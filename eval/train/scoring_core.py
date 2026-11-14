"""Method-agnostic scoring for the interaction-retargeting benchmark.

A *method* is any executable registered as ``scripts/retargeters/<name>.sh``.
It is invoked as ``bash scripts/retargeters/<name>.sh <input_dir> <output_dir>``
from the workspace root and must write ``<output_dir>/<instance_id>.json``
containing ``{"q": [[11 floats] x num_frames]}`` for every input file.

Everything scored is recomputed here from ``q`` with this module's own copy of
the robot kinematics and scene geometry; nothing else a method writes is read.
A method run fails as a whole (and receives the worst value of every metric in
every experiment) if it crashes, exceeds its CPU budget, stops making progress,
misses any instance, or returns a trajectory that is malformed, non-finite, or
outside the robot's joint or velocity limits.

Budget: CPU time (user + system) of the method's whole process tree, counted by
the evaluator -- not wall-clock time, so the result does not depend on machine
load.  Methods run with one BLAS/OpenMP thread.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Robot model (independent copy; must match the published robot definition)
# ---------------------------------------------------------------------------
THIGH, SHIN, TORSO = 0.40, 0.39, 0.48
SHOULDER_X, SHOULDER_DROP = 0.16, 0.06
UPPER_ARM, FOREARM = 0.27, 0.25
Q_MIN = np.array([-3.0, 0.30, -0.80, -1.50, -0.30, -1.50, -2.40, -3.20, -2.60, -0.60, -2.60])
Q_MAX = np.array([3.0, 1.60, 0.80, 1.50, 2.40, 1.50, 0.30, 0.60, 2.60, 3.20, 2.60])
V_MAX = np.array([1.5, 1.5, 4.0, 8.0, 10.0, 8.0, 10.0, 8.0, 10.0, 8.0, 10.0])
LINKS = [(0, 1), (1, 6), (1, 9), (0, 2), (2, 3), (0, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)]
PELVIS, LEFT_FOOT, RIGHT_FOOT, LEFT_HAND, RIGHT_HAND = 0, 3, 5, 8, 11
END_EFFECTORS = {"left_foot": LEFT_FOOT, "right_foot": RIGHT_FOOT, "left_hand": LEFT_HAND, "right_hand": RIGHT_HAND}
KEYPOINT_NAMES = ["pelvis", "neck", "left_knee", "left_foot", "right_knee", "right_foot",
                  "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
LIMIT_TOL = 1e-6

EXPERIMENTS = [
    "object_interaction_kinematics",
    "terrain_interaction_kinematics",
    "augmented_interaction_kinematics",
]
METRICS = [
    "penetration_duration",
    "max_penetration_cm",
    "foot_skating_duration",
    "max_foot_skating_cm_s",
    "contact_preservation",
    "spurious_contact_duration",
    "pelvis_path_error_cm",
    "motion_velocity_error_cm_s",
]
HIGHER_IS_BETTER = {m: (m == "contact_preservation") for m in METRICS}
# Unbounded metrics are capped; the cap is also the value of a failed run.
CAPS = {
    "penetration_duration": 1.0,
    "max_penetration_cm": 30.0,
    "foot_skating_duration": 1.0,
    "max_foot_skating_cm_s": 300.0,
    "contact_preservation": 1.0,
    "spurious_contact_duration": 1.0,
    "pelvis_path_error_cm": 100.0,
    "motion_velocity_error_cm_s": 300.0,
}
WORST = {m: (0.0 if HIGHER_IS_BETTER[m] else CAPS[m]) for m in METRICS}

PENETRATION_FRAME_DEPTH = 0.01   # m: a frame counts as penetrating beyond 1 cm
SKATING_SPEED = 0.01             # m/s: stance-foot speed above this is skating
CONTACT_TOLERANCE = 0.02         # m: |distance to contact surface| for a kept contact
CLEARANCE_DEMO = 0.03            # m: demonstrator end effector (robot scale) this far from every surface ...
CLEARANCE_TOUCH = 0.01           # m: ... while the robot's is within this of a surface = spurious contact
LINK_SAMPLES = 21

CPU_BUDGET_PER_INSTANCE = 20.0   # CPU seconds per instance (reference method: about 0.6)
IDLE_POLL_LIMIT = 1800           # consecutive 1 s polls without any CPU progress = hung method
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def fk_batch(Q: np.ndarray) -> np.ndarray:
    x, z, tor, lh, lk, rh, rk, ls, le, rs, re = np.asarray(Q, dtype=float).T

    def d(a):
        return np.stack([np.sin(a), -np.cos(a)], axis=-1)

    pelvis = np.stack([x, z], axis=-1)
    neck = pelvis + TORSO * np.stack([np.sin(tor), np.cos(tor)], axis=-1)
    l_sh = neck + np.array([-SHOULDER_X, -SHOULDER_DROP])
    r_sh = neck + np.array([SHOULDER_X, -SHOULDER_DROP])
    l_knee = pelvis + THIGH * d(lh)
    l_foot = l_knee + SHIN * d(lh + lk)
    r_knee = pelvis + THIGH * d(rh)
    r_foot = r_knee + SHIN * d(rh + rk)
    l_elb = l_sh + UPPER_ARM * d(ls)
    l_hand = l_elb + FOREARM * d(ls + le)
    r_elb = r_sh + UPPER_ARM * d(rs)
    r_hand = r_elb + FOREARM * d(rs + re)
    return np.stack([pelvis, neck, l_knee, l_foot, r_knee, r_foot,
                     l_sh, l_elb, l_hand, r_sh, r_elb, r_hand], axis=1)


def link_samples(P: np.ndarray, n: int = LINK_SAMPLES) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    a = P[:, [i for i, _ in LINKS]]
    b = P[:, [j for _, j in LINKS]]
    s = a[:, :, None, :] + (b - a)[:, :, None, :] * u[:, None]
    return s.reshape(P.shape[0], -1, 2)


# ---------------------------------------------------------------------------
# Scene geometry (independent copy)
# ---------------------------------------------------------------------------

def terrain_sdf(P, verts) -> np.ndarray:
    v = np.asarray(verts, dtype=float)
    P = np.asarray(P, dtype=float)
    A, B = v[:-1], v[1:]
    AB = B - A
    t = np.clip(np.sum((P[..., None, :] - A) * AB, axis=-1) / np.sum(AB * AB, axis=1), 0.0, 1.0)
    C = A + t[..., None] * AB
    d = np.sqrt(np.min(np.sum((P[..., None, :] - C) ** 2, axis=-1), axis=-1))
    inside = P[..., 1] < np.interp(P[..., 0], v[:, 0], v[:, 1])
    return np.where(inside, -d, d)


def box_sdf(P, center, size) -> np.ndarray:
    q = np.abs(np.asarray(P, dtype=float) - np.asarray(center, dtype=float)) - 0.5 * np.asarray(size, dtype=float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def surface_distance(P: np.ndarray, scene: dict) -> np.ndarray:
    """Unsigned distance of per-frame points P (T, 2) to the nearest surface of ``scene``."""
    d = np.abs(terrain_sdf(P, scene["terrain"]))
    obj = scene.get("object")
    if obj:
        d = np.minimum(d, np.abs(box_sdf(P, np.asarray(obj["center"], dtype=float), obj["size"])))
    return d


# ---------------------------------------------------------------------------
# Output validation and per-instance metrics
# ---------------------------------------------------------------------------

def validate_output(path: Path, inp: dict) -> tuple[np.ndarray | None, str]:
    if not path.is_file():
        return None, "missing output"
    try:
        data = json.loads(path.read_text())
        q = np.asarray(data["q"], dtype=float)
    except Exception as exc:  # malformed JSON / ragged arrays / missing key
        return None, f"malformed output: {type(exc).__name__}"
    N = int(inp["num_frames"])
    if q.shape != (N, len(Q_MIN)):
        return None, f"q has shape {q.shape}, expected {(N, len(Q_MIN))}"
    if not np.all(np.isfinite(q)):
        return None, "non-finite configuration"
    if np.any(q < Q_MIN - LIMIT_TOL) or np.any(q > Q_MAX + LIMIT_TOL):
        return None, "joint limit violated"
    step = np.abs(np.diff(q, axis=0))
    if np.any(step > V_MAX * float(inp["dt"]) + LIMIT_TOL):
        return None, "velocity limit violated"
    return q, ""


def _ground_relative_path(pel: np.ndarray, terrain) -> np.ndarray:
    """Pelvis displacement from frame 0: horizontal, and height above the ground below it."""
    v = np.asarray(terrain, dtype=float)
    h = pel[:, 1] - np.interp(pel[:, 0], v[:, 0], v[:, 1])
    return np.stack([pel[:, 0] - pel[0, 0], h - h[0]], axis=1)


def instance_metrics(inp: dict, ann: dict, q: np.ndarray, extras: dict | None = None) -> dict[str, float]:
    dt = float(inp["dt"])
    alpha = float(ann["alpha"])  # robot-to-demonstrator height ratio (direct scaling)
    P = fk_batch(q)
    src = np.asarray(inp["source_keypoints"], dtype=float)
    scene = inp["target_scene"]
    src_scene = inp["source_scene"]
    terrain = np.asarray(scene["terrain"], dtype=float)
    obj = scene.get("object")
    centers = np.asarray(obj["center"], dtype=float) if obj else None
    size = np.asarray(obj["size"], dtype=float) if obj else None

    # Penetration of any robot link into the target scene.
    S = link_samples(P)
    d_terr = np.maximum(0.0, -terrain_sdf(S, terrain))
    d_all = d_terr
    if obj:
        d_obj = np.maximum(0.0, -box_sdf(S, centers[:, None, :], size))
        d_all = np.maximum(d_terr, d_obj)
    depth = np.max(d_all, axis=1)
    if extras is not None and float(np.max(depth)) > 0.0:  # diagnostics only (not scored)
        t_star = int(np.argmax(depth))
        k_star = int(np.argmax(d_all[t_star]))
        i, j = LINKS[k_star // LINK_SAMPLES]
        extras["max_penetration_frame"] = t_star
        extras["max_penetration_link"] = f"{KEYPOINT_NAMES[i]}-{KEYPOINT_NAMES[j]}"
        extras["max_penetration_obstacle"] = "object" if obj and d_obj[t_star, k_star] >= d_terr[t_star, k_star] else "terrain"

    # Stance feet: skating and contact with the target terrain.
    speeds, contacts = [], []
    for name, idx in (("left_foot", LEFT_FOOT), ("right_foot", RIGHT_FOOT)):
        st = np.asarray(ann["stance"][name], dtype=bool)
        ev = st[1:] & st[:-1]
        v = np.linalg.norm(P[1:, idx] - P[:-1, idx], axis=1) / dt
        speeds.append(v[ev])
        contacts.append(np.abs(terrain_sdf(P[st, idx], terrain)) <= CONTACT_TOLERANCE)
    n_foot_events = int(sum(len(c) for c in contacts))
    # Demonstrated hand contacts with the object or the terrain.
    for name, idx in (("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        for t, label in enumerate(ann["hand_contact"][name]):
            if label == "object" and obj:
                d = float(box_sdf(P[t, idx], centers[t], size))
            elif label == "terrain":
                d = float(terrain_sdf(P[t, idx], terrain))
            else:
                continue
            contacts.append(np.array([abs(d) <= CONTACT_TOLERANCE]))
    speeds = np.concatenate(speeds)
    contacts = np.concatenate(contacts)

    # Spurious contacts: an end effector touches a surface where the demonstrator's was clear of every surface.
    clear_events, spurious = 0, 0
    for name, idx in END_EFFECTORS.items():
        clear = alpha * surface_distance(src[:, idx], src_scene) >= CLEARANCE_DEMO
        clear_events += int(np.sum(clear))
        spurious += int(np.sum(surface_distance(P[:, idx], scene)[clear] <= CLEARANCE_TOUCH))

    if extras is not None:  # diagnostics only (not scored)
        extras["foot_contact_rate"] = float(np.mean(contacts[:n_foot_events])) if n_foot_events else None
        extras["hand_contact_rate"] = float(np.mean(contacts[n_foot_events:])) if len(contacts) > n_foot_events else None
        extras["clearance_events"] = clear_events

    # Fidelity to the demonstration brought to the robot's scale (alpha); the pelvis height
    # is measured above the ground below it, so following a modified terrain is not penalised.
    pel_err = np.linalg.norm(_ground_relative_path(P[:, PELVIS], terrain)
                             - alpha * _ground_relative_path(src[:, PELVIS], src_scene["terrain"]), axis=1)
    vel_err = np.linalg.norm(np.diff(P, axis=0) - alpha * np.diff(src, axis=0), axis=2) / dt

    raw = {
        "penetration_duration": float(np.mean(depth > PENETRATION_FRAME_DEPTH)),
        "max_penetration_cm": 100.0 * float(np.max(depth)),
        "foot_skating_duration": float(np.mean(speeds > SKATING_SPEED)) if len(speeds) else 0.0,
        "max_foot_skating_cm_s": 100.0 * float(np.max(speeds)) if len(speeds) else 0.0,
        "contact_preservation": float(np.mean(contacts)) if len(contacts) else 0.0,
        "spurious_contact_duration": spurious / clear_events if clear_events else 0.0,
        "pelvis_path_error_cm": 100.0 * float(np.mean(pel_err)),
        "motion_velocity_error_cm_s": 100.0 * float(np.mean(vel_err)),
    }
    return {m: float(min(max(raw[m], 0.0), CAPS[m])) for m in METRICS}


def aggregate(per_instance: dict[str, dict], anns: dict[str, dict]) -> dict[str, dict[str, float]]:
    out = {}
    for exp in EXPERIMENTS:
        ids = [i for i in per_instance if anns[i]["experiment"] == exp]
        out[exp] = {m: float(np.mean([per_instance[i][m] for i in ids])) for m in METRICS}
    return out


# ---------------------------------------------------------------------------
# Running registered methods under a counted CPU budget
# ---------------------------------------------------------------------------
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CLK = os.sysconf("SC_CLK_TCK")


def discover_methods(root: Path) -> list[tuple[str, Path]]:
    reg = root / "scripts" / "retargeters"
    out = []
    for p in sorted(reg.glob("*.sh")):
        if NAME_RE.match(p.stem):
            out.append((p.stem, p))
    return out


def _session_cpu(sid: int) -> float:
    """CPU seconds of the live processes of session ``sid`` (incl. their reaped children)."""
    total = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                rest = fh.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(rest[3]) == sid:
            total += int(rest[11]) + int(rest[12]) + int(rest[13]) + int(rest[14])
    return total / _CLK


def _children_cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def run_method(root: Path, script: Path, in_dir: Path, out_dir: Path, budget: float, log_path: Path) -> tuple[str, float]:
    """Run one method; returns (failure reason or '', CPU seconds used)."""
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env["PYTHONPATH"] = str(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cpu0 = _children_cpu()
    reason = ""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(["bash", str(script), str(in_dir), str(out_dir)], cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last, idle = -1.0, 0
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            used = _session_cpu(proc.pid)
            if used > budget:
                reason = f"CPU budget exceeded ({budget:.0f} CPU s)"
            idle = idle + 1 if used <= last + 1e-9 else 0
            last = max(last, used)
            if idle >= IDLE_POLL_LIMIT:
                reason = f"no CPU progress for {IDLE_POLL_LIMIT} polls (hung)"
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
    cpu = _children_cpu() - cpu0
    if not reason and proc.returncode != 0:
        reason = f"exit code {proc.returncode}"
    if not reason and cpu > budget:
        reason = f"CPU budget exceeded ({cpu:.0f} > {budget:.0f} CPU s)"
    return reason, cpu


def run_and_score(root: Path, name: str, script: Path, inputs: dict[str, dict], anns: dict[str, dict],
                  log_dir: Path, keep_dir: Path | None = None) -> tuple[dict, dict]:
    """Run one registered method on all instances and score it."""
    diag: dict = {"status": "ok", "reason": "", "per_instance": {}}
    work = Path(tempfile.mkdtemp(prefix=f"retarget_{name}_"))
    in_dir, out_dir = work / "inputs", work / "outputs"
    in_dir.mkdir()
    out_dir.mkdir()
    for iid, inp in inputs.items():
        (in_dir / f"{iid}.json").write_text(json.dumps(inp))
    log_dir.mkdir(parents=True, exist_ok=True)
    budget = CPU_BUDGET_PER_INSTANCE * len(inputs)
    reason, cpu = run_method(root, script, in_dir, out_dir, budget, log_dir / f"{name}.log")
    diag["cpu_s"] = round(cpu, 1)
    diag["cpu_budget_s"] = budget
    if reason:
        diag["status"], diag["reason"] = "failed", reason

    per_instance = {}
    if diag["status"] == "ok":
        for iid, inp in inputs.items():
            q, why = validate_output(out_dir / f"{iid}.json", inp)
            if q is None:
                diag["status"], diag["reason"] = "failed", f"{iid}: {why}"
                break
            extras = {}
            per_instance[iid] = instance_metrics(inp, anns[iid], q, extras)
            diag.setdefault("per_instance_extras", {})[iid] = extras
    if keep_dir is not None:  # audit option: keep the raw outputs for rescoring
        shutil.copytree(out_dir, keep_dir / name, dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)

    if diag["status"] != "ok":
        return {exp: dict(WORST) for exp in EXPERIMENTS}, diag
    diag["per_instance"] = {iid: {"experiment": anns[iid]["experiment"], **m} for iid, m in per_instance.items()}
    return aggregate(per_instance, anns), diag


def evaluate_all(root: Path, inputs: dict[str, dict], anns: dict[str, dict], split: str,
                 score_out: Path, diag_out: Path, log_dir: Path) -> dict:
    methods = discover_methods(root)
    keep = os.environ.get("RETARGET_KEEP_OUTPUTS")
    keep_dir = Path(keep) / split if keep else None
    scores = {"experiments": {exp: {"results": {}} for exp in EXPERIMENTS}}
    diagnostics = {"split": split, "num_instances": len(inputs),
                   "instances_per_experiment": {e: sum(a["experiment"] == e for a in anns.values()) for e in EXPERIMENTS},
                   "cpu_budget_per_instance_s": CPU_BUDGET_PER_INSTANCE,
                   "methods": {}}
    score_out.parent.mkdir(parents=True, exist_ok=True)
    diag_out.parent.mkdir(parents=True, exist_ok=True)
    for name, script in methods:
        results, diag = run_and_score(root, name, script, inputs, anns, log_dir, keep_dir)
        for exp in EXPERIMENTS:
            scores["experiments"][exp]["results"][name] = results[exp]
        diagnostics["methods"][name] = diag
        print(f"[{split}] {name}: {diag['status']} {diag['reason']} ({diag.get('cpu_s', 0)} CPU s)", flush=True)
        score_out.write_text(json.dumps(scores, indent=2))
        diag_out.write_text(json.dumps(diagnostics, indent=2))
    score_out.write_text(json.dumps(scores, indent=2))
    diag_out.write_text(json.dumps(diagnostics, indent=2))
    return scores
