"""Method-agnostic scoring for the humanoid interaction-retargeting benchmark (train copy).

A *method* is any executable registered as ``scripts/retargeters/<name>.sh``.
It is invoked as ``bash scripts/retargeters/<name>.sh <input_dir> <output_dir>``
from the workspace root and must write ``<output_dir>/<instance_id>.json``
containing ``{"q": [[36 floats] x num_frames]}`` for every input file: the
Unitree G1 configuration in MuJoCo order (base position, base quaternion
w-x-y-z, 29 joint angles in the order of the robot model).

Every scored number is recomputed here from ``q`` with this module's own copy
of the robot model (``assets/``) and of the scene geometry; nothing else a
method writes is read.  A run fails as a whole -- and receives the worst value
of every metric in every experiment -- if the method crashes, exceeds its
counted CPU budget, stops making progress, misses an instance, or returns a
trajectory that is malformed, non-finite, has a non-unit base quaternion,
teleports, or leaves the robot's joint position or velocity limits.

Budget: CPU time (user + system) of the method's whole process tree, counted
by the evaluator -- not wall-clock time, so results do not depend on machine
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
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
ROBOT_XML = ASSETS / "g1_29dof_spherehand_hull.xml"
LIMITS = json.loads((ASSETS / "g1_limits.json").read_text())
Q_LO = np.array([j["range"][0] for j in LIMITS["joints"]])
Q_HI = np.array([j["range"][1] for j in LIMITS["joints"]])
V_MAX = np.array([j["velocity"] for j in LIMITS["joints"]])
NQ = 7 + len(Q_LO)
LIMIT_TOL = 1e-4
QUAT_TOL = 1e-3
MAX_BASE_STEP = 0.2          # m per frame (6 m/s, 4x the fastest demonstrated pelvis speed): larger base jumps are teleports

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
    "pose_error_cm",
]
HIGHER_IS_BETTER = {m: (m == "contact_preservation") for m in METRICS}
CAPS = {
    "penetration_duration": 1.0,
    "max_penetration_cm": 30.0,
    "foot_skating_duration": 1.0,
    "max_foot_skating_cm_s": 300.0,
    "contact_preservation": 1.0,
    "spurious_contact_duration": 1.0,
    "pelvis_path_error_cm": 100.0,
    "pose_error_cm": 100.0,
}
WORST = {m: (0.0 if HIGHER_IS_BETTER[m] else CAPS[m]) for m in METRICS}

# Kinematic-quality definitions (paper Sec. V-B, released evaluation code).
PENETRATION_TOL = 0.01        # m: a frame is penetrating if any robot geom is > 1 cm inside the scene
DETECT = 0.1                  # m: distance query cut-off
STANCE_STEP = 0.01            # m per frame: scaled source toe slower than this = stance foot
SKATE_STEP = 0.01             # m per frame: robot stance toe faster than this = skating
OBJ_HAND_RADIUS = 0.40        # m: demonstrator wrist this close to the box origin while the box is carried = hand contact
OBJ_HAND_CLEAR = 0.50         # m: demonstrator wrist farther than this from the box origin = hand clearly off the box
OBJ_LIFT, OBJ_MOVE = 0.03, 0.003   # m above its first-frame height / m per frame: the box is being carried
TERRAIN_HAND = 0.05           # m: demonstrator hand joint within this of a terrain box = contact
TERRAIN_TOE, TERRAIN_ANKLE = 0.06, 0.11   # m: demonstrator toe / ankle within this of a terrain box = foot contact
TERRAIN_HAND_CLEAR = 0.15     # m: demonstrator hand farther than this from every terrain box = clear
ROBOT_CONTACT = 0.03          # m: robot limb geometry within this of the surface = contact kept
ROBOT_TOUCH = 0.01            # m: robot hand this close to a surface while the demonstrator's is clear = spurious

ROBOT_LIMB_BODIES = {
    "left_hand": ["left_sphere_hand_link", "left_sphere_hand_tip_link", "left_wrist_yaw_link",
                  "left_wrist_pitch_link", "left_wrist_roll_link"],
    "right_hand": ["right_sphere_hand_link", "right_sphere_hand_tip_link", "right_wrist_yaw_link",
                   "right_wrist_pitch_link", "right_wrist_roll_link"],
    "left_foot": ["left_ankle_roll_link", "left_ankle_pitch_link"]
                 + [f"left_ankle_roll_sphere_{k}_link" for k in range(1, 6)],
    "right_foot": ["right_ankle_roll_link", "right_ankle_pitch_link"]
                  + [f"right_ankle_roll_sphere_{k}_link" for k in range(1, 6)],
}
ROBOT_TOE = {"left": "left_ankle_roll_sphere_5_link", "right": "right_ankle_roll_sphere_5_link"}
ROBOT_PELVIS = "pelvis_contour_link"
SMPLH_MAP = {
    "Pelvis": "pelvis_contour_link", "L_Hip": "left_hip_pitch_link", "R_Hip": "right_hip_pitch_link",
    "L_Knee": "left_knee_link", "R_Knee": "right_knee_link", "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link", "L_Elbow": "left_elbow_link", "R_Elbow": "right_elbow_link",
    "L_Ankle": "left_ankle_intermediate_1_link", "R_Ankle": "right_ankle_intermediate_1_link",
    "L_Toe": "left_ankle_roll_sphere_5_link", "R_Toe": "right_ankle_roll_sphere_5_link",
    "L_Wrist": "left_sphere_hand_link", "R_Wrist": "right_sphere_hand_link",
}
MOCAP_MAP = {
    "Spine1": "pelvis_contour_link", "LeftUpLeg": "left_hip_pitch_link", "LeftLeg": "left_knee_link",
    "LeftToeBase": "left_ankle_roll_sphere_5_link", "RightUpLeg": "right_hip_pitch_link",
    "RightLeg": "right_knee_link", "RightToeBase": "right_ankle_roll_sphere_5_link",
    "LeftArm": "left_shoulder_roll_link", "LeftForeArm": "left_elbow_link",
    "LeftHandMiddle3": "left_sphere_hand_link", "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link", "RightHandMiddle3": "right_sphere_hand_link",
    "LeftFoot": "left_ankle_intermediate_1_link", "RightFoot": "right_ankle_intermediate_1_link",
}
SOURCE_TOES = {"smplh": ("L_Toe", "R_Toe"), "mocap": ("LeftToeBase", "RightToeBase")}
SOURCE_PELVIS = {"smplh": "Pelvis", "mocap": "Hips"}

CPU_BUDGET_PER_INSTANCE = 30.0   # CPU seconds per instance
IDLE_POLL_LIMIT = 1800           # consecutive 1 s polls without any CPU progress = hung method
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


# ---------------------------------------------------------------------------
# Scene geometry (independent copy)
# ---------------------------------------------------------------------------

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def box_frames(box: dict, T: int):
    P = np.asarray(box["poses"], dtype=float) if "poses" in box else np.repeat(
        np.asarray(box["pose"], dtype=float)[None], T, axis=0)
    R = quat_to_mat(P[:, 3:7])
    c = P[:, :3] + np.einsum("tij,j->ti", R, np.asarray(box.get("offset", [0, 0, 0]), dtype=float))
    return P, c, R, np.asarray(box["half_size"], dtype=float)


def box_sdf(points: np.ndarray, c: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance of points (T, K, 3) to per-frame boxes (c (T,3), R (T,3,3))."""
    local = np.einsum("tji,tkj->tki", R, points - c[:, None, :])
    q = np.abs(local) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def terrain_sdf(points: np.ndarray, scene: dict) -> np.ndarray:
    T = points.shape[0]
    d = np.full(points.shape[:-1], np.inf)
    for b in scene["boxes"]:
        if b.get("kind") == "terrain":
            _, c, R, h = box_frames(b, T)
            d = np.minimum(d, box_sdf(points, c, R, h))
    return d


def build_scene_model(scene: dict):
    root = ET.parse(ROBOT_XML).getroot()
    wb = root.find("worldbody")
    for g in root.iter("geom"):
        g.set("contype", "1")
        g.set("conaffinity", "0")
        g.set("margin", str(DETECT))
    ET.SubElement(wb, "geom", name="ground", type="plane", size="100 100 0.1",
                  pos=f"0 0 {float(scene.get('ground_z', 0.0))}", contype="0", conaffinity="1", margin=str(DETECT))
    for k, box in enumerate(scene["boxes"]):
        hs = " ".join(f"{float(v):.6f}" for v in box["half_size"])
        off = " ".join(f"{float(v):.6f}" for v in box.get("offset", [0, 0, 0]))
        if "poses" in box:
            b = ET.SubElement(wb, "body", name=f"box{k}", mocap="true")
        else:
            p = [float(v) for v in box["pose"]]
            b = ET.SubElement(wb, "body", name=f"box{k}", pos=" ".join(map(str, p[:3])),
                              quat=" ".join(map(str, p[3:7])))
        ET.SubElement(b, "geom", name=f"box{k}", type="box", size=hs, pos=off, contype="0", conaffinity="1",
                      margin=str(DETECT))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, mujoco.MjData(model)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(path: Path, inp: dict) -> tuple[np.ndarray | None, str]:
    if not path.is_file():
        return None, "missing output"
    try:
        data = json.loads(path.read_text())
        q = np.asarray(data["q"], dtype=float)
    except Exception as exc:  # malformed JSON / ragged arrays / missing key
        return None, f"malformed output: {type(exc).__name__}"
    T = int(inp["num_frames"])
    if q.shape != (T, NQ):
        return None, f"q has shape {q.shape}, expected {(T, NQ)}"
    if not np.all(np.isfinite(q)):
        return None, "non-finite configuration"
    qn = np.linalg.norm(q[:, 3:7], axis=1)
    if np.any(np.abs(qn - 1.0) > QUAT_TOL):
        return None, "base quaternion is not unit length"
    q = q.copy()
    q[:, 3:7] /= qn[:, None]
    if np.any(q[:, 7:] < Q_LO - LIMIT_TOL) or np.any(q[:, 7:] > Q_HI + LIMIT_TOL):
        return None, "joint position limit violated"
    dt = 1.0 / float(inp["fps"])
    if np.any(np.abs(np.diff(q[:, 7:], axis=0)) > V_MAX * dt + LIMIT_TOL):
        return None, "joint velocity limit violated"
    if np.any(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) > MAX_BASE_STEP):
        return None, "base teleport"
    return q, ""


# ---------------------------------------------------------------------------
# Per-instance measurements
# ---------------------------------------------------------------------------

def experiment_of(inp: dict) -> str:
    if inp.get("augmentation"):
        return "augmented_interaction_kinematics"
    return "object_interaction_kinematics" if inp["task"] == "object" else "terrain_interaction_kinematics"


def annotate(inp: dict) -> dict:
    """Labels derived from the demonstration and its scene: stance feet, desired and clear contacts."""
    names = inp["joint_names"]
    fmt = "smplh" if "L_Wrist" in names else "mocap"
    J = np.asarray(inp["human_joints"], dtype=float)
    T = len(J)
    alpha = float(inp["alpha"])
    src = inp["source_scene"]
    toes = [names.index(t) for t in SOURCE_TOES[fmt]]
    step = alpha * np.linalg.norm(np.diff(J[:, toes, :2], axis=0), axis=2)
    stance = np.vstack([np.zeros((1, 2), bool), step <= STANCE_STEP])
    contact, clear = {}, {}
    if inp["task"] == "object":
        P, _, R, _ = box_frames(src["boxes"][0], T)
        step_obj = np.concatenate([[0.0], np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)])
        carried = (P[:, 2] - P[0, 2] > OBJ_LIFT) | (step_obj > OBJ_MOVE)
        for side, jn in (("left_hand", "L_Wrist"), ("right_hand", "R_Wrist")):
            r = np.linalg.norm(J[:, names.index(jn)] - P[:, :3], axis=1)
            contact[side] = carried & (r <= OBJ_HAND_RADIUS)
            clear[side] = r > OBJ_HAND_CLEAR
    else:
        def sd(*joints):
            return np.min(terrain_sdf(J[:, [names.index(j) for j in joints]], src), axis=1)
        for side, pre in (("left", "Left"), ("right", "Right")):
            dh = sd(f"{pre}HandMiddle3", f"{pre}Hand")
            contact[f"{side}_hand"] = dh <= TERRAIN_HAND
            clear[f"{side}_hand"] = dh > TERRAIN_HAND_CLEAR
            contact[f"{side}_foot"] = (sd(f"{pre}ToeBase") <= TERRAIN_TOE) | (sd(f"{pre}Foot") <= TERRAIN_ANKLE)
    kmap = SMPLH_MAP if fmt == "smplh" else MOCAP_MAP
    return {
        "experiment": experiment_of(inp),
        "format": fmt,
        "stance": {"left": stance[:, 0], "right": stance[:, 1]},
        "contact": contact,
        "clear": clear,
        "src_kp": alpha * J[:, [names.index(SOURCE_PELVIS[fmt])] + [names.index(k) for k in list(kmap)[1:]]],
        "kp_bodies": list(kmap.values()),
    }


def instance_measures(inp: dict, ann: dict, q: np.ndarray) -> dict:
    """Raw per-instance counts and values; ``aggregate`` turns them into the experiment metrics."""
    scene = inp["target_scene"]
    model, data = build_scene_model(scene)
    T = len(q)
    robot_geoms = [g for g in range(model.ngeom) if model.geom_contype[g] == 1]
    body_geoms = {}
    for g in robot_geoms:
        body_geoms.setdefault(model.body(int(model.geom_bodyid[g])).name, []).append(g)
    limb_geoms = {k: [g for b in v for g in body_geoms.get(b, [])] for k, v in ROBOT_LIMB_BODIES.items()}
    kinds = [b.get("kind", "terrain") for b in scene["boxes"]]
    box_geoms = [model.geom(f"box{k}").id for k in range(len(scene["boxes"]))]
    contact_geoms = [g for g, kd in zip(box_geoms, kinds) if kd == ("object" if inp["task"] == "object" else "terrain")]
    mocap = {k: int(model.body_mocapid[model.body(f"box{k}").id]) for k, b in enumerate(scene["boxes"]) if "poses" in b}
    kp_ids = [model.body(b).id for b in ann["kp_bodies"]]
    toe_ids = {s: model.body(b).id for s, b in ROBOT_TOE.items()}
    pel_id = model.body(ROBOT_PELVIS).id

    depth = np.zeros(T)
    limb_dist = {k: np.full(T, np.inf) for k in ROBOT_LIMB_BODIES}
    support_dist = {s: np.full(T, np.inf) for s in ("left", "right")}
    ground_id = model.geom("ground").id
    support_geoms = [ground_id] + [g for g, kd in zip(box_geoms, kinds) if kd == "terrain"]
    kp = np.zeros((T, len(kp_ids), 3))
    toes = {s: np.zeros((T, 3)) for s in toe_ids}
    pelvis = np.zeros((T, 3))
    fromto = np.zeros(6)
    for t in range(T):
        for k, mid in mocap.items():
            p = scene["boxes"][k]["poses"][t]
            data.mocap_pos[mid] = p[:3]
            data.mocap_quat[mid] = p[3:7]
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        worst = 0.0
        pairs = set(zip(data.contact.geom1[:data.ncon].tolist(), data.contact.geom2[:data.ncon].tolist()))
        for g1, g2 in pairs:
            dist = mujoco.mj_geomDistance(model, data, int(g1), int(g2), DETECT, fromto)
            worst = max(worst, -dist)
        depth[t] = worst
        for limb, gs in limb_geoms.items():
            best = np.inf
            for g in gs:
                for s in contact_geoms:
                    best = min(best, mujoco.mj_geomDistance(model, data, g, s, DETECT, fromto))
            limb_dist[limb][t] = best
        for side in ("left", "right"):
            best = np.inf
            for g in limb_geoms[f"{side}_foot"]:
                for s in support_geoms:
                    best = min(best, mujoco.mj_geomDistance(model, data, g, s, DETECT, fromto))
            support_dist[side][t] = best
        kp[t] = data.xpos[kp_ids]
        for s, b in toe_ids.items():
            toes[s][t] = data.xpos[b]
        pelvis[t] = data.xpos[pel_id]

    # foot skating on demonstrated stance frames
    stick = slide = 0
    max_slide = 0.0
    dt = 1.0 / float(inp["fps"])
    step = {s: np.concatenate([[0.0], np.linalg.norm(np.diff(toes[s][:, :2], axis=0), axis=1)]) for s in toes}
    st = ann["stance"]
    either = st["left"] | st["right"]
    sliding = (st["left"] & (step["left"] > SKATE_STEP)) | (st["right"] & (step["right"] > SKATE_STEP))
    stick, slide = int(either.sum()), int(sliding.sum())
    if slide:
        v = np.maximum(np.where(st["left"] & (step["left"] > SKATE_STEP), step["left"], 0.0),
                       np.where(st["right"] & (step["right"] > SKATE_STEP), step["right"], 0.0))
        max_slide = float(v.max()) / dt
    # contacts: demonstrated hand/foot contacts with the object or terrain, and planted (stance) feet
    # on a supporting surface (floor or terrain)
    desired = kept = clear_n = spurious = 0
    for limb, want in ann["contact"].items():
        desired += int(want.sum())
        kept += int(np.sum(want & (limb_dist[limb] <= ROBOT_CONTACT)))
    for side in ("left", "right"):
        want = ann["stance"][side]
        desired += int(want.sum())
        kept += int(np.sum(want & (support_dist[side] <= ROBOT_CONTACT)))
    for limb, cl in ann["clear"].items():
        clear_n += int(cl.sum())
        spurious += int(np.sum(cl & (limb_dist[limb] <= ROBOT_TOUCH)))
    # fidelity: pelvis path (horizontal) and body pose relative to the pelvis, vs the scaled demonstration
    src = ann["src_kp"]
    pel_err = np.linalg.norm(pelvis[:, :2] - src[:, 0, :2], axis=1)
    pose_err = np.linalg.norm((kp[:, 1:] - kp[:, :1]) - (src[:, 1:] - src[:, :1]), axis=2)
    return {
        "frames": T,
        "penetrating_frames": int(np.sum(depth > PENETRATION_TOL)),
        "max_depth": float(depth.max()),
        "stance_frames": stick,
        "skating_frames": slide,
        "max_skating_speed": max_slide,
        "desired_contacts": desired,
        "kept_contacts": kept,
        "clear_events": clear_n,
        "spurious_events": spurious,
        "pelvis_error": float(pel_err.mean()),
        "pose_error": float(pose_err.mean()),
    }


def aggregate(measures: dict[str, dict], anns: dict[str, dict]) -> dict[str, dict[str, float]]:
    out = {}
    for exp in EXPERIMENTS:
        ms = [measures[i] for i in measures if anns[i]["experiment"] == exp]
        if not ms:
            continue
        s = lambda k: float(sum(m[k] for m in ms))  # noqa: E731
        raw = {
            "penetration_duration": float(np.mean([m["penetrating_frames"] / m["frames"] for m in ms])),
            "max_penetration_cm": 100.0 * float(np.mean([m["max_depth"] for m in ms])),
            "foot_skating_duration": s("skating_frames") / max(s("stance_frames"), 1.0),
            "max_foot_skating_cm_s": 100.0 * float(np.mean([m["max_skating_speed"] for m in ms])),
            "contact_preservation": s("kept_contacts") / s("desired_contacts") if s("desired_contacts") else 1.0,
            "spurious_contact_duration": s("spurious_events") / max(s("clear_events"), 1.0),
            "pelvis_path_error_cm": 100.0 * float(np.mean([m["pelvis_error"] for m in ms])),
            "pose_error_cm": 100.0 * float(np.mean([m["pose_error"] for m in ms])),
        }
        out[exp] = {m: float(min(max(raw[m], 0.0), CAPS[m])) for m in METRICS}
    return out


# ---------------------------------------------------------------------------
# Running registered methods under a counted CPU budget
# ---------------------------------------------------------------------------
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CLK = os.sysconf("SC_CLK_TCK")


def discover_methods(root: Path) -> list[tuple[str, Path]]:
    reg = root / "scripts" / "retargeters"
    return [(p.stem, p) for p in sorted(reg.glob("*.sh")) if NAME_RE.match(p.stem)]


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


def score_outputs(out_dir: Path, inputs: dict[str, dict], anns: dict[str, dict], diag: dict) -> dict | None:
    """Validate and measure every instance; None if the run is invalid (reason in diag)."""
    measures = {}
    for iid, inp in inputs.items():
        q, why = validate_output(out_dir / f"{iid}.json", inp)
        if q is None:
            diag["status"], diag["reason"] = "failed", f"{iid}: {why}"
            return None
        measures[iid] = instance_measures(inp, anns[iid], q)
    return measures


def run_and_score(root: Path, name: str, script: Path, inputs: dict[str, dict], anns: dict[str, dict],
                  log_dir: Path, keep_dir: Path | None = None) -> tuple[dict, dict]:
    """Run one registered method on all instances and score it."""
    diag: dict = {"status": "ok", "reason": ""}
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
    measures = score_outputs(out_dir, inputs, anns, diag) if diag["status"] == "ok" else None
    if keep_dir is not None:  # audit option: keep the raw outputs for rescoring
        shutil.copytree(out_dir, keep_dir / name, dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    if measures is None:
        return {exp: dict(WORST) for exp in EXPERIMENTS}, diag
    diag["per_instance"] = {iid: {"experiment": anns[iid]["experiment"], **m} for iid, m in measures.items()}
    return aggregate(measures, anns), diag


def evaluate_all(root: Path, inputs: dict[str, dict], split: str, score_out: Path, diag_out: Path,
                 log_dir: Path) -> dict:
    anns = {iid: annotate(inp) for iid, inp in inputs.items()}
    methods = discover_methods(root)
    keep = os.environ.get("RETARGET_KEEP_OUTPUTS")
    keep_dir = Path(keep) / split if keep else None
    scores = {"experiments": {exp: {"results": {}} for exp in EXPERIMENTS}}
    diagnostics = {"split": split, "num_instances": len(inputs),
                   "instances_per_experiment": {e: sum(a["experiment"] == e for a in anns.values())
                                                for e in EXPERIMENTS},
                   "cpu_budget_per_instance_s": CPU_BUDGET_PER_INSTANCE, "methods": {}}
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
    return scores


def rescore_dir(out_dir: Path, inputs: dict[str, dict]) -> tuple[dict, dict]:
    """Score saved outputs of one method (audit helper; same validation and metrics)."""
    anns = {iid: annotate(inp) for iid, inp in inputs.items()}
    diag: dict = {"status": "ok", "reason": ""}
    measures = score_outputs(out_dir, inputs, anns, diag)
    if measures is None:
        return {exp: dict(WORST) for exp in EXPERIMENTS}, diag
    diag["per_instance"] = {iid: {"experiment": anns[iid]["experiment"], **m} for iid, m in measures.items()}
    return aggregate(measures, anns), diag
