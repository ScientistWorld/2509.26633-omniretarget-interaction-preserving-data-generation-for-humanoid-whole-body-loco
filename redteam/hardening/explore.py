"""Design exploration for the iteration-12 hardening (one job, many candidate setups).

For each candidate setup (scene convention, proportion jitter, augmentation size)
a dev split is generated with a dev seed (neither the train nor the test seed),
every candidate method is run on it, and metrics are computed from q:

  v1  the iteration-11 metric set (fidelity vs the raw demonstration)
  v2  fidelity vs the demonstration at the robot's scale, pelvis height taken
      relative to the supporting ground (adaptation to a changed scene is not
      penalised)

Rank deltas vs the candidate reference use the gym's formula (weight x |coef| x
directed improvement / population sd across honest methods), with paired
bootstrap intervals over instances.

Usage: python redteam/hardening/explore.py [run|summary] [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from baseline import keypoint_ik as KIK
from data import task_generator as G
from eval.train.scoring_core import (CAPS, CONTACT_TOLERANCE, LINKS, PENETRATION_FRAME_DEPTH, Q_MAX, Q_MIN,
                                     SKATING_SPEED, V_MAX, box_sdf, fk_batch, link_samples, terrain_sdf)
from method import omniretarget as OR
from redteam.hardening import variants as V

ROOT = Path(__file__).resolve().parents[2]
DEV_SEED = 777001
PER_EXP = int(os.environ.get("EXPLORE_PER_EXP", "20"))

SETUPS = {
    "C0_human_small": dict(scene_mode="human", prop_jitter=0.06, aug_level="small", scale_mode="per_frame"),
    "C1_robot_small": dict(scene_mode="robot", prop_jitter=0.06, aug_level="small", scale_mode="global"),
    "C2_robot_large": dict(scene_mode="robot", prop_jitter=0.06, aug_level="large", scale_mode="global"),
    "C3_robot_large_prop12": dict(scene_mode="robot", prop_jitter=0.12, aug_level="large", scale_mode="global"),
    "C4_robot_large_objramp": dict(scene_mode="robot", prop_jitter=0.06, aug_level="large", scale_mode="global",
                                   object_terrain=True),
}
GEN_KEYS = ("scene_mode", "prop_jitter", "aug_level", "object_terrain")


def standing(task):
    src = np.asarray(task["source_keypoints"], dtype=float)
    q = np.array([src[0, 0, 0], 0.78, 0.0, -0.18, 0.36, 0.18, -0.36, -0.2, 0.3, 0.2, -0.3])
    return np.repeat(q[None], len(src), axis=0)


def run_method(name: str, task: dict, sm: str) -> np.ndarray:
    if name == "refA":
        return OR.retarget(task, scale_mode=sm, ground_frame0=False, augment_anchor=False)
    if name == "refB":
        return OR.retarget(task, scale_mode=sm)
    if name == "refC":
        return OR.retarget(task, scale_mode=sm, terrain_spacing=0.025, box_per_side=12)
    if name == "no_mesh":
        return OR.retarget(task, objective="keypoint", scale_mode=sm)
    if name == "kc":
        return OR.retarget(task, objective="keypoint", extra_factory=V.contact_terms, scale_mode=sm)
    if name == "kc100":
        return OR.retarget(task, objective="keypoint", scale_mode=sm,
                           extra_factory=lambda tk, sc: V.contact_terms(tk, sc, 100.0, 100.0))
    if name == "kc_it11":
        return OR.retarget(task, objective="keypoint", extra_factory=V.contact_terms, scale_mode=sm,
                           ground_frame0=False, augment_anchor=False)
    if name == "ca":
        return OR.retarget(task, extra_factory=V.contact_terms, scale_mode=sm)
    if name == "kik":
        return KIK.retarget(task, scale_mode=sm)
    if name == "const":
        return standing(task)
    raise KeyError(name)


METHODS = ["refB", "refA", "refC", "no_mesh", "kc", "kc100", "kc_it11", "ca", "kik", "const"]
HONEST = ["refB", "no_mesh", "kc", "ca", "kik", "const"]
EXPERIMENTS = ["object_interaction_kinematics", "terrain_interaction_kinematics", "augmented_interaction_kinematics"]
WEIGHTS = {"object_interaction_kinematics": 0.35, "terrain_interaction_kinematics": 0.35,
           "augmented_interaction_kinematics": 0.30}
COEF = {"penetration_duration": 1.0, "max_penetration_cm": 0.5, "foot_skating_duration": 0.8,
        "max_foot_skating_cm_s": 0.4, "contact_preservation": 1.0, "pelvis_path_error_cm": 0.3,
        "motion_velocity_error_cm_s": 0.3}
METRIC_SETS = {
    "v1": {m: m for m in COEF},
    "v2": {**{m: m for m in COEF}, "pelvis_path_error_cm": "pelvis_path_error_cm_v2",
           "motion_velocity_error_cm_s": "motion_velocity_error_cm_s_v2"},
}
HIGHER = {"contact_preservation"}


def valid(q, inp):
    N = int(inp["num_frames"])
    if q.shape != (N, 11) or not np.all(np.isfinite(q)):
        return "shape/finite"
    if np.any(q < Q_MIN - 1e-6) or np.any(q > Q_MAX + 1e-6):
        return "joint limit"
    if np.any(np.abs(np.diff(q, axis=0)) > V_MAX * float(inp["dt"]) + 1e-6):
        return "velocity limit"
    return ""


def metrics(inp: dict, ann: dict, q: np.ndarray) -> dict:
    dt = float(inp["dt"])
    P = fk_batch(q)
    src = np.asarray(inp["source_keypoints"], dtype=float)
    scene = inp["target_scene"]
    terrain = np.asarray(scene["terrain"], dtype=float)
    src_terrain = np.asarray(inp["source_scene"]["terrain"], dtype=float)
    obj = scene.get("object")
    centers = np.asarray(obj["center"], dtype=float) if obj else None
    size = np.asarray(obj["size"], dtype=float) if obj else None
    S = link_samples(P)
    d_all = np.maximum(0.0, -terrain_sdf(S, terrain))
    if obj:
        d_all = np.maximum(d_all, np.maximum(0.0, -box_sdf(S, centers[:, None, :], size)))
    depth = np.max(d_all, axis=1)
    speeds, foot_c, hand_c, scuff = [], [], [], []
    for name, idx in (("left_foot", 3), ("right_foot", 5)):
        st = np.asarray(ann["stance"][name], dtype=bool)
        ev = st[1:] & st[:-1]
        v = np.linalg.norm(P[1:, idx] - P[:-1, idx], axis=1) / dt
        speeds.append(v[ev])
        foot_c.append(np.abs(terrain_sdf(P[st, idx], terrain)) <= CONTACT_TOLERANCE)
        h_demo = src[:, idx, 1] - np.interp(src[:, idx, 0], src_terrain[:, 0], src_terrain[:, 1])
        sw = h_demo >= 0.03
        if np.any(sw):
            scuff.append(np.abs(terrain_sdf(P[sw, idx], terrain)) <= 0.01)
    for name, idx in (("left_hand", 8), ("right_hand", 11)):
        for t, label in enumerate(ann["hand_contact"][name]):
            if label == "object" and obj:
                d = float(box_sdf(P[t, idx], centers[t], size))
            elif label == "terrain":
                d = float(terrain_sdf(P[t, idx], terrain))
            else:
                continue
            hand_c.append(abs(d) <= CONTACT_TOLERANCE)
    speeds = np.concatenate(speeds)
    foot_c = np.concatenate(foot_c)
    hand_c = np.asarray(hand_c, dtype=bool)
    contacts = np.concatenate([foot_c, hand_c])
    pel_err = np.linalg.norm((P[:, 0] - P[0, 0]) - (src[:, 0] - src[0, 0]), axis=1)
    vel_err = np.linalg.norm(np.diff(P, axis=0) - np.diff(src, axis=0), axis=2) / dt
    alpha = float(ann.get("alpha", 1.0))

    def rel(pel, terr):
        h = pel[:, 1] - np.interp(pel[:, 0], terr[:, 0], terr[:, 1])
        return np.stack([pel[:, 0] - pel[0, 0], h - h[0]], axis=1)

    pel2 = np.linalg.norm(rel(P[:, 0], terrain) - alpha * rel(src[:, 0], src_terrain), axis=1)
    vel2 = np.linalg.norm(np.diff(P, axis=0) - alpha * np.diff(src, axis=0), axis=2) / dt
    raw = {
        "penetration_duration": float(np.mean(depth > PENETRATION_FRAME_DEPTH)),
        "max_penetration_cm": 100.0 * float(np.max(depth)),
        "foot_skating_duration": float(np.mean(speeds > SKATING_SPEED)) if len(speeds) else 0.0,
        "max_foot_skating_cm_s": 100.0 * float(np.max(speeds)) if len(speeds) else 0.0,
        "contact_preservation": float(np.mean(contacts)) if len(contacts) else 0.0,
        "pelvis_path_error_cm": 100.0 * float(np.mean(pel_err)),
        "motion_velocity_error_cm_s": 100.0 * float(np.mean(vel_err)),
    }
    out = {m: float(min(max(raw[m], 0.0), CAPS[m])) for m in raw}
    out["pelvis_path_error_cm_v2"] = float(min(100.0 * np.mean(pel2), 100.0))
    out["motion_velocity_error_cm_s_v2"] = float(min(100.0 * np.mean(vel2), 300.0))
    out["foot_contact"] = float(np.mean(foot_c)) if len(foot_c) else None
    out["hand_contact"] = float(np.mean(hand_c)) if len(hand_c) else None
    out["swing_scuff"] = float(np.mean(np.concatenate(scuff))) if scuff else None
    return out


def _job(args):
    setup, method, inp, ann = args
    sm = SETUPS[setup]["scale_mode"]
    t0 = time.process_time()
    rec = {"setup": setup, "method": method, "iid": inp["instance_id"], "experiment": ann["experiment"]}
    try:
        q = np.asarray(run_method(method, inp, sm), dtype=float)
        rec["cpu_s"] = time.process_time() - t0
        why = valid(q, inp)
        rec["status"] = "ok" if not why else why
        rec.update(metrics(inp, ann, q))
    except Exception as exc:  # keep going; record the failure
        rec["cpu_s"] = time.process_time() - t0
        rec["status"] = f"error: {type(exc).__name__}: {exc}"
        rec["trace"] = traceback.format_exc()[-2000:]
    return rec


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "records.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done.add((r["setup"], r["method"], r["iid"]))
    jobs = []
    for setup, cfg in SETUPS.items():
        t0 = time.time()
        split = G.make_split(DEV_SEED, PER_EXP, "dev", cfg={k: cfg[k] for k in GEN_KEYS if k in cfg})
        print(f"[gen] {setup}: {len(split)} instances in {time.time() - t0:.1f}s", flush=True)
        for method in METHODS:
            for inp, ann in split:
                if (setup, method, inp["instance_id"]) not in done:
                    jobs.append((setup, method, inp, ann))
    ncpu = max(1, len(os.sched_getaffinity(0)))
    print(f"[run] {len(jobs)} jobs on {ncpu} cpus", flush=True)
    t0 = time.time()
    with open(log, "a") as fh, Pool(ncpu) as pool:
        for i, rec in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0 or rec["status"] != "ok":
                print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {rec['setup']} {rec['method']} {rec['iid']} "
                      f"{rec['status']} cpu={rec.get('cpu_s', 0):.2f}", flush=True)


def _means(recs, method, exp, keys):
    rs = [r for r in recs if r["method"] == method and r["experiment"] == exp]
    out = {}
    for k in keys:
        vals = [r[k] for r in rs if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _delta(mc, mr, sds, mset):
    tot = 0.0
    for exp in EXPERIMENTS:
        for m, key in METRIC_SETS[mset].items():
            sd = sds[exp][key]
            if sd <= 0:
                continue
            sgn = 1.0 if m in HIGHER else -1.0
            tot += WEIGHTS[exp] * COEF[m] * sgn * (mc[exp][key] - mr[exp][key]) / sd
    return tot


def summary(out_dir: Path) -> None:
    recs = [json.loads(l) for l in (out_dir / "records.jsonl").read_text().splitlines()]
    keys = sorted({k for m in METRIC_SETS.values() for k in m.values()} |
                  {"foot_contact", "hand_contact", "swing_scuff", "cpu_s"})
    lines = []
    report = {}
    for setup in SETUPS:
        R = [r for r in recs if r["setup"] == setup]
        bad = [r for r in R if r["status"] != "ok"]
        methods = [m for m in METHODS if any(r["method"] == m for r in R)]
        lines.append(f"\n## {setup}  ({len(R)} records, {len(bad)} not ok)\n")
        for r in bad[:10]:
            lines.append(f"- NOT OK {r['method']} {r['iid']}: {r['status']}")
        ok_methods = [m for m in methods if not any(r["method"] == m and r["status"] != "ok" for r in R)]
        means = {m: {e: _means(R, m, e, keys) for e in EXPERIMENTS} for m in methods}
        for exp in EXPERIMENTS:
            lines.append(f"\n{exp}\n")
            hdr = ["pen_d", "pen_max", "sk_d", "sk_max", "contact", "foot_c", "hand_c", "pel1", "vel1", "pel2", "vel2", "scuff", "cpu"]
            ks = ["penetration_duration", "max_penetration_cm", "foot_skating_duration", "max_foot_skating_cm_s",
                  "contact_preservation", "foot_contact", "hand_contact", "pelvis_path_error_cm",
                  "motion_velocity_error_cm_s", "pelvis_path_error_cm_v2", "motion_velocity_error_cm_s_v2",
                  "swing_scuff", "cpu_s"]
            lines.append("| method | " + " | ".join(hdr) + " |")
            lines.append("|---" * (len(hdr) + 1) + "|")
            for m in methods:
                v = means[m][exp]
                lines.append(f"| {m} | " + " | ".join("-" if v[k] is None else f"{v[k]:.3f}" for k in ks) + " |")
        honest = [m for m in HONEST if m in ok_methods]
        rep = {}
        for mset in METRIC_SETS:
            sds = {e: {key: float(np.std([means[h][e][key] for h in honest]))
                       for key in METRIC_SETS[mset].values()} for e in EXPERIMENTS}
            ids = {e: sorted({r["iid"] for r in R if r["experiment"] == e}) for e in EXPERIMENTS}
            per = {m: {r["iid"]: r for r in R if r["method"] == m} for m in methods}
            rng = np.random.default_rng(0)
            picks = {e: rng.integers(0, len(ids[e]), size=(1000, len(ids[e]))) for e in EXPERIMENTS}
            for ref in ("refB", "refA", "refC"):
                if ref not in ok_methods:
                    continue
                rows = []
                for m in methods:
                    if m == ref or m not in ok_methods:
                        continue
                    d = _delta(means[m], means[ref], sds, mset)
                    boots = np.zeros(1000)
                    for e in EXPERIMENTS:
                        for mname, key in METRIC_SETS[mset].items():
                            sd = sds[e][key]
                            if sd <= 0:
                                continue
                            diff = np.array([per[m][i][key] - per[ref][i][key] for i in ids[e]])
                            sgn = 1.0 if mname in HIGHER else -1.0
                            boots += WEIGHTS[e] * COEF[mname] * sgn * diff[picks[e]].mean(axis=1) / sd
                    boots.sort()
                    rows.append((m, d, float(boots[25]), float(boots[974])))
                rep[f"{mset}_vs_{ref}"] = {m: [d, lo, hi] for m, d, lo, hi in rows}
                lines.append(f"\nrank delta ({mset}) vs {ref}: " + ", ".join(
                    f"{m} {d:+.2f} [{lo:+.2f},{hi:+.2f}]" for m, d, lo, hi in sorted(rows, key=lambda x: -x[1])))
        report[setup] = {"means": means, "ranks": rep}
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "redteam/hardening/results/explore"
    if mode in ("run", "all"):
        run(out)
    if mode in ("summary", "all", "run"):
        summary(out)
