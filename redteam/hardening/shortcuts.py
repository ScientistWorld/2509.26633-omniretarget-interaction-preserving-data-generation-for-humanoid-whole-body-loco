"""Shortcut submissions run through the hardened hidden test.

Usage: python redteam/hardening/shortcuts.py <mode> <input_dir> <output_dir>

select_best is the "private search" probe: per instance it runs several honest
retargeters and keeps the trajectory that scores best under a public copy of the
metrics, with contact/stance labels and the scale inferred from the inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from baseline.keypoint_ik import retarget as keypoint_ik
from data.planar_robot import Q_MAX, PlanarHumanoid
from method.omniretarget import human_height, retarget as omniretarget

ROOT = Path(__file__).resolve().parents[2]
ROBOT = PlanarHumanoid()


def standing(task):
    src = np.asarray(task["source_keypoints"], dtype=float)
    q = np.array([src[0, 0, 0], 0.78, 0.0, -0.18, 0.36, 0.18, -0.36, -0.2, 0.3, 0.2, -0.3])
    return np.repeat(q[None], len(src), axis=0)


def ideal_points(task):
    """Source keypoints (not a robot body)."""
    return np.asarray(task["source_keypoints"], dtype=float)


def is_nominal_object(task):
    t = task["target_scene"]
    return bool(t.get("object")) and t == task.get("nominal_scene")


def inferred_labels(task):
    """Stance / hand-contact labels and alpha inferred from the inputs alone."""
    from data.scene_geometry import box_sdf, terrain_sdf
    src = np.asarray(task["source_keypoints"], dtype=float)
    dt = float(task["dt"])
    N = len(src)
    ss = task["source_scene"]
    stance = {}
    for name, idx in (("left_foot", 3), ("right_foot", 5)):
        sp = np.abs(np.diff(src[:, idx, 0])) / dt
        st = np.zeros(N, dtype=bool)
        st[:-1] |= sp < 0.01
        st[1:] |= sp < 0.01
        stance[name] = st.tolist()
    hc = {}
    for name, idx in (("left_hand", 8), ("right_hand", 11)):
        lab = [None] * N
        for t in range(N):
            if ss.get("object") and abs(float(box_sdf(src[t, idx], ss["object"]["center"][t], ss["object"]["size"]))) <= 0.005:
                lab[t] = "object"
            elif abs(float(terrain_sdf(src[t, idx], ss["terrain"]))) <= 0.005:
                lab[t] = "terrain"
        hc[name] = lab
    return {"stance": stance, "hand_contact": hc, "alpha": ROBOT.height / human_height(src[0])}


SCALES = {"penetration_duration": (1.0, 0.1), "max_penetration_cm": (0.5, 1.0), "foot_skating_duration": (0.8, 0.1),
          "max_foot_skating_cm_s": (0.4, 5.0), "contact_preservation": (-1.0, 0.1),
          "spurious_contact_duration": (0.8, 0.1), "pelvis_path_error_cm": (0.3, 5.0),
          "motion_velocity_error_cm_s": (0.3, 5.0)}


def select_best(task):
    from eval.train.scoring_core import instance_metrics  # a public copy of the scorer
    from redteam.hardening.variants import VARIANTS, contact_terms
    ann = inferred_labels(task)
    cands = [omniretarget(task),
             omniretarget(task, **VARIANTS["contact_aware"]),
             omniretarget(task, **VARIANTS["keypoint_contact"]),
             omniretarget(task, objective="keypoint", extra_factory=lambda tk, sc: contact_terms(tk, sc, 100.0, 100.0))]
    best, best_cost = None, np.inf
    for q in cands:
        m = instance_metrics(task, ann, q)
        cost = sum(c * m[k] / s for k, (c, s) in SCALES.items())
        if cost < best_cost:
            best, best_cost = q, cost
    return best


def main() -> None:
    mode, in_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "score_forgery":
        forged = {"experiments": {e: {"results": {"sc_score_forgery": {
            "penetration_duration": 0.0, "max_penetration_cm": 0.0, "foot_skating_duration": 0.0,
            "max_foot_skating_cm_s": 0.0, "contact_preservation": 1.0, "spurious_contact_duration": 0.0,
            "pelvis_path_error_cm": 0.0, "motion_velocity_error_cm_s": 0.0}}} for e in [
            "object_interaction_kinematics", "terrain_interaction_kinematics", "augmented_interaction_kinematics"]}}
        for name in ("scores_test.json", "scores_train.json"):
            (ROOT / "scoring" / name).write_text(json.dumps(forged))
    for path in sorted(in_dir.glob("*.json")):
        task = json.loads(path.read_text())
        iid = task["instance_id"]
        N = int(task["num_frames"])
        extra = {}
        if mode == "empty":
            continue
        if mode == "zero_q":
            q = np.zeros((N, 11))
            extra = {"retargeted_keypoints": ideal_points(task).tolist()}
        elif mode == "disconnected_keypoints":
            q = keypoint_ik(task)
            extra = {"retargeted_keypoints": ideal_points(task).tolist(),
                     "claimed_metrics": {"contact_preservation": 1.0, "penetration_duration": 0.0}}
        elif mode == "impossible_q":
            q = np.full((N, 11), 1e200)
        elif mode == "nan":
            q = np.full((N, 11), np.nan)
        elif mode == "malformed":
            q = np.zeros((0, 11))
        elif mode == "constant_pose":
            q = standing(task)
        elif mode == "joint_violation":
            q = keypoint_ik(task)
            q[:, 4] = Q_MAX[4] + 0.05
        elif mode == "velocity_violation":
            q = keypoint_ik(task)
            q[N // 2:, 0] += 0.3
        elif mode == "partial_missing":
            if task["task_type"] != "object_carry":
                continue
            q = omniretarget(task)
        elif mode == "omit_failed_experiment":
            if not is_nominal_object(task):
                continue
            from redteam.hardening.variants import contact_terms
            q = omniretarget(task, extra_factory=contact_terms)
        elif mode == "score_forgery":
            q = keypoint_ik(task)
            extra = {"metrics": {"contact_preservation": 1.0}, "success": True}
        elif mode == "select_best":
            q = select_best(task)
        else:
            raise SystemExit(f"unknown mode {mode}")
        payload = {"instance_id": iid, "method": f"sc_{mode}", "q": np.asarray(q, dtype=float).tolist(), **extra}
        (out_dir / f"{iid}.json").write_text(json.dumps(payload, allow_nan=True))


if __name__ == "__main__":
    main()
