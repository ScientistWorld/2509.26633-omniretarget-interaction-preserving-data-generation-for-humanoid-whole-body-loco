"""Shortcut submissions run through the hardened hidden test.

Usage: python redteam/hardening/shortcuts.py <mode> <input_dir> <output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from baseline.keypoint_ik import retarget as keypoint_ik
from data.planar_robot import Q_MAX, PlanarHumanoid
from method.omniretarget import retarget as omniretarget

ROOT = Path(__file__).resolve().parents[2]
ROBOT = PlanarHumanoid()


def standing(task):
    src = np.asarray(task["source_keypoints"], dtype=float)
    q = np.array([src[0, 0, 0], 0.78, 0.0, -0.18, 0.36, 0.18, -0.36, -0.2, 0.3, 0.2, -0.3])
    return np.repeat(q[None], len(src), axis=0)


def ideal_points(task):
    """Source keypoints with hands snapped onto the target scene (not a robot body)."""
    return np.asarray(task["source_keypoints"], dtype=float)


def is_nominal_object(task):
    s, t = task["source_scene"], task["target_scene"]
    return bool(s.get("object")) and s == t


def main() -> None:
    mode, in_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "score_forgery":
        forged = {"experiments": {e: {"results": {"sc_score_forgery": {
            "penetration_duration": 0.0, "max_penetration_cm": 0.0, "foot_skating_duration": 0.0,
            "max_foot_skating_cm_s": 0.0, "contact_preservation": 1.0, "pelvis_path_error_cm": 0.0,
            "motion_velocity_error_cm_s": 0.0}}} for e in [
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
            q = omniretarget(task, extra_residual=contact_terms(task))
        elif mode == "score_forgery":
            q = keypoint_ik(task)
            extra = {"metrics": {"contact_preservation": 1.0}, "success": True}
        else:
            raise SystemExit(f"unknown mode {mode}")
        payload = {"instance_id": iid, "method": f"sc_{mode}", "q": np.asarray(q, dtype=float).tolist(), **extra}
        (out_dir / f"{iid}.json").write_text(json.dumps(payload, allow_nan=True))


if __name__ == "__main__":
    main()
