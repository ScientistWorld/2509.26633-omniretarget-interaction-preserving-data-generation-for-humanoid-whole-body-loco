"""Evaluate retargeted trajectories without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NAMES = ["pelvis", "neck", "left_knee", "left_foot", "right_knee", "right_foot",
         "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
BODY_FOR_OBJECT_COLLISION = [n for n in NAMES if n not in ("left_hand", "right_hand")]


def terrain_height(x: np.ndarray | float, task: str) -> np.ndarray | float:
    x_arr = np.asarray(x)
    if task == "terrain_step":
        ramp = np.clip((x_arr - 0.40) / 0.15, 0.0, 1.0)
        out = 0.32 * ramp
    else:
        out = np.zeros_like(x_arr, dtype=float)
    return float(out) if np.isscalar(x) else out


def rect_signed_distance(p: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    q = np.abs(p - center) - size / 2
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1]), 0.0)
    return float(outside + inside)


def task_metrics(task: dict, output: dict) -> dict[str, float]:
    pts = np.array(output["retargeted_keypoints"], dtype=float)
    dt = float(task["dt"])
    pen_depths = []
    pen_frames = 0
    for t, frame in enumerate(pts):
        frame_pen = 0.0
        for i, p in enumerate(frame):
            d = p[1] - terrain_height(p[0], task["task"])
            frame_pen = max(frame_pen, max(0.0, -d))
            obj = task["objects"][t]
            if obj and NAMES[i] in BODY_FOR_OBJECT_COLLISION:
                od = rect_signed_distance(p, np.array(obj["center"]), np.array(obj["size"]))
                frame_pen = max(frame_pen, max(0.0, -od))
        pen_depths.append(frame_pen)
        pen_frames += int(frame_pen > 1e-4)

    skate_speeds = []
    stance_count = 0
    skate_count = 0
    src = np.array(task["source_keypoints"], dtype=float)
    for t in range(1, len(pts)):
        for foot in ("left_foot", "right_foot"):
            idx = NAME_TO_IDX[foot]
            src_v = abs(src[t, idx, 0] - src[t - 1, idx, 0]) / dt
            prev_h = terrain_height(src[t - 1, idx, 0], task["task"])
            cur_h = terrain_height(src[t, idx, 0], task["task"])
            same_surface = abs(cur_h - prev_h) < 0.05
            if src_v < 0.35 and same_surface and src[t, idx, 1] <= cur_h + 0.03:
                stance_count += 1
                v = np.linalg.norm(pts[t, idx] - pts[t - 1, idx]) / dt
                skate_speeds.append(v)
                skate_count += int(v > 0.01)

    contact_desired = 0
    contact_hit = 0
    for t, contact in enumerate(task["contacts"]):
        obj = task["objects"][t]
        for hand in contact.get("object", []):
            if not obj:
                continue
            contact_desired += 1
            d = rect_signed_distance(pts[t, NAME_TO_IDX[hand]], np.array(obj["center"]), np.array(obj["size"]))
            contact_hit += int(abs(d) < 0.055)
        for name in [n for n in contact.get("terrain", []) if "hand" in n]:
            contact_desired += 1
            p = pts[t, NAME_TO_IDX[name]]
            contact_hit += int(abs(p[1] - terrain_height(p[0], task["task"])) < 0.08)

    return {
        "penetration_duration": float(pen_frames / len(pts)),
        "max_penetration_cm": float(100.0 * max(pen_depths or [0.0])),
        "foot_skating_duration": float(skate_count / max(1, stance_count)),
        "max_foot_skating_cm_s": float(100.0 * max(skate_speeds or [0.0])),
        "contact_preservation": float(contact_hit / max(1, contact_desired)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("checkpoints/retargeting"))
    parser.add_argument("--score-out", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    scores = {"experiments": {"object_interaction_kinematics": {"results": {}}, "terrain_interaction_kinematics": {"results": {}}}}
    mapping = {"object_carry": "object_interaction_kinematics", "terrain_step": "terrain_interaction_kinematics"}
    for path in sorted(args.outputs.glob("*.json")):
        out = json.loads(path.read_text())
        task = json.loads(Path(out["source_file"]).read_text())
        exp = mapping[out["task"]]
        scores["experiments"][exp]["results"][out["method"]] = task_metrics(task, out)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
