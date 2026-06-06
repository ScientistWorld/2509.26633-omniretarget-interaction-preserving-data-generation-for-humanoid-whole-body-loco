"""Score the held-out test slice of retargeting artifacts.

This evaluator reads method-produced JSON artifacts from an output directory.
It does not import method code or regenerate trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SLICE = "test"
FRAME_POLICY = "odd_frame_indices"
NAMES = [
    "pelvis",
    "neck",
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
]
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
BODY_FOR_OBJECT_COLLISION = [n for n in NAMES if n not in ("left_hand", "right_hand")]
TRUSTED_TASKS = {
    "object_carry": Path("data/synthetic_tasks/object_carry.json"),
    "terrain_step": Path("data/synthetic_tasks/terrain_step.json"),
}
EXPERIMENTS = {
    "object_carry": "object_interaction_kinematics",
    "terrain_step": "terrain_interaction_kinematics",
}


def split_indices(num_frames: int) -> list[int]:
    return list(range(1, num_frames, 2))


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
    if len(pts) != len(task["source_keypoints"]):
        raise ValueError(
            f"Artifact for {output.get('task')} has {len(pts)} frames; "
            f"trusted task has {len(task['source_keypoints'])} frames"
        )

    indices = split_indices(len(pts))
    dt = float(task["dt"])
    src = np.array(task["source_keypoints"], dtype=float)
    pen_depths = []
    pen_frames = 0
    for t in indices:
        frame = pts[t]
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
    for prev_t, t in zip(indices, indices[1:]):
        dt_gap = (t - prev_t) * dt
        for foot in ("left_foot", "right_foot"):
            idx = NAME_TO_IDX[foot]
            src_v = abs(src[t, idx, 0] - src[prev_t, idx, 0]) / dt_gap
            prev_h = terrain_height(src[prev_t, idx, 0], task["task"])
            cur_h = terrain_height(src[t, idx, 0], task["task"])
            same_surface = abs(cur_h - prev_h) < 0.05
            if src_v < 0.35 and same_surface and src[t, idx, 1] <= cur_h + 0.03:
                stance_count += 1
                v = np.linalg.norm(pts[t, idx] - pts[prev_t, idx]) / dt_gap
                skate_speeds.append(v)
                skate_count += int(v > 0.01)

    contact_scores = []
    for t in indices:
        contact = task["contacts"][t]
        obj = task["objects"][t]
        for hand in contact.get("object", []):
            if not obj:
                continue
            tolerance = 0.055
            d = rect_signed_distance(pts[t, NAME_TO_IDX[hand]], np.array(obj["center"]), np.array(obj["size"]))
            contact_scores.append(max(0.0, 1.0 - abs(d) / tolerance))
        for name in [n for n in contact.get("terrain", []) if "hand" in n]:
            tolerance = 0.08
            p = pts[t, NAME_TO_IDX[name]]
            d = abs(p[1] - terrain_height(p[0], task["task"]))
            contact_scores.append(max(0.0, 1.0 - d / tolerance))

    pelvis_idx = NAME_TO_IDX["pelvis"]
    src_origin = src[indices[0], pelvis_idx]
    pts_origin = pts[indices[0], pelvis_idx]
    src_pelvis_path = src[indices, pelvis_idx] - src_origin
    pts_pelvis_path = pts[indices, pelvis_idx] - pts_origin
    pelvis_path_error = np.linalg.norm(pts_pelvis_path - src_pelvis_path, axis=1)

    velocity_errors = []
    for prev_t, t in zip(indices, indices[1:]):
        dt_gap = (t - prev_t) * dt
        output_velocity = (pts[t] - pts[prev_t]) / dt_gap
        source_velocity = (src[t] - src[prev_t]) / dt_gap
        velocity_errors.append(np.linalg.norm(output_velocity - source_velocity, axis=1))

    return {
        "penetration_duration": float(pen_frames / max(1, len(indices))),
        "max_penetration_cm": float(100.0 * max(pen_depths or [0.0])),
        "foot_skating_duration": float(skate_count / max(1, stance_count)),
        "max_foot_skating_cm_s": float(100.0 * max(skate_speeds or [0.0])),
        "contact_preservation": float(np.mean(contact_scores)) if contact_scores else 0.0,
        "pelvis_path_error_cm": float(100.0 * np.mean(pelvis_path_error)),
        "motion_velocity_error_cm_s": float(100.0 * np.mean(velocity_errors)) if velocity_errors else 0.0,
    }


def expected_methods(train_scores: Path) -> dict[str, set[str]] | None:
    if not train_scores.exists():
        return None

    data = json.loads(train_scores.read_text())
    experiments = data.get("experiments", data)
    expected: dict[str, set[str]] = {}
    for exp, exp_data in experiments.items():
        if exp.startswith("_"):
            continue
        results = exp_data.get("results", exp_data)
        expected[exp] = {method for method in results if not method.startswith("_")}
    return expected


def score_outputs(outputs: Path, train_scores: Path) -> tuple[dict, dict]:
    expected = expected_methods(train_scores)
    exp_names = sorted(set(EXPERIMENTS.values()))
    scores = {"experiments": {exp: {"results": {}} for exp in exp_names}}
    diagnostics = {
        "slice": SLICE,
        "frame_policy": FRAME_POLICY,
        "train_scores": str(train_scores),
        "experiments": {exp: {"results": {}} for exp in exp_names},
    }
    seen: set[tuple[str, str]] = set()

    for path in sorted(outputs.glob("*.json")):
        out = json.loads(path.read_text())
        task_name = out["task"]
        method = out["method"]
        if task_name not in TRUSTED_TASKS:
            raise ValueError(f"Unknown task id in artifact {path}: {task_name}")
        key = (method, task_name)
        if key in seen:
            raise ValueError(f"Duplicate artifact for method={method!r}, task={task_name!r}")
        seen.add(key)

        exp = EXPERIMENTS[task_name]
        if expected is not None and method not in expected.get(exp, set()):
            continue

        task = json.loads(TRUSTED_TASKS[task_name].read_text())
        metrics = task_metrics(task, out)
        scores["experiments"][exp]["results"][method] = metrics
        diagnostics["experiments"][exp]["results"][method] = {
            "slice": SLICE,
            "task": task_name,
            "num_scored_frames": len(split_indices(len(task["source_keypoints"]))),
            "metrics": metrics,
        }

    if expected is not None:
        for exp, methods in expected.items():
            scored = set(scores["experiments"].get(exp, {"results": {}})["results"])
            if scored != methods:
                missing = sorted(methods - scored)
                extra = sorted(scored - methods)
                raise ValueError(
                    f"Test slice method keys for {exp} do not match scores_train.json; "
                    f"missing={missing}, extra={extra}"
                )

    return scores, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("checkpoints/retargeting"))
    parser.add_argument("--score-out", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--diagnostics-out", type=Path, default=Path("scoring/diagnostics_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    scores, diagnostics = score_outputs(args.outputs, args.train_scores)
    args.score_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.score_out.write_text(json.dumps(scores, indent=2))
    args.diagnostics_out.write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
