"""Synthetic interaction tasks for the OmniRetarget reproduction.

The original paper evaluates on OMOMO, LAFAN1, and private MoCap. The
full robot assets and released retargeted data are not available in this
workspace, so this module defines small deterministic human-object and
human-terrain demonstrations that preserve the paper's problem structure:
retarget human keypoints to a morphology-mismatched humanoid while
preserving contact and satisfying kinematic constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYPOINTS = [
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


def terrain_height(x: np.ndarray | float, task: str) -> np.ndarray | float:
    x_arr = np.asarray(x)
    if task == "terrain_step":
        ramp = np.clip((x_arr - 0.40) / 0.15, 0.0, 1.0)
        out = 0.32 * ramp
    else:
        out = np.zeros_like(x_arr, dtype=float)
    if np.isscalar(x):
        return float(out)
    return out


def _human_body(t: float, task: str) -> dict[str, np.ndarray]:
    stride = 0.75
    x = -0.45 + stride * t
    ground = terrain_height(x, task)
    pelvis = np.array([x, ground + 0.92 + 0.025 * np.sin(2 * np.pi * t)])
    neck = pelvis + np.array([0.03 * np.sin(2 * np.pi * t), 0.52])
    shoulders = {
        "left_shoulder": neck + np.array([-0.19, -0.08]),
        "right_shoulder": neck + np.array([0.19, -0.08]),
    }
    phase = 2 * np.pi * t
    left_foot_x = x - 0.20 + 0.16 * np.sin(phase)
    right_foot_x = x + 0.20 - 0.16 * np.sin(phase)
    left_swing = max(0.0, np.sin(phase))
    right_swing = max(0.0, -np.sin(phase))
    left_foot = np.array([left_foot_x, terrain_height(left_foot_x, task) + 0.08 * left_swing])
    right_foot = np.array([right_foot_x, terrain_height(right_foot_x, task) + 0.08 * right_swing])
    left_knee = 0.54 * pelvis + 0.46 * left_foot + np.array([-0.04, 0.10])
    right_knee = 0.54 * pelvis + 0.46 * right_foot + np.array([0.04, 0.10])
    body = {
        "pelvis": pelvis,
        "neck": neck,
        "left_knee": left_knee,
        "left_foot": left_foot,
        "right_knee": right_knee,
        "right_foot": right_foot,
        **shoulders,
    }
    return body


def generate_task(task: str, frames: int = 80, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, frames)
    keypoints = np.zeros((frames, len(KEYPOINTS), 2), dtype=float)
    objects = []
    contacts: list[dict[str, list[str]]] = []
    terrains = []

    for ti, t in enumerate(ts):
        body = _human_body(float(t), task)
        if task == "object_carry":
            obj_center = np.array([-0.35 + 0.70 * t, 0.88 + 0.05 * np.sin(np.pi * t)])
            obj_size = np.array([0.34, 0.24])
            body["left_hand"] = obj_center + np.array([-obj_size[0] / 2 - 0.01, 0.01])
            body["right_hand"] = obj_center + np.array([obj_size[0] / 2 + 0.01, 0.01])
            body["left_elbow"] = 0.55 * body["left_shoulder"] + 0.45 * body["left_hand"] + np.array([-0.04, 0.04])
            body["right_elbow"] = 0.55 * body["right_shoulder"] + 0.45 * body["right_hand"] + np.array([0.04, 0.04])
            objects.append({"center": obj_center.tolist(), "size": obj_size.tolist(), "theta": 0.0})
            contacts.append({"object": ["left_hand", "right_hand"], "terrain": ["left_foot", "right_foot"]})
        else:
            # Hands brace on the upper platform surface while stepping up.
            rail = np.array([0.63, terrain_height(0.63, task) + 0.02])
            reach = np.clip((t - 0.28) / 0.30, 0.0, 1.0)
            body["left_hand"] = (1 - reach) * (body["left_shoulder"] + np.array([-0.10, -0.38])) + reach * (rail + np.array([-0.10, 0.0]))
            body["right_hand"] = (1 - reach) * (body["right_shoulder"] + np.array([0.10, -0.38])) + reach * (rail + np.array([0.10, 0.0]))
            body["left_elbow"] = 0.5 * (body["left_shoulder"] + body["left_hand"]) + np.array([-0.03, 0.06])
            body["right_elbow"] = 0.5 * (body["right_shoulder"] + body["right_hand"]) + np.array([0.03, 0.06])
            objects.append(None)
            hand_contacts = ["left_hand", "right_hand"] if reach > 0.9 else []
            contacts.append({"object": [], "terrain": ["left_foot", "right_foot"] + hand_contacts})

        for ki, name in enumerate(KEYPOINTS):
            keypoints[ti, ki] = body[name]

        xs = np.linspace(-0.9, 0.95, 18)
        zs = terrain_height(xs, task)
        terrains.append(np.stack([xs, zs], axis=1).tolist())

    # Deterministic dense surface points are part of the problem data, not method code.
    object_surface = []
    for obj in objects:
        if obj is None:
            object_surface.append([])
            continue
        c = np.array(obj["center"])
        sx, sz = obj["size"]
        pts = []
        for a in np.linspace(-0.5, 0.5, 5):
            pts.append(c + np.array([a * sx, -0.5 * sz]))
            pts.append(c + np.array([a * sx, 0.5 * sz]))
            pts.append(c + np.array([-0.5 * sx, a * sz]))
            pts.append(c + np.array([0.5 * sx, a * sz]))
        object_surface.append(np.unique(np.round(np.array(pts), 6), axis=0).tolist())

    return {
        "task": task,
        "dt": 1.0 / 30.0,
        "keypoint_names": KEYPOINTS,
        "source_keypoints": keypoints.tolist(),
        "objects": objects,
        "object_surface": object_surface,
        "terrain_points": terrains,
        "contacts": contacts,
        "notes": "Scaled synthetic substitute for unavailable OmniRetarget datasets.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_tasks"))
    parser.add_argument("--frames", type=int, default=80)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task in ["object_carry", "terrain_step"]:
        payload = generate_task(task, frames=args.frames)
        (args.out / f"{task}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
