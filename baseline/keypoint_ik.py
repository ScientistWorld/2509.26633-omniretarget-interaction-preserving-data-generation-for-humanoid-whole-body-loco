"""Keypoint-matching IK baseline (GMR / PHC objective family, paper App. B).

Per frame, minimise sum_i || f_i(q) - p_i^source ||^2 subject only to joint
limits (and the robot's velocity limits, as box bounds), warm-started from
the previous frame.  Following the paper's "direct scaling" preprocessing
(App. A, Eq. 7/8), human keypoints are scaled by the robot-to-human height
ratio about the ground point below the demonstrator's initial pelvis (the
same scaling that brings the demonstration scene to the robot's scale).
The baseline does not look at objects or terrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from data.planar_robot import NAME_TO_IDX, Q_MAX, Q_MIN, PlanarHumanoid, per_frame_velocity_bounds
from data.scene_geometry import terrain_height

ROBOT = PlanarHumanoid()


def human_height(src0: np.ndarray) -> float:
    P = lambda n: src0[NAME_TO_IDX[n]]  # noqa: E731
    torso = np.linalg.norm(P("neck") - P("pelvis"))
    thigh = 0.5 * (np.linalg.norm(P("left_knee") - P("pelvis")) + np.linalg.norm(P("right_knee") - P("pelvis")))
    shin = 0.5 * (np.linalg.norm(P("left_foot") - P("left_knee")) + np.linalg.norm(P("right_foot") - P("right_knee")))
    return float(torso + thigh + shin)


def scaled_source(task: dict, scale_mode: str = "global") -> np.ndarray:
    src = np.asarray(task["source_keypoints"], dtype=float)
    alpha = ROBOT.height / human_height(src[0])
    pel = src[:, NAME_TO_IDX["pelvis"]]
    ground = terrain_height(pel[:, 0], task["source_scene"]["terrain"])
    anchor = np.stack([pel[:, 0], ground], axis=1)[:, None, :]
    if scale_mode == "global":
        anchor = np.repeat(anchor[:1], len(src), axis=0)
    return anchor + alpha * (src - anchor)


def retarget(task: dict, scale_mode: str = "global") -> np.ndarray:
    tgt = scaled_source(task, scale_mode)
    N = len(tgt)
    vstep = per_frame_velocity_bounds(float(task["dt"]))
    q_prev = None
    out = np.zeros((N, len(Q_MIN)))
    for t in range(N):
        target = tgt[t]
        if q_prev is None:
            pel = target[NAME_TO_IDX["pelvis"]]
            q0 = np.array([pel[0], min(pel[1], 0.78), 0.0, -0.15, 0.3, 0.15, -0.3, -0.3, 0.3, 0.3, -0.3])
            lo, hi = Q_MIN, Q_MAX
        else:
            q0 = q_prev
            lo = np.maximum(Q_MIN, q_prev - vstep)
            hi = np.minimum(Q_MAX, q_prev + vstep)
        q0 = np.clip(q0, lo, hi)

        def objective(q, q0=q0, target=target):
            fk, J = ROBOT.fk_jacobian(q)
            e = fk - target
            val = float(np.sum(e ** 2) + 1e-3 * np.sum((q - q0) ** 2))
            grad = 2.0 * np.einsum("kd,kdj->j", e, J) + 2e-3 * (q - q0)
            return val, grad

        res = minimize(objective, q0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options={"maxiter": 300 if q_prev is None else 80})
        q_prev = np.clip(np.asarray(res.x, dtype=float), lo, hi)
        out[t] = q_prev
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    args = parser.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.inputs.glob("*.json")):
        task = json.loads(path.read_text())
        q = retarget(task)
        out = {"instance_id": task["instance_id"], "method": "keypoint_ik", "q": np.round(q, 10).tolist()}
        (args.outputs / f"{task['instance_id']}.json").write_text(json.dumps(out))
        print(f"[keypoint_ik] {task['instance_id']} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
