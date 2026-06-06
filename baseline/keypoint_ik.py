"""Keypoint-position IK baseline, matching the PHC/GMR objective family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from data.planar_robot import KEYPOINTS as NAMES
from data.planar_robot import PlanarHumanoid, initial_guess


def run(task_path: Path, out_path: Path) -> None:
    task = json.loads(task_path.read_text())
    robot = PlanarHumanoid()
    q_prev = None
    qs, fks = [], []
    q_min = np.array([-2.0, 0.45, -0.8, -1.6, -0.5, -1.6, -2.1, -2.8, -0.1, -0.1, -2.8])
    q_max = np.array([2.0, 1.55, 0.8, 1.6, 2.1, 1.6, 0.5, 0.3, 2.8, 2.8, 0.3])
    for source_list in task["source_keypoints"]:
        source = np.array(source_list, dtype=float)
        q0 = q_prev if q_prev is not None else initial_guess(source)

        def objective(q: np.ndarray) -> float:
            return float(np.sum((robot.fk(q) - source) ** 2) + 0.01 * np.sum((q - q0) ** 2))

        res = minimize(objective, q0, method="L-BFGS-B", bounds=list(zip(q_min, q_max)), options={"maxiter": 80})
        q_prev = np.asarray(res.x, dtype=float)
        qs.append(q_prev.tolist())
        fks.append(robot.fk(q_prev).tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"method": "keypoint_ik", "task": task["task"], "dt": task["dt"],
                                    "q": qs, "keypoint_names": NAMES,
                                    "retargeted_keypoints": fks, "source_file": str(task_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.out)


if __name__ == "__main__":
    main()
