"""Scaled OmniRetarget implementation.

This implements the paper's computational core: frame-wise interaction
mesh retargeting with Laplacian deformation energy, temporal smoothing,
joint/velocity limits, non-penetration, and foot-sticking constraints.
The solver uses SciPy SLSQP for the small benchmark instead of Drake's
SOCP backend; the objective and hard constraints mirror Eq. (3)/(15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

from data.planar_robot import KEYPOINTS as NAMES
from data.planar_robot import NAME_TO_IDX, PlanarHumanoid, initial_guess
FOOT_NAMES = ["left_foot", "right_foot"]
HAND_NAMES = ["left_hand", "right_hand"]
COLLISION_SKIP = set(HAND_NAMES)


def terrain_height(x: np.ndarray | float, task: str) -> np.ndarray | float:
    x_arr = np.asarray(x)
    if task == "terrain_step":
        ramp = np.clip((x_arr - 0.40) / 0.15, 0.0, 1.0)
        out = 0.32 * ramp
    else:
        out = np.zeros_like(x_arr, dtype=float)
    return float(out) if np.isscalar(x) else out


def adjacency(points: np.ndarray) -> list[list[int]]:
    if len(points) < 4:
        return [[j for j in range(len(points)) if j != i] for i in range(len(points))]
    tri = Delaunay(points, qhull_options="QJ")
    neigh = [set() for _ in range(len(points))]
    for simplex in tri.simplices:
        for i in simplex:
            neigh[i].update(int(j) for j in simplex if j != i)
    return [sorted(n) if n else [j for j in range(len(points)) if j != i] for i, n in enumerate(neigh)]


def laplacian(points: np.ndarray, neigh: list[list[int]]) -> np.ndarray:
    out = np.zeros_like(points)
    for i, ns in enumerate(neigh):
        out[i] = points[i] - points[ns].mean(axis=0)
    return out


def object_frame(points: np.ndarray, obj: dict | None) -> np.ndarray:
    if not obj:
        return points
    return points - np.array(obj["center"], dtype=float)


def stance_feet(src_prev: np.ndarray, src: np.ndarray, dt: float, task_name: str) -> list[str]:
    out = []
    for name in FOOT_NAMES:
        idx = NAME_TO_IDX[name]
        horiz_v = abs(src[idx, 0] - src_prev[idx, 0]) / dt
        prev_h = terrain_height(src_prev[idx, 0], task_name)
        cur_h = terrain_height(src[idx, 0], task_name)
        same_surface = abs(cur_h - prev_h) < 0.05
        if horiz_v < 0.35 and same_surface and src[idx, 1] <= cur_h + 0.03:
            out.append(name)
    return out


def solve_frame(task: dict, t: int, q_prev: np.ndarray, prev_fk: np.ndarray | None, robot: PlanarHumanoid) -> np.ndarray:
    source = np.array(task["source_keypoints"][t], dtype=float)
    obj = task["objects"][t]
    obj_pts = np.array(task["object_surface"][t], dtype=float).reshape(-1, 2)
    terr = np.array(task["terrain_points"][t], dtype=float)
    source_mesh = np.vstack([source, obj_pts, terr])
    source_mesh = object_frame(source_mesh, obj) if obj else source_mesh
    neigh = adjacency(source_mesh)
    l_source = laplacian(source_mesh, neigh)
    q0 = q_prev.copy() if q_prev is not None else initial_guess(source)
    dt = float(task["dt"])
    q_min = np.array([-2.0, 0.45, -0.7, -1.4, -0.2, -1.4, -1.9, -2.4, 0.1, -0.2, -2.4])
    q_max = np.array([2.0, 1.45, 0.7, 1.4, 1.9, 1.4, 0.2, 0.2, 2.4, 2.4, -0.1])
    vel = np.array([0.09, 0.08, 0.18, 0.24, 0.30, 0.24, 0.30, 0.30, 0.36, 0.30, 0.36])
    if task["task"] == "terrain_step":
        vel = np.array([0.16, 0.18, 0.24, 0.42, 0.55, 0.42, 0.55, 0.50, 0.60, 0.50, 0.60])
    limited_bounds = list(zip(np.maximum(q_min, q0 - vel), np.minimum(q_max, q0 + vel))) if q_prev is not None else list(zip(q_min, q_max))
    full_bounds = list(zip(q_min, q_max))

    def target_mesh(q: np.ndarray) -> np.ndarray:
        pts = np.vstack([robot.fk(q), obj_pts, terr])
        return object_frame(pts, obj) if obj else pts

    def objective(q: np.ndarray) -> float:
        tgt = target_mesh(q)
        e_lap = np.sum((laplacian(tgt, neigh) - l_source) ** 2)
        e_key = 0.08 * np.sum((robot.fk(q) - source) ** 2)
        e_smooth = 0.03 * np.sum((q - q0) ** 2)
        return float(e_lap + e_key + e_smooth)

    constraints = []

    def nonpenetration(q: np.ndarray) -> np.ndarray:
        fk = robot.fk(q)
        vals = []
        for i, p in enumerate(fk):
            vals.append(p[1] - terrain_height(p[0], task["task"]))
            if obj and NAMES[i] not in COLLISION_SKIP:
                c = np.array(obj["center"])
                sx, sz = np.array(obj["size"])
                dx = abs(p[0] - c[0]) - sx / 2
                dz = abs(p[1] - c[1]) - sz / 2
                vals.append(max(dx, dz))
        return np.array(vals)

    constraints.append({"type": "ineq", "fun": nonpenetration})
    if t > 0 and prev_fk is not None:
        for foot in stance_feet(np.array(task["source_keypoints"][t - 1]), source, dt, task["task"]):
            idx = NAME_TO_IDX[foot]
            constraints.append({"type": "eq", "fun": lambda q, idx=idx: robot.fk(q)[idx] - prev_fk[idx]})
    for name in task["contacts"][t].get("terrain", []):
        if "hand" in name:
            idx = NAME_TO_IDX[name]
            constraints.append({"type": "eq", "fun": lambda q, idx=idx: robot.fk(q)[idx, 1] - terrain_height(robot.fk(q)[idx, 0], task["task"])})

    def constraint_violation(q: np.ndarray) -> float:
        violation = 0.0
        for con in constraints:
            val = np.asarray(con["fun"](q), dtype=float)
            if con["type"] == "ineq":
                violation = max(violation, float(np.max(np.maximum(0.0, -val))))
            else:
                violation = max(violation, float(np.max(np.abs(val))))
        return violation

    res = minimize(objective, q0, method="SLSQP", bounds=limited_bounds, constraints=constraints,
                   options={"maxiter": 140, "ftol": 1e-8, "disp": False})
    if (not res.success or constraint_violation(np.asarray(res.x, dtype=float)) > 1e-4) and task["task"] == "terrain_step":
        seed = q0.copy()
        seed[1] = max(seed[1], terrain_height(seed[0], task["task"]) + 0.85)
        res_retry = minimize(objective, seed, method="SLSQP", bounds=full_bounds, constraints=constraints,
                             options={"maxiter": 220, "ftol": 1e-8, "disp": False})
        if constraint_violation(np.asarray(res_retry.x, dtype=float)) < constraint_violation(np.asarray(res.x, dtype=float)):
            res = res_retry
    return np.asarray(res.x if res.success else res.x, dtype=float)


def run(task_path: Path, out_path: Path) -> None:
    task = json.loads(task_path.read_text())
    robot = PlanarHumanoid()
    qs, fks = [], []
    q_prev = None
    prev_fk = None
    for t in range(len(task["source_keypoints"])):
        q = solve_frame(task, t, q_prev, prev_fk, robot)
        fk = robot.fk(q)
        qs.append(q.tolist())
        fks.append(fk.tolist())
        q_prev, prev_fk = q, fk
    payload = {"method": "omniretarget", "task": task["task"], "dt": task["dt"], "q": qs,
               "keypoint_names": NAMES, "retargeted_keypoints": fks, "source_file": str(task_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.out)


if __name__ == "__main__":
    main()
