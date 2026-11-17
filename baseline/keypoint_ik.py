"""Keypoint-matching IK baseline (GMR / PHC family, paper App. B-1/B-2).

Per frame, the robot configuration minimises the distance between its 15 key
bodies and the demonstrator's key joints brought to the robot's scale by
direct scaling (alpha = robot / demonstrator height, demonstrator lowered so
its toes touch the supporting surface), with joint position and velocity
limits and a small smoothness term -- no collision, contact or foot-sticking
handling and no awareness of the object or terrain.  Solved by bounded
Gauss-Newton steps (scipy lsq_linear), warm-started from the previous frame.

Usage: python baseline/keypoint_ik.py --inputs DIR --outputs DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "data/robot/g1_29dof_spherehand_hull.xml"
LIMITS = json.loads((ROOT / "data/robot/g1_limits.json").read_text())
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
TOES = {"smplh": ("L_Toe", "R_Toe"), "mocap": ("LeftToeBase", "RightToeBase")}
HIPS = {"smplh": ("L_Hip", "R_Hip"), "mocap": ("LeftUpLeg", "RightUpLeg")}
SMOOTH, STEP, ITERS_FIRST, ITERS = 0.05, 0.2, 50, 10


def _support(xy: np.ndarray, scene: dict) -> np.ndarray:
    h = np.zeros(xy.shape[:-1])
    for b in scene["boxes"]:
        if "pose" not in b:
            continue
        p = np.asarray(b["pose"], dtype=float)
        yaw = 2 * np.arctan2(p[6], p[3])
        c, s = np.cos(yaw), np.sin(yaw)
        loc = (xy - p[:2]) @ np.array([[c, -s], [s, c]])
        half = np.asarray(b["half_size"], dtype=float)
        inside = np.all(np.abs(loc) <= half[:2], axis=-1)
        h = np.where(inside, np.maximum(h, p[2] + half[2]), h)
    return h


def retarget(inst: dict) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    names = inst["joint_names"]
    fmt = "smplh" if "L_Wrist" in names else "mocap"
    kmap = SMPLH_MAP if fmt == "smplh" else MOCAP_MAP
    J = np.asarray(inst["human_joints"], dtype=float)
    toes = [names.index(t) for t in TOES[fmt]]
    z_min = float(np.min(J[:, toes, 2] - _support(J[:, toes, :2], inst["source_scene"])))
    J = J.copy()
    J[:, :, 2] -= z_min
    target = float(inst["alpha"]) * J[:, [names.index(k) for k in kmap]]
    bodies = [model.body(b).id for b in kmap.values()]
    lo = np.array([j["range"][0] for j in LIMITS["joints"]])
    hi = np.array([j["range"][1] for j in LIMITS["joints"]])
    vmax = np.array([j["velocity"] for j in LIMITS["joints"]]) / float(inst["fps"])
    hips = [names.index(h) for h in HIPS[fmt]]
    fwd = np.cross(J[0, hips[0]] - J[0, hips[1]], [0, 0, 1.0])
    yaw = np.arctan2(fwd[1], fwd[0])
    q = np.zeros(model.nq)
    q[:3] = target[0, 0]
    q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    nv = model.nv
    out = []
    q_prev = q.copy()
    for t in range(len(target)):
        for _ in range(ITERS_FIRST if t == 0 else ITERS):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            rows, res = [], []
            for b, p in zip(bodies, target[t]):
                jp = np.zeros((3, nv))
                mujoco.mj_jac(model, data, jp, None, data.xpos[b], b)
                rows.append(jp)
                res.append(p - data.xpos[b])
            dprev = np.zeros(nv)
            mujoco.mj_differentiatePos(model, dprev, 1.0, q, q_prev)
            A = np.vstack(rows + [np.sqrt(SMOOTH) * np.eye(nv)])
            b = np.concatenate(res + [np.sqrt(SMOOTH) * dprev])
            lb = np.full(nv, -STEP)
            ub = np.full(nv, STEP)
            jlo = np.maximum(lo, q_prev[7:] - vmax) if t > 0 else lo
            jhi = np.minimum(hi, q_prev[7:] + vmax) if t > 0 else hi
            lb[6:] = np.minimum(np.maximum(lb[6:], jlo - q[7:]), 0.0)
            ub[6:] = np.maximum(np.minimum(ub[6:], jhi - q[7:]), 0.0)
            dq = lsq_linear(A, b, bounds=(lb, ub), method="bvls").x
            mujoco.mj_integratePos(model, q, dq, 1.0)
            q[3:7] /= np.linalg.norm(q[3:7])
            q[7:] = np.clip(q[7:], lo, hi)
            if np.linalg.norm(dq) < 1e-4:
                break
        out.append(q.copy())
        q_prev = q.copy()
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, required=True)
    args = ap.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.inputs.glob("*.json")):
        inst = json.loads(path.read_text())
        q = retarget(inst)
        (args.outputs / path.name).write_text(json.dumps({"q": np.round(q, 6).tolist()}))
        print(f"{inst['instance_id']}: {len(q)} frames", flush=True)


if __name__ == "__main__":
    main()
