"""Hidden test instances, built in memory from eval/test/data/test_demonstrations.npz.

Same construction as the public split (data/tasks.py), with an independent
copy of the code, held-out demonstrations (OMOMO subjects sub7/sub16, climbing
sequences 0/2/3) and a private seed:
  * object_interaction    48 box-carrying windows (nominal scene),
  * terrain_interaction   every 150-frame window of the 3 climbing sequences (14),
  * augmented_interaction 24 box-carrying windows with an offset/rotated/resized
                          box + the 14 climbing windows with rescaled platform heights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "test_demonstrations.npz"
SEED = 7130291
ROBOT_HEIGHT = 1.32
FPS = 30
WINDOW, MIN_FRAMES = 150, 90
MOCAP_HUMAN_HEIGHT = 1.78
N_OBJECT, N_OBJECT_AUG = 48, 24
OMOMO_HEIGHTS = {"sub7": 1.8078, "sub16": 1.7230}
LARGEBOX_HALF = [0.235577, 0.229365, 0.203948]
LARGEBOX_OFFSET = [0.001494, -0.000715, 0.005756]
TAU_P, TAU_THETA = 50.0, 25.0
SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2",
    "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Thorax",
    "R_Shoulder", "R_Elbow", "R_Wrist", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2",
    "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1",
    "R_Thumb2", "R_Thumb3",
]
MOCAP_JOINTS = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandIndex1", "RightHandIndex2",
    "RightHandIndex3", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandRing1",
    "RightHandRing2", "RightHandRing3", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "LeftUpLeg",
    "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "LeftFootMod",
    "RightFootMod",
]


def _r(a, nd=5):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _moving_frame(P, thresh=0.0025):
    v = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return int(np.argmax(v > thresh)) if np.any(v > thresh) else 0


def _object_scenes(poses, alpha):
    src = {"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": LARGEBOX_HALF, "offset": LARGEBOX_OFFSET,
                                        "poses": _r(poses)}]}
    nom = poses.copy()
    nom[:, :2] *= alpha
    nom[:, 2] = nom[0, 2] + alpha * (nom[:, 2] - nom[0, 2])
    return src, {"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": LARGEBOX_HALF,
                                             "offset": LARGEBOX_OFFSET, "poses": _r(nom)}]}


def _augment_object(nominal, root0, rng):
    box = nominal["boxes"][0]
    P = np.asarray(box["poses"], dtype=float)
    T = len(P)
    tm = _moving_frame(P)
    d_local = np.array([rng.uniform(0.0, 0.25), rng.uniform(-0.15, 0.15), 0.0])
    theta = float(rng.uniform(-0.6, 0.6))
    scale = rng.uniform(0.8, 1.2, size=3)
    to_obj = P[0, :2] - root0[:2]
    x = np.array([*to_obj / (np.linalg.norm(to_obj) + 1e-9), 0.0])
    y = np.cross([0.0, 0.0, 1.0], x)
    dp = np.column_stack([x, y, [0, 0, 1.0]]) @ d_local
    half = np.asarray(box["half_size"]) * scale
    off = np.asarray(box["offset"]) * scale
    dp[2] = np.abs(_quat_mat(P[0, 3:7])[2]) @ (half - np.asarray(box["half_size"]))
    k = np.arange(T)
    dec_p = np.where(k < tm, 1.0, np.exp(-(k - tm) / TAU_P))
    dec_r = np.where(k < tm, 1.0, np.exp(-(k - tm) / TAU_THETA))
    out = P.copy()
    out[:, :3] += dec_p[:, None] * dp
    ang = theta * dec_r
    qz = np.stack([np.cos(ang / 2), np.zeros(T), np.zeros(T), np.sin(ang / 2)], 1)
    out[:, 3:7] = _quat_mul(qz, P[:, 3:7])
    return ({"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": _r(half), "offset": _r(off),
                                         "poses": _r(out)}]},
            {"type": "object_pose_shape", "translation_demo_frame": _r(d_local, 4), "yaw": round(theta, 4),
             "scale": _r(scale, 4), "moving_frame": tm})


def _terrain_boxes(box_vertices):
    boxes = []
    for V in box_vertices:
        V = np.asarray(V, dtype=float)
        z0, z1 = V[:, 2].min(), V[:, 2].max()
        xy = np.unique(V[:, :2].round(6), axis=0)
        d = xy - xy[0]
        o = np.argsort(np.linalg.norm(d, axis=1))
        a, b = d[o[1]], d[o[2]]
        half = [np.linalg.norm(a) / 2, np.linalg.norm(b) / 2, (z1 - z0) / 2]
        if max(half) > 5.0:
            continue
        yaw = float(np.arctan2(a[1], a[0]))
        c = xy.mean(0)
        boxes.append({"kind": "terrain", "half_size": _r(half),
                      "pose": _r([c[0], c[1], (z0 + z1) / 2, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])})
    return boxes


def _terrain_scenes(boxes, alpha):
    nom = []
    for b in boxes:
        p = np.asarray(b["pose"], dtype=float)
        nom.append({"kind": "terrain", "half_size": _r(np.asarray(b["half_size"]) * alpha),
                    "pose": _r([*(p[:3] * alpha), *p[3:7]])})
    return {"ground_z": 0.0, "boxes": boxes}, {"ground_z": 0.0, "boxes": nom}


def _augment_terrain(nominal, rng):
    s = float(rng.choice([rng.uniform(0.7, 0.9), rng.uniform(1.1, 1.3)]))
    out = []
    for b in nominal["boxes"]:
        p = np.asarray(b["pose"], dtype=float)
        h = np.asarray(b["half_size"], dtype=float)
        out.append({"kind": "terrain", "half_size": _r([h[0], h[1], h[2] * s]),
                    "pose": _r([p[0], p[1], p[2] - h[2] + h[2] * s, *p[3:7]])})
    return {"ground_z": 0.0, "boxes": out}, {"type": "terrain_height", "height_scale": round(s, 4)}


def _instance(iid, task, names, joints, h, source, nominal, target=None, aug=None):
    return {"instance_id": iid, "task": task, "fps": FPS, "num_frames": int(len(joints)),
            "human_height": round(float(h), 4), "robot_height": ROBOT_HEIGHT,
            "alpha": round(ROBOT_HEIGHT / float(h), 6), "robot_model": "data/robot/g1_29dof_spherehand_hull.xml",
            "joint_names": list(names), "human_joints": _r(joints, 4), "source_scene": source,
            "nominal_scene": nominal, "target_scene": target if target is not None else nominal,
            "augmentation": aug}


def test_split() -> dict[str, dict]:
    """instance_id -> input dict (the evaluator derives every label from these inputs)."""
    z = np.load(DATA)
    meta = json.loads(str(z["meta"]))
    rng = np.random.default_rng(SEED)
    out = {}
    clips = meta["omomo"]
    order = rng.permutation(len(clips))
    for rank, idx in enumerate(order[: N_OBJECT + N_OBJECT_AUG]):
        name = clips[idx]
        joints = z[f"omomo/{name}/joints"].astype(float)
        poses = z[f"omomo/{name}/poses"].astype(float)
        h = OMOMO_HEIGHTS[name.split("_")[0]]
        alpha = ROBOT_HEIGHT / h
        src, nom = _object_scenes(poses, alpha)
        if rank < N_OBJECT:
            x = _instance(f"test_object_{name}", "object", SMPLH_JOINTS, joints, h, src, nom)
        else:
            tgt, aug = _augment_object(nom, alpha * joints[0, 0], rng)
            x = _instance(f"test_objaug_{name}", "object", SMPLH_JOINTS, joints, h, src, nom, tgt, aug)
        out[x["instance_id"]] = x
    alpha = ROBOT_HEIGHT / MOCAP_HUMAN_HEIGHT
    for seq in meta["climb"]:
        sid = seq["sequence"]
        J = z[f"climb/{sid}/joints"].astype(float)
        src, nom = _terrain_scenes(_terrain_boxes(seq["box_vertices"]), alpha)
        for s in range(0, len(J), WINDOW):
            e = min(len(J), s + WINDOW)
            if e - s < MIN_FRAMES:
                continue
            x = _instance(f"test_terrain_seq{sid}_f{s:04d}", "terrain", MOCAP_JOINTS, J[s:e], MOCAP_HUMAN_HEIGHT,
                          src, nom)
            out[x["instance_id"]] = x
            tgt, aug = _augment_terrain(nom, rng)
            x = _instance(f"test_terraug_seq{sid}_f{s:04d}", "terrain", MOCAP_JOINTS, J[s:e], MOCAP_HUMAN_HEIGHT,
                          src, nom, tgt, aug)
            out[x["instance_id"]] = x
    return out
