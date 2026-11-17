"""Retargeting instances from real human demonstrations (public train split).

Sources (fetched by scripts/download.sh):
  * OMOMO box carrying (InterMimic-processed release used by the paper's code):
    52 SMPL-H joints + the large box's 6-DoF trajectory, 30 fps.  Only the
    public train subjects are fetched; the test subjects are held out.
  * The paper's in-house climbing MoCap (released with its code): 53 joints at
    120 fps (used at 30 fps) + the box terrain of each sequence.  Only the
    public train sequences are fetched.

An *instance* (one JSON file) holds the demonstration and three scenes:
  ``source_scene``  the demonstrator's scene (human scale),
  ``nominal_scene`` the same scene brought to the robot's scale by direct
                    scaling with alpha = robot height / demonstrator height
                    (paper App. A); the manipulated box keeps its true size,
  ``target_scene``  the scene the robot must act in: the nominal scene, or an
                    augmented copy (object pose/shape or terrain height; paper
                    Sec. III-B and App. C-1).
The robot model is data/robot/g1_29dof_spherehand_hull.xml (Unitree G1, 29 DoF).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from data.omomo_io import SMPLH_JOINTS, load_pt_array, parse_intermimic

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "downloads"
ROBOT_HEIGHT = 1.32          # Unitree G1 (paper code default)
FPS = 30
WINDOW = 150                 # terrain frames per instance (5 s)
OBJECT_WINDOW = 600          # up to 20 s: pickup and subsequent carrying motion
MIN_FRAMES = 90
MOCAP_HUMAN_HEIGHT = 1.78    # paper code default for its in-house MoCap
MOCAP_DOWNSAMPLE = 4         # 120 fps -> 30 fps
OMOMO_HEIGHTS = {"sub1": 1.6637, "sub2": 1.8438, "sub3": 1.9411, "sub4": 1.7904, "sub5": 1.8171, "sub6": 1.7622,
                 "sub7": 1.8078, "sub8": 1.8449, "sub9": 1.7532, "sub10": 1.6510, "sub11": 1.7666, "sub12": 1.8030,
                 "sub13": 1.8110, "sub14": 1.8263, "sub15": 1.6846, "sub16": 1.7230, "sub17": 1.7366}
OMOMO_TRAIN_SUBJECTS = ("sub3", "sub8", "sub10", "sub12")
CLIMB_TRAIN_SEQUENCES = (1,)    # sequence 4 (thin wall reaching below the floor) is not used
# Large box (OMOMO "largebox"): bounds of the released mesh, in the object frame.
LARGEBOX_HALF = [0.235577, 0.229365, 0.203948]
LARGEBOX_OFFSET = [0.001494, -0.000715, 0.005756]
# Augmentation decay constants of the paper code (frames), Eq. (14).
TAU_P, TAU_THETA = 50.0, 25.0

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


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def object_first_moving_frame(poses: np.ndarray, thresh: float = 0.0025) -> int:
    """First frame where the object pose changes by more than ``thresh`` (paper code)."""
    v = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    return int(np.argmax(v > thresh)) if np.any(v > thresh) else 0


def _round(a, nd=5):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def object_scenes(obj_poses: np.ndarray, alpha: float) -> tuple[dict, dict]:
    """Source (human-scale) and nominal (robot-scale) scenes of a box-carrying clip.

    Direct scaling as in the paper code: the box trajectory's horizontal
    position is scaled by alpha, its height displacement from frame 0 too; the
    box keeps its true size."""
    src = {"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": LARGEBOX_HALF, "offset": LARGEBOX_OFFSET,
                                        "poses": _round(obj_poses)}]}
    nom = obj_poses.copy()
    nom[:, :2] *= alpha
    nom[:, 2] = nom[0, 2] + alpha * (nom[:, 2] - nom[0, 2])
    nominal = {"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": LARGEBOX_HALF, "offset": LARGEBOX_OFFSET,
                                            "poses": _round(nom)}]}
    return src, nominal


def augment_object_scene(nominal: dict, human_root0: np.ndarray, rng: np.random.Generator) -> tuple[dict, dict]:
    """Paper Sec. III-B / App. C-1: offset the box's initial pose (translation in the
    demonstrator's frame and a yaw rotation), decayed exponentially after the box
    starts moving (Eq. 14), and scale its three dimensions."""
    box = nominal["boxes"][0]
    P = np.asarray(box["poses"], dtype=float)
    T = len(P)
    tm = object_first_moving_frame(P)
    d_local = np.array([rng.uniform(0.0, 0.25), rng.uniform(-0.15, 0.15), 0.0])
    theta = float(rng.uniform(-0.6, 0.6))
    scale = rng.uniform(0.8, 1.2, size=3)
    # demonstrator frame at frame 0: x from the root towards the box (paper code)
    to_obj = P[0, :2] - human_root0[:2]
    x = np.array([*to_obj / (np.linalg.norm(to_obj) + 1e-9), 0.0])
    y = np.cross([0.0, 0.0, 1.0], x)
    dp = np.column_stack([x, y, [0, 0, 1.0]]) @ d_local
    half = np.asarray(box["half_size"]) * scale
    off = np.asarray(box["offset"]) * scale
    R0 = _quat_mat(P[0, 3:7])
    dp[2] = np.abs(R0[2]) @ (half - np.asarray(box["half_size"]))   # keep the box resting on the floor
    k = np.arange(T)
    dec_p = np.where(k < tm, 1.0, np.exp(-(k - tm) / TAU_P))
    dec_r = np.where(k < tm, 1.0, np.exp(-(k - tm) / TAU_THETA))
    out = P.copy()
    out[:, :3] += dec_p[:, None] * dp
    ang = theta * dec_r
    qz = np.stack([np.cos(ang / 2), np.zeros(T), np.zeros(T), np.sin(ang / 2)], 1)
    out[:, 3:7] = _quat_mul(qz, P[:, 3:7])
    target = {"ground_z": 0.0, "boxes": [{"kind": "object", "half_size": _round(half), "offset": _round(off),
                                           "poses": _round(out)}]}
    params = {"type": "object_pose_shape", "translation_demo_frame": _round(d_local, 4), "yaw": round(theta, 4),
              "scale": _round(scale, 4), "moving_frame": tm}
    return target, params


def terrain_boxes_from_obj(seq_dir: Path) -> list[dict]:
    """Oriented boxes (human scale) from the released per-box OBJ files; ground slabs are dropped."""
    boxes = []
    for f in sorted(glob.glob(str(seq_dir / "box_models" / "*.obj"))):
        V = np.array([ln.split()[1:4] for ln in open(f) if ln.startswith("v ")], dtype=float)
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
        boxes.append({"kind": "terrain", "half_size": _round(half),
                      "pose": _round([c[0], c[1], (z0 + z1) / 2, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])})
    return boxes


def terrain_scenes(boxes: list[dict], alpha: float) -> tuple[dict, dict]:
    """Source and nominal scenes of a climbing sequence (terrain scaled by alpha about the origin)."""
    src = {"ground_z": 0.0, "boxes": boxes}
    nom = []
    for b in boxes:
        p = np.asarray(b["pose"], dtype=float)
        nom.append({"kind": "terrain", "half_size": _round(np.asarray(b["half_size"]) * alpha),
                    "pose": _round([*(p[:3] * alpha), *p[3:7]])})
    return src, {"ground_z": 0.0, "boxes": nom}


def augment_terrain_scene(nominal: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Paper Sec. III-B / Fig. 4a: scale the platforms' height."""
    s = float(rng.choice([rng.uniform(0.7, 0.9), rng.uniform(1.1, 1.3)]))
    out = []
    for b in nominal["boxes"]:
        p = np.asarray(b["pose"], dtype=float)
        h = np.asarray(b["half_size"], dtype=float)
        bottom = p[2] - h[2]
        out.append({"kind": "terrain", "half_size": _round([h[0], h[1], h[2] * s]),
                    "pose": _round([p[0], p[1], bottom + h[2] * s, *p[3:7]])})
    return {"ground_z": 0.0, "boxes": out}, {"type": "terrain_height", "height_scale": round(s, 4)}


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------

def make_instance(iid: str, task: str, joint_names: list[str], joints: np.ndarray, human_height: float,
                  source: dict, nominal: dict, target: dict | None = None, augmentation: dict | None = None) -> dict:
    return {
        "instance_id": iid,
        "task": task,
        "fps": FPS,
        "num_frames": int(len(joints)),
        "human_height": round(float(human_height), 4),
        "robot_height": ROBOT_HEIGHT,
        "alpha": round(ROBOT_HEIGHT / float(human_height), 6),
        "robot_model": "data/robot/g1_29dof_spherehand_hull.xml",
        "joint_names": list(joint_names),
        "human_joints": _round(joints, 4),
        "source_scene": source,
        "nominal_scene": nominal,
        "target_scene": target if target is not None else nominal,
        "augmentation": augmentation,
    }


def omomo_window(joints: np.ndarray, poses: np.ndarray) -> tuple[int, int] | None:
    """Keep pickup and subsequent carrying (up to 20 s, starting 1 s before motion)."""
    tm = object_first_moving_frame(poses)
    start = max(0, tm - 30)
    end = min(len(joints), start + OBJECT_WINDOW)
    return (start, end) if end - start >= MIN_FRAMES else None


def omomo_instances(clips: list[tuple[str, np.ndarray]], rng: np.random.Generator,
                    n_nominal: int, n_augmented: int, prefix: str = "") -> list[dict]:
    """clips: (name like 'sub3_largebox_003', raw (T, 591) array)."""
    out, used = [], 0
    order = rng.permutation(len(clips))
    for idx in order:
        name, arr = clips[idx]
        joints, poses = parse_intermimic(arr)
        win = omomo_window(joints, poses)
        if win is None:
            continue
        s, e = win
        joints, poses = joints[s:e], poses[s:e]
        h = OMOMO_HEIGHTS[name.split("_")[0]]
        alpha = ROBOT_HEIGHT / h
        src, nom = object_scenes(poses, alpha)
        if used < n_nominal:
            out.append(make_instance(f"{prefix}object_{name}", "object", SMPLH_JOINTS, joints, h, src, nom))
        elif used < n_nominal + n_augmented:
            tgt, params = augment_object_scene(nom, alpha * joints[0, 0], rng)
            out.append(make_instance(f"{prefix}objaug_{name}", "object", SMPLH_JOINTS, joints, h, src, nom,
                                     tgt, params))
        else:
            break
        used += 1
    return out


def climb_windows(T: int, stride: int = WINDOW) -> list[tuple[int, int]]:
    wins = []
    for s in range(0, T, stride):
        e = min(T, s + WINDOW)
        if e - s >= MIN_FRAMES:
            wins.append((s, e))
    return wins


def climb_instances(seqs: list[tuple[int, np.ndarray, list[dict]]], rng: np.random.Generator,
                    prefix: str = "", stride: int = WINDOW) -> list[dict]:
    """seqs: (sequence id, joints at 30 fps (T, 53, 3), terrain boxes). Nominal and height-augmented windows."""
    h = MOCAP_HUMAN_HEIGHT
    alpha = ROBOT_HEIGHT / h
    out = []
    for sid, joints, boxes in seqs:
        src, nom = terrain_scenes(boxes, alpha)
        for s, e in climb_windows(len(joints), stride):
            j = joints[s:e]
            out.append(make_instance(f"{prefix}terrain_seq{sid}_f{s:04d}", "terrain", MOCAP_JOINTS, j, h, src, nom))
            tgt, params = augment_terrain_scene(nom, rng)
            out.append(make_instance(f"{prefix}terraug_seq{sid}_f{s:04d}", "terrain", MOCAP_JOINTS, j, h, src, nom,
                                     tgt, params))
    return out


def load_climb_sequence(seq_dir: Path) -> np.ndarray:
    f = sorted(glob.glob(str(seq_dir / "*_joint_positions_*.npy")))[0]
    return np.load(f)[::MOCAP_DOWNSAMPLE].astype(np.float64)


def build_train(out_dir: Path, seed: int = 20251001) -> list[str]:
    """Public train split: 20 box-carrying + 10 augmented-box clips (train subjects), and
    150-frame windows (every 75 frames) of the public climbing sequence, nominal + height-augmented."""
    rng = np.random.default_rng(seed)
    clips = []
    for f in sorted(glob.glob(str(DOWNLOADS / "omomo" / "largebox" / "*.pt"))):
        name = Path(f).stem
        if name.split("_")[0] in OMOMO_TRAIN_SUBJECTS:
            clips.append((name, load_pt_array(Path(f).read_bytes())))
    if not clips:
        raise SystemExit("no OMOMO train clips found; run scripts/download.sh")
    insts = omomo_instances(clips, rng, n_nominal=20, n_augmented=10)
    seqs = []
    for sid in CLIMB_TRAIN_SEQUENCES:
        d = DOWNLOADS / "holosoma" / "demo_data" / "climb" / f"mocap_climb_seq_{sid}"
        seqs.append((sid, load_climb_sequence(d), terrain_boxes_from_obj(d)))
    insts += climb_instances(seqs, rng, stride=WINDOW // 2)   # overlapping windows: one public sequence
    inp = out_dir / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    for old in inp.glob("*.json"):
        old.unlink()
    for x in insts:
        (inp / f"{x['instance_id']}.json").write_text(json.dumps(x))
    return [x["instance_id"] for x in insts]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the public train instances")
    ap.add_argument("--out", type=Path, default=DOWNLOADS / "tasks" / "train")
    a = ap.parse_args()
    ids = build_train(a.out)
    print(f"wrote {len(ids)} train instances to {a.out}")
