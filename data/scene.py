"""Scene description shared by the public data tools, methods and baselines.

A *scene* (JSON-able dict) is the ground plane z = 0 plus a list of oriented
boxes.  Every box has
  ``half_size`` [hx, hy, hz]            box half extents in its own frame,
  ``offset``    [ox, oy, oz]            box centre in the body frame (object meshes are not centred),
  ``poses``     (T, 7) or ``pose`` (7)  body pose per frame (dynamic) or fixed, [x, y, z, qw, qx, qy, qz],
  ``kind``      "object" (manipulated) or "terrain" (static).

``build_scene_model`` returns a MuJoCo model with the robot plus the scene:
robot geoms only collide with scene geoms (no self pairs), dynamic boxes are
mocap bodies moved by ``set_frame``.  The evaluator keeps its own copy of
this geometry; this module is for methods.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "data" / "robot" / "g1_29dof_spherehand_hull.xml"


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] (..., 4) -> rotation matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def box_poses(box: dict, T: int) -> np.ndarray:
    """(T, 7) body poses of a box."""
    if "poses" in box:
        return np.asarray(box["poses"], dtype=float)
    return np.repeat(np.asarray(box["pose"], dtype=float)[None], T, axis=0)


def box_frames(box: dict, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame box centre (T,3), rotation (T,3,3) and half size (3,)."""
    P = box_poses(box, T)
    R = quat_to_mat(P[:, 3:7])
    c = P[:, :3] + np.einsum("tij,j->ti", R, np.asarray(box.get("offset", [0, 0, 0]), dtype=float))
    return c, R, np.asarray(box["half_size"], dtype=float)


def box_sdf(points: np.ndarray, centre: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance of points (..., 3) to one oriented box (negative inside)."""
    local = np.einsum("ji,...j->...i", R, np.asarray(points, dtype=float) - centre)
    q = np.abs(local) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def scene_sdf(points: np.ndarray, scene: dict, include_ground: bool = True, kinds=("object", "terrain")) -> np.ndarray:
    """Signed distance of per-frame points (T, K, 3) to the scene (min over ground and boxes)."""
    P = np.asarray(points, dtype=float)
    T = P.shape[0]
    d = P[..., 2] - float(scene.get("ground_z", 0.0)) if include_ground else np.full(P.shape[:-1], np.inf)
    for box in scene["boxes"]:
        if box.get("kind", "terrain") not in kinds:
            continue
        c, R, h = box_frames(box, T)
        dd = np.stack([box_sdf(P[t], c[t], R[t], h) for t in range(T)])
        d = np.minimum(d, dd)
    return d


def build_scene_model(scene: dict, robot_xml: Path = ROBOT_XML):
    """MuJoCo model of robot + scene; returns (model, data, dynamic_box_mocap_ids)."""
    import mujoco

    root = ET.parse(robot_xml).getroot()
    wb = root.find("worldbody")
    for g in root.iter("geom"):
        g.set("contype", "1")
        g.set("conaffinity", "0")
    ET.SubElement(wb, "geom", name="ground", type="plane", size="50 50 0.1",
                  pos=f"0 0 {float(scene.get('ground_z', 0.0))}", contype="0", conaffinity="1")
    for k, box in enumerate(scene["boxes"]):
        hs = " ".join(f"{v:.6f}" for v in box["half_size"])
        off = " ".join(f"{v:.6f}" for v in box.get("offset", [0, 0, 0]))
        if "poses" in box:
            b = ET.SubElement(wb, "body", name=f"box{k}", mocap="true")
        else:
            p = np.asarray(box["pose"], dtype=float)
            b = ET.SubElement(wb, "body", name=f"box{k}", pos=" ".join(map(str, p[:3])),
                              quat=" ".join(map(str, p[3:7])))
        ET.SubElement(b, "geom", name=f"scene_{box.get('kind', 'terrain')}_{k}", type="box", size=hs, pos=off,
                      contype="0", conaffinity="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mocap = {k: int(model.body_mocapid[model.body(f"box{k}").id]) for k, b in enumerate(scene["boxes"]) if "poses" in b}
    return model, data, mocap


def set_frame(data, scene: dict, mocap: dict, t: int) -> None:
    """Move the dynamic boxes of ``scene`` to frame ``t``."""
    for k, mid in mocap.items():
        p = scene["boxes"][k]["poses"][t]
        data.mocap_pos[mid] = p[:3]
        data.mocap_quat[mid] = p[3:7]
