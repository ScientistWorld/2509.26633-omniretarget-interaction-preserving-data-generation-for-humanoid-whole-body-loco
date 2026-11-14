"""Honest method variants used to check that the hardened test rewards better retargeting.

contact_aware          OmniRetarget + task-specific contact costs (App. B-5 allows
                       augmenting the core objective): hands that touch an object or
                       the terrain in the demonstration, and planted feet, are pulled
                       to the corresponding point of the target scene.
keypoint_contact       The same contact costs with keypoint matching instead of the
                       interaction mesh (same solver, constraints, scaling and
                       augmentation anchoring as the reference).
keypoint_contact_it11  The iteration-11 version of keypoint_contact (no frame-0
                       grounding, no Eq. 4/5 anchoring), kept for before/after.
no_mesh                Keypoint matching in place of the mesh, nothing else changed.

Usage: python redteam/hardening/variants.py <variant> <input_dir> <output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from data.planar_robot import NAME_TO_IDX
from data.scene_geometry import box_sdf, terrain_sdf
from method.omniretarget import retarget

HANDS = [NAME_TO_IDX["left_hand"], NAME_TO_IDX["right_hand"]]
FEET = [NAME_TO_IDX["left_foot"], NAME_TO_IDX["right_foot"]]
W_HAND = 10.0
W_FOOT = 10.0


def _polyline_closest(p, verts):
    A, B = verts[:-1], verts[1:]
    AB = B - A
    t = np.clip(np.sum((p - A) * AB, axis=1) / np.sum(AB * AB, axis=1), 0.0, 1.0)
    C = A + t[:, None] * AB
    d = np.linalg.norm(p - C, axis=1)
    k = int(np.argmin(d))
    return k, float(t[k]), float(d[k])


def _box_surface_uv(p, c, size):
    rel = (p - c) / size
    if np.max(np.abs(rel)) < 0.5:
        ax = int(np.argmax(np.abs(rel)))
        rel = rel.copy()
        rel[ax] = 0.5 * np.sign(rel[ax] if rel[ax] != 0 else 1.0)
        return rel
    return np.clip(rel, -0.5, 0.5)


def contact_terms(task: dict, scene: dict, w_hand: float = W_HAND, w_foot: float = W_FOOT):
    """Contact costs mapping demonstrated contacts onto ``scene`` (the scene being solved for)."""
    src = np.asarray(task["source_keypoints"], dtype=float)
    ss, ts = task["source_scene"], scene
    terr_s = np.asarray(ss["terrain"], dtype=float)
    terr_t = np.asarray(ts["terrain"], dtype=float)
    N = len(src)
    targets: list[dict[int, np.ndarray]] = []
    for t in range(N):
        tg = {}
        for idx in HANDS:
            p = src[t, idx]
            if ss.get("object"):
                cs = np.asarray(ss["object"]["center"][t])
                zs = np.asarray(ss["object"]["size"])
                if abs(float(box_sdf(p, cs, zs))) <= 0.025:
                    ct = np.asarray(ts["object"]["center"][t])
                    zt = np.asarray(ts["object"]["size"])
                    tg[idx] = ct + _box_surface_uv(p, cs, zs) * zt
                    continue
            if abs(float(terrain_sdf(p, terr_s))) <= 0.025:
                k, u, _ = _polyline_closest(p, terr_s)
                tg[idx] = terr_t[k] + u * (terr_t[k + 1] - terr_t[k])
        for idx in FEET:
            p = src[t, idx]
            if abs(float(terrain_sdf(p, terr_s))) <= 0.01:
                k, u, _ = _polyline_closest(p, terr_s)
                tg[idx] = terr_t[k] + u * (terr_t[k + 1] - terr_t[k])
        targets.append(tg)

    def extra(t, fk, Jfk):
        rows, Js = [], []
        for idx, tp in targets[t].items():
            w = np.sqrt(w_hand if idx in HANDS else w_foot)
            rows.append(w * (fk[idx] - tp))
            Js.append(w * Jfk[idx])
        if not rows:
            return None
        return np.concatenate(rows), np.vstack(Js)

    return extra


VARIANTS = {
    "contact_aware": dict(objective="mesh", extra_factory=contact_terms),
    "keypoint_contact": dict(objective="keypoint", extra_factory=contact_terms),
    "keypoint_contact_it11": dict(objective="keypoint", extra_factory=contact_terms,
                                  ground_frame0=False, augment_anchor=False),
    "no_mesh": dict(objective="keypoint"),
}


def run(variant: str, task: dict) -> np.ndarray:
    return retarget(task, **VARIANTS[variant])


def main() -> None:
    variant, in_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.json")):
        task = json.loads(path.read_text())
        q = run(variant, task)
        (out_dir / f"{task['instance_id']}.json").write_text(
            json.dumps({"instance_id": task["instance_id"], "method": variant, "q": np.round(q, 10).tolist()}))
        print(f"[{variant}] {task['instance_id']} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
