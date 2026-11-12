"""Scene geometry helpers: terrain polylines and axis-aligned boxes.

A terrain is a polyline ``[[x0, z0], [x1, z1], ...]`` with strictly increasing
x; the solid region is everything below it.  An object is an axis-aligned box
with ``size = [width, height]`` and a per-frame ``center``.

Signed distances are negative inside solids.  Gradients point away from the
solid (outward normal direction).
"""

from __future__ import annotations

import numpy as np


def terrain_height(x, verts) -> np.ndarray:
    v = np.asarray(verts, dtype=float)
    return np.interp(x, v[:, 0], v[:, 1])


def terrain_sdf(P, verts, with_grad: bool = False):
    """Signed distance of points ``P`` (..., 2) to the terrain polyline."""
    v = np.asarray(verts, dtype=float)
    P = np.asarray(P, dtype=float)
    A, B = v[:-1], v[1:]
    AB = B - A
    L2 = np.sum(AB * AB, axis=1)
    AP = P[..., None, :] - A
    t = np.clip(np.sum(AP * AB, axis=-1) / L2, 0.0, 1.0)
    C = A + t[..., None] * AB
    D2 = np.sum((P[..., None, :] - C) ** 2, axis=-1)
    k = np.argmin(D2, axis=-1)
    d = np.sqrt(np.take_along_axis(D2, k[..., None], axis=-1)[..., 0])
    inside = P[..., 1] < terrain_height(P[..., 0], v)
    sign = np.where(inside, -1.0, 1.0)
    sdf = sign * d
    if not with_grad:
        return sdf
    Ck = np.take_along_axis(C, k[..., None, None].repeat(2, axis=-1), axis=-2)[..., 0, :]
    diff = P - Ck
    nrm = np.linalg.norm(diff, axis=-1, keepdims=True)
    seg = AB[k]
    seg_n = np.stack([-seg[..., 1], seg[..., 0]], axis=-1)
    seg_n = seg_n / np.linalg.norm(seg_n, axis=-1, keepdims=True)
    grad = np.where(nrm > 1e-9, sign[..., None] * diff / np.maximum(nrm, 1e-12), seg_n)
    return sdf, grad


def box_sdf(P, center, size, with_grad: bool = False):
    """Signed distance of points ``P`` (..., 2) to an axis-aligned box."""
    P = np.asarray(P, dtype=float)
    c = np.asarray(center, dtype=float)
    b = 0.5 * np.asarray(size, dtype=float)
    rel = P - c
    q = np.abs(rel) - b
    qpos = np.maximum(q, 0.0)
    outside = np.linalg.norm(qpos, axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    sdf = outside + inside
    if not with_grad:
        return sdf
    sgn = np.where(rel >= 0, 1.0, -1.0)
    g_out = sgn * qpos / np.maximum(outside, 1e-12)[..., None]
    axis = np.argmax(q, axis=-1)
    g_in = np.zeros_like(P)
    np.put_along_axis(g_in, axis[..., None], np.take_along_axis(sgn, axis[..., None], axis=-1), axis=-1)
    grad = np.where((outside > 1e-12)[..., None], g_out, g_in)
    return sdf, grad


def box_perimeter_uv(n_per_side: int = 8) -> np.ndarray:
    """Normalised box-surface coordinates in [-0.5, 0.5]^2 (counter-clockwise)."""
    s = np.linspace(-0.5, 0.5, n_per_side, endpoint=False)
    bottom = np.stack([s, np.full_like(s, -0.5)], axis=1)
    right = np.stack([np.full_like(s, 0.5), s], axis=1)
    top = np.stack([-s, np.full_like(s, 0.5)], axis=1)
    left = np.stack([np.full_like(s, -0.5), -s], axis=1)
    return np.concatenate([bottom, right, top, left], axis=0)


def scene_object(scene: dict, t: int):
    """(center, size) of the scene object at frame t, or None."""
    obj = scene.get("object")
    if not obj:
        return None
    return np.asarray(obj["center"][t], dtype=float), np.asarray(obj["size"], dtype=float)
