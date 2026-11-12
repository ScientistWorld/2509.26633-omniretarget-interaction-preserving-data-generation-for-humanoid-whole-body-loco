"""Planar humanoid robot used by every retargeting method in this benchmark.

The robot is a 2D (frontal-plane) kinematic chain with an 11-dimensional
configuration ``q``::

    [base_x, base_z, torso, left_hip, left_knee, right_hip, right_knee,
     left_shoulder, left_elbow, right_shoulder, right_elbow]

``base_x/base_z`` are metres, the rest radians.  A link direction angle ``a``
points along ``(sin a, -cos a)``: 0 hangs straight down, +pi/2 points to +x.

Joint limits (``Q_MIN``/``Q_MAX``) and per-second velocity limits (``V_MAX``)
are hard robot constraints: an output trajectory that leaves them is not a
valid robot motion.  ``LINKS`` lists the rigid links (pairs of keypoint
indices) whose whole length must stay out of objects and terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

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
NAME_TO_IDX = {n: i for i, n in enumerate(KEYPOINTS)}

Q_NAMES = [
    "base_x", "base_z", "torso",
    "left_hip", "left_knee", "right_hip", "right_knee",
    "left_shoulder", "left_elbow", "right_shoulder", "right_elbow",
]
Q_MIN = np.array([-3.0, 0.30, -0.80, -1.50, -0.30, -1.50, -2.40, -3.20, -2.60, -0.60, -2.60])
Q_MAX = np.array([3.0, 1.60, 0.80, 1.50, 2.40, 1.50, 0.30, 0.60, 2.60, 3.20, 2.60])
# Velocity limits per second (m/s for the base, rad/s for joints).
V_MAX = np.array([1.5, 1.5, 4.0, 8.0, 10.0, 8.0, 10.0, 8.0, 10.0, 8.0, 10.0])

LINKS = [
    (0, 1),    # pelvis - neck (torso)
    (1, 6),    # neck - left shoulder
    (1, 9),    # neck - right shoulder
    (0, 2),    # pelvis - left knee
    (2, 3),    # left knee - left foot
    (0, 4),    # pelvis - right knee
    (4, 5),    # right knee - right foot
    (6, 7),    # left upper arm
    (7, 8),    # left forearm
    (9, 10),   # right upper arm
    (10, 11),  # right forearm
]


@dataclass
class PlanarHumanoid:
    thigh: float = 0.40
    shin: float = 0.39
    torso: float = 0.48
    shoulder_x: float = 0.16
    shoulder_drop: float = 0.06
    upper_arm: float = 0.27
    forearm: float = 0.25

    @property
    def height(self) -> float:
        """Foot-to-neck height of the straight standing robot."""
        return self.torso + self.thigh + self.shin

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics: (11,) configuration -> (12, 2) keypoints."""
        return self.fk_batch(np.asarray(q, dtype=float)[None])[0]

    def fk_batch(self, Q: np.ndarray) -> np.ndarray:
        """Vectorised forward kinematics: (T, 11) -> (T, 12, 2)."""
        Q = np.asarray(Q, dtype=float)
        x, z, tor, lh, lk, rh, rk, ls, le, rs, re = Q.T

        def d(a):
            return np.stack([np.sin(a), -np.cos(a)], axis=-1)

        pelvis = np.stack([x, z], axis=-1)
        neck = pelvis + self.torso * np.stack([np.sin(tor), np.cos(tor)], axis=-1)
        l_sh = neck + np.array([-self.shoulder_x, -self.shoulder_drop])
        r_sh = neck + np.array([self.shoulder_x, -self.shoulder_drop])
        l_knee = pelvis + self.thigh * d(lh)
        l_foot = l_knee + self.shin * d(lh + lk)
        r_knee = pelvis + self.thigh * d(rh)
        r_foot = r_knee + self.shin * d(rh + rk)
        l_elb = l_sh + self.upper_arm * d(ls)
        l_hand = l_elb + self.forearm * d(ls + le)
        r_elb = r_sh + self.upper_arm * d(rs)
        r_hand = r_elb + self.forearm * d(rs + re)
        return np.stack([pelvis, neck, l_knee, l_foot, r_knee, r_foot,
                         l_sh, l_elb, l_hand, r_sh, r_elb, r_hand], axis=1)

    def fk_jacobian(self, q: np.ndarray, h: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (12, 2) and their Jacobian (12, 2, 11) by central differences."""
        q = np.asarray(q, dtype=float)
        E = np.eye(len(q)) * h
        Qs = np.concatenate([q[None], q[None] + E, q[None] - E], axis=0)
        P = self.fk_batch(Qs)
        n = len(q)
        J = (P[1:n + 1] - P[n + 1:]) / (2 * h)  # (11, 12, 2)
        return P[0], np.transpose(J, (1, 2, 0))


def link_samples(points: np.ndarray, n: int = 5) -> np.ndarray:
    """Points sampled along every robot link (endpoints included).

    ``points`` has shape (..., 12, 2); the result has shape (..., len(LINKS)*n, 2).
    """
    u = np.linspace(0.0, 1.0, n)
    a = points[..., [i for i, _ in LINKS], :]
    b = points[..., [j for _, j in LINKS], :]
    s = a[..., :, None, :] + (b - a)[..., :, None, :] * u[:, None]
    return s.reshape(*points.shape[:-2], len(LINKS) * n, 2)


def link_sample_weights(n: int = 5) -> np.ndarray:
    """Interpolation matrix W (len(LINKS)*n, 12) with samples = W @ keypoints."""
    u = np.linspace(0.0, 1.0, n)
    W = np.zeros((len(LINKS) * n, len(KEYPOINTS)))
    for k, (i, j) in enumerate(LINKS):
        W[k * n:(k + 1) * n, i] += 1.0 - u
        W[k * n:(k + 1) * n, j] += u
    return W


def per_frame_velocity_bounds(dt: float) -> np.ndarray:
    return V_MAX * float(dt)


def clip_to_limits(q: np.ndarray, q_prev: np.ndarray | None, dt: float) -> np.ndarray:
    """Project a configuration onto the joint box and the velocity box."""
    lo, hi = Q_MIN.copy(), Q_MAX.copy()
    if q_prev is not None:
        v = per_frame_velocity_bounds(dt)
        lo = np.maximum(lo, q_prev - v)
        hi = np.minimum(hi, q_prev + v)
    return np.clip(q, lo, hi)
