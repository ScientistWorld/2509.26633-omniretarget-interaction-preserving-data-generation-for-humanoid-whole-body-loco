"""Planar humanoid kinematics shared by data, baselines, and methods."""

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


@dataclass
class PlanarHumanoid:
    thigh: float = 0.40
    shin: float = 0.39
    torso: float = 0.48
    shoulder_x: float = 0.16
    upper_arm: float = 0.27
    forearm: float = 0.25

    def fk(self, q: np.ndarray) -> np.ndarray:
        x, z, torso, lh, lk, rh, rk, ls, le, rs, re = q
        pelvis = np.array([x, z])
        neck = pelvis + self.torso * np.array([np.sin(torso), np.cos(torso)])
        l_sh = neck + np.array([-self.shoulder_x, -0.06])
        r_sh = neck + np.array([self.shoulder_x, -0.06])
        l_knee = pelvis + self.thigh * np.array([np.sin(lh), -np.cos(lh)])
        l_foot = l_knee + self.shin * np.array([np.sin(lh + lk), -np.cos(lh + lk)])
        r_knee = pelvis + self.thigh * np.array([np.sin(rh), -np.cos(rh)])
        r_foot = r_knee + self.shin * np.array([np.sin(rh + rk), -np.cos(rh + rk)])
        l_elbow = l_sh + self.upper_arm * np.array([np.sin(ls), -np.cos(ls)])
        l_hand = l_elbow + self.forearm * np.array([np.sin(ls + le), -np.cos(ls + le)])
        r_elbow = r_sh + self.upper_arm * np.array([np.sin(rs), -np.cos(rs)])
        r_hand = r_elbow + self.forearm * np.array([np.sin(rs + re), -np.cos(rs + re)])
        return np.stack([pelvis, neck, l_knee, l_foot, r_knee, r_foot, l_sh, l_elbow, l_hand, r_sh, r_elbow, r_hand])


def initial_guess(source: np.ndarray) -> np.ndarray:
    pelvis = source[NAME_TO_IDX["pelvis"]]
    return np.array([pelvis[0], pelvis[1] - 0.03, 0.0, -0.18, 0.25, 0.18, -0.25, -1.05, 1.15, 1.05, -1.15], dtype=float)
