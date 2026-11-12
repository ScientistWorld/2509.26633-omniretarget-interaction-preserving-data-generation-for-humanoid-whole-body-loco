# Overview

OmniRetarget (arXiv 2509.26633) addresses humanoid motion retargeting for
object and terrain interaction. The paper argues that preserving the spatial
relationships between body, object and terrain (an interaction mesh) while
enforcing hard kinematic constraints yields higher-quality robot reference
motions than keypoint-matching retargeting: less penetration and foot skating,
with contacts preserved, including when the scene is modified for data
augmentation.

This workspace is a scaled planar reproduction packaged as a benchmark:
procedurally generated human demonstrations with varied body proportions, a
fixed planar humanoid, a hidden held-out test split generated at test time,
an evaluator that scores only the robot joint trajectory, a keypoint-IK
baseline and the OmniRetarget reference method.
