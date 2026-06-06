# Overview

OmniRetarget addresses humanoid motion retargeting for object and terrain interaction. The paper argues that preserving interaction geometry and enforcing hard kinematic constraints produces higher-quality reference trajectories than keypoint-only retargeting, reducing penetration and foot skating while maintaining contacts.

This workspace packages a scaled reproduction of the core kinematic claim. It provides procedural interaction tasks, a keypoint IK baseline, an OmniRetarget-style interaction-mesh optimizer, and a reusable evaluator for the paper's main kinematic metrics.
