# Method (reference implementation: `method/omniretarget.py`)

OmniRetarget retargets a demonstration frame by frame by preserving an
*interaction mesh* while enforcing hard kinematic constraints (paper Sec. III-A,
Alg. 5, App. D).

- **Preprocessing (App. A, Eq. 7/8):** the demonstration is scaled by
  `alpha = h_robot / h_human` (heights from the demonstrator's bone lengths)
  about the ground point below the initial pelvis. The human keypoints and the
  source-scene samples are scaled together, so the scaled demonstration is
  self-consistent; it coincides with the nominal scene the robot acts in (the
  benchmark's `nominal_scene`), except that the object keeps its true size.
- **Interaction mesh:** Delaunay triangulation (the 2D analogue of the paper's
  tetrahedralisation) of the scaled source keypoints plus dense samples of the
  object surface (32 points) and the terrain (every 5 cm, including all polyline
  vertices). The target mesh uses the same connectivity with the robot's
  forward-kinematic keypoints and the corresponding target-scene samples. Meshes
  are expressed in the object frame when an object is present (App. C-2).
- **Per-frame problem (Eq. 3):** minimise the uniform-weight Laplacian
  deformation energy plus `||q - q_{t-1}||_Q^2`, subject to non-penetration of
  sampled points along every robot link (3b), joint limits (3c), per-frame
  velocity limits (3d), and stance-foot sticking (3e). A foot is in stance when
  its horizontal speed in the source motion is below 1 cm/s.
- **Solver (App. D):** sequential convex programming warm-started from
  `q*_{t-1}`: each iteration uses the Gauss-Newton quadratic model of the
  Laplacian residuals, linearised collision and stance constraints, exact
  joint/velocity bounds, and the second-order-cone trust region
  `||dq||_2 <= 0.2`; at most 10 iterations per frame. Each convex sub-problem is
  solved exactly with SciPy SLSQP; an L-infinity elastic slack on the linearised
  collision rows keeps it feasible. Output configurations are projected onto the
  joint/velocity box.
- **Frame 0:** keypoint IK on the scaled demonstration, the demonstrator's
  initially planted feet constrained onto the target ground, 40 iterations.
- **Augmentation (Sec. III-B, App. C):** object augmentations first solve the
  nominal motion `q*_bar` in the nominal scene, then re-solve in the augmented
  scene with `||q_t - q*_bar_t||_W^2` (W = 20 on base and legs, 0 on torso and
  arms; Eq. 4) and the initial feet fixed to the nominal ones (Eq. 5). Terrain
  augmentations add the section's "additional constraints" for stable ground
  contact on the modified terrain: a foot the demonstrator plants is constrained
  onto the target terrain in the frame it lands ((3e) then keeps it there).

Scaled-down relative to the paper: planar 11-DoF robot instead of a 3D Unitree
G1 with quaternion floating base, sampled link points instead of collision
meshes, procedural demonstrations instead of OMOMO/LAFAN1/MoCap, SciPy instead of
Drake. Not reproduced: downstream RL policies and hardware experiments.

The baseline `baseline/keypoint_ik.py` is the GMR/PHC objective family: per-frame
keypoint-position IK on the directly scaled demonstration, with joint and
velocity limits only and no knowledge of the object or terrain.
