# Method (reference implementation: `method/omniretarget.py`)

OmniRetarget retargets a demonstration frame by frame by preserving an
*interaction mesh* while enforcing hard kinematic constraints (paper Sec. III-A,
Alg. 5, App. D).

- **Preprocessing (App. A, Eq. 7/8):** the demonstration is scaled by
  `alpha = h_robot / h_human` (heights from the demonstrator's bone lengths). The
  human keypoints and the source-scene samples are scaled together about the
  ground point below the pelvis in each frame, so the scaled demonstration stays
  self-consistent and keeps its walking path.
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

Scaled-down relative to the paper: planar 11-DoF robot instead of a 3D Unitree
G1 with quaternion floating base, sampled link points instead of collision
meshes, procedural demonstrations instead of OMOMO/LAFAN1/MoCap, SciPy instead of
Drake. Not reproduced: downstream RL policies and hardware experiments.

The baseline `baseline/keypoint_ik.py` is the GMR/PHC objective family: per-frame
keypoint-position IK on the directly scaled demonstration, with joint and
velocity limits only and no knowledge of the object or terrain.
