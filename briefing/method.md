# Method

OmniRetarget builds an interaction mesh from body keypoints plus dense object and terrain samples. It computes uniform-weight Laplacian coordinates over Delaunay mesh neighbors and retargets each frame by minimizing Laplacian deformation between source and target meshes.

For each frame, the implementation solves for a planar humanoid configuration with:

- Laplacian deformation objective over robot, object, and terrain mesh vertices.
- Temporal smoothness from the previous frame's configuration.
- Joint bounds and per-frame velocity bounds.
- Terrain and object non-penetration constraints.
- Stance-foot equality constraints when source foot horizontal velocity is low.
- Warm start from the previous optimized frame.

The paper uses a custom sequential SOCP solver with Drake automatic differentiation for Unitree G1/H1/Booster models. This reproduction uses SciPy SLSQP on a smaller planar humanoid benchmark while preserving the paper's computational formulation: interaction mesh Laplacian retargeting with hard kinematic constraints. The baseline is a keypoint-position IK method from the PHC/GMR objective family, which lacks interaction mesh preservation and hard contact constraints.
