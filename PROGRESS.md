# Progress

## What Works

- Read the OmniRetarget paper and extracted the core algorithm, baselines, metrics, and Table II reported numbers.
- Implemented a scaled planar humanoid benchmark with object-carry and terrain-step interaction tasks.
- Implemented the paper's core retargeting computation: interaction-mesh Laplacian objective, Delaunay neighbors, object/terrain mesh samples, warm-started per-frame constrained optimization, joint/velocity limits, non-penetration constraints, and stance-foot sticking constraints.
- Implemented a keypoint-position IK baseline from the paper's baseline family.
- Implemented method-independent evaluation that reads artifacts and writes `scoring/scores.json`.
- Job `d59dfd81-fe3` ran the full pipeline successfully in the compute container.

## Results

Job `d59dfd81-fe3` produced the first reproduced scores. On `object_interaction_kinematics`, OmniRetarget reduced maximum penetration from `9.73 cm` to `1.00 cm`, reduced foot skating duration from `1.00` to `0.00`, reduced maximum foot skating speed from `35.10 cm/s` to approximately `0`, and preserved contact at `1.00`. However, penetration duration still reported `1.00`, so the object setting does not yet satisfy the core claim.

On `terrain_interaction_kinematics`, OmniRetarget reduced penetration duration from `0.05` to `0.00`, but terrain hand contact preservation remained `0.00` and maximum foot skating speed remained too high. This is not yet a successful reproduction of the terrain claim.

## Remaining

- Run the packaged reproduction in the compute container.
- Rerun after tightening object non-penetration and terrain hand-contact constraints.
- Reach `core_claim` on at least one interaction setting, then expand to the second setting.

## Issues

- The paper states that OmniRetarget code, retargeted datasets, and trained policies will be released, but the markdown provides no repository URL for the proposed method.
- Full Unitree/Drake/RL reproduction is outside the current workspace assets; this first job targets the core kinematic retargeting claim.
- A local smoke test could not run before container setup because `numpy` is not installed in the login Python.
- Job `bd8cb121-d23` failed due to missing local package initialization, fixed in the next revision.
- Job `d59dfd81-fe3` reached `method_runs` but exposed two method issues: object collision constraints allowed a 1 cm overlap and terrain hand contacts were only represented in the mesh, not constrained.

## Deviations from Paper

- Paper data: OMOMO, LAFAN1, and in-house MoCap. Substitute: deterministic synthetic object and terrain interaction tasks in `data/`. Reason: the OmniRetarget release and in-house MoCap are not available from the paper markdown, and a small benchmark is needed for a reusable gym.
- Paper robot/solver: Unitree humanoids, Drake autodiff, custom sequential SOCP. Substitute: planar humanoid model and SciPy SLSQP constrained optimization. Reason: preserve the same objective and hard-constraint formulation at a scale that can run in this environment.
- Paper downstream RL: BeyondMimic-style policy training. Substitute: kinematic metrics only for the first milestone. Reason: the paper's central retargeting claim is directly measured by Table II kinematic metrics, while full RL requires unavailable simulator assets and policies.
