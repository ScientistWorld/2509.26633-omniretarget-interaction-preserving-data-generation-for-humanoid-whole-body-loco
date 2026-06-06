# Progress

## What Works

- Read the OmniRetarget paper and extracted the core algorithm, baselines, metrics, and Table II reported numbers.
- Implemented a scaled planar humanoid benchmark with object-carry and terrain-step interaction tasks.
- Implemented the paper's core retargeting computation: interaction-mesh Laplacian objective, Delaunay neighbors, object/terrain mesh samples, warm-started per-frame constrained optimization, joint/velocity limits, non-penetration constraints, and stance-foot sticking constraints.
- Implemented a keypoint-position IK baseline from the paper's baseline family.
- Implemented method-independent evaluation that reads artifacts and writes `scoring/scores.json`.
- Job `d59dfd81-fe3` ran the full pipeline successfully in the compute container.

## Results

Job `12774676-633` reached `core_claim` on `object_interaction_kinematics`. Keypoint IK had penetration duration `1.00`, max penetration `9.73 cm`, foot skating duration `1.00`, and max skating speed `35.10 cm/s`. OmniRetarget achieved penetration duration `0.00`, max penetration approximately `0 cm`, foot skating duration `0.00`, max skating speed approximately `0 cm/s`, and contact preservation `1.00`.

Job `d59dfd81-fe3` produced the first reproduced scores. On `object_interaction_kinematics`, OmniRetarget reduced maximum penetration from `9.73 cm` to `1.00 cm`, reduced foot skating duration from `1.00` to `0.00`, reduced maximum foot skating speed from `35.10 cm/s` to approximately `0`, and preserved contact at `1.00`. However, penetration duration still reported `1.00`, so the object setting does not yet satisfy the core claim.

On `terrain_interaction_kinematics`, OmniRetarget reduced penetration duration from `0.05` to `0.00`, but terrain hand contact preservation remained `0.00` and maximum foot skating speed remained too high. This is not yet a successful reproduction of the terrain claim.

Job `e0708e5b-dfc` fixed object penetration: object penetration duration dropped from `1.00` for keypoint IK to `0.00` for OmniRetarget, with contact preservation `1.00`. Foot skating still did not satisfy the claim because the optimizer used a stricter stance detector than the evaluator (`0.12 m/s` vs `0.35 m/s`). The next revision aligned these thresholds.

## Remaining

- Run the packaged reproduction in the compute container.
- Rerun after fixing inconsistent terrain hand-contact generation.
- Expand from `core_claim` to `core_claim_plus` by making the terrain-step setting meaningful and successful.

## Issues

- The paper states that OmniRetarget code, retargeted datasets, and trained policies will be released, but the markdown provides no repository URL for the proposed method.
- Full Unitree/Drake/RL reproduction is outside the current workspace assets; this first job targets the core kinematic retargeting claim.
- A local smoke test could not run before container setup because `numpy` is not installed in the login Python.
- Job `bd8cb121-d23` failed due to missing local package initialization, fixed in the next revision.
- Job `d59dfd81-fe3` reached `method_runs` but exposed two method issues: object collision constraints allowed a 1 cm overlap and terrain hand contacts were only represented in the mesh, not constrained.
- Job `12774676-633` showed that the terrain task generator placed labeled hand contacts well above the terrain surface, making contact preservation impossible. The generator now places support hands on the elevated platform surface.

## Deviations from Paper

- Paper data: OMOMO, LAFAN1, and in-house MoCap. Substitute: deterministic synthetic object and terrain interaction tasks in `data/`. Reason: the OmniRetarget release and in-house MoCap are not available from the paper markdown, and a small benchmark is needed for a reusable gym.
- Paper robot/solver: Unitree humanoids, Drake autodiff, custom sequential SOCP. Substitute: planar humanoid model and SciPy SLSQP constrained optimization. Reason: preserve the same objective and hard-constraint formulation at a scale that can run in this environment.
- Paper downstream RL: BeyondMimic-style policy training. Substitute: kinematic metrics only for the first milestone. Reason: the paper's central retargeting claim is directly measured by Table II kinematic metrics, while full RL requires unavailable simulator assets and policies.
