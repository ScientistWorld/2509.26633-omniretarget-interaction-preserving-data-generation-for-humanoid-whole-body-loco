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

Job `9563a6d6-a93` kept the object-interaction core claim intact and improved terrain foot skating: OmniRetarget reduced terrain foot-skating duration from `1.00` for keypoint IK to `0.048` and max skating speed from `42.75 cm/s` to `2.70 cm/s`. It did not reach `core_claim_plus` because terrain penetration remained high (`24.37 cm`) and terrain hand-contact preservation was only `0.08`.

Job `361e82c5-d0d` improved terrain contact preservation to `0.75`, above the paper's reported `0.72`, but still did not reach `core_claim_plus`: terrain penetration duration was `0.125`, max penetration was `21.50 cm`, and foot-skating duration was `0.286`. Artifact inspection showed failed hard constraints during the step-up transition, caused by too-tight sequential velocity bounds for the scaled planar robot.

Job `10633b10-ff5` further improved terrain contact preservation to `0.917` and reduced max penetration to `7.06 cm`, but still did not reach `core_claim_plus`: penetration duration was `0.225` and one stance frame produced a `97.98 cm/s` skating spike. The remaining failure came from the procedural terrain's discontinuous height jump and a retry seed that could move away from an already established stance foot.

## Remaining

- Run the packaged reproduction in the compute container.
- Rerun after replacing the synthetic terrain discontinuity with a short continuous ramp and seeding terrain retries from the previous frame.
- Expand from `core_claim` to `core_claim_plus` by making the terrain-step setting meaningful and successful.

## Issues

- The paper states that OmniRetarget code, retargeted datasets, and trained policies will be released, but the markdown provides no repository URL for the proposed method.
- Full Unitree/Drake/RL reproduction is outside the current workspace assets; this first job targets the core kinematic retargeting claim.
- A local smoke test could not run before container setup because `numpy` is not installed in the login Python.
- Job `bd8cb121-d23` failed due to missing local package initialization, fixed in the next revision.
- Job `d59dfd81-fe3` reached `method_runs` but exposed two method issues: object collision constraints allowed a 1 cm overlap and terrain hand contacts were only represented in the mesh, not constrained.
- Job `12774676-633` showed that the terrain task generator placed labeled hand contacts well above the terrain surface, making contact preservation impossible. The generator now places support hands on the elevated platform surface.
- Job `9563a6d6-a93` showed two remaining terrain protocol mismatches: contact labels started before the generated hands reached the support surface, and foot stance intervals crossed the discontinuous step edge. Both are now corrected in the data/evaluation definitions.
- Job `361e82c5-d0d` showed optimizer infeasibility during the step-up transition. The next revision keeps the same OmniRetarget objective and constraints but retries terrain frames with relaxed joint-velocity bounds when SLSQP returns a constraint-violating iterate.
- Job `10633b10-ff5` showed that the discontinuous synthetic step could classify near-boundary hand support as penetration and that retrying from a fresh initial pose could break a previous stance foot. The terrain is now a ramped step, and retries remain anchored to the previous frame.

## Deviations from Paper

- Paper data: OMOMO, LAFAN1, and in-house MoCap. Substitute: deterministic synthetic object and terrain interaction tasks in `data/`. Reason: the OmniRetarget release and in-house MoCap are not available from the paper markdown, and a small benchmark is needed for a reusable gym.
- Paper robot/solver: Unitree humanoids, Drake autodiff, custom sequential SOCP. Substitute: planar humanoid model and SciPy SLSQP constrained optimization. Reason: preserve the same objective and hard-constraint formulation at a scale that can run in this environment.
- Paper downstream RL: BeyondMimic-style policy training. Substitute: kinematic metrics only for the first milestone. Reason: the paper's central retargeting claim is directly measured by Table II kinematic metrics, while full RL requires unavailable simulator assets and policies.
