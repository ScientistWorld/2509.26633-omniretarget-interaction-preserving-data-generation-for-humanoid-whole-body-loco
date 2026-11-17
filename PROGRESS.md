# Progress

## What Works

- Read the OmniRetarget paper and extracted the core algorithm, baselines, metrics, and Table II reported numbers.
- Implemented a scaled planar humanoid benchmark with object-carry and terrain-step interaction tasks.
- Implemented the paper's core retargeting computation: interaction-mesh Laplacian objective, Delaunay neighbors, object/terrain mesh samples, warm-started per-frame constrained optimization, joint/velocity limits, non-penetration constraints, and stance-foot sticking constraints.
- Implemented a keypoint-position IK baseline from the paper's baseline family.
- Implemented method-independent evaluation that reads artifacts, loads trusted benchmark tasks by task ID, writes reference-compatible `scoring/scores.json`, and writes all artifact metrics to `scoring/diagnostics.json`.
- Job `d59dfd81-fe3` ran the full pipeline successfully in the compute container.

## Results

Job `12774676-633` reached `core_claim` on `object_interaction_kinematics`. Keypoint IK had penetration duration `1.00`, max penetration `9.73 cm`, foot skating duration `1.00`, and max skating speed `35.10 cm/s`. OmniRetarget achieved penetration duration `0.00`, max penetration approximately `0 cm`, foot skating duration `0.00`, max skating speed approximately `0 cm/s`, and contact preservation `1.00`.

Job `d59dfd81-fe3` produced the first reproduced scores. On `object_interaction_kinematics`, OmniRetarget reduced maximum penetration from `9.73 cm` to `1.00 cm`, reduced foot skating duration from `1.00` to `0.00`, reduced maximum foot skating speed from `35.10 cm/s` to approximately `0`, and preserved contact at `1.00`. However, penetration duration still reported `1.00`, so the object setting does not yet satisfy the core claim.

On `terrain_interaction_kinematics`, OmniRetarget reduced penetration duration from `0.05` to `0.00`, but terrain hand contact preservation remained `0.00` and maximum foot skating speed remained too high. This is not yet a successful reproduction of the terrain claim.

Job `e0708e5b-dfc` fixed object penetration: object penetration duration dropped from `1.00` for keypoint IK to `0.00` for OmniRetarget, with contact preservation `1.00`. Foot skating still did not satisfy the claim because the optimizer used a stricter stance detector than the evaluator (`0.12 m/s` vs `0.35 m/s`). The next revision aligned these thresholds.

Job `9563a6d6-a93` kept the object-interaction core claim intact and improved terrain foot skating: OmniRetarget reduced terrain foot-skating duration from `1.00` for keypoint IK to `0.048` and max skating speed from `42.75 cm/s` to `2.70 cm/s`. It did not reach `core_claim_plus` because terrain penetration remained high (`24.37 cm`) and terrain hand-contact preservation was only `0.08`.

Job `361e82c5-d0d` improved terrain contact preservation to `0.75`, above the paper's reported `0.72`, but still did not reach `core_claim_plus`: terrain penetration duration was `0.125`, max penetration was `21.50 cm`, and foot-skating duration was `0.286`. Artifact inspection showed failed hard constraints during the step-up transition, caused by too-tight sequential velocity bounds for the scaled planar robot.

Job `10633b10-ff5` further improved terrain contact preservation to `0.917` and reduced max penetration to `7.06 cm`, but still did not reach `core_claim_plus`: penetration duration was `0.225` and one stance frame produced a `97.98 cm/s` skating spike. The remaining failure came from the procedural terrain's discontinuous height jump and a retry seed that could move away from an already established stance foot.

Job `8f7aad22-38d` reached `core_claim_plus`. On `object_interaction_kinematics`, OmniRetarget achieved penetration duration `0.00`, max penetration approximately `0 cm`, foot-skating duration `0.00`, max skating speed approximately `0 cm/s`, and contact preservation `1.00`. On `terrain_interaction_kinematics`, OmniRetarget achieved penetration duration `0.00`, max penetration approximately `0 cm`, foot-skating duration `0.00`, max skating speed approximately `0 cm/s`, and contact preservation `1.00`. The keypoint IK diagnostic baseline retained severe skating or interaction failures, so the benchmark distinguishes the paper's method, but it is omitted from final `scores.json` because the substitute-data baseline is not directly paper-consistent.

Audit job `424fab31-265` confirmed the packaged run. The generated keypoint IK artifacts remain useful diagnostics on the synthetic benchmark: on the object task, keypoint IK has penetration duration `1.00`, max penetration `9.73 cm`, foot-skating duration `1.00`, and max skating speed `35.10 cm/s`; on the terrain task, keypoint IK has foot-skating duration `1.00`, max skating speed `62.06 cm/s`, and contact preservation `0.00`. OmniRetarget retained near-zero penetration and skating with `1.00` contact preservation on both tasks. `python validate.py` passes.

Current revision tightens scoring integrity before the next audit job: `eval/` now ignores artifact-supplied `source_file` paths and loads tracked task files from `data/synthetic_tasks/`; normal method and baseline scripts no longer regenerate those tracked benchmark files; and `reference.json` names the paper Table II baselines as `PHC`, `GMR`, and `VideoMimic` rather than conflating them with the local keypoint IK diagnostic baseline.

Job `9e421c18-e41` verified the hardened trusted-task evaluator computed the expected metrics, but failed validation because the local `keypoint_ik` diagnostic baseline was written into `scores.json` while it is not a paper-reported method in `reference.json`. The evaluator now writes `keypoint_ik` and other extra artifacts to `scoring/diagnostics.json` and keeps `scores.json` restricted to methods present in `reference.json`.

Job `f03c43a4-801` passed the packaged CPU audit after the diagnostics split. `scripts/reproduce.sh` runs baseline artifacts, OmniRetarget artifacts, standalone evaluation, and `python validate.py --compare` successfully in the compute container. A later packaging audit tightened the metric contract so `scores.json`, `scores_train.json`, and `scores_test.json` all use metrics declared in `reference.json`, while the diagnostic keypoint IK baseline remains in `scoring/diagnostics*.json`.

Current final scoring supports `core_claim` conservatively. On `terrain_interaction_kinematics`, OmniRetarget achieves `0.00` penetration duration, approximately `0 cm` max penetration, `0.00` foot-skating duration, approximately `0 cm/s` max foot skating, and `1.00` normalized contact preservation. On `object_interaction_kinematics`, OmniRetarget achieves near-zero penetration and skating, but the stricter normalized contact score is `0.705`, below the paper's reported `0.96`; this is useful evidence for the hard-constraint part of the method but not enough to claim an additional milestone.

Final verification rejected an attempted object-contact equality constraint: CPU audit jobs showed it did not improve the object contact score and introduced extra foot skating. The final method is therefore the last validated interaction-mesh constrained optimizer without that additional unscored constraint.

## Remaining

- Downstream RL policy learning and full Unitree/Drake retargeting are not reproduced because the required simulator assets, robot-specific setup, and released OmniRetarget code are unavailable.
- Higher milestones would require implementing additional paper sections beyond the Table II kinematic retargeting experiments.

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
- Job `8f7aad22-38d` passed the packaged reproduction and no remaining issue blocks the reusable kinematic gym.
- The local keypoint IK baseline is a diagnostic baseline for the synthetic substitute tasks, not a reproduced PHC/GMR/VideoMimic paper baseline. It appears in `scoring/diagnostics.json`, while `scoring/scores.json` is restricted to paper-reference method names so `validate.py` remains strict.
- Job `f03c43a4-801` confirmed the packaged reproduction in the compute container. A subsequent audit found stale reference/split-score inconsistencies; these are now fixed and `python validate.py --compare` passes locally.
- Final milestone is downgraded from `core_claim_plus` to `core_claim` because the implementation uses SciPy SLSQP on planar synthetic data, and no augmentation or downstream RL experiment is scored.
- An attempted object-contact equality constraint was tested and removed because it worsened the scored trajectory constraints; the retained scores correspond to the validated pre-attempt method.

## Deviations from Paper

- Paper data: OMOMO, LAFAN1, and in-house MoCap. Substitute: deterministic synthetic object and terrain interaction tasks in `data/`. Reason: the OmniRetarget release and in-house MoCap are not available from the paper markdown, and a small benchmark is needed for a reusable gym.
- Paper robot/solver: Unitree humanoids, Drake autodiff, custom sequential SOCP. Substitute: planar humanoid model and SciPy SLSQP constrained optimization. Reason: preserve the same objective and hard-constraint formulation at a scale that can run in this environment.
- Paper downstream RL: BeyondMimic-style policy training. Substitute: kinematic metrics only for the first milestone. Reason: the paper's central retargeting claim is directly measured by Table II kinematic metrics, while full RL requires unavailable simulator assets and policies.

### [2026-11-12 04:40] - core_claim (iteration 11, hardener)
- Jobs: `47e6081b-653` (cancelled: baseline used numeric gradients; reference floated without the paper's direct scaling), `ca61137d-357` (36-instance test; fixes validated), `9f74fa16-ef7` (final: 60 train / 120 test instances, official scores + combined variants/shortcuts run; optional combined-train pass cancelled after a node slowdown).
- Final official scores: `scoring/scores_train.json`, `scoring/scores_test.json`; audit results in `redteam/hardening/results/`; report in `HARDENING.md`.

### [2026-11-14] - core_claim (iteration 12, hardener)
- Jobs: `7a16b974-f50` (exploration, cancelled: 8 workers x multi-threaded BLAS on a 1-CPU allocation inflated CPU time ~300x), `a6db59d2-208` (exploration of 5 candidate setups x 10 methods on a 60-instance dev split, `redteam/hardening/results/explore/`), `80d6fa58-151` (final check; official run completed, cancelled after 3 preemptions), `301cd2e9-fd7` (resumable chunked audit: variants, shortcuts, private-search probe, tamper check, CPU-budget checks, train + test).
- Final official scores: `scoring/scores_train.json`, `scoring/scores_test.json` (= `redteam/hardening/results/official/`); audit in `redteam/hardening/results/combined_{test,train}/`; report in `HARDENING.md`.

## Final hardening, 2026-11-17

Read the paper and both reports. Completed the 3D/public-real-motion evaluator boundary and metadata fixes together before computation. Removed the penalty-slack solver fallback, corrected stance/skating units, separated support/contact constraints, protected hidden assets with Landlock, and built a single sequential official/variant/shortcut audit. The incoming stored reference was failed; final scientific headroom remains to be verified in the submitted job. Earlier planar summaries are historical.
