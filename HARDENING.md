# Final hardening

The final 325-input audit passes: a genuine contact-aware improvement scores above the reference beyond paired recording noise, the no-mesh ablation scores below it beyond noise, and all replayed major shortcuts score below it. The reference repeat has identical per-instance measurements. One honest alternative, keypoint matching plus contact costs, remains statistically tied; that is a remaining resolution limit.

The final evidence is in `eval/test/audit_results/ae348dd707e3/`. Job `b6ab06be-ec7` completed the resumed audit with exit 0 and `python validate.py` passed. Official `scoring/scores_test.json` and `scores_train.json` contain only the final end-to-end reference and independent keypoint-IK baseline. Audit ranks and shortcut results are diagnostic evidence and are never ranked artifacts.

## Paper fidelity and REVIEW.md

Read `paper/2509.26633.md`, especially Sections III/V and Appendices A–D. REVIEW's planar substitute has been replaced by real OMOMO large-box manipulation and public climbing motion capture, the 3D Unitree G1 (36 configuration coordinates, 29 articulated joints), and volumetric Delaunay interaction meshes. Held-out subjects/recordings, private construction and private diagnostics live only in `eval/test/`.

The reference implements the paper's actual computation: uniformly weighted Laplacian deformation, body/scene correspondences, object-local coordinates, previous-frame warm starts, sequential quadratic approximation with SOCP trust regions, hard linearized collision/joint/velocity/stance constraints, and nominal-then-augmented retargeting with lower-body anchoring and initial-foot constraints. It uses 15 anatomical points, 100 deterministic surface samples, up to 10 iterations per frame (50 initially), trust radius 0.2, mesh weight 10 and smoothing 0.2. Clarabel and MuJoCo replace Drake's solver/derivative infrastructure; no learned surrogate replaces the algorithm.

Two port errors were corrected after the first combined audit. Crouched zero-joint initialization could lie almost 50 cm underground, beyond the trust radius. The authors' released initial-frame retry, removing only the SOC step bound while retaining physical inequalities, is restored. The soft-slack fallback is removed. Collision derivatives now query canonical geometry-type order and explicitly restore witness order: [MuJoCo 3.3.0's native-CCD early return](https://github.com/google-deepmind/mujoco/blob/3.3.0/src/engine/engine_support.c#L1287) otherwise leaves the pair internally reordered. The unchanged finite-difference check passes on 1,324 directions/pairs (median absolute error 8.39e-11, 90th percentile 0.00413; thresholds 0.001/0.02). Maximum discrepancy is 0.08262, so this is not a claim of exact derivatives at every collision feature.

REVIEW's mesh-effect uncertainty prompted more representative motion coverage, without changing optimizer, thresholds or weights: all 99 usable held-out box clips, up to 600 frames (20 seconds) instead of 150, and two modified scenes per clip. Public object windows received the same length increase. Final test: 99 nominal object, 14 nominal terrain and 212 modified-scene instances; train: 44 inputs. The final mesh ablation is significantly worse overall, rather than an effect asserted from the augmented experiment alone. Keypoint-plus-contact remains unresolved as reported below.

## Better and weaker methods, before and after

Rank differences are relative to reference 0. Each audit uses its own cohort population standard deviations; different datasets/cohorts have different units. Before/after columns document the failures and repairs, not a common-scale treatment effect. The first 3D audit preceded initialization/Jacobian fixes. The corrected 100-input audit established headroom but left the no-mesh effect uncertain. All final comparisons were rerun from final scientific code/data, including the reference and baseline.

| Method | First 3D audit | Corrected 100 inputs | Final 325 inputs | Final paired 95% interval |
|---|---:|---:|---:|---|
| Reference | +0.00000 | +0.00000 | +0.00000 | — |
| Contact cost 10 | +0.04326 | +0.07009 | +0.04250 | [+0.00952, +0.08008] |
| Contact cost 100 | -0.03511 | +0.10513 | +0.07166 | [+0.02836, +0.12420] |
| Contact cost 1000 | -0.18614 | +0.16292 | +0.14581 | [+0.06832, +0.23076] |
| Contact + support cost 10 | not run | +0.06752 | +0.03653 | [-0.00064, +0.07403] |
| Contact + support cost 100 | not run | +0.09830 | +0.05390 | [+0.00113, +0.11510] |
| Smaller steps + contacts | not run | -0.07260 | -0.11731 | [-0.22538, -0.00952] |
| No mesh, constrained keypoints | -0.36767 | -0.16463 | -0.12629 | [-0.25677, -0.01740] |
| Keypoints + contacts | -0.18960 | +0.00659 | -0.04076 | [-0.15491, +0.07247] |
| Longer private simulator search | -0.01267 | +0.01564 | +0.00605 | [-0.00720, +0.02128] |
| Independent keypoint IK | -4.22915 | -4.98447 | -4.98135 | [-5.58277, -4.33671] |
| Exact reference repeat | +0.00000 | +0.00000 | +0.00000 | [+0.00000, +0.00000] |

Intervals use 1,000 fixed-seed paired bootstrap draws of complete recordings: 99 object clips and three climbing recordings, stratified by task. Every method sees the same inputs and draws. All nominal and modified windows of a recording remain together; 325 windows are not treated as independent recordings. Normalization is fixed to the measured cohort while bootstrapping.

The contact-1000 variant improves contact from 0.53071/0.68401/0.55622 to 0.55775/0.75635/0.58235 (object/terrain/modified). Its gains survive the other collision, support and fidelity terms. On the honest-only normalization cohort its final advantage is +0.30780, versus -0.75259 for no-mesh and -11.91030 for keypoint IK, so the ordering is not created by invalid probes in the cohort. The larger contact cost does not reach the metric bounds.

Train uses the same interface and measurements on comparable real motions. Contact-1000 improves by +0.10263 [0.02104,0.18160]; no-mesh loses by -1.02829 [-1.60144,-0.40277]; keypoint-plus-contact loses by -1.25102 [-1.73813,-0.70660]; keypoint IK loses by -8.83871. Smaller steps score +0.41181 on train (interval crosses zero) and -0.11731 on test (interval below zero), illustrating that a public-split preference need not transfer.

Original REDTEAM planar results were contact refinement +1.1089, no-mesh **+0.3468**, keypoint IK -3.5768, with reference 0. The preceding planar hardening reported contact refinement +0.15 [0.09,0.23], no-mesh -0.57 [-0.73,-0.41], and keypoint-plus-contact +0.05 [-0.04,0.15]. Those historical synthetic results did not establish the 3D claim and are not retained as this gym's reference.

## Reference before and after; bounds and headroom

The incoming stored 3D reference was failed: one modified trajectory jumped its base by 55 cm/frame. It therefore received all 27 worst values (contact 0; durations 1; penetration cap 30 cm, skating cap 300 cm/s, path/pose caps 100 cm). A preceding base-velocity fix had not been rerun over the official split. The final reference is valid. Its self-relative rank remains 0 by definition, so raw quantities provide the useful before/after comparison.

For the first valid 3D audit, before the initialization/Jacobian repair, the reference had the following values; the final column also reflects the expanded corpus and longer motions:

| Experiment | Contact before → final | Penetration duration before → final | Support failure before → final |
|---|---|---|---|
| Object | 0.35404 → 0.53071 | 0.53683 → 0.19648 | 0.51515 → 0.00149 |
| Terrain | 0.68430 → 0.68401 | 0.17381 → 0.14286 | 0.20247 → 0.20247 |
| Modified scenes | 0.55332 → 0.55622 | 0.39045 → 0.20276 | 0.39803 → 0.01448 |

Final reference values (durations are fractions):

| Metric | Object | Terrain | Modified scenes |
|---|---:|---:|---:|
| Penetration duration ↓ | 0.196484 | 0.142857 | 0.202764 |
| Maximum penetration, cm ↓ | 12.123565 | 3.743095 | 12.345218 |
| Skating duration ↓ | 0.000400 | 0.000000 | 0.000246 |
| Maximum skating, cm/s ↓ | 0.477958 | 0.364773 | 0.458088 |
| Contact preservation ↑ | 0.530711 | 0.684006 | 0.556221 |
| Unwanted contact duration ↓ | 0.564775 | 0.151399 | 0.529193 |
| Support failure duration ↓ | 0.001487 | 0.202472 | 0.014485 |
| Pelvis path error, cm ↓ | 6.647290 | 6.553555 | 10.509128 |
| Body pose error, cm ↓ | 10.972963 | 13.918308 | 11.599551 |

The reference is at a bound only for nominal-terrain skating duration (0). Its maximum skating speed still varies below the event threshold. No scored metric is constant across the honest cohort. Headroom comes from real held-out interaction and changed-scene transfer, not matching paper numbers or merely observing an unsaturated metric. The simple keypoint baseline is far below the reference, and the tested improvement does not saturate the task.

The final reference substantially reduces penetration and skating relative to keypoint IK; object/modified contact also improves. It is not best on every metric: keypoint IK has better source fidelity and nominal-terrain contact/support. Reference mean-window maximum penetration remains 12.12/3.74/12.35 cm, much worse than the paper's reported approximately 1.3 cm. This reproduction establishes a constrained kinematic tradeoff and mesh benefit on the subset, not the paper's exact numerical quality or complete physical feasibility.

## Major shortcuts: fixes and replay scores

Before values below are REDTEAM's original planar ranks. After values use the common final 29-method audit cohort. Every after case ran through `bash scripts/evaluate_test.sh` with no arguments on the full hidden split. An attack returning a legal but poor configuration is scored on that configuration; an invalid run receives the complete worst vector.

| Shortcut | Before | Fix | Final after |
|---|---:|---|---:|
| Disconnected body / unchecked points | +3.1164 / +2.2503 | Ignore submitted body points; recompute all geometry from q and evaluator-owned robot | -3.60438 (detached-points replay) |
| Mutable task / comparison assets | +3.1625 | Fresh per-method inputs; measurement uses trusted in-memory originals and private assets | -3.60438 |
| Impossible configurations | 0 tie | Finite/shape/base/quaternion/joint/velocity checks before scoring | -12.88119, failed |
| Empty output | 0 tie | Prepopulate every experiment's worst vector; missing one input fails whole run | -12.88119, failed |
| Missing required experiment | 0; improved-object-only +1.1089 | Same whole-run completeness gate regardless of the supplied experiment's quality | -12.88119, failed |
| Edit train score list / use a new name | 0 / 0 ties | Discover every registration on disk; train scores never select test candidates | -3.60438 / -3.60438 |
| Claimed metrics, flags or forged score files | -2.7917 | Only raw q is read; evaluator overwrites all scored results | -4.98135 (actual IK output) |
| NaN / malformed output | undefined / no score | Fail closed and emit all 27 finite worst values | -12.88119 each, failed |
| Joint, velocity, quaternion or base-teleport violations | not previously tested separately | Explicit whole-trajectory validity gates | -12.88119 each, failed |
| Hidden answers / chmod attempt | not previously tested | Landlock denies private test and git-history reads to method and descendants | -3.60438 |
| Freeze a constant pose | -9.5616 | Contact, unwanted contact, support and fidelity all measured | -3.60438 |
| Declare contact at every frame | not previously tested | Contact/clearance labels are reconstructed from source, not method flags | -0.21626 |

All nine invalid/missing probes have every scored metric in every experiment populated, not an empty entry. Constant, detached-points, mutable-input, filter and hidden-read probes return the same measured constant trajectory and therefore the same score. The malformed probe cannot preserve stale scores. No fixed shortcut ties the reference.

Original finite-grid brute force (3^11 configurations per frame, 14.17 million evaluations) lost by -5.675 on the planar task. That finite enumeration is not viable at the new 36-coordinate floating-base scale and unseen motions. Rather than claim to have exhaustively enumerated a continuous 3D problem, the final audit measures private search using its own public simulator: 150 initial/30 subsequent iterations instead of 50/10. Its +0.00605 [-0.00720,0.02128] is indistinguishable from the reference and below the honest improvement. Both search and evaluation-oracle reimplementations are bound by process-tree CPU accounting, not a spoofable method-reported evaluation count.

## Scoring and execution contract

- The method submits raw finite `[T,36]` configurations only. Test owns geometry, limits, thresholds, annotations, test size and every comparison. Extra numbers, points, success flags and verdicts are ignored. Contact uses absolute surface distance so penetration cannot masquerade as contact. Support loss and unwanted hand/swing-foot contact close floating/freezing/overcontact gains.
- Any crash, malformed/missing/partial output, impossible value, joint/base constraint violation or CPU-budget failure receives contact 0 and all other worst values in all three experiments. Thus failed runs are coordinatewise dominated by solving runs; another metric cannot compensate for failure.
- Ranking sums `experiment_weight * abs(coefficient) * direction * (value - reference) / population_std_over_attempts`. Direction comes only from `higher_is_better`; zero-variance terms contribute zero. Only each experiment's `results` in the two split score files are ranked. Audit rank/verdict files are not inputs to ranking.
- Weights are object 0.35, terrain 0.35, modified scenes 0.30. Metric coefficient magnitudes in the table's order are 1, 0.5, 0.8, 0.4, 1, 0.8, 0.8, 0.3, 0.3. Support failure was added at 0.8 to give stable support comparable importance to skating/unwanted contact; this closes an unmeasured constraint. No weight or threshold was tuned after the corrected/expanded audit. Paper numbers remain unchanged under `_paper_reported_results`, outside ranked results.
- This method needs no training. The evaluator invokes arbitrary registered procedures on unseen inputs, then constructs annotations and scores raw outputs. Landlock restricts methods/descendants to public reads and invocation/tmp writes and denies `eval/test/` and git history, except their own fresh invocation directory. This boundary protects current test inputs/answers, not leftover processes or a Python environment.
- Each method runs sequentially with nothing beside it. Fixed input construction, sampling and solver ordering yield an exact repeated reference (maximum metric difference 0, all per-instance measurements identical). The only budget is 30 process-tree CPU seconds per instance (9,750 test; 1,320 train), with one numerical-library thread. There is no wall-clock cutoff anywhere on the method path. Final reference used 773.9 CPU seconds; baseline 102.6; improved contact-1000 821.6. Runtime is never scored.
- Train and test are independent implementations with no imports between them; neither imports `method/`. Both use the same input/output contract. All answer-bearing corpora, constructors and detailed test audits are under `eval/test/`. Public files contain aggregate results only.

## Remaining limits and shortcuts left

Keypoint matching plus contact is an honest alternative, not an exploit: final rank -0.04076 [-0.15491,0.07247]. Expanded coverage resolves the no-mesh ablation overall but does not distinguish this alternative reliably on test. No hard-coded rule beating the honest improvement was found; this is not a proof against every possible rule. Real unseen human trajectories replace the hand-writable synthetic process. No known paying shortcut is intentionally left open. General private simulator search remains expressible and allowed within the measured CPU budget; the tested longer search does not win beyond noise. Previously losing copy-input, mutable-baseline and impossible-keypoint variants were not replayed separately: baseline artifacts cannot alter fixed reference values, and detached/impossible points have no effect on q-derived measurements. Their major mechanisms are covered by the replayed forgery, mutation and detached-point probes.

There are only two held-out object subjects and three climbing recordings (one public climbing recording), so recording-bootstrap intervals do not establish subject-population generalization. Contact labels are geometric, not force annotations. Self-collision, balance, torque, downstream RL and hardware success are unmeasured. Convex collision meshes are reduced to at most 128 vertices; MuJoCo/Clarabel numerical behavior differs from Drake. The nonlinear constraint residuals and imperfect contact above remain. PHC, GMR and VideoMimic are preserved paper table entries, not reproduced baseline implementations. Milestone is `core_claim` for this kinematic subset, with these limits; no higher milestone or full-paper reproduction is claimed.

## Packaging and verification

`bash scripts/download.sh` contains real idempotent public fetches for the pinned authors' release, robot resources, OMOMO and public climbing data; it retains only public train subjects/recording. The small generated private pack is tracked in `eval/test/data/`, with private rebuild provenance there. Downloads, runtime outputs and containers are ignored. No committed code references `shared/`. Dependencies are pinned in `environment/setup.sh`; OS dependencies are in `environment/container.def`.

The final audit and validator ran in the compute container. Scheduler interruptions required checkpointed resumption: scientific manifests cover sources, assets, registrations, dependencies and scoring definitions, and no completed run from a changed scientific setup was reused. The first full audit failed the headroom check; the derivative check caught a real port error before further science; the corrected 100-input audit motivated the final coverage increase. The last job completed every remaining shortcut replay and paired analysis. No training/inference/data generation ran on the login node. `scripts/run.sh` is restored to end-to-end reference reproduction; private `eval.test.audit` retains the complete verification workflow.

`validate.py` passes. Its two heuristic warnings concern fixed input-scene generation and bootstrap resampling in private helper modules; the actual evaluator reads q artifacts and recomputes every scored number. Official test/train score files are restored solely from final reference/baseline runs; paper-reported rows are unchanged. Briefing and scoring documents describe the final task without prescribing the reference method.
