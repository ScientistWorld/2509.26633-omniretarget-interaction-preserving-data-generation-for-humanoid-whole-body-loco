# Does this gym reward better research?

**Only partly. A genuine improvement scores higher and simple unconstrained IK scores lower, but removing the paper's interaction-mesh objective also improves the audit rank. The synthetic gym therefore does not establish the benefit of that central idea. More seriously, disconnected body points beat the reference on every scored metric, and omitting a required experiment can still beat the reference. This scoring is not a reliable measure of scientific contribution.**

## Scientific-quality checks

These are actual hidden-test runs, with the reference assigned rank difference zero. Each linked directory contains its own `run.sh`, `status.json`, and the test's unmodified `scores_test.json`.

| Variant | What changed | Rank difference | Finding |
|---|---|---:|---|
| [Reference, seed 0](redteam/reference_seed0/) | Original method, original public tasks | 0 | Reproduces the packaged reference scores exactly. |
| [Genuine contact refinement](redteam/better_contact_refinement/) | Optimize four arm angles using robot forward kinematics, contact error, joint limits, per-frame increment limits and collision inequalities; retain the other reference coordinates | **+1.1089** | Object contact rises **0.708525 → 0.999477**, velocity error falls **11.2185 → 10.8523 cm/s**, and every other score is unchanged. A genuine Pareto improvement is rewarded. |
| [More optimization](redteam/better_budget/) | Increase solver iteration limits 140/220 → 700/1100 and tighten tolerance | −0.4482 | Additional compute worsens terrain skating: duration 0 → 0.1, maximum speed approximately 0 → 157.30 cm/s. The lower rank is appropriate for these outputs; this is not evidence of a better result being punished. |
| [Remove the mesh objective](redteam/weaker_no_mesh/) | Keep keypoint matching, smoothing and constraints; remove Laplacian deformation energy | **+0.3468** | Object contact improves to 0.827820, while maximum skating and some motion errors worsen. This honest simpler alternative outranks the reference under the audit normalization. The synthetic problem does not demonstrate that the mesh term helps overall. |
| [Weaker keypoint IK](redteam/weaker_keypoint_ik/) | Run the supplied basic baseline, identified as the candidate for scoring | **−3.5768** | Large penetration/skating losses outweigh better source matching. The basic baseline is clearly below the reference. |
| [Brute-force grid](redteam/brute_force_grid/) | Enumerate all configurations in a 3-level grid over all 11 coordinates at each frame | **−5.6750** | Does not solve the test. |

All nonzero changes exceed the measured seed spread, which is zero. The refinement is a real improvement over the delivered reference, not a claim to reproduce the paper's full algorithm or repair its existing terrain velocity violations.

**Rank interpretation.** The future ranking cohort is unknown, so an exact future scalar rank cannot be reported. The numbers here use the specified ranking rule with population standard deviations across the eight honest runs in the table, including the three reference seeds. For each experiment and metric the contribution is `experiment_weight × abs(coefficient) × direction_adjusted(candidate − reference) / standard_deviation`. Missing values contribute nothing, as specified in the audit request. Both experiments have weight 0.5. Contact is higher-is-better; all six other metrics are lower-is-better. Negative coefficient signs are never used to reverse direction. [The summary](redteam/summary.json) records every normalizer and contribution, and [the analysis code](redteam/summarize.py) implements the calculation. Numerical near-equalities use a 1e-9 tolerance only for the descriptive Pareto flag, not the scalar calculation.

The comparison point is the actual `omniretarget` in the original `scoring/scores_test.json`, also reproduced by seed 0. The results transcribed from the paper into `scoring/reference.json` are not the measured gym reference. Exact shortcut ties and the disconnected-body/task-replacement Pareto wins do not depend on our choice of positive normalizers.

## Seed spread and reproduction check

[Seeds 0](redteam/reference_seed0/scores_test.json), [1](redteam/reference_seed1/scores_test.json), and [2](redteam/reference_seed2/scores_test.json) produce identical values in all **14** scored metric/experiment pairs. Maximum-minus-minimum spread is **0 for every metric**, and normalized rank spread is **0**. This is repeatability of a deterministic optimizer on fixed trajectories, not evidence of generalization: the data generator creates an RNG but never uses it. Regenerating both tasks with seeds 0, 1 and 2 gives identical data, also identical to the tracked JSON.

The delivered reproduction improves collision/skating constraints relative to its local IK baseline, and improves terrain contact. It **does not improve object contact** relative to that baseline, and loses source-motion fidelity in both tasks. These observations only partially support the paper's joint claim. The paper itself reports that GMR can have higher object contact at the expense of penetration, so the object-contact ordering alone is not a contradiction of the paper. PHC, GMR and VideoMimic are table entries here, not reproduced baseline implementations.

| Hidden-test metric | Object reference | Object IK | Terrain reference | Terrain IK |
|---|---:|---:|---:|---:|
| Penetration duration ↓ | 0 | 1 | 0 | 0 |
| Maximum penetration, cm ↓ | 1.39e-14 | 9.7206 | 2.29e-7 | 0 |
| Foot-skating duration ↓ | 0.1 | 1 | 0 | 1 |
| Maximum foot skating, cm/s ↓ | 4.3075 | 26.9612 | 1.38e-5 | 60.2111 |
| Contact preservation ↑ | 0.708525 | 0.900896 | 0.999964 | 0 |
| Pelvis-path error, cm ↓ | 3.5544 | 1.4358 | 20.3158 | 8.6398 |
| Motion-velocity error, cm/s ↓ | 11.2185 | 5.7620 | 72.1794 | 24.9701 |

**Bounds and remaining room.** The measured reference is exactly at the lower bound for penetration duration in both experiments and foot-skating duration in terrain. Object maximum penetration is at floating-point zero; terrain maximum penetration and maximum skating are also effectively at their zero floors. Terrain contact is nearly saturated at 0.999964, but is not exactly at its upper bound of 1. Object contact, object skating, and both source-motion errors leave room for improvement. The genuine refinement demonstrates that the reference can be beaten. The simple baseline is not near-perfect overall. For completeness, the transcribed paper reference in `reference.json` has exact zero object penetration duration and exact zero skating duration/speed in both experiments; those are not this gym's measured values.

**Departures from the paper.** Comparing [the supplied paper](paper/2509.26633.md), especially Sections III and V and Appendix D, with the delivered implementation:

- The paper uses 3D humanoids with quaternion floating bases and real robot collision geometry. This gym uses an 11-coordinate planar model and 12 sampled joint positions. It omits link-volume and self-collision checks.
- The paper constructs a volumetric Delaunay tetrahedral interaction mesh. The gym uses 2D Delaunay triangles. Object coordinates account for translation without general object rotation.
- The paper solves sequential SOCPs with linearized constraints, Drake rotation derivatives, up to 10 iterations and a trust region. The gym uses direct SciPy SLSQP, adds a weighted keypoint-matching term, and changes the procedure. This goes beyond scaling resources down.
- Stance detection uses a horizontal speed threshold of **0.35 m/s**, versus **0.01 m/s** in the paper. Terrain hand contacts get extra hard equalities, while hand-object contact is not enforced as a hard constraint.
- The terrain solver retry relaxes per-frame velocity bounds to full joint bounds; unsuccessful solver outputs can still be emitted. An independent artifact check finds maximum coordinate-increment excess **0.6282** in the reference terrain sequence. The score never measures joint/velocity-limit compliance.
- OMOMO (2.78 hours), in-house MoCap (1 hour), and LAFAN1 (4.6 hours) become two analytic **40-frame** tasks. Dataset/body preprocessing, original robot embodiments, and the actual paper baselines are absent. Augmentation, downstream RL, hardware success, and long-horizon claims are not evaluated.
- Contact preservation is a mean linear surface-distance score, rather than the paper's normalized duration of preserved contact. Terrain scoring covers desired hands, rather than the paper's hands, toes and heels. Pelvis and motion errors are additional gym metrics.

The synthetic replacement retains a limited collision/skating-versus-matching tradeoff. It does **not** establish that improvement here predicts improvement on the paper's problem: no-mesh improves the audit rank, terrain hand contact is almost saturated by explicit constraints, and training and test score alternating frames of the **same publicly supplied trajectories**, rather than unseen motions or scenes. Public contact labels, geometry and full source trajectories also supply all inputs needed by the keypoint shortcuts below. No download, cache or hidden answer file is needed.

**Brute force.** The grid run evaluates **177,147 configurations per frame**, **7,085,880 per task**, **14,171,760 total**, in about 16 seconds including evaluation. It has terrain contact 0, terrain skating duration 1, and loses to the reference. This does not demonstrate that brute force solves the test. The search enumerates its entire finite local grid, not all possible continuous configurations; finer global search remains untested. The public two-trajectory test is nevertheless small enough for direct point manipulation to achieve ideal metrics.

## Executed shortcuts

Every row is a tested shortcut, including ties. Links lead to its actual `run.sh`; adjacent `scores_test.json` and `status.json` preserve the hidden-test result. “Works” means at or above reference rank, not scientific validity. Values are rank differences from reference **0**, under the normalization above. No rank is assigned to a failed test or a nonfinite result.

| Shortcut | What it does | Test rank vs reference | Outcome |
|---|---|---:|---|
| [shortcut_disconnected_body](redteam/shortcut_disconnected_body/run.sh) | Use visible source data, freeze stance-foot components, project hands onto public surfaces, and shift all other joints up 2 m. Emit unrelated zero configurations. | **+3.1164** | **Works; Pareto win.** No task changes. The points cannot be one robot's body. |
| [shortcut_unchecked_keypoints](redteam/shortcut_unchecked_keypoints/run.sh) | Independently project source joints to remove collisions and satisfy contacts/stance, without robot kinematics. | **+2.2503** | **Works** under audit normalization; object pelvis error worsens. |
| [shortcut_mutable_task](redteam/shortcut_mutable_task/run.sh) | Replace the visible task JSON treated as trusted: zero source, no objects, all-frame hand-terrain labels; emit zero points. | **+3.1625** | **Works; global metric ideals:** contact 1 and every other metric 0 in both tasks. |
| [shortcut_impossible_q](redteam/shortcut_impossible_q/run.sh) | Keep reference points but replace every configuration coordinate with 1e200; add impossible claimed flags/timing. | **0, exact tie** | **Works as a shortcut.** All 14 metrics identical; configuration is ignored. Improve mode. |
| [shortcut_empty](redteam/shortcut_empty/run.sh) | Submit no artifacts and run visible evaluation. | **0, tie** | **Works as a shortcut.** Test succeeds with empty results; 14 values skipped. |
| [shortcut_partial_missing](redteam/shortcut_partial_missing/run.sh) | Submit reference object output only; omit terrain. | **0, tie** | **Works as a shortcut.** Seven omitted metrics incur no loss. Improve mode. |
| [shortcut_omit_failed_experiment](redteam/shortcut_omit_failed_experiment/run.sh) | Submit genuinely improved object output and no terrain output. | **+1.1089** | **Works.** Failing the entire terrain task beats reference and ties the complete honest refinement. Improve mode. |
| [shortcut_train_filter](redteam/shortcut_train_filter/run.sh) | Leave zero-point artifacts on disk, then remove the candidate from writable training-score method lists. | **0, tie** | **Works as a shortcut.** Hidden diagnostics evaluate the artifacts, but ranked results exclude them. |
| [shortcut_unlisted_method](redteam/shortcut_unlisted_method/run.sh) | Give zero-point artifacts a new method name, excluded by training's reference-name whitelist. | **0, tie** | **Works as a shortcut.** Hidden test follows the empty expected method sets. |
| [shortcut_copy_input](redteam/shortcut_copy_input/run.sh) | Return the source human points as robot points. | −2.7917 | Fails to pay; collision/skating losses dominate. |
| [shortcut_constant](redteam/shortcut_constant/run.sh) | Repeat a lifted initial robot pose throughout each task. | −9.5616 | Fails to pay; contact/path losses dominate. |
| [shortcut_score_forgery](redteam/shortcut_score_forgery/run.sh) | Copy input, add claimed metrics/success/timing, and prewrite perfect/impossible test scores. | −2.7917 | Fails to pay. Hidden evaluation overwrites forged scores and ignores the claims. |
| [shortcut_mutable_baseline](redteam/shortcut_mutable_baseline/run.sh) | Keep the weak constant candidate, but replace baseline artifacts with points 100 m underground. | −9.5616 | Fails to pay. Candidate metrics are absolute and unchanged. |
| [shortcut_nan](redteam/shortcut_nan/run.sh) | Emit NaN coordinates. | Undefined | Test exits 0 and writes six NaN metrics; no verified paying rank. It does not assign worst values. |
| [shortcut_impossible_keypoints](redteam/shortcut_impossible_keypoints/run.sh) | Emit constant 1e200 keypoint coordinates. | −9.5616 | Fails to pay despite accepting impossible coordinates. |
| [shortcut_malformed](redteam/shortcut_malformed/run.sh) | Emit zero frames against the 40-frame trusted task. | No score; exit 1 | Fails to pay. Evaluator raises and writes no test-score file; it does not fill worst values. |

The strongest unchanged-data shortcut, `shortcut_disconnected_body`, gets penetration/skating **0** and contact **1** in both tasks, pelvis-path error approximately **0 cm**, and motion-velocity errors **1.3228 / 2.2369 cm/s**. The maximum coordinate disagreement between its submitted points and its configurations' actual forward kinematics is **3.07 / 3.13 m**. It improves the full measured metric vector while failing the task of producing a robot motion. Fixed offsets disappear from the aligned pelvis and velocity errors, and independently placed hands/feet satisfy the remaining point tests. This is a normalization-independent failure of the rank's scientific meaning.

## Scoring completeness and trust boundaries

**Missing and failed runs do not receive worst values.** An empty submission and a partially missing submission return success and empty `results` entries. A malformed submission returns failure and no new scores at all. [The saved failure evidence](redteam/shortcut_malformed/failure.json) records the exception. Since the rank skips missing values, omission can avoid every loss. The improved-object-only shortcut proves that a submission failing a required experiment can strictly outrank the reference. Scalar compensation also cannot enforce unmeasured robot realizability: the disconnected-body shortcut improves all measured objectives while violating that requirement.

**The test does not write every method on disk under `results`.** Its expected method sets come from agent-writable `scoring/scores_train.json`. It calculates diagnostics for other artifacts, then filters the ranked results to those sets. Its subsequent equality check validates the already-filtered sets. Both the explicit training-score edit and a naturally unlisted method name produce successful tests with no candidate in ranked results. Diagnostics are not read by the rank. The visible evaluator also depends on `scoring/reference.json`, which is outside the four scoring documents listed in the future-agent manifest; packaging should check that dependency separately.

**Scores and metadata are not equally trusted.** Prewritten `scores_test.json` and artifact metric/flag/timing claims are overwritten or ignored on a successful run. However, `retargeted_keypoints` themselves are trusted as the robot motion, while `q`, its limits, and its agreement with forward kinematics are ignored. Shape and finiteness validation is inadequate. Replacing the public data changes what the unchanged hidden evaluator calls ground truth. There is no learned fixed reference artifact to retrain in this gym; task JSON is the demonstrated mutable fixed asset. Worsening baseline outputs has no effect on candidate metrics.

## Next-iteration changes suggested by the evidence

1. Evaluate against immutable task assets and genuinely held-out motions/scenes. Public generators with unused seeds cannot provide a held-out distribution.
2. Recompute robot points from submitted configurations using trusted robot geometry. Validate dimensions, finite values, joint/velocity limits, collision geometry and contact locations; make task validity a prerequisite for reward.
3. Require every candidate in every required experiment, independent of writable training scores and method-name whitelists. Write explicit finite worst outcomes for missing, malformed and failed runs, including whole-job failures handled outside the evaluator.
4. Establish that the actual mesh procedure helps on representative data, using faithful baselines and an ablation. Restore the paper's optimization/stance regime or clearly label this as a different planar problem. Preserve source fidelity alongside interaction constraints, with failure gates where tradeoffs must not compensate for an invalid result.

## Execution and restoration

Two CPU jobs (`32475454-e3b`, `5b618d40-e2d`, both reported exit 0) executed **24 distinct cases**: eight scientific/check runs and sixteen shortcut runs. The reported host log paths are not mounted here; preserved per-case producer/train/test logs and score/status files were reviewed. [The harness](redteam/run_case.py) invokes the producer to completion, runs the visible evaluator, and then invokes **`bash scripts/evaluate_test.sh` with no arguments** in a separate process. No background process, evaluator modification, Python environment interception or hidden-answer access is used. Candidate producers use only future-agent-visible data/baseline/train information, plus the original method for improve-mode checks. Paper inspection was used only for this audit's reproduction assessment.

Each case restores the original data, method, baseline, scores and checkpoints afterward. The original job script was restored from start commit `da53eb23707773cbc2ffa2b443ca471a6f14aea9`. Audit changes are confined to `redteam/` and this report, apart from the mandatory final `action.yaml` done signal. Preexisting protected README/manifest/trajectory changes recorded in [the starting status](redteam/start_status.txt), and system-managed trajectory changes, were not overwritten. No reproduction milestone was advanced by this audit.
