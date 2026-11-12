# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 07:39] - core_claim_plus
- Implemented a scaled OmniRetarget interaction-mesh constrained optimizer and keypoint IK diagnostic baseline.
- Reproduced the kinematic core claim on both object and terrain interaction tasks in compute jobs.
- Tightened the reusable evaluator to load trusted benchmark task files by task ID rather than trusting artifact-provided paths.

### [2026-06-06 07:47] - core_claim_plus
- Compute job `f03c43a4-801` passed `scripts/reproduce.sh` and `python validate.py --compare` in the container.
- Final scoring keeps paper-reference `omniretarget` results in `scoring/scores.json` and local substitute-task diagnostics in `scoring/diagnostics.json`.
- No packaging issue remains for the reusable kinematic reproduction gym.

### [2026-06-06 08:18] - core_claim
- Independent audit confirmed that the implementation captures the OmniRetarget interaction-mesh constrained retargeting core, but the SciPy SLSQP solver and planar synthetic benchmark are significant scale deviations from the paper's Drake sequential SOCP and Unitree datasets.
- Final milestone is conservatively set to `core_claim`: the terrain interaction benchmark preserves contact while enforcing no penetration and no skating; object interaction preserves hard constraints but has lower normalized contact score under the stricter metric.

### [2026-06-06 08:25] - core_claim
- Final verification cleaned method-agnostic documentation and restored the local keypoint IK baseline to official scores as a substitute comparator.
- `python validate.py --compare` passes after the verification fixes.
- Workspace is ready as a partial gym for the scaled kinematic retargeting problem; higher milestones require original-scale assets, solver fidelity, augmentation, or downstream RL.

### [2026-06-06 08:26] - core_claim
- Added a method-side hard object-contact constraint for labeled hand-object contacts to improve the object interaction benefit metric.
- Submitted a CPU audit job to regenerate packaged reproduction, train-slice scores, and test-slice scores before considering any milestone change.

### [2026-06-06 08:35] - core_claim
- Final verification rejected the object-contact constraint attempt because audit jobs showed no object benefit and worse foot-skating metrics.
- Removed the attempted constraint and kept the validated scaled interaction-mesh constrained optimizer.
- Milestone remains `core_claim`.

### [2026-11-12 04:40] - core_claim
- Hardening (iteration 11): rebuilt the benchmark around a hidden, generated 120-instance test, a q-only evaluator with a validity gate (worst values on failure) and an every-registered-method interface; restored the paper's per-frame sequential convex solver (trust region 0.2, <=10 iterations), 1 cm/s stance detection and direct scaling; added the scene-augmentation experiment (Sec. III-B).
- Hidden test: OmniRetarget has ~0 penetration and skating vs 0.36-0.43 / 0.41-0.56 for keypoint IK, and higher contact in all three experiments (0.83/0.86/0.71 vs 0.71/0.86/0.54); rank delta of the baseline -7.4. A contact-aware variant beats the reference by +1.47 (paired 95% CI [+1.07, +1.90]); all red-team shortcuts now fail or score as their real trajectory. Details in HARDENING.md.
- Milestone kept at `core_claim`: scaled planar substitute, no downstream RL; the mesh-vs-constrained-keypoint ablation is a tie.
