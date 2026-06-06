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
