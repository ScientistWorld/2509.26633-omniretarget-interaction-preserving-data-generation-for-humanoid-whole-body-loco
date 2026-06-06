# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 07:29] - core_claim_plus
- Packaged reproduction validates both object-interaction and terrain-interaction kinematic experiments.
- OmniRetarget implementation preserves contacts while satisfying penetration and foot-skating constraints on the reusable scaled benchmark.
- `python validate.py --compare` passes all structure and scoring checks.

### [2026-06-06 07:35] - core_claim
- Continuing audit found the evaluator skipped the generated keypoint IK baseline artifacts.
- Fixed the evaluator to include every produced method artifact in `scoring/scores.json`.
- Submitting a CPU container test to regenerate scores and verify the packaged reproduction with baseline results included.

## Stop Justification
- Re-audit in progress; previous `core_claim_plus` claim is being revalidated after evaluator fix.
