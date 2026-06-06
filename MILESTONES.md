# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-06 00:00] - none
- Read the paper and identified OmniRetarget's core as interaction-mesh Laplacian retargeting with hard kinematic constraints.
- Implemented the initial scaled benchmark, proposed method, baseline, evaluator, scripts, and scoring reference.
- Prepared a compute-container job to run the method end-to-end; no milestone above `none` is claimed until the job finishes successfully.

### [2026-06-06 03:15] - none
- Reviewed failed job `bd8cb121-d23`.
- Diagnosed a packaging/import failure before algorithm execution: `ModuleNotFoundError: No module named 'data'`.
- Added package initializers and script-level `PYTHONPATH` export before resubmission.

### [2026-06-06 03:35] - method_runs
- Job `d59dfd81-fe3` executed the full reproduction script successfully.
- Generated synthetic object and terrain tasks, ran the keypoint IK baseline, ran the OmniRetarget interaction-mesh optimizer, evaluated artifacts, and passed `validate.py --compare`.
- Current results show the algorithm runs end-to-end, but object penetration duration and terrain contact preservation are not yet sufficient for `core_claim`.

### [2026-06-06 03:50] - method_runs
- Job `e0708e5b-dfc` fixed object penetration and preserved object contact, but foot-skating metrics still failed due to a stance-threshold mismatch between method and evaluator.
- Aligned the optimizer's stance detector with the evaluation threshold before resubmission.

### [2026-06-06 04:05] - core_claim
- Job `12774676-633` reproduced the core kinematic claim on object interaction.
- Compared with keypoint IK, OmniRetarget eliminated penetration and foot skating while preserving hand-object contact.
- Terrain setting remains below claim because the synthetic generator labeled hand contacts above the actual terrain surface; fixed generator and resubmitted to target `core_claim_plus`.

## Stop Justification
- Not stopped.
