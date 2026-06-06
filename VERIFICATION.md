# Verification Report

## Initial Assessment

The workspace contained a working scaled reproduction and had previously claimed `core_claim_plus`, but a closer audit found that the implementation and benchmark are substantially scaled down: SciPy SLSQP on a planar humanoid and synthetic tasks rather than the paper's Drake sequential SOCP, Unitree assets, and original datasets. The code does implement the central interaction-preserving constrained retargeting computation, but the evidence does not justify `core_claim_plus`.

## Issues Found

- **Critical:** Method-agnostic documentation leaked solution-specific framing in `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, and `scoring/DIRECTION.md`.
- **Critical:** Official scores omitted the local baseline, leaving the gym without an official non-trivial comparator even though diagnostics showed one.
- **Minor:** The official evaluator registered only methods listed in `reference.json`; current reproducibility is acceptable after registering the local baseline, but future agents should use registered method keys for strict validation.
- **Minor:** The git tree still reports unrelated system-file noise: `README.md` and `manifest.json` are character devices, and tracked `trajectory/turn_*.jsonl` files appear deleted. These are outside the reproduction files and were not changed.

## Fixes Applied

- Rewrote method-agnostic documentation to describe only the task, metrics, constraints, and allowed research direction without naming the paper's specific construction.
- Registered `keypoint_ik` as a substitute local baseline in `scoring/reference.json` without inventing paper-reported metric values.
- Restored `keypoint_ik` metrics in `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json` from existing diagnostic outputs.
- Re-ran `python validate.py --compare`; all scoring and structure checks pass.

## Final Judgment

The workspace is ready for partial gym use. It is meaningful for improving a scaled kinematic retargeting benchmark with object and terrain interaction, and it has train/test slices plus a non-trivial local baseline. It is not a full paper reproduction: original datasets, robot models, sequential SOCP implementation, augmentation experiments, and downstream RL are absent.

- Milestone: core_claim
- Ready for gym use: partial
- Confidence: medium
- Key limitation: scaled synthetic planar kinematics only
