> **Superseded:** this report describes the pre-hardening gym (iterations 1-9). The benchmark, evaluator and reference implementation were rebuilt in iteration 11; see `HARDENING.md` for the current state.

# Verification Report

## Initial Assessment

The workspace started this final check at `core_claim`. The claimed milestone is justified for a scaled kinematic gym: the reproduced method runs end to end, writes reusable artifacts, and the evaluator measures interaction-retargeting quality from those artifacts rather than rerunning the method. The implementation is not an original-scale reproduction of the paper's full Unitree/Drake/RL stack, so the previous downgrade from `core_claim_plus` to `core_claim` was appropriate.

## Issues Found

- Minor: Generated job logs and raw trajectory transcripts were tracked in git. These are system artifacts and do not belong in the reusable gym.
- Minor: A late object-contact constraint edit was present after the accepted scores were produced. Audit jobs showed it did not improve the object contact metric and worsened skating, so it should not be part of the final method.
- Minor: The root has pre-existing system-file noise in `README.md` and `manifest.json` that appears as special-file modifications in git status. I did not alter these unrelated system files.
- Limitation: The object-interaction benchmark still has weaker normalized contact preservation than the diagnostic keypoint baseline, although the reproduced method satisfies the hard penetration and skating constraints. This supports `core_claim`, but not a higher milestone.

## Fixes Applied

- Removed tracked generated `jobs/` logs and `trajectory/` raw transcripts, and added `jobs/` to `.gitignore`.
- Removed the unverified object-contact equality constraint attempt from `method/omniretarget.py`, restoring consistency with the validated packaged scores.
- Updated `PROGRESS.md` and `MILESTONES.md` to document the rejected audit attempt and the conservative final milestone.
- Re-ran `python validate.py --compare`; validation passes, including scoring schema checks, import separation, train/test wiring checks, and method-output-based evaluation checks.

## Final Judgment

The workspace is ready as a partial research gym for the scaled kinematic interaction-retargeting problem. Improve-mode agents can build on the packaged method and scores, while from-scratch agents can use the method-agnostic problem, data, evaluation, and scoring documents without depending on `method/`.

- Milestone: core_claim
- Ready for gym use: partial
- Confidence: medium
- Key limitation: scaled synthetic planar kinematics only; the original robot assets, Drake sequential SOCP solver stack, augmentation regime, and downstream RL claims are not reproduced.
