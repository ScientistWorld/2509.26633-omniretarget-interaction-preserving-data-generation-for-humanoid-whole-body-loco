# Verification

Current verification entrypoint is `scripts/run.sh`: one sequential final-code job reruns the reference and baseline end to end, then honest alternatives and fixed shortcuts through the same no-argument split evaluators. Test audit evidence stays under `eval/test/audit_results/`; the official score files contain only the shipped reference and keypoint baseline. `HARDENING.md` records the final evidence after that job.

Before submission: Python source compilation, shell parsing, independent train/test import inspection and a stdlib-only kernel filesystem isolation check. No training, inference, scientific data generation or metric computation was run on the login node. Full `validate.py` runs in the compute container after the audit.

The public task uses actual 3D motion and G1 kinematics. Scope is a subset of the paper's kinematic claim, with reduced collision hulls and Clarabel/MuJoCo replacing Drake. No downstream policy, dynamics or hardware reproduction is claimed.
