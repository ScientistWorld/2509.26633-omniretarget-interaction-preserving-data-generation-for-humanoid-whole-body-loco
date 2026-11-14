#!/bin/bash
# Reproduce the gym end to end inside the container (no internet needed):
# regenerate the public train instances, score every registered method (the
# OmniRetarget reference and the keypoint-IK baseline) on train and on the
# hidden test split, and validate the workspace.
#
# The iteration-12 hardening audit (honest variants and shortcut submissions
# through the hidden test, CPU-budget checks) is reproduced with
#   python redteam/hardening/harness.py all        # resumable; delete redteam/hardening/results/{official,chunks} to redo
#   python redteam/hardening/summarize.py redteam/hardening/results/combined_test test
#   python redteam/hardening/summarize.py redteam/hardening/results/combined_train train
# and the design exploration with python redteam/hardening/explore.py run <out_dir>.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python --version
rm -f scripts/retargeters/var_*.sh scripts/retargeters/sc_*.sh
bash scripts/reproduce.sh
