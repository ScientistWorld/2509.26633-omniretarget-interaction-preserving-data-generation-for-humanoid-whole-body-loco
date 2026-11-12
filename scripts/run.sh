#!/bin/bash
# Reproduce the gym end to end inside the container (no internet needed):
# regenerate the public train instances, score every registered method
# (OmniRetarget reference and keypoint-IK baseline) on train and on the hidden
# test split, and validate the workspace.
#
# The hardening audit (variants + shortcut submissions through the hidden test)
# is reproduced with:
#   python redteam/hardening/harness.py official
#   python redteam/hardening/harness.py combined
#   python redteam/hardening/summarize.py redteam/hardening/results/combined_test
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python --version
bash scripts/reproduce.sh
