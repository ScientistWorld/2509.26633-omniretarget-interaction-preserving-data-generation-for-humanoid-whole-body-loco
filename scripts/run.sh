#!/bin/bash
# Reproduce the gym end to end inside the container (no internet needed):
#   1. regenerate the public train instances;
#   2. score the shipped methods (OmniRetarget reference, keypoint-IK baseline) on
#      train and on the hidden test split, with nothing else running beside them;
#   3. hardening audit: honest variants and shortcut submissions through the
#      unchanged hidden test (and train), CPU-budget / hang-guard checks;
#   4. summaries and validation.
# Every audit stage is resumable (done markers under redteam/hardening/results/),
# because compute nodes may preempt and restart the job.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python --version
rm -f scripts/retargeters/var_*.sh scripts/retargeters/sc_*.sh
rm -rf data/tasks/train
python data/task_generator.py --out data/tasks/train > /dev/null
python redteam/hardening/harness.py official
cp scoring/scores_train.json scoring/scores.json
cp scoring/diagnostics_train.json scoring/diagnostics.json
python redteam/hardening/harness.py combined
python redteam/hardening/harness.py budget
python redteam/hardening/summarize.py redteam/hardening/results/combined_test test
python redteam/hardening/summarize.py redteam/hardening/results/combined_train train
python validate.py
