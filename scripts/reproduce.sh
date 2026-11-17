#!/bin/bash
# Reproduce the gym end to end (no internet needed once scripts/download.sh has run):
# rebuild the public train instances, score every registered method (the OmniRetarget reference
# and the keypoint-IK baseline) on train and on the hidden test split, and validate the workspace.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m data.tasks --out data/downloads/tasks/train
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
cp scoring/diagnostics_train.json scoring/diagnostics.json
if [ -f eval/test/evaluate_test.py ]; then
  bash scripts/evaluate_test.sh
fi
python validate.py || true
