#!/bin/bash
# Reproduce the gym end to end: regenerate the public train instances, score
# every registered method (OmniRetarget and the keypoint-IK baseline) on train
# and, when the hidden evaluator is present, on the hidden test split.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -rf data/tasks/train
python data/task_generator.py --out data/tasks/train
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
cp scoring/diagnostics_train.json scoring/diagnostics.json
if [ -f eval/test/evaluate_test.py ]; then
  bash scripts/evaluate_test.sh
fi
python validate.py
