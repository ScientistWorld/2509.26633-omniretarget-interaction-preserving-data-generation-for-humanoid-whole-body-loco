#!/bin/bash
# Held-out test-slice evaluator. Reads method-produced artifacts and writes
# scoring/scores_test.json. This script does not train or run method code.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

OUTPUT_DIR="${1:-checkpoints/retargeting}"
python eval/test/evaluate_retargeting_slice.py \
  --outputs "$OUTPUT_DIR" \
  --score-out scoring/scores_test.json \
  --diagnostics-out scoring/diagnostics_test.json \
  --train-scores scoring/scores_train.json
