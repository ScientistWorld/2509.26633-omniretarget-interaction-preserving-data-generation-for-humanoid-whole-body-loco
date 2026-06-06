#!/bin/bash
# Visible train-slice evaluator. Reads method-produced artifacts and writes
# scoring/scores_train.json. This script does not train or run method code.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

OUTPUT_DIR="${1:-checkpoints/retargeting}"
python eval/train/evaluate_retargeting_slice.py \
  --outputs "$OUTPUT_DIR" \
  --score-out scoring/scores_train.json \
  --diagnostics-out scoring/diagnostics_train.json
