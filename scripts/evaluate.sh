#!/bin/bash
# Standard evaluation entry point.
#
# Scores every method registered in scripts/retargeters/ on the public train
# instances (same code path as scripts/evaluate_train.sh) and writes
# scoring/scores.json (plus scoring/diagnostics.json with per-instance values
# and failure reasons). Uses only eval/ and data/; never imports method/.
#
# To evaluate a new method, add scripts/retargeters/<name>.sh that takes
# <input_dir> <output_dir> and writes <output_dir>/<instance_id>.json = {"q": ...}.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
python eval/train/evaluate_train.py --score-out scoring/scores.json --diagnostics-out scoring/diagnostics.json
