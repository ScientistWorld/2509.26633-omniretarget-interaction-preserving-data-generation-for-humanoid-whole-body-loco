#!/bin/bash
# Visible train evaluator (no arguments needed).
# Runs every method registered as scripts/retargeters/<name>.sh on the public
# train instances in data/tasks/train, recomputes all metrics from the returned
# joint trajectories and writes scoring/scores_train.json.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
python eval/train/evaluate_train.py
