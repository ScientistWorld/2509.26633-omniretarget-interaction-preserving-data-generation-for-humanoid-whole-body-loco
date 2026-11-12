#!/bin/bash
# Hidden test evaluator (no arguments needed).
# Generates the private test instances, runs every method registered as
# scripts/retargeters/<name>.sh on them, recomputes all metrics from the
# returned joint trajectories and writes scoring/scores_test.json.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p scoring
python eval/test/evaluate_test.py
