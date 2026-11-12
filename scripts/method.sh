#!/bin/bash
# Run the paper's method (OmniRetarget) on the public train instances and keep
# the joint trajectories under checkpoints/retargeting/train/omniretarget/.
# Scoring is done by scripts/evaluate*.sh, which run every registered method.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p checkpoints/retargeting/train
bash scripts/retargeters/omniretarget.sh data/tasks/train/inputs checkpoints/retargeting/train/omniretarget
