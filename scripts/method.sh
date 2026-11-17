#!/bin/bash
# Run the paper's method (OmniRetarget) on the public train instances and keep the robot
# trajectories under checkpoints/retargeting/train/omniretarget/.
# Scoring is done by scripts/evaluate*.sh, which run every registered method.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p checkpoints/retargeting/train
bash scripts/retargeters/omniretarget.sh data/downloads/tasks/train/inputs checkpoints/retargeting/train/omniretarget
