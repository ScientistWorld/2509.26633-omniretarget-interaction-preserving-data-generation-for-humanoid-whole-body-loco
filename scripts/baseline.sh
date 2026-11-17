#!/bin/bash
# Run the keypoint-IK baseline (GMR/PHC objective family) on the public train instances and keep
# the robot trajectories under checkpoints/retargeting/train/keypoint_ik/.
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p checkpoints/retargeting/train
bash scripts/retargeters/keypoint_ik.sh data/downloads/tasks/train/inputs checkpoints/retargeting/train/keypoint_ik
