#!/bin/bash
# Run the keypoint-IK baseline (GMR/PHC objective family) on the public train
# instances and keep the joint trajectories under checkpoints/retargeting/train/keypoint_ik/.
# Usage: bash scripts/baseline.sh
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
mkdir -p checkpoints/retargeting/train
bash scripts/retargeters/keypoint_ik.sh data/tasks/train/inputs checkpoints/retargeting/train/keypoint_ik
