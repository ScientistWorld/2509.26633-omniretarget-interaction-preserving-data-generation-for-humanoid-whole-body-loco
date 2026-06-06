#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

mkdir -p data/synthetic_tasks checkpoints/retargeting
python data/synthetic_locomotion.py --out data/synthetic_tasks --frames "${FRAMES:-80}"

python baseline/keypoint_ik.py --task data/synthetic_tasks/object_carry.json --out checkpoints/retargeting/keypoint_ik_object_carry.json
python baseline/keypoint_ik.py --task data/synthetic_tasks/terrain_step.json --out checkpoints/retargeting/keypoint_ik_terrain_step.json

bash scripts/evaluate.sh checkpoints/retargeting
