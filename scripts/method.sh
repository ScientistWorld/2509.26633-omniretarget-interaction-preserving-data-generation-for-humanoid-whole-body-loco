#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p data/synthetic_tasks checkpoints/retargeting
python data/synthetic_locomotion.py --out data/synthetic_tasks --frames "${FRAMES:-80}"

python method/omniretarget.py --task data/synthetic_tasks/object_carry.json --out checkpoints/retargeting/omniretarget_object_carry.json
python method/omniretarget.py --task data/synthetic_tasks/terrain_step.json --out checkpoints/retargeting/omniretarget_terrain_step.json

bash scripts/evaluate.sh checkpoints/retargeting
