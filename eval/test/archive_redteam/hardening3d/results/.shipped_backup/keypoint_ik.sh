#!/bin/bash
# Registered method: keypoint-matching IK baseline (GMR/PHC family; joint limits only).
# Contract: bash scripts/retargeters/<name>.sh <input_dir> <output_dir>
set -e
cd /home/user
python baseline/keypoint_ik.py --inputs "$1" --outputs "$2"
