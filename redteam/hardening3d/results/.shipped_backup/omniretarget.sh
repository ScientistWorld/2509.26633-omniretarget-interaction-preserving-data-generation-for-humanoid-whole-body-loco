#!/bin/bash
# Registered method: OmniRetarget (paper method; interaction mesh + hard constraints, sequential SOCP).
# Contract: bash scripts/retargeters/<name>.sh <input_dir> <output_dir>
set -e
cd /home/user
python method/omniretarget.py --inputs "$1" --outputs "$2"
