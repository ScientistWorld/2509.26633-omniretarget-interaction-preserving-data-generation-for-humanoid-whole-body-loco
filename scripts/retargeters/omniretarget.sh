#!/bin/bash
# Registered method: OmniRetarget (paper method, scaled planar reproduction).
# Contract: bash scripts/retargeters/<name>.sh <input_dir> <output_dir>
set -e
cd /home/user
python method/omniretarget.py --inputs "$1" --outputs "$2"
