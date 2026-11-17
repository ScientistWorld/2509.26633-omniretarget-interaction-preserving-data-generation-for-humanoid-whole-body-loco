#!/bin/bash
set -e
cd /home/user
python baseline/keypoint_ik.py --inputs "$1" --outputs "$2"
