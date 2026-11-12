#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash redteam/run_followup.sh
