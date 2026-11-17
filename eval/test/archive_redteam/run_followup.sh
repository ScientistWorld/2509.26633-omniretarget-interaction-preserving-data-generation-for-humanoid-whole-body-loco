#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash redteam/shortcut_disconnected_body/run.sh
bash redteam/shortcut_omit_failed_experiment/run.sh
python redteam/summarize.py
