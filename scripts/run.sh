#!/bin/bash
# Iteration-12 hardening, step 1: design exploration (candidate setups x methods on a dev split).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python --version
python -c "import numpy, scipy, os; print(numpy.__version__, scipy.__version__, len(os.sched_getaffinity(0)), 'cpus')"
python redteam/hardening/explore.py run redteam/hardening/results/explore
