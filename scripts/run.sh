#!/bin/bash
# Iteration-12 hardening, step 1: design exploration (candidate setups x methods on a dev split).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python --version
python -c "import numpy, scipy, os; print(numpy.__version__, scipy.__version__, len(os.sched_getaffinity(0)), 'cpus in mask')"
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
EXPLORE_WORKERS=${EXPLORE_WORKERS:-1} python redteam/hardening/explore.py run redteam/hardening/results/explore
