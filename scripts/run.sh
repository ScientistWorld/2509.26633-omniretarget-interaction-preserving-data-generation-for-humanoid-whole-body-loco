#!/bin/bash
# Resume the combined audit after infrastructure preemption. Scientific sources
# and fixed assets are checked before reusing completed reference/variant runs.
# Finish the paired repeat, shortcut replay and recording-clustered analysis.
# Finalization restores this entrypoint to reference reproduction.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m eval.test.audit
