#!/bin/bash
# Corrective combined audit: author-release initialization, numerical geometry check,
# reference/baselines, honest headroom variants, paired repeat and shortcut replay.
# Finalization restores this entrypoint to reference reproduction.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m eval.test.audit
