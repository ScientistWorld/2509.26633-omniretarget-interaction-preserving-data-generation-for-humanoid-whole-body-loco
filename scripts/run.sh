#!/bin/bash
# Extend held-out real motions, then run all scientific and shortcut comparisons
# in one sequential job. No optimizer or scoring definition changes in this run.
# Finalization restores this entrypoint to reference reproduction.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python eval/test/build_test_data.py --if-needed
python -m eval.test.audit
