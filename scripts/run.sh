#!/bin/bash
# Job entry point: reproduce the gym end to end inside the container (no internet needed; the
# public assets in data/downloads/ come from scripts/download.sh).
#   1. rebuild the public train instances,
#   2. official run: the shipped retargeters alone (OmniRetarget reference, keypoint-IK baseline)
#      through scripts/evaluate_train.sh and scripts/evaluate_test.sh -> scoring/scores_*.json,
#   3. iteration-13 hardening audit: honest variants and shortcut submissions through the same
#      unchanged evaluators (resumable chunks; redteam/hardening3d/results/).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python --version
python -c "import numpy, scipy, mujoco, clarabel; print(numpy.__version__, scipy.__version__, mujoco.__version__, clarabel.__version__)"
rm -f scripts/retargeters/var_*.sh scripts/retargeters/sc_*.sh
if [ ! -f redteam/hardening3d/results/official/exit_codes.json ]; then
  python -m data.tasks --out data/downloads/tasks/train
fi
python redteam/hardening3d/harness.py all
for s in test train; do
  python redteam/hardening3d/summarize.py redteam/hardening3d/results/combined_$s $s || true
done
python validate.py || true
