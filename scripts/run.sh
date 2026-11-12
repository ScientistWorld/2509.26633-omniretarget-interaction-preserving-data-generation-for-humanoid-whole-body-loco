#!/bin/bash
# Hardening job: regenerate the public train instances, score the shipped
# methods on train and on the hidden test (official numbers), then rerun the
# hidden test with honest variants and shortcut submissions registered
# alongside, and summarise rank differences with paired bootstrap intervals.
set -x
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python --version
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

rm -rf data/tasks/train
python data/task_generator.py --out data/tasks/train || exit 1

python redteam/hardening/harness.py official
python redteam/hardening/summarize.py redteam/hardening/results/official || true
cp scoring/scores_train.json scoring/scores.json
cp scoring/diagnostics_train.json scoring/diagnostics.json

python redteam/hardening/harness.py combined
python redteam/hardening/summarize.py redteam/hardening/results/combined_test || true

python validate.py || true
