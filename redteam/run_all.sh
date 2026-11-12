#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
for candidate in reference_seed0 reference_seed1 reference_seed2 better_budget better_contact_refinement weaker_no_mesh weaker_keypoint_ik brute_force_grid shortcut_unchecked_keypoints shortcut_impossible_q shortcut_copy_input shortcut_constant shortcut_score_forgery shortcut_mutable_task shortcut_mutable_baseline shortcut_empty shortcut_partial_missing shortcut_train_filter shortcut_unlisted_method shortcut_nan shortcut_impossible_keypoints shortcut_malformed; do
  bash "redteam/$candidate/run.sh"
done
python redteam/summarize.py
