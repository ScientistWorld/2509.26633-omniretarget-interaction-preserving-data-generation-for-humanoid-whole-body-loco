Checked the existing train/test split, evaluator scripts, scoring schema, and static leakage patterns. The scripts already pointed to the correct slice folders and wrote `scoring/scores_train.json` / `scoring/scores_test.json`; no train/test score stubs existed.

Changed the split metrics to better distinguish real retargeting improvement from metric-gaming:

- Added `pelvis_path_error_cm` and `motion_velocity_error_cm_s` to both train and test evaluators. These penalize methods that satisfy contacts or avoid penetration by distorting the source motion, freezing the body, or moving the trajectory away from the demonstrated path.
- Added matching metric entries and descriptions in `scoring/reference.json`. Existing collision, skating, and contact metrics now also have required descriptions.
- Fixed a frame-split leakage pattern in transition metrics. Foot-skating and velocity scoring now use only within-slice frame pairs instead of scoring an even train frame against the previous odd test frame, or vice versa.
- Removed generated Python cache directories from the workspace.

Runtime concerns for the next runner iteration:

- I did not run evaluators or compute scores in this design-only iteration. The runner should confirm the new fidelity metrics do not dominate the constraint/contact metrics and do not saturate at identical values across train and test.
- The split is still parity-based over two synthetic tasks, so it is lightweight and repeatable but not a strong distribution shift. The added motion-fidelity metrics improve shortcut resistance, but future work would be stronger if additional tracked held-out task variants were available.
- The held-out test evaluator optionally reads `scoring/scores_train.json` only to require matching method keys when train scores already exist. If final testing runs without train scores present, it will score every artifact in the output directory.

Static checks performed:

- `bash -n scripts/evaluate_train.sh`
- `bash -n scripts/evaluate_test.sh`
- Grepped for train/test cross-imports and forbidden paper-agreement metric names.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
