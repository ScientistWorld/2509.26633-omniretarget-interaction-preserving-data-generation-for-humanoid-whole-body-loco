# Evaluation

Run `bash scripts/evaluate_train.sh` or `bash scripts/evaluate_test.sh` with no arguments. Each scores every `scripts/retargeters/*.sh` executable on disk, independently on the same inputs, and writes each method under each experiment's `results` in `scoring/scores_train.json` or `scoring/scores_test.json`. `scripts/evaluate.sh` is the public train alias.

The test uses held-out real demonstrations. Its corpus, instance construction, robot assets, ground truth, detailed diagnostics and audit raw outputs stay under `eval/test/`. A kernel Landlock boundary prevents the method and descendants from reading that directory or git history, except its own fresh invocation input/output directory. Methods may read public code/data and system libraries, and write their invocation directory and `/tmp`. Ground-truth annotations are built after the method returns. There is no training in the hidden test.

All numbers are recomputed from `q` using evaluator-owned MuJoCo kinematics, collision geometry, source annotations and target scene. Extra keypoints, metadata, verdicts, claimed timing and scores are ignored. Thresholds and test size are fixed by the evaluator.

| Metric | Measurement | Better |
|---|---|---|
| `penetration_duration` | Mean window fraction of frames with any robot-scene penetration deeper than 1 cm | lower |
| `max_penetration_cm` | Mean window maximum penetration, capped at 30 cm | lower |
| `foot_skating_duration` | Fraction of demonstrated stance-foot events with robot toe horizontal speed above 1 cm/s | lower |
| `max_foot_skating_cm_s` | Mean window maximum horizontal speed on stance events, including speeds below 1 cm/s; cap 300 | lower |
| `contact_preservation` | Interaction contact events kept within 3 cm absolute signed surface distance: hands for manipulated objects; hands and feet for terrain | higher |
| `spurious_contact_duration` | Clear-hand and swing-foot events in which the robot touches a surface anyway (distance at most 1 cm, including penetration) | lower |
| `support_failure_duration` | Stance-foot events more than 3 cm absolute distance from the floor or supporting terrain | lower |
| `pelvis_path_error_cm` | Mean horizontal pelvis-position error against the scaled demonstration; cap 100 cm | lower |
| `pose_error_cm` | Mean body-relative keypoint error against the scaled demonstration; cap 100 cm | lower |

A source foot is in stance when its scaled horizontal speed is at most 1 cm/s. Source object-hand contact uses wrist distance at most 40 cm from the box origin while lifted by 3 cm or moving more than 3 mm/frame. Terrain contacts use hand/toe/ankle surface thresholds of 5/6/11 cm. Clear hands are more than 50 cm from the box origin or 15 cm from terrain; clear swing toes are more than 6 cm above their supporting surfaces. These are geometric annotations, not force measurements. Contact, support and skating ratios pool their eligible events; penetration and motion errors average windows equally.

A crash, missing/malformed/nonfinite trajectory, nonunit quaternion, impossible base position, joint position or velocity violation, base translation above 20 cm/frame, base angular speed above 12 rad/s, or exceeded CPU budget fails the entire method. It receives every metric's worst value in every experiment: contact 0, duration 1, and the stated error caps. No entries are omitted. Numerical validation tolerance is 1e-4 for joint limits and 1e-3 for quaternion norm.

The budget is 30 CPU seconds per instance, summed over the method's process tree: 1320 CPU seconds for train and 9750 for test. One BLAS/OpenMP thread is supplied. No wall-clock timeout is used anywhere on the method path. A hidden test runs alone in a fresh container.

The three experiments have weights 0.35 (nominal objects), 0.35 (nominal terrain), and 0.30 (modified scenes). `scoring/reference.json` defines metric directions and coefficient magnitudes. Ranking adds weighted signed improvements over the stored reference, each divided by that metric's population standard deviation over the attempts being ranked; zero-variance metrics contribute zero. Coefficient signs do not reverse directions. Failed runs are coordinatewise worst and cannot outrank a valid solving run through another metric. Historical paper numbers are underscore metadata, not scaled-subset targets. Diagnostics and audit ranks are not ranked results.
