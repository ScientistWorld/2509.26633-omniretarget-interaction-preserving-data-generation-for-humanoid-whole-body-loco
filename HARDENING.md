# Final hardening

The final combined verification is queued; the numerical tables below will be completed from that job. No new result is claimed before it finishes.

The paper was read in `paper/2509.26633.md`, especially Sections III/V and Appendices A–D. REVIEW.md correctly identified that a planar procedural task did not establish the paper's 3D claim. This iteration completes the interrupted move to real OMOMO large-box manipulation, public climbing motion capture, the Unitree G1 with floating base and 29 joints, Delaunay volumetric interaction meshes, and sequential conic optimization with the paper's hard constraints. Test uses held-out subjects and recordings; modified boxes and platform heights provide further difficulty.

The incoming stored 3D test reference was failed: an augmented trajectory jumped its base by 55 cm/frame. Its score was the worst vector in every experiment. A previous fix had added floating-base velocity bounds but had never been rerun over the official split. The new audit reruns the final method and every comparison from source. Historical planar rank differences are not numerically comparable with the 3D task.

Changes decided together from REDTEAM.md and REVIEW.md:

- Configurations alone determine scores. Evaluation-owned robot, scenes, thresholds, annotations and split size determine all comparisons. Detached keypoints and claimed metrics are ignored.
- Every failure emits all nine worst metrics for all three experiments. Wrong shape, missing/partial output, malformed JSON, NaN, impossible base/quaternion, joint/velocity violations and teleporting cannot tie the reference through an empty entry.
- Inputs are fresh per method and trusted in-memory copies drive measurement. Mutating temporary inputs, train-score membership or candidate score files cannot change the trusted comparison.
- Test data, construction, diagnostics, old audits and raw trajectories are confined to `eval/test/`. Linux Landlock denies method/descendant reads of that directory and git history, except the invocation's own inputs/outputs. Ground truth is built after the method returns. Kernel denial was checked with a stdlib-only subprocess before submitting model computation.
- Stance and skating use 1 cm/s rather than 1 cm/frame. Maximum stance speed includes subthreshold movement. Object contact covers hands only. Deeply penetrating limbs cannot count as preserved contact. A separate support-loss metric closes floating-foot wins; swing-foot clearance is also measured.
- Ranking metadata uses the actual pose metric, has directions and positive coefficient magnitudes, and includes every scored constraint. Weights remain 0.35/0.35/0.30; support failure adds coefficient 0.8 because stable support is part of the paper's claim. The paper's reported numbers are preserved unchanged under underscore keys because the subset and collision hulls differ.
- Removed the soft slack fallback: infeasible linearized subproblems now reject the SQP step instead of replacing hard constraints with penalties. Foot equality tolerance is 0.1 mm. Joint/velocity inequalities no longer expand to accommodate an out-of-bounds iterate. The original 10-step sequential procedure, mesh, smoothing and augmentation pipeline remain intact.
- Fixed deterministic sampling, fixed source inputs, sorted collision pairs, sequential isolated runs and CPU-time budgets remove seed/load dependence. No wall-clock watchdog remains in the method path.

The audit runs reference, independent keypoint IK, no-mesh ablation, three contact-cost variants, keypoint plus contact, longer private simulator search, an exact repeat, a constant pose and the fixed shortcut cases. Paired intervals resample complete recordings, sharing each draw across methods and nominal/augmented versions. All raw measurements are private and only official reference/baseline results are restored to scoring files.

Historical before values: original REDTEAM reference 0, contact improvement +1.1089, no-mesh +0.3468, keypoint IK -3.5768; previous hardening contact +0.15 [0.09,0.23], keypoint+contact +0.05 [-0.04,0.15], no-mesh -0.57 [-0.73,-0.41]. Major original shortcuts: disconnected body +3.1164, unchecked points +2.2503, mutable task +3.1625; impossible/empty/partial/filter/unlisted candidates tied the reference at 0. These are historical planar tests, not before/after estimates on a common scientific scale.

Pending final measurements: reference values, honest variant ranks and paired intervals, shortcut after scores, repeat difference, active metric bounds and train/test comparison. Remaining limitations and the final milestone will be stated after the audit, including any failures rather than claiming them fixed.
