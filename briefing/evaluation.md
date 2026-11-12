# Evaluation

`bash scripts/evaluate_train.sh` (and `bash scripts/evaluate.sh`) score every
method registered in `scripts/retargeters/` on the public train split and write
`scoring/scores_train.json` (`scoring/scores.json`). The hidden test runs
`bash scripts/evaluate_test.sh` in a fresh container: it creates held-out
instances of the same kinds (same generating distribution, different draws, 40
per experiment), runs every registered method on them through the same command,
and writes `scoring/scores_test.json`. Methods are therefore scored on
demonstrations and scenes they have never seen.

## What is measured

Everything is recomputed from the returned joint trajectory `q` with the
evaluator's own copy of the robot kinematics and scene geometry. Robot keypoints
written by a method, claimed metrics, flags or score files are never read.

Per instance, against the **target scene**:

| metric | meaning | better |
|---|---|---|
| `penetration_duration` | fraction of frames where any point along any robot link is more than 1 cm inside the terrain or the object | lower |
| `max_penetration_cm` | deepest such penetration (cm, capped at 30) | lower |
| `foot_skating_duration` | fraction of desired stance-foot frame pairs (the demonstrator's foot was planted in both frames) in which the robot foot moves faster than 1 cm/s | lower |
| `max_foot_skating_cm_s` | largest robot foot speed over those frame pairs (cm/s, capped at 300) | lower |
| `contact_preservation` | fraction of desired contact events kept: each frame where a demonstrator hand touched the object or terrain, and each frame where a demonstrator foot was planted, counts as one event; it is kept if the corresponding robot point is within 2 cm of the corresponding target-scene surface | higher |
| `pelvis_path_error_cm` | mean distance between the robot's pelvis displacement (from its first frame) and the demonstrator's (cm, capped at 100) | lower |
| `motion_velocity_error_cm_s` | mean keypoint velocity mismatch between robot and demonstrator (cm/s, capped at 300) | lower |

An experiment's value for a metric is the mean over its instances.

## Validity gate

A method's run is **failed** if it exits non-zero, exceeds the time limit
(2400 s per split), misses any instance, or returns any `q` that is malformed,
has the wrong number of frames, contains non-finite values, leaves the joint
limits, or changes any coordinate between consecutive frames by more than the
velocity limit times `dt` (tolerance 1e-6). A failed run receives the worst value
of every metric in every experiment (1 for durations, 0 for contact, the caps
above for the rest). There is no partial credit for skipping instances or
experiments.

## Experiments

- `object_interaction_kinematics` (weight 0.35): nominal object-carry instances.
- `terrain_interaction_kinematics` (weight 0.35): nominal terrain instances.
- `augmented_interaction_kinematics` (weight 0.30): instances whose target scene
  differs from the demonstration scene (half object, half terrain).

`scoring/reference.json` lists metric directions and coefficients. Per-instance
values and failure reasons are written to `scoring/diagnostics_*.json`
(diagnostics are not ranked).
