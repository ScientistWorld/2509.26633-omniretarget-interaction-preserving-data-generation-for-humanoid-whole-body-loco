# Evaluation

`bash scripts/evaluate_train.sh` (and `bash scripts/evaluate.sh`) score every
method registered in `scripts/retargeters/` on the public train split and write
`scoring/scores_train.json` (`scoring/scores.json`). The hidden test runs
`bash scripts/evaluate_test.sh` in a fresh container: it creates held-out
instances of the same kinds (same generating distribution, different draws, 60
per experiment), runs every registered method on them through the same command,
and writes `scoring/scores_test.json`. Methods are therefore scored on
demonstrations and scenes they have never seen.

## What is measured

Everything is recomputed from the returned joint trajectory `q` with the
evaluator's own copy of the robot kinematics and scene geometry. Robot keypoints
written by a method, claimed metrics, flags or score files are never read.

Per instance, against the **target scene** (`alpha` is the robot-to-demonstrator
height ratio used to build the nominal scene):

| metric | meaning | better |
|---|---|---|
| `penetration_duration` | fraction of frames where any point along any robot link is more than 1 cm inside the terrain or the object | lower |
| `max_penetration_cm` | deepest such penetration (cm, capped at 30) | lower |
| `foot_skating_duration` | fraction of desired stance-foot frame pairs (the demonstrator's foot was planted in both frames) in which the robot foot moves faster than 1 cm/s | lower |
| `max_foot_skating_cm_s` | largest robot foot speed over those frame pairs (cm/s, capped at 300) | lower |
| `contact_preservation` | fraction of desired contact events kept: each frame where a demonstrator hand touched the object or terrain, and each frame where a demonstrator foot was planted, counts as one event; it is kept if the corresponding robot point is within 2 cm of the corresponding target-scene surface | higher |
| `spurious_contact_duration` | fraction of clearance events in which the robot touches a surface anyway: a clearance event is a frame where a demonstrator hand or foot was at least 3 cm (times `alpha`) from every source-scene surface; it is violated if the corresponding robot hand or foot is within 1 cm of a target-scene surface (dragged swing feet, hands brushing obstacles) | lower |
| `pelvis_path_error_cm` | mean distance between the robot pelvis path and the demonstrator's pelvis path scaled by `alpha`; both paths are measured from their first frame, horizontally and as height above the ground directly below the pelvis, so following a raised or lowered terrain is not penalised (cm, capped at 100) | lower |
| `motion_velocity_error_cm_s` | mean mismatch between robot keypoint velocities and the demonstrator's keypoint velocities scaled by `alpha` (cm/s, capped at 300) | lower |

An experiment's value for a metric is the mean over its instances.

## Validity gate and budget

A method's run is **failed** if it exits non-zero, exceeds its CPU budget, stops
making progress, misses any instance, or returns any `q` that is malformed, has
the wrong number of frames, contains non-finite values, leaves the joint limits,
or changes any coordinate between consecutive frames by more than the velocity
limit times `dt` (tolerance 1e-6). A failed run receives the worst value of every
metric in every experiment (1 for durations, 0 for contact, the caps above for
the rest). There is no partial credit for skipping instances or experiments.

The budget is counted by the evaluator, not measured in wall-clock time: 6 CPU
seconds (user + system, summed over the method's whole process tree) per
instance of the split, i.e. 1080 CPU s for the 180 test instances and 360 CPU s
for the 60 train instances. Methods run with one BLAS/OpenMP thread. A method
whose process tree uses no CPU across 1800 consecutive one-second checks is
treated as hung. Wall-clock time is not scored.

## Experiments

- `object_interaction_kinematics` (weight 0.35): nominal box-carry instances
  (with a ramp up to a platform).
- `terrain_interaction_kinematics` (weight 0.35): nominal terrain instances.
- `augmented_interaction_kinematics` (weight 0.30): instances whose target scene
  differs from the nominal scene (half object, half terrain).

`scoring/reference.json` lists metric directions and coefficients. Per-instance
values and failure reasons are written to `scoring/diagnostics_*.json`
(diagnostics are not ranked).
