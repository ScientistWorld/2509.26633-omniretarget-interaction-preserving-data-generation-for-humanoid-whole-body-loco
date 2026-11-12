# Problem Definition

**Kinematic motion retargeting for humanoid whole-body interaction.** A human
demonstration (a time series of body keypoints) was recorded while the human
interacted with a scene (a carried object and/or terrain). Produce a joint
trajectory for a humanoid robot with different body proportions that

- preserves the demonstrated interactions in the scene where the robot must act
  (hands stay on the object or on the terrain surface they touched; feet stay
  planted on the ground while the demonstrator's feet were planted),
- respects the robot's hard kinematic limits (joint ranges and per-frame joint
  velocity limits), does not push any part of the body into objects or terrain,
  and does not let stance feet slide,
- stays faithful to the demonstrated motion (pelvis path and keypoint
  velocities).

These goals conflict: the robot's limbs and torso differ in length from the
human's, so copying keypoint positions breaks contacts and causes penetration
or skating, while enforcing contacts alone can distort the motion.

## Robot

`data/planar_robot.py` defines the robot: a planar (frontal-plane) humanoid with
12 keypoints (pelvis, neck, knees, feet, shoulders, elbows, hands) and an
11-dimensional configuration `q = [base_x, base_z, torso, left_hip, left_knee,
right_hip, right_knee, left_shoulder, left_elbow, right_shoulder, right_elbow]`.
It provides forward kinematics, the joint limits `Q_MIN`/`Q_MAX`, the velocity
limits `V_MAX` (per second; per frame multiply by `dt`) and the list of rigid
links whose full length must stay out of solids. `data/scene_geometry.py`
provides signed distances to terrain polylines and boxes.

## Instances

Each instance is one demonstration (2.6–3.2 s at 30 Hz) of one of two task types:

- `object_carry`: the human side-steps while holding a box overhead with both
  hands on its sides; the box also moves up and down.
- `terrain_step`: the human side-steps up a ramp onto a platform and braces the
  right hand on top of a tall column standing on the platform.

Demonstrators vary in overall size (roughly 0.9–1.15 times the robot) and in
individual segment proportions. Every instance carries two scenes:

- `source_scene`: the scene the human interacted with,
- `target_scene`: the scene in which the robot must act. In most instances it is
  identical to the source scene; in *augmented* instances it is a modified copy
  (the box is resized and starts displaced, with the displacement decaying over
  time, or the platform and column heights change). Scene correspondences are
  explicit: terrain polylines have the same vertices in both scenes, and the box
  keeps its identity (`center` per frame, `size`).

Input file format (`data/tasks/train/inputs/<instance_id>.json`):

```
instance_id, task_type, dt, num_frames, keypoint_names,
source_keypoints  [num_frames][12][2]   (metres; x horizontal, z up)
source_scene      {"terrain": [[x, z], ...], "object": null | {"size": [w, h], "center": [num_frames][2]}}
target_scene      same structure
```

No contact labels or stance labels are provided to methods; they must be
inferred from the demonstration, as in the original retargeting setting.

## What a method must produce

A method is an executable `scripts/retargeters/<name>.sh` invoked as

```
bash scripts/retargeters/<name>.sh <input_dir> <output_dir>
```

from the workspace root. For every `<input_dir>/<instance_id>.json` it must write
`<output_dir>/<instance_id>.json` containing `{"q": [[11 numbers] x num_frames]}`.
Only `q` is read; any other fields are ignored. The same command is used for the
public train split and for the hidden test split.

Public train instances (inputs and evaluation labels) are in `data/tasks/train/`
and can be regenerated with `python data/task_generator.py`.
