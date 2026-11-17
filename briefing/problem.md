# Problem

Retarget a real human demonstration to a humanoid with different proportions while preserving its interactions with a carried box or climbing terrain. Produce motion that keeps demonstrated contacts, avoids scene penetration and foot skating, respects kinematic limits, and follows the human's path and body pose. Contacts, feasibility and motion fidelity must improve together.

The robot is the public Unitree G1 with 29 articulated joints and a floating base. Its self-contained MuJoCo model and URDF joint limits are in `data/robot/`. Collision meshes use deterministic convex hulls with at most 128 vertices. This is a kinematic task; balance, torque feasibility and downstream control are not measured.

Public instances are real OMOMO box-manipulation clips and public climbing motion capture, cut into 90–150-frame windows at 30 Hz. Test uses different subjects and climbing recordings. The train split has 44 windows; the test split has 100. The source scene is at human scale. Nominal terrain and object trajectories are scaled by `alpha = 1.32 / human_height`; the manipulated box retains its physical size. Some target scenes have different box dimensions and initial poses, or changed platform heights.

Each input JSON contains `instance_id`, `task` (`object` or `terrain`), `fps`, `num_frames`, `human_height`, `robot_height`, `alpha`, `joint_names`, `human_joints[T][J][3]`, `source_scene`, `nominal_scene`, `target_scene`, and optional `augmentation`. Distances are metres. A scene contains `ground_z` and oriented `boxes`: `kind`, `half_size`, optional local centre `offset`, and either a static `pose[7]` or a `poses[T][7]` trajectory. Poses are xyz followed by a wxyz quaternion. No contact or stance labels are supplied.

Register any procedure as `scripts/retargeters/<name>.sh`. Both evaluators invoke:

```bash
bash scripts/retargeters/<name>.sh <input_dir> <output_dir>
```

For every input write `<output_dir>/<instance_id>.json` containing `{"q": ...}` with shape `[num_frames, 36]`: base xyz, unit wxyz quaternion, then the model's 29 joint positions. Only this raw configuration trajectory is scored. Methods can supply their own optimization procedure; they are not limited to choosing settings of a shipped method.

Run `bash scripts/download.sh` to fetch public assets and prepare train inputs under `data/downloads/tasks/train/inputs/`. Download commands are idempotent and the compute node needs no internet. This task needs no model training.
