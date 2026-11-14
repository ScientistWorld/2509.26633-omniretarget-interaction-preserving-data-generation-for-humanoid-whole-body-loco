# Targets

Every experiment below is scored in `scoring/scores_train.json` (public train)
and `scoring/scores_test.json` (hidden test), with the same eight metrics:

| experiment | benefit metric | constraint metrics | fidelity metrics |
|---|---|---|---|
| `object_interaction_kinematics` | `contact_preservation` (primary) | `penetration_duration`, `max_penetration_cm`, `foot_skating_duration`, `max_foot_skating_cm_s`, `spurious_contact_duration` | `pelvis_path_error_cm`, `motion_velocity_error_cm_s` |
| `terrain_interaction_kinematics` | `contact_preservation` (primary) | same | same |
| `augmented_interaction_kinematics` | `contact_preservation` (primary) | same | same |

Hard requirements (validity gate, not traded off): joint limits, per-frame
velocity limits, a well-formed finite trajectory for every instance, and the
counted CPU budget (6 CPU s per instance).

Goal: preserve more demonstrated contacts without more penetration, foot
skating or spurious contact, and without drifting away from the demonstrated
motion.
