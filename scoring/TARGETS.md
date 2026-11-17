# Targets

Each of `object_interaction_kinematics`, `terrain_interaction_kinematics` and `augmented_interaction_kinematics` has a corresponding entry in both split score files and measures:

- Benefit: `contact_preservation` (higher).
- Constraints: `penetration_duration`, `max_penetration_cm`, `foot_skating_duration`, `max_foot_skating_cm_s`, `spurious_contact_duration`, `support_failure_duration` (lower).
- Motion fidelity: `pelvis_path_error_cm`, `pose_error_cm` (lower).

The goal is better contact preservation and feasible motion together. Joint/base validity, complete output and CPU budget are hard gates. No downstream RL, hardware, model-size, or wall-clock target is scored.
