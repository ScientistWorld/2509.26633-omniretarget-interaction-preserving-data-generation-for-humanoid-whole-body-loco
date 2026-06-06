# Targets

Optimize the retargeted trajectory quality on:

- `object_interaction_kinematics`
- `terrain_interaction_kinematics`

Primary constraints are low `penetration_duration`, low `max_penetration_cm`, low `foot_skating_duration`, and low `max_foot_skating_cm_s`.

The benefit metric is `contact_preservation`: methods should preserve intended object or terrain contacts while satisfying the constraints.
