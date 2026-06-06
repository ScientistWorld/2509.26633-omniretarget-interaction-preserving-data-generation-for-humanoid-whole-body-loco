# Evaluation

Evaluation reads method artifacts from `checkpoints/retargeting/*.json` and writes `scoring/scores.json`. It does not import from `method/`.

Metrics mirror the paper's kinematic quality metrics:

- `penetration_duration`: fraction of frames with robot-object or robot-terrain penetration.
- `max_penetration_cm`: maximum penetration depth in centimeters.
- `foot_skating_duration`: fraction of desired stance-foot contacts where the robot foot moves above the tolerance.
- `max_foot_skating_cm_s`: maximum stance-foot speed in centimeters per second.
- `contact_preservation`: fraction of desired hand-object or hand-terrain contacts that remain close enough to contact.

The core claim is supported when the proposed method reduces penetration and foot skating while preserving task contacts compared with a keypoint-matching IK baseline on at least one interaction setting. Higher milestones require the same effect across additional interaction settings.
