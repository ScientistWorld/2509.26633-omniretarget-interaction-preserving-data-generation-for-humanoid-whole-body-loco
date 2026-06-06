# Evaluation

Evaluation reads candidate artifacts from `checkpoints/retargeting/*.json` and writes `scoring/scores.json`. It does not import from any solution implementation. The evaluator also writes `scoring/diagnostics.json` with every artifact it scored.

Metrics measure kinematic interaction quality:

- `penetration_duration`: fraction of frames with robot-object or robot-terrain penetration.
- `max_penetration_cm`: maximum penetration depth in centimeters.
- `foot_skating_duration`: fraction of desired stance-foot contacts where the robot foot moves above the tolerance.
- `max_foot_skating_cm_s`: maximum stance-foot speed in centimeters per second.
- `contact_preservation`: fraction of desired hand-object or hand-terrain contacts that remain close enough to contact.

A successful candidate should reduce physical artifacts while preserving task contacts. Higher-quality results improve the same metrics across both interaction settings.
