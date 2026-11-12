# Constraints

- A method is a registered executable `scripts/retargeters/<name>.sh <input_dir> <output_dir>`;
  it must write `{"q": ...}` for every input instance. Only `q` is scored.
- Outputs must respect the robot defined in `data/planar_robot.py`: joint limits
  `Q_MIN`/`Q_MAX` and per-frame velocity limits `V_MAX * dt`. Violations, missing
  instances, malformed or non-finite trajectories fail the whole run (worst value
  everywhere).
- The method must not use evaluation labels or the evaluator's code: it receives
  only the instance inputs (demonstration keypoints, source scene, target scene,
  timing). Do not read or import anything under `eval/`.
- Do not change the robot definition, the scene geometry helpers, the task
  generator, the evaluator, the thresholds or the train data to change scores;
  the hidden test uses its own copies and never reads them.
- Per-split time limit: 2400 s for all instances of a split (the reference
  method needs about a minute). Wall-clock time is not scored.
