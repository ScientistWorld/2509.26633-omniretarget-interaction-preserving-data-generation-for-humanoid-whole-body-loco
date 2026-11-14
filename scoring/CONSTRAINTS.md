# Constraints

- A method is a registered executable `scripts/retargeters/<name>.sh <input_dir> <output_dir>`;
  it must write `{"q": ...}` for every input instance. Only `q` is scored.
- Outputs must respect the robot defined in `data/planar_robot.py`: joint limits
  `Q_MIN`/`Q_MAX` and per-frame velocity limits `V_MAX * dt`. Violations, missing
  instances, malformed or non-finite trajectories fail the whole run (worst value
  everywhere).
- Budget: 6 CPU seconds per instance (user + system time of the method's whole
  process tree, counted by the evaluator; one BLAS/OpenMP thread). The reference
  method needs about 0.7. Exceeding it, or making no CPU progress for 1800
  consecutive one-second checks, fails the run. Wall-clock time is not scored.
- The method must not use evaluation labels or the evaluator's code: it receives
  only the instance inputs (demonstration keypoints, source / nominal / target
  scenes, timing). Do not read or import anything under `eval/`.
- Do not change the robot definition, the scene geometry helpers, the task
  generator, the evaluator, the thresholds or the train data to change scores;
  the hidden test uses its own copies and never reads them.
