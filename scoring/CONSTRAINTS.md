# Constraints

- Return finite `[T,36]` G1 configurations for every instance through the registered executable interface. Respect the model's joint ranges and URDF joint velocity limits. Base quaternions must be unit length, translation is limited to 20 cm/frame, angular speed to 12 rad/s, and base coordinates to 100 m in absolute value.
- A violation, crash, missing output or CPU-budget excess fails the whole run, with worst values for every metric in all three experiments.
- Budget: 30 CPU seconds per instance, summed over the process tree; one supplied BLAS/OpenMP thread. There is no wall-clock cutoff.
- Read only instance inputs, public train data, public source and system libraries. Do not read hidden test assets, git history or evaluation code. Write temporary state and outputs in the invocation workspace or `/tmp`.
- Do not change robot assets, evaluator code, thresholds, test construction, score metadata or comparison methods. Hidden evaluation owns all of these independently.
- No training or internet access is required at test time. An arbitrary new method can run through the same interface as the reference and baselines.
