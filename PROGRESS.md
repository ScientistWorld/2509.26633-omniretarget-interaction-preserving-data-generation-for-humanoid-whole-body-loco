# Progress

## What Works

- Read the OmniRetarget paper and extracted the core algorithm, baselines, metrics, and Table II reported numbers.
- Implemented a scaled planar humanoid benchmark with object-carry and terrain-step interaction tasks.
- Implemented the paper's core retargeting computation: interaction-mesh Laplacian objective, Delaunay neighbors, object/terrain mesh samples, warm-started per-frame constrained optimization, joint/velocity limits, non-penetration constraints, and stance-foot sticking constraints.
- Implemented a keypoint-position IK baseline from the paper's baseline family.
- Implemented method-independent evaluation that reads artifacts and writes `scoring/scores.json`.

## Results

No reproduced scores have been produced yet. Job `bd8cb121-d23` reached the compute container and confirmed Python 3.11, but failed before running the optimizer because `baseline/keypoint_ik.py` could not import the local `data` package. The fix adds package initializers and exports `PYTHONPATH=/home/user` in runnable scripts.

## Remaining

- Run the packaged reproduction in the compute container.
- Inspect `scoring/scores.json` and compare against the expected qualitative pattern.
- If the method does not beat the baseline on the constraint metrics, tune the synthetic task or optimizer without changing the paper's algorithm.

## Issues

- The paper states that OmniRetarget code, retargeted datasets, and trained policies will be released, but the markdown provides no repository URL for the proposed method.
- Full Unitree/Drake/RL reproduction is outside the current workspace assets; this first job targets the core kinematic retargeting claim.
- A local smoke test could not run before container setup because `numpy` is not installed in the login Python.
- Job `bd8cb121-d23` failed due to missing local package initialization, fixed in the next revision.

## Deviations from Paper

- Paper data: OMOMO, LAFAN1, and in-house MoCap. Substitute: deterministic synthetic object and terrain interaction tasks in `data/`. Reason: the OmniRetarget release and in-house MoCap are not available from the paper markdown, and a small benchmark is needed for a reusable gym.
- Paper robot/solver: Unitree humanoids, Drake autodiff, custom sequential SOCP. Substitute: planar humanoid model and SciPy SLSQP constrained optimization. Reason: preserve the same objective and hard-constraint formulation at a scale that can run in this environment.
- Paper downstream RL: BeyondMimic-style policy training. Substitute: kinematic metrics only for the first milestone. Reason: the paper's central retargeting claim is directly measured by Table II kinematic metrics, while full RL requires unavailable simulator assets and policies.
