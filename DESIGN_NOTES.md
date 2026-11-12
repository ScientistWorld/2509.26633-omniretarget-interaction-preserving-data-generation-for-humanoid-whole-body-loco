# Design notes (current gym, after iteration 11 hardening)

- **Split:** public train = 60 procedurally generated demonstrations in `data/tasks/train/` (20 per experiment, generator `data/task_generator.py`, public seed). Hidden test = 120 demonstrations (40 per experiment) generated in memory at test time by `eval/test/hidden_tasks.py` (self-contained copy of the generator, private seed). Same distribution, disjoint draws.
- **Method interface:** every `scripts/retargeters/<name>.sh <input_dir> <output_dir>` is discovered and run by both evaluators on fresh temporary copies of the inputs; outputs are `{"q": [...]}` per instance.
- **Scoring:** `eval/train/scoring_core.py` and `eval/test/scoring_core.py` (independent copies) recompute all seven metrics from `q` with their own robot model and scene geometry. Validity gate (shape, finiteness, joint limits, per-frame velocity limits, every instance present, exit code, 2400 s per split); a failed run gets the worst value of every metric in every experiment.
- **Experiments:** object (0.35), terrain (0.35), augmented scene (0.30). See `scoring/EXPERIMENTS.md`, `briefing/evaluation.md`.
- **Audit:** `HARDENING.md` and `redteam/hardening/` (harness, variants, shortcut submissions, results).
