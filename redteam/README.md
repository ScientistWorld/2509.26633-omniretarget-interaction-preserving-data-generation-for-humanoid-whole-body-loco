# Gym scoring audit

Run `bash redteam/run_all.sh` on a CPU compute node. Every case has its own
`run.sh`; each produces its candidate artifacts, invokes the visible evaluator,
then invokes `bash scripts/evaluate_test.sh` **without arguments** in a separate
process. The evaluator and its environment are never modified. No background
process persists. The orchestrator restores data, checkpoints, and scoring
after each case, including after an exception.

`scores_test.json`, `diagnostics_test.json`, and `status.json` in each case
preserve the actual hidden-test result. A test failure has no fabricated score.
Artifacts and logs remain locally available but are ignored by git. The audit
uses only visible data/baseline code for from-scratch shortcuts; experiments
that refine or reuse the reference are explicitly improve-mode experiments.

`summary.json` reports a cohort-normalized rank *difference* from the reference,
using experiment weights and absolute coefficients. The future ranking cohort
does not exist yet: these numbers are illustrative, and the underlying metric
vectors are authoritative. A Pareto improvement or exact tie is independent of
the unknown positive normalizers. Missing results contribute zero, as specified
in the audit request. NaN handling is reported separately, not assumed safe.
