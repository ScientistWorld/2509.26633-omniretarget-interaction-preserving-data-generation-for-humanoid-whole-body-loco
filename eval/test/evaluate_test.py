"""Hidden test-split evaluator.

Builds the held-out test instances in memory (eval/test/hidden_tasks.py), runs
every method registered in ``scripts/retargeters/`` on them (a fresh temporary
copy of the inputs per method; labels exist only in this process's memory),
recomputes all metrics from the returned robot trajectories and writes
``scoring/scores_test.json``.  Methods that fail any instance (or exceed the
counted CPU budget) get the worst value of every metric in every experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.test.hidden_tasks import test_split
from eval.test.scoring_core import evaluate_all

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score-out", type=Path, default=ROOT / "scoring/scores_test.json")
    parser.add_argument("--diagnostics-out", type=Path, default=ROOT / "scoring/diagnostics_test.json")
    args = parser.parse_args()
    evaluate_all(ROOT, test_split(), "test", args.score_out, args.diagnostics_out, ROOT / "logs/eval_test")


if __name__ == "__main__":
    main()
