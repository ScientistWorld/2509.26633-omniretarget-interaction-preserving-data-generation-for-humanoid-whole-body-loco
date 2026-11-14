"""Visible train-split evaluator.

Runs every method registered in ``scripts/retargeters/`` on the public train
instances (``data/tasks/train/inputs``), recomputes all metrics from the
returned joint trajectories using the public train annotations, and writes
``scoring/scores_train.json`` (and a per-instance diagnostics file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.train.scoring_core import evaluate_all

ROOT = Path(__file__).resolve().parents[2]


def load_split(task_dir: Path) -> tuple[dict, dict]:
    inputs, anns = {}, {}
    for path in sorted((task_dir / "inputs").glob("*.json")):
        inp = json.loads(path.read_text())
        iid = inp["instance_id"]
        inputs[iid] = inp
        anns[iid] = json.loads((task_dir / "annotations" / f"{iid}.json").read_text())
    if not inputs:
        raise SystemExit(f"no train instances found in {task_dir}")
    return inputs, anns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/tasks/train")
    parser.add_argument("--score-out", type=Path, default=ROOT / "scoring/scores_train.json")
    parser.add_argument("--diagnostics-out", type=Path, default=ROOT / "scoring/diagnostics_train.json")
    args = parser.parse_args()
    inputs, anns = load_split(args.tasks)
    evaluate_all(ROOT, inputs, anns, "train", args.score_out, args.diagnostics_out, ROOT / "logs/eval_train")


if __name__ == "__main__":
    main()
