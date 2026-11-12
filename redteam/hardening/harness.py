"""Hardening check: official runs, then variants + shortcuts through the unchanged evaluators.

1. Official gym: only the shipped retargeters registered; ``bash scripts/evaluate_train.sh``
   and ``bash scripts/evaluate_test.sh`` (no arguments).  Results saved to results/official.
2. Combined: register honest variants and shortcut submissions next to the shipped
   methods, apply the state-tampering shortcuts (edited train scores, rewritten public
   train tasks), and run ``bash scripts/evaluate_test.sh`` and ``evaluate_train.sh``
   again.  Results saved to results/combined_{test,train}.
3. Restore all tampered state and the official score files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/user")
REG = ROOT / "scripts" / "retargeters"
OUT = ROOT / "redteam" / "hardening" / "results"
SCORE_FILES = ["scores_train.json", "diagnostics_train.json", "scores_test.json", "diagnostics_test.json"]

VARIANTS = {
    "var_no_mesh": 'python method/omniretarget.py --objective keypoint --name var_no_mesh --inputs "$1" --outputs "$2"',
    "var_contact_aware": 'python redteam/hardening/variants.py contact_aware "$1" "$2"',
    "var_keypoint_contact": 'python redteam/hardening/variants.py keypoint_contact "$1" "$2"',
}
SHORTCUT_MODES = [
    "zero_q", "disconnected_keypoints", "impossible_q", "nan", "malformed", "empty", "constant_pose",
    "joint_violation", "velocity_violation", "partial_missing", "omit_failed_experiment", "score_forgery",
]


def sh(cmd: str, log: Path) -> int:
    with open(log, "w") as fh:
        r = subprocess.run(["bash", "-c", cmd], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"$ {cmd} -> {r.returncode}", flush=True)
    return r.returncode


def register(name: str, body: str) -> Path:
    p = REG / f"{name}.sh"
    p.write_text("#!/bin/bash\nset -e\ncd /home/user\n" + body + "\n")
    return p


def save(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in SCORE_FILES:
        src = ROOT / "scoring" / f
        if src.exists():
            shutil.copy2(src, dest / f)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage in ("all", "official"):
        status = {"train": sh("bash scripts/evaluate_train.sh", OUT / "official_train.log"),
                  "test": sh("bash scripts/evaluate_test.sh", OUT / "official_test.log")}
        save(OUT / "official")
        (OUT / "official" / "exit_codes.json").write_text(json.dumps(status))

    if stage in ("all", "combined"):
        added = [register(n, b) for n, b in VARIANTS.items()]
        added += [register(f"sc_{m}", f'python redteam/hardening/shortcuts.py {m} "$1" "$2"') for m in SHORTCUT_MODES]
        backup = Path(tempfile.mkdtemp(prefix="hardening_backup_"))
        shutil.copytree(ROOT / "data" / "tasks", backup / "tasks")
        shutil.copytree(ROOT / "scoring", backup / "scoring")
        try:
            # Tampering that the hidden test must ignore: drop methods from the writable
            # train scores and replace every public train task with trivial content.
            train_scores = ROOT / "scoring" / "scores_train.json"
            if train_scores.exists():
                data = json.loads(train_scores.read_text())
                for exp in data["experiments"].values():
                    exp["results"] = {}
                train_scores.write_text(json.dumps(data))
            for p in (ROOT / "data/tasks/train/inputs").glob("*.json"):
                inp = json.loads(p.read_text())
                inp["source_keypoints"] = [[[0.0, 0.0]] * 12] * inp["num_frames"]
                inp["target_scene"] = {"terrain": [[-4.0, -5.0], [4.0, -5.0]], "object": None}
                p.write_text(json.dumps(inp))
            for p in (ROOT / "data/tasks/train/annotations").glob("*.json"):
                ann = json.loads(p.read_text())
                ann["stance"] = {k: [False] * len(v) for k, v in ann["stance"].items()}
                ann["hand_contact"] = {k: [None] * len(v) for k, v in ann["hand_contact"].items()}
                p.write_text(json.dumps(ann))
            code = sh("bash scripts/evaluate_test.sh", OUT / "combined_test.log")
            save(OUT / "combined_test")
            (OUT / "combined_test" / "exit_code.txt").write_text(str(code))
        finally:
            shutil.rmtree(ROOT / "data" / "tasks")
            shutil.copytree(backup / "tasks", ROOT / "data" / "tasks")
        code = sh("bash scripts/evaluate_train.sh", OUT / "combined_train.log")
        save(OUT / "combined_train")
        (OUT / "combined_train" / "exit_code.txt").write_text(str(code))
        for p in added:
            p.unlink()
        for f in SCORE_FILES:
            src = OUT / "official" / f
            if src.exists():
                shutil.copy2(src, ROOT / "scoring" / f)
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    main()
