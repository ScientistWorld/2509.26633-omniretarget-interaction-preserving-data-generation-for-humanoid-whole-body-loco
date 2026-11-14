"""Hardening check (iteration 12): official runs, then variants + shortcuts through the unchanged evaluators.

1. official  Only the shipped retargeters registered; ``bash scripts/evaluate_train.sh`` and
             ``bash scripts/evaluate_test.sh`` (no arguments), nothing else running beside them.
             Results saved to results/official (raw outputs kept for rescoring).
2. combined  Register honest variants and shortcut submissions next to the shipped methods,
             apply the state-tampering shortcuts (emptied train scores, rewritten public train
             tasks), run ``bash scripts/evaluate_test.sh``; restore the train tasks and run
             ``bash scripts/evaluate_train.sh``.  Results saved to results/combined_{test,train}.
             All tampered state and the official score files are restored afterwards.
3. budget    Direct checks of the counted CPU budget and the hang guard on tiny budgets.
"""

from __future__ import annotations

import json
import os
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
    f"var_{v}": f'python redteam/hardening/variants.py {v} "$1" "$2"'
    for v in ["no_mesh", "contact_aware", "keypoint_contact", "keypoint_contact_w100", "keypoint_contact_it11"]
}
SHORTCUT_MODES = [
    "zero_q", "disconnected_keypoints", "impossible_q", "nan", "malformed", "empty", "constant_pose",
    "joint_violation", "velocity_violation", "partial_missing", "omit_failed_experiment", "score_forgery",
    "select_best",
]


def sh(cmd: str, log: Path, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env.update(env_extra or {})
    with open(log, "w") as fh:
        r = subprocess.run(["bash", "-c", cmd], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"$ {cmd} -> {r.returncode}", flush=True)
    return r.returncode


def register(name: str, body: str) -> Path:
    p = REG / f"{name}.sh"
    p.write_text("#!/bin/bash\nset -e\ncd /home/user\n" + body + "\n")
    return p


def save(dest: Path, files=SCORE_FILES) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = ROOT / "scoring" / f
        if src.exists():
            shutil.copy2(src, dest / f)


def official() -> None:
    d = OUT / "official"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    keep = {"RETARGET_KEEP_OUTPUTS": str(d / "outputs")}
    status = {"train": sh("bash scripts/evaluate_train.sh", OUT / "official_train.log", keep),
              "test": sh("bash scripts/evaluate_test.sh", OUT / "official_test.log", keep)}
    save(d)
    (d / "exit_codes.json").write_text(json.dumps(status))


def combined() -> None:
    added = [register(n, b) for n, b in VARIANTS.items()]
    added += [register(f"sc_{m}", f'python redteam/hardening/shortcuts.py {m} "$1" "$2"') for m in SHORTCUT_MODES]
    backup = Path(tempfile.mkdtemp(prefix="hardening_backup_"))
    shutil.copytree(ROOT / "data" / "tasks", backup / "tasks")
    shutil.copytree(ROOT / "scoring", backup / "scoring")
    try:
        for d in ("combined_test", "combined_train", "combined_outputs"):
            shutil.rmtree(OUT / d, ignore_errors=True)
        try:
            # Tampering the hidden test must ignore: drop methods from the writable train
            # scores and replace every public train task with trivial content.
            train_scores = ROOT / "scoring" / "scores_train.json"
            if train_scores.exists():
                data = json.loads(train_scores.read_text())
                for exp in data["experiments"].values():
                    exp["results"] = {}
                train_scores.write_text(json.dumps(data))
            for p in (ROOT / "data/tasks/train/inputs").glob("*.json"):
                inp = json.loads(p.read_text())
                inp["source_keypoints"] = [[[0.0, 0.0]] * 12] * inp["num_frames"]
                for key in ("nominal_scene", "target_scene"):
                    inp[key] = {"terrain": [[-4.0, -5.0], [4.0, -5.0]], "object": None}
                p.write_text(json.dumps(inp))
            for p in (ROOT / "data/tasks/train/annotations").glob("*.json"):
                ann = json.loads(p.read_text())
                ann["stance"] = {k: [False] * len(v) for k, v in ann["stance"].items()}
                ann["hand_contact"] = {k: [None] * len(v) for k, v in ann["hand_contact"].items()}
                p.write_text(json.dumps(ann))
            code = sh("bash scripts/evaluate_test.sh", OUT / "combined_test.log",
                      {"RETARGET_KEEP_OUTPUTS": str(OUT / "combined_outputs")})
            save(OUT / "combined_test", ["scores_test.json", "diagnostics_test.json"])
            (OUT / "combined_test" / "exit_code.txt").write_text(str(code))
        finally:
            shutil.rmtree(ROOT / "data" / "tasks")
            shutil.copytree(backup / "tasks", ROOT / "data" / "tasks")
        code = sh("bash scripts/evaluate_train.sh", OUT / "combined_train.log",
                  {"RETARGET_KEEP_OUTPUTS": str(OUT / "combined_outputs")})
        save(OUT / "combined_train", ["scores_train.json", "diagnostics_train.json"])
        (OUT / "combined_train" / "exit_code.txt").write_text(str(code))
    finally:
        for p in added:
            p.unlink(missing_ok=True)
        for f in SCORE_FILES:
            src = OUT / "official" / f
            if src.exists():
                shutil.copy2(src, ROOT / "scoring" / f)
        shutil.rmtree(backup, ignore_errors=True)


def budget() -> None:
    """The counted CPU budget and the hang guard, exercised directly with tiny limits."""
    sys.path.insert(0, str(ROOT))
    from eval.test import scoring_core as sc
    work = Path(tempfile.mkdtemp(prefix="budget_"))
    cases = {
        "cpu_burn": "python -c 'while True: pass'",
        "cpu_burn_in_child": "bash -c \"python -c 'while True: pass' & wait\"",
        "sleep_forever": "sleep 100000",
        "quick_ok": "python -c 'print(1)'",
    }
    out = {}
    sc.IDLE_POLL_LIMIT = 10
    for name, body in cases.items():
        script = work / f"{name}.sh"
        script.write_text("#!/bin/bash\n" + body + "\n")
        reason, cpu = sc.run_method(ROOT, script, work, work / "out", budget=5.0, log_path=work / f"{name}.log")
        out[name] = {"reason": reason, "cpu_s": round(cpu, 2)}
        print(name, out[name], flush=True)
    (OUT / "budget_checks.json").write_text(json.dumps(out, indent=2))
    shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "official"):
        official()
    if stage in ("all", "combined"):
        combined()
    if stage in ("all", "budget"):
        budget()


if __name__ == "__main__":
    main()
