"""Hardening check (iteration 12): official runs, then variants + shortcuts through the unchanged evaluators.

Resumable (the compute nodes preempt jobs): every stage/chunk writes a done marker and is
skipped when rerun; every invocation first restores clean state (no audit registrations,
shipped retargeters in place, official score files in scoring/).

1. official  Only the shipped retargeters registered; ``bash scripts/evaluate_train.sh`` and
             ``bash scripts/evaluate_test.sh`` (no arguments), nothing else running beside them.
             Results saved to results/official (raw outputs kept for rescoring).
2. combined  Chunks of honest variants and shortcut submissions registered as methods and run
             through ``bash scripts/evaluate_test.sh`` (and ``evaluate_train.sh``).  The shipped
             retargeters are unregistered during the chunks (their scores are taken from the
             official run; every method is scored independently) except in the ``tamper`` chunk,
             which reruns them while the public train tasks and train scores are overwritten.
             Chunk results are merged into results/combined_{test,train}.
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
CHUNKS_DIR = OUT / "chunks"
SHIPPED_BACKUP = OUT / ".shipped_backup"
TAMPER_MARK = OUT / ".train_tampered"
SHIPPED = ["omniretarget", "keypoint_ik"]
SCORE_FILES = ["scores_train.json", "diagnostics_train.json", "scores_test.json", "diagnostics_test.json"]


def variant(v: str) -> str:
    return f'python redteam/hardening/variants.py {v} "$1" "$2"'


def shortcut(m: str) -> str:
    return f'python redteam/hardening/shortcuts.py {m} "$1" "$2"'


TEST_CHUNKS = {
    "var_a": {"var_no_mesh": variant("no_mesh"), "var_contact_aware": variant("contact_aware")},
    "var_b": {"var_keypoint_contact": variant("keypoint_contact"),
              "var_keypoint_contact_w100": variant("keypoint_contact_w100")},
    "var_c": {"var_keypoint_contact_it11": variant("keypoint_contact_it11")},
    "shortcuts_fast": {f"sc_{m}": shortcut(m) for m in ["zero_q", "impossible_q", "nan", "malformed", "empty",
                                                         "constant_pose"]},
    "shortcuts_kik": {f"sc_{m}": shortcut(m) for m in ["disconnected_keypoints", "score_forgery",
                                                        "joint_violation", "velocity_violation"]},
    "shortcuts_ref": {f"sc_{m}": shortcut(m) for m in ["partial_missing", "omit_failed_experiment"]},
    "tamper": {},  # shipped methods only, with tampered public state
    "select_best": {"sc_select_best": shortcut("select_best")},
}
TRAIN_CHUNKS = {
    "train_var_a": {"var_no_mesh": variant("no_mesh"), "var_contact_aware": variant("contact_aware"),
                    "sc_constant_pose": shortcut("constant_pose")},
    "train_var_b": {"var_keypoint_contact": variant("keypoint_contact"),
                    "var_keypoint_contact_w100": variant("keypoint_contact_w100"),
                    "var_keypoint_contact_it11": variant("keypoint_contact_it11")},
    "train_select_best": {"sc_select_best": shortcut("select_best")},
}


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


def save(dest: Path, files) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = ROOT / "scoring" / f
        if src.exists():
            shutil.copy2(src, dest / f)


def restore_state() -> None:
    for p in REG.glob("*.sh"):
        if p.stem.startswith(("var_", "sc_")):
            p.unlink()
    if SHIPPED_BACKUP.exists():
        for p in SHIPPED_BACKUP.glob("*.sh"):
            shutil.copy2(p, REG / p.name)
        shutil.rmtree(SHIPPED_BACKUP)
    if TAMPER_MARK.exists():
        shutil.rmtree(ROOT / "data/tasks/train", ignore_errors=True)
        subprocess.run([sys.executable, "data/task_generator.py", "--out", "data/tasks/train"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, check=True)
        TAMPER_MARK.unlink()
    for f in SCORE_FILES:
        src = OUT / "official" / f
        if src.exists():
            shutil.copy2(src, ROOT / "scoring" / f)


def official() -> None:
    d = OUT / "official"
    if (d / "exit_codes.json").exists() and json.loads((d / "exit_codes.json").read_text()) == {"train": 0, "test": 0}:
        print("[official] already done", flush=True)
        return
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    keep = {"RETARGET_KEEP_OUTPUTS": str(d / "outputs")}
    status = {"train": sh("bash scripts/evaluate_train.sh", OUT / "official_train.log", keep),
              "test": sh("bash scripts/evaluate_test.sh", OUT / "official_test.log", keep)}
    save(d, SCORE_FILES)
    (d / "exit_codes.json").write_text(json.dumps(status))


def tamper_public_state() -> None:
    TAMPER_MARK.write_text("1")
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


def run_chunk(name: str, methods: dict, split: str) -> None:
    d = CHUNKS_DIR / name
    if (d / "done").exists():
        print(f"[{name}] already done", flush=True)
        return
    shutil.rmtree(d, ignore_errors=True)
    restore_state()
    try:
        if name != "tamper":
            SHIPPED_BACKUP.mkdir(parents=True, exist_ok=True)
            for s in SHIPPED:
                shutil.move(str(REG / f"{s}.sh"), SHIPPED_BACKUP / f"{s}.sh")
        else:
            tamper_public_state()
        for n, body in methods.items():
            register(n, body)
        code = sh(f"bash scripts/evaluate_{split}.sh", OUT / f"chunk_{name}.log",
                  {"RETARGET_KEEP_OUTPUTS": str(OUT / "combined_outputs")})
        save(d, [f"scores_{split}.json", f"diagnostics_{split}.json"])
        (d / "exit_code.txt").write_text(str(code))
        (d / "done").write_text("1")
    finally:
        restore_state()


def merge(split: str, chunks: list[str]) -> None:
    """Combine the official run and the chunk runs of one split into one score/diagnostics pair."""
    dest = OUT / f"combined_{split}"
    dest.mkdir(parents=True, exist_ok=True)
    scores = json.loads((OUT / "official" / f"scores_{split}.json").read_text())
    diag = json.loads((OUT / "official" / f"diagnostics_{split}.json").read_text())
    sources = {m: "official" for m in diag["methods"]}
    for c in chunks:
        d = CHUNKS_DIR / c
        if not (d / "done").exists():
            continue
        cs = json.loads((d / f"scores_{split}.json").read_text())
        cd = json.loads((d / f"diagnostics_{split}.json").read_text())
        for m, v in cd["methods"].items():
            if c == "tamper":  # shipped methods rerun under tampering: kept as evidence, official values ranked
                continue
            diag["methods"][m] = v
            sources[m] = c
            for exp in cs["experiments"]:
                scores["experiments"][exp]["results"][m] = cs["experiments"][exp]["results"][m]
    diag["sources"] = sources
    (dest / f"scores_{split}.json").write_text(json.dumps(scores, indent=2))
    (dest / f"diagnostics_{split}.json").write_text(json.dumps(diag, indent=2))
    if split == "test" and (CHUNKS_DIR / "tamper" / "done").exists():
        t = json.loads((CHUNKS_DIR / "tamper" / "scores_test.json").read_text())["experiments"]
        o = scores["experiments"]
        same = {m: all(t[e]["results"][m] == o[e]["results"][m] for e in o) for m in SHIPPED}
        (dest / "tamper_check.json").write_text(json.dumps({"identical_to_official_under_tampering": same}, indent=2))
        print("[tamper] shipped scores identical to official:", same, flush=True)


def combined() -> None:
    for name, methods in TEST_CHUNKS.items():
        run_chunk(name, methods, "test")
    merge("test", list(TEST_CHUNKS))
    for name, methods in TRAIN_CHUNKS.items():
        run_chunk(name, methods, "train")
    merge("train", list(TRAIN_CHUNKS))


def budget() -> None:
    """The counted CPU budget and the hang guard, exercised directly with tiny limits."""
    if (OUT / "budget_checks.json").exists():
        print("[budget] already done", flush=True)
        return
    sys.path.insert(0, str(ROOT))
    from eval.test import scoring_core as sc
    work = Path(tempfile.mkdtemp(prefix="budget_"))
    cases = {
        "cpu_burn": "python -c 'while True: pass'",
        "cpu_burn_in_child": "bash -c \"python -c 'while True: pass' & wait\"",
        "sleep_forever": "sleep 100000",
        "quick_ok": "python -c 'print(1)'",
    }
    out = {"budget_cpu_s": 5.0, "idle_poll_limit": 10}
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
    restore_state()
    if stage in ("all", "official"):
        official()
    if stage in ("all", "combined"):
        combined()
    if stage in ("all", "budget"):
        budget()


if __name__ == "__main__":
    main()
