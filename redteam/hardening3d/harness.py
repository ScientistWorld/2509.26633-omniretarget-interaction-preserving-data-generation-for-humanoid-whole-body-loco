"""Hardening audit (iteration 13, 3D gym): official runs, then honest variants and shortcut
submissions through the unchanged evaluators.

Resumable (compute nodes preempt jobs): every stage/chunk writes a done marker and is skipped when
rerun; every invocation first restores clean state (no audit registrations, shipped retargeters in
place, official score files in scoring/).

1. official  only the shipped retargeters registered; ``bash scripts/evaluate_train.sh`` and
             ``bash scripts/evaluate_test.sh`` (no arguments), nothing else running beside them.
             Results in results/official (raw outputs kept for rescoring).
2. chunks    honest variants / shortcuts registered as methods (shipped ones unregistered; every
             method is scored independently) and run through ``bash scripts/evaluate_test.sh``
             (variants also through ``evaluate_train.sh``).  Merged into results/combined_{test,train}.

Usage: python redteam/hardening3d/harness.py [official|chunks|merge|all]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/user")
REG = ROOT / "scripts" / "retargeters"
OUT = ROOT / "redteam" / "hardening3d" / "results"
CHUNKS_DIR = OUT / "chunks"
BACKUP = OUT / ".shipped_backup"
SHIPPED = ["omniretarget", "keypoint_ik"]
SCORE_FILES = ["scores_train.json", "diagnostics_train.json", "scores_test.json", "diagnostics_test.json"]


def method(flags: str) -> str:
    return f'python method/omniretarget.py --inputs "$1" --outputs "$2" {flags}'


def shortcut(m: str) -> str:
    return f'python redteam/hardening3d/shortcuts.py {m} "$1" "$2"'


VARIANTS = {
    "var_no_mesh": method("--objective keypoint"),
    "var_contact": method("--contact-weight 100"),
    "var_keypoint_contact": method("--objective keypoint --contact-weight 100"),
}
CHUNKS = [  # (chunk name, methods, splits)
    ("var_no_mesh", {"var_no_mesh": VARIANTS["var_no_mesh"]}, ("test", "train")),
    ("var_contact", {"var_contact": VARIANTS["var_contact"]}, ("test", "train")),
    ("var_keypoint_contact", {"var_keypoint_contact": VARIANTS["var_keypoint_contact"]}, ("test", "train")),
    ("sc_fast", {f"sc_{m}": shortcut(m) for m in ["constant_pose", "impossible_q", "nan", "malformed", "empty"]},
     ("test",)),
    ("sc_constant_train", {"sc_constant_pose": shortcut("constant_pose")}, ("train",)),
    ("sc_kik", {f"sc_{m}": shortcut(m) for m in ["joint_violation", "score_forgery"]}, ("test",)),
    ("sc_ref", {f"sc_{m}": shortcut(m) for m in ["partial", "velocity_violation", "unnormalized_quat",
                                                  "teleport"]}, ("test",)),
    ("sc_always_contact", {"sc_always_contact": shortcut("always_contact")}, ("test",)),
]


def sh(cmd: str, log: Path, env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env.update(env_extra or {})
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        r = subprocess.run(["bash", "-c", cmd], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"$ {cmd} -> {r.returncode}", flush=True)
    return r.returncode


def register(name: str, body: str) -> None:
    (REG / f"{name}.sh").write_text("#!/bin/bash\nset -e\ncd /home/user\n" + body + "\n")


def save(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in SCORE_FILES:
        src = ROOT / "scoring" / f
        if src.exists():
            shutil.copy2(src, dest / f)


def restore_state() -> None:
    for p in REG.glob("*.sh"):
        if p.stem.startswith(("var_", "sc_")):
            p.unlink()
    if BACKUP.exists():
        for p in BACKUP.glob("*.sh"):
            shutil.copy2(p, REG / p.name)
        shutil.rmtree(BACKUP)
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
    status = {"train": sh("bash scripts/evaluate_train.sh", d / "train.log", keep),
              "test": sh("bash scripts/evaluate_test.sh", d / "test.log", keep)}
    save(d)
    (d / "exit_codes.json").write_text(json.dumps(status))


def run_chunk(name: str, methods: dict, splits) -> None:
    d = CHUNKS_DIR / name
    if (d / "done").exists():
        print(f"[{name}] already done", flush=True)
        return
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    for s in SHIPPED:
        shutil.move(str(REG / f"{s}.sh"), BACKUP / f"{s}.sh")
    for m, body in methods.items():
        register(m, body)
    codes = {}
    keep = {"RETARGET_KEEP_OUTPUTS": str(d / "outputs")}
    for split in splits:
        codes[split] = sh(f"bash scripts/evaluate_{split}.sh", d / f"{split}.log", keep)
        for f in (f"scores_{split}.json", f"diagnostics_{split}.json"):
            if (ROOT / "scoring" / f).exists():
                shutil.copy2(ROOT / "scoring" / f, d / f)
    restore_state()
    (d / "exit_codes.json").write_text(json.dumps(codes))
    (d / "done").write_text("1")


def merge() -> None:
    for split in ("test", "train"):
        scores, diags = None, {}
        src = [OUT / "official"] + [CHUNKS_DIR / c for c, _, sp in CHUNKS if split in sp]
        for d in src:
            sf, df = d / f"scores_{split}.json", d / f"diagnostics_{split}.json"
            if not sf.exists():
                continue
            s, dg = json.loads(sf.read_text()), json.loads(df.read_text())
            if scores is None:
                scores = s
            else:
                for e, v in s["experiments"].items():
                    scores["experiments"][e]["results"].update(v["results"])
            diags.update(dg["methods"])
        if scores is None:
            continue
        out = OUT / f"combined_{split}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"scores_{split}.json").write_text(json.dumps(scores, indent=1))
        (out / f"diagnostics_{split}.json").write_text(json.dumps({"methods": diags}, indent=1))
        print(f"[merge] {split}: {sorted(diags)}", flush=True)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    restore_state()
    if stage in ("official", "all"):
        official()
    if stage in ("chunks", "all"):
        for name, methods, splits in CHUNKS:
            run_chunk(name, methods, splits)
    if stage in ("merge", "all", "chunks"):
        merge()
    restore_state()


if __name__ == "__main__":
    main()
