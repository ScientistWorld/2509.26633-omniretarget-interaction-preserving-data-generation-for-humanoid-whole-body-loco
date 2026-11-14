"""Rank differences vs the reference for every method in a combined hidden-test run.

rank(c) - rank(ref) = sum_e w_e sum_m |coef_m| * dir_m * (c_m - ref_m) / sd_{e,m}
with dir_m = +1 for higher-is-better metrics and -1 otherwise.  Two normalizers:

  cohort6    population sd across six honest methods (reference, keypoint-IK baseline,
             no-mesh ablation, contact-aware, keypoint+contact, constant pose), as in iteration 11
  nearref    population sd across the reference and the honest variants only (no baseline,
             no constant pose): the case where the ranked attempts are all close to the reference

Paired 95% intervals: test instances are resampled within each experiment, the same resample for
both methods.  Also writes the per-metric contributions of selected comparisons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
REF = "omniretarget"
COHORTS = {
    "cohort6": ["omniretarget", "keypoint_ik", "var_no_mesh", "var_contact_aware", "var_keypoint_contact",
                "sc_constant_pose"],
    "nearref": ["omniretarget", "var_no_mesh", "var_contact_aware", "var_keypoint_contact",
                "var_keypoint_contact_w100", "var_keypoint_contact_it11"],
}
BREAKDOWN = ["var_keypoint_contact", "var_keypoint_contact_w100", "var_no_mesh", "var_contact_aware", "sc_select_best"]


def load(run_dir: Path, split: str):
    scores = json.loads((run_dir / f"scores_{split}.json").read_text())["experiments"]
    diag = json.loads((run_dir / f"diagnostics_{split}.json").read_text())["methods"]
    return scores, diag


def contributions(vals_c, vals_r, sds, reference):
    out = {}
    for exp, spec in reference["experiments"].items():
        for m, md in spec["metrics"].items():
            sd = sds[exp][m]
            if sd <= 0:
                continue
            sgn = 1.0 if md["higher_is_better"] else -1.0
            out[f"{exp}/{m}"] = spec["weight"] * abs(md["coefficient"]) * sgn * (vals_c[exp][m] - vals_r[exp][m]) / sd
    return out


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "redteam/hardening/results/combined_test"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    reference = json.loads((ROOT / "scoring/reference.json").read_text())
    exps = list(reference["experiments"])
    scores, diag = load(run_dir, split)
    methods = sorted(diag)
    vals = {name: {exp: scores[exp]["results"][name] for exp in exps} for name in methods}
    per_ref = diag[REF]["per_instance"]
    ids = {exp: sorted(i for i, v in per_ref.items() if v["experiment"] == exp) for exp in exps}
    rng = np.random.default_rng(0)
    picks = {exp: rng.integers(0, len(ids[exp]), size=(2000, len(ids[exp]))) for exp in exps}
    out = {"split": split, "normalizers": {}, "methods": {}}
    for cname, cohort in COHORTS.items():
        honest = [m for m in cohort if m in diag and diag[m]["status"] == "ok"]
        sds = {exp: {m: float(np.std([vals[h][exp][m] for h in honest])) for m in reference["experiments"][exp]["metrics"]}
               for exp in exps}
        out["normalizers"][cname] = {"methods": honest, "sd": sds}
        for name in methods:
            entry = out["methods"].setdefault(name, {"status": diag[name]["status"], "reason": diag[name]["reason"],
                                                     "cpu_s": diag[name].get("cpu_s"), "metrics": vals[name]})
            contrib = contributions(vals[name], vals[REF], sds, reference)
            entry[f"rank_delta_{cname}"] = float(sum(contrib.values()))
            if name in BREAKDOWN:
                entry[f"contributions_{cname}"] = {k: round(v, 4) for k, v in contrib.items() if abs(v) > 1e-4}
            if diag[name]["status"] == "ok" and name != REF:
                per_c = diag[name]["per_instance"]
                boots = np.zeros(2000)
                for exp, spec in reference["experiments"].items():
                    for m, md in spec["metrics"].items():
                        sd = sds[exp][m]
                        if sd <= 0:
                            continue
                        diff = np.array([per_c[i][m] - per_ref[i][m] for i in ids[exp]])
                        sgn = 1.0 if md["higher_is_better"] else -1.0
                        boots += spec["weight"] * abs(md["coefficient"]) * sgn * diff[picks[exp]].mean(axis=1) / sd
                boots.sort()
                entry[f"ci95_{cname}"] = [float(boots[50]), float(boots[1949])]
    (run_dir / f"summary_{split}.json").write_text(json.dumps(out, indent=2))
    lines = [f"Split: {split}. Rank delta vs `{REF}` (paired 95% CI).", "",
             "| method | status | CPU s | cohort6 | 95% CI | nearref | 95% CI |", "|---|---|---:|---:|---|---:|---|"]
    fmt = lambda ci: f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""  # noqa: E731
    for name, e in sorted(out["methods"].items(), key=lambda kv: -kv[1]["rank_delta_cohort6"]):
        lines.append(f"| {name} | {e['status']} {e['reason']} | {e['cpu_s']} | {e['rank_delta_cohort6']:+.3f} | "
                     f"{fmt(e.get('ci95_cohort6'))} | {e['rank_delta_nearref']:+.3f} | {fmt(e.get('ci95_nearref'))} |")
    lines += ["", "Per-experiment metric means:", ""]
    mets = list(next(iter(reference["experiments"].values()))["metrics"])
    for exp in exps:
        lines += [f"**{exp}**", "", "| method | " + " | ".join(mets) + " |", "|---" * (len(mets) + 1) + "|"]
        for name in methods:
            if diag[name]["status"] != "ok":
                continue
            lines.append(f"| {name} | " + " | ".join(f"{vals[name][exp][m]:.4f}" for m in mets) + " |")
        lines.append("")
    (run_dir / f"summary_{split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
