"""Rank differences vs the reference for every method in a combined hidden-test run.

rank(c) - rank(ref) = sum_e w_e sum_m |coef_m| * dir_m * (c_m - ref_m) / sd_{e,m}
with dir_m = +1 for higher-is-better metrics and -1 otherwise, and sd the
population standard deviation of the metric across the honest methods of the
run (failed shortcuts excluded).  Paired 95% intervals come from resampling
test instances within each experiment (same resample for both methods).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
REF = "omniretarget"
HONEST = ["omniretarget", "keypoint_ik", "var_no_mesh", "var_contact_aware", "var_keypoint_contact", "sc_constant_pose"]


def load(run_dir: Path):
    scores = json.loads((run_dir / "scores_test.json").read_text())["experiments"]
    diag = json.loads((run_dir / "diagnostics_test.json").read_text())["methods"]
    return scores, diag


def rank_delta(vals_c, vals_r, sds, reference):
    total = 0.0
    for exp, spec in reference["experiments"].items():
        for m, md in spec["metrics"].items():
            sd = sds[exp][m]
            if sd <= 0:
                continue
            sgn = 1.0 if md["higher_is_better"] else -1.0
            total += spec["weight"] * abs(md["coefficient"]) * sgn * (vals_c[exp][m] - vals_r[exp][m]) / sd
    return total


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "redteam/hardening/results/combined_test"
    reference = json.loads((ROOT / "scoring/reference.json").read_text())
    scores, diag = load(run_dir)
    methods = sorted(diag)
    honest = [m for m in HONEST if m in diag and diag[m]["status"] == "ok"]
    sds = {exp: {m: float(np.std([scores[exp]["results"][h][m] for h in honest]))
                 for m in spec["metrics"]} for exp, spec in reference["experiments"].items()}
    vals = {name: {exp: scores[exp]["results"][name] for exp in reference["experiments"]} for name in methods}
    rng = np.random.default_rng(0)
    per_ref = diag[REF]["per_instance"]
    ids_by_exp = {exp: sorted(i for i, v in per_ref.items() if v["experiment"] == exp) for exp in reference["experiments"]}
    out = {"normalizers": sds, "honest_methods": honest, "methods": {}}
    for name in methods:
        d = rank_delta(vals[name], vals[REF], sds, reference)
        entry = {"status": diag[name]["status"], "reason": diag[name]["reason"],
                 "runtime_s": diag[name].get("runtime_s"), "rank_delta": d, "metrics": vals[name]}
        if diag[name]["status"] == "ok" and name != REF:
            per_c = diag[name]["per_instance"]
            boots = []
            for _ in range(2000):
                vc, vr = {}, {}
                for exp, ids in ids_by_exp.items():
                    pick = rng.choice(ids, size=len(ids), replace=True)
                    vc[exp] = {m: float(np.mean([per_c[i][m] for i in pick])) for m in reference["experiments"][exp]["metrics"]}
                    vr[exp] = {m: float(np.mean([per_ref[i][m] for i in pick])) for m in reference["experiments"][exp]["metrics"]}
                boots.append(rank_delta(vc, vr, sds, reference))
            entry["rank_delta_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            entry["rank_delta_boot_sd"] = float(np.std(boots))
            # Per-instance paired win rate on the scalar rank (each instance alone).
        out["methods"][name] = entry
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    lines = ["| method | status | rank delta vs reference | paired 95% CI |", "|---|---|---:|---|"]
    for name, e in sorted(out["methods"].items(), key=lambda kv: -kv[1]["rank_delta"]):
        ci = e.get("rank_delta_ci95")
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        lines.append(f"| {name} | {e['status']} {e['reason']} | {e['rank_delta']:+.3f} | {ci_s} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for name in methods:
        print(name, json.dumps({exp: {m: round(v, 4) for m, v in vals[name][exp].items()} for exp in vals[name]}))


if __name__ == "__main__":
    main()
