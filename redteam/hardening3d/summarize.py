"""Rank differences vs the reference for every method of a combined run (iteration 13, 3D gym).

rank(c) - rank(ref) = sum_e w_e sum_m |coef_m| * dir_m * (c_m - ref_m) / sd_{e,m}
with dir_m = +1 for higher-is-better metrics and -1 otherwise, and sd_{e,m} the population sd of
the metric across a cohort of honest methods (reference, keypoint-IK baseline, the honest variants
and the constant-pose submission).  Paired 95% intervals: instances are resampled within each
experiment (the same resample for both methods) and the experiment metrics are recomputed from the
per-instance measurements exactly as the evaluator aggregates them.

Usage: python redteam/hardening3d/summarize.py <combined_dir> <split>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.train.scoring_core import aggregate  # noqa: E402  (same aggregation as eval/test)

REF = "omniretarget"
COHORT = ["omniretarget", "keypoint_ik", "var_no_mesh", "var_contact", "var_keypoint_contact", "sc_constant_pose"]
N_BOOT = 2000


def rank_delta(vals_c, vals_r, sds, reference):
    tot, parts = 0.0, {}
    for exp, spec in reference["experiments"].items():
        for m, md in spec["metrics"].items():
            sd = sds[exp][m]
            if sd <= 0 or exp not in vals_c or exp not in vals_r:
                continue
            sgn = 1.0 if md["higher_is_better"] else -1.0
            v = float(spec["weight"]) * abs(md["coefficient"]) * sgn * (vals_c[exp][m] - vals_r[exp][m]) / sd
            parts[f"{exp}/{m}"] = v
            tot += v
    return tot, parts


def main() -> None:
    run_dir = Path(sys.argv[1])
    split = sys.argv[2]
    reference = json.loads((ROOT / "scoring/reference.json").read_text())
    scores = json.loads((run_dir / f"scores_{split}.json").read_text())["experiments"]
    diag = json.loads((run_dir / f"diagnostics_{split}.json").read_text())["methods"]
    exps = list(reference["experiments"])
    methods = sorted(diag)
    vals = {n: {e: scores[e]["results"][n] for e in exps if n in scores[e]["results"]} for n in methods}
    honest = [m for m in COHORT if m in diag and diag[m]["status"] == "ok"]
    sds = {e: {m: float(np.std([vals[h][e][m] for h in honest])) for m in reference["experiments"][e]["metrics"]}
           for e in exps}
    per_ref = diag[REF]["per_instance"]
    ids = {e: sorted(i for i, v in per_ref.items() if v["experiment"] == e) for e in exps}
    rng = np.random.default_rng(0)
    picks = {e: rng.integers(0, len(ids[e]), size=(N_BOOT, len(ids[e]))) for e in exps}

    def boot_vals(per, b):
        meas, anns = {}, {}
        for e in exps:
            for k, j in enumerate(picks[e][b]):
                iid = ids[e][j]
                meas[f"{iid}#{k}"] = per[iid]
                anns[f"{iid}#{k}"] = {"experiment": e}
        return aggregate(meas, anns)

    ref_boot = [boot_vals(per_ref, b) for b in range(N_BOOT)]
    out = {"split": split, "cohort": honest, "sd": sds, "methods": {}}
    for name in methods:
        d = diag[name]
        tot, parts = rank_delta(vals[name], vals[REF], sds, reference)
        entry = {"status": d["status"], "reason": d["reason"], "cpu_s": d.get("cpu_s"), "metrics": vals[name],
                 "rank_delta": round(tot, 4),
                 "contributions": {k: round(v, 4) for k, v in parts.items() if abs(v) > 1e-3}}
        if d["status"] == "ok" and name != REF:
            per_c = d["per_instance"]
            bs = np.array([rank_delta(boot_vals(per_c, b), ref_boot[b], sds, reference)[0] for b in range(N_BOOT)])
            entry["rank_delta_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out["methods"][name] = entry
    (run_dir / f"summary_{split}.json").write_text(json.dumps(out, indent=1))
    for name, e in sorted(out["methods"].items(), key=lambda kv: -kv[1]["rank_delta"]):
        print(f"{name:28s} {e['status']:6s} rank {e['rank_delta']:+9.3f} {e.get('rank_delta_ci95', '')} "
              f"cpu {e['cpu_s']} {e['reason'][:60]}")


if __name__ == "__main__":
    main()
