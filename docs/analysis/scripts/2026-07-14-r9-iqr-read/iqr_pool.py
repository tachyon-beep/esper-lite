#!/usr/bin/env python3
"""Pool seed41+seed42 quote/sigma arrays: pooled IQR_q, Delta_material, pooled strata,
pooled near-cliff sigma. numpy percentile method='linear'."""
import json
import numpy as np

D = "/tmp/claude-1000/-home-john-esper-lite/279590eb-15ce-4785-b4c2-78ac9a9d9dd7/scratchpad"
a41 = json.load(open(f"{D}/arrays_seed41.json"))
a42 = json.load(open(f"{D}/arrays_seed42.json"))


def block(vals):
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, float)
    p = lambda q: float(np.percentile(a, q))
    return {"n": int(a.size), "p10": p(10), "p25": p(25), "p50": p(50), "p75": p(75),
            "p90": p(90), "iqr": p(75) - p(25), "idr_10_90": p(90) - p(10),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "n_distinct": int(len(set(round(v, 9) for v in vals)))}


def qvals(a, key, pred=None):
    return [x["q"] for x in a[key] if (pred is None or pred(x))]


def frac_ge(vals, t):
    return (sum(1 for v in vals if v >= t) / len(vals)) if vals else None


out = {}
for name, key in [("prior1", "q1"), ("prior3", "q3")]:
    pooled = qvals(a41, key) + qvals(a42, key)
    b = block(pooled)
    b["delta_material"] = (0.10 / b["iqr"]) if b.get("iqr") else None
    b["n_lifecycles_note"] = "pooled across 2 seed-runs; per-seed n_lifecycles in per-seed JSON"
    out[f"quote_{name}_POOLED"] = b

# pooled strata on prior1
dwell_lo = lambda x: x["eis"] is not None and x["eis"] <= 2
dwell_hi = lambda x: x["eis"] is not None and x["eis"] >= 3
prior_lo = lambda x: x["np"] <= 2
prior_hi = lambda x: x["np"] >= 3
for nm, pred in [("dwell_1_2", dwell_lo), ("dwell_3plus", dwell_hi),
                 ("prior_1_2", prior_lo), ("prior_3plus", prior_hi)]:
    pooled = qvals(a41, "q1", pred) + qvals(a42, "q1", pred)
    b = block(pooled)
    b["delta_material"] = (0.10 / b["iqr"]) if b.get("iqr") else None
    out[f"stratum_{nm}_POOLED"] = b

# pooled near-cliff sigma
for nm, k in [("primary_0.5_1.5", "sig_primary"), ("sens_0.7_1.3", "sig_sens")]:
    pooled = list(a41[k]) + list(a42[k])
    b = block(pooled)
    b["frac_ge_0.1"] = frac_ge(pooled, 0.1)
    b["frac_ge_0.2"] = frac_ge(pooled, 0.2)
    b["frac_ge_0.3"] = frac_ge(pooled, 0.3)
    out[f"near_cliff_sigma_{nm}_POOLED"] = b

print(json.dumps(out, indent=2))
