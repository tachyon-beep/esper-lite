#!/usr/bin/env python3
"""Pool seed41 + seed42 arrays into the pooled Option-A lock-burden report."""
import json
import os
from collections import Counter

import numpy as np

MORPH = ("GERMINATE", "PRUNE", "SET_ALPHA_TARGET", "FOSSILIZE")
D = os.path.dirname(os.path.abspath(__file__))


def load(label):
    with open(os.path.join(D, f"arrays_{label}.json")) as f:
        return json.load(f)


def pctls(vals):
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    p = lambda q: float(np.percentile(a, q))
    return {"n": int(a.size), "mean": float(a.mean()), "p50": p(50), "p75": p(75),
            "p90": p(90), "p99": p(99), "max": float(a.max())}


a1, a2 = load("seed41"), load("seed42")
tee = a1["total_episode_epochs"] + a2["total_episode_epochs"]
twe = a1["total_window_epochs"] + a2["total_window_epochs"]
type_totals = Counter(a1["type_totals"]); type_totals.update(a2["type_totals"])
supp = Counter(a1["suppressed_by_type"]); supp.update(a2["suppressed_by_type"])
frac_ep = a1["per_episode_locked_frac"] + a2["per_episode_locked_frac"]
qd = a1["queue_depths"] + a2["queue_depths"]
n_parts = a1["n_partitions"] + a2["n_partitions"]
any_lock = a1["n_episodes_any_locked_epoch"] + a2["n_episodes_any_locked_epoch"]

out = {
    "pooled_locked_fraction": twe / tee if tee else None,
    "total_window_epochs": twe,
    "total_episode_epochs": tee,
    "n_partitions": n_parts,
    "n_windows": a1["n_windows"] + a2["n_windows"],
    "late_masked": a1["late_masked"] + a2["late_masked"],
    "type_totals": dict(type_totals),
    "suppressed_by_type": dict(supp),
    "suppressed_frac_of_type": {
        k: (supp.get(k, 0) / type_totals[k]) if type_totals.get(k) else None for k in MORPH},
    "suppressed_hold_PRUNE": a1["suppressed_hold_PRUNE"] + a2["suppressed_hold_PRUNE"],
    "suppressed_hold_GERMINATE": a1["suppressed_hold_GERMINATE"] + a2["suppressed_hold_GERMINATE"],
    "frac_supp_PRUNE_holding": (
        (a1["suppressed_hold_PRUNE"] + a2["suppressed_hold_PRUNE"]) / supp["PRUNE"]
        if supp.get("PRUNE") else None),
    "n_episodes_any_locked_epoch": any_lock,
    "frac_episodes_any_lock": any_lock / n_parts if n_parts else None,
    "per_episode_locked_frac_pctls": pctls(frac_ep),
    "queue_depth_distribution": dict(sorted(Counter(qd).items())),
    "sum_queued_fossilizes": int(sum(qd)),
    "consistency_queued_eq_suppFOSS": int(sum(qd)) == supp.get("FOSSILIZE", 0),
    ">25pct_escalation_fires": (twe / tee) > 0.25 if tee else None,
}
with open(os.path.join(D, "out_pooled.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
