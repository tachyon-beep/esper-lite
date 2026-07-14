#!/usr/bin/env python3
"""r9 q_decision-IQR + near-cliff quote-noise read (task esper-lite-e12e2d1543).

READ-ONLY streaming analysis of SEALED archival telemetry. One pass per file.
Spec: iqr_read_spec.md (pre-registered before any result computed).

Usage:
  iqr_read.py <events.jsonl> <label> [--gate N | --full] [--maxlines M]

--gate N : prefix pass over first N lines: assert regime stamp, run n=1
           CF-vs-seed_contribution validation gate, report n-distribution +
           value-recoverability. No downstream stats.
--full   : full pass, emit per-seed JSON (q_decision-IQR + near-cliff sigma).

Contribution reconstruction (spec §3): c_i = val_acc(all-True config)
  − accuracy(config all-True-except-i). Mirrors vectorized_trainer.py:1379 (LOO). Exact n=1.
"""
import json
import sys
import math
from collections import defaultdict, Counter

import numpy as np

ALPHA = 1.0 / 3.0          # span-5 EWMA, adjust=False
CLIFF_PRIMARY = (0.5, 1.5)
CLIFF_SENS = (0.7, 1.3)


def is_finite_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def reconstruct_contributions(slot_ids, configs):
    """slot_id -> (c or None, recoverable). c_i = val_acc(all-True) − acc(all-True-except-i)."""
    n = len(slot_ids)
    by_mask = {}
    for c in configs:
        sm = c.get("seed_mask")
        if sm is None:
            continue
        by_mask[tuple(bool(b) for b in sm)] = c.get("accuracy")
    val_acc = by_mask.get(tuple([True] * n))
    val_ok = is_finite_num(val_acc)
    out = {}
    for i, slot in enumerate(slot_ids):
        loo = by_mask.get(tuple(j != i for j in range(n)))
        out[slot] = ((float(val_acc) - float(loo), True)
                     if (val_ok and is_finite_num(loo)) else (None, False))
    return out, n


REGIME_REQUIRED = {
    "recurrent_n_epochs": 1,
    "reward_mode": "shaped",
    "per_head_advantage_norm": False,
    "max_seeds": 3,
    "gamma": 0.995,
}


def assert_regime(regime, label):
    if regime is None:
        raise SystemExit(f"[{label}] REGIME ABORT: no TRAINING_STARTED found")
    problems = []
    for k, v in REGIME_REQUIRED.items():
        if regime.get(k) != v:
            problems.append(f"{k}={regime.get(k)!r} (want {v!r})")
    cl, me = regime.get("chunk_length"), regime.get("max_epochs")
    if not (cl == me == 150):
        problems.append(f"chunk_length/max_epochs={cl}/{me} (want 150/150)")
    obs_v4 = [k for k in regime.keys() if "obs_v4" in k]
    if obs_v4:
        problems.append(f"obs_v4 key present: {obs_v4}")
    stamp = {k: regime.get(k) for k in
             ["recurrent_n_epochs", "reward_mode", "chunk_length", "max_epochs",
              "per_head_advantage_norm", "max_seeds", "gamma", "task", "reward_family",
              "n_envs", "gae_lambda", "n_slots", "topology"]}
    stamp["has_obs_v4_key"] = bool(obs_v4)
    if problems:
        raise SystemExit(f"[{label}] REGIME ABORT (off-regime): " + "; ".join(problems)
                         + f"\n  stamp={json.dumps(stamp)}")
    return stamp


def process_file(path, gate_lines=None, max_lines=None):
    slot_seed = defaultdict(dict)        # (env,ep) -> {slot: seed_id}
    pending = defaultdict(dict)          # (env,ep) -> {slot: (c, recoverable, n)}
    life_meas = defaultdict(list)        # L -> [(epoch, c, recoverable, n)] fresh measurements
    life_hold = defaultdict(list)        # L -> [(epoch, epochs_in_stage)]
    life_meas_stage = defaultdict(list)  # L -> [(epoch, stage_at_meas)]
    n_hist = Counter(); strat_hist = Counter(); recov = Counter()
    n_cf = 0; regime = None; n_lines = 0

    gate = gate_lines is not None
    la = {}                              # (env,ep,epoch,slot) -> (seed_contribution, seed_stage, val_acc)
    gate_matches = []

    with open(path) as f:
        for line in f:
            n_lines += 1
            if gate_lines and n_lines > gate_lines:
                break
            if max_lines and n_lines > max_lines:
                break

            if regime is None and '"TRAINING_STARTED"' in line:
                e = json.loads(line)
                if e.get("event_type") == "TRAINING_STARTED":
                    regime = e["data"]
                    continue

            if '"COUNTERFACTUAL_MATRIX_COMPUTED"' in line:
                e = json.loads(line)
                if e.get("event_type") != "COUNTERFACTUAL_MATRIX_COMPUTED":
                    continue
                d = e["data"]; env = d.get("env_id"); ep = d.get("episode_idx")
                if env is None or ep is None:
                    continue
                n_cf += 1
                recon, n = reconstruct_contributions(list(d.get("slot_ids") or []),
                                                     d.get("configs") or [])
                n_hist[n] += 1; strat_hist[d.get("strategy")] += 1
                for slot, (c, ok) in recon.items():
                    pending[(env, ep)][slot] = (c, ok, n)
                    recov[(n, ok)] += 1

            elif '"EPOCH_COMPLETED"' in line:
                e = json.loads(line)
                if e.get("event_type") != "EPOCH_COMPLETED":
                    continue
                d = e["data"]; env = d.get("env_id"); ep = d.get("episode_idx")
                if env is None or ep is None:
                    continue
                p = (env, ep); epoch = e.get("epoch"); seeds = d.get("seeds") or {}
                for slot, (c, ok, n) in pending.get(p, {}).items():
                    seed_id = slot_seed[p].get(slot) or f"{slot}:UNKNOWN"
                    L = (env, ep, slot, seed_id)
                    life_meas[L].append((epoch, c, ok, n))
                    if gate and ok and n == 1:
                        rec = la.get((env, ep, epoch, slot))
                        if rec is not None:
                            gate_matches.append((c, rec[0], rec[1], rec[2]))
                pending[p] = {}
                for slot, v in seeds.items():
                    if v.get("stage") != "HOLDING":
                        continue
                    seed_id = slot_seed[p].get(slot) or f"{slot}:UNKNOWN"
                    life_hold[(env, ep, slot, seed_id)].append((epoch, v.get("epochs_in_stage")))

            elif gate and '"ANALYTICS_SNAPSHOT"' in line and '"last_action"' in line:
                e = json.loads(line)
                if e.get("event_type") != "ANALYTICS_SNAPSHOT":
                    continue
                d = e["data"]
                if d.get("kind") != "last_action":
                    continue
                rc = d.get("reward_components")
                if not isinstance(rc, dict) or rc.get("seed_contribution") is None:
                    continue
                env = d.get("env_id"); ep = d.get("episode_idx")
                epoch = d.get("inner_epoch"); slot = d.get("slot_id")
                if None in (env, ep, epoch, slot):
                    continue
                la[(env, ep, epoch, slot)] = (
                    float(rc["seed_contribution"]), rc.get("seed_stage"), rc.get("val_acc"))

            elif '"SEED_STAGE_CHANGED"' in line or '"SEED_GERMINATED"' in line:
                e = json.loads(line)
                if e.get("event_type") not in ("SEED_STAGE_CHANGED", "SEED_GERMINATED"):
                    continue
                d = e["data"]; env = d.get("env_id"); ep = d.get("episode_idx")
                slot = d.get("slot_id"); sid = e.get("seed_id")
                if None in (env, ep, slot, sid):
                    continue
                slot_seed[(env, ep)][slot] = sid

    return {
        "regime": regime, "n_lines": n_lines, "n_cf": n_cf,
        "n_hist": dict(n_hist), "strat_hist": dict(strat_hist),
        "recov": {f"n{n}_{'ok' if ok else 'bad'}": c for (n, ok), c in recov.items()},
        "life_meas": life_meas, "life_hold": life_hold, "life_meas_stage": life_meas_stage,
        "gate_matches": gate_matches,
    }


# ---------------- EWMA + q_decision ----------------
def ewma_stream(values):
    """adjust=False EWMA seeded at first value. Returns list s_k aligned to values."""
    out = []
    s = None
    for v in values:
        s = v if s is None else ALPHA * v + (1 - ALPHA) * s
        out.append(s)
    return out


def build_quotes(life_meas, life_hold, min_prior=1):
    """For each lifecycle, EWMA over full-lifecycle recoverable measurement stream (epoch-ordered).
    q_decision(t) = EWMA value from measurements at epochs < t. Return list of dicts per eligible
    HOLDING epoch with a valid quote."""
    quotes = []  # dict(seed_run filled by caller): q, epoch, epochs_in_stage, n_prior, L
    for L, hold in life_hold.items():
        meas = sorted([(ep, c) for (ep, c, ok, n) in life_meas.get(L, []) if ok and c is not None],
                      key=lambda x: x[0])
        if not meas:
            continue
        m_epochs = [m[0] for m in meas]
        m_vals = [m[1] for m in meas]
        s = ewma_stream(m_vals)  # s[k] = EWMA including measurement k
        for (epoch, eis) in hold:
            # measurements strictly before epoch
            k = _last_before(m_epochs, epoch)
            if k < 0:
                continue
            n_prior = k + 1
            if n_prior < min_prior:
                continue
            quotes.append({"q": s[k], "epoch": epoch, "epochs_in_stage": eis,
                           "n_prior": n_prior, "L": L})
    return quotes


def _last_before(sorted_epochs, epoch):
    """index of last element strictly < epoch, or -1."""
    lo, hi = 0, len(sorted_epochs)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_epochs[mid] < epoch:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def pctls(vals):
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    p = lambda q: float(np.percentile(a, q))  # method='linear' default
    return {"n": int(a.size), "p10": p(10), "p25": p(25), "p50": p(50),
            "p75": p(75), "p90": p(90), "iqr": p(75) - p(25), "idr_10_90": p(90) - p(10),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


# ---------------- Quantity 2: near-cliff window sigma ----------------
def near_cliff_sigmas(life_meas, life_hold, band):
    """Per HOLDING span (>=3 fresh recoverable measured c), rolling <=5 windows; keep windows with
    mean in band; window sigma = sample SD (ddof=1). Return list of sigmas + fraction thresholds."""
    sigmas = []
    lo, hi = band
    for L, hold in life_hold.items():
        hold_epochs = set(ep for ep, _ in hold)
        if not hold_epochs:
            continue
        # measured c's at HOLDING epochs, epoch-ordered
        mc = sorted([(ep, c) for (ep, c, ok, n) in life_meas.get(L, [])
                     if ok and c is not None and ep in hold_epochs], key=lambda x: x[0])
        if len(mc) < 3:
            continue
        # split into integer-contiguous HOLDING spans
        for run in _contig_runs(mc):
            vals = [c for _, c in run]
            if len(vals) < 3:
                continue
            w = min(5, len(vals))
            for i in range(0, len(vals) - w + 1):
                win = vals[i:i + w]
                mu = float(np.mean(win))
                if lo <= mu <= hi:
                    sigmas.append(float(np.std(win, ddof=1)))
    return sigmas


def _contig_runs(mc):
    """mc = sorted [(epoch, c)]; split into maximal integer-consecutive-epoch runs."""
    runs = []; cur = []; prev = None
    for ep, c in mc:
        if prev is not None and ep != prev + 1:
            runs.append(cur); cur = []
        cur.append((ep, c)); prev = ep
    if cur:
        runs.append(cur)
    return runs


def frac_ge(sigmas, thr):
    if not sigmas:
        return None
    return sum(1 for s in sigmas if s >= thr) / len(sigmas)


if __name__ == "__main__":
    path = sys.argv[1]; label = sys.argv[2]
    mode = "--full"
    gate_lines = None; max_lines = None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--gate":
            mode = "--gate"; gate_lines = int(args[i + 1]); i += 2
        elif args[i] == "--full":
            mode = "--full"; i += 1
        elif args[i] == "--maxlines":
            max_lines = int(args[i + 1]); i += 2
        else:
            i += 1

    res = process_file(path, gate_lines=gate_lines, max_lines=max_lines)
    stamp = assert_regime(res["regime"], label)

    if mode == "--gate":
        gm = res["gate_matches"]
        diffs = [abs(c - sc) for (c, sc, st, va) in gm]
        report = {
            "label": label, "mode": "gate", "n_lines": res["n_lines"], "n_cf": res["n_cf"],
            "regime_stamp": stamp, "n_hist": res["n_hist"], "strat_hist": res["strat_hist"],
            "recoverability": res["recov"],
            "gate_n_matches": len(gm),
            "gate_max_abs_diff": (max(diffs) if diffs else None),
            "gate_mean_abs_diff": (sum(diffs) / len(diffs) if diffs else None),
            "gate_examples": [
                {"c_recon": round(c, 6), "seed_contribution": round(sc, 6),
                 "seed_stage": st, "abs_diff": round(abs(c - sc), 9)}
                for (c, sc, st, va) in gm[:8]],
            "gate_stage_counts": dict(Counter(st for (_, _, st, _) in gm)),
        }
        print(json.dumps(report, indent=2, default=str))
    else:
        # full stats
        q1 = build_quotes(res["life_meas"], res["life_hold"], min_prior=1)
        q3 = build_quotes(res["life_meas"], res["life_hold"], min_prior=3)

        def quote_block(quotes):
            vals = [x["q"] for x in quotes]
            L_set = set(x["L"] for x in quotes)
            block = pctls(vals)
            block["n_lifecycles"] = len(L_set)
            block["n_distinct_quote_values"] = len(set(round(v, 9) for v in vals))
            return block

        # strata
        def stratum(quotes, pred):
            return quote_block([x for x in quotes if pred(x)])

        dwell_lo = lambda x: (x["epochs_in_stage"] is not None and x["epochs_in_stage"] <= 2)
        dwell_hi = lambda x: (x["epochs_in_stage"] is not None and x["epochs_in_stage"] >= 3)
        prior_lo = lambda x: x["n_prior"] <= 2
        prior_hi = lambda x: x["n_prior"] >= 3

        sig_p = near_cliff_sigmas(res["life_meas"], res["life_hold"], CLIFF_PRIMARY)
        sig_s = near_cliff_sigmas(res["life_meas"], res["life_hold"], CLIFF_SENS)

        def sig_block(sigmas):
            b = pctls(sigmas) if sigmas else {"n": 0}
            b["frac_ge_0.1"] = frac_ge(sigmas, 0.1)
            b["frac_ge_0.2"] = frac_ge(sigmas, 0.2)
            b["frac_ge_0.3"] = frac_ge(sigmas, 0.3)
            return b

        # pre-HOLDING measurement fraction (does full-stream choice bite?)
        n_meas_total = 0; n_meas_preholding = 0
        hold_L = set(res["life_hold"].keys())
        for L, ms in res["life_meas"].items():
            hold_epochs = set(ep for ep, _ in res["life_hold"].get(L, []))
            for (ep, c, ok, n) in ms:
                if ok and c is not None:
                    n_meas_total += 1
                    if ep not in hold_epochs:
                        n_meas_preholding += 1

        report = {
            "label": label, "mode": "full", "n_lines": res["n_lines"], "n_cf": res["n_cf"],
            "regime_stamp": stamp, "n_hist": res["n_hist"], "strat_hist": res["strat_hist"],
            "recoverability": res["recov"],
            "n_holding_lifecycles": len(hold_L),
            "n_recoverable_measurements": n_meas_total,
            "frac_measurements_pre_holding": (n_meas_preholding / n_meas_total if n_meas_total else None),
            "quote_min_prior1": quote_block(q1),
            "quote_min_prior3": quote_block(q3),
            "delta_material_prior1": (0.10 / quote_block(q1)["iqr"] if quote_block(q1).get("iqr") else None),
            "delta_material_prior3": (0.10 / quote_block(q3)["iqr"] if quote_block(q3).get("iqr") else None),
            "stratum_dwell_1_2": stratum(q1, dwell_lo),
            "stratum_dwell_3plus": stratum(q1, dwell_hi),
            "stratum_prior_1_2": stratum(q1, prior_lo),
            "stratum_prior_3plus": stratum(q1, prior_hi),
            "near_cliff_sigma_primary_0.5_1.5": sig_block(sig_p),
            "near_cliff_sigma_sens_0.7_1.3": sig_block(sig_s),
        }
        # dump raw arrays for pooling across seeds
        import os
        outdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(outdir, f"arrays_{label}.json"), "w") as fh:
            json.dump({
                "q1": [{"q": x["q"], "eis": x["epochs_in_stage"], "np": x["n_prior"]} for x in q1],
                "q3": [{"q": x["q"], "eis": x["epochs_in_stage"], "np": x["n_prior"]} for x in q3],
                "sig_primary": sig_p, "sig_sens": sig_s,
            }, fh)
        print(json.dumps(report, indent=2, default=str))
