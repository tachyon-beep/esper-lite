#!/usr/bin/env python3
"""Option-A global audit-lock intervention-burden overlay on SEALED r9 telemetry.

READ-ONLY streaming analysis. One pass per file. Task esper-lite-8af424bb66.

Model (PDR-0090 #5, ratified-conditionally):
  While a fossilization settlement window is open (request -> boundary), ALL policy
  morphology ops (GERMINATE / PRUNE / SET_ALPHA_TARGET / additional FOSSILIZE) are
  suspended; one pending fossilization at a time. Host train+val continue.
  Boundary schedule: fixed cadence B in {10,20,...,150}; a request at epoch R settles
  at first B >= R+5 (min_window=5). Measurement cadence 100% (PDR-0087) => no extension.

Overlay (first-order; r9 had NO lock -> suppressed ops WOULD have changed the stream):
  Per (env_id, episode_idx) partition, walk decisions in epoch order (1 decision/epoch).
  - close an open window once a decision at epoch e > B is reached.
  - if window open: morphology decisions counted WOULD-BE-SUPPRESSED by type;
    a FOSSILIZE during-window increments this window's queue_depth (serialization);
    PRUNE/GERMINATE with seed_stage==HOLDING(6) => consequential-overlap.
  - if no window open and a successful FOSSILIZE: B=ceil((R+5)/10)*10; B>150 -> late-masked
    (no window); else open window [R+1, B], locked epochs = min(B,E_max) - R.
  queue_depth := # additional FOSSILIZE requests beyond the opener during the window
  (opener + queue_depth = commits contending for that one settlement slot).

Usage:
  lock_burden.py <events.jsonl> <label>     # full pass -> out_<label>.json + arrays_<label>.json
  lock_burden.py --selftest                 # state-machine unit test (no data)
"""
import json
import math
import os
import sys
from collections import defaultdict, Counter

HOLDING = 6                       # SeedStage.HOLDING (leyline/stages.py)
MORPH = ("GERMINATE", "PRUNE", "SET_ALPHA_TARGET", "FOSSILIZE")
W_SETTLE = 10
MIN_WINDOW = 5
MAX_EPOCH = 150

REGIME_REQUIRED = {
    "recurrent_n_epochs": 1,
    "reward_mode": "shaped",
    "per_head_advantage_norm": False,
    "max_seeds": 3,
    "gamma": 0.995,
    "task": "cifar_baseline",
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
              "n_envs", "gae_lambda"]}
    if problems:
        raise SystemExit(f"[{label}] REGIME ABORT (off-regime): " + "; ".join(problems)
                         + f"\n  stamp={json.dumps(stamp)}")
    return stamp


def boundary(R):
    """First multiple of W_SETTLE that is >= R + MIN_WINDOW."""
    return int(math.ceil((R + MIN_WINDOW) / W_SETTLE) * W_SETTLE)


def simulate_partition(decisions):
    """decisions: list of (epoch, action, success, stage). Returns per-partition overlay dict."""
    decisions.sort(key=lambda d: d[0])
    epochs = [d[0] for d in decisions]
    n_dup = len(epochs) - len(set(epochs))
    e_max = epochs[-1] if epochs else 0
    n_epochs = len(set(epochs))               # episode-epochs (denominator)

    window_open = False
    B = None
    queue_depth = 0
    window_epochs_total = 0
    n_windows = 0
    queue_depths = []                         # per-window additional-fossilize counts
    supp = Counter()                          # suppressed by type
    supp_hold = Counter()                     # suppressed PRUNE/GERMINATE targeting HOLDING
    late_masked = 0

    for (e, a, succ, stage) in decisions:
        if window_open and e > B:
            queue_depths.append(queue_depth)  # bank on close
            window_open = False
            B = None
            queue_depth = 0
        if window_open:
            if a in MORPH:
                supp[a] += 1
                if a == "FOSSILIZE":
                    queue_depth += 1
                if a in ("PRUNE", "GERMINATE") and stage == HOLDING:
                    supp_hold[a] += 1
            # WAIT and any non-morphology decision: not suppressed
        else:
            if a == "FOSSILIZE" and succ:
                Bcand = boundary(e)
                if Bcand > MAX_EPOCH:
                    late_masked += 1
                else:
                    window_open = True
                    B = Bcand
                    queue_depth = 0
                    n_windows += 1
                    locked = min(B, e_max) - e   # epochs R+1..min(B,E_max)
                    window_epochs_total += max(0, locked)
            # non-fossilize morphology when unlocked: proceeds normally (not counted)

    if window_open:                            # BLOCKER-1 fix: flush trailing open window
        queue_depths.append(queue_depth)

    return {
        "n_epochs": n_epochs,
        "e_max": e_max,
        "n_dup_epoch": n_dup,
        "window_epochs": window_epochs_total,
        "n_windows": n_windows,
        "queue_depths": queue_depths,
        "supp": dict(supp),
        "supp_hold": dict(supp_hold),
        "late_masked": late_masked,
        "locked_frac": (window_epochs_total / n_epochs) if n_epochs else 0.0,
    }


def process_file(path, label):
    regime = None
    # partition -> list of (epoch, action, success, stage)
    parts = defaultdict(list)
    type_totals = Counter()                    # total decisions of each morphology type (denominators)
    n_last_action = 0
    n_lines = 0
    success_by_type = defaultdict(Counter)

    with open(path) as f:
        for line in f:
            n_lines += 1
            if regime is None and '"TRAINING_STARTED"' in line:
                e = json.loads(line)
                if e.get("event_type") == "TRAINING_STARTED":
                    regime = e["data"]
                    continue
            if '"ANALYTICS_SNAPSHOT"' not in line or '"last_action"' not in line:
                continue
            e = json.loads(line)
            if e.get("event_type") != "ANALYTICS_SNAPSHOT":
                continue
            d = e["data"]
            if d.get("kind") != "last_action":
                continue
            env = d.get("env_id"); ep = d.get("episode_idx"); epoch = d.get("inner_epoch")
            if None in (env, ep, epoch):
                continue
            a = d.get("action_name")
            succ = bool(d.get("action_success"))
            rc = d.get("reward_components") or {}
            stage = rc.get("seed_stage")
            n_last_action += 1
            if a in MORPH:
                type_totals[a] += 1
                success_by_type[a][succ] += 1
            parts[(env, ep)].append((epoch, a, succ, stage))

    stamp = assert_regime(regime, label)

    # aggregate overlay across partitions
    agg = {
        "label": label, "path": path, "regime_stamp": stamp,
        "n_lines": n_lines, "n_last_action": n_last_action,
        "n_partitions": len(parts),
        "type_totals": dict(type_totals),
        "success_by_type": {k: dict(v) for k, v in success_by_type.items()},
    }
    total_episode_epochs = 0
    total_window_epochs = 0
    total_windows = 0
    total_late = 0
    n_dup = 0
    supp_total = Counter()
    supp_hold_total = Counter()
    per_ep_frac = []          # per-episode locked fraction
    per_ep_windowed = 0       # episodes with >=1 window opened
    per_ep_any_lock = 0       # episodes with >=1 locked epoch
    queue_depths_all = []     # per-window queue depths
    e_max_hist = Counter()

    for p, decs in parts.items():
        r = simulate_partition(decs)
        total_episode_epochs += r["n_epochs"]
        total_window_epochs += r["window_epochs"]
        total_windows += r["n_windows"]
        total_late += r["late_masked"]
        n_dup += r["n_dup_epoch"]
        for k, v in r["supp"].items():
            supp_total[k] += v
        for k, v in r["supp_hold"].items():
            supp_hold_total[k] += v
        per_ep_frac.append(r["locked_frac"])
        if r["n_windows"] > 0:
            per_ep_windowed += 1
        if r["window_epochs"] > 0:
            per_ep_any_lock += 1
        queue_depths_all.extend(r["queue_depths"])
        e_max_hist[r["e_max"]] += 1

    # consistency check: sum queued fossilizes == suppressed FOSSILIZE (second-request) count
    sum_queued = sum(queue_depths_all)
    supp_foss = supp_total.get("FOSSILIZE", 0)
    agg["CONSISTENCY_queued_eq_suppFOSS"] = (sum_queued == supp_foss)
    agg["sum_queued_fossilizes"] = sum_queued
    agg["suppressed_FOSSILIZE_second_request"] = supp_foss
    if n_dup:
        agg["WARN_duplicate_epoch_decisions"] = n_dup

    agg["total_episode_epochs"] = total_episode_epochs
    agg["total_window_epochs"] = total_window_epochs
    agg["locked_fraction_pooled"] = (total_window_epochs / total_episode_epochs) if total_episode_epochs else None
    agg["n_windows"] = total_windows
    agg["late_masked"] = total_late
    agg["suppressed_by_type"] = dict(supp_total)
    agg["suppressed_frac_of_type"] = {
        k: (supp_total.get(k, 0) / type_totals[k]) if type_totals.get(k) else None
        for k in MORPH
    }
    agg["suppressed_hold_PRUNE"] = supp_hold_total.get("PRUNE", 0)
    agg["suppressed_hold_GERMINATE"] = supp_hold_total.get("GERMINATE", 0)
    agg["frac_supp_PRUNE_holding"] = (
        supp_hold_total.get("PRUNE", 0) / supp_total["PRUNE"] if supp_total.get("PRUNE") else None)
    agg["n_episodes_with_window"] = per_ep_windowed
    agg["n_episodes_any_locked_epoch"] = per_ep_any_lock
    agg["frac_episodes_any_lock"] = (per_ep_any_lock / len(parts)) if parts else None
    agg["per_episode_locked_frac_pctls"] = pctls(per_ep_frac)
    agg["queue_depth_distribution"] = dict(sorted(Counter(queue_depths_all).items()))
    agg["e_max_hist"] = dict(sorted(e_max_hist.items()))

    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, f"arrays_{label}.json"), "w") as fh:
        json.dump({
            "label": label,
            "total_episode_epochs": total_episode_epochs,
            "total_window_epochs": total_window_epochs,
            "type_totals": dict(type_totals),
            "suppressed_by_type": dict(supp_total),
            "late_masked": total_late,
            "n_windows": total_windows,
            "suppressed_hold_PRUNE": supp_hold_total.get("PRUNE", 0),
            "suppressed_hold_GERMINATE": supp_hold_total.get("GERMINATE", 0),
            "n_partitions": len(parts),
            "n_episodes_any_locked_epoch": per_ep_any_lock,
            "per_episode_locked_frac": per_ep_frac,
            "queue_depths": queue_depths_all,
        }, fh)
    return agg


def pctls(vals):
    if not vals:
        return {}
    import numpy as np
    a = np.asarray(vals, dtype=float)
    p = lambda q: float(np.percentile(a, q))
    return {"n": int(a.size), "mean": float(a.mean()), "p50": p(50), "p75": p(75),
            "p90": p(90), "p99": p(99), "max": float(a.max())}


# ---------------------------------------------------------------- self test
def selftest():
    # single partition, epochs 1..150 assumed length via e_max
    D = [
        (1, "WAIT", True, None),
        (5, "FOSSILIZE", True, HOLDING),     # opener -> window [6,10], locked=5
        (8, "PRUNE", True, HOLDING),         # suppressed + HOLDING overlap
        (9, "FOSSILIZE", True, HOLDING),     # queued (queue_depth->1), no new window
        (10, "GERMINATE", True, None),       # suppressed (still in window, e==B)
        (11, "WAIT", True, None),            # closes window (banks qd=1)
        (12, "PRUNE", True, 3),              # unlocked -> NOT suppressed
        (147, "FOSSILIZE", True, HOLDING),   # late-masked (B=160>150)
        (150, "WAIT", True, None),           # sets e_max=150
    ]
    r = simulate_partition(list(D))
    assert boundary(5) == 10, boundary(5)
    assert boundary(6) == 20, boundary(6)
    assert boundary(145) == 150, boundary(145)
    assert boundary(146) == 160, boundary(146)
    assert r["n_windows"] == 1, r
    assert r["window_epochs"] == 5, r          # epochs 6..10
    assert r["supp"].get("PRUNE") == 1, r
    assert r["supp"].get("GERMINATE") == 1, r
    assert r["supp"].get("FOSSILIZE") == 1, r  # the queued one at epoch 9
    assert r["supp_hold"].get("PRUNE") == 1, r
    assert r["late_masked"] == 1, r
    assert r["queue_depths"] == [1], r          # banked at close
    assert abs(r["locked_frac"] - (5 / len({d[0] for d in D}))) < 1e-9, r
    # trailing-open-window flush: opener near end, window never closes before episode end
    D2 = [(140, "FOSSILIZE", True, HOLDING),    # B=150, window [141,150]
          (145, "FOSSILIZE", True, HOLDING),    # queued during open window
          (150, "WAIT", True, None)]
    r2 = simulate_partition(list(D2))
    assert r2["n_windows"] == 1, r2
    assert r2["window_epochs"] == 10, r2        # 141..150
    assert r2["queue_depths"] == [1], r2        # flushed at end (would else be lost)
    assert r2["supp"].get("FOSSILIZE") == 1, r2
    print("SELFTEST OK")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    path = sys.argv[1]; label = sys.argv[2]
    res = process_file(path, label)
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, f"out_{label}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
