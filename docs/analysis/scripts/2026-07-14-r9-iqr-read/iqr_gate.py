#!/usr/bin/env python3
"""Validation gate (post-pass match): CF-reconstructed c vs reward_components.seed_contribution
on fresh n=1 epochs. Ordering-robust: collect both streams over a prefix, then intersect keys.

Also reports: regime stamp, n-distribution, value recoverability, and the stage distribution of
non-null seed_contribution (to confirm the overlap sits on active-seed decisions).
"""
import json
import sys
import math
from collections import defaultdict, Counter

sys.path.insert(0, "/tmp/claude-1000/-home-john-esper-lite/279590eb-15ce-4785-b4c2-78ac9a9d9dd7/scratchpad")
from iqr_read import reconstruct_contributions, assert_regime, is_finite_num  # noqa


def gate(path, label, gate_lines):
    slot_seed = defaultdict(dict)
    pending = defaultdict(dict)
    fresh = {}      # (env,ep,epoch,slot) -> (c, n)  recoverable fresh measurements
    la = {}         # (env,ep,epoch,slot) -> (seed_contribution, seed_stage, val_acc)
    n_hist = Counter(); strat_hist = Counter(); recov = Counter()
    la_stage_nonnull = Counter()
    regime = None; n_lines = 0; n_cf = 0; la_total = 0; la_nonnull = 0

    with open(path) as f:
        for line in f:
            n_lines += 1
            if n_lines > gate_lines:
                break
            if regime is None and '"TRAINING_STARTED"' in line:
                e = json.loads(line)
                if e.get("event_type") == "TRAINING_STARTED":
                    regime = e["data"]; continue

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
                p = (env, ep); epoch = e.get("epoch")
                for slot, (c, ok, n) in pending.get(p, {}).items():
                    if ok:
                        fresh[(env, ep, epoch, slot)] = (c, n)
                pending[p] = {}

            elif '"ANALYTICS_SNAPSHOT"' in line and '"last_action"' in line:
                e = json.loads(line)
                if e.get("event_type") != "ANALYTICS_SNAPSHOT":
                    continue
                d = e["data"]
                if d.get("kind") != "last_action":
                    continue
                la_total += 1
                rc = d.get("reward_components")
                if not isinstance(rc, dict) or rc.get("seed_contribution") is None:
                    continue
                la_nonnull += 1
                la_stage_nonnull[rc.get("seed_stage")] += 1
                env = d.get("env_id"); ep = d.get("episode_idx")
                epoch = d.get("inner_epoch"); slot = d.get("slot_id")
                if None in (env, ep, epoch, slot):
                    continue
                la[(env, ep, epoch, slot)] = (float(rc["seed_contribution"]),
                                              rc.get("seed_stage"), rc.get("val_acc"))

            elif '"SEED_STAGE_CHANGED"' in line or '"SEED_GERMINATED"' in line:
                e = json.loads(line)
                if e.get("event_type") not in ("SEED_STAGE_CHANGED", "SEED_GERMINATED"):
                    continue
                d = e["data"]; env = d.get("env_id"); ep = d.get("episode_idx")
                slot = d.get("slot_id"); sid = e.get("seed_id")
                if None in (env, ep, slot, sid):
                    continue
                slot_seed[(env, ep)][slot] = sid

    stamp = assert_regime(regime, label)

    # post-pass match on n=1 fresh + non-null seed_contribution, same key
    matches = []
    for key, (c, n) in fresh.items():
        if n != 1:
            continue
        rec = la.get(key)
        if rec is None:
            continue
        matches.append((c, rec[0], rec[1], rec[2]))

    diffs = [abs(c - sc) for (c, sc, st, va) in matches]
    # also try n=2/n=3 recoverable matches (LOO from full_factorial) as a bonus check
    matches_multi = []
    for key, (c, n) in fresh.items():
        if n == 1:
            continue
        rec = la.get(key)
        if rec is not None:
            matches_multi.append((c, rec[0], n, rec[1]))
    diffs_multi = [abs(c - sc) for (c, sc, n, st) in matches_multi]

    report = {
        "label": label, "n_lines": n_lines, "n_cf": n_cf,
        "regime_stamp": stamp,
        "n_hist": dict(n_hist), "strat_hist": dict(strat_hist),
        "recoverability": {f"n{n}_{'ok' if ok else 'bad'}": cc for (n, ok), cc in recov.items()},
        "n_fresh_recoverable_keys": len(fresh),
        "la_total": la_total, "la_nonnull_seed_contribution": la_nonnull,
        "la_nonnull_stage_dist": dict(la_stage_nonnull),
        "GATE_n1_matches": len(matches),
        "GATE_n1_max_abs_diff": (max(diffs) if diffs else None),
        "GATE_n1_mean_abs_diff": (sum(diffs) / len(diffs) if diffs else None),
        "GATE_n1_examples": [{"c_recon": round(c, 6), "seed_contribution": round(sc, 6),
                              "seed_stage": st, "abs_diff": round(abs(c - sc), 9)}
                             for (c, sc, st, va) in matches[:10]],
        "GATE_n1_stage_dist": dict(Counter(st for (_, _, st, _) in matches)),
        "GATE_multi_matches": len(matches_multi),
        "GATE_multi_max_abs_diff": (max(diffs_multi) if diffs_multi else None),
        "GATE_multi_examples": [{"c_recon": round(c, 6), "seed_contribution": round(sc, 6),
                                 "n": n, "seed_stage": st, "abs_diff": round(abs(c - sc), 9)}
                                for (c, sc, n, st) in matches_multi[:10]],
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    gate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
