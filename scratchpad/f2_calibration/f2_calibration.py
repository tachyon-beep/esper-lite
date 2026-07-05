"""F2 calibration data: (A) offline Welford reconstruction of the reward-
normalizer std trajectory (the divisor the credit meets at terminal steps),
(B) empirical gap/G priors from logged 2- and 3-slot CF factorials,
(C) normalized-currency context. Read-only. Gate criterion 2."""

import json
import sys
import time

import duckdb
import numpy as np

sys.path.insert(0, "src")
from esper.simic.rewards.committed_shapley import compute_committed_shapley_topup

GLOB = "telemetry/causal_r1_n5/control_s4[1-5]/telemetry_*/events.jsonl"
TAU = 0.28

con = duckdb.connect()
con.execute("SET threads TO 8")
t0 = time.time()

# ---- Part A: reward stream per run, replay Welford, read std at terminals ----
rows = con.execute(f"""
SELECT regexp_extract(filename, 'control_(s\\d+)', 1) run,
       json_extract(data, '$.episode_idx')::INTEGER ep,
       epoch,
       json_extract(data, '$.env_id')::INTEGER env,
       json_extract(data, '$.total_reward')::DOUBLE r
FROM read_json('{GLOB}', format='newline_delimited',
  columns={{'event_type':'VARCHAR','epoch':'BIGINT','data':'JSON'}}, filename=true,
  maximum_object_size=33554432)
WHERE event_type='ANALYTICS_SNAPSHOT'
  AND json_extract_string(data, '$.kind')='last_action'
""").fetch_df()
print(f"loaded {len(rows)} reward rows in {time.time()-t0:.0f}s")


def prefix_std(x: np.ndarray) -> np.ndarray:
    """Sample std of x[:i+1] at each i (Welford-equivalent, float64)."""
    n = np.arange(1, len(x) + 1, dtype=np.float64)
    cs, cs2 = np.cumsum(x), np.cumsum(x * x)
    mean = cs / n
    var = np.full(len(x), np.nan)
    var[1:] = np.maximum(0.0, (cs2[1:] - n[1:] * mean[1:] ** 2) / (n[1:] - 1))
    return np.sqrt(var)


print("\n== A: terminal-step running std per run (order: ep, epoch, env asc) ==")
print(f"{'run':4} {'first_term':>10} {'min_term':>9} {'p1':>7} {'p5':>7} {'p50':>7} "
      f"{'final':>7} {'ordsens%':>8}")
conv_std = {}
term_stats = {}
for run, g in rows.groupby("run"):
    g_asc = g.sort_values(["ep", "epoch", "env"], kind="mergesort")
    s_asc = prefix_std(g_asc["r"].to_numpy())
    term_mask = (g_asc["epoch"] == 150).to_numpy()
    t_asc = s_asc[term_mask]
    # order sensitivity: env descending within epoch
    g_dsc = g.sort_values(["ep", "epoch", "env"], ascending=[True, True, False],
                          kind="mergesort")
    s_dsc = prefix_std(g_dsc["r"].to_numpy())
    t_dsc = s_dsc[(g_dsc["epoch"] == 150).to_numpy()]
    sens = 100 * np.nanmax(np.abs(np.sort(t_asc) - np.sort(t_dsc)) / np.sort(t_asc))
    conv_std[run] = s_asc[-1]
    term_stats[run] = t_asc
    print(f"{run:4} {t_asc[0]:10.4f} {np.nanmin(t_asc):9.4f} "
          f"{np.nanpercentile(t_asc,1):7.4f} {np.nanpercentile(t_asc,5):7.4f} "
          f"{np.nanpercentile(t_asc,50):7.4f} {s_asc[-1]:7.4f} {sens:8.3f}")

all_term = np.concatenate(list(term_stats.values()))
print(f"POOLED terminal std: min {np.nanmin(all_term):.4f}  p1 "
      f"{np.nanpercentile(all_term,1):.4f}  p50 {np.nanpercentile(all_term,50):.4f}  "
      f"max {np.nanmax(all_term):.4f}")

# ---- Part B: factorial gap/G priors ----
t1 = time.time()
mats = con.execute(f"""
SELECT json_extract_string(data, '$.slot_ids') slot_ids,
       json_extract_string(data, '$.configs') configs
FROM read_json('{GLOB}', format='newline_delimited',
  columns={{'event_type':'VARCHAR','data':'JSON'}}, filename=true,
  maximum_object_size=33554432)
WHERE event_type='COUNTERFACTUAL_MATRIX_COMPUTED'
  AND json_array_length(data, '$.slot_ids') >= 2
""").fetchall()
print(f"\nloaded {len(mats)} multi-slot matrices in {time.time()-t1:.0f}s")

gaps2, gs2, i_half = [], [], []
res3 = []
for slot_ids_s, configs_s in mats:
    slots = json.loads(slot_ids_s)
    configs = json.loads(configs_s)
    accs = {}
    for c in configs:
        key = frozenset(s for s, on in zip(slots, c["seed_mask"]) if on)
        accs[key] = c["accuracy"]  # duplicates identical; last wins
    if len(slots) == 2:
        a, b = slots
        try:
            v00 = accs[frozenset()]; v10 = accs[frozenset({a})]
            v01 = accs[frozenset({b})]; v11 = accs[frozenset({a, b})]
        except KeyError:
            continue
        inter = v11 - v10 - v01 + v00
        i_half.append(inter / 2)
        gaps2.append(max(0.0, inter / 2 - TAU))
        gs2.append(max(0.0, v11 - v00))
    else:
        try:
            r = compute_committed_shapley_topup(
                accs, slots, scale=1.0, cap=1e9, tau=TAU)
        except ValueError:
            continue
        res3.append(r)

i_half = np.array(i_half); gaps2 = np.array(gaps2); gs2 = np.array(gs2)
pos = gaps2[gaps2 > 0]
print("== B: k=2 factorial prior (transient-seed proxy, tau=0.28, scale=1) ==")
print(f"n={len(gaps2)}  P(I/2 > tau) = {100*np.mean(gaps2>0):.1f}%")
print(f"I/2 quantiles: p50 {np.percentile(i_half,50):+.3f}  p90 "
      f"{np.percentile(i_half,90):+.3f}  p99 {np.percentile(i_half,99):+.3f}  "
      f"max {i_half.max():+.2f}")
if len(pos):
    print(f"positive gap: p50 {np.percentile(pos,50):.3f}  p90 "
          f"{np.percentile(pos,90):.3f}  p99 {np.percentile(pos,99):.3f}  "
          f"max {pos.max():.2f} pp")
    # G-clamp binding at scale=1, cap=inf: sum_raw = 2*gap vs G
    paying = gaps2 > 0
    bind = (2 * gaps2[paying]) > gs2[paying]
    print(f"G-clamp binds on {100*np.mean(bind):.1f}% of paying k=2 events "
          f"(G p50 among paying: {np.percentile(gs2[paying],50):.2f} pp)")

if res3:
    g3 = np.array([t.gap for r in res3 for t in r.per_slot.values()])
    pay3 = np.array([any(t.gap > 0 for t in r.per_slot.values()) for r in res3])
    bind3 = np.array([r.clamp_binding for r in res3 if r.sum_raw > 0])
    print(f"== k=3 ({len(res3)} matrices): P(any slot pays) = "
          f"{100*pay3.mean():.1f}%; positive gaps p50/p99 = "
          f"{np.percentile(g3[g3>0],50):.3f}/{np.percentile(g3[g3>0],99):.3f} pp; "
          f"G-clamp binds {100*bind3.mean():.1f}% of paying ==" if (g3 > 0).any()
          else f"== k=3: no positive gaps in {len(res3)} matrices ==")

# ---- Part C: normalized-currency context ----
print("\n== C: buffer-unit context (per run converged std) ==")
print(f"{'run':4} {'std':>6} {'foss_pkg~1.55':>13} {'term_bonus~2.5':>14} "
      f"{'tau=0.28':>9} {'gap_p90/std':>11}")
gp90 = np.percentile(pos, 90) if len(pos) else float("nan")
for run, sd in conv_std.items():
    print(f"{run:4} {sd:6.3f} {1.55/sd:13.2f} {2.5/sd:14.2f} {0.28/sd:9.3f} "
          f"{gp90/sd:11.2f}")
print(f"\ntotal {time.time()-t0:.0f}s")
