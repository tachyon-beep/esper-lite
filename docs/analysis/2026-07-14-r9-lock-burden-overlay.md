# Option-A global audit-lock — intervention-burden overlay on sealed r9 telemetry

> **Main-session review (2026-07-14): ACCEPTED.** Internal reconciliation is exact (4,151 FOSSILIZE
> decisions = 3,999 window-openers + 52 queued + 100 late-masked), and 4,151 independently matches the
> phase-1 PBRS scan's commit count across both seeds — two instruments, same total. The axis caution is
> correctly drawn (26.85% is episode INCIDENCE, not the epoch-fraction gate; the gate reads 1.75%). The
> first-order-overlay caveat is carried: this measures how much of r9's realized activity the lock would
> intercept, not the locked policy's dynamics. Scripts + JSON preserved under
> `docs/analysis/scripts/2026-07-14-r9-lock-burden/`.

**Task:** esper-lite-8af424bb66 · **Date:** 2026-07-14 · **Mode:** READ-ONLY streaming overlay, one pass/file.
**Data:** `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/events.jsonl` (~6.8 GB each).
**Scripts:** `lock_burden.py` (per-seed overlay + self-test), `lock_pool.py` (pool). Outputs: `out_{seed41,seed42,pooled}.json`, `arrays_*.json`.

## Regime stamp (asserted in-script; both seeds identical, on-regime)

| key | value | key | value |
|-----|-------|-----|-------|
| recurrent_n_epochs | 1 | reward_mode | shaped |
| per_head_advantage_norm | false | max_seeds | 3 |
| gamma | 0.995 | task | cifar_baseline |
| chunk_length / max_epochs | 150 / 150 | reward_family | contribution |
| n_envs | 12 | gae_lambda | 0.95 |

`obs_v4` absent. Off-regime → `SystemExit` abort. Both files passed.

## Method

Per `(env_id, episode_idx)` partition (one policy decision per epoch — confirmed: 1 `ANALYTICS_SNAPSHOT/last_action`
per `(env,ep,inner_epoch)`, `inner_epoch ∈ [1,150]` == `reward_components.epoch`), walk decisions in epoch order:

- **Boundary schedule:** fixed cadence `B ∈ {10,20,…,150}`; a FOSSILIZE request at epoch `R` settles at `B = ceil((R+5)/10)·10`
  (min_window = 5). Measurement cadence is 100% (PDR-0087) → no extension modeling.
- **Open:** a *successful* FOSSILIZE (`action_success==True`; every morphology decision in r9 is `action_success==True`)
  with **no** window open → open `[R+1, B]`, locked epochs `= min(B, E_max) − R`. If `B > 150` (R ∈ 146..150) → **late-masked** (no window).
- **While open (one pending at a time):** GERMINATE / PRUNE / SET_ALPHA_TARGET / additional FOSSILIZE → **would-be-suppressed** by type.
  A during-window FOSSILIZE also increments that window's **queue_depth** (additional commits contending for the single settlement slot).
  A suppressed PRUNE/GERMINATE with `seed_stage==HOLDING(6)` → **consequential-overlap** (`seed_stage` = the decision-carried stage of the
  target seed; `SeedStage.HOLDING==6` per `leyline/stages.py`).
- **Close** once a decision at epoch `e > B` is reached (banks queue_depth); a trailing still-open window is flushed at episode end.

**Reconciliation (both seeds):** pooled FOSSILIZE decisions 4151 = window-openers 3999 + queued-second-requests 52 + late-masked 100. ✔
**Consistency (both seeds):** Σ queued-fossilizes == suppressed-FOSSILIZE count (28==28, 24==24). ✔ Episodes run full length (E_max=150 for 14396/14400; 149 for 4).

## Results

### 1 — Locked-epoch fraction (headline)

| | seed41 | seed42 | **pooled** |
|---|---|---|---|
| episode-epochs (Σ) | 1,079,932 | 1,079,950 | 2,159,882 |
| window (locked) epochs | 18,628 | 19,269 | 37,897 |
| **locked fraction** | **1.72%** | **1.78%** | **1.75%** |
| windows opened | 1,968 | 2,031 | 3,999 |
| episodes with ≥1 lock | 26.64% | 27.07% | **26.85%** |

Per-episode locked-fraction distribution (n=14,400 pooled): **P50 = 0.0**, P75 = 0.033, P90 = 0.073, P99 = 0.093, **max = 0.187**.
(The lock is bimodal at the episode grain: ~73% of episodes never fossilize → 0% locked; among the ~27% that do, the load is a few 5–10-epoch windows.)

### 2 — Would-be-suppressed morphology ops (pooled; fraction of that op-type's run-wide total)

| op | suppressed | total | **fraction** |
|----|-----------:|------:|-----:|
| GERMINATE | 3,104 | 177,931 | 1.74% |
| PRUNE | 3,443 | 163,931 | 2.10% |
| SET_ALPHA_TARGET | 798 | 66,954 | 1.19% |
| FOSSILIZE (2nd request) | 52 | 4,151 | 1.25% |
| **late-masked FOSSILIZE requests** | **100** | 4,151 | 2.41% |

Per-seed: seed41 {G 1525, P 1700, SA 391, F2 28, late 47}; seed42 {G 1579, P 1743, SA 407, F2 24, late 53}. Highly reproducible.

### 3 — Serialization pathology (queue depth = additional commits behind one open settlement)

| queue depth | windows (pooled) |
|---|---|
| 0 | 3,947 |
| 1 | 52 |
| ≥2 | **0** |

**No window ever had more than one fossilization queued behind it** (seed41 max 1, seed42 max 1). 98.7% of windows settle with zero contention;
the remaining 1.3% have exactly one commit that would wait. Serialization is a non-issue at r9's fossilization rate.

### 4 — High-value overlap (consequential suppressed ops targeting a HOLDING seed)

- Suppressed **PRUNE targeting HOLDING**: **32 / 3,443 = 0.93%** of suppressed prunes (seed41 19, seed42 13).
- Suppressed **GERMINATE targeting HOLDING: 0** — structural: GERMINATE targets DORMANT slots, so its `seed_stage` is always null; a germinate cannot target a HOLDING seed. Report as ~0 by construction.

Consequential overlap is tiny: across the whole run the lock would have suspended 32 prunes of a HOLDING (near-fossilization-decision) seed — ~16/seed.

## Escalation check (PDR-0084 → Option-B trigger, threshold >25% of epochs locked)

**Does NOT fire.** Pooled locked fraction **1.75%** ≪ 25% (seed41 1.72%, seed42 1.78%). The trigger also fails under the
stricter *per-episode* reading: the **maximum single-episode** locked fraction is **18.7%**, P99 = 9.3% — no individual episode crosses 25% either.
Both the pooled-mean and per-episode-tail interpretations of the gate stay comfortably below threshold.

## Interpretation caveats (first-order overlay — NOT a locked-policy simulation)

1. **Counterfactual overlay on unlocked behavior.** r9 ran with no lock. Suppressed ops (and queued/late-masked fossilizes) *did* execute in r9
   and changed the subsequent decision stream; under a real lock the policy would face a different observation/reward trajectory. These are
   **first-order burden estimates** (how much of r9's realized activity the lock *would have* intercepted), not the behavior of the locked policy.
2. **Queued/late-masked fossilizes are counted once and dropped**, not re-queued into a later window (the overlay models "one pending at a time"
   but does not re-drive the suppressed commit forward). At observed rates (max queue depth 1; 52 queued + 100 late over 4,151) this is negligible.
3. **HOLDING overlap uses the decision-carried `seed_stage`** (cheap proxy) rather than the authoritative `EPOCH_COMPLETED.seeds[slot].stage`
   census. For PRUNE/FOSSILIZE/SET_ALPHA_TARGET this field is populated with the target seed's stage; adequate for a fraction estimate.
4. **`action_success` is uniformly True** for all morphology decisions in both files, so the "successful FOSSILIZE" opener filter is not load-bearing here.
5. **Two different axes — do not conflate.** The >25% gate is on **locked-epoch fraction (1.75%)**. The separate figure **26.85% of episodes touched by ≥1 lock** is *episode incidence* (a different denominator) and does **not** feed the epoch gate.
6. **Numerator/denominator basis.** Locked epochs use epoch-number arithmetic (`B−R`); the denominator `total_episode_epochs` is the decision count (1,079,932 for seed41 vs 1,079,998 for a fully contiguous run — ~66 epochs are the known EPOCH gap artifact the measure_rate read flagged). A window spanning a gap epoch is in the numerator but not the denominator → ≈+0.006% over-statement, i.e. **conservative** for the >25% question. Immaterial at a 1.75% result.
