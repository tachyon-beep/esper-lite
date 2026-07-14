# r9 HOLDING per-epoch counterfactual measurement-rate read

> **Main-session review (2026-07-14): ACCEPTED.** The mechanical grounding (a solo ablation
> config is built for every active slot on every validation pass, no rotation —
> `vectorized_trainer.py:984-992`) was independently confirmed against the same trainer block
> read during the G-LEDGER Phase-1 work (`:959-978`). The **obs-v3 indirect verification is
> RATIFIED**: no positive obs-version stamp exists in this telemetry generation, but obs
> version affects only how a measured LOO is *encoded*, never *when* the solo-eval runs, and
> every other regime stamp verified positively. (The missing positive obs-version stamp is
> filed as a telemetry-programme observation.) Analysis script preserved at
> `docs/analysis/scripts/2026-07-14-r9-measurement-rate/measure_rate.py`
> (SHA-256 `8f14063a8016356e94aa7a94495db1cf2ee16b4e42d9d45936aaf7643e0dfdf2`), with the
> per-seed JSON outputs alongside.

**Task:** esper-lite-5e9affe207 (queued pre-freeze read for the "permanence-visible" settlement,
`docs/analysis/2026-07-14-permanence-visible-preregistration.md` §2.3; PDR-0084 reversal trigger #3).
**Date:** 2026-07-14. **Mode:** read-only streaming analysis of sealed r9 archival telemetry.
**Scripts (scratchpad):** `measure_rate.py` (main streaming pass), inline eis cross-check.

---

## Headline (pooled, both seeds)

| Quantity | seed41 | seed42 | **Pooled** |
|---|---|---|---|
| HOLDING seed-epochs (denominator) | 12,953 | 13,750 | **26,703** |
| …with a VALID counterfactual measurement that epoch | 12,953 | 13,750 | **26,703** |
| **Measured fraction** | **100.000%** | **100.000%** | **100.000%** |
| Unmeasured HOLDING seed-epochs | 0 | 0 | **0** |
| Gaps (consecutive-unmeasured runs) | 0 | 0 | **0** |
| Gap length P50 / P90 / P99 / max | — / — / — / — | — / — / — / — | **0 / 0 / 0 / 0** (no gaps exist) |
| ≥5-valid-in-5 windows (count) | 183 | 163 | **346** |
| …windows fully satisfied (all 5 measured) | 183 (100%) | 163 (100%) | **346 (100%)** |
| **Implied extension-clause frequency** | ~0% | ~0% | **~0%** |

**Every single HOLDING seed-epoch in both r9 runs had a fresh valid LOO measurement.** There are
zero gaps, so the gap-length distribution is empty. Under the measurement cadence observed in r9, a
5-epoch confirmation window is **always** fully measured; the settlement's insufficient-measurements
extension clause (§2.3) would be the **rare/never** path, **not** the common path. **The r9 evidence
does not support firing PDR-0084 reversal trigger #3** (subject to the pinning caveat below — this is
evidence against the trigger, conditioned on the frozen-alpha bridge assumption, not an unconditional
clearance).

**Strength of the "zero" (rule of three):** 0 unmeasured in 26,703 HOLDING seed-epochs bounds the
per-epoch miss probability at **≤ ~0.011%** (95%, 3/26,703); the probability a pinned 5-epoch window
contains ≥1 miss is therefore **≤ ~0.056%**. Even at the CI upper bound the extension clause is not
the common path.

---

## Regime-stamp assertion (MANDATORY — all verified)

Extracted from the `TRAINING_STARTED` event `data` of each file (identical on both):

| Stamp | Required | seed41 | seed42 | Verdict |
|---|---|---|---|---|
| `recurrent_n_epochs` (K) | 1 | 1 | 1 | ✅ |
| `reward_mode` | shaped | shaped | shaped | ✅ |
| `chunk_length` | =max_epochs=150 | 150 | 150 | ✅ |
| `max_epochs` | 150 | 150 | 150 | ✅ |
| `per_head_advantage_norm` | False | false | false | ✅ |
| `max_seeds` | 3 | 3 | 3 | ✅ |
| `gamma` | 0.995 | 0.995 | 0.995 | ✅ |
| `task` | cifar_baseline | cifar_baseline | cifar_baseline | ✅ |
| (context) `n_envs` | — | 12 | 12 | — |
| (context) `gae_lambda` | — | 0.95 | 0.95 | — |
| (context) `reward_family` | — | contribution | contribution | — |

**obs-v3:** verified **indirectly** and reported loudly here. There is **no positive obs-version
stamp emitted** anywhere in the telemetry (no `obs_version`/`feature_schema_version` field;
`observation_stats` reports only feature-group means/stds, not dimensionality). Verification: (a)
`TRAINING_STARTED.data` contains **no `obs_v4*` / `obs_version` key** (`regime_has_obs_v4_key=false`
on both files) → the run did not override `TrainingConfig.obs_v4_contribution_state`, whose code
default is **OFF = V3** (`leyline/__init__.py:693-698`); (b) the trainer's staleness-reset site
is explicitly commented "Obs V3" (`vectorized_trainer.py:1394`). **This does not hard-STOP the read,
and obs-version is immaterial to the measured quantity:** V3 vs V4 differ only in how a measured LOO
is *encoded into the observation* (V4 adds 3 per-slot status dims), NOT in *when the solo-eval runs*.
The counterfactual computation path is identical.

Integrity: `n_epoch_completed = 1,080,000 = 7200 episodes × 150 epochs` and `n_partitions = 7200`
on **both** files — the exact full-run counts, confirming each 6.8 GB file was streamed to EOF with
no truncation. Files (sealed, unmodified; SHA-256 in `2026-07-14-r9-archival-record.md` A4):
`telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/events.jsonl`.

---

## Operationalization of "VALID measurement" (stated explicitly)

A **valid counterfactual (LOO) measurement of a HOLDING seed at `(env, episode, slot, epoch)`** is
recorded **iff that slot appears in the `slot_ids` of the `COUNTERFACTUAL_MATRIX_COMPUTED` event
belonging to that epoch, and that event's config accuracies are finite.**

Grounding (code, not inference):
- The LOO value is written **only** when a `"solo"` ablation config is evaluated for the slot:
  `new_contribution = val_acc − solo_acc`, which resets `epochs_since_counterfactual[slot] = 0` and
  calls `record_counterfactual_measurement` → `ContributionState.measured_this_epoch = True`
  (`vectorized_trainer.py:1368-1402`). This is exactly the FRESH/`measured_this_epoch` condition of
  `leyline/contribution_state.py`.
- A `"solo"` config is built for **every** active slot (`alpha>0`, non-fossilized) at **every**
  validation pass — **no rotation across slots, no per-slot interval** (`vectorized_trainer.py:984-992`).
  A HOLDING seed has `alpha=1.0`, so it is always active → always gets a solo config.
- `COUNTERFACTUAL_MATRIX_COMPUTED` is emitted whenever `active_slots` is non-empty, listing all
  measured slots (`emitters.py:187-251`). Its `slot_ids` membership is therefore the faithful
  per-epoch record of which slots were freshly measured.

**Why event-presence is faithful (not a dashboard artifact):** the emit gate is
`_should_emit("ops_normal")` → `should_collect` → `level >= TelemetryLevel.NORMAL`, a **static
deterministic boolean** — no token bucket, no rate limiter, no RNG (`telemetry_config.py:61-72`).
`EPOCH_COMPLETED` and `COUNTERFACTUAL_MATRIX_COMPUTED` share this gate, so ops_normal is collected
all-or-nothing. Empirically confirmed: `partitions_with_epoch_number_gaps = 0` (no `EPOCH_COMPLETED`
tick is ever missing within any of the 7200 episodes), and `n_cf_invalid = 0` (all counterfactual
events carry finite accuracies).

This is a related-but-DIFFERENT quantity from the banked **98–99% "decisions acting on fresh data"**
(`2026-07-13-decision-point-diagnosis.md`), which is measured *at FOSSILIZE/PRUNE decision rows*.
The present read is a **per-(env,slot,epoch) census over all HOLDING seed-epochs**, not per-decision.
**Why 100% here vs 98–99% there — expected, not a discrepancy:** the two measure different things and
the direction is predicted. The 98–99% is per-*decision* freshness at *start*-of-epoch (a decision can
act on the *prior* epoch's measurement before this epoch's solo-eval lands → a small stale tail); the
present census asks whether *this* epoch's own solo-eval ran → 100%. A per-epoch census sitting a
point or two above per-decision freshness is exactly what the mechanism predicts.

---

## Method (reconstruction)

One streaming pass per file, partitioned by `(env_id, episode_idx)` (n_envs=12 → episodes
interleaved in one file; keyed apart). A fast raw-line substring gate limits `json.loads` to the
four relevant event types.

- **Epoch clock:** `EPOCH_COMPLETED.epoch` (within-episode, 1..150). The top-level `epoch` field on
  `SEED_*` events is a **stuck constant `12`** and is ignored (verified on the env2/ep2 trace).
- **Ordering:** a `COUNTERFACTUAL_MATRIX_COMPUTED` is emitted ~1 ms **before** its epoch's
  `EPOCH_COMPLETED`; stage transitions are emitted **after** and take effect next epoch. Pending
  measurements are flushed at each `EPOCH_COMPLETED`.
- **HOLDING census (throttle-immune):** `EPOCH_COMPLETED.data.seeds[slot].stage == 'HOLDING'`.
- **Lifecycle identity:** `(env, episode, slot, seed_id)`; `seed_id` tracked from `SEED_GERMINATED`
  / `SEED_STAGE_CHANGED` top-level `seed_id`. `life_gen_unknown = 0` (every HOLDING seed-epoch had a
  resolvable generation — no orphan rows).
- **Gap:** a maximal run of consecutive unmeasured epochs within a contiguous HOLDING span. **≥5-in-5:**
  over each contiguous HOLDING span of length ≥5, every sliding 5-epoch window that has all 5 measured.

**Validation performed:**
1. Micro-validation on env2/ep2: r0c2 had a 1-epoch HOLDING span at epoch 25 (BLENDING→HOLDING after
   epoch 24, HOLDING→BLENDING after epoch 25, `epochs_in_stage=1`); the `EPOCH_COMPLETED` census read
   `r0c2=HOLDING, alpha=1.0` at epoch 25, and a counterfactual for r0c2 was present → measured. All
   three reconstructions (transition spans, census, counterfactual join) agree.
2. Independent denominator cross-check via `sum(epochs_in_stage)` at HOLDING-exit transitions (a
   separate event path): 12,918 (s41) / 13,725 (s42) vs census 12,953 / 13,750 — agree to 0.18–0.27%,
   census slightly higher exactly as expected (seeds still HOLDING at epoch-150 truncation have no
   exit transition).

---

## Contiguous HOLDING span-length distribution (context for ≥5-in-5)

| span length (epochs) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | ≥5 total | all spans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seed41 | 6160 | 1842 | 554 | 202 | 73 | 26 | 7 | 4 | 3 | 1 | 114 | 8,872 |
| seed42 | 6758 | 1965 | 605 | 177 | 48 | 31 | 10 | 3 | 1 | 1 | 94 | 9,599 |

HOLDING spans are **short** (span P50 = 1 epoch, P90 = 2, max = 10). Only ~208 of 18,471 pooled
contiguous HOLDING spans naturally reach ≥5 epochs — see caveat 1.

---

## Validity caveats — what this read structurally CANNOT tell us

1. **Unpinned cadence (the load-bearing caveat).** r9 has **no settlement / no commit**, so a seed is
   never *pinned* to HOLDING. Naturally-occurring HOLDING spans are mostly 1–2 epochs; a literal
   5-consecutive-HOLDING window forms only ~208 times (346 windows) across ~26.7k HOLDING seed-epochs.
   The 100% ≥5-in-5 figure is computed **only on that sparse, self-selected subset.** The transferable,
   robust quantity is the **per-epoch measured fraction = 100% with zero gaps** — a mechanical property
   (validation runs every epoch, solo config per active slot, no rotation). Under the settlement the
   committed seed's α is *frozen at the request instant* (§2.1) and stays ablatable to the boundary
   (§2.0), so it would remain active (α=1 → measured) every epoch of the pinned window; on that
   mechanism a pinned 5-epoch window is fully measured. **But r9 has no frozen-alpha window, so this
   remains a PROXY for the settlement's frozen-alpha confirmation condition.**
   **Named bridge assumption (state, don't assume):** r9-100%-cadence transfers to the settlement only
   if a committed/PENDING seed keeps landing in `active_slot_list` (α frozen >0, still ablated to the
   boundary). This is a spec-reading of §2.0/§2.1, not an r9 observation. It survives §2.5.7: the
   Option-A global audit lock suspends *morphology* ops during the window, but the counterfactual pass
   is driven by *validation* (host training continues), so measurement continues under the lock. The
   one implementation that would break it is a settlement that excludes pending seeds from ablation
   before the boundary — which §2.0 explicitly forbids ("stays validly measurable to the boundary").
2. **Solo-eval "rotation" is regime-specific.** In r9 there is **no rotation** — every active slot is
   measured every epoch. The §2.3 "one skipped solo-eval rotation" hypothetical does not occur here
   (zero skips observed). A future run that changed the eval cadence, or a settlement implementation
   that *excluded pending/committed seeds from ablation before the boundary*, could change the cadence;
   §2.0/§2.1 as written keep the seed measurable, but that path is untested in r9.
3. **HOLDING-only denominator.** TRAINING-stage seeds (α=0) are structurally unmeasured (100% `None`);
   they are correctly excluded — this read is about HOLDING (α=1) seeds only, where measurement is
   universal.
4. **obs-v3 verified indirectly** (no positive obs-version stamp exists); immaterial to cadence (see above).
5. **n=2 trained controllers, K=1 regime.** Findings are within the sealed r9 regime and become
   non-extendable at the first K>1 / obs-v4 run.

## Bottom line
Measurement cadence for HOLDING seeds in r9 is **100% per-epoch, zero gaps, both seeds** — the
zero-slack `min_window=5 / ≥5-valid` confirmation window is feasible and the extension clause is not
the common path. The single material caveat is that r9 cannot exhibit a *pinned/frozen-alpha* window;
the result is a mechanically-grounded proxy for that condition, not a direct observation of it.
