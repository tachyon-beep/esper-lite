# r9 q_decision-IQR + near-cliff quote-noise — RESULTS

> **Main-session review (2026-07-14): ACCEPTED.** The dispatched agent failed with an API stall while
> composing its final message — AFTER all artifacts were written; the work is complete. Reviewed: the
> spec-first discipline held (spec durable before computation, structural peek disclosed); the
> reconstruction validation gate is the right trust anchor (186,786 bit-exact matches against the
> trainer's own `seed_contribution`); the gate-form verdict is consistent with the phase-1
> threshold-variance surface independently (at raw per-measurement σ≈0.8 pp, both the raw-EWMA and
> LCB(z=1) windfalls exceed ΔP=0.10 by the surface's own numbers). Caveats carried, esp. #1
> (dwell-weighted IQR) and #2 (132-window tail sample — direction robust, magnitude approximate).
> Scripts + per-seed JSON preserved under `docs/analysis/scripts/2026-07-14-r9-iqr-read/`
> (`iqr_read.py` SHA-256 prefix `49202bdb00a77f1d`).

Task: esper-lite-e12e2d1543 · Governing decision: PDR-0089 (ΔP_req = 0.10 RATIFIED) · Date 2026-07-14
Data: SEALED archival telemetry `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/events.jsonl`
Discipline: the pre-registered read spec (`iqr_read_spec.md`) was written and made durable **BEFORE any
result was computed or inspected** (PDR-0089 #2). Only event *schema* (field names + one event's mask
shape) was inspected pre-spec — disclosed in the spec header. Analysis is READ-ONLY, one stream per file.

Artifacts (this scratchpad): `iqr_read_spec.md` (pre-reg spec), `iqr_read.py` (extractor),
`iqr_gate.py` (validation gate), `iqr_pool.py` (pooling), `out_full_seed{41,42}.json`,
`arrays_seed{41,42}.json`, `out_pooled.json`.

---

## 1. Regime stamp — PASS (both seeds, asserted before analysis)
Extractor fails loudly off-regime. Both seeds asserted independently; both PASS.

| field | required | seed41 | seed42 |
|---|---|---|---|
| recurrent_n_epochs (K) | 1 | 1 ✓ | 1 ✓ |
| reward_mode | shaped | shaped ✓ | shaped ✓ |
| chunk_length == max_epochs | 150 == 150 | 150/150 ✓ | 150/150 ✓ |
| per_head_advantage_norm | False | False ✓ | False ✓ |
| max_seeds | 3 | 3 ✓ | 3 ✓ |
| gamma | 0.995 | 0.995 ✓ | 0.995 ✓ |
| task / reward_family | cifar_baseline / contribution | ✓ | ✓ |
| n_envs / gae_lambda | (context) | 12 / 0.95 | 12 / 0.95 |
| **any `obs_v4` key present** | **absent** | **absent ✓** | **absent ✓** |

obs-v3 has no positive stamp — verified indirectly & ratified previously (PDR); NOT gated on, only noted.
The off-regime marker is the ABSENCE of any `obs_v4` key (mirrors the prior read's `regime_has_obs_v4_key`).
**Completeness cross-check:** the read observed exactly **1,079,932** last-action decisions on seed41 —
identical to the archival record's decision count — evidence the whole 6.84 GB file was streamed to EOF.

## 2. Quote reconstruction & VALIDATION GATE (the trust anchor)
Per-epoch contribution `c = val_acc(all-True config) − accuracy(config all-True-except-s)`, i.e. the
leave-one-out marginal contribution, mirroring `vectorized_trainer.py:1379`
(`new_contribution = val_acc − baseline_accs[slot]`). Accuracies are **percent** → **c is in
percentage-points (pp)**. Fresh-measurement epochs are defined exactly as the prior r9 measurement-rate
read (slot in CF `slot_ids` with finite config accuracies).

**Gate (post-pass, ordering-robust, on a 4.33 M-line prefix):** on the clean overlap — fresh **n=1**
epochs where the same-epoch `last_action` targets that slot — CF-reconstructed `c` was compared to the
trainer's own `reward_components.seed_contribution`:

> **186,786 matches · max |Δ| = 0.0 · mean |Δ| = 0.0** (bit-exact).

This simultaneously validates sign, pp-units, and mask identification. n=1 dominates the regime
(**92 %** of CF events; max_seeds=3). **Value-recoverability = 100 %** — every fresh measurement (n=1,
n=2, n=3) had its LOO config present (n≥2 events are `full_factorial`, which emits the needed configs),
so the exclusion fraction is **≈ 0** and the `seed_contribution` fallback was **not** invoked.
(n≥2 `seed_contribution` shows occasional ~0.6 pp disagreements — that is `seed_contribution`'s
target-slot/staleness semantics, not a reconstruction error; the CF reconstruction is the primary,
internally-exact source.)

BLENDING=4, HOLDING=6 (leyline `SeedStage`). `seed_contribution` is populated only at BLENDING+HOLDING
decisions — the population of interest. Eligibility uses the throttle-immune `EPOCH_COMPLETED` HOLDING
census (string stage), independent of whether a decision fired.

---

## 3. HEADLINE — Quantity 1: q_decision IQR (the Δ_material denominator)
q_decision = lagged, event-time EWMA (span 5, α=1/3, adjust=False, seeded at first measurement) over the
seed-lifecycle's **full** fresh-measurement stream, evaluated at every eligible-HOLDING epoch with ≥1
prior measurement. Units: **pp**. No clipping/winsorization.

### Primary (≥1 prior measurement)
| | q25 | q75 | **IQR_q** | median | 10–90 IDR | n (epochs) | n_lifecycles | n_distinct_q |
|---|---|---|---|---|---|---|---|---|
| seed41 | 0.721 | 14.438 | **13.717** | 6.634 | 20.585 | 12,953 | 6,455 | 12,097 |
| seed42 | 0.786 | 13.763 | **12.977** | 6.152 | 20.207 | 13,750 | 6,648 | 12,779 |
| **POOLED** | **0.760** | **14.084** | **13.324** | **6.400** | **20.391** | **26,703** | 13,103 | 24,245 |

### **Δ_material = 0.10 / IQR_q** (ΔP_req = 0.10, owner-ratified)
| | Δ_material (per pp) | equivalently: quote move for ΔP=0.10 |
|---|---|---|
| seed41 | **0.00729** | 13.72 pp |
| seed42 | **0.00771** | 12.98 pp |
| **POOLED** | **0.00751** | **13.32 pp** |

Interpretation (per PDR-0089): a full interquartile quote move (≈13.3 pp) corresponds to a 0.10
request-probability shift; the slope is 0.10/IQR_q ≈ 0.0075 per pp. IQR_q is **NOT degenerate**
(PDR-0086 trigger does not fire).

### Sensitivity (≥3 prior measurements): pooled IQR_q = **13.574**, Δ_material = **0.00737** — essentially unchanged.

### Strata (descriptive only, pooled, primary population)
| stratum | IQR_q | median | n |
|---|---|---|---|
| HOLDING dwell 1–2 | 13.139 | 6.154 | 24,024 |
| HOLDING dwell 3+ | 14.722 | 8.270 | 2,679 |
| #prior meas 1–2 | 2.520 | 0.240 | 1,785 |
| #prior meas 3+ | 13.574 | 7.323 | 24,918 |
Dwell strata: near-identical spread. #prior-measurement strata: quotes with only 1–2 prior measurements
have a much smaller IQR (2.5) near ~0 — an EWMA warm-up artifact; the population IQR is driven by the
≥3-measurement bulk (~93 %).

---

## 4. HEADLINE — Quantity 2: within-window quote noise NEAR THE CLIFF
Per-lifecycle window-σ (ddof=1) of consecutive per-epoch contributions detrended by the ≤5-epoch window
mean, HOLDING spans with ≥3 measured contributions, windows whose mean ∈ [0.5, 1.5] pp (straddling the
**FOSSILIZE cliff = 1.0 pp**, `DEFAULT_MIN_FOSSILIZE_CONTRIBUTION = 1.0`, "minimum causal contribution
(%)"). Same pp unit as the gate-model σ.

| band | seed | P25 | **P50** | P75 | **P90** | frac σ≥0.1 | **frac σ≥0.2** | frac σ≥0.3 | n_windows |
|---|---|---|---|---|---|---|---|---|---|
| [0.5,1.5] | seed41 | 0.484 | **0.725** | 1.187 | **1.682** | 1.00 | 0.984 | 0.938 | 64 |
| [0.5,1.5] | seed42 | 0.713 | **1.010** | 1.369 | **2.079** | 1.00 | 0.985 | 0.985 | 68 |
| **[0.5,1.5]** | **POOLED** | 0.580 | **0.821** | 1.275 | **1.871** | **1.00** | **0.985** | **0.962** | **132** |
| [0.7,1.3] (sens) | POOLED | 0.604 | 0.862 | 1.213 | 1.871 | 1.00 | 0.988 | 0.963 | 81 |

## 5. GATE-FORM VERDICT (pre-registered thresholds, PDR-0089 #5)
Pre-registered rule: raw EWMA gate fails for near-cliff σ ≳ **0.15–0.2 pp**; the LCB-qualified (z=1)
variant holds only to σ ≈ **0.3 pp**.

Measured in-regime near-cliff σ (pooled): **P50 ≈ 0.82 pp, P90 ≈ 1.87 pp; 98.5 % of windows σ ≥ 0.2 pp;
96 % σ ≥ 0.3 pp.** This is **~4–5× the raw-gate failure threshold and ~2.7× the LCB-variant ceiling.**

> **VERDICT: BOTH pre-registered gate forms are insufficient at the measured noise scale.
> Raw EWMA gate FAILS; the LCB-qualified variant is ALSO breached (σ ≫ 0.3 pp for the vast majority of
> near-cliff windows). → FLAG FOR OWNER.** Under the ΔP < 0.10 materiality bound, neither pre-registered
> mechanism confines windfall pay-probability near the cliff; a stronger confirmation mechanism
> (more confirmation samples / wider margin / structural change) is indicated at freeze.

The direction is unambiguous (σ is an order of magnitude above the raw threshold, not marginal), but the
percentiles rest on a **small tail sample** (132 pooled windows) — see caveats.

---

## 6. Method notes
- Partition (env_id, episode_idx); epoch clock = `EPOCH_COMPLETED.epoch` (top-level SEED_*/CF epoch is a
  stuck constant, ignored). CF measurements flushed to the next `EPOCH_COMPLETED` epoch. Lifecycle id
  (env, ep, slot, seed_id/generation) tracked from `SEED_GERMINATED`/`SEED_STAGE_CHANGED`.
- Percentiles: numpy `method="linear"` (matches prior read). EWMA is event-time (per measurement), lagged
  to epochs strictly < t. Full-lifecycle stream (not HOLDING-only).
- 100 % value-recoverability; no fallback used; no clipping/winsorization.
- **95 % of a seed's fresh measurements occur PRE-HOLDING** (frac_measurements_pre_holding: s41 0.952,
  s42 0.946). With short HOLDING dwells (P50≈1–2 epochs, prior read), the HOLDING-decision quote is
  dominated by TRAINING/BLENDING measurements — this is why the full-lifecycle EWMA is the only viable
  choice (a HOLDING-only EWMA would be near-empty), and it means the "quote" is a pre-HOLDING-weighted
  contribution history.

## 7. Validity caveats — what this read STRUCTURALLY cannot tell us
1. **Autocorrelation / dwell-weighting (load-bearing).** q_decision is a step function that changes only
   on a new measurement and is flat between (sparse) measurements. Per-epoch pooling (as pre-registered)
   repeats the same quote across gap epochs → **IQR_q is dwell-weighted and heavily autocorrelated, not an
   i.i.d. sample**. Effective independent n ≪ 26,703 (n_lifecycles ≈ 13,103; n_distinct_q ≈ 24,245).
   Δ_material inherits this — it is calibrated against a dwell-weighted quote spread.
2. **Near-cliff σ is a small tail sample (132 windows).** Near-cliff (mean c ∈ [0.5,1.5] pp) HOLDING spans
   with ≥3 measured contributions are rare because HOLDING dwells are short and most quotes sit far above
   the 1 pp cliff (median q ≈ 6.4 pp). The verdict's DIRECTION is robust (σ ≈ 0.8 pp is ~4× the threshold),
   the exact percentiles are tail estimates.
3. **Regime analogy.** r9 is K=1 / obs-v3. The target regime is K=4 / obs-v4, whose in-regime quote noise
   MAY differ; these numbers ANCHOR the scale, they are not a guarantee for the target regime.
4. **Short-span σ limitation.** σ is estimated on the longer-dwell HOLDING tail; it may not represent the
   modal (1–2 epoch) HOLDING episode, which contributes no ≥3-point window at all.
5. **SEALED / one-shot.** r9 is non-extendable once a K>1/obs-v4 run happens; this is a calibration read,
   not a re-runnable estimator.


---

# APPENDIX A — PRE-REGISTERED READ SPECIFICATION (verbatim, written before any result)

# r9 q_decision-IQR + near-cliff noise — PRE-REGISTERED READ SPECIFICATION

Task: esper-lite-e12e2d1543. Governing decision: PDR-0089 (ΔP_req = 0.10 RATIFIED).
This file is written and durable **BEFORE any result is computed or inspected** (PDR-0089 #2,
gpt item-5 discipline). No q_decision, IQR, σ, or aggregate has been looked at at authoring time.

## 0. What was inspected pre-spec (full disclosure)
To author a correct reconstruction formula I inspected, before writing this spec:
- Source code: `emitters.py` (CF-matrix + last_action + epoch emitters), `vectorized_trainer.py`
  (contribution computation ~L1360-1402), `action_execution.py` (seed_contribution), the
  `CounterfactualMatrixPayload`/`RewardComponentsTelemetry`/`ContributionState` dataclasses,
  and the prior `measure_rate.py` read.
- The **structural shape of ONE event of each type** in seed41 (keys + one CF event's masks):
  the first CF event is n=1, `strategy="ablation_only"`, `configs` masks `[[False],[True],[True]]`.
  No analysis outcome (no quote/IQR/σ) was computed. This structural peek is required to
  pre-register a reconstruction formula against real field names; it does not reveal the read result.

## 1. Regime stamp assertion (STEP 2, runs before analysis)
Extract the first `TRAINING_STARTED.data` per file. **Fail loudly** unless ALL hold:
- `recurrent_n_epochs == 1` (K=1)
- `reward_mode == "shaped"`
- `chunk_length == max_epochs == 150`
- `per_head_advantage_norm == False`
- `max_seeds == 3`
- `gamma == 0.995`
- task/dataset == cifar_baseline (assert `task`/`reward_family`/`n_envs` match the archival record)
- **ABSENCE of any key containing `obs_v4`** in the regime dict (the off-regime marker; mirrors the
  prior script's `regime_has_obs_v4_key`). obs-v3 has no positive stamp — verified indirectly &
  ratified previously (PDR); NOT gated on here, only noted.
Any mismatch → abort with a loud error. Both seeds asserted independently.

## 2. Partition, epoch clock, lifecycle (mirrors prior measure_rate.py, verified against code)
- Partition = `(env_id, episode_idx)` from event `data`.
- Epoch clock = `EPOCH_COMPLETED.epoch` (within-episode 1..150). The top-level `epoch` on SEED_*/CF
  events is a stuck constant and is IGNORED.
- Ordering: `COUNTERFACTUAL_MATRIX_COMPUTED` is emitted immediately BEFORE its epoch's
  `EPOCH_COMPLETED`; accumulate CF measurements into `pending[partition]` and stamp them to the
  epoch of the next `EPOCH_COMPLETED` for that partition; then clear pending.
- Seed-lifecycle id `L = (env, episode, slot, seed_id)`. `seed_id` (the "generation") tracked per
  `(env,ep,slot)` from top-level `seed_id` on `SEED_GERMINATED`/`SEED_STAGE_CHANGED`. Unknown-gen
  epochs keyed `slot:UNKNOWN` and counted separately.

## 3. Per-epoch counterfactual contribution c_t — QUOTE RECONSTRUCTION FORMULA
The authoritative per-epoch contribution the trainer computes (`vectorized_trainer.py:1379`) is
`new_contribution = env_state.val_acc − baseline_accs[slot]`, where `baseline_accs[slot]` is the
**leave-one-out (LOO) ablation** accuracy (slot s disabled, all other active slots enabled;
"accuracy drop when seed is disabled", `action_execution.py:1082-1083`). Accuracies are in
**percent** (`100·correct/total`); therefore c is in **percentage points (pp)**.

A FRESH measurement of seed-lifecycle L at an epoch exists IFF L's slot appears in that epoch's CF
event `slot_ids` with finite config accuracies (identical to the prior read's measurement definition;
this is the condition that resets `epochs_since_counterfactual→0`).

**Primary value source — CF-matrix reconstruction (per active slot):** from the CF event's `configs`
(each = `{seed_mask: [bool]*n, accuracy}`, aligned to `slot_ids`):
- `val_acc` = accuracy of the config whose `seed_mask` is **all-True** (all active slots on). If more
  than one all-True config (happens at n=1, where solo≡all-on), they are equal; take one.
- For target slot at index `i`, `LOO_i` = accuracy of the config whose `seed_mask` is **True
  everywhere except index i** (popcount n−1). n=1 → the all-False (all-off) config; n=2 → the other
  slot's solo config; n=3 → the complementary pair config.
- `c_i = val_acc − LOO_i`.
This mirrors trainer L1379 exactly and is EXACT for n=1 (the dominant regime under max_seeds=3).

**Hard validation gate (run on a prefix BEFORE the full pass):** on the clean overlap — fresh
n=1 epochs where a same-epoch `last_action` targets that slot (`reward_components.seed_stage` present)
— CF-reconstructed `c_i` MUST equal `reward_components.seed_contribution` to float tolerance
(|Δ| ≤ 1e-6 · max(1,|c|)). This simultaneously validates sign, pp-units, and mask identification.
If the gate fails, the reconstruction is wrong and NO downstream number is reported; stop and diagnose.

**Missing-value / exclusion treatment (no imputation):**
- If for a fresh-measured slot the required LOO config (popcount n−1) is ABSENT from the event
  (e.g. n≥2 under `ablation_only`, which does not emit pairs), or any needed accuracy is non-finite,
  the VALUE is UNRECOVERABLE at that epoch: it is EXCLUDED from the c-stream and COUNTED.
  Report the excluded fraction of fresh HOLDING measurements. Named fallback (only invoked, and only
  if the excluded fraction is material, i.e. >5% of eligible-HOLDING quote inputs): substitute
  `reward_components.seed_contribution` for target-slot epochs. If exclusion is immaterial the
  fallback is NOT used (the extra seam is not worth it). Whether the fallback fired is reported.
- No clipping. No winsorization. No outlier removal. Raw pp values enter every statistic.

## 4. q_decision — LAGGED EWMA (pre-reg §2.3)
Per seed-lifecycle L, over the **full-lifecycle ordered stream** of its FRESH measured contributions
`c_(1),c_(2),…` (all stages, not HOLDING-only — faithful to "the seed's per-epoch counterfactual
contributions"; report how many measurements precede HOLDING so the choice's bite is visible):
- Event-time recursion, `adjust=False`, seeded at the first measurement:
  `s_1 = c_(1); s_k = α·c_(k) + (1−α)·s_(k−1)`, `α = 2/(span+1) = 2/6 = 1/3`, span = 5.
- `q_decision(t)` at an epoch t = `s_k` where k indexes the LAST measurement at an epoch **< t**
  (LAGGED: excludes epoch t's own measurement).

## 5. Eligibility population (Quantity 1)
ELIGIBLE-HOLDING decision epochs: seed-lifecycle L is in HOLDING at epoch t
(`EPOCH_COMPLETED.data.seeds[slot].stage == "HOLDING"` — throttle-immune per-(env,epoch) census,
authoritative, matches prior read). HOLDING is the FOSSILIZE legality condition; age masks do not
bite (prior reads). A VALID quote requires ≥1 prior measurement (k≥1 exists at an epoch < t).
Compute q_decision(t) at **every** eligible HOLDING epoch with a valid quote; pool per seed-run
(41, 42) and pooled. Sensitivity: also report requiring ≥3 prior measurements (k≥3).

## 6. IQR convention & reported statistics (Quantity 1)
- Percentile method: **linear interpolation between closest ranks** (numpy default, method="linear";
  identical to the prior read's `pctl`). Applied to the pooled multiset of q_decision values.
- Report per seed AND pooled: q25, q75, **IQR_q = q75 − q25**, the 10–90 **interdecile range**
  (P90 − P10), median (P50), and n (= number of eligible-HOLDING quote-epochs). ALSO report
  `n_lifecycles` (distinct HOLDING lifecycles contributing) and `n_distinct_quote_values`
  (the quote is a step function; per-epoch pooling repeats values across gap epochs).
- **Δ_material = 0.10 / IQR_q** (ΔP_req = 0.10, owner-ratified PDR-0089). Units: request-probability
  per pp of quote. Reported per seed and pooled. If IQR_q is degenerate (≈0), PDR-0086 trigger
  governs (flag; do not fabricate).

## 7. Strata (descriptive only, Quantity 1)
- By HOLDING dwell at the decision epoch: `epochs_in_stage ∈ {1,2}` vs `≥3`
  (`EPOCH_COMPLETED.data.seeds[slot].epochs_in_stage`).
- By number of prior measurements feeding the quote: `{1,2}` vs `≥3`.
Report q25/q75/IQR_q/median/n per stratum per seed and pooled.

## 8. Quantity 2 — within-window quote noise near the cliff
Uses the RAW per-epoch fresh contribution stream c_t (Section 3 values), per HOLDING seed-lifecycle.
- A "HOLDING span" = a maximal run of integer-consecutive epochs during which L is HOLDING. Restrict
  to spans with **≥3 fresh measured contributions** (short-span limitation stated as caveat).
- Rolling windows of up to 5 consecutive measured contributions (ordered by epoch). For spans with
  <5 measured points, the window is the whole set (that span's measured c's).
- Window local mean μ_w = mean of the window's c values. **Near-cliff restriction:** keep windows
  with μ_w ∈ [0.5, 1.5] pp (primary); also report [0.7, 1.3] pp (sensitivity).
- Window σ = sample SD of the window's c values (detrended by μ_w; centering does not change SD),
  **ddof = 1**.
- Report per seed and pooled: window-σ distribution P25/P50/P75/P90, and the FRACTION of near-cliff
  windows with σ ≥ 0.1 / ≥ 0.2 / ≥ 0.3 pp. Also report window count.

## 9. Gate-form verdict rule (pre-registered thresholds, PDR-0089 #5, decision #5)
Read the verdict from the DATA against the pre-registered bound — do not choose post hoc:
- If near-cliff in-regime σ (headline P50/P75) < ~0.1 pp → **raw EWMA gate within bound**; LCB
  fallback NOT forced.
- If σ ≳ 0.15–0.2 pp → **raw EWMA gate FAILS** the ΔP<0.10 bound near the cliff; the **LCB-qualified
  (z=1) variant** fires by pre-registration (it holds to σ ≈ 0.3).
- If σ ≳ 0.3 pp → even the LCB variant is stressed; flag for owner.
Report which branch the measured σ selects.

## 10. Validity caveats (what this read STRUCTURALLY cannot tell us)
1. **Autocorrelation / dwell-weighting (load-bearing).** q_decision is a step function that changes
   only on a new measurement; between (sparse, often long) measurements it is flat. Per-epoch pooling
   (as the pre-reg specifies) repeats the same quote across gap epochs, so IQR_q is **dwell-weighted
   and heavily autocorrelated, not an i.i.d. sample**. Δ_material inherits this: it is calibrated
   against a dwell-weighted quote spread, not an effective-independent one. `n_distinct_quote_values`
   and `n_lifecycles` are reported so the effective sample size is visible.
2. **Regime analogy for σ.** r9 is K=1 / obs-v3. The K=4 / obs-v4 in-regime quote noise MAY differ;
   this number ANCHORS the scale, it is not a guarantee for the target regime.
3. **Short HOLDING spans.** r9 HOLDING dwells are natural (no frozen-α pinning; P50≈1–2 epochs per
   prior read), so many spans have <3 measured points and are excluded from σ; σ is estimated on the
   longer-dwell tail, which may not represent the modal (short) HOLDING episode.
4. **n≥2 reconstruction gap.** Under `ablation_only`, LOO configs for n≥2 are not emitted; those
   fresh measurements' VALUES are unrecoverable from the CF matrix (Section 3 exclusion). The read is
   VALUE-complete only where the LOO config is present (exact for n=1). Excluded fraction reported.
5. **SEALED / non-extendable.** r9 becomes off-regime the moment a K>1/obs-v4 run happens; this is a
   one-shot calibration read, not a re-runnable estimator.

## 11. Outputs
- `iqr_read_spec.md` (this file, verbatim in the results doc).
- Analysis script(s) + per-seed JSON (raw stats).
- `r9_iqr_results.md`: spec, regime-stamp table, headline numbers (IQR_q per seed + pooled;
  Δ_material; near-cliff σ distribution + ≥0.2 fraction; gate-form verdict), method notes, caveats.
