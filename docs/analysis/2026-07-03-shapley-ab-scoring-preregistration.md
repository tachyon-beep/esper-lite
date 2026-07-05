# Committed-Shapley OFF/ON A/B — scoring pre-registration (2026-07-03)

**Status:** PRE-REGISTERED before any A/B run exists. Scoring rules below are fixed;
deviations at scoring time must be flagged as such, not silently applied.
**Authority:** PDR-0019 (primary gates + effect-size floor + asymmetric null),
PDR-0018 (safety floors), F2 freeze + drl reviews (gate comments #105–#106),
plan `docs/plans/ready/2026-07-03-pre-ab-build.md`.
**Arms:** OFF = `configs/config-3slot-3seed-baseline-shaped.json` (scale=0.0);
ON = `configs/config-3slot-3seed-baseline-shaped-shapley-on.json` (scale=1.0,
tau=0.28, cap=5.0, std_floor=0.25, normalized_cap=3.0). Paired seeds 41–45,
**same commit for both arms** (PDR-0015 NUM_BLUEPRINTS constraint) — record the
launch commit hash here at launch: **`6dd80716`** (launched 2026-07-03, owner GPU-go;
working tree carried doc/tooling-only churn — no source diffs — noted for provenance).
**Launch plan:** telemetry → `telemetry/shapley_ab_n5/{off,on}_s<seed>`; per GPU, the
OFF wave runs first then the ON wave (cuda:0 = seeds {41,43,45}, cuda:1 = {42,44},
mirroring the causal_r1_n5 sweep pattern); command per run:
`uv run python -m esper.scripts.train ppo --config-json <arm-config> --seed <s>
--device cuda:<i> --gpu-preload --no-tui --telemetry-dir telemetry/shapley_ab_n5/<arm>_s<s>`.

## Primary gates — design criteria (i)–(v), operationalized

| # | Criterion (design doc §falsifiable) | Measured by | Falsified when |
|---|---|---|---|
| i | Enabling stems get top_up > 0 | `COMMITTED_SHAPLEY_TOPUP` events: top_up distribution over r0c0-fossilized slots (the suppress-confirmed cohort, PDR-0009) | r0c0 fossils systematically receive 0 while co-fossilized |
| ii | Freeloaders ≤ no-op | Null-player enforcement: no payment at `phi − c_paid ≤ tau` (structural; verify zero violations in events); ON null-player excess P99 vs tau (see ON-recalibration below) | Any payment below the deadband, or P99 > 2× placebo tau (PDR-0017 trigger → deadband method reopens) |
| iii | acc-per-param RISES and the G-clamp binds somewhere | Off-switch-J efficiency read ON vs OFF; `clamp_binding` flag count > 0 across the ON runs | Efficiency flat/down at n=5 direction level; or clamp never binds (ceiling untested) |
| iv | Commitment-avoidance no worse; no degenerate fossilize spike | fossilize/ep paired ON vs OFF (baseline 0.207±0.006); G2 line below | ON fossilize/ep < OFF (worse avoidance), or G2 trips |
| v | Learnability: the retro-written credit moves the FOSSILIZE decision | Op-head probability mass on FOSSILIZE conditioned on a HOLDING seed present (S8 probe machinery), ON vs OFF; advantage at credited t_f steps. NOTE: the γλ=0.945 retro-write tail also lifts approach ops (WAIT/SET_ALPHA/ADVANCE into t_f) — expected, not an anomaly | No detectable shift on an INFORMATIVE read (see asymmetric null) |

## Effect-size floor (the YELLOW-band license, PDR-0019)

**Episode-level paired Δcorr(reward, J) ≥ +0.10 absolute at n=5** (corr of
episode-summed reward vs episode J, per seed, paired OFF→ON; baseline 0.212±0.046,
probe recheck 0.179–0.261). Sign consistency: **≥ 4/5 seeds** with Δcorr > 0 (or a
criterion-(v) head shift). At **n=10**: paired-bootstrap lower bound > 0, preferred
point ≥ +0.10. Step-level corr is NOT a gate (F2 sampling ceiling). The retired
`Δfossilize_payable_J` median is NOT a gate at any n (structurally zero; PDR-0019).

## Safety gates (PDR-0018 §3, HARD)

Final-accuracy median paired Δ ≥ −0.3 pp; committed params/compute not materially up
unless accuracy/J justifies; fossilize opportunity not suppressed (criterion iv).

## Guard rules at scoring time

- **G1 (in-run, wired):** ≥13 fossils/batch × 2 consecutive batches aborts the ON run.
  **Pairing rule (pre-registered): an aborted ON seed is neither positive nor null — its
  pair is DROPPED from direction scoring and the abort investigated before any re-run.**
- **G2 (backstop):** end-of-run ON fossilize/ep > 3× paired OFF ⇒ not bankable
  regardless of Δcorr. A gross-spike line only; concentration can sit under it.
- **G3 (HARD):** ON efficiency (off-switch-J read) degraded > 10% vs OFF ⇒ not bankable.
  The 10% is PROVISIONAL pending a banked paired noise floor for the efficiency read
  (drl MINOR-3) — bank that floor before scoring or state its absence in the verdict.
- **G4 (paid-event review trigger — NEVER auto-reject):** paid events (`top_up > 0`)
  per ON run > 3× the ~12 expected (> 36) ⇒ the G3 + Δcorr adjudication is FORCED
  before banking a positive. Rationale: the honest success case also raises paid count;
  a hard count gate would Goodhart the guard (drl MAJOR-2). G3 + Δcorr are the
  correctness adjudicators; G1/G2 are breakers.

## Confound gate (drl MAJOR-3 condition)

If the direction effect is carried by paid events with `std_used < 1.0` (early-run
amplified — check the per-event `running_std_raw`/`std_used` fields), treat the result
as divisor-drift-confounded: re-confirm with `std_floor ≥ 1.0` before banking.

## Asymmetric null (PDR-0019, pre-registered)

A **positive** n=5 is informative and proceeds to n=10. A **null** n=5 is NOT
informative (~12 paid events/run; payment floor = k≥2 episodes ≈ 1.1% at OFF behavior)
— it does NOT fire PDR-0018 §7's pre-fossil follow-on and does NOT count against the
term until re-run with materially more accrued paid events. §7 fires only on an
informative underperformance.

## Descriptive-only tables (never gates)

- Conditional-on-payment credit magnitude (mean/median top_up and credit_buf among
  paying events) vs the 2×tau_provisional diagnostic line.
- Paid-event count and k-distribution per run (feeds G4).
- normalized_cap / std_floor / G-clamp bind rates (per-event flags).
- The raw v(S) tables (`v_table_masks`/`v_table_accs`) vs the transient-factorial
  prior (P(pay)=45.4%, gap p50 1.58 pp) — retiring the F2 calibration's proxy caveat.

## ON-run tau recalibration (MANDATORY before n=10 magnitude claims)

From the non-truncated `COMMITTED_SHAPLEY_TOPUP` emission (every k≥1 env, including
k=1 and non-payers): P99 of `phi − c_paid` over null-player slots. PDR-0017 trigger:
P99 > 2× placebo tau (> 0.56 pp) reopens the deadband method (option-a supplementation
or magnitude-scaled tau) before the A/B is read.

## FLAGGED post-launch deviation note (2026-07-04) — TOPUP `episode_idx` decode

**This is a telemetry-DECODE correction, not a scoring-rule change.** The launch
commit `6dd80716` carries bug esper-lite-e780981fe7: the `COMMITTED_SHAPLEY_TOPUP`
payload's `episode_idx` field is stamped with the **batch index**, not the episode id
(codebase convention `episodes_completed + env_idx`). The event is emitted directly to
the hub (no env-context wrapper), so the payload field is the ONLY episode identity on
the event. The payload's `env_id` field is correct.

**Decode for the in-flight ON runs (all at `6dd80716`; fix landed post-launch on the
branch, runs unaffected):** with `n_envs = 12` (both arm configs), the true id is

    episode_idx_true = stored_episode_idx * 12 + env_id

Exact for every batch including a final partial one (all *prior* batches are full, so
`episodes_completed` at batch b is 12·b). Any scoring read that joins TOPUP events
per-episode — criterion (v) advantage-at-credited-t_f in particular — MUST apply this
decode. Per-event reads (criteria i/ii, tau recalibration, G4 counts, bind-rate
tables) are unaffected.
