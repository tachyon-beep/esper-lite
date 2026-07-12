# 600-round seed-41 OFF diagnostic — PDR-0055 read

Date: 2026-07-12 · Run: `telemetry/stage2_off_longdiag/seed41/telemetry_2026-07-11_192943/`
Commit: `7ea84dc6` (schedule-fixed code) · Completion: 600/600 rounds, rc=0, 0 crashes.

## Question (pre-committed, PDR-0055 carried from 0053/0054)

Is the within-run non-stationarity that demoted LEG-B and the cf-plateau criterion at
the n=5 screen (cf/return scale drifting ~2–4 → ~24–28 within every 200-round A/B arm)
a **schedule transient** (entropy anneal + penalty schedule moving through rounds
0–250) or **structural morphogenetic churn** (persists at constant schedules)?

Pre-committed rule: stationary EV/Var(returns)/cf-scale within the all-flat window
(rounds 250–600) ⇒ schedule-transient ⇒ a post-anneal scored window can re-power LEG-B
and re-pose the cf-plateau criterion. Continued drift ⇒ structural ⇒ no horizon fixes
it; mechanistic claims need detrended/relative metrics.

Schedule geometry (this run): penalty boost <50, baseline 50–149, decay 150–199, flat
0.5× ≥200; entropy anneal spans 3000 ep / 12 envs = 250 rounds → entropy flat ≥250.
**All-schedules-flat window = rounds 250–600 (350 rounds).**

## Verdict: SCHEDULE-TRANSIENT (reward/return-side)

The dramatic scale growth is confined to the scheduled phases; the all-flat window is
stationary (flat-to-slightly-declining). Per-round-index medians:

| Metric | boost 0–50 | baseline 50–150 | decay 150–200 | FLAT 250–425 | FLAT 425–600 | flat-window trend |
|--------|-----------|-----------------|---------------|--------------|--------------|-------------------|
| `value_target_scale` (return scale) | 3.66 | 11.30 | 21.10 | 28.82 | 26.14 | slope −1.10/100rd, ratio **0.907** |
| `return_std` | 4.28 | 12.78 | 23.15 | 25.71 | 23.09 | ratio 0.898 |
| `var_G` = Var(returns) | 18.4 | 163 | 536 | 660 | 533 | ratio 0.807 |
| `return_var_cf_share` (cf dominance) | 0.877 | 1.019 | 1.021 | 1.018 | 1.019 | slope **−0.0001**, ratio **1.000** |
| `explained_variance` | 0.012 | 0.094 | 0.566 | 0.596 | 0.625 | ratio 1.047 (improving) |

Reads:
- **The runaway scale is the transient.** `value_target_scale` / `var_G` climb ~7× from
  round 0 to the anneal boundary (~250), then STOP and slightly decline once both
  schedules are constant. The A/B arms (200 rounds) were measured *entirely inside* this
  transient — which is exactly why they saw monotonic 2–4 → 24–28 growth.
- **cf-dominance is perfectly stationary** at ~1.018 across the flat window (matches the
  Stage-0 gate median 1.017) — the cf stream's share of return variance does not drift.
- **EV is stationary-to-improving** — the critic keeps fitting; no late-training decay.
- Normalizer-lag proxy (`vg_mean_g − vg_mean_v`) oscillates near zero, no systematic lag.

## Consequence for the epic direction

- A **post-anneal scored window (rounds ≥250)** is a valid instrument: LEG-B
  (advantage-path volatility) and the cf-plateau criterion — both demoted to descriptive
  at the n=5 screen *because* of the non-stationarity (PDR-0052) — become re-powerable in
  a future pre-registration.
- This does **not** resurrect the rejected HRA: LEG-A failed on the absolute ev_sum
  deficit (PDR-0059), a separate finding. What it rehabilitates is the *measurability* of
  the volatility/plateau claims, strengthening the case that a redesigned instrument could
  get a clean read where the 200-round screen could not.

## Caveats (do not overclaim)

- **n=1, OFF leg.** This establishes the reward/return-side non-stationarity is
  schedule-transient. It does NOT observe the ON-leg V_cf-bootstrap component (PDR-0060
  endogeneity: `returns_cf = A_cf + V_cf`, self-inflicted) — there is no V_cf on this OFF
  run. That architecture-induced component remains for the held-out Gram read on a future
  instrumented ON leg (the vg telemetry now emits exactly what that needs).
- `value_target_scale` here is the TOTAL return scale; the cf-scale inference is indirect
  (cf carries ~all return variance, `return_var_main_share` ≈ 0.0025, so total ≈ cf).
- Rounds 0–200 reproduce the seed-41 OFF A/B arm's scale-growth shape (qualitative
  continuation corroboration; not claimed bitwise — GPU nondeterminism, PDR-0035).
