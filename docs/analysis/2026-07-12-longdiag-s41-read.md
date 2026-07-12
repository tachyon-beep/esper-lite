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

## Verdict: EARLY-TRAINING-TRANSIENT — feared structural churn RULED OUT; cause of stabilization UNIDENTIFIED

**Correction (advisor-reviewed):** an earlier draft called this "schedule-transient."
That over-attributes. The entropy anneal (0–250), the penalty schedule (flat ≥200), AND
the host learning curve (acc 40%→48%, then plateau) all progress and flatten over the
SAME rounds; at n=1 they are perfectly confounded in time and nothing in this run assigns
the stabilization to the schedule specifically. Indeed `value_target_scale` is still
*rising* into the 250–425 bin (24.7→28.8) before it declines — it does not snap flat at
the schedule boundary, which is mildly more consistent with a smooth learning-curve
plateau than a schedule step.

**What IS established:** the feared *ever-growing structural churn* (scale compounding
without bound) is ruled out — the scale stabilizes and slightly DECLINES rather than
compounding. **What is NOT established:** whether the schedule or host-convergence caused
the stabilization. That separation is not identifiable in this single OFF run, and saying
so is the correct result, not a gap to close.

"Stationary" below means **not drifting** (1st/2nd-half ratios ≤ 1.0) — NOT small or
quiescent. Absolute variance stays HIGH; the scale is stable. Per-round-index medians:

| Metric | boost 0–50 | baseline 50–150 | decay 150–200 | FLAT 250–425 | FLAT 425–600 | flat-window trend |
|--------|-----------|-----------------|---------------|--------------|--------------|-------------------|
| `value_target_scale` (return scale) | 3.66 | 11.30 | 21.10 | 28.82 | 26.14 | slope −1.10/100rd, ratio **0.907** |
| `return_std` | 4.28 | 12.78 | 23.15 | 25.71 | 23.09 | ratio 0.898 |
| `var_G` = Var(returns) | 18.4 | 163 | 536 | 660 | 533 | ratio 0.807 |
| `return_var_cf_share` (cf dominance) | 0.877 | 1.019 | 1.021 | 1.018 | 1.019 | slope **−0.0001**, ratio **1.000** |
| `explained_variance` | 0.012 | 0.094 | 0.566 | 0.596 | 0.625 | ratio 1.047 (improving) |

Reads:
- **The runaway scale stops.** `value_target_scale` / `var_G` climb ~7× from round 0 to
  ~round 250, then plateau and slightly decline. The A/B arms (200 rounds) were measured
  *entirely inside* this early-training growth — which is exactly why they saw monotonic
  2–4 → 24–28 growth. (Whether the plateau is caused by schedules going flat or the host
  converging is not separable here — see the verdict.)
- **cf-dominance is stationary** at ~1.018 across the flat window (matches the Stage-0
  gate median 1.017) — the cf stream's share of return variance does not drift.
- **EV is stationary-to-improving** — the critic keeps fitting; no late-training decay.
- **The adjudicating datum (advisor):** `var_G` stays HIGH (~530–660) and flat across
  250–600. A *quiescent, converged* host would show return variance COLLAPSING, not
  sitting high. Persistent high variance says the system is still ACTIVE — churning at a
  stable scale, not gone quiet. Weak (n=1) but real; it tilts the confound toward
  "active-but-schedule-flat" over "host-quiescent" — the reading under which a post-anneal
  LEG-B stays diagnostic rather than trivially-stable.
- Normalizer-lag proxy (`vg_mean_g − vg_mean_v`) oscillates near zero, no systematic lag.

## Consequence for the epic direction

- A **post-anneal scored window (rounds ≥250)** makes LEG-B (advantage-path volatility)
  and the cf-plateau criterion — both demoted to descriptive at the n=5 screen *because*
  of the early-training drift (PDR-0052) — COMPUTABLE as pre-registered metrics, which the
  transient denied. Caveat that controls whether that computation MEANS anything: if the
  window is stable because the host went quiescent, a post-anneal LEG-B is trivially
  "stable" (easy regime) and tells you nothing about a redesigned critic. The high-`var_G`
  datum above argues weakly against that (system still active) — but it is the load-bearing
  assumption, not a settled fact.
- This does **not** resurrect the rejected HRA: LEG-A failed on the absolute ev_sum
  deficit (PDR-0059), a separate finding.
- **Ranking limiter (do not lead with the schedule/host confound above this):** the
  DOMINANT unknown is the **unobserved ON-leg V_cf-bootstrap endogeneity.** This OFF run
  has no V_cf, so the self-inflicted, architecture-induced half of the non-stationarity —
  the half PDR-0060 says a redesign actually has to fix — is ENTIRELY INVISIBLE here. The
  honest framing is therefore: *the reward-side confound is escapable (a windowed
  instrument exists); the architecture-side is still unseen.* NOT "the measurement path is
  cleared for a redesigned HRA." That shifts weight toward getting an ON-leg Gram read (or
  TIP-first to build it) BEFORE committing to a large A′/B experiment, rather than jumping
  straight into A′/B on the strength of this OFF read.

## Caveats (ranked)

- **[dominant] ON-leg endogeneity unmeasured.** The V_cf-bootstrap non-stationarity
  (PDR-0060: `returns_cf = A_cf + V_cf`, self-inflicted) is invisible on this OFF leg (no
  V_cf). The vg Gram telemetry now emits exactly what a future instrumented ON leg needs
  to read it — this run does not.
- **[secondary] schedule vs host-convergence confound.** Not identifiable at n=1 (above).
- `value_target_scale` here is the TOTAL return scale; the cf-scale inference is indirect
  (cf carries ~all return variance, `return_var_main_share` ≈ 0.0025, so total ≈ cf).
- Rounds 0–200 reproduce the seed-41 OFF A/B arm's scale-growth shape (qualitative
  continuation corroboration; not claimed bitwise — GPU nondeterminism, PDR-0035).
