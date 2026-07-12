# 600-round seed-41 OFF diagnostic — PDR-0055 read

Date: 2026-07-12 · Run: `telemetry/stage2_off_longdiag/seed41/telemetry_2026-07-11_192943/`
Commit: `7ea84dc6` (schedule-fixed code) · Completion: 600/600 rounds, rc=0, 0 crashes.
Reviewed by: advisor + external peer (ChatGPT-pro), 2026-07-12 — both integrated below.

## Question (pre-committed, PDR-0055 carried from 0053/0054)

Is the within-run non-stationarity that demoted LEG-B and the cf-plateau criterion at
the n=5 screen (cf/return scale drifting ~2–4 → ~24–28 within every 200-round A/B arm)
a **schedule transient** (entropy anneal + penalty schedule moving through rounds
0–250) or **structural morphogenetic churn** (persists at constant schedules)?

Pre-committed rule: stationary EV/Var(returns)/cf-scale within the all-flat window
(rounds 250–600) ⇒ **schedule-transient** ⇒ a post-anneal scored window can re-power
LEG-B and re-pose the cf-plateau criterion. Continued drift ⇒ structural ⇒ no horizon
fixes it; mechanistic claims need detrended/relative metrics.

Schedule geometry (this run): penalty boost <50, baseline 50–149, decay 150–199, flat
0.5× ≥200; entropy anneal spans 3000 ep / 12 envs = 250 rounds → entropy flat ≥250.
**All-schedules-flat window = rounds 250–600 (350 rounds).**

## Verdict

**Operational (PDR-0055, pre-registered): SCHEDULE-TRANSIENT / POST-ANNEAL PLATEAU.**
The pre-committed rule mapped "no persistent drift in rounds 250–600" → "schedule-
transient"; that condition is met, so the pre-registered call fires. Honoring the
pre-registration, the operational label stands.

**Scientific (softened — do not over-lean): the cause is NOT causally isolated.** The
entropy anneal (0–250), the penalty schedule (flat ≥200), AND the host learning curve
(acc 40%→48%, then slowing) all progress and flatten over the SAME rounds; at n=1 they
are perfectly confounded, and no matched counterfactual (schedules flat from the start,
or breakpoints shifted with all else held) exists. `value_target_scale` even keeps
*rising* into the 250–425 bin (24.7→28.8) before declining — it does not snap flat at the
schedule boundary. So:
- **Established (very likely):** in this seed, the reward/return scale stops exhibiting
  *persistent upward drift* once both known schedules are flat — which **rules against
  unavoidable structural scale growth in this run**.
- **NOT established:** that the schedules (rather than ordinary training maturation)
  *caused* the earlier rise; nor that this generalizes beyond n=1.

"Plateaued / no persistent upward drift over the frozen window" is the honest phrasing;
"stationary" is used loosely (half-window medians + slope + the pre-committed binary),
NOT as a formal covariance-stationarity / change-point result.

## The numbers (per-round-index medians)

| Quantity | boost 0–50 | baseline 50–150 | decay 150–200 | FLAT 250–425 | FLAT 425–600 | flat-window |
|----------|-----------|-----------------|---------------|--------------|--------------|-------------|
| `value_target_scale` — **running normalizer scale** | 3.66 | 11.30 | 21.10 | 28.82 | 26.14 | ratio 0.907 |
| `return_std` — **instantaneous batch σ** | 4.28 | 12.78 | 23.15 | 25.71 | 23.09 | ratio 0.898 |
| `var_G` = Var(returns) — **instantaneous (Gram)** | 18.4 | 163 | 536 | 660 | 533 | ratio 0.807 |
| `return_var_cf_share` (cf covariance share) | 0.877 | 1.019 | 1.021 | 1.018 | 1.019 | ratio 1.000 |
| `explained_variance` | 0.012 | 0.094 | 0.566 | 0.596 | 0.625 | ratio 1.047 |

Reads:
- **The runaway scale stops.** Scale climbs ~7× from round 0 to ~250, then plateaus and
  slightly declines. Rephrased carefully (peer): the 200-round A/B arms *ended before the
  post-anneal plateau was observable*, so they could not distinguish a finite training
  transient from persistent structural drift — not "they were inside the transient, which
  is why they grew" (that asserts causation the run doesn't isolate).
- **Running vs instantaneous scale agree.** The running normalizer scale (28.8→26.1)
  tracks the instantaneous `return_std` (25.7→23.1) and `var_G` — the plateau is a real
  property of the return process, NOT a sluggish running-stat artifact.
- **EV improves in absolute terms too, not just fractionally.** Unexplained variance ≈
  (1−EV)·Var(G) falls **267 → 200** across the two flat-window halves ((1−0.596)·660 →
  (1−0.625)·533). The critic genuinely fits better; not a relative-metric illusion.
- **cf-dominance is flat** at ~1.018 across the window (matches the Stage-0 gate median
  1.017) — the cf stream's share of return variance does not drift.
- **Adjudicating datum (advisor):** `var_G` stays HIGH (~530–660) and flat. A *quiescent,
  converged* host would show variance COLLAPSING; persistent high variance says the system
  is still ACTIVE — churning at a stable scale, not gone quiet. Weak (n=1) but real; it
  tilts the schedule-vs-host confound toward "active-but-flat," the reading under which a
  post-anneal LEG-B stays diagnostic rather than trivially-stable.
- **Mean value bias is small** (`vg_mean_g − vg_mean_v` ≈ 0). NOTE (peer correction): this
  is `mean(returns) − mean(values)` = a mean-residual / value-bias measure, **NOT a
  normalizer-lag proxy** (an earlier draft mislabeled it). A true normalizer-lag read needs
  instantaneous target mean/std vs the running-normalizer state; small mean bias does not
  by itself certify the normalizer tracks scale.

## Consequence for the epic direction

- A **post-anneal window (≥250) is a plausible measurement population** for LEG-B and a
  cf-plateau criterion — but do NOT restore the old gates unchanged (see design notes).
  Guard: if the window were flat because the host went quiescent, a post-anneal LEG-B is
  trivially "stable" and uninformative; the high-`var_G` datum argues against that but it
  is the load-bearing assumption, not settled.
- This does **not** resurrect the rejected HRA: LEG-A failed on the absolute ev_sum
  deficit (PDR-0059), a separate finding a later OFF plateau cannot reverse.
- **Dominant limiter (ranked above the schedule/host confound):** the unobserved ON-leg
  **V_cf-bootstrap endogeneity.** This OFF run has no V_cf, so the self-inflicted,
  architecture-induced half of the non-stationarity — the half a redesign must fix — is
  invisible here. Honest framing: *the reward-side confound is escapable; the
  architecture-side is unseen.* Weight shifts toward an instrumented ON-leg Gram read
  BEFORE a large A′/B experiment.
- Path balance (peer-concordant): **Objective A remains rejected**; **Path A′ modestly
  MORE credible** (target can settle → cf not proven intrinsically structurally
  non-stationary); **Path B still slightly-to-moderately favoured** until an ON run shows
  post-anneal V_cf becomes reliably useful.

## Design notes for a FUTURE pre-registration (do not carry the old gates forward)

- **LEG-B needs multi-seed OFF calibration** over the frozen post-anneal window: n≥5 OFF
  seeds, fixed rounds 250–600, ε frozen from OFF spread only, same scored-update population
  in ON. One seed proves *feasibility*, not a noise floor.
- **cf-plateau: split the one `cf_value_loss` rule** (found unsatisfiable in the 200-round
  ON data, PDR-0052) into three distinct, separately-tested plateaus, all readable from the
  vg Gram telemetry: (1) *target-process* plateau — slope/change in Var(G_cf) / instantaneous
  cf target scale; (2) *learner* plateau — slope/change in cf residual variance or EV_cf;
  (3) *normalizer* plateau — instantaneous/running cf scale ratio stabilizes. Do not
  collapse them back into one loss-plateau rule.

## Caveats (ranked)

- **[dominant] ON-leg endogeneity unmeasured** (above). The vg Gram telemetry now emits
  what a future instrumented ON leg needs; this run does not.
- **[secondary] schedule vs host-convergence confound.** Not identifiable at n=1.
- **[secondary] n=1, one OFF seed.** Project scar applies directly: an n=1 synergy read
  once reversed under the systematic n=5 read (PDR-0026). Do not harden a directional
  result from a narrow/mismatched population — this supports feasibility, not generality.
- Rounds 0–200 reproduce the seed-41 OFF A/B arm's scale-growth shape (qualitative
  continuation corroboration; not claimed bitwise — GPU nondeterminism, PDR-0035).
