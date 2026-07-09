# PDR-0054 — Long-horizon diagnostic read window corrected pre-data

Date: 2026-07-10
Status: accepted (pre-data amendment under the run-authorization grant)
Supersedes: PDR-0053 (in part — the determinism cross-check claim and the read-window
definition; the run itself, its config, and its purpose stand)

## Context

The high-effort review of the last 6 hours' commits found (verified CONFIRMED, with
exact schedule math) that PDR-0053's design rested on an incomplete schedule audit:
the **entropy-floor penalty schedule normalizes by total horizon**
(`total_train_steps = n_episodes × ppo_updates_per_batch`, config.py:396 →
`_get_penalty_schedule` progress at ppo_agent.py:1465). Breakpoints: 1.5× boost for
progress < 0.25, 1.0× for 0.25–0.75, decay 1.0→0.5 over 0.75–1.0. The penalty is
gradient-active in these runs (slot raw entropy ~0.04–0.09 vs floor 0.15 at all 200
observed updates). PDR-0053 checked the entropy anneal (genuinely horizon-independent
at 250 rounds) and missed this second, unique horizon-normalized schedule.

## Corrections (recorded BEFORE the diagnostic produces any data; the waiter is still
holding for cuda:1)

1. **Determinism cross-check claim RETRACTED.** Rounds 1–50 of the 600-round run
   reproduce the 200-round seed-41 arm; divergence onset is **round 51** (penalty
   multiplier 1.5× vs 1.0×). A round-51+ divergence is EXPECTED and must not be read
   as an infrastructure/determinism fault. The rounds-1–50 prefix remains a valid
   (weaker) determinism check.
2. **Pre-registered read window: rounds 251–451.** That is the all-schedules-constant
   window (entropy coef flat at 0.08 from round 250; penalty multiplier flat at 1.0×
   from round 151 to 451). The transient-vs-structural verdict reads ONLY this window:
   EV/Var(returns)/cf-scale stationarity within rounds 251–451 answers the question.
3. **Rounds 452–600 are schedule-confounded** (penalty decay 1.0→0.5) — descriptive
   only, never part of the verdict.
4. **External-validity caveat.** The 600-round run's early phase differs from the A/B
   arms (1.5× boost held to round 150 vs 50), so it is "a system with a longer
   exploration-boost phase", not a literal continuation. Acceptable for the
   stationarity question (which concerns the flat window's internal behavior), but any
   comparison of absolute levels against the A/B arms carries this caveat.

## Why the run proceeds unchanged

The config still delivers a ~200-round all-schedules-constant observation window — the
decisive instrument for transient-vs-structural. Removing the confound entirely would
require making the penalty schedule horizon-absolute (a training-path code change,
frozen until the ON wave completes) — not available, and not necessary for the
question as re-posed. No relaunch, no config change; only the reading is corrected.

## Reversal trigger

If the run shows the penalty-schedule difference visibly distorts even the flat window
(e.g., rounds 251–451 behavior inconsistent with the 200-round arms' overlapping
rounds in a way the boost-phase difference plausibly explains), discard the diagnostic
verdict and redesign post-freeze with a horizon-absolute penalty schedule.
