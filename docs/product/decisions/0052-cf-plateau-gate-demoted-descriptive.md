# PDR-0052 — cf_value_loss plateau: gap closed, then gate demoted to descriptive

Date: 2026-07-10
Status: accepted (owner-ruled via explicit disposition)
Decider: owner (john); implementation + evidence by owner-agent
Relates-to: PDR-0050 (§11.1 LEG-B demotion — same root cause)

## Context

A code review found two correctness gaps in the Stage-2 acceptance reader
(`stage2_acceptance_io.py`):
1. **[P1]** The §11 W-rule declares an ON arm INVALID if `cf_value_loss` has not
   plateaued by the burn-in window W, but the reader never selected `cf_value_loss`
   and no validation performed the check — a still-warming cf head could reach
   SCREEN_PASS/ACCEPT on unstable EV.
2. **[P2]** Terminal `EPISODE_OUTCOME` safety readers (G1/G2) averaged `final_accuracy`
   and `param_ratio` with no bounds check — impossible telemetry (acc outside 0–100,
   param_ratio < 1) could become passing safety evidence.

Both were genuine (the reader code confirmed the omissions). Fixed TDD-style.

## What was implemented (both gaps closed)

- **P2 (stands as a hard gate):** `_per_env_terminal_mean` now validates every env's
  terminal value is finite and inside a contract range BEFORE averaging (`final_accuracy`
  ∈ [0,100]; `param_ratio` ∈ [1,∞)), per-env so two insane values cannot cancel into a
  plausible mean. Impossible telemetry → pair INVALID. Regression tests added.
- **P1 (implemented, then demoted — see ruling):** `cf_value_loss` ingested into
  `UpdateRow` + `_PPO_UPDATE_COLUMNS`; `plateau()` implemented to the frozen rule;
  `validate_pair` wired to check it on the ON leg.

## The finding that forced a decision

With the plateau gate implemented faithfully, **every real ON arm is INVALID at any W.**
Live ON telemetry (seeds 41/42): `cf_value_loss` oscillates 0→50 per update; neither the
raw series nor its trailing-8 median ever plateaus, and the smooth `cf_value_target_scale`
analog also fails on seed42 (step-jumps past update 17). Root cause = the SAME within-run
non-stationarity that demoted LEG-B (PDR-0050): the cf target scale grows all run, so
"cf head stopped warming" is unsatisfiable on this horizon. A literal gate would make the
entire pre-registered A/B unscorable for a reason unrelated to HRA's effect.

## The call

**Demote the cf-plateau clause to DESCRIPTIVE** (§11.2 amendment). Retain as hard gates:
`cf_value_loss` PRESENCE on every ON update (signature completeness) and finiteness on
scored updates. `plateau()` is kept and reported per-arm descriptively; a genuinely
warming-then-flat cf head would still show a finite plateau, so the oscillatory reality
is itself the informative datum. Screen predicate unchanged: LEG-A ∧ MECH ∧ G1–G4.

## Why not the alternatives

- **Gate on `cf_value_target_scale` plateau instead:** data does not support it (seed42
  fails); would discard arms for the return-growth confound, not cf warmup.
- **Keep the literal gate:** every ON arm INVALID; whole experiment unscorable and needs
  a redesigned criterion + full re-run — which faces the identical non-stationarity.

## Reversal trigger

If a future ON wave (different horizon, curriculum, or a detrended cf-scale metric) shows
`cf_value_loss` plateauing on ≥⌈0.8n⌉ arms, a warmup-stability gate becomes meaningful and
should be re-proposed and pre-registered before use. Until then, warmup is descriptive.

## Provenance / freeze discipline

Read-path-only change (reader + validity + pure-layer helper); training path untouched;
same-commit discipline intact (ON wave still on `fe177844`). Recorded while the ON wave
was in flight — the finding derives from `cf_value_loss` SHAPE (which HRA cannot make
stationary), not from any EV/verdict quantity, so it is not a peek at the outcome.
191 harness tests green; ruff + mypy clean; calibrate CLI re-verified (δ unchanged).
