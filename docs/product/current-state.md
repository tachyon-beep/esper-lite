# Current State — Esper        Checkpoint: 2026-07-05 ~16:00 (checkpoint #16 — TERM PARKED (owner call a); EV track is the Now bet; PDR-0027)

## The bet right now
**EV-stabilization — joint value-target variance reduction for recurrent
factored-action PPO** (esper-lite-f25b71c165, moved Next → Now by PDR-0027).
Metric: per-stream EV / value-target variance; secondary product signal =
fossilize/ep and k≥2 co-fossilization frequency on control runs (the
organic-coverage trigger that can reopen the Shapley A/B re-run).

## What just closed
**Committed-Shapley top-up: PARKED by explicit owner word** ("checkpoint for
A (parking the term)", 2026-07-05). The n=5 A/B scored NULL-NOT-INFORMATIVE
(PDR-0026: paid episodes 0.49%, k≥2 = 1.10% — the pre-registered coverage
floor, hit exactly; mean paired Δcorr −0.017 = pure noise) with the mechanism
ALL GREEN. Instrument banked as validated; flag stays 0.0 and owner-gated;
enablement task esper-lite-f22a1d48a7 CLOSED. Scorecard:
`docs/analysis/2026-07-05-shapley-ab-n5-scoring-verdict.md`.

## In flight
- Nothing running. GPUs free. All 10 A/B runs banked in
  `telemetry/shapley_ab_n5/` (do not delete — owner-gated; quarantine dirs
  from the aborted prelaunch remain by rename only).

## Facts the next session must not relitigate
- The null does NOT count against the term (asymmetric null); PDR-0018 §7
  does NOT fire; GATE-2 does NOT reopen.
- Parking ≠ killing: the term passed every validity/safety/anti-farming test;
  the finding was channel bandwidth (1.1% coverage), not credit-math error.
- Re-run licensing lives in PDR-0027 reversal triggers (organic k≥2 rise ≥3×
  → owner decides; ≥10× coverage null = informative). n=10 magnitude path
  needs the τ null-player placebo leg (incl. GATE-placebo, addendum §5c).
- Mid-run "47% paid" flag was a reference-class error (44.7% vs 45.4% design
  prior); governor rollbacks ON 12.4 vs OFF 7.4/run (s43=21) — RCA if a
  future ON leg shows it again.
- Standing: scale>0 owner-gated ALWAYS; identity = tachyon-beep; no
  push/tag/release without ask.

## Open questions / blocked-on-owner
- None blocking the Now bet. (Owner may later reopen (b) coverage-leg or
  (c) pre-fossil reshape — recorded in PDR-0027, not live.)
- Standing placeholders: north-star/rent TARGETs (metrics.md); G3 paired
  noise floor unbanked (matters to any future efficiency gate).

## Last checkpoint did (checkpoint #16)
- PDR-0027 (owner-ratified park + EV promotion); roadmap: Shapley → Parked
  band, EV-stabilization → Now.
- Tracker: esper-lite-f22a1d48a7 closed (comments 122–123 carry verdict +
  parking); esper-lite-f25b71c165 annotated as Now bet (comment 124).
- No metric changes since #15 (PDR-0026 rows already dated 2026-07-05).

## Next session, start here
**Pick up the EV epic esper-lite-f25b71c165: first leg is Stage-0
instrumentation esper-lite-3d67b09687** (per-component return variance +
per-stream EV telemetry). Related ready items: esper-lite-cfbdfdf040
(specialist review of the staged variance-reduction plan) — review gate per
CLAUDE.md applies (drl-expert + yzmir-deep-rl). Memory pointers:
`ev-stab-stage2-impl-state` (HRA branch feat/ev-stab-stage2-hra, resume at
buffer per-stream GAE), `ev-variance-research-verdict` (sequencing:
marginal-V(s) first, then per-head-norm flag, then HRA head).
