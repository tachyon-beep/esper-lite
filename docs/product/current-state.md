# Current State — Esper        Checkpoint: 2026-07-05 ~15:30 (checkpoint #15 — A/B SCORED: null-not-informative; PDR-0026)

## The bet right now
**Reward credit-assignment redesign — the n=5 A/B is SCORED.** Verdict
(PDR-0026): **NULL — NOT INFORMATIVE** under the pre-registered asymmetric
null. Effect-size floor failed (mean paired Δcorr −0.017, 2/5 positive vs
+0.10 / 4-of-5) because the payment channel is structurally sparse: paid
episodes 0.49%, k≥2 coalitions 1.10% (the pre-registered floor, hit exactly).
**Mechanism validity ALL GREEN** — the instrument works exactly as calibrated
(P(pay|k≥2) 44.7% vs 45.4% prior); the finding is channel bandwidth, not
credit-math error. Full scorecard:
`docs/analysis/2026-07-05-shapley-ab-n5-scoring-verdict.md`.

## In flight
- Nothing running. All 10 runs banked integrity-clean in
  `telemetry/shapley_ab_n5/` (ON: 2400/2400 × 5, zero tracebacks, every TOPUP
  finite + tau-stamped). GPUs free.

## BLOCKED ON OWNER — the next-leg decision (PDR-0026 call 6)
The prereg licenses an A/B re-run only with "materially more accrued paid
events." Options on the table:
(a) **park the term** (stays default-OFF) and start the queued
    EV/advantage-pathology track (esper-lite-f25b71c165);
(b) **coverage-raising leg** — design change to raise k≥2 co-fossilization
    above 1.1% of episodes, then re-run;
(c) **reshape the credit design** (pre-fossil territory = PDR-0018 §7 — which
    a null explicitly does NOT license by itself).
Standing: shapley_synergy_scale > 0 stays owner-gated. n=10 magnitude path
additionally requires the τ null-player placebo leg (incl. GATE-placebo,
addendum §5c) — labeling gap recorded in the verdict doc §5.

## Scoring facts to not relitigate
- Null does NOT count against the term; PDR-0018 §7 does NOT fire; GATE-2 does
  NOT reopen (criterion v falsifies only on an informative read).
- Mid-run "47% k=2 paid" flag RESOLVED: reference-class error (design prior
  45.4%, observed 44.7%; ~1% was the placebo-null rate).
- OFF-arm corr replication 0.214 vs banked 0.212 validates the estimator.
- If a ≥10×-coverage re-run is STILL null, THAT null is informative and counts
  (PDR-0026 reversal trigger).
- Secondary: governor rollbacks ON 12.4 vs OFF 7.4 mean/run (s43=21) — RCA
  before scoring any future ON leg if it recurs.

## Last checkpoint did (checkpoint #15)
- Integrity-checked all five ON completions; launched + banked on_s45.
- Scored the A/B strictly per prereg + addendum (floor, criteria i–v, safety,
  G1–G4, confound gate, τ-recalibration gap, descriptives).
- PDR-0026; verdict doc; metrics.md rows updated (corr, coverage, ON-wave
  validity); tracker comment on esper-lite-f22a1d48a7.

## Next session, start here
**(1) Put the next-leg decision (a/b/c above) to the owner — nothing runs
until they choose.** (2) If (a): pick up esper-lite-f25b71c165 (EV track).
(3) Whatever the choice, the Shapley instrument + telemetry + scoring scripts
(`ab_pass1_extract.py` / `ab_pass2_score.py` in the job tmp; estimator
documented in the verdict doc) are reusable as-is.
