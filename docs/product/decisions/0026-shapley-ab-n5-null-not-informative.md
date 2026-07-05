# PDR-0026 — Shapley A/B n=5 verdict: null-not-informative (coverage-bound); mechanism validated; term neither banked nor condemned

Date: 2026-07-05   Status: accepted (scoring executed strictly per prereg;
next-leg decision escalated to owner)
Author: Claude (agent)   Related: PDR-0019 (gates), PDR-0023/0025 (relaunch),
prereg + addendum (docs/analysis/2026-07-03 + 2026-07-04), verdict doc
`docs/analysis/2026-07-05-shapley-ab-n5-scoring-verdict.md`, tracker
esper-lite-f22a1d48a7.

## Context

All five ON runs banked integrity-clean 2026-07-05 against the banked OFF set.
Scored strictly per the frozen prereg: effect-size floor FAILED (mean paired
Δcorr −0.017, 2/5 positive vs the +0.10 / 4-of-5 gate) while every validity,
safety, and guard read came back green. Payment coverage landed exactly on the
pre-registered floor: k≥2 co-fossilization 1.10% of episodes (pre-registered
1.1%), paid episodes 0.49%, P(pay|k≥2) 44.7% vs the 45.4% factorial prior.

## Calls

1. **Verdict recorded as NULL — NOT INFORMATIVE** under the prereg's asymmetric
   null: the term touched too few episodes (0.49%) for an episode-level
   correlation to move; the paired-delta spread (SD 0.066) equals the
   pure-noise prediction (√2·0.046). The null does NOT count against the term
   and does NOT fire PDR-0018 §7.
2. **Mechanism BANKED as validated** (validity, not efficacy): criteria (i)/(ii)
   pass, clamp/ncap exercised, zero deadband violations, G1/G2/G4 clear, safety
   gates pass, GATE fossils pay through the production seam, no hacking
   signature. Any future leg builds on this instrument as-is.
3. **Mid-run G4 flag RESOLVED as a reference-class error:** 44.7% observed vs
   45.4% design prior (the ~1% figure was the placebo null-player rate). No
   GATE-2 reopen; PDR-0025's reversal trigger not met.
4. **No GATE-2 reopen from criterion (v):** it falsifies only on an INFORMATIVE
   read; this read was pre-registered-uninformative at 0.5% coverage.
5. **τ-recalibration labeling gap recorded:** null-player P99 is not computable
   from ON telemetry (censoring at τ); an injected-placebo leg (incl. the
   addendum §5(c) GATE-placebo leg) is a mandatory precondition for any n=10
   MAGNITUDE claim. Nothing blocked today (no magnitude claim made).
6. **No autonomous next leg.** The prereg licenses a re-run only with
   "materially more accrued paid events" — i.e., raising k≥2 co-fossilization
   frequency (design/curriculum/scope change) or a much larger run budget.
   That is an owner decision between at least: (a) park the term default-OFF
   and switch to the queued EV/advantage-pathology track
   (esper-lite-f25b71c165); (b) design a coverage-raising leg and re-run;
   (c) accept the coverage ceiling as a finding about WHERE this term can act
   and reshape the credit design (PDR-0018 §7 territory — which a null
   explicitly does not license by itself). **shapley_synergy_scale stays
   owner-gated; no scale>0 run without the explicit word.**

## Rationale

The experiment did what a well-designed experiment does on a sparse channel:
it validated the instrument end-to-end and measured the channel's bandwidth.
The design priors were strikingly accurate (1.1% coverage predicted/observed;
45.4% vs 44.7% pay rate), which means the null was structurally determined
before launch — the honest reading is "the payment gate is too narrow at
current behavior," not "the credit is wrong." Banking that distinction now
prevents relitigating the term later on the wrong grounds.

## Reversal triggers

- If a future leg materially raises accrued paid events (≥10× episode
  coverage) and the direction read is STILL null → that null IS informative
  and counts against the term (prereg asymmetry consumed).
- If the governor-rollback elevation (ON 12.4 vs OFF 7.4 per run; s43=21)
  recurs in any future ON leg → RCA before scoring that leg.
- Criterion (v) failing on an informative read reopens GATE 2 (PDR-0011,
  unchanged).
