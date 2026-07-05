# Design note — future Committed-Shapley coverage leg (NOT committed work)

**Status:** concept only. The term is PARKED (PDR-0027, owner call "A"); this
note exists so that IF the owner later reopens path (b) coverage-raising or
(c) credit reshape, the design starts from the n=5 findings instead of
re-deriving them. Nothing here is licensed to run. Sources: PDR-0026 verdict
(`docs/analysis/2026-07-05-shapley-ab-n5-scoring-verdict.md`), external-review
concurrence 2026-07-05 (GPT-Pro read, owner-forwarded; adopted items marked).

## The binding facts any future leg must respect

- **Frequency, not magnitude, is the failure mode.** Paid magnitudes p50 5.08
  / max 10.0 (≈15–30% of a typical episode reward) on 0.49% of episodes.
- **Payment quality is calibrated:** P(pay|k≥2) 44.7% vs 45.4% transient-
  factorial prior — a match that also retires the F2 proxy caveat (the prereg's
  descriptive table anticipated exactly this read on real fossil coalitions).
- **The channel is behavior-bound:** k≥2 co-fossilization = 1.10% of episodes
  under the current policy ecology. This is a design fact, not underpowering.

## Anti-pattern (adopted from external review — agreed)

**Do not respond by twiddling scalars.** Raising `shapley_synergy_scale`,
lowering τ, or loosening cap/ncap makes *rare* events *louder* — high-variance
reward spikes on 0.5% of episodes — without touching coverage. Any knob change
without a coverage mechanism is the wrong first move.

## Licensing (already in PDR-0026/0027 — restated, not new)

- Re-run informative only at **≥10× PER-RUN paid-episode coverage** (~5%).
  More same-design seeds do NOT qualify: n=10 same-design tightens the
  estimate of a ~0 Δcorr but cannot fix within-run dilution.
- τ null-player floor is NOT computable from ON telemetry (censoring at τ):
  any MAGNITUDE claim first needs an injected-placebo leg incl. the
  GATE-placebo PIN-E leg (prereg addendum §5c).
- Pre-fossil credit (path c) is PDR-0018 §7 territory — a null does not
  license it; it needs its own design + review cycle.

## Coverage-raising candidates (path b — sketches only)

- Curriculum/regime with more stable co-residence (longer HOLDING before
  terminal; slot-pressure regimes where k≥2 is common).
- Organic route (the PDR-0027 mainline): the EV track fixes advantage noise →
  commitment may rise on its own; watch fossilize/ep and k≥2 frequency on
  post-EV control runs (reversal trigger: ≥3× the 1.1% baseline).

## Metrics for a sparse term (adopted from external review — WITH a correction)

Episode-level corr(reward, J) is insensitive at <1% coverage; a future leg
should ADD event-level reads while keeping the episode-level metric for
behavioral relevance. Adopted candidates:

- **Paid-event precision/recall against J-positive structures** — does credit
  land on fossils/coalitions that prove J-positive? Non-circular, directly
  the "pays the right structures" question.
- **Conditional alignment on the eligible subpopulation** — corr(reward, J)
  restricted to TOPUP-eligible (k≥2) episodes, where the term actually acts.

**Rejected from the external list:** corr(TOPUP_paid, event-level Shapley
excess) — near-tautological: top_up is a deterministic clamp of the excess;
this correlates the mechanism with itself and would always "pass."

## Secondary watch item

Governor rollbacks ran ON 12.4 vs OFF 7.4 per run (s43 = 21). No gate, small
absolute rates; RCA before scoring if any future ON leg reproduces it.

## Verdict wording to reuse (sharpened, adopted)

"**Instrument banked; efficacy untested; payment-gate bandwidth measured as
too narrow** (coverage-bound null)." Avoids reading "neither banked nor
condemned" as "nothing was learned" — the implementation, GATE-fossil path,
caps/clamps, G4 resolution, and the coverage ceiling are all banked knowledge.
