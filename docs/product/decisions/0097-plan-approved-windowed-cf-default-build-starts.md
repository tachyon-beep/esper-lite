# PDR-0097 — Plan APPROVED (three review rounds, zero open blockers); windowed-cf ratio denominator adopted as DEFAULT; Phase-2 build STARTS

Date: 2026-07-14   Status: accepted
Follows PDR-0096. Input: drl-expert final eyeball — APPROVE (B1–B5, N1–N5, W1-A/B, N-r2a/b/c, B-epoch cleanup all resolved; "the fix improved the gate, not just closed the hole" — the symmetric-inclusion law makes G-CONTINUITY exact for the constant-input synthetic).

## Decisions
1. **Plan APPROVED — the build begins** under the PDR-0090 authorization (default-off impl + tests + replay +
   drl pre-commit review + docs; no training activation, no GPU, no obs changes pending F2).
2. **Windowed-cf denominator adopted as the DEFAULT construction** (upgraded from fallback-if-misfires): the
   annuity ratio check compares `q_settle` (EWMA) against `cf_w` = EWMA of the window's per-epoch clean
   counterfactual measurements, same span — like-with-like; the spot `cf_B` form is rejected as noise-fragile in
   both directions (reviewer's structural argument, adopted whole: a smoothed/spot ratio can misfire on
   legitimate seeds AND silently skip ransomware whose cf spikes at B). The replay case matrix must test BOTH
   failure directions.
3. Build order per plan: leyline contracts → settlement pure core (TDD) → kasmina fields → action layer →
   reward path → replay B path. Reviewer verifies at the pre-commit code review.

## Reversal trigger
- Inherited unchanged from PDR-0096 (misfire on legit-low-cf seeds → smoothing refinement, never removal) and
  PDR-0091 (placement contradictions halt the build).
