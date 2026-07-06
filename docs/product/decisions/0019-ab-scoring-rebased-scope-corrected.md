# PDR-0019 — A/B scoring re-based on design criteria (i)–(v) with Δcorr as effect-size floor; scope corrected (synergy-credit, k=1 pays zero); asymmetric null recorded

Date: 2026-07-03   Status: accepted (owner-ruled)   Supersedes: PDR-0018 §3 (the
Δfossilize_payable_J overlay) and §6 (scope framing) — PDR-0018 otherwise stands
Related: PDR-0018; gate esper-lite-f22a1d48a7 comments #104–#105;
docs/analysis/2026-07-03-f2-scale-cap-calibration.md (incl. drl-expert review outcome)

## Context
F2 calibration froze with all four knobs accepted (scale=1.0, cap=5.0pp, std_floor=0.25,
normalized_cap=3.0). Two facts surfaced during calibration, both code-verified and
drl-review-confirmed: (1) k=1 coalitions pay structurally zero (φ ≡ c_paid ⇒ gap =
max(0, −τ) = 0), so the term pays ONLY synergy excess in k≥2 episodes — 1.1% of episodes
at OFF-arm behavior, not the 18.5–22% bootstrap floor cited at PDR-0018 ratification;
(2) the PDR-0018 §3 primary target ("median paired Δfossilize_payable_J ≥ 2×tau") is
structurally unscoreable — the per-episode median is identically zero for both arms.
The drl-expert review rated (2) a BLOCKER to scoring and recommended re-basing (its
option c), explicitly rejecting per-run paid mass as Goodhart-aligned with the
over-fossilization surface.

## The rulings (owner, 2026-07-03)

1. **Scoring re-based (supersedes PDR-0018 §3 overlay).** Primary gates for the paired
   OFF/ON A/B = the design doc's pre-registered falsifiable criteria (i)–(v). The YELLOW
   large-effect floor = **episode-level paired Δcorr(reward, J) ≥ +0.10 absolute** (as
   defined in PDR-0018: episode-summed reward vs episode J, the level where the 0.212
   baseline reproduces). Sign consistency (≥4/5 seeds) re-anchors to Δcorr and/or a
   criterion-(v) FOSSILIZE head-probability shift. Per-run aggregate paid mass is
   REJECTED as a gate (it rises under the over-fossilization hacking direction);
   conditional-on-payment credit magnitude is DESCRIPTIVE ONLY (reported against
   2×tau as a diagnostic, never gated). The same re-basing extends to the n=10
   magnitude gate: Δcorr lower bound > 0 (preferred point ≥ +0.10) + criteria (i)–(v)
   at n=10, with the payable-J-vs-tau point-estimate test dropped there for the same
   unit reason. Safety floors from PDR-0018 §3 are unchanged and HARD.
2. **Asymmetric null recorded as fact (owner: not a decision point).** At ~12 expected
   paid events per 200-episode run, a positive n=5 result is informative; a null is not.
   A null n=5 does NOT fire PDR-0018 §7's pre-fossil follow-on and does NOT count
   against the term; §7 fires only on an informative (adequately powered)
   underperformance — i.e., after enough paid events have accrued to distinguish
   "term inert" from "term never got enough shots."
3. **Scope correction (supersedes PDR-0018 §6).** The A/B is a **synergy-credit
   experiment gated on multi-fossil episodes** (payment floor = k≥2 ≈ 1.1% of episodes
   at OFF behavior), NOT a "terminal commitment-credit, not synergy" experiment as §6
   stated. Substrate note: the "entropy-degenerate population" caveat carried by the
   occurrence-probe doc and the drl review is partially stale — PDR-0006 (2026-07-01)
   reclassified the entropy collapse as a telemetry artifact (decision-step op entropy
   ≈ 0.88, healthy); the ~12-event power limit stands on its own.

## Options considered
Keeping the payable-J primary with a per-run aggregate re-basing was rejected on the drl
review's Goodhart argument; conditional-magnitude-as-gate rejected on n≈12 thinness.
Sequencing entropy hygiene before the A/B (the review's falsifiability option) was not
taken — the substrate premise is stale per PDR-0006, and the asymmetric-null rule covers
the residual risk without delaying the positive read.

## Reversal triggers
- PDR-0017's ON-recalibration trigger unchanged (first ON run P99 > 2× placebo tau
  reopens the deadband method).
- Design criterion (v) failing on an INFORMATIVE read reopens GATE 2 (PDR-0011).
- If the ON arm's realized paid-event count per run materially exceeds the ~12 expected
  (fossilize-rate inflation), the over-fossilization guard rules (gate criterion 3 +
  hard safety gates) adjudicate before any positive Δcorr is banked.
