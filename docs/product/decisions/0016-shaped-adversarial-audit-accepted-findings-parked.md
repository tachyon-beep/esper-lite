# PDR-0016 — SHAPED adversarial audit commissioned and accepted; findings PARKED pending the data-informed prioritization review

Date: 2026-07-03   Status: accepted (commissioning owner-directed in-session; acceptance
within grant)   Author: Claude (agent)
Related: esper-lite-a7ef375203 (audit task, open), docs/analysis/2026-07-03-shaped-reward-adversarial-audit.md
(committed f3d8db95), PDR-0009 (interpretation caveat), esper-lite-f22a1d48a7 (A/B interaction)

## Context
The owner, probing whether the current work "takes the existing reward systems into
account," named the residual fear: "there's an underlying flaw in the shaped result even
if I can't see it," and commissioned a Fable-powered specialist audit — with the explicit
directive that prioritization of findings waits for the next data set "so we're not
making decisions blind."

## The call
Audit commissioned (reward-function-reviewer, read-only, three lenses: per-additend
hacking surfaces; PBRS invariance conditions; behavioral-sink register). Delivered same
day: 9 findings, all mechanism-confirmed with line cites. Acceptance was gated on the
orchestrator independently re-verifying the load-bearing claims in code — 5/5 CONFIRMED:

- F1 (HIGH): the Phase −1 attribution clip is positive-only and applied BEFORE the PRUNE
  sign-flip → pruning a −15 pp seed pays +15, unbounded, clip-exempt.
- F2 (HIGH): step reward reads only the TARGETED slot and WAIT canonicalizes to r0c0 →
  ~⅔ of steps' dense credit keys on r0c0; structural ceiling on corr(reward, J); caveats
  the PDR-0009 narrative (position-asymmetric credit vs learned stem preference — the
  J-side direction result stands, the "policy learned to build stems at r0c0" story is
  confounded).
- F3 (MED-HIGH): PBRS invariance violated four ways → pbrs_bonus is ordinary shaping.
- F5 (MED): min-over-LOO all-off fallback spuriously suppresses fossilizing
  coalition-carried seeds — a commitment suppressor the Committed-Shapley top-up does NOT
  fix (different gate).

Owner-endorsed synthesis (in-session): F1 (prune lucrative) + F5 (fossilize punished) +
F3 (commitment is a potential drop) + known coalition blindness jointly rationalize the
observed commitment avoidance (germinate 12.4 / prune 11.6 / fossilize 0.207 per episode;
committed-J ≈ 0). Mechanisms are code-confirmed; occurrence rates are NOT yet measured.

## Disposition
All findings PARKED per the owner directive. No fixes, no reprioritization. The
zero-GPU occurrence probes (scripted SQL over existing n=5 telemetry) are defined in the
audit doc and ready to run at the review. One time-sensitive note recorded on both
tracker issues: the F1 channel is exactly what the Phase −1 clip arm exempts — run the
F1 probe before reading any clip-arm verdict.

## Reversal trigger
The parking decision expires at the prioritization review (trigger: PIN-E tau + gate-leg
data delivered — which occurred the same day; the review is now unblocked). If the review
does not occur before the next enablement-gate leg is dispatched, the A/B-informativeness
risk (see PDR-0017) forces the F1/F5 question to be answered first.
