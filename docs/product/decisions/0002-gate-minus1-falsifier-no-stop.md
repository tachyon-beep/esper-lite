# PDR-0002 — GATE −1 cheap-fix falsifier verdict: NO STOP (redesign SURVIVES)

Date: 2026-06-28   Status: accepted   Author: Claude (agent)   Owner sign-off: yes (grant — accept against pre-registered criteria)
Supersedes: —   Related: PDR-0003, metrics.md, esper-lite-a221da47ea, docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md

## Context
The Now bet (reward credit-assignment redesign, Phase 0) hinged on a pre-registered
falsifier: would a *cheap rescale* of the existing reward (attribution unit-normalize,
shaped-attribution clip ×2/×5, escrow obs-feed) fix the pathology — committed structure
carrying ~0 committed-per-param counterfactual value (J) — and thereby make a full
credit-assignment redesign unnecessary? The pre-registered STOP rule: STOP (rescale is
enough) iff a non-escrow lever shows **churn↓ AND committed-acc↑** on a **seed-level**
paired bootstrap (resampling the ≥5 seeds as the unit, NOT the 12 vec-envs — the
pre-registered invalidator).

## Options considered
1. **Score the completed 25/25 sweep on the seed-level paired bootstrap and read the
   pre-registered rule** — pro: honors the falsifier as written; con: n=5 seeds is thin.
2. Score on the vec-env unit (n≈60) for tighter CIs — REJECTED: the pre-registered
   invalidator explicitly forbids it (vec-envs are not independent; would falsely tighten).
3. Defer the verdict pending more seeds — REJECTED: the sweep was complete and the rule
   was pre-committed; deferring would be moving the goalposts.

## The call
Option 1. Verdict: **NO STOP — the redesign SURVIVES the falsifier.** No non-escrow cheap
lever produced churn↓ AND committed-acc↑ on the seed-level bootstrap. A classification bug
(escrow arm mislabeled) was caught and corrected mid-scoring. **SURVIVE ≠ ADMIT:** the
verdict does not greenlight building the redesign — it only closes the cheap-rescale
escape hatch. The redesign-build itself stays in Next, gated on the (a)/(b) fork (PDR-0003).

## Rationale
A pre-registered falsifier is only worth running if its verdict binds. The cheap levers
had a fair, full-power test on the correct unit and did not clear the bar; banking "rescale
is enough" would have been unsupported. Advisor-vetted; concordance-checked across arms.

## Reversal trigger
Reopen if a **larger or differently-designed falsifier** finds a non-escrow cheap-rescale
lever that achieves **churn↓ AND committed-acc↑** on the seed-level paired bootstrap
(metrics.md: corr(reward,J) ↑ and committed-J ↑ without a redesign). Also reopen if the
escrow obs-feed (the one lever NOT falsified here) is later shown to clear the bar on its
own — escrow was excluded from the STOP rule, not proven inert.
