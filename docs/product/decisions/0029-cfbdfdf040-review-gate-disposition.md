# PDR-0029 — cfbdfdf040 review-gate disposition: Stage 3 scoped out; MAJOR-1/-3 folded into Stage 2 acceptance

Date: 2026-07-05   Status: accepted (within grant: prioritize / scope the backlog)
Author: Claude (agent)   Related: PDR-0028 (the reconciliation), task
esper-lite-cfbdfdf040 (the review gate), epic esper-lite-f25b71c165.

## Context

The formal drl-expert + yzmir-deep-rl review gate (esper-lite-cfbdfdf040) ran this
session, verified against landed code (the removed research reports were not needed).
Verdicts: Stage 1 (marginal-V verify) **approved** — already satisfied by the
directly-learned op-independent V(s), the stronger variant, no residual action-coupling.
Stages 0, 1b, 2 **approved-with-changes**. Stage 3 (escrow telescoping) **needs-revision**
— diagnosis sound but no authored plan, and the index conflates the clip-invariance
defect with the observability defect (plus an off-by-one: feed `escrow_credit_prev`,
not `_target`). BLOCKER: none (the normalizer-misrouting candidate verified clean).

Two MAJOR findings on Stage 2 acceptance:
- **MAJOR-1:** R_cf still shapes the trunk via the POLICY gradient (A_total includes
  A_cf), so an `ev_main`-IDR win can be hollow / value-neutral.
- **MAJOR-3 (provenance):** the HRA Stage-0 gate appears unrecorded; the only
  Stage-0-ish A/B in git history is SHAPLEY, and `config.py:509-516` makes
  shapley⊕HRA mutually exclusive — so shapley evidence must NOT be credited to the
  HRA gate.

## Options

- **(a)** Scope Stage 3 OUT of this gate (record its diagnosis as drl-approved, require
  a separate authored Stage-3 plan + fresh review); pass 0/1/1b/2 contingent on
  MAJOR-1/-3 folded into Stage 2 acceptance.
- **(b)** Hold cfbdfdf040 open pending an authored Stage-3 plan.

## Call

**(a).** Stage 3's escrow diagnosis is banked as drl-approved; a NEW authored Stage-3
plan + fresh drl/yzmir review is required before any Stage-3 code. The review gate
(cfbdfdf040) is closed as complete for Stages 0/1/1b/2. Acceptance criteria folded
into the Stage-2 task:
- **MAJOR-1:** `ev_sum` is promoted from sanity-check to a HARD acceptance floor
  (median `ev_sum_ON ≥ ev_sum_OFF` within noise) + one downstream signal
  (advantage-variance reduction or a return/val-acc trend) — so a hollow win fails.
- **MAJOR-3:** capture the VALUE-FREE HRA Stage-0 gate BEFORE any HRA ON-leg run;
  never credit shapley evidence to the HRA gate.
- **Stage-0 change:** `r_main_var_share` = Var(R_main)/Var(R) is the load-bearing
  smoothness leg (variance-share, not the fragile CoV) — already implemented as such
  in `compute_return_variance_shares` (PDR-0028).

## Rationale

Scoping Stage 3 out unblocks the majority of the epic (0/1/1b/2 are clean) while giving
the escrow correctness fix — which touches the observation contract — the dedicated,
separately-reviewed plan it needs. Folding MAJOR-1/-3 into Stage-2 acceptance ensures
the de-shaping is judged on a real, provenanced win rather than a bookkeeping artifact.

## Reversal triggers

- If the authored Stage-3 plan shows the escrow fix is entangled with Stage-2 and
  cannot be sequenced independently → re-fold Stage 3 into the gate.
- If a Stage-2 ON-leg run shows `ev_sum` degrade below the floor → Stage-2 acceptance
  FAILS (MAJOR-1 fired), regardless of `ev_main` movement.
