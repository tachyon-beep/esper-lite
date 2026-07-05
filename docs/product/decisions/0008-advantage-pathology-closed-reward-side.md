# PDR-0008 — Advantage-pathology track closed: optimizer adequate, the defect is reward-side

Date: 2026-07-01   Status: accepted   Author: Claude (agent)   Owner sign-off: yes (grant — analysis)
Supersedes: —   Related: PDR-0006, docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md

## Context
PDR-0006's `does_not_fix` queued the "advantage-pathology family" (op-conditioned Q baseline, global advantage-norm
diluting sparse-head credit, single-sample bootstrap variance) as the next-most-likely credit-assignment culprit.
Investigated right-sized (orientation probe + advisor; no workflow — the check was a per-op probe, not a sweep).

## The call
**CLOSE the track as NOT the binding constraint.** Evidence: the op-conditioned-baseline confound is already fixed (the
baseline is op-independent V(s), P0-1, `vectorized_trainer.py:2452`; Q(s,op) is detached telemetry); sparse-head
advantage std holds 0.79–0.98 (global norm not crushing it); critic EV lifts 0.004→~0.55, and — crossing the fossil
timing — the committed structure (back half) is decided under adequate EV. Prior notes reached the same verdict
independently ("binding constraint = signal generation, not optimizer"). ⇒ the OPTIMIZER is adequate; the
credit-assignment defect is **REWARD-side** (the J-undervaluation the redesign targets). This is a positive result — it
confirms the redesign is aimed at the right layer.

## Caveat / open corner
One check is unresolved on current telemetry: a per-op advantage SIGN bias (a looks-good/actually-bad a mean/std washes
out) needs the raw per-step advantages+ops, which are not emitted → marked verify-if-instrumented, not a blocker.

## Reversal trigger
Reopen if a healthy-policy run shows a per-op advantage sign bias, or if sparse-head credit demonstrably fails to
propagate — specifically if the reward-credit term's GATE 2 learnability check fails: a terminal-sparse credit is a
DIFFERENT signal regime where "optimizer adequate" does NOT transfer.
