# PDR-0009 — n=5 J-read: r0c0 is an efficiency-enabling stem (b) — banked at total-system scope

Date: 2026-07-02   Status: accepted (DIRECTION banked within grant; the n=10-vs-n=5 MAGNITUDE call is FLAGGED for owner)   Author: Claude (agent)
Supersedes: —   Related: PDR-0007, PDR-0006, PDR-0004, docs/analysis/2026-07-02-n5-j-read-result.md

## Context
The n=5 causal-contribution run (PDR-0007) resolves the (a)/(b)/(c) fork: is the early-converging r0c0 cohort a
freeloader (a), fungible (c), or an LOO-undervalued enabling stem (b)? Discriminator = paired Δ(acc-per-param),
seed-level (the pre-registered estimand). All 10 runs completed (the seed-44 telemetry crash was fixed, db61dc3a).

## The call
**BANK (b): r0c0 is an efficiency-enabling stem.** All 5 seeds show suppressing r0c0 ~halves system param-efficiency
(Δeff −4.4…−12.6 pp/M-param, median −7.0), all gates pass (offset-free, zero-r0c0, finiteness, decision-step entropy),
sign-test p≈0.03. ⇒ the reward should CREDIT r0c0's contribution, not penalize it — refutes the freeloader fix (a); (c)
fungible is refuted (the replacement costs ~2–2.7× the params).

## Scope guards (advisor — do NOT let these harden)
- **TOTAL-SYSTEM, not mechanistic:** the gap is ENTIRELY the denominator — accuracy contribution is ~8pp in BOTH arms;
  only total params move (~400k→~1M). Claim: "the system is less param-efficient without r0c0 *available*," NOT "r0c0
  *mechanistically enables* the downstream structure."
- **DIRECTION banked, MAGNITUDE not:** the reported "CI [−12.6,−4.4]" is min/max of 5 points = a sign test, not a CI.

## Reversal trigger
Reopen the DIRECTION only if n=10 (or a healthy, well-trained-policy re-run) does not keep Δeff negative across seeds.
The MAGNITUDE is explicitly unbanked pending the owner's n=10 decision. Also: the read is on a 200-episode
(healthy-exploration, not fully-converged) policy — a much-longer-trained controller could in principle differ.
