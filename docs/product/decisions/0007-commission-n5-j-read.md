# PDR-0007 — Commission the n=5 J-read (extend the causal pilot to begin banking the fork)

Date: 2026-07-01   Status: accepted   Author: Claude (agent)   Owner sign-off: PARTIAL — analysis/prep autonomous within grant; **Phase 1 GPU spend (~13h) awaits explicit owner go**
Supersedes: —   Related: PDR-0006, PDR-0004, PDR-0003, docs/plans/ready/2026-07-01-n5-j-read-run-sheet.md

## Context
PDR-0006 reframed the (a)/(b)/(c) fork discriminator to **J (acc-per-param)** and showed an n=3
pilot leaning to (b): r0c0 is an efficiency-enabling stem (suppressing it ~halves system
efficiency). The design pre-registered **n=5 to BEGIN causal evidence** (n=10 floor). The next
decision-relevant step is to extend to n=5 and read ΔJ/acc-per-param at the seed level — NOT a
collapse-fix rerun (there was no collapse; PDR-0006).

## Options considered
1. **Extend to n=5 (reuse 41–43 + run 44–45) and read ΔJ/acc-per-param** — pro: cheapest path to
   begin banking (4 runs, ~13h); con: n=5 may still be underpowered (escalate to n=10).
2. Jump straight to n=10 — pro: floor power; con: ~33h, premature if n=5 already separates.
3. Settle PDR-0004 estimand scope first — orthogonal; the J-read is estimand-invariant.

## The call
Option 1. **Phase 0 (prep) done autonomously:** preserved the 41–43 telemetry
(`telemetry/causal_r1_n5/`), promoted the J analyzer to `scripts/causal_contribution_j_analyze.py`
(validated — reproduces the pilot, all gates pass), and PRE-REGISTERED the run sheet (estimand,
gates, decision rule, n=10 contingency). **Phase 1 (the ~13h GPU spend) awaits explicit owner go.**
The decision rule is pre-registered: negative Δeff CI excl 0 → (b) credit enabling; includes 0 →
(c) fungible; positive → (a) penalize; wide → n=10.

## Rationale
J is the project's north-star and the metric that actually separates the fork (accuracy ties by
construction here); n=5 is the pre-registered threshold to begin evidence; reusing 41–43 makes it
cheap; pre-registration prevents goalpost-moving on a result the whole arc has circled.

## Reversal trigger
If the n=5 seed-level Δeff CI does NOT exclude 0, (b) is not banked — escalate to n=10 (or flip to
(a) if positive). If a gate trips on 44–45 (offset-free, zero-r0c0, finiteness, decision-step
entropy), discard that pair, do not bank. Abandon the n-extension path if the harness cannot
produce a clean read at n=5 (route the fork to a different method).
