# PDR-0004 — Causal-run estimand scope: total-system pivot vs mechanistic-via-placebo

Date: 2026-06-28   Status: **accepted** (owner ratified total-system 2026-07-01)   Author: Claude (agent)   Owner sign-off: YES — ratified 2026-07-01
Supersedes: —   Related: PDR-0003, docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md (header: SUPERSEDE PENDING)

## Context
The v1 causal design used a placebo arm to isolate a *mechanistic* claim ("r0c0 specifically
*enables* downstream structure"). Adversarial review found the placebo cannot bound the
action-space artifact it was meant to bound. The v2 design resolves this by **pivoting the
estimand to total-system**: it deletes the placebo and reduces Goal-1 to "the *system* is
worse without r0c0's commit" (system-level dependence) rather than "r0c0 mechanistically
enables its neighbours." This narrows what the experiment can claim, and is a scope decision
about the redesign's evidential basis — it is **not** purely internal methodology.

## Options considered
1. **Adopt the total-system estimand (v2)** — pro: identifiable, resolves three review
   blockers, the harness is already estimand-invariant so it runs either way; con: gives up
   the mechanistic "enabling" claim — the redesign would rest on system-level dependence.
2. **Commission a concrete placebo / DUMMY-R0C0 design** to try to preserve the mechanistic
   claim — pro: keeps the stronger claim if it can be made to bound the artifact; con: no
   such design exists yet; may not be constructible.
3. Run with v1 as-is — REJECTED: the placebo is known not to bound the artifact.

## The call (PROPOSED — not yet enacted)
Lean Option 1 (total-system). **This crosses into owner-gated territory**: it changes the
evidential claim the credit-assignment redesign would stand on. Per the authority grant
(escalate vision/strategy-scope changes), it is recorded `proposed` and surfaced for owner
sign-off. The pilot (PDR-0003) is estimand-invariant and proceeded without prejudging this.

## Rationale
The total-system estimand is the only one currently identifiable; the mechanistic claim has
no working design. But choosing "system-level dependence" as the bar the redesign must clear
is a scope call the owner should ratify, not the agent.

## Reversal trigger
Resolve on owner decision: **(1)** owner ratifies total-system → mark `accepted`, v2 design
supersedes v1; **(2)** owner wants the mechanistic claim → commission the placebo/DUMMY-R0C0
design as a Next bet and this PDR is `superseded` by that design's PDR. Until the owner
rules, the causal read is reported as total-system with the mechanistic claim explicitly
out of scope.

## Owner ratification (2026-07-01)
Owner chose **Option 1 — ratify total-system**; mechanistic path DEFERRED. Deciding rationale:
(1) the J-reframe (PDR-0006) makes the total-system read produce a clean, decision-relevant
signal (n=3 pilot: suppressing r0c0 ~halves system efficiency); (2) the read is **r0c0-specific**
(SUPPRESS-SLOT targets r0c0), so it is "the system is less efficient without r0c0," not a weak
"needs some structure" claim; (3) the reward-design goal (credit vs penalize r0c0-like stems, and
how much) is answerable at the system level — credit r0c0's system counterfactual J; (4) the
mechanistic path blocks on an unsolved placebo/DUMMY design (v1 review: cannot bound the
action-space artifact) and per-pair synergy has not been shown necessary. The causal run's Goal-1
is therefore **system-level dependence**; the mechanistic-enabling claim is out of scope.

**Mechanistic revisit-trigger (deferred, not dead):** re-open only if the reward design turns out
to need per-pair synergy credit, or a publication-grade mechanistic claim is wanted — and even
then, a placebo/DUMMY-R0C0 design that bounds the action-space artifact must be solved first.
