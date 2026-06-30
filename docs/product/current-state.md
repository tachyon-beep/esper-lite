# Current State — Esper        Checkpoint: 2026-06-30 (checkpoint #2 — causal-run arc + collapse finding)

## The bet right now
**Reward credit-assignment redesign — causal resolution of the (a)/(b) fork.** The
cheap-rescale escape hatch is CLOSED (GATE −1 verdict NO STOP / SURVIVE, PDR-0002). The active
work is the owner-gated causal-contribution run to decide freeloader (a) vs LOO-undervalued
enabling stem (b) — opposite reward fixes. **Immediate Now: fix the pervasive policy entropy
collapse (PDR-0005)** — it gates any healthy-policy causal read. Metric: restore the
policy-entropy guardrail → then committed-J / corr(reward,J).

## In flight
- **Entropy-collapse fix** (the active Now prerequisite) — scoped levers: anneal-window
  (3000 vs 200-ep), std-floor / entropy-floor penalty on the raw distribution, fp32 logits;
  validate entropy holds on a smoke before resuming the causal R1. · tracker:
  esper-lite-425dcc4ca2 (new), esper-lite-3d67b09687 (Stage-0 instrument)
- **Causal-contribution harness** — BUILT, R2-reviewed, validated on GPU (Stage-1 + R1
  full-scale offset-free), and COMMITTED (68fca06d + escrow fix e0134230) on branch
  feat/phase-minus1-scale-falsifier (**NOT pushed**). Waiting on the collapse fix. Design:
  docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md
- **The (a)/(b) fork** — still OPEN; now blocked on the collapse fix (the seed-identity
  reconstruction path is superseded by the causal-run approach).

## Open questions / blocked-on-owner
- **Causal estimand SCOPE (PDR-0004, PROPOSED — owner sign-off pending):** ratify the
  total-system pivot (Goal-1 = system-level dependence, mechanistic-enabling claim out of
  scope) OR commission a placebo/DUMMY-R0C0 design to try to keep the mechanistic claim. The
  run is estimand-invariant; this decides what it may CLAIM. *(blocked-on-owner)*
- **metrics.md TARGET numbers still `<owner-set>` placeholders** — highest-value owner input;
  no acceptance/kill fires until the committed-J north-star + guardrail floors carry a
  number+date. Same for vision.md inferred audience/secondary. *(blocked-on-owner)*
- **Policy-entropy guardrail is BREACHED** (collapse, metrics.md) — it is the active Now work
  (PDR-0005), not a kill signal yet (PDR-0005's reversal trigger fires only if the fix fails).

## Last checkpoint did
- **GATE −1 falsifier scored → NO STOP / SURVIVE** (PDR-0002); closed esper-lite-a221da47ea.
- **Commissioned + built + GPU-validated + committed the causal-contribution harness**
  (PDR-0003); estimand SCOPE flagged for owner (PDR-0004, proposed).
- **R1 pilot ran (n=3): harness validated, NO causal read (collapsed policies).** Collapse
  investigation → morphogenesis REAL (+7.69pp / n=6), collapse general + pre-existing →
  **fix the collapse first** (PDR-0005); created esper-lite-425dcc4ca2.
- Escrow config fix (prior blocked-on-owner) committed with disclosure (e0134230).

## Next session, start here
**Implement the entropy-collapse fix** (esper-lite-425dcc4ca2): the scoped levers, then a
healthy-policy smoke confirming slot-head entropy holds across the run, then resume the
causal R1 on a healthy policy (re-measure the paired-Δ SD → n=5 → n=10, the morphogenesis seed
floor). Substantive detail: docs/analysis/2026-06-30-r1-pilot-result.md + project memory
(reward-redesign-phase0-state).
