# PDR-0070 — Next bet: differentiable mixture-floor fix (staged experiment) + harness enablers

Date: 2026-07-13   Status: **superseded by PDR-0071** (two-build staged experiment → one primitive + λ-sweep; ship path → co-requisite with `hindsight_credit`). The harness enablers (accepted, no GPU) carry forward unchanged.
Related: PDR-0069 (root cause), PDR-0066 (critic is wrong lever). Analysis: `docs/analysis/2026-07-13-decision-point-diagnosis.md`. Memory: `floor-gradient-dead-zone`.

## Context
PDR-0069 root-caused suppressed commitment to the anti-WAIT floor gradient dead-zone. The fix is a change to the exploration primitive. Both external reviewers (claude-prime, gpt-prime) converged on the primitive and the experiment staging, and both rejected the straight-through estimator I first proposed.

## What does this buy, and how is that measured?  (REQUIRED — PDR-0068)
A floor that prevents WAIT-collapse WITHOUT censoring the gradient of the actions it forces — so the policy can finally learn the commit decision from the reward it already receives. Establishing whether it does is the experiment. Primary (mechanism) metrics: zero-gradient/floor-bound sample rate (target: near-zero), per-head floor-binding rates, current-LOO sensitivity of the FOSSILIZE/PRUNE logits, WAIT share on genuine multi-choice states. Guardrail (product) metrics: terminal accuracy contribution, parameter footprint, developmental compute, destructive-intervention rate. **Do NOT score "more fossilization" as success by itself** — current LOO still cannot measure transferred/retained value (that is the separate track).

## Options considered
- **Straight-through estimator through the floor** — REJECTED (both primes): back-props the unfloored gradient while sampling the floored distribution → biased PPO importance ratio; the exact class of bug this codebase is made of.
- **Differentiable mixture floor** — `q = (1−λ)·softmax(z_masked) + λ·uniform(legal)`; guarantees `q_i ≥ λ/n`, smooth, exact `log q` for a valid ratio, non-zero gradient everywhere. Behaviourally near-identical cap (λ=0.6,n=4 → 0.15 floor / 0.55 cap).
- **Smooth `P(non-WAIT) ≥ ε`** (gpt-prime) — reserve ε of WAIT's mass, redistribute across non-WAIT by LEARNED relative probs; no per-op floor on irreversible actions; closest to the original anti-WAIT intent. Subsumes the mixture as a special case.

## The call (proposed)
1. **Fix primitive:** a differentiable floor — mixture or smooth `P(non-WAIT)≥ε` — NOT straight-through. Requires **drl-expert review** before landing (a change to the action distribution / PPO ratio is exactly what caused this arc).
2. **Staged experiment (separate the two questions — gpt-prime):**
   - **Stage 1 — no-floor NECESSITY smoke:** current reward + code, probability floor DISABLED, frozen abort (e.g. WAIT share > 95% on genuine multi-choice op states for N updates). Tests whether the Dec-2025 99.9%-WAIT pathology still exists after the reward changed ≥twice. No collapse ⇒ the hard floor may be obsolete. Collapse ⇒ anti-WAIT intervention still needed → Stage 2.
   - **Stage 2 — paired A/B (n=5):** hard per-action floor (control) vs differentiable floor (treatment), fresh-init paired seeds.
3. **Harness enablers (accepted, no GPU, land first):** default-ON checkpointing (weights + optimiser + normaliser + RNG + config, at milestones not just final) and per-decision diagnostic logging (pre/post-floor op logits + advantage + effective floor + floor-bound mask). The seed41/42 weights + advantage + pre-floor logits were NOT captured this session — the same PDR-0026 "coverage-bound" failure — so the advantage-vs-LOO gate and pre-floor policy were unreadable.

## Before-a-run check  (PDR-0067 G1 / PDR-0068)
Already on disk: the reward-vs-LOO gate (−0.7→+8.2) is a strong proxy that the commit reward signal is live, but the TRUE gate (per-decision advantage-vs-LOO) is not logged — the Stage-2 run's pre-committed reading must add it. Remaining zero-GPU reads to run BEFORE any GPU arm: entropy-coefficient audit (rule out uniform-forcing as a competing cause) and whether floor-bound zero-gradient samples still enter global advantage standardisation. Pre-committed reading gets the PDR-0066 pre-launch adversarial review.

## Reversal trigger
- Stage-1 no collapse → skip Stage-2's anti-WAIT assumption; investigate whether any floor is needed at all.
- Stage-2 treatment shows no commit-selectivity improvement despite near-zero floor-binding AND a logged live advantage-vs-LOO → the dead-zone was not binding; re-open PDR-0069's alternative (critic/reward line) with a measured baseline.
- Terminal product guardrails (accuracy contribution / destructive-intervention rate) regress under the differentiable floor → the fix trades collapse-safety for instability; revert and reconsider ε/λ or a state-dependent floor.
