# PDR-0006 — The entropy collapse was a MEASUREMENT ARTIFACT; the (a)/(b) fork discriminator is J, not accuracy

Date: 2026-07-01   Status: accepted   Author: Claude (agent)   Owner sign-off: yes (grant — analysis + reprioritize within the active bet)
Supersedes: PDR-0005   Related: PDR-0003, metrics.md, docs/analysis/2026-06-30-r1-pilot-result.md, esper-lite-425dcc4ca2

## Context
PDR-0005 made "fix the pervasive entropy collapse" the Now prerequisite, on the premise that the R1 policies were
degenerate (wrong population for the (a)/(b) causal read). A diagnose-first ultracode workflow (4 agents + adversarial
synthesis, very-high confidence) + independent telemetry verification overturned that premise.

## What changed (the finding)
1. **NO training collapse — a telemetry/analyser artifact.** The anomaly detector (ppo_coordinator.py:619) reads
   `head_X_entropy` = per-head entropy averaged over ALL steps; on the ~60% forced/no-decision steps normalized entropy
   is EXACTLY 0, so `head_X_entropy = learnable_fraction × conditional_entropy` (a decision-DENSITY proxy). Verified
   identity: `head_slot_entropy == head_slot_learnable_fraction = 0.0872` at birth (ratio 1.0000). The slot threshold
   (0.10) sits below that diluted baseline from update 1 → ~411 anomalies are FALSE and can never clear. Decision-step
   slot entropy is HEALTHY (control 1.0→0.70, suppress 1.0→0.60, never <0.1); op (undiluted) ~0.88. The entropy bonus is
   alive (entropy_loss=0.0 is a hardcoded emitter stub); floor anti-collapse; anneal/cardinality ruled out; GATE-1
   reproduces it → structural. ⇒ the R1 population was HEALTHY-EXPLORATION; the "wrong population" block is RETRACTED.
2. **The fork discriminator is J (committed counterfactual per param), NOT accuracy.** Re-analyzing the EXISTING R1
   (no GPU): both arms tie on accuracy (~+7.7pp) but suppress spends ~2–2.7× the params for it (control 12–20 vs
   suppress 7–9 pp/M-param; median Δ −7.0). Dropping r0c0 makes the system LESS efficient — the opposite of a freeloader.

## The call
- Retract PDR-0005's "training collapse / wrong population" premise (false). The "fix collapse first" sequence is mooted.
- **Bank the METHOD:** the (a)/(b)/(c) fork is discriminated by ΔJ / Δ(acc-per-param), not accuracy.
- The n=3 PILOT direction (leans away from freeloader, toward an efficiency-enabling stem the LOO reward undervalues) is
  recorded but **NOT banked** (n=3; healthy-exploration ≠ converged; design needs n=5 to begin, n=10 floor).
- Ship the telemetry read-path swap (wire `conditional_head_entropies` to the detector; fix the `entropy_loss` stub;
  emit conditional entropy; relabel raw as density) as HYGIENE under a **bit-identical regression guard** — it changes
  no policy, only the diagnostic.

## Rationale
The verified identity + the op-head control + GATE-1 reproduction make the artifact diagnosis very-high-confidence; J is
the project's own north-star and is the metric that actually separates the fork (accuracy ties by construction here).
Banking the direction at n=3 would repeat the over-read that the diagnose-verify-reconcile discipline has caught
repeatedly this arc.

## Reversal trigger
Reopen "policy health" only if the conditional (decision-step) entropy, once wired into telemetry, shows a genuine
sub-0.1 decision-step collapse on a future run (vs the verified healthy 0.30–1.0). Reopen the fork DIRECTION (not the
method) at n=5: if ΔJ/acc-per-param does NOT stay negative (suppress less efficient) across ≥5 seeds, the enabling-stem
lean is not banked and (c) fungible / (a) freeloader return to contention.
