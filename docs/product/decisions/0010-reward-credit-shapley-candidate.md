# PDR-0010 — Reward-credit design: Committed-Shapley synergy top-up is the surviving candidate (gated)

Date: 2026-07-02   Status: accepted as a CANDIDATE DESIGN (default-OFF; ENABLING the reward-behavior change is FLAGGED for owner + reward-function-reviewer sign-off)   Author: Claude (agent)
Supersedes: —   Related: PDR-0009, PDR-0006, PDR-0004, docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md

## Context
(b) banked (PDR-0009) ⇒ the reward must credit r0c0's enabling contribution without re-opening the freeloader hole (the
per-step LOO the reward credits by ignores synergy — phase0 §1.2). Designed via a workflow (3 candidates:
Shapley-synergy / hindsight-developmental / efficiency-frontier; B+C stalled on infra) + adversarial synthesis + advisor
reconcile.

## The call
**Candidate A — a default-OFF, terminal, additive Committed-Shapley top-up — is the SURVIVING candidate.** Shapley's
null-player axiom IS the freeloader guard (an inert committed seed scores 0 exactly); an episode-level efficiency clamp
bounds the credit pot. **GATE 1 (proxy-vs-estimand) PASSED:** r0c0's terminal leave-out drop is large on all 5 seeds
(median 9.8pp) → terminally load-bearing, NOT the scaffolding archetype that would make the terminal proxy miss it. The
**mirror (over-credit)** is reassuring (terminal-drop is selective, not uniform) but NOT fully cleared (the clean config
has no known entrenched-worthless seed to stress it) → entrenchment stays a monitored gate. A survives over candidate B
(temporal/hindsight — the pre-identified salvage had GATE 1 failed; it didn't).

## Gates remaining before it becomes "the design" — and before ANY enablement
- **GATE 2 — learnability (UNRUN):** does a once-per-episode terminal credit to a long-past FOSSILIZE action propagate
  to that action's advantage? (Per PDR-0008, "optimizer adequate" does NOT transfer to this sparse-terminal regime.)
- Then **reward-function-reviewer + owner sign-off.** The flag stays default-OFF throughout.
- **Verification owed before wiring:** which signal does `fossilize_contribution_scale` multiply?
  `SEED_FOSSILIZED.counterfactual` reads ~10 for r0c0 (a joint/ensemble measure), not the ~0 per-seed LOO the
  "undervalued" premise is about.

## Reversal trigger
Drop A for B (temporal/hindsight credit) if GATE 2 fails, OR if an entrenchment mirror on a run with a known-worthless
seed shows terminal-drop cannot separate enabling from entrenched. **Never enable the term** (shapley_synergy_scale > 0)
without GATE 2 pass + owner + reward-function-reviewer sign-off.
