# Current State — Esper        Checkpoint: 2026-07-02 (checkpoint #4 — n=5 banks (b); reward-credit design gated)

## The bet right now
**Reward credit-assignment redesign — design the credit term for the enabling stem.** The (a)/(b) fork is RESOLVED:
the n=5 J-read banks **(b)** — r0c0 is an efficiency-enabling stem (suppressing it ~halves system param-efficiency, 5/5
seeds; PDR-0009), and the optimizer is adequate so the defect is reward-side (PDR-0008). The reward must CREDIT the
enabling contribution the per-step LOO undervalues. Metric it moves: committed-J / corr(reward,J) once the term is A/B'd.

## In flight
- **Reward-credit term** — a default-OFF Committed-Shapley synergy top-up (PDR-0010, surviving candidate; design
  `docs/plans/concepts/2026-07-01-reward-credit-shapley-synergy-design.md`). GATE 1 (proxy-vs-estimand) PASSED (r0c0
  terminally load-bearing, not scaffolding); mirror reassuring (terminal-drop selective) but entrenchment is a monitored
  gate. **Next concrete step: GATE 2 — learnability probe** (can a once-per-episode terminal credit propagate to the
  FOSSILIZE action?). · tracker: esper-lite-254175df90
- **Telemetry hygiene** (esper-lite-425dcc4ca2): detector read-path fix LANDED (a84f9a78); observability-emit follow-up
  (relabel raw → density, emit conditional) still open. Also LANDED: the seed-44 telemetry crash fix (db61dc3a).

## Open questions / blocked-on-owner
- **n=10 vs bank-at-n=5 (MAGNITUDE):** (b) direction is banked (5/5, sign-test p≈0.03); the "CI" is a sign test, not a
  CI. Extending to n=10 (the morphogenesis floor) is the only way to bank the effect *size*. **Owner's call.**
- **Reward-term ENABLEMENT is gated:** the Shapley top-up stays default-OFF (shapley_synergy_scale=0.0). Enabling it
  (a reward-behavior change) needs GATE 2 pass + reward-function-reviewer + **owner sign-off** — do NOT enable
  autonomously. *(blocked-on-owner)*
- **metrics.md TARGET numbers** still `<owner-set>` placeholders (committed-J north-star, guardrail floors).
- Verify before wiring the term: which signal `fossilize_contribution_scale` multiplies (SEED_FOSSILIZED.counterfactual
  reads ~10 for r0c0 = joint, not the ~0 per-seed LOO the "undervalued" premise is about).

## Last checkpoint did (checkpoint #4)
- **n=5 J-read → (b) banked** at total-system scope (PDR-0009; all 10 runs, 5/5 negative Δeff, gates pass); closed
  esper-lite-8190ff1c95.
- **Advantage-pathology track CLOSED** — optimizer adequate, defect reward-side (PDR-0008), confirming the redesign
  targets the right layer.
- **Reward-credit design** — Committed-Shapley top-up is the surviving candidate (PDR-0010); GATE 1 + mirror run on the
  n=5 data; created esper-lite-254175df90.
- Fixed the seed-44 telemetry crash (db61dc3a; a diagnostic must not crash training) — n=5 completed clean.

## Next session, start here
**GATE 2 — the reward-credit learnability probe** (esper-lite-254175df90): does a synthetic once-per-episode terminal
top-up on a FOSSILIZE action measurably move that action's advantage in this recurrent PPO? Cheap, no GPU dependency,
and it decides whether candidate A can even propagate before anyone builds the Shapley apparatus. If it fails → pivot to
candidate B (temporal/hindsight credit). In parallel, the owner's two calls above (n=10; enablement sign-off).
