# Current State — Esper        Checkpoint: 2026-07-02 (checkpoint #6 — reward-credit term BUILT default-OFF; enablement is the next gate)

## The bet right now
**Reward credit-assignment redesign.** The Committed-Shapley top-up is **BUILT and ACCEPTED
(PDR-0013)**: owner signed off in-session, same-day plan → specialist review → TDD build → verification
(reviewer re-pass APPROVE_WITH_CHANGES with its one change closed in-build; adversarial code review
CLEAN; ~100 new tests; suites green except a pre-existing Phase 0 item). `shapley_synergy_scale=0.0`
everywhere — nothing is enabled. Metric it moves: committed-J / corr(reward,J) in the enablement A/B.

## In flight
- **Enablement gate** (esper-lite-f22a1d48a7, open, owner-gated): tau from the PIN-E placebo
  (esper-lite-3d67b09687), F2 scale/cap/normalized_cap calibration, HARD off-switch-J efficiency +
  fossilize-count gate, entrenchment monitor, ON-run sibling-dormancy recheck, then the paired
  ≥5-seed OFF/ON A/B. Carries the build-time residuals (terminal-vs-t_f std drift on the credit
  divisor; cuDNN terminal-val_acc confound at scale>0).
- **Telemetry hygiene** (esper-lite-425dcc4ca2): observability-emit follow-up — unchanged.

## Open questions / blocked-on-owner
- **START THE ENABLEMENT GATE?** Its first executable leg is the PIN-E placebo harness
  (esper-lite-3d67b09687, GATE-0's last open leg) — needed for tau regardless.
- **n=10 vs bank-at-n=5 (MAGNITUDE, PDR-0009):** still unanswered; independent of the build.
- **metrics.md TARGET numbers** still `<owner-set>` placeholders.
- **Owner's working tree:** phase-0 doc/CLAUDE.md/AGENTS.md/.gitignore edits remain uncommitted
  (untouched by the agent). Related: Phase 0 seed_residency test-expectation gap filed as
  esper-lite-obs-7240279be3.

## Last checkpoint did (checkpoint #6)
- **Built the term (PDR-0013)** across 7ec67de7 + a479d776..2d7e0db5: exact 2^k committed-coalition
  Shapley in the terminal fused pass; retro-write at t_f pre-GAE (min(normalized_cap,
  top_up/max(std, std_floor))); five default-0.0 flags; F1/F8 sibling mutual exclusion; HRA hard
  exclusion; `synergy_bonus`→`interaction_bonus` rename (No-Legacy); COMMITTED_SHAPLEY_TOPUP telemetry.
- **Process:** 3 scout agents → plan (drl+pytorch approved-with-changes; both caught the same
  divide_by_std API hole before code existed) → TDD (mandated GAE test first) → WI-8/WI-9 delegated →
  reviewer re-pass + adversarial code review.
- Tracker: esper-lite-254175df90 CLOSED (close_commit 2d7e0db5); enablement successor
  esper-lite-f22a1d48a7 created; plan moved to docs/plans/completed/; design doc amended (c_paid
  semantic shift); PLAN_TRACKER updated.

## Next session, start here
**The enablement gate is the bet's critical path**: build the PIN-E placebo harness
(esper-lite-3d67b09687) to set tau — everything else in esper-lite-f22a1d48a7 hangs off it. The n=10
magnitude run remains available in parallel if the owner wants the (b) effect size banked.
