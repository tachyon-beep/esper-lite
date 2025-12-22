# BUG-021: Auto-prune penalty path is dead (`step_epoch()` never returns True)

- **Title:** `pending_auto_prune_penalty` is never accrued because `SeedSlot.step_epoch()` never reports an auto-prune
- **Context:** Vectorized PPO loop expects `was_auto_pruned = slot.step_epoch()` to detect system-initiated prunes and penalize next-step reward
- **Impact:** P3 – misleading reward-shaping code and dead telemetry path; if auto-prunes are added/expected, the penalty mechanism won’t engage
- **Environment:** HEAD @ workspace; multi-slot PPO (`src/esper/simic/training/vectorized.py`)
- **Reproduction Steps:**
  1. Run PPO with `reward_config.auto_prune_penalty != 0.0`.
  2. Observe `env_state.pending_auto_prune_penalty` remains 0.0 for the entire rollout.
- **Expected Behavior:** Either:
  - A) True environment-initiated prunes should be surfaced and penalized, or
  - B) The penalty mechanism should be removed if Phase 4 intentionally eliminated auto-prunes.
- **Observed Behavior:** `SeedSlot.step_epoch()` always returns `False` and never signals a prune (even when it calls `self.prune(...)` during scheduled prune completion).
- **Logs/Telemetry:** None specific; current telemetry distinguishes auto-prunes via `SeedMetrics.auto_pruned`, but Simic doesn’t consume it for penalties.
- **Hypotheses:** Phase 4 removed auto-prunes (by design) but the Simic penalty plumbing wasn’t cleaned up; `step_epoch()`’s return contract drifted.
- **Fix Plan (choose one):**
  - A) Delete the auto-prune penalty code path in Simic and associated comments if auto-prunes are no longer a concept, or
  - B) Redefine/implement `step_epoch()`’s return value to mean “a prune occurred that was not the current step’s RL action”, and drive penalties off that, or
  - C) Drive penalties off Kasmina telemetry/state (`SeedMetrics.auto_pruned` / `SEED_PRUNED.auto_pruned==True`) instead of `step_epoch()` return value.
- **Validation Plan:** Add an integration test that triggers a non-policy prune and verifies a penalty is applied exactly once on the subsequent step.
- **Status:** Open
- **Links:**
  - Kasmina contract/docs: `src/esper/kasmina/slot.py:2004`
  - Penalty apply: `src/esper/simic/training/vectorized.py:2303`
  - Penalty accrue: `src/esper/simic/training/vectorized.py:2647`

