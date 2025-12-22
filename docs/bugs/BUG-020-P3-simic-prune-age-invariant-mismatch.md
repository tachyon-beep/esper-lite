# BUG-020: `MIN_PRUNE_AGE` enforced in masks but not in execution/validity checks

- **Title:** `MIN_PRUNE_AGE` is applied in `compute_action_masks` but not enforced in Simic’s `PRUNE` execution branch or reward-validity logic
- **Context:** Tamiyo masks treat “prunable” as `(stage allows PRUNED) ∧ (alpha_mode==HOLD) ∧ (seed_age_epochs>=MIN_PRUNE_AGE)`
- **Impact:** P3 – contract mismatch; if masks drift or are bypassed (debug/heuristic/testing), Simic can execute early prunes the mask would have blocked
- **Environment:** HEAD @ workspace; multi-slot PPO (`src/esper/simic/training/vectorized.py`)
- **Reproduction Steps:**
  1. Create a seed with `metrics.epochs_total==0` in a prunable stage and `alpha_mode==HOLD`.
  2. Bypass masking and force `op=PRUNE`.
  3. Simic will allow the prune (stage/alpha checks pass) even though the mask would have blocked it.
- **Expected Behavior:** Execution-time legality checks should match masking invariants for safety and future-proofing.
- **Observed Behavior:** Execution and reward-validity checks omit `MIN_PRUNE_AGE`.
- **Logs/Telemetry:** None specific; manifests as unexpected early prunes if invalid actions slip through.
- **Hypotheses:** `MIN_PRUNE_AGE` was added to masks as a policy regularizer but wasn’t propagated into the env legality checks.
- **Fix Plan:** Add `seed_state.metrics.epochs_total >= MIN_PRUNE_AGE` to:
  - `action_valid_for_reward` for `PRUNE`, and
  - `OP_PRUNE` execution branch gate (alongside HOLD + can_transition checks).
- **Validation Plan:** Unit test that `PRUNE` is rejected (treated as invalid/WAIT) when `epochs_total < MIN_PRUNE_AGE`.
- **Status:** Open
- **Links:**
  - Masking invariant: `src/esper/tamiyo/policy/action_masks.py:220`
  - Reward-validity: `src/esper/simic/training/vectorized.py:2140`
  - Execution gate: `src/esper/simic/training/vectorized.py:2374`
  - Constant: `src/esper/leyline/__init__.py:33`

