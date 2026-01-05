# Phase 0 Simic: Execution + Reward (A0 + C0)

Phase 0 adds two new lifecycle ops. Simic must execute them in the vectorized loop without adding Python bottlenecks or breaking action/obs parity.

## 1) Vectorized execution of internal ops

**Primary execution surface:** `src/esper/simic/training/vectorized.py`

Changes required:

- Add internal-op validity checks in `_parse_sampled_action(...)`:
  - op is invalid for reward if:
    - target slot is disabled
    - there is no active seed in the slot
    - seed is not a ladder seed (`internal_kind == NONE`)
    - op would not change level (already at boundary)
    - seed is terminal/frozen (recommended: disallow in `FOSSILIZED`)
  - invalid ops must be converted to `LifecycleOp.WAIT` for reward (existing pattern).

- Add execution branches alongside existing op handlers:
  - `OP_GROW_INTERNAL` → call a SeedSlot method that updates internal_level and returns success
  - `OP_SHRINK_INTERNAL` → same

**Obs parity (must remain true):**

- `env_state.last_action_success` and `env_state.last_action_op` are updated every step.
- Because `NUM_OPS` changes, feature extraction (`src/esper/tamiyo/policy/features.py`) must see the new width immediately.

## 2) Reward semantics for internal ops

**Primary reward surface:** `src/esper/simic/rewards/rewards.py`

Minimum required behavior:

- Internal ops must have explicit intervention-cost semantics in `INTERVENTION_COSTS`.
  - Default recommendation: set cost equal to `set_alpha_target_cost` (Phase 0), then tune later.
  - Alternative: set to `0.0` and rely on rent/ROI pressure; only acceptable if thrash remains low.

Where to update:

- `INTERVENTION_COSTS` dict must include the new ops, otherwise `get_intervention_cost()` returns 0 and hides drift.

## 3) Rent accounting / ROI learnability

Rent is computed via `compute_rent_and_shock_inputs()` in `src/esper/simic/training/helpers.py` using:

- cached per-slot `slot.state.metrics.seed_param_count`
- and alpha-weighted effective overhead

Implication for C0:

- When `internal_level` changes, **Kasmina must update** `SeedState.metrics.seed_param_count` to reflect the active trainable set at the new level, otherwise internal growth won’t show up in the rent signal.

## 4) Safety constraints

- **Throughput:** internal ops must be O(1) per step (simple state updates + optional requires_grad toggles).
- **Governor interactions:** internal ops increase/decrease capacity; governor rollbacks must remain coherent (no partial internal state divergence).
- **DDP symmetry:** internal ops cannot be rank-local.

