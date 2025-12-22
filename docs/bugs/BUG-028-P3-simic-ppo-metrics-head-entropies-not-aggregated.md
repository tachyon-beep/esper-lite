# BUG-028: PPO metrics aggregation drops per-head histories (uses first update only)

- **Title:** `_aggregate_ppo_metrics()` claims to aggregate `head_entropies` but returns only the first update’s values (and `head_grad_norms` also implicitly becomes “first update only”)
- **Context:** Vectorized PPO training when `ppo_updates_per_batch > 1`; aggregated metrics feed telemetry (`PPO_UPDATE_COMPLETED`) and dashboards
- **Impact:** P3 – observability bug. Per-head entropy/grad-norm telemetry can be misleading, making exploration-collapse debugging and head-dominance diagnosis unreliable.
- **Environment:** HEAD @ workspace; any run using multiple PPO updates per batch (`ppo_updates_per_batch >= 2`)
- **Reproduction Steps:**
  1. Call `_aggregate_ppo_metrics([{"head_entropies": {"slot": [1.0]}}, {"head_entropies": {"slot": [2.0]}}])`.
  2. Observe the result contains `{"head_entropies": {"slot": [1.0]}}` (second update dropped).
  3. Repeat for `head_grad_norms` and observe only the first dict is kept due to the generic `dict` branch.
- **Expected Behavior:** Either:
  - A) concatenate per-head lists across updates, or
  - B) compute a true aggregate (e.g., mean over updates) and log that explicitly.
- **Observed Behavior:** `head_entropies` path returns `values[0]` despite comment saying “concatenate”; `head_grad_norms` falls through to `dict` handling and also returns `values[0]`.
- **Logs/Telemetry:** `PPO_UPDATE_COMPLETED` fields like `slot_entropy` reflect only the first PPO update in the batch.
- **Hypotheses:** Aggregator implementation drift during multi-update support; comment updated but logic not.
- **Fix Plan:** Update `_aggregate_ppo_metrics()` to merge per-head dicts:
  - For `head_entropies`/`head_grad_norms`: for each head key, concatenate lists across updates (or compute an aggregate statistic) and return the merged dict.
  - Ensure telemetry emits values that match the aggregator semantics.
- **Validation Plan:**
  - Unit test for `_aggregate_ppo_metrics()` covering multi-update aggregation of `head_entropies` and `head_grad_norms`.
  - Smoke PPO run with `ppo_updates_per_batch=2` and confirm telemetry head entropies change when the second update differs.
- **Status:** Open
- **Links:**
  - Aggregator: `src/esper/simic/training/vectorized.py:255`

