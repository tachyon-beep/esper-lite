# BUG-010: Heuristic embargo tracked globally, not per slot

- **Title:** Heuristic embargo/cull cooldown tracked globally, not per slot
- **Context:** Tamiyo heuristic (`src/esper/tamiyo/heuristic.py`)
- **Impact:** P3 – Design smell in baseline mode, no production impact
- **Environment:** Main branch
- **Status:** Deferred (downgraded from P1)

## Analysis (2025-12-17)

**Low priority.** The bug is real but impact is minimal:

### 1. Heuristic is a Baseline, Not Production

The heuristic mode (`train_heuristic`) is a comparison baseline, not production code.
PPO (`train_ppo`) is the production path and doesn't have this issue - it learns
optimal germination timing through rewards, not explicit embargo rules.

### 2. Heuristic Already Limited to One Action Per Epoch

The heuristic makes ONE decision per `decide()` call (one per epoch). Even with
multiple empty slots, it would only germinate in one slot per epoch anyway.

```python
# training.py lines 506-509
germinate_slot = next(
    (slot_id for slot_id in enabled_slots if not model.has_active_seed_in_slot(slot_id)),
    None,
)
```

### 3. No Multi-Slot Heuristic Tests Exist

There are no tests for multi-slot heuristic behavior, suggesting this
scenario isn't a supported use case.

## The Bug (Low Priority)

`_last_cull_epoch` is a single int instead of per-slot dict:

```python
# Current (global)
self._last_cull_epoch: int = -100

# Correct (per-slot)
self._last_cull_epoch: dict[str, int] = {}
```

This means culling in slot A blocks germination in slot B until embargo expires.

## Future Fix (If Needed)

Change `_last_cull_epoch` from `int` to `dict[str, int]` and:
1. Track cull epoch per slot_id
2. Pass target slot to `_decide_germination`
3. Check embargo only for the target slot

This would require design changes to pass slot context through the decision flow.

## Links

- `src/esper/tamiyo/heuristic.py` (embargo logic in `_decide_germination`, `_cull_seed`)
- Production path: `src/esper/simic/vectorized.py` (no explicit embargo - learns via rewards)
