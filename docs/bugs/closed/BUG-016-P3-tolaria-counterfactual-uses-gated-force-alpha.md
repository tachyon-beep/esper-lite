# BUG-016: Counterfactual validation uses mutation-based force_alpha (not a bug)

- **Title:** Counterfactual validation uses mutation-based `force_alpha`, unsafe for future DDP
- **Context:** Tolaria `validate_with_attribution` uses `seed_slot.force_alpha(0.0)` (mutation) to compute baseline
- **Impact:** P3 – Design smell / future tech debt. No current production impact.
- **Environment:** Main branch
- **Status:** Closed (Not a bug)
- **Resolution:** `force_alpha` is explicitly documented as not thread-safe/DDP-safe and is only used in non-production attribution helpers/tests; production training uses per-env model instances so mutations are isolated.

## Analysis (2025-12-17)

**Not an active bug.** Investigation found:

1. **`validate_with_attribution` is not used in production code** - Only defined, exported, and tested
2. **Vectorized path (production) is safe** - Each environment has its own model instance via `create_model()`, so `force_alpha` mutations are isolated
3. **Docstring already documents the limitation** - Clear warnings about thread safety

## Current Safe Patterns

| Code Path | Safety | Reason |
|-----------|--------|--------|
| `validate_with_attribution` | N/A | Not called in production |
| Vectorized counterfactual | Safe | Each env has own model instance |
| Heuristic training | Safe | Single-threaded |

## Future Consideration

When implementing DDP support, counterfactual evaluation will need redesign:

**Options:**
1. **Functional alpha override** - Pass alpha to forward() instead of mutating state
2. **Thread-local storage** - Isolate force_alpha per thread/process
3. **Model cloning** - Create temporary model copy for baseline evaluation

## Links

- `src/esper/tolaria/trainer.py::validate_with_attribution` (unused but exported)
- `src/esper/kasmina/slot.py::force_alpha` (has TODO for DDP)
- `src/esper/simic/training/vectorized.py` (safe implementation with per-env models)
