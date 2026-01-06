# BUG-001: Heuristic training API fails when `slots` defaults to `None`

- **Title:** Heuristic `train_heuristic()` defaults `slots=None` and throws before training
- **Context:** Simic heuristic training (`src/esper/simic/training/helpers.py::train_heuristic`, `run_heuristic_episode`)
- **Impact:** Medium – programmatic callers hit `ValueError` before training starts; inconsistent with the CLI which supplies default slot IDs.
- **Environment:** Main branch; `PYTHONPATH=src`; reproducible on CPU
- **Status:** FIXED (2025-12-17)

## Reproduction (pre-fix)

```bash
PYTHONPATH=src uv run python - <<'PY'
from esper.simic.training.helpers import train_heuristic
train_heuristic(n_episodes=1, max_epochs=1, max_batches=1, device="cpu")
PY
```

## Root Cause Analysis

`train_heuristic(..., slots=None)` forwarded `slots=None` into `run_heuristic_episode()`,
which then called `create_model(..., slots=None)`. Model creation requires a non-empty
slot list, so training failed before any episode work could begin.

## Fix

- Default `slots=None` to `SlotConfig.default().slot_ids` in:
  - `src/esper/simic/training/helpers.py::train_heuristic`
  - `src/esper/simic/training/helpers.py::run_heuristic_episode`
- Validate `slots` is non-empty and deduplicated before calling `create_model()`.

This mirrors the heuristic CLI defaults (`--slots r0c0 r0c1 r0c2` in `src/esper/scripts/train.py`).

## Validation

```bash
PYTHONPATH=src uv run python - <<'PY'
from esper.simic.training.helpers import train_heuristic
train_heuristic(n_episodes=1, max_epochs=1, max_batches=1, device="cpu")
print("OK")
PY

PYTHONPATH=src uv run python -m esper.scripts.train heuristic --episodes 1 --max-epochs 1 --max-batches 1 --device cpu
```

## Links

- Fix: `src/esper/simic/training/helpers.py`
- CLI defaults: `src/esper/scripts/train.py`
