# Remove Legacy Telemetry Shims Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete all backwards-compat/dual-schema telemetry shims and the V3 `FastTrainingSignals` path so the codebase has a single, strict telemetry+features contract (no legacy support).

**Architecture:** Standardize on one telemetry schema (no alternate key names, no versioned fallbacks). Make “missing required fields” fail fast (raise) or fail safe (ignore) instead of silently mis-bucketing data. Add tests that assert removed legacy APIs cannot be imported, mirroring the existing “legacy API removed” pattern in `tests/test_blueprint_registry.py`.

**Tech Stack:** Python 3.11, `pytest`, `uv`, `ruff`, `mypy`.

---

## Scope

**In scope (legacy shims to remove):**
- V3 observation path: `TensorSchema`, `TENSOR_SCHEMA_SIZE`, `FastTrainingSignals`, and `TrainingSignals.to_fast()` in `src/esper/leyline/signals.py`.
- Telemetry env namespacing shim: `env_idx` fallback for `env_id` in `src/esper/karn/collector.py`.
- UI schema shims: `batch` fallback for `batch_idx`, and `rolling_avg_accuracy` / `avg_accuracy` fallback for `rolling_accuracy` in `src/esper/karn/tui.py` (BATCH_COMPLETED formatting/handling only).
- “Backwards compatibility” behavior in PPO anomaly detection: remove the “no episode info” fallback in `src/esper/simic/anomaly_detector.py`.
- Test-only “Backwards compatibility aliases” in `tests/test_stabilization_tracking.py`.

**Explicitly out of scope (needs separate decision):**
- Removing “shaped reward” as a reward family/mode (some code calls it “legacy shaped reward”, but it may still be a supported mode rather than a compatibility shim).

---

## Preflight (worktree + baseline)

### Task 0: Create a dedicated worktree

**Files:**
- None (new worktree only).

**Step 1: Create worktree**

Run:
```bash
git status --porcelain
git worktree add ../esper-lite-no-legacy-shims -b chore/remove-legacy-telemetry-shims
```

Expected:
- Clean status (or you intentionally stash/commit before proceeding).
- New worktree created at `../esper-lite-no-legacy-shims`.

**Step 2: Enter worktree and run baseline tests (spot-check)**

Run:
```bash
cd ../esper-lite-no-legacy-shims
uv run pytest -m "not slow" -q
```

Expected:
- PASS (baseline snapshot before removals).

**Step 3: Commit baseline metadata (optional)**

Run:
```bash
git commit --allow-empty -m "chore: start legacy-shim removal"
```

Expected:
- Empty commit created (optional, but helps checkpoints).

---

## V3 FastTrainingSignals removal (single features path)

### Task 1: Add failing tests that enforce V3 signals API is removed

**Files:**
- Modify: `tests/test_simic_features.py`

**Step 1: Replace the existing V3 tests with removal-enforcement tests**

Edit `tests/test_simic_features.py` to:
```python
import pytest


def test_v3_tensor_schema_removed():
    with pytest.raises(ImportError):
        from esper.leyline.signals import TensorSchema  # noqa: F401


def test_v3_tensor_schema_size_removed():
    with pytest.raises(ImportError):
        from esper.leyline.signals import TENSOR_SCHEMA_SIZE  # noqa: F401


def test_v3_fast_training_signals_removed():
    with pytest.raises(ImportError):
        from esper.leyline.signals import FastTrainingSignals  # noqa: F401


def test_leyline_no_longer_exports_v3_fast_signals():
    with pytest.raises(ImportError):
        from esper.leyline import FastTrainingSignals  # noqa: F401
```

**Step 2: Run the test to verify it fails**

Run:
```bash
uv run pytest tests/test_simic_features.py -v
```

Expected: FAIL because those imports currently succeed.

**Step 3: Commit test change**

Run:
```bash
git add tests/test_simic_features.py
git commit -m "test: enforce removal of v3 signals API"
```

---

### Task 2: Remove V3 symbols from `leyline/signals.py`

**Files:**
- Modify: `src/esper/leyline/signals.py`

**Step 1: Make the removal-enforcement tests pass by deleting V3 constructs**

In `src/esper/leyline/signals.py`:
- Delete `TensorSchema` and `TENSOR_SCHEMA_SIZE`
- Delete `FastTrainingSignals`
- Delete `TrainingSignals.to_fast(...)` (and any helpers only used by it)
- Remove now-unused imports (e.g., `IntEnum`, `NamedTuple`)
- Update the module docstring so it no longer claims a “two tier” signals design

Minimal target shape at top-of-file:
```python
"""Leyline Signals - Training state observations.

TrainingSignals is the rich, structured signals contract produced by training
and consumed by Tamiyo/Simic.
"""
```

**Step 2: Run the tests to verify they pass**

Run:
```bash
uv run pytest tests/test_simic_features.py -v
```

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add src/esper/leyline/signals.py
git commit -m "refactor(leyline): delete v3 fast signals path"
```

---

### Task 3: Remove V3 exports from `leyline/__init__.py`

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Step 1: Stop importing/exporting removed V3 symbols**

In `src/esper/leyline/__init__.py`:
- Remove `TensorSchema`, `TENSOR_SCHEMA_SIZE`, `FastTrainingSignals` from the `from esper.leyline.signals import (...)` block
- Remove them from `__all__`

**Step 2: Run the enforcement tests**

Run:
```bash
uv run pytest tests/test_simic_features.py -v
```

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add src/esper/leyline/__init__.py
git commit -m "refactor(leyline): stop exporting removed v3 signals API"
```

---

### Task 4: Remove references to `TensorSchema` in non-test code/docs

**Files:**
- Modify: `src/esper/nissa/config.py:179`

**Step 1: Update feature_count_estimate to not reference removed V3 schema**

Change:
```python
count = 35  # Base training signals features (see leyline.TensorSchema)
```

To something that doesn’t import cross-domain modules:
```python
count = 50  # Base PPO observation dims (MULTISLOT_FEATURE_SIZE)
```

**Step 2: Run a focused test/import check**

Run:
```bash
uv run python -c "from esper.nissa.config import TelemetryConfig; print(TelemetryConfig.standard().feature_count_estimate())"
```

Expected:
- Prints an integer, no ImportError.

**Step 3: Commit**

Run:
```bash
git add src/esper/nissa/config.py
git commit -m "docs(nissa): remove v3 TensorSchema reference"
```

---

### Task 5: Update architecture docs that still describe the removed V3 path

**Files:**
- Modify: `docs/arch-analysis-2025-12-13-2143/02-subsystem-catalog.md`
- Modify: `docs/arch-analysis-2025-12-13-2143/03-diagrams.md`
- Modify: `docs/arch-analysis-2025-12-13-2143/04-final-report.md`
- Modify: `docs/results/google_analysis.md`

**Step 1: Edit docs to reflect the single-path (V4) reality**

Guidelines for edits:
- Replace “FastTrainingSignals (NamedTuple) hot path” with “V4 multislot features: `esper.simic.features.obs_to_multislot_features` + per-slot telemetry padding”.
- Remove or rewrite any sections praising `TensorSchema` indices as the PPO input contract.

**Step 2: Verify no remaining references**

Run:
```bash
rg -n "FastTrainingSignals|TensorSchema|TENSOR_SCHEMA_SIZE" docs src/esper | head
```

Expected:
- No matches in `src/esper/` (docs may still have archived references if you choose not to update archives).

**Step 3: Commit**

Run:
```bash
git add docs/arch-analysis-2025-12-13-2143/02-subsystem-catalog.md
git add docs/arch-analysis-2025-12-13-2143/03-diagrams.md
git add docs/arch-analysis-2025-12-13-2143/04-final-report.md
git add docs/results/google_analysis.md
git commit -m "docs: update analysis docs for v4-only feature pipeline"
```

---

## Telemetry env namespacing: remove `env_idx` compatibility

### Task 6: Add failing test asserting `env_idx` is ignored (not supported)

**Files:**
- Modify: `tests/karn/test_collector_multienv.py`

**Step 1: Replace the legacy-acceptance test with a strictness test**

Replace `test_counterfactual_env_idx_fallback_namespaces_by_env` with:
```python
def test_counterfactual_env_idx_is_ignored_to_avoid_misbucketing():
    from esper.karn.collector import KarnCollector

    collector = KarnCollector()
    store = collector.store

    collector.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={"episode_id": "test_cf_env_idx", "max_epochs": 5, "n_envs": 2},
    ))

    # Legacy schema: env_idx only (should be ignored, not silently bucketed)
    collector.emit(TelemetryEvent(
        event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
        slot_id="mid",
        data={"env_idx": 1, "contribution": 0.9},
    ))

    slots = store.current_epoch.slots
    assert "env1:mid" not in slots
```

**Step 2: Run the test to verify it fails**

Run:
```bash
uv run pytest tests/karn/test_collector_multienv.py::TestMultiEnvSlotTracking::test_counterfactual_env_idx_is_ignored_to_avoid_misbucketing -v
```

Expected: FAIL because the current implementation accepts `env_idx`.

**Step 3: Commit**

Run:
```bash
git add tests/karn/test_collector_multienv.py
git commit -m "test(karn): forbid legacy env_idx in counterfactual events"
```

---

### Task 7: Delete the `env_idx` fallback in `KarnCollector`

**Files:**
- Modify: `src/esper/karn/collector.py:245`

**Step 1: Implement strict env_id handling**

In `src/esper/karn/collector.py`:
- In `_handle_seed_event`, replace:
  ```python
  env_id = data.get("env_id", data.get("env_idx", 0))
  ```
  with:
  ```python
  env_id = data.get("env_id")
  if env_id is None:
      return
  ```

- In `_handle_counterfactual_computed`, replace:
  ```python
  env_id = data.get("env_id", data.get("env_idx", 0))
  ```
  with the same strict pattern (`env_id` required; ignore event if missing).

**Step 2: Run the focused test**

Run:
```bash
uv run pytest tests/karn/test_collector_multienv.py::TestMultiEnvSlotTracking::test_counterfactual_env_idx_is_ignored_to_avoid_misbucketing -v
```

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add src/esper/karn/collector.py
git commit -m "fix(karn): remove env_idx telemetry compatibility shim"
```

---

## Telemetry UI schema shims: remove alternate key names

### Task 8: Add failing tests that reject alternate BATCH_COMPLETED keys

**Files:**
- Modify: `tests/karn/test_tui_state.py`

**Step 1: Add a test proving we don’t accept `rolling_avg_accuracy` for BATCH_COMPLETED**

Add:
```python
def test_batch_completed_does_not_fallback_to_rolling_avg_accuracy():
    from esper.karn.tui import TUIOutput

    tui = TUIOutput()
    tui._handle_training_started(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={"n_envs": 1, "max_epochs": 10, "task": "cifar10"},
    ))

    tui._handle_batch_completed(TelemetryEvent(
        event_type=TelemetryEventType.BATCH_COMPLETED,
        data={
            "batch_idx": 1,
            "episodes_completed": 1,
            "total_episodes": 10,
            "avg_accuracy": 50.0,
            "rolling_avg_accuracy": 60.0,  # Legacy key (should NOT be read)
            "avg_reward": 0.5,
        },
    ))

    assert tui.state.host_accuracy != 60.0
```

**Step 2: Run the test to verify it fails**

Run:
```bash
uv run pytest tests/karn/test_tui_state.py::TestMultiEnvAggregation::test_batch_completed_does_not_fallback_to_rolling_avg_accuracy -v
```

Expected: FAIL because current code falls back to `rolling_avg_accuracy`.

**Step 3: Commit**

Run:
```bash
git add tests/karn/test_tui_state.py
git commit -m "test(karn): forbid legacy BATCH_COMPLETED key fallbacks"
```

---

### Task 9: Remove BATCH_COMPLETED key fallbacks in the TUI formatter + handler

**Files:**
- Modify: `src/esper/karn/tui.py:589`
- Modify: `src/esper/karn/tui.py:853`

**Step 1: Implement strict key usage**

In `src/esper/karn/tui.py`:
- In `_format_event_for_log` for `BATCH_COMPLETED`, replace:
  ```python
  batch_idx = data.get("batch_idx", data.get("batch", "?"))
  rolling_acc = data.get("rolling_accuracy", data.get("rolling_avg_accuracy", data.get("avg_accuracy")))
  ```
  with:
  ```python
  batch_idx = data.get("batch_idx", "?")
  rolling_acc = data.get("rolling_accuracy")
  ```

- In `_handle_batch_completed`, replace:
  ```python
  current_acc = data.get("rolling_accuracy", data.get("rolling_avg_accuracy", data.get("avg_accuracy", 0.0)))
  ```
  with:
  ```python
  current_acc = data.get("rolling_accuracy", 0.0)
  ```

**Step 2: Run the new strictness test**

Run:
```bash
uv run pytest tests/karn/test_tui_state.py::TestMultiEnvAggregation::test_batch_completed_does_not_fallback_to_rolling_avg_accuracy -v
```

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add src/esper/karn/tui.py
git commit -m "refactor(karn): remove legacy BATCH_COMPLETED schema fallbacks"
```

---

## PPO anomaly detector: remove backwards-compat “unknown total episodes” path

### Task 10: Make tests require episode info for EV thresholding

**Files:**
- Modify: `tests/simic/test_anomaly_detector.py`

**Step 1: Replace backwards-compat tests with strictness tests**

Replace `test_backwards_compatible_without_episode_info` with:
```python
import pytest


def test_requires_episode_info_for_value_collapse_thresholds():
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.check_value_function(explained_variance=0.05)
```

Replace `test_value_collapse_detail_shows_unknown_when_no_total` with:
```python
import pytest


def test_requires_total_episodes_for_value_collapse_thresholds():
    detector = AnomalyDetector()
    with pytest.raises(ValueError):
        detector.check_value_function(
            explained_variance=0.05,
            current_episode=10,
            total_episodes=0,
        )
```

**Step 2: Run the test file to verify failures**

Run:
```bash
uv run pytest tests/simic/test_anomaly_detector.py -v
```

Expected: FAIL until implementation is updated.

**Step 3: Commit tests**

Run:
```bash
git add tests/simic/test_anomaly_detector.py
git commit -m "test(simic): require episode context for anomaly thresholds"
```

---

### Task 11: Enforce strict episode context in `AnomalyDetector`

**Files:**
- Modify: `src/esper/simic/anomaly_detector.py:62`

**Step 1: Remove fallbacks and raise on missing episode info**

Update `get_ev_threshold`:
```python
if total_episodes <= 0:
    raise ValueError("total_episodes must be > 0 for phase-dependent EV thresholds")
```

Update `check_value_function`:
- Remove `current_episode: int = 0, total_episodes: int = 0` defaults
- Require explicit args and raise if invalid:
```python
if current_episode <= 0 or total_episodes <= 0:
    raise ValueError("current_episode and total_episodes are required (> 0)")
```

Update `check_all`:
- Remove default values for `current_episode` and `total_episodes` and require callers to pass them.

**Step 2: Update call sites**

`src/esper/simic/vectorized.py` already calls `check_all(... current_episode=..., total_episodes=...)`.
Update any remaining test call sites (e.g., `tests/simic/test_anomaly_detector.py`) to pass explicit episode info where appropriate (besides the strictness tests).

**Step 3: Run tests**

Run:
```bash
uv run pytest tests/simic/test_anomaly_detector.py -v
```

Expected: PASS.

**Step 4: Commit**

Run:
```bash
git add src/esper/simic/anomaly_detector.py
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): remove backwards-compat anomaly threshold fallback"
```

---

## Test cleanup: remove “Backwards compatibility aliases”

### Task 12: Delete alias constants in stabilization tracking tests

**Files:**
- Modify: `tests/test_stabilization_tracking.py:9`

**Step 1: Delete the alias block**

Remove:
```python
# Backwards compatibility aliases
STABILIZATION_THRESHOLD = DEFAULT_STABILIZATION_THRESHOLD
STABILIZATION_EPOCHS = DEFAULT_STABILIZATION_EPOCHS
```

And update usages to reference:
- `DEFAULT_STABILIZATION_THRESHOLD`
- `DEFAULT_STABILIZATION_EPOCHS`

**Step 2: Run the file’s tests**

Run:
```bash
uv run pytest tests/test_stabilization_tracking.py -v
```

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add tests/test_stabilization_tracking.py
git commit -m "test: remove backwards-compat alias constants"
```

---

## Final verification

### Task 13: Grep-based “no legacy shims” sanity check

**Files:**
- None

**Step 1: Search for known legacy shim patterns**

Run:
```bash
rg -n "env_idx|Backwards compatibility|backwards compatibility" src/esper tests | head -n 200
rg -n "rolling_avg_accuracy\"\\)" src/esper/karn/tui.py
rg -n "FastTrainingSignals|TensorSchema|TENSOR_SCHEMA_SIZE" src/esper tests | head -n 200
```

Expected:
- No remaining `env_idx` fallbacks in `src/esper`.
- No remaining `FastTrainingSignals`/`TensorSchema` symbols in `src/esper`.
- Tests may still mention “compatibility” in historical contexts, but not as active shims.

---

### Task 14: Run CI-parity checks

**Files:**
- None

**Step 1: Tests**

Run:
```bash
uv run pytest
```

Expected: PASS.

**Step 2: Lint**

Run:
```bash
uv run ruff check src/ tests/
```

Expected: PASS.

**Step 3: Types**

Run:
```bash
uv run mypy src/
```

Expected: PASS.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-12-16-remove-legacy-telemetry-shims.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) — Open new session with executing-plans, batch execution with checkpoints

Which approach?

