# Remove Legacy V3 Signals (FastTrainingSignals/TensorSchema) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Remove the legacy V3 flat observation-vector API (`FastTrainingSignals`, `FastTrainingSignals.to_vector()`, `TensorSchema`, `TENSOR_SCHEMA_SIZE`, and `TrainingSignals.to_fast()`), leaving only the current multi-slot PPO observation path (`simic.features` + `simic.ppo.signals_to_features`).

**Architecture:** Make “no legacy V3 signals API” a test-enforced contract, then delete the V3 implementation and its exports (no shims). Update any comments/docs that reference the removed symbols so the codebase stays truthful pre-release.

**Tech Stack:** Python 3.11+, pytest, ruff, git.

---

## Preconditions

- Start from a clean working tree.
- If you want isolation: create a worktree with `superpowers:using-git-worktrees`.

---

### Task 1: Add a failing test that forbids the V3 API (RED)

**Files:**
- Create: `tests/leyline/test_no_legacy_v3_signals.py`
- Modify: `src/esper/leyline/__init__.py`
- Delete: `tests/test_simic_features.py`

**Step 1: Write a failing test for *public exports only***

```python
import importlib


def test_leyline_does_not_export_v3_vector_primitives() -> None:
    leyline = importlib.import_module("esper.leyline")
    assert getattr(leyline, "FastTrainingSignals", None) is None
    assert getattr(leyline, "TensorSchema", None) is None
    assert getattr(leyline, "TENSOR_SCHEMA_SIZE", None) is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest -q tests/leyline/test_no_legacy_v3_signals.py`

Expected: FAIL (attributes currently exist).

**Step 3: Make it pass (remove public exports + delete legacy tests)**

1. In `src/esper/leyline/__init__.py`, remove imports and `__all__` entries for:
   - `FastTrainingSignals`
   - `TensorSchema`
   - `TENSOR_SCHEMA_SIZE`
2. Delete `tests/test_simic_features.py` (it asserts V3 indices and calls `to_vector()`).

**Step 4: Run tests to verify GREEN**

Run: `uv run pytest -q tests/leyline/test_no_legacy_v3_signals.py`

Expected: PASS.

**Step 5: Commit (green only)**

Run:
```bash
git add -A
git commit -m "refactor(leyline): remove legacy v3 signals exports"
```

---

### Task 2: Delete the V3 implementation from Leyline (legacy code removal)

**Files:**
- Modify: `src/esper/leyline/signals.py`
- Modify: `src/esper/tamiyo/tracker.py`
- Modify: `tests/tamiyo/test_tracker_unit.py`
- Modify: `tests/leyline/test_no_legacy_v3_signals.py`

**Step 1: Extend the test to forbid V3 *internal* symbols (RED)**

Add this second test to `tests/leyline/test_no_legacy_v3_signals.py`:

```python
def test_signals_module_has_no_v3_vector_primitives() -> None:
    signals = importlib.import_module("esper.leyline.signals")
    assert getattr(signals, "FastTrainingSignals", None) is None
    assert getattr(signals, "TensorSchema", None) is None
    assert getattr(signals, "TENSOR_SCHEMA_SIZE", None) is None
```

**Step 2: Run tests to verify RED**

Run: `uv run pytest -q tests/leyline/test_no_legacy_v3_signals.py`

Expected: FAIL (those symbols still exist in `esper.leyline.signals`).

**Step 3: Delete the V3 implementation + dead wiring (GREEN)**

1. In `src/esper/leyline/signals.py`, delete:
   - `class TensorSchema(IntEnum): ...`
   - `TENSOR_SCHEMA_SIZE = ...`
   - `class FastTrainingSignals(NamedTuple): ...` (including `empty()` and `to_vector()`)
   - `TrainingSignals.to_fast(...)` (it only exists to create `FastTrainingSignals`)
2. Remove the now-dead `TrainingSignals.seed_counterfactual` field if it is only used for the deleted V3 path.
3. In `src/esper/tamiyo/tracker.py`, remove any `seed_counterfactual` wiring and stop passing it into `TrainingSignals(...)`.
4. In `tests/tamiyo/test_tracker_unit.py`, remove the unit test that asserted `signals.seed_counterfactual` is populated.

**Step 4: Run tests to verify GREEN**

Run:
- `uv run pytest -q tests/leyline/test_no_legacy_v3_signals.py`
- `uv run pytest -q tests/tamiyo/test_tracker_unit.py`

Expected: PASS.

**Step 5: Commit (green only)**

Run:
```bash
git add -A
git commit -m "refactor(leyline): delete legacy v3 signals api"
```

---

### Task 3: Remove stale references to TensorSchema in telemetry config

**Files:**
- Modify: `src/esper/nissa/config.py`

**Step 1: Update `feature_count_estimate()` comment**

Replace the `leyline.TensorSchema` reference (since it no longer exists) with a truthful statement, e.g.:
- “Base feature count is task/observation-path specific; this estimate covers telemetry add-ons.”

Keep behavior unchanged unless you find a real user of this estimate.

**Step 2: Run ruff**

Run: `uv run ruff check src/ tests/`

Expected: PASS.

**Step 3: Commit**

Run:
```bash
git add src/esper/nissa/config.py
git commit -m "docs(nissa): remove stale TensorSchema reference"
```

---

### Task 4: Docs cleanup (non-archive docs that reference removed symbols)

**Files:**
- Modify: `docs/results/google_analysis.md` (and any other non-archive docs found by grep)

**Step 1: Find references**

Run: `rg -n "FastTrainingSignals|TensorSchema|TENSOR_SCHEMA_SIZE|to_vector\\(" docs/`

**Step 2: Update or annotate**

For docs that are meant to stay current (not `docs/*/archive/*`), update to describe the current observation path:
- `src/esper/simic/features.py` (`obs_to_multislot_features`)
- `src/esper/simic/ppo.py` (`signals_to_features`)

**Step 3: Commit**

Run:
```bash
git add docs/
git commit -m "docs: remove references to legacy v3 signals api"
```

---

## Final Verification (required before merging)

Run:
- `uv run ruff check src/ tests/`
- `uv run pytest -q`
- `rg -n "FastTrainingSignals|TensorSchema|TENSOR_SCHEMA_SIZE|to_vector\\(|to_fast\\(" src/ tests/` (should return nothing)

Expected: All green, and no legacy V3 API remains anywhere in runtime code or tests.
