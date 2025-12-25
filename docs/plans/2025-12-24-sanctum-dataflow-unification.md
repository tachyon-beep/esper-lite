# Sanctum Data Flow Unification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify Sanctum's split-brain architecture by making SanctumBackend use AggregatorRegistry internally, enabling A/B testing to work in production.

**Architecture:** The backend will own the AggregatorRegistry (currently orphaned in app). The app will call `backend.get_all_snapshots()` to get per-group snapshots, create TamiyoBrain widgets dynamically, and delete all dead event-driven code paths.

**Tech Stack:** Python 3.12, Textual TUI, pytest, threading (lock-based synchronization)

---

## Problem Summary

Two separate data paths exist that never synchronize:

| Path | Entry Point | Aggregator | Widgets Updated | Status |
|------|-------------|------------|-----------------|--------|
| Timer-driven | `_poll_and_refresh()` | `backend._aggregator` (single) | All except TamiyoBrain (until workaround) | **PRODUCTION** |
| Event-driven | `handle_telemetry_event()` | `app._aggregator_registry` (multi) | TamiyoBrain only | **DEAD** (never called) |

**Root Cause:** The app has two aggregators - one in backend (production), one in registry (orphaned). Events flow to backend's aggregator, but A/B widget code reads from registry (which is empty).

**Impact:** A/B testing mode is completely broken in production. Tests pass but validate dead code.

---

## Solution Overview

### Phase 1: Backend Uses Registry
- Replace `SanctumBackend._aggregator: SanctumAggregator` with `SanctumBackend._registry: AggregatorRegistry`
- Add `backend.get_all_snapshots() -> dict[str, SanctumSnapshot]`
- Keep `backend.get_snapshot()` as backward-compatible merged snapshot

### Phase 2: App Uses Multi-Snapshot API
- Update `_refresh_all_panels()` to call `backend.get_all_snapshots()`
- Create TamiyoBrain widgets dynamically per group (like old `_update_widgets()`)
- Update RunHeader with A/B comparison when 2+ groups
- Remove workaround default widget code

### Phase 3: Delete Dead Code
- Delete `app.handle_telemetry_event()` method
- Delete `app._aggregator_registry` field
- Delete `app._update_widgets()` method
- Keep `AggregatorRegistry` class (now used by backend)

### Phase 4: Fix Tests
- Update integration tests to use timer-based path
- Tests call `backend.emit()` then trigger `_poll_and_refresh()`
- Remove tests that call deleted methods

---

## Task 1: Add get_all_snapshots() to Backend

**Files:**
- Modify: `src/esper/karn/sanctum/backend.py`
- Test: `tests/karn/sanctum/test_backend.py`

### Step 1.1: Write failing test for get_all_snapshots

Add to `tests/karn/sanctum/test_backend.py`:

```python
class TestBackendMultiGroupAPI:
    """Tests for multi-group snapshot API."""

    def test_get_all_snapshots_empty_initially(self):
        """get_all_snapshots returns empty dict before any events."""
        backend = SanctumBackend(num_envs=4)
        backend.start()

        snapshots = backend.get_all_snapshots()

        assert snapshots == {}

    def test_get_all_snapshots_single_group(self):
        """get_all_snapshots returns single group after events."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        backend = SanctumBackend(num_envs=4)
        backend.start()

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id="A",
            data={"policy_loss": 0.1},
        )
        backend.emit(event)

        snapshots = backend.get_all_snapshots()

        assert "A" in snapshots
        assert len(snapshots) == 1

    def test_get_all_snapshots_multiple_groups(self):
        """get_all_snapshots returns all groups for A/B testing."""
        from esper.leyline import TelemetryEvent, TelemetryEventType

        backend = SanctumBackend(num_envs=4)
        backend.start()

        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                group_id=group_id,
                data={"policy_loss": 0.1},
            )
            backend.emit(event)

        snapshots = backend.get_all_snapshots()

        assert "A" in snapshots
        assert "B" in snapshots
        assert len(snapshots) == 2
```

### Step 1.2: Run test to verify it fails

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_backend.py::TestBackendMultiGroupAPI -v
```

Expected: FAIL with `AttributeError: 'SanctumBackend' object has no attribute 'get_all_snapshots'`

### Step 1.3: Implement get_all_snapshots in backend

Replace single `_aggregator` with `_registry` in `src/esper/karn/sanctum/backend.py`:

```python
"""Sanctum Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the AggregatorRegistry for TUI consumption.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from esper.karn.sanctum.registry import AggregatorRegistry

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.sanctum.schema import SanctumSnapshot

_logger = logging.getLogger(__name__)


class SanctumBackend:
    """OutputBackend that feeds telemetry to Sanctum TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot()/get_all_snapshots() called from UI thread.

    Uses AggregatorRegistry internally to support A/B testing with
    multiple policy groups. Each group_id gets its own aggregator.
    """

    def __init__(self, num_envs: int = 16, max_event_log: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_event_log: Maximum events to keep in log.
        """
        self._registry = AggregatorRegistry(
            num_envs=num_envs,
            max_event_log=max_event_log,
        )
        self._started = False
        self._event_count = 0

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True
        _logger.info("SanctumBackend started")

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to appropriate aggregator.

        Routes to aggregator based on event.group_id.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            _logger.warning("SanctumBackend.emit() called before start()")
            return
        self._event_count += 1
        self._registry.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "SanctumSnapshot":
        """Get merged SanctumSnapshot for backward compatibility.

        For single-policy mode or legacy callers. Returns the first
        group's snapshot, or empty snapshot if no events received.

        Returns:
            Snapshot of current aggregator state.
        """
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            # No events yet - return empty snapshot
            from esper.karn.sanctum.schema import SanctumSnapshot
            snapshot = SanctumSnapshot()
        else:
            # Return first group's snapshot (alphabetically)
            group_id = sorted(snapshots.keys())[0]
            snapshot = snapshots[group_id]

        # Add event count for debugging
        snapshot.total_events_received = self._event_count
        return snapshot

    def get_all_snapshots(self) -> dict[str, "SanctumSnapshot"]:
        """Get snapshots for all policy groups.

        For A/B testing mode. Each group_id maps to its aggregator's snapshot.

        Returns:
            Dict mapping group_id to SanctumSnapshot.
        """
        snapshots = self._registry.get_all_snapshots()
        # Add event count to each snapshot
        for snapshot in snapshots.values():
            snapshot.total_events_received = self._event_count
        return snapshots

    def toggle_decision_pin(self, decision_id: str) -> bool:
        """Toggle pin status for a decision.

        Args:
            decision_id: ID of the decision to toggle.

        Returns:
            New pin status (True if pinned, False if unpinned).
        """
        # Pin applies to first group (or specific group if ID contains group prefix)
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            return False
        group_id = sorted(snapshots.keys())[0]
        aggregator = self._registry.get_or_create(group_id)
        return aggregator.toggle_decision_pin(decision_id)

    def toggle_best_run_pin(self, record_id: str) -> bool:
        """Toggle pin status for a best run record.

        Args:
            record_id: ID of the record to toggle.

        Returns:
            New pin status (True if pinned, False if unpinned).
        """
        # Pin applies to first group
        snapshots = self._registry.get_all_snapshots()
        if not snapshots:
            return False
        group_id = sorted(snapshots.keys())[0]
        aggregator = self._registry.get_or_create(group_id)
        return aggregator.toggle_best_run_pin(record_id)
```

### Step 1.4: Update AggregatorRegistry to accept max_event_log

Modify `src/esper/karn/sanctum/registry.py`:

```python
class AggregatorRegistry:
    """Manages multiple SanctumAggregators for A/B testing.

    Each PolicyGroup gets its own aggregator, keyed by group_id.
    The registry creates aggregators on-demand when first accessed.
    """

    def __init__(self, num_envs: int = 4, max_event_log: int = 100) -> None:
        self._num_envs = num_envs
        self._max_event_log = max_event_log
        self._aggregators: dict[str, SanctumAggregator] = {}

    def get_or_create(self, group_id: str) -> "SanctumAggregator":
        """Get existing aggregator or create new one for group."""
        if group_id not in self._aggregators:
            from esper.karn.sanctum.aggregator import SanctumAggregator
            self._aggregators[group_id] = SanctumAggregator(
                num_envs=self._num_envs,
                max_event_log=self._max_event_log,
            )
        return self._aggregators[group_id]
```

### Step 1.5: Run tests to verify they pass

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_backend.py -v
```

Expected: All tests PASS

### Step 1.6: Commit

```bash
git add src/esper/karn/sanctum/backend.py src/esper/karn/sanctum/registry.py tests/karn/sanctum/test_backend.py
git commit -m "feat(sanctum): backend uses AggregatorRegistry for multi-group support

Replace single SanctumAggregator with AggregatorRegistry to enable A/B testing.
Add get_all_snapshots() API for accessing per-group snapshots.
Keep get_snapshot() for backward compatibility (returns first group).

Part of data flow unification to fix dead event-driven path."
```

---

## Task 2: Update App to Use Multi-Snapshot API

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Test: `tests/karn/sanctum/test_app_integration.py`

### Step 2.1: Write failing test for multi-group widget creation

Add to `tests/karn/sanctum/test_app_integration.py`:

```python
@pytest.mark.asyncio
async def test_backend_emits_create_multiple_tamiyo_widgets():
    """Backend emitting A/B events should create two TamiyoBrain widgets."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()

    # Emit events for two groups through backend (production path)
    for group_id in ["A", "B"]:
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id=group_id,
            data={"policy_loss": 0.1},
        )
        backend.emit(event)

    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Trigger refresh (simulates timer firing)
        app._poll_and_refresh()
        await pilot.pause()

        # Should have two TamiyoBrain widgets
        widgets = list(app.query(TamiyoBrain))
        assert len(widgets) == 2, f"Expected 2 TamiyoBrain widgets, got {len(widgets)}"

        # Each should have correct group class
        classes = [" ".join(w.classes) for w in widgets]
        assert any("group-a" in c for c in classes), "Missing group-a widget"
        assert any("group-b" in c for c in classes), "Missing group-b widget"
```

### Step 2.2: Run test to verify it fails

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app_integration.py::test_backend_emits_create_multiple_tamiyo_widgets -v
```

Expected: FAIL - only one "default" widget exists

### Step 2.3: Update _refresh_all_panels to use get_all_snapshots

Modify `_refresh_all_panels()` in `src/esper/karn/sanctum/app.py`:

```python
def _refresh_all_panels(self, snapshot: "SanctumSnapshot") -> None:
    """Refresh all panels with new snapshot data.

    Args:
        snapshot: The current telemetry snapshot (for non-TamiyoBrain widgets).
    """
    # Update run header first (most important context)
    try:
        self.query_one("#run-header", RunHeader).update_snapshot(snapshot)
    except NoMatches:
        pass  # Widget hasn't mounted yet
    except Exception as e:
        self.log.warning(f"Failed to update run-header: {e}")

    # Update anomaly strip (after run header)
    try:
        self.query_one("#anomaly-strip", AnomalyStrip).update_snapshot(snapshot)
    except NoMatches:
        pass  # Widget hasn't mounted yet
    except Exception as e:
        self.log.warning(f"Failed to update anomaly-strip: {e}")

    # Update each widget - query by ID and call update_snapshot
    try:
        self.query_one("#env-overview", EnvOverview).update_snapshot(snapshot)
    except NoMatches:
        pass  # Widget hasn't mounted yet
    except Exception as e:
        self.log.warning(f"Failed to update env-overview: {e}")

    try:
        self.query_one("#scoreboard", Scoreboard).update_snapshot(snapshot)
    except NoMatches:
        pass  # Widget hasn't mounted yet
    except Exception as e:
        self.log.warning(f"Failed to update scoreboard: {e}")

    # Update TamiyoBrain widgets using multi-group API
    self._refresh_tamiyo_widgets()

    try:
        self.query_one("#event-log", EventLog).update_snapshot(snapshot)
    except NoMatches:
        pass  # Widget hasn't mounted yet
    except Exception as e:
        self.log.warning(f"Failed to update event-log: {e}")

    # Update EnvDetailScreen modal if displayed
    if len(self.screen_stack) > 1:
        current_screen = self.screen_stack[-1]
        if isinstance(current_screen, EnvDetailScreen):
            env = snapshot.envs.get(current_screen.env_id)
            if env is not None:
                try:
                    current_screen.update_env_state(env)
                except Exception as e:
                    self.log.warning(f"Failed to update env-detail-screen: {e}")


def _refresh_tamiyo_widgets(self) -> None:
    """Refresh TamiyoBrain widgets from backend's multi-group snapshots.

    Creates widgets dynamically for each policy group. Handles both
    single-policy mode (one widget) and A/B testing (two+ widgets).
    """
    snapshots = self._backend.get_all_snapshots()

    # Handle empty case (no events yet) - create default widget
    if not snapshots:
        try:
            widget = self._get_or_create_tamiyo_widget("default")
            from esper.karn.sanctum.schema import SanctumSnapshot as SnapshotClass
            widget.update_snapshot(SnapshotClass())
        except NoMatches:
            pass
        return

    # Create/update widget for each group
    for group_id, group_snapshot in snapshots.items():
        try:
            widget = self._get_or_create_tamiyo_widget(group_id)
            widget.update_snapshot(group_snapshot)
        except NoMatches:
            pass  # Container hasn't mounted yet
        except Exception as e:
            self.log.warning(f"Failed to update tamiyo widget for {group_id}: {e}")

    # Update RunHeader with A/B comparison data when 2+ policies
    if len(snapshots) >= 2:
        try:
            run_header = self.query_one("#run-header", RunHeader)
            group_ids = sorted(snapshots.keys())
            snapshot_a = snapshots[group_ids[0]]
            snapshot_b = snapshots[group_ids[1]]

            run_header.update_comparison(
                group_a_accuracy=snapshot_a.aggregate_mean_accuracy,
                group_b_accuracy=snapshot_b.aggregate_mean_accuracy,
                group_a_reward=snapshot_a.aggregate_mean_reward,
                group_b_reward=snapshot_b.aggregate_mean_reward,
            )
        except NoMatches:
            pass
        except Exception as e:
            self.log.warning(f"Failed to update run header comparison: {e}")
```

### Step 2.4: Run test to verify it passes

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app_integration.py::test_backend_emits_create_multiple_tamiyo_widgets -v
```

Expected: PASS

### Step 2.5: Run full test suite to verify no regressions

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests PASS

### Step 2.6: Commit

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app_integration.py
git commit -m "feat(sanctum): app uses backend.get_all_snapshots() for A/B mode

Update _refresh_all_panels() to use new multi-group API.
Extract _refresh_tamiyo_widgets() for cleaner separation.
TamiyoBrain widgets now created dynamically per group.
A/B testing now works in production path."
```

---

## Task 3: Delete Dead Event-Driven Code

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Modify: `tests/karn/sanctum/test_app_integration.py`

### Step 3.1: Identify dead code to delete

In `src/esper/karn/sanctum/app.py`:
- Line 22: `from esper.karn.sanctum.registry import AggregatorRegistry` - DELETE (no longer used in app)
- Line 178: `self._aggregator_registry = AggregatorRegistry(num_envs=num_envs)` - DELETE
- Lines 229-236: `handle_telemetry_event()` method - DELETE
- Lines 264-297: `_update_widgets()` method - DELETE

### Step 3.2: Delete orphaned app._aggregator_registry

Remove the import and field initialization:

```python
# DELETE this import (line 22)
# from esper.karn.sanctum.registry import AggregatorRegistry

# DELETE this line in __init__ (line 178)
# self._aggregator_registry = AggregatorRegistry(num_envs=num_envs)
```

### Step 3.3: Delete handle_telemetry_event method

```python
# DELETE this entire method (lines 229-236)
# def handle_telemetry_event(self, event: "TelemetryEvent") -> None:
#     """Route telemetry event to appropriate aggregator."""
#     self._aggregator_registry.process_event(event)
#     self._update_widgets()
```

### Step 3.4: Delete _update_widgets method

```python
# DELETE this entire method (lines 264-297)
# def _update_widgets(self) -> None:
#     """Update all TamiyoBrain widgets with latest snapshots."""
#     snapshots = self._aggregator_registry.get_all_snapshots()
#     ...
```

### Step 3.5: Update integration tests to use production path

Modify tests in `tests/karn/sanctum/test_app_integration.py` that called `handle_telemetry_event()`:

**Before:**
```python
app.handle_telemetry_event(event)
```

**After:**
```python
backend.emit(event)
app._poll_and_refresh()
await pilot.pause()
```

Update these tests:
- `test_sanctum_app_shows_multiple_tamiyo_widgets`
- `test_keyboard_switches_between_policies`
- `test_run_header_shows_ab_comparison`
- `test_run_header_no_ab_comparison_in_single_mode`

### Step 3.6: Run tests to verify all pass

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests PASS

### Step 3.7: Commit

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app_integration.py
git commit -m "refactor(sanctum): delete dead event-driven code path

Remove orphaned AggregatorRegistry from app (now in backend).
Delete handle_telemetry_event() - was never called in production.
Delete _update_widgets() - replaced by _refresh_tamiyo_widgets().

All telemetry now flows through single path:
  backend.emit() → backend._registry → app._poll_and_refresh()

Tests updated to use production path via backend.emit()."
```

---

## Task 4: Final Verification

### Step 4.1: Run full test suite

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests PASS

### Step 4.2: Run Sanctum manually to verify A/B mode

```bash
# In one terminal, start training with A/B mode (if available)
PYTHONPATH=src uv run python -m esper.scripts.train ppo --dual-ab

# Or verify single-policy mode works
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 5
```

### Step 4.3: Verify architecture is unified

Check that:
- `SanctumBackend` owns the only `AggregatorRegistry`
- `SanctumApp` has no aggregator fields
- All telemetry flows: `emit()` → `registry` → `_poll_and_refresh()` → widgets

### Step 4.4: Final commit (if any fixes needed)

```bash
git add -A
git commit -m "fix(sanctum): final verification fixes

[Any final fixes discovered during manual testing]"
```

---

## Summary

| Phase | Task | Files Changed | Tests |
|-------|------|---------------|-------|
| 1 | Backend uses registry | `backend.py`, `registry.py` | `test_backend.py` |
| 2 | App uses multi-snapshot API | `app.py` | `test_app_integration.py` |
| 3 | Delete dead code | `app.py`, tests | Update 4 tests |
| 4 | Final verification | - | Full suite |

**Before:** Two data paths, orphaned registry, broken A/B mode
**After:** Single unified path, A/B mode works, no dead code

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Backend API change breaks callers | Low | Medium | `get_snapshot()` preserved for compat |
| Tests using dead path fail | Medium | Low | Tests updated in Task 3 |
| Widget mounting race conditions | Low | Low | Existing NoMatches handling |
| Thread safety issues | Low | High | Registry has same locking as aggregator |

**Overall Risk: LOW** - Changes are surgical and well-contained.
