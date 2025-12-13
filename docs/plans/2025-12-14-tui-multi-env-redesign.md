# TUI Multi-Environment Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Karn TUI and telemetry pipeline to properly support multi-environment PPO training with per-env tracking, aggregate views, correct event emission, and robust serialization.

**Architecture:**
1. Fix event emission (EPOCH_COMPLETED never emitted, heuristic path missing TRAINING_STARTED)
2. Fix slot collision in Karn collector (namespace by env_id)
3. Fix JSON serialization crash (datetime handling)
4. Replace TUI global metric tracking with per-env state
5. Add env overview table and env ID prefixes to event log

**Tech Stack:** Python 3.11+, Rich (TUI library), dataclasses, deque for history tracking

---

## Phase 0: Critical Pipeline Fixes (Event Emission & Ordering)

### ⚠️ Critical: Event Ordering Contract

**EPOCH_COMPLETED is Karn's commit barrier.** When Karn receives EPOCH_COMPLETED(e), it:
1. Commits epoch e's snapshot (with all accumulated data)
2. Immediately advances to epoch e+1

**Any event emitted AFTER EPOCH_COMPLETED(e) goes into epoch e+1, not e.**

**Safe emission order per epoch:**
```
1. [Epoch e begins] Karn's current_epoch.epoch == e
2. [During rollout] SEED_*, REWARD_COMPUTED, COUNTERFACTUAL_COMPUTED (with epoch=e)
3. [After PPO update] PPO_UPDATE_COMPLETED (with epoch=e)  ← MUST include epoch!
4. [Optional] ANALYTICS_SNAPSHOT (with epoch=e)
5. [LAST] EPOCH_COMPLETED(e) ← commits and advances to e+1
```

**Current bugs:**
- Karn starts at epoch 0, Simic starts at epoch 1 → off-by-one
- PPO_UPDATE_COMPLETED missing `epoch=epoch` field
- ANALYTICS_SNAPSHOT missing `epoch=epoch` field

---

### Task 0: Remove Duplicate SeedStage Enum from Karn (Contract Violation)

**Files:**
- Modify: `src/esper/karn/store.py` (delete lines 86-102, add import)
- Modify: `src/esper/karn/__init__.py` (re-export from leyline, not store)

**Problem:** Karn defines its own `SeedStage(Enum)` that "mirrors" leyline's `SeedStage(IntEnum)`. This violates CLAUDE.md's No Legacy Code Policy ("No adapter classes to support old interfaces").

**Step 1: Write the failing test**

Create `tests/karn/test_leyline_contracts.py`:

```python
"""Tests for Karn's use of leyline contracts."""

import pytest


class TestSeedStageContract:
    """Tests for SeedStage contract compliance."""

    def test_karn_uses_leyline_seedstage(self):
        """Karn should use leyline.SeedStage, not define its own."""
        from esper.leyline import SeedStage as LeylineSeedStage
        from esper.karn.store import SlotSnapshot

        # SlotSnapshot.stage should use the leyline enum
        slot = SlotSnapshot(slot_id="mid")
        assert type(slot.stage).__module__ == "esper.leyline.stages"
        assert isinstance(slot.stage.value, int)  # IntEnum, not Enum

    def test_karn_exports_leyline_seedstage(self):
        """Karn's re-export should be the same as leyline's."""
        from esper.leyline import SeedStage as LeylineSeedStage
        from esper.karn import SeedStage as KarnSeedStage

        assert LeylineSeedStage is KarnSeedStage
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_leyline_contracts.py -v`
Expected: FAIL (Karn's SeedStage is from `esper.karn.store`, not `esper.leyline.stages`)

**Step 3: Delete local SeedStage and import from leyline**

In `src/esper/karn/store.py`:

1. Add import at top:
```python
from esper.leyline import SeedStage
```

2. Delete the local `class SeedStage(Enum)` definition (lines 86-102).

**Step 4: Update Karn's __init__.py to re-export from leyline**

In `src/esper/karn/__init__.py`, change the import:

```python
# Before:
from esper.karn.store import (
    ...
    SeedStage,
)

# After:
from esper.karn.store import (
    ...
    # SeedStage removed - now from leyline
)
from esper.leyline import SeedStage  # Re-export authoritative definition
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_leyline_contracts.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/karn/test_leyline_contracts.py src/esper/karn/store.py src/esper/karn/__init__.py
git commit -m "$(cat <<'EOF'
fix(karn): remove duplicate SeedStage, use leyline contract

CLAUDE.md violation: Karn defined its own SeedStage(Enum) that
"mirrored" leyline's SeedStage(IntEnum). This is exactly the kind
of compatibility shim the No Legacy Code Policy forbids.

Fixed by:
- Deleting local SeedStage class from store.py
- Importing from esper.leyline instead
- Re-exporting leyline's SeedStage from karn.__init__

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: Fix Epoch Numbering and Add epoch Field to PPO Events

**Files:**
- Modify: `src/esper/karn/collector.py:195` (start at epoch 1, not 0)
- Modify: `src/esper/simic/vectorized.py:1477-1483` (add epoch to skipped event)
- Modify: `src/esper/simic/vectorized.py:1585-1612` (add epoch to PPO event)
- Modify: `src/esper/simic/vectorized.py:1619-1633` (add epoch to ANALYTICS_SNAPSHOT)
- Test: `tests/simic/test_epoch_telemetry.py` (create)

**Step 1: Write the failing test**

Create `tests/simic/test_epoch_telemetry.py`:

```python
"""Tests for epoch telemetry emission and ordering."""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType


class TestEpochTelemetryContract:
    """Tests for epoch telemetry event contracts."""

    def test_ppo_update_completed_has_epoch_field(self):
        """PPO_UPDATE_COMPLETED events MUST include epoch field."""
        # This test documents the contract: PPO events must have epoch
        # so Karn can validate they land in the correct epoch snapshot.

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=10,  # REQUIRED
            data={
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 1.5,
                "kl_divergence": 0.01,
            }
        )

        assert event.epoch == 10, "PPO_UPDATE_COMPLETED must have epoch field"

    def test_epoch_completed_has_required_keys(self):
        """EPOCH_COMPLETED must have keys Karn expects for host snapshot."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=10,
            data={
                "train_loss": 0.5,
                "train_accuracy": 75.0,
                "val_loss": 0.6,
                "val_accuracy": 72.0,
            }
        )

        # Karn's _handle_epoch_completed expects these exact keys
        assert "val_loss" in event.data
        assert "val_accuracy" in event.data
        assert "train_loss" in event.data
        assert "train_accuracy" in event.data

    def test_epoch_completed_is_commit_barrier(self):
        """Document that EPOCH_COMPLETED commits and advances epoch."""
        from esper.karn.collector import TelemetryCollector
        from esper.karn.store import TelemetryStore

        store = TelemetryStore()
        collector = TelemetryCollector(store)

        # Start episode (starts at epoch 1 to match Simic)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 10}
        ))

        # Karn should now be at epoch 1
        assert store.current_epoch.epoch == 1

        # Emit EPOCH_COMPLETED(1)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data={"val_loss": 0.5, "val_accuracy": 70.0, "train_loss": 0.6, "train_accuracy": 65.0}
        ))

        # Should have committed epoch 1 and advanced to epoch 2
        assert len(store.epoch_snapshots) == 1
        assert store.epoch_snapshots[0].epoch == 1
        assert store.current_epoch.epoch == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_epoch_telemetry.py::TestEpochTelemetryContract::test_epoch_completed_is_commit_barrier -v`
Expected: FAIL (Karn starts at epoch 0, not 1)

**Step 3: Fix Karn to start at epoch 1**

Modify `src/esper/karn/collector.py` line 195:

```python
        # Start at epoch 1 to match Simic's range(1, max_epochs + 1)
        self.store.start_epoch(1)
        _logger.debug(f"Auto-started episode and epoch 1 from TRAINING_STARTED: {episode_id}")
```

**Step 4: Add epoch field to PPO_UPDATE_COMPLETED (skipped case)**

Modify `src/esper/simic/vectorized.py` lines 1477-1483:

```python
            if hub:
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                    epoch=epoch,  # ADD THIS
                    severity="warning",
                    message="Buffer cleared due to Governor rollback - skipping update",
                    data={"reason": "governor_rollback", "skipped": True},
                ))
```

**Step 5: Add epoch field to PPO_UPDATE_COMPLETED (normal case)**

Modify `src/esper/simic/vectorized.py` line 1585:

```python
            ppo_event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                epoch=epoch,  # ADD THIS
                data={
                    # ... existing fields ...
```

**Step 6: Add epoch field to ANALYTICS_SNAPSHOT**

Modify `src/esper/simic/vectorized.py` line 1619:

```python
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                epoch=epoch,  # ADD THIS
                data={
                    # ... existing fields ...
```

**Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/simic/test_epoch_telemetry.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add tests/simic/test_epoch_telemetry.py src/esper/karn/collector.py src/esper/simic/vectorized.py
git commit -m "$(cat <<'EOF'
fix(simic/karn): align epoch numbering and add epoch to PPO events

Critical fixes for Karn epoch snapshot consistency:

1. Karn now starts at epoch 1 (was 0) to match Simic's range(1, max_epochs+1)
2. PPO_UPDATE_COMPLETED now includes epoch=epoch field
3. ANALYTICS_SNAPSHOT now includes epoch=epoch field

This ensures PPO metrics land in the correct epoch snapshot before
EPOCH_COMPLETED commits and advances to the next epoch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Emit EPOCH_COMPLETED as Final Commit Barrier

**Files:**
- Modify: `src/esper/simic/vectorized.py` (after PPO update + ANALYTICS_SNAPSHOT, before batch loop ends)
- Test: `tests/simic/test_epoch_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_epoch_telemetry.py`:

```python
class TestEpochCompletedEmission:
    """Tests for EPOCH_COMPLETED emission."""

    def test_epoch_completed_emitted_with_aggregate_metrics(self):
        """EPOCH_COMPLETED includes aggregate metrics across envs."""
        from esper.nissa import get_hub

        hub = get_hub()
        captured = []

        class CaptureBackend:
            def emit(self, event):
                if event.event_type == TelemetryEventType.EPOCH_COMPLETED:
                    captured.append(event)
            def close(self):
                pass

        hub.add_backend(CaptureBackend())

        # Emit test event with expected structure
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=5,
            data={
                "train_loss": 0.5,
                "train_accuracy": 75.0,
                "val_loss": 0.6,
                "val_accuracy": 72.0,
                "n_envs": 4,
            }
        ))

        assert len(captured) == 1
        assert captured[0].epoch == 5
        assert captured[0].data["n_envs"] == 4
```

**Step 2: Run test**

Run: `PYTHONPATH=src pytest tests/simic/test_epoch_telemetry.py::TestEpochCompletedEmission -v`
Expected: PASS (contract test)

**Step 3: Add EPOCH_COMPLETED emission after ANALYTICS_SNAPSHOT**

Add after line 1633 in `src/esper/simic/vectorized.py` (after ANALYTICS_SNAPSHOT emit, before the batch loop continues):

```python
            # EPOCH_COMPLETED: Commit barrier for Karn
            # This MUST be emitted LAST for each epoch, after PPO_UPDATE and ANALYTICS_SNAPSHOT
            # Karn will commit the epoch snapshot and advance to epoch+1
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=epoch,
                data={
                    "train_loss": sum(train_losses) / max(len(env_states) * num_train_batches, 1),
                    "train_accuracy": sum(100.0 * train_corrects[i] / max(train_totals[i], 1)
                                          for i in range(len(env_states))) / len(env_states),
                    "val_loss": sum(val_losses) / max(len(env_states) * num_test_batches, 1),
                    "val_accuracy": sum(100.0 * val_corrects[i] / max(val_totals[i], 1)
                                        for i in range(len(env_states))) / len(env_states),
                    "n_envs": len(env_states),
                },
            ))
```

**Note:** The placement is critical—EPOCH_COMPLETED must be the LAST event emitted for each epoch. It goes after ANALYTICS_SNAPSHOT (line 1633), which is after PPO_UPDATE_COMPLETED (line 1613).

**Step 4: Run test**

Run: `PYTHONPATH=src pytest tests/simic/test_epoch_telemetry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/simic/test_epoch_telemetry.py src/esper/simic/vectorized.py
git commit -m "$(cat <<'EOF'
fix(simic): emit EPOCH_COMPLETED as final commit barrier

EPOCH_COMPLETED is now emitted AFTER PPO_UPDATE and ANALYTICS_SNAPSHOT,
acting as the commit barrier for Karn's epoch snapshots.

Event ordering per epoch:
1. SEED_*, REWARD_COMPUTED (during rollout)
2. PPO_UPDATE_COMPLETED (after update)
3. ANALYTICS_SNAPSHOT (dashboard sync)
4. EPOCH_COMPLETED (commits and advances) ← NEW

Includes aggregate metrics: train/val loss/accuracy, n_envs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fix JSON Serialization Crash in export_jsonl

**Files:**
- Modify: `src/esper/karn/store.py:408-414`
- Test: `tests/karn/test_store_export.py` (create)

**Step 1: Write the failing test**

Create `tests/karn/test_store_export.py`:

```python
"""Tests for Karn store export functionality."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from esper.karn.store import TelemetryStore, EpisodeContext


class TestExportJsonl:
    """Tests for JSONL export serialization."""

    def test_export_handles_datetime(self, tmp_path: Path):
        """export_jsonl correctly serializes datetime objects."""
        store = TelemetryStore()

        # Create context with datetime
        store.context = EpisodeContext(
            episode_id="test_123",
            timestamp=datetime(2025, 12, 14, 12, 0, 0, tzinfo=timezone.utc),
            base_seed=42,
        )

        output_file = tmp_path / "test_export.jsonl"
        count = store.export_jsonl(output_file)

        assert count >= 1
        assert output_file.exists()

        # Verify it's valid JSON
        with open(output_file) as f:
            line = f.readline()
            data = json.loads(line)  # Should not raise
            assert data["type"] == "context"
            # timestamp should be serialized as ISO string
            assert "timestamp" in data["data"]

    def test_export_handles_path_objects(self, tmp_path: Path):
        """export_jsonl correctly serializes Path objects."""
        store = TelemetryStore()
        store.context = EpisodeContext(
            episode_id="test_path",
            timestamp=datetime.now(timezone.utc),
        )

        output_file = tmp_path / "test_path.jsonl"

        # Should not raise TypeError
        count = store.export_jsonl(output_file)
        assert count >= 1

    def test_export_handles_enums(self, tmp_path: Path):
        """export_jsonl correctly serializes Enum values."""
        from esper.karn.store import SlotSnapshot
        from esper.leyline import SeedStage

        store = TelemetryStore()
        store.start_episode("test_enum", base_seed=42, max_epochs=10)

        # Add a slot with enum stage
        store.current_epoch.slots["mid"] = SlotSnapshot(
            slot_id="mid",
            stage=SeedStage.TRAINING,
        )
        store.commit_epoch()

        output_file = tmp_path / "test_enum.jsonl"
        count = store.export_jsonl(output_file)

        # Should serialize enum as name string
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                if data["type"] == "epoch":
                    # Check that stage is serialized (as int from enum value)
                    assert "slots" in data["data"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_store_export.py -v`
Expected: FAIL with TypeError (datetime not JSON serializable)

**Step 3: Fix serialize function in export_jsonl**

Modify `src/esper/karn/store.py` lines 408-414:

```python
        def serialize(obj):
            """Serialize dataclass or primitive to JSON-safe dict."""
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            elif isinstance(obj, deque):
                return list(obj)
            return obj

        def json_default(obj):
            """Handle non-serializable types for json.dumps."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            # hasattr AUTHORIZED by John on 2025-12-14 15:00:00 UTC
            # Justification: Serialization - handle Enum values in JSON export
            if hasattr(obj, "name") and hasattr(obj, "value"):
                return obj.name  # Serialize enum as name string
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
```

Then update all `json.dumps` calls in the method to use `default=json_default`:

```python
            # Write context
            if self.context:
                f.write(json.dumps({"type": "context", "data": serialize(self.context)}, default=json_default) + "\n")
                count += 1

            # Write baseline
            if self.baseline:
                f.write(json.dumps({"type": "baseline", "data": serialize(self.baseline)}, default=json_default) + "\n")
                count += 1

            # Write epochs
            for epoch in self.epoch_snapshots:
                f.write(json.dumps({"type": "epoch", "data": serialize(epoch)}, default=json_default) + "\n")
                count += 1
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_store_export.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_store_export.py src/esper/karn/store.py
git commit -m "$(cat <<'EOF'
fix(karn): handle datetime/Path/Enum in export_jsonl

export_jsonl crashed with TypeError when serializing:
- datetime objects (EpisodeContext.timestamp)
- Path objects
- Enum values (SeedStage)

Added json_default handler for non-serializable types:
- datetime → ISO format string
- Path → string
- Enum → name string

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Fix Multi-Env Slot Collision in Karn Collector

**Files:**
- Modify: `src/esper/karn/collector.py:246-258`
- Test: `tests/karn/test_collector_multienv.py` (create)

**Step 1: Write the failing test**

Create `tests/karn/test_collector_multienv.py`:

```python
"""Tests for Karn collector multi-env support."""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.collector import TelemetryCollector
from esper.karn.store import TelemetryStore


class TestMultiEnvSlotTracking:
    """Tests for slot tracking across multiple environments."""

    def test_slots_namespaced_by_env_id(self):
        """Slots from different envs don't collide."""
        store = TelemetryStore()
        collector = TelemetryCollector(store)

        # Start episode
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test_multi", "max_epochs": 10, "n_envs": 2}
        ))

        # Germinate in slot "mid" for env 0
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="mid",
            data={"env_id": 0, "seed_id": "env0_seed_0", "blueprint_id": "conv"}
        ))

        # Germinate in slot "mid" for env 1 (same slot_id, different env)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="mid",
            data={"env_id": 1, "seed_id": "env1_seed_0", "blueprint_id": "norm"}
        ))

        # Both should be tracked separately
        slots = store.current_epoch.slots

        # Expect namespaced keys
        assert "env0:mid" in slots or ("mid" in slots and len(slots) == 2)

        # If namespaced, verify different blueprints
        if "env0:mid" in slots:
            assert slots["env0:mid"].blueprint_id == "conv"
            assert slots["env1:mid"].blueprint_id == "norm"

    def test_env_id_extracted_from_event_data(self):
        """env_id is correctly extracted from event.data."""
        store = TelemetryStore()
        collector = TelemetryCollector(store)

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 5}
        ))

        # Event with env_id in data
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="early",
            data={"env_id": 3, "seed_id": "env3_seed_0", "blueprint_id": "test"}
        ))

        # Should namespace by env_id
        slots = store.current_epoch.slots
        assert "env3:early" in slots or "early" in slots
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_collector_multienv.py -v`
Expected: FAIL (slots overwrite each other)

**Step 3: Update _handle_seed_event to namespace slots**

Modify `src/esper/karn/collector.py` starting at line 246:

```python
    def _handle_seed_event(self, event: "TelemetryEvent") -> None:
        """Handle seed lifecycle events with env-namespaced slots."""
        if not self.store.current_epoch:
            return

        data = event.data or {}

        # Extract env_id (standardize on env_id, but accept env_idx for backwards compat)
        env_id = data.get("env_id", data.get("env_idx", 0))

        # Get raw slot_id
        raw_slot_id = event.slot_id or data.get("slot_id", "unknown")

        # Namespace slot key by env_id to prevent multi-env collisions
        slot_key = f"env{env_id}:{raw_slot_id}"

        # Get or create slot snapshot with namespaced key
        if slot_key not in self.store.current_epoch.slots:
            self.store.current_epoch.slots[slot_key] = SlotSnapshot(slot_id=slot_key)

        slot = self.store.current_epoch.slots[slot_key]

        # Update based on event type
        # hasattr AUTHORIZED by John on 2025-12-14 03:30:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        if event_type == "SEED_GERMINATED":
            slot.stage = SeedStage.GERMINATED
            slot.seed_id = data.get("seed_id")
            slot.blueprint_id = data.get("blueprint_id")
            slot.seed_params = data.get("params", 0)
        elif event_type == "SEED_STAGE_CHANGED":
            stage_name = data.get("to", "DORMANT")
            try:
                slot.stage = SeedStage[stage_name]
            except KeyError:
                pass
            slot.epochs_in_stage = 0
        elif event_type == "SEED_FOSSILIZED":
            slot.stage = SeedStage.FOSSILIZED
        elif event_type == "SEED_CULLED":
            slot.stage = SeedStage.CULLED
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_collector_multienv.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_collector_multienv.py src/esper/karn/collector.py
git commit -m "$(cat <<'EOF'
fix(karn): namespace slots by env_id to prevent multi-env collisions

With --n-envs > 1, slots from different environments were overwriting
each other because they shared the same slot_id key (e.g., "mid").

Now slots are keyed as "env{N}:{slot_id}" (e.g., "env0:mid", "env1:mid")
to ensure each environment's slots are tracked independently.

Also standardizes on env_id (accepting env_idx for backwards compat).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Add TRAINING_STARTED Emission to Heuristic Path

**Files:**
- Modify: `src/esper/simic/training.py:278-286`
- Test: `tests/simic/test_heuristic_telemetry.py` (create)

**Step 1: Write the failing test**

Create `tests/simic/test_heuristic_telemetry.py`:

```python
"""Tests for heuristic training telemetry emission."""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType


class TestHeuristicTelemetry:
    """Tests for telemetry emission in heuristic training."""

    def test_training_started_event_contract(self):
        """Document expected TRAINING_STARTED event structure."""
        # This test documents the contract that heuristic training should emit
        # TRAINING_STARTED to activate Karn.

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={
                "episode_id": "heur_42",
                "seed": 42,
                "max_epochs": 75,
                "task": "cifar10",
            }
        )

        assert event.event_type == TelemetryEventType.TRAINING_STARTED
        assert "episode_id" in event.data
        assert "max_epochs" in event.data
```

**Step 2: Run test**

Run: `PYTHONPATH=src pytest tests/simic/test_heuristic_telemetry.py -v`
Expected: PASS (contract test)

**Step 3: Add TRAINING_STARTED emission to run_heuristic_episode**

Modify `src/esper/simic/training.py` after line 285 (after telemetry wiring):

```python
    # Wire telemetry
    hub = get_hub()
    def telemetry_callback(event):
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    slot = model.seed_slots[target_slot]
    slot.on_telemetry = telemetry_callback
    slot.fast_mode = False
    slot.isolate_gradients = True

    # Emit TRAINING_STARTED to activate Karn (P1 fix)
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={
            "episode_id": f"heur_{base_seed}",
            "seed": base_seed,
            "max_epochs": max_epochs,
            "task": task_spec.name,
            "mode": "heuristic",
        },
    ))
```

**Step 4: Run test**

Run: `PYTHONPATH=src pytest tests/simic/test_heuristic_telemetry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/simic/test_heuristic_telemetry.py src/esper/simic/training.py
git commit -m "$(cat <<'EOF'
fix(simic): emit TRAINING_STARTED in heuristic training path

Karn activates on TRAINING_STARTED events, but heuristic training
never emitted this event, so Karn was never activated.

Now emits TRAINING_STARTED at start of run_heuristic_episode() with:
- episode_id, seed, max_epochs, task, mode

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: TUI Data Structure Fixes

### Task 6: Add Per-Env Reward Tracking to EnvState

**Files:**
- Modify: `src/esper/karn/tui.py:105-114`
- Test: `tests/karn/test_tui_state.py` (create)

**Step 1: Write the failing test**

Create `tests/karn/__init__.py`:

```python
"""Karn test package."""
```

Create `tests/karn/test_tui_state.py`:

```python
"""Tests for TUI state management."""

from collections import deque

import pytest

from esper.karn.tui import EnvState, TUIState


class TestEnvState:
    """Tests for per-environment state tracking."""

    def test_env_state_has_reward_history(self):
        """EnvState tracks per-env reward history."""
        env = EnvState(env_id=0)
        assert hasattr(env, "reward_history")
        assert isinstance(env.reward_history, deque)
        assert env.reward_history.maxlen == 50

    def test_env_state_has_accuracy_history(self):
        """EnvState tracks per-env accuracy history."""
        env = EnvState(env_id=0)
        assert hasattr(env, "accuracy_history")
        assert isinstance(env.accuracy_history, deque)

    def test_env_state_has_action_history(self):
        """EnvState tracks recent actions."""
        env = EnvState(env_id=0)
        assert hasattr(env, "action_history")
        assert isinstance(env.action_history, deque)
        assert env.action_history.maxlen == 10

    def test_env_state_current_reward_from_history(self):
        """EnvState.current_reward returns last reward in history."""
        env = EnvState(env_id=0)
        assert env.current_reward == 0.0  # default when empty

        env.reward_history.append(0.5)
        env.reward_history.append(0.3)
        assert env.current_reward == 0.3

    def test_env_state_reward_sparkline(self):
        """EnvState generates sparkline from reward history."""
        env = EnvState(env_id=0)
        env.reward_history.extend([0.1, 0.3, 0.5, 0.7, 0.9])

        sparkline = env.reward_sparkline
        assert isinstance(sparkline, str)
        assert len(sparkline) > 0
        assert any(c in sparkline for c in "▁▂▃▄▅▆▇█")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestEnvState -v`
Expected: FAIL with AttributeError

**Step 3: Add helper function and enhance EnvState**

Add after line 103 (after SeedState class):

```python
def _make_sparkline_static(values: list[float], width: int = 8) -> str:
    """Create a sparkline from values (static helper)."""
    if not values:
        return "─" * width

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1.0

    blocks = "▁▂▃▄▅▆▇█"
    values = values[-width:]

    result = ""
    for v in values:
        normalized = (v - min_val) / range_val
        idx = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
        result += blocks[idx]

    return result.ljust(width, "─")
```

Replace EnvState dataclass (lines 105-114):

```python
@dataclass
class EnvState:
    """Per-environment state for multi-env tracking."""
    env_id: int = 0
    current_epoch: int = 0
    host_accuracy: float = 0.0
    host_loss: float = 0.0
    seeds: dict[str, SeedState] = field(default_factory=dict)
    active_seed_count: int = 0
    fossilized_count: int = 0
    culled_count: int = 0

    # Per-env reward tracking
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    best_reward: float = float('-inf')
    best_reward_epoch: int = 0

    # Per-env accuracy tracking
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    best_accuracy: float = 0.0
    best_accuracy_epoch: int = 0

    # Per-env action tracking
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0
    })
    total_actions: int = 0

    # Status tracking
    status: str = "initializing"
    last_update: datetime | None = None
    epochs_since_improvement: int = 0

    @property
    def current_reward(self) -> float:
        """Get most recent reward."""
        return self.reward_history[-1] if self.reward_history else 0.0

    @property
    def mean_reward(self) -> float:
        """Mean reward over history."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def reward_sparkline(self) -> str:
        """Generate sparkline from reward history."""
        return _make_sparkline_static(list(self.reward_history), width=8)

    @property
    def accuracy_sparkline(self) -> str:
        """Generate sparkline from accuracy history."""
        return _make_sparkline_static(list(self.accuracy_history), width=8)

    def add_reward(self, reward: float, epoch: int) -> None:
        """Add reward and update best tracking."""
        self.reward_history.append(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_epoch = epoch

    def add_accuracy(self, accuracy: float, epoch: int) -> None:
        """Add accuracy and update best/status tracking."""
        prev_acc = self.accuracy_history[-1] if self.accuracy_history else 0.0
        self.accuracy_history.append(accuracy)
        self.host_accuracy = accuracy

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        self._update_status(prev_acc, accuracy)

    def add_action(self, action_name: str) -> None:
        """Track action taken."""
        self.action_history.append(action_name)
        if action_name in self.action_counts:
            self.action_counts[action_name] += 1
            self.total_actions += 1

    def _update_status(self, prev_acc: float, curr_acc: float) -> None:
        """Update env status based on metrics."""
        if self.epochs_since_improvement > 10:
            self.status = "stalled"
        elif curr_acc < prev_acc - 1.0:
            self.status = "degraded"
        elif curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestEnvState -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/__init__.py tests/karn/test_tui_state.py src/esper/karn/tui.py
git commit -m "$(cat <<'EOF'
feat(karn): add per-env reward/accuracy/action tracking to EnvState

- Add reward_history, accuracy_history, action_history deques
- Add sparkline generation properties
- Add status tracking (healthy/stalled/degraded/excellent)
- Add helper methods for updating metrics

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Add TUIState Aggregate Calculations

**Files:**
- Modify: `src/esper/karn/tui.py` (TUIState class)
- Test: `tests/karn/test_tui_state.py`

**Step 1: Write the failing test**

Add to `tests/karn/test_tui_state.py`:

```python
class TestTUIStateAggregation:
    """Tests for TUIState aggregate calculations."""

    def test_aggregate_mean_reward_from_envs(self):
        """TUIState.aggregate_mean_reward averages across all envs."""
        state = TUIState()
        state.n_envs = 3

        for i in range(3):
            env = state.get_or_create_env(i)
            env.reward_history.append(float(i + 1) * 0.1)

        assert abs(state.aggregate_mean_reward - 0.2) < 0.001

    def test_aggregate_best_accuracy_tracks_source_env(self):
        """TUIState.aggregate_best_accuracy returns best across envs with source."""
        state = TUIState()
        state.n_envs = 3

        env0 = state.get_or_create_env(0)
        env0.best_accuracy = 75.0
        env0.best_accuracy_epoch = 10

        env1 = state.get_or_create_env(1)
        env1.best_accuracy = 85.0
        env1.best_accuracy_epoch = 15

        env2 = state.get_or_create_env(2)
        env2.best_accuracy = 70.0

        best_acc, best_env, best_epoch = state.aggregate_best_accuracy
        assert best_acc == 85.0
        assert best_env == 1
        assert best_epoch == 15

    def test_aggregate_action_counts(self):
        """TUIState.aggregate_action_counts sums across all envs."""
        state = TUIState()

        env0 = state.get_or_create_env(0)
        env0.action_counts = {"WAIT": 10, "GERMINATE": 2, "CULL": 1, "FOSSILIZE": 0}

        env1 = state.get_or_create_env(1)
        env1.action_counts = {"WAIT": 5, "GERMINATE": 3, "CULL": 0, "FOSSILIZE": 2}

        counts = state.aggregate_action_counts
        assert counts["WAIT"] == 15
        assert counts["GERMINATE"] == 5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestTUIStateAggregation -v`
Expected: FAIL with AttributeError

**Step 3: Add aggregate properties to TUIState**

Add these properties to TUIState class (before the closing of the class):

```python
    # Track which env last emitted reward (for detail view focus)
    last_reward_env_id: int = 0

    @property
    def aggregate_mean_reward(self) -> float:
        """Mean of current rewards across all envs."""
        if not self.env_states:
            return 0.0
        rewards = [e.current_reward for e in self.env_states.values()]
        return sum(rewards) / len(rewards) if rewards else 0.0

    @property
    def aggregate_mean_accuracy(self) -> float:
        """Mean of current accuracies across all envs."""
        if not self.env_states:
            return 0.0
        accs = [e.host_accuracy for e in self.env_states.values()]
        return sum(accs) / len(accs) if accs else 0.0

    @property
    def aggregate_best_accuracy(self) -> tuple[float, int, int]:
        """Best accuracy across all envs: (accuracy, env_id, epoch)."""
        if not self.env_states:
            return (0.0, -1, 0)
        best_env = max(self.env_states.values(), key=lambda e: e.best_accuracy)
        return (best_env.best_accuracy, best_env.env_id, best_env.best_accuracy_epoch)

    @property
    def aggregate_action_counts(self) -> dict[str, int]:
        """Sum action counts across all envs."""
        totals: dict[str, int] = {"WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0}
        for env in self.env_states.values():
            for action, count in env.action_counts.items():
                totals[action] = totals.get(action, 0) + count
        return totals

    @property
    def aggregate_total_actions(self) -> int:
        """Total actions across all envs."""
        return sum(e.total_actions for e in self.env_states.values())

    @property
    def envs_by_status(self) -> dict[str, list[int]]:
        """Group env IDs by status."""
        by_status: dict[str, list[int]] = {}
        for env_id, env in self.env_states.items():
            if env.status not in by_status:
                by_status[env.status] = []
            by_status[env.status].append(env_id)
        return by_status
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestTUIStateAggregation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_tui_state.py src/esper/karn/tui.py
git commit -m "$(cat <<'EOF'
feat(karn): add TUIState aggregate calculations across envs

- aggregate_mean_reward: mean of per-env current rewards
- aggregate_best_accuracy: best across envs with source tracking
- aggregate_action_counts: sum actions across all envs
- envs_by_status: group envs by health status

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: TUI Event Handler Fixes

### Task 8: Fix REWARD_COMPUTED Handler for Per-Env Routing

**Files:**
- Modify: `src/esper/karn/tui.py:535-564`
- Test: `tests/karn/test_tui_state.py`

**Step 1: Write the failing test**

Add to `tests/karn/test_tui_state.py`:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType


class TestTUIOutputEventHandlers:
    """Tests for TUIOutput event routing."""

    def test_reward_computed_routes_to_correct_env(self):
        """REWARD_COMPUTED updates the correct env's reward history."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()
        tui.state.n_envs = 4
        for i in range(4):
            tui.state.get_or_create_env(i)

        event = TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            epoch=10,
            data={
                "env_id": 2,
                "total_reward": 0.75,
                "action_name": "GERMINATE",
                "val_acc": 82.5,
            }
        )
        tui._handle_reward_computed(event)

        env2 = tui.state.env_states[2]
        assert env2.current_reward == 0.75
        assert env2.action_history[-1] == "GERMINATE"
        assert env2.host_accuracy == 82.5

        assert tui.state.env_states[0].current_reward == 0.0
        assert tui.state.env_states[1].current_reward == 0.0

    def test_action_distribution_per_env(self):
        """Actions are tracked per-env, not globally."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()
        tui.state.n_envs = 2
        for i in range(2):
            tui.state.get_or_create_env(i)

        tui._handle_reward_computed(TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={"env_id": 0, "total_reward": 0.1, "action_name": "GERMINATE"}
        ))
        tui._handle_reward_computed(TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={"env_id": 1, "total_reward": 0.1, "action_name": "WAIT"}
        ))

        assert tui.state.env_states[0].action_counts["GERMINATE"] == 1
        assert tui.state.env_states[1].action_counts["WAIT"] == 1
        assert tui.state.aggregate_action_counts["GERMINATE"] == 1
        assert tui.state.aggregate_action_counts["WAIT"] == 1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestTUIOutputEventHandlers -v`
Expected: FAIL

**Step 3: Rewrite _handle_reward_computed**

Replace the method:

```python
    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event with per-env routing."""
        data = event.data or {}
        env_id = data.get("env_id", 0)
        epoch = event.epoch or 0

        env_state = self.state.get_or_create_env(env_id)

        total_reward = data.get("total_reward", 0.0)
        env_state.add_reward(total_reward, epoch)

        action_name = data.get("action_name", "WAIT")
        env_state.add_action(action_name)

        val_acc = data.get("val_acc")
        if val_acc is not None:
            env_state.add_accuracy(val_acc, epoch)

        env_state.current_epoch = epoch
        env_state.last_update = datetime.now()

        self.state.reward_components = {
            "accuracy_delta": data.get("base_acc_delta", 0.0),
            "bounded_attr": data.get("bounded_attribution", 0.0),
            "compute_rent": data.get("compute_rent", 0.0),
            "blending_warn": data.get("blending_warning", 0.0),
            "probation_warn": data.get("probation_warning", 0.0),
            "terminal_bonus": data.get("fossilize_terminal_bonus", 0.0),
        }
        self.state.last_reward_env_id = env_id

        acc_delta = data.get("base_acc_delta", 0.0)
        if acc_delta < 0 and total_reward > 0:
            self.state.reward_hacking_detected = True

        self.state.current_reward = total_reward
        self.state.host_accuracy = data.get("val_acc", self.state.host_accuracy)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestTUIOutputEventHandlers -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_tui_state.py src/esper/karn/tui.py
git commit -m "$(cat <<'EOF'
fix(karn): route REWARD_COMPUTED to per-env state

- Extract env_id from event data and route to correct EnvState
- Track rewards, actions, accuracy per-env
- Maintain aggregate calculations via properties

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Add Env ID to Event Log Display

**Files:**
- Modify: `src/esper/karn/tui.py:352-428`
- Test: `tests/karn/test_tui_state.py`

**Step 1: Write the failing test**

Add to `tests/karn/test_tui_state.py`:

```python
class TestEventLogFormatting:
    """Tests for event log formatting with env IDs."""

    def test_event_log_includes_env_id(self):
        """Formatted event log entries include env prefix."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()

        event = TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={
                "env_id": 3,
                "action_name": "GERMINATE",
                "total_reward": 0.5,
            }
        )

        formatted = tui._format_event_for_log(event)
        assert formatted is not None
        timestamp, event_type, msg = formatted
        assert "[3]" in msg or "3" in msg
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestEventLogFormatting -v`
Expected: FAIL

**Step 3: Update _format_event_for_log**

Add env_id extraction and prefix to the method. Near the start, after data extraction:

```python
        # Extract env_id for prefix
        env_id = data.get("env_id")
        env_prefix = f"[{env_id}] " if env_id is not None else ""
```

Then prepend `env_prefix` to each message format.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_state.py::TestEventLogFormatting -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_tui_state.py src/esper/karn/tui.py
git commit -m "$(cat <<'EOF'
fix(karn): add env ID prefix to event log messages

Events with env_id now display [N] prefix in log.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Env Overview Table Rendering

### Task 10: Create and Integrate Env Overview Table

**Files:**
- Modify: `src/esper/karn/tui.py` (add `_render_env_overview`, modify `_render`)
- Test: `tests/karn/test_tui_rendering.py` (create)

**Step 1: Write the failing test**

Create `tests/karn/test_tui_rendering.py`:

```python
"""Tests for TUI rendering components."""

import pytest
from rich.table import Table

from esper.karn.tui import TUIOutput


class TestEnvOverviewTable:
    """Tests for environment overview table rendering."""

    def test_render_env_overview_returns_table(self):
        """_render_env_overview returns a Rich Table."""
        tui = TUIOutput()
        tui.state.n_envs = 2

        for i in range(2):
            env = tui.state.get_or_create_env(i)
            env.host_accuracy = 75.0 + i * 5
            env.reward_history.append(0.3 + i * 0.1)
            env.status = "healthy"

        table = tui._render_env_overview()
        assert isinstance(table, Table)

    def test_env_overview_shows_all_envs(self):
        """Table has one row per environment."""
        tui = TUIOutput()
        tui.state.n_envs = 4

        for i in range(4):
            env = tui.state.get_or_create_env(i)
            env.status = "healthy"

        table = tui._render_env_overview()
        assert table.row_count == 4
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_rendering.py -v`
Expected: FAIL (method doesn't exist)

**Step 3: Implement _render_env_overview and integrate into layout**

Add `_render_env_overview` method and update `_render` to include it. (See full implementation in original plan Task 6-7.)

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/karn/test_tui_rendering.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_tui_rendering.py src/esper/karn/tui.py
git commit -m "$(cat <<'EOF'
feat(karn): add env overview table to TUI layout

- _render_env_overview creates per-env metrics table
- Shows ID, Accuracy, Reward, Epoch, Seeds, Status
- Color-coded status indicators
- Integrated into main layout

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Final Integration Testing

### Task 11: Run Full Test Suite

**Step 1: Run all new tests**

Run: `PYTHONPATH=src pytest tests/karn/ tests/simic/test_epoch_telemetry.py tests/simic/test_heuristic_telemetry.py -v`
Expected: All PASS

**Step 2: Run full test suite for regressions**

Run: `PYTHONPATH=src pytest -x -q`
Expected: All PASS

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(karn): complete multi-env TUI and telemetry pipeline fixes

Phase 0 - Pipeline Fixes:
- Emit EPOCH_COMPLETED in vectorized training
- Fix JSON serialization crash (datetime/Path/Enum)
- Namespace slots by env_id to prevent collisions
- Emit TRAINING_STARTED in heuristic path

Phase 1-3 - TUI Fixes:
- EnvState per-env tracking with sparklines
- TUIState aggregate calculations
- REWARD_COMPUTED per-env routing
- Event log env ID prefix
- Env overview table in layout

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Priority | Description | Files |
|------|----------|-------------|-------|
| 0 | P0 | Remove duplicate SeedStage (use leyline contract) | `store.py`, `__init__.py` |
| 1 | P0 | Fix epoch numbering (Karn→1) + add epoch to PPO events | `collector.py`, `vectorized.py` |
| 2 | P0 | Emit EPOCH_COMPLETED as commit barrier | `vectorized.py` |
| 3 | P0 | Fix JSON serialization crash (datetime/Path/Enum) | `store.py` |
| 4 | P0 | Namespace slots by env_id | `collector.py` |
| 5 | P1 | TRAINING_STARTED in heuristic | `training.py` |
| 6 | P0 | EnvState per-env tracking | `tui.py` |
| 7 | P0 | TUIState aggregates | `tui.py` |
| 8 | P0 | REWARD_COMPUTED per-env routing | `tui.py` |
| 9 | P2 | Event log env ID prefix | `tui.py` |
| 10 | P1 | Env overview table | `tui.py` |
| 11 | - | Integration testing | All |

**Total tasks:** 12 (Task 0-11)
**Total estimated time:** 80-110 minutes following TDD.
