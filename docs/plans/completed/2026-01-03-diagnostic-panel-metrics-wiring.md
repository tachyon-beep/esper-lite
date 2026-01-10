# Diagnostic Panel Metrics Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up the new SeedLifecycleStats, ObservationStats, and EpisodeStats schema fields end-to-end from telemetry emission through aggregator computation to widget rendering.

**Architecture:** The metrics flow through 4 layers:
1. **Telemetry Emission** (simic/training) - Emit raw data in existing or new payloads
2. **Aggregator Computation** (karn/sanctum/aggregator.py) - Process events and populate schema fields
3. **Snapshot Propagation** (schema.py) - Carry computed metrics to widgets
4. **Widget Rendering** (tamiyo_brain/*.py) - Display the metrics

**Tech Stack:** Python dataclasses, Textual TUI, Rich text rendering

---

## Current State Analysis

### Already Available in Aggregator:
- `_cumulative_fossilized` and `_cumulative_pruned` counters
- `_cumulative_blueprint_spawns` (tracks germinations per blueprint)
- Slot stage counts computed in `_get_snapshot_unlocked()`
- Per-env seed states with `epochs_total` at fossilize/prune time

### Missing Data Sources:
- `_cumulative_germinated` - not tracked directly (can derive from blueprint_spawns sum)
- Seed lifespan history - epochs_total is available at terminal events
- Observation feature statistics - not emitted by simic
- Episode length/outcome telemetry - not emitted by simic

---

## Task 1: Add Germination Counter to Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:175-252` (counters section)
- Modify: `src/esper/karn/sanctum/aggregator.py:900-912` (SEED_GERMINATED handler)

**Step 1: Add `_cumulative_germinated` counter**

```python
# In SanctumAggregator.__post_init__() and field declarations
_cumulative_germinated: int = 0

# In __post_init__():
self._cumulative_germinated = 0
```

**Step 2: Increment counter in SEED_GERMINATED handler**

```python
# In _handle_seed_event(), after line 903 (env.active_seed_count += 1)
self._cumulative_germinated += 1
```

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): track cumulative germination count"
```

---

## Task 2: Add Seed Lifespan Tracking

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:125` (add field)
- Modify: `src/esper/karn/sanctum/aggregator.py:960-978` (SEED_FOSSILIZED handler)
- Modify: `src/esper/karn/sanctum/aggregator.py:1000-1028` (SEED_PRUNED handler)

**Step 1: Add lifespan history deque**

```python
# In SanctumAggregator class fields
_seed_lifespan_history: deque[int] = field(default_factory=lambda: deque(maxlen=100))

# In __post_init__():
self._seed_lifespan_history = deque(maxlen=100)
```

**Step 2: Track lifespan at fossilize**

```python
# In SEED_FOSSILIZED handler, after line 964 (seed.epochs_total = ...)
if fossilized_payload.epochs_total > 0:
    self._seed_lifespan_history.append(fossilized_payload.epochs_total)
```

**Step 3: Track lifespan at prune**

```python
# In SEED_PRUNED handler, after line 1002 (seed.epochs_total = ...)
if pruned_payload.epochs_total > 0:
    self._seed_lifespan_history.append(pruned_payload.epochs_total)
```

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): track seed lifespan at terminal events"
```

---

## Task 3: Add Rate History for Trend Detection

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:125` (add fields)
- Modify: `src/esper/karn/sanctum/aggregator.py:1064-1215` (BATCH_EPOCH_COMPLETED handler)

**Step 1: Add rate history deques**

```python
# In SanctumAggregator class fields
_germination_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
_prune_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
_fossilize_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))

# In __post_init__():
self._germination_rate_history = deque(maxlen=20)
self._prune_rate_history = deque(maxlen=20)
self._fossilize_rate_history = deque(maxlen=20)
```

**Step 2: Compute and append rates at batch end**

```python
# In _handle_batch_epoch_completed(), after line 1086 (episode return tracking)
# Compute per-episode rates for trend tracking
if self._current_episode > 0:
    germ_rate = self._cumulative_germinated / self._current_episode
    prune_rate = self._cumulative_pruned / self._current_episode
    foss_rate = self._cumulative_fossilized / self._current_episode
    self._germination_rate_history.append(germ_rate)
    self._prune_rate_history.append(prune_rate)
    self._fossilize_rate_history.append(foss_rate)
```

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): track lifecycle rate history for trends"
```

---

## Task 4: Compute and Populate SeedLifecycleStats in Snapshot

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:23-42` (imports)
- Modify: `src/esper/karn/sanctum/aggregator.py:444-495` (_get_snapshot_unlocked)

**Step 1: Import SeedLifecycleStats**

```python
# Add to imports
from esper.karn.sanctum.schema import (
    # ... existing imports ...
    SeedLifecycleStats,
)
```

**Step 2: Add helper function for trend detection**

```python
# Add after normalize_action() function
def detect_rate_trend(history: deque[float]) -> str:
    """Detect trend in rate history (rising/stable/falling)."""
    if len(history) < 5:
        return "stable"

    recent = list(history)[-5:]
    older = list(history)[-10:-5] if len(history) >= 10 else list(history)[:5]

    if not older:
        return "stable"

    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)

    # 20% change threshold
    threshold = 0.2 * max(abs(older_mean), 0.01)
    diff = recent_mean - older_mean

    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    return "stable"
```

**Step 3: Compute SeedLifecycleStats in _get_snapshot_unlocked**

```python
# Add after avg_epochs calculation (around line 442), before snapshot = SanctumSnapshot(...)

# Compute seed lifecycle stats
blend_success = (
    self._cumulative_fossilized / max(1, self._cumulative_fossilized + self._cumulative_pruned)
)
avg_lifespan = (
    sum(self._seed_lifespan_history) / len(self._seed_lifespan_history)
    if self._seed_lifespan_history else 0.0
)
current_ep = max(1, self._current_episode)

seed_lifecycle = SeedLifecycleStats(
    germination_count=self._cumulative_germinated,
    prune_count=self._cumulative_pruned,
    fossilize_count=self._cumulative_fossilized,
    active_count=active_slots,
    total_slots=total_slots,
    germination_rate=self._cumulative_germinated / current_ep,
    prune_rate=self._cumulative_pruned / current_ep,
    fossilize_rate=self._cumulative_fossilized / current_ep,
    blend_success_rate=blend_success,
    avg_lifespan_epochs=avg_lifespan,
    germination_trend=detect_rate_trend(self._germination_rate_history),
    prune_trend=detect_rate_trend(self._prune_rate_history),
    fossilize_trend=detect_rate_trend(self._fossilize_rate_history),
)
```

**Step 4: Add seed_lifecycle to snapshot construction**

```python
# In SanctumSnapshot(...) constructor, add:
seed_lifecycle=seed_lifecycle,
```

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): compute SeedLifecycleStats in aggregator"
```

---

## Task 5: Add cumulative_germinated to Snapshot

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:484-488`

**Step 1: Add cumulative_germinated to snapshot**

```python
# In SanctumSnapshot(...) constructor, alongside other cumulative fields:
cumulative_germinated=self._cumulative_germinated,
```

**Step 2: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): include cumulative_germinated in snapshot"
```

---

## Task 6: Verify SlotsPanel Rendering Works

**Files:**
- No changes needed - SlotsPanel already updated to use snapshot.seed_lifecycle

**Step 1: Run quick visual test**

```bash
# Start a training run with Sanctum
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 5 --tui sanctum
```

**Step 2: Verify lifecycle section appears**

Expected: After the stage distribution bars, you should see:
- Separator line
- "Active: X/Y  Foss: Z  Prune: W  Germ: V"
- Rate lines with trend arrows
- Lifespan and Blend success rate

**Step 3: Document any visual issues**

If metrics show all zeros, the aggregator computation is working but telemetry may not be flowing. Check that SEED_GERMINATED events are being emitted.

---

## Task 7: Stub ObservationStats (Defer Telemetry)

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:444`

**Step 1: Add placeholder ObservationStats**

```python
# In _get_snapshot_unlocked(), after seed_lifecycle computation:
# ObservationStats requires new telemetry from simic - stub for now
observation_stats = ObservationStats()  # Default values until telemetry added
```

**Step 2: Add to snapshot**

```python
# In SanctumSnapshot(...) constructor:
observation_stats=observation_stats,
```

**Step 3: Add import**

```python
from esper.karn.sanctum.schema import (
    # ... existing ...
    ObservationStats,
)
```

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): stub ObservationStats in aggregator (telemetry pending)"
```

---

## Task 8: Stub EpisodeStats (Defer Telemetry)

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:444`

**Step 1: Add placeholder EpisodeStats**

```python
# In _get_snapshot_unlocked(), after observation_stats:
# EpisodeStats requires episode length/outcome telemetry - stub for now
episode_stats = EpisodeStats(
    total_episodes=self._current_episode,
)
```

**Step 2: Add to snapshot**

```python
# In SanctumSnapshot(...) constructor:
episode_stats=episode_stats,
```

**Step 3: Add import**

```python
from esper.karn.sanctum.schema import (
    # ... existing ...
    EpisodeStats,
)
```

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): stub EpisodeStats in aggregator (telemetry pending)"
```

---

## Task 9: Create EpisodeMetricsPanel Widget

**Files:**
- Create: `src/esper/karn/sanctum/widgets/tamiyo_brain/episode_metrics_panel.py`
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain/__init__.py`

**Step 1: Create the widget file**

```python
"""EpisodeMetricsPanel - Episode-level training metrics.

Displays:
- Episode length statistics (mean/std/min/max)
- Outcome rates (timeout, success, early termination)
- Completion trend indicator
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class EpisodeMetricsPanel(Static):
    """Episode-level metrics panel.

    Extends Static directly for minimal layout overhead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "EPISODE HEALTH"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        """Render episode metrics."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        stats = self._snapshot.episode_stats
        result = Text()

        # Line 1: Length statistics
        result.append("Length  ", style="dim")
        result.append(f"μ:{stats.length_mean:.0f}", style="cyan")
        result.append(f" σ:{stats.length_std:.0f}", style="dim")
        result.append(f" [{stats.length_min}-{stats.length_max}]", style="dim")
        result.append("\n")

        # Line 2: Outcome rates
        result.append("Timeout ", style="dim")
        timeout_style = "red" if stats.timeout_rate > 0.2 else "yellow" if stats.timeout_rate > 0.1 else "green"
        result.append(f"{stats.timeout_rate:.0%}", style=timeout_style)

        result.append("  Success ", style="dim")
        success_style = "green" if stats.success_rate > 0.7 else "yellow" if stats.success_rate > 0.5 else "red"
        result.append(f"{stats.success_rate:.0%}", style=success_style)

        result.append("  Early ", style="dim")
        result.append(f"{stats.early_termination_rate:.0%}", style="dim")
        result.append("\n")

        # Line 3: Total episodes + trend
        result.append("Episodes ", style="dim")
        result.append(f"{stats.total_episodes}", style="cyan")

        trend_map = {
            "improving": ("↗", "green"),
            "stable": ("→", "dim"),
            "declining": ("↘", "red"),
        }
        arrow, style = trend_map.get(stats.completion_trend, ("→", "dim"))
        result.append(f"  Trend: {arrow}", style=style)

        return result
```

**Step 2: Add export to __init__.py**

```python
# In __init__.py, add to imports and __all__:
from .episode_metrics_panel import EpisodeMetricsPanel
```

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain/episode_metrics_panel.py
git add src/esper/karn/sanctum/widgets/tamiyo_brain/__init__.py
git commit -m "feat(sanctum): add EpisodeMetricsPanel widget"
```

---

## Task 10: Update TamiyoBrain Layout with Episode Row

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain/tamiyo_brain.py`

**Step 1: Import EpisodeMetricsPanel**

```python
from .episode_metrics_panel import EpisodeMetricsPanel
```

**Step 2: Update CSS heights**

```python
# In DEFAULT_CSS, adjust heights:
# top-row: 13 -> 14 (add 1 for expanded content)
# heads-row: 27 -> 22 (reduce by 5)
# Add new episode-row: 6

"""
    #top-row {
        height: 14;
        width: 100%;
    }

    #heads-row {
        height: 22;
        width: 100%;
    }

    #episode-row {
        height: 6;
        width: 100%;
    }

    #episode-metrics-panel {
        width: 100%;
        height: 1fr;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }
"""
```

**Step 3: Add episode row to compose()**

```python
# In compose(), after heads-row:
with Horizontal(id="episode-row"):
    yield EpisodeMetricsPanel(id="episode-metrics-panel")
```

**Step 4: Add update call in update_snapshot()**

```python
# Add after other query_one calls:
self.query_one("#episode-metrics-panel", EpisodeMetricsPanel).update_snapshot(snapshot)
```

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain/tamiyo_brain.py
git commit -m "feat(sanctum): add EpisodeMetricsPanel to TamiyoBrain layout"
```

---

## Task 11: Add Observation Stats to HealthStatusPanel (Deferred)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py`

**Step 1: Add observation stats rendering**

```python
# Add at the end of render(), before return result:

# Separator before observation stats
result.append("─" * 36, style="dim")
result.append("\n")

# Observation stats (stub until telemetry available)
obs = self._snapshot.observation_stats
result.append("Obs Stats ", style="dim")
if obs.nan_count > 0 or obs.inf_count > 0:
    result.append(f"NaN:{obs.nan_count} Inf:{obs.inf_count}", style="red bold")
else:
    result.append("healthy", style="green")
```

**Step 2: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py
git commit -m "feat(sanctum): add ObservationStats stub to HealthStatusPanel"
```

---

## Task 12: Write Integration Test for Lifecycle Stats

**Files:**
- Create: `tests/karn/sanctum/test_lifecycle_stats.py`

**Step 1: Write test**

```python
"""Test SeedLifecycleStats computation in aggregator."""

import pytest
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import (
    TelemetryEvent,
    EventType,
    SeedGerminatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
)


def test_lifecycle_stats_computed():
    """Verify germination count, rates, and blend success are computed."""
    agg = SanctumAggregator(num_envs=2)

    # Emit TRAINING_STARTED
    from esper.leyline import TrainingStartedPayload
    agg.process_event(TelemetryEvent(
        event_type=EventType.TRAINING_STARTED,
        data=TrainingStartedPayload(
            episode_id="test",
            task="test",
            n_envs=2,
            max_epochs=10,
            max_batches=5,
            host_params=1000,
            slot_ids=("r0c0", "r0c1"),
        ),
    ))

    # Emit 3 germinations
    for i in range(3):
        agg.process_event(TelemetryEvent(
            event_type=EventType.SEED_GERMINATED,
            slot_id="r0c0",
            data=SeedGerminatedPayload(
                env_id=0,
                slot_id="r0c0",
                blueprint_id="test_bp",
            ),
        ))

    # Emit 1 fossilize
    agg.process_event(TelemetryEvent(
        event_type=EventType.SEED_FOSSILIZED,
        slot_id="r0c0",
        data=SeedFossilizedPayload(
            env_id=0,
            slot_id="r0c0",
            blueprint_id="test_bp",
            epochs_total=50,
        ),
    ))

    # Emit 1 prune
    agg.process_event(TelemetryEvent(
        event_type=EventType.SEED_PRUNED,
        slot_id="r0c0",
        data=SeedPrunedPayload(
            env_id=0,
            slot_id="r0c0",
            reason="test",
            blueprint_id="test_bp",
            epochs_total=30,
        ),
    ))

    snapshot = agg.get_snapshot()
    lifecycle = snapshot.seed_lifecycle

    # Verify counts
    assert lifecycle.germination_count == 3
    assert lifecycle.fossilize_count == 1
    assert lifecycle.prune_count == 1

    # Verify blend success rate (1 fossilized / (1 fossilized + 1 pruned) = 0.5)
    assert lifecycle.blend_success_rate == pytest.approx(0.5)

    # Verify lifespan is tracked (avg of 50 and 30 = 40)
    assert lifecycle.avg_lifespan_epochs == pytest.approx(40.0)
```

**Step 2: Run test**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_lifecycle_stats.py -v
```

**Step 3: Commit**

```bash
git add tests/karn/sanctum/test_lifecycle_stats.py
git commit -m "test(sanctum): add integration test for SeedLifecycleStats"
```

---

## Future Work (Telemetry Required)

### ObservationStats Telemetry
To fully populate ObservationStats, simic needs to emit observation feature statistics. Suggested approach:
1. Add `OBSERVATION_STATS` event type to leyline
2. Compute feature stats in vectorized training loop (every N batches)
3. Add handler in aggregator to populate ObservationStats fields

### EpisodeStats Telemetry
To fully populate EpisodeStats, simic needs to emit episode outcome details. Suggested approach:
1. Extend `EPISODE_OUTCOME` payload with length, timeout_flag, success_flag
2. Track episode lengths in aggregator's `_handle_episode_outcome()`
3. Compute rolling statistics from outcome history

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Add germination counter | Ready |
| 2 | Add seed lifespan tracking | Ready |
| 3 | Add rate history for trends | Ready |
| 4 | Compute SeedLifecycleStats | Ready |
| 5 | Add cumulative_germinated to snapshot | Ready |
| 6 | Verify SlotsPanel rendering | Ready |
| 7 | Stub ObservationStats | Ready |
| 8 | Stub EpisodeStats | Ready |
| 9 | Create EpisodeMetricsPanel widget | Ready |
| 10 | Update TamiyoBrain layout | Ready |
| 11 | Add ObservationStats to HealthStatusPanel | Ready |
| 12 | Write integration test | Ready |
