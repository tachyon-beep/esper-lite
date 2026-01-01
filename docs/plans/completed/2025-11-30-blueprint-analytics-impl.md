# Blueprint Analytics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add telemetry to track blueprint performance (fossilization rates, accuracy deltas, churn) and per-environment seed scoreboards.

**Architecture:** Create `BlueprintAnalytics` as a Nissa `OutputBackend` that aggregates telemetry events. Kasmina emits enriched events with `blueprint_id`, `improvement`, and `params_added`. Simic injects `env_id` via callback and prints periodic summaries.

**Tech Stack:** Python dataclasses, defaultdict, Nissa OutputBackend protocol

---

## Task 1: Create BlueprintStats and SeedScoreboard Dataclasses

**Files:**
- Create: `src/esper/nissa/analytics.py`
- Test: `tests/test_nissa_analytics.py`

**Step 1: Write the failing test**

Create `tests/test_nissa_analytics.py`:

```python
"""Tests for Nissa Blueprint Analytics."""

import pytest
from esper.nissa.analytics import BlueprintStats, SeedScoreboard


class TestBlueprintStats:
    """Tests for BlueprintStats dataclass."""

    def test_initial_values(self):
        """Stats start at zero."""
        stats = BlueprintStats()
        assert stats.germinated == 0
        assert stats.fossilized == 0
        assert stats.culled == 0
        assert stats.acc_deltas == []
        assert stats.churns == []

    def test_mean_acc_delta_empty(self):
        """Empty acc_deltas returns 0."""
        stats = BlueprintStats()
        assert stats.mean_acc_delta == 0.0

    def test_mean_acc_delta_with_values(self):
        """Mean accuracy delta calculated correctly."""
        stats = BlueprintStats(acc_deltas=[1.0, 2.0, 3.0])
        assert stats.mean_acc_delta == 2.0

    def test_fossilization_rate_no_terminal(self):
        """Rate is 0% when no seeds reached terminal state."""
        stats = BlueprintStats(germinated=5)
        assert stats.fossilization_rate == 0.0

    def test_fossilization_rate_all_fossilized(self):
        """Rate is 100% when all seeds fossilized."""
        stats = BlueprintStats(germinated=5, fossilized=5, culled=0)
        assert stats.fossilization_rate == 100.0

    def test_fossilization_rate_mixed(self):
        """Rate calculated correctly for mixed outcomes."""
        stats = BlueprintStats(germinated=10, fossilized=3, culled=7)
        assert stats.fossilization_rate == 30.0


class TestSeedScoreboard:
    """Tests for SeedScoreboard dataclass."""

    def test_initial_values(self):
        """Scoreboard starts empty."""
        sb = SeedScoreboard()
        assert sb.total_germinated == 0
        assert sb.total_fossilized == 0
        assert sb.params_added == 0
        assert sb.live_blueprint is None

    def test_compute_cost_empty(self):
        """Empty scoreboard has 1.0x cost."""
        sb = SeedScoreboard()
        assert sb.compute_cost == 1.0

    def test_compute_cost_with_fossilized(self):
        """Compute cost accumulates from fossilized seeds."""
        sb = SeedScoreboard()
        sb.fossilized_by_blueprint["depthwise"] = 2  # 2 * 0.08 = 0.16 extra
        sb.fossilized_by_blueprint["attention"] = 1  # 1 * 0.35 = 0.35 extra
        # Total: 1.0 + 0.16 + 0.35 = 1.51
        assert abs(sb.compute_cost - 1.51) < 0.01

    def test_params_percentage(self):
        """Params percentage calculated correctly."""
        sb = SeedScoreboard(params_added=10000, host_params=100000)
        assert sb.params_percentage == 10.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_nissa_analytics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'esper.nissa.analytics'`

**Step 3: Write minimal implementation**

Create `src/esper/nissa/analytics.py`:

```python
"""Nissa Analytics - Blueprint performance aggregation.

Aggregates telemetry events into strategic dashboards:
- BlueprintStats: Per-blueprint fossilization rates and accuracy metrics
- SeedScoreboard: Per-environment cumulative seed tracking
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Callable


# =============================================================================
# Compute Cost Multipliers
# =============================================================================

BLUEPRINT_COMPUTE_MULTIPLIERS: dict[str, float] = {
    "depthwise": 1.08,      # Cheap - depthwise separable
    "conv_enhance": 1.15,   # Moderate - adds conv layers
    "norm": 1.02,           # Minimal - just normalization
    "attention": 1.35,      # Expensive - O(n²) attention
}


def compute_cost_for_blueprint(blueprint_id: str) -> float:
    """Return compute multiplier for a blueprint type."""
    return BLUEPRINT_COMPUTE_MULTIPLIERS.get(blueprint_id, 1.1)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BlueprintStats:
    """Performance statistics for a single blueprint type."""

    germinated: int = 0
    fossilized: int = 0
    culled: int = 0
    acc_deltas: list[float] = field(default_factory=list)
    churns: list[float] = field(default_factory=list)

    @property
    def mean_acc_delta(self) -> float:
        """Mean accuracy improvement at terminal state."""
        return sum(self.acc_deltas) / len(self.acc_deltas) if self.acc_deltas else 0.0

    @property
    def mean_churn(self) -> float:
        """Mean accuracy change on cull (usually negative)."""
        return sum(self.churns) / len(self.churns) if self.churns else 0.0

    @property
    def fossilization_rate(self) -> float:
        """Percentage of terminal seeds that fossilized (not culled)."""
        total = self.fossilized + self.culled
        return (self.fossilized / total * 100) if total > 0 else 0.0


@dataclass
class SeedScoreboard:
    """Cumulative seed tracking for an environment."""

    total_germinated: int = 0
    total_fossilized: int = 0
    total_culled: int = 0
    fossilized_by_blueprint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    live_blueprint: str | None = None
    params_added: int = 0
    host_params: int = 0

    @property
    def compute_cost(self) -> float:
        """Estimated compute cost relative to baseline (1.0)."""
        cost = 1.0
        for bp_id, count in self.fossilized_by_blueprint.items():
            cost += (compute_cost_for_blueprint(bp_id) - 1.0) * count
        return cost

    @property
    def params_percentage(self) -> float:
        """Params added as percentage of host."""
        return (self.params_added / self.host_params * 100) if self.host_params > 0 else 0.0


__all__ = [
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
    "BlueprintStats",
    "SeedScoreboard",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_nissa_analytics.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/esper/nissa/analytics.py tests/test_nissa_analytics.py
git commit -m "feat(nissa): add BlueprintStats and SeedScoreboard dataclasses"
```

---

## Task 2: Implement BlueprintAnalytics OutputBackend

**Files:**
- Modify: `src/esper/nissa/analytics.py`
- Test: `tests/test_nissa_analytics.py`

**Step 1: Write the failing test**

Add to `tests/test_nissa_analytics.py`:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.analytics import BlueprintAnalytics


class TestBlueprintAnalytics:
    """Tests for BlueprintAnalytics OutputBackend."""

    def test_tracks_germination(self):
        """Germination events increment counters."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            seed_id="seed_001",
            data={"blueprint_id": "depthwise", "env_id": 0, "params": 5000},
        )
        analytics.emit(event)

        assert analytics.stats["depthwise"].germinated == 1
        assert analytics.scoreboards[0].total_germinated == 1
        assert analytics.scoreboards[0].live_blueprint == "depthwise"

    def test_tracks_fossilization(self):
        """Fossilization events update stats and scoreboard."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_FOSSILIZED,
            seed_id="seed_001",
            data={
                "blueprint_id": "depthwise",
                "env_id": 0,
                "improvement": 2.5,
                "params_added": 10000,
            },
        )
        analytics.emit(event)

        assert analytics.stats["depthwise"].fossilized == 1
        assert analytics.stats["depthwise"].acc_deltas == [2.5]
        assert analytics.scoreboards[0].total_fossilized == 1
        assert analytics.scoreboards[0].params_added == 10000
        assert analytics.scoreboards[0].fossilized_by_blueprint["depthwise"] == 1

    def test_tracks_cull(self):
        """Cull events update stats with churn."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_CULLED,
            seed_id="seed_001",
            data={
                "blueprint_id": "attention",
                "env_id": 1,
                "improvement": -0.3,
                "reason": "no_improvement",
            },
        )
        analytics.emit(event)

        assert analytics.stats["attention"].culled == 1
        assert analytics.stats["attention"].churns == [-0.3]
        assert analytics.scoreboards[1].total_culled == 1

    def test_ignores_irrelevant_events(self):
        """Non-lifecycle events are ignored."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            seed_id="seed_001",
            data={"val_accuracy": 75.0},
        )
        analytics.emit(event)

        assert len(analytics.stats) == 0
        assert len(analytics.scoreboards) == 0

    def test_summary_table_format(self):
        """Summary table is formatted correctly."""
        analytics = BlueprintAnalytics()
        # Add some test data
        analytics.stats["depthwise"].germinated = 10
        analytics.stats["depthwise"].fossilized = 6
        analytics.stats["depthwise"].culled = 4
        analytics.stats["depthwise"].acc_deltas = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0]

        table = analytics.summary_table()

        assert "Blueprint Stats:" in table
        assert "depthwise" in table
        assert "60.0%" in table  # fossilization rate

    def test_snapshot_serializable(self):
        """Snapshot returns serializable dict."""
        analytics = BlueprintAnalytics()
        analytics.stats["depthwise"].germinated = 5
        analytics.scoreboards[0].total_germinated = 5

        snapshot = analytics.snapshot()

        assert "stats" in snapshot
        assert "scoreboards" in snapshot
        assert snapshot["stats"]["depthwise"]["germinated"] == 5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_nissa_analytics.py::TestBlueprintAnalytics -v`
Expected: FAIL with `ImportError: cannot import name 'BlueprintAnalytics'`

**Step 3: Write minimal implementation**

Add to `src/esper/nissa/analytics.py`:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import OutputBackend


class BlueprintAnalytics(OutputBackend):
    """Aggregates blueprint performance from telemetry events.

    Implements OutputBackend to receive events from NissaHub.
    Tracks:
    - Per-blueprint stats (germinated, fossilized, culled, accuracy)
    - Per-environment scoreboards (params, compute cost, distribution)
    """

    def __init__(self):
        self.stats: dict[str, BlueprintStats] = defaultdict(BlueprintStats)
        self.scoreboards: dict[int, SeedScoreboard] = {}

    def emit(self, event: TelemetryEvent) -> None:
        """Process lifecycle events to update stats."""
        if event.event_type == TelemetryEventType.SEED_GERMINATED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)

            self.stats[bp_id].germinated += 1
            sb = self._get_scoreboard(env_id)
            sb.total_germinated += 1
            sb.live_blueprint = bp_id

        elif event.event_type == TelemetryEventType.SEED_FOSSILIZED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            improvement = event.data.get("improvement", 0.0)
            params = event.data.get("params_added", 0)

            self.stats[bp_id].fossilized += 1
            self.stats[bp_id].acc_deltas.append(improvement)

            sb = self._get_scoreboard(env_id)
            sb.total_fossilized += 1
            sb.fossilized_by_blueprint[bp_id] += 1
            sb.params_added += params
            sb.live_blueprint = None

        elif event.event_type == TelemetryEventType.SEED_CULLED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            improvement = event.data.get("improvement", 0.0)

            self.stats[bp_id].culled += 1
            self.stats[bp_id].churns.append(improvement)

            sb = self._get_scoreboard(env_id)
            sb.total_culled += 1
            sb.live_blueprint = None

    def _get_scoreboard(self, env_id: int) -> SeedScoreboard:
        """Get or create scoreboard for environment."""
        if env_id not in self.scoreboards:
            self.scoreboards[env_id] = SeedScoreboard()
        return self.scoreboards[env_id]

    def summary_table(self) -> str:
        """Pretty-print blueprint performance stats."""
        lines = ["Blueprint Stats:"]
        lines.append("  " + "-" * 75)
        lines.append(
            f"  {'Blueprint':<14} {'Germ':>5} {'Foss':>5} {'Cull':>5} "
            f"{'Rate':>6} {'ΔAcc':>8} {'Churn':>8}"
        )
        lines.append("  " + "-" * 75)

        for bp_id in sorted(self.stats.keys()):
            s = self.stats[bp_id]
            lines.append(
                f"  {bp_id:<14} {s.germinated:>5} {s.fossilized:>5} "
                f"{s.culled:>5} {s.fossilization_rate:>5.1f}% "
                f"{s.mean_acc_delta:>+7.2f}% {s.mean_churn:>+7.2f}%"
            )
        return "\n".join(lines)

    def scoreboard_table(self, env_id: int = 0) -> str:
        """Pretty-print scoreboard for an environment."""
        sb = self._get_scoreboard(env_id)

        dist = ", ".join(
            f"{bp} x{count}" for bp, count in sb.fossilized_by_blueprint.items()
        )

        lines = [
            f"Seed Scoreboard (env {env_id}):",
            f"  Fossilized: {sb.total_fossilized} "
            f"(+{sb.params_added/1000:.1f}K params, +{sb.params_percentage:.1f}% of host)",
            f"  Compute cost: {sb.compute_cost:.2f}x baseline",
            f"  Distribution: {dist or 'none'}",
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return serializable snapshot for history."""
        return {
            "stats": {
                bp: {
                    "germinated": s.germinated,
                    "fossilized": s.fossilized,
                    "culled": s.culled,
                    "mean_acc_delta": s.mean_acc_delta,
                    "mean_churn": s.mean_churn,
                    "fossilization_rate": s.fossilization_rate,
                }
                for bp, s in self.stats.items()
            },
            "scoreboards": {
                env_id: {
                    "total_germinated": sb.total_germinated,
                    "total_fossilized": sb.total_fossilized,
                    "total_culled": sb.total_culled,
                    "params_added": sb.params_added,
                    "compute_cost": sb.compute_cost,
                }
                for env_id, sb in self.scoreboards.items()
            },
        }


# Update __all__
__all__ = [
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
    "BlueprintStats",
    "SeedScoreboard",
    "BlueprintAnalytics",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_nissa_analytics.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/nissa/analytics.py tests/test_nissa_analytics.py
git commit -m "feat(nissa): add BlueprintAnalytics OutputBackend"
```

---

## Task 3: Update Nissa __init__.py Exports

**Files:**
- Modify: `src/esper/nissa/__init__.py`

**Step 1: Verify current exports**

Run: `python -c "from esper.nissa import BlueprintAnalytics"` should fail.

**Step 2: Add exports**

Edit `src/esper/nissa/__init__.py`, add after line 45:

```python
from esper.nissa.analytics import (
    BlueprintStats,
    SeedScoreboard,
    BlueprintAnalytics,
    BLUEPRINT_COMPUTE_MULTIPLIERS,
    compute_cost_for_blueprint,
)
```

And update `__all__` to include:

```python
__all__ = [
    # Config
    "TelemetryConfig",
    "GradientConfig",
    "LossLandscapeConfig",
    "PerClassConfig",
    # Tracker
    "DiagnosticTracker",
    "GradientStats",
    "GradientHealth",
    "EpochSnapshot",
    # Output
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "NissaHub",
    "get_hub",
    "emit",
    # Analytics
    "BlueprintStats",
    "SeedScoreboard",
    "BlueprintAnalytics",
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
]
```

**Step 3: Verify import works**

Run: `PYTHONPATH=src python -c "from esper.nissa import BlueprintAnalytics; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/nissa/__init__.py
git commit -m "feat(nissa): export analytics module"
```

---

## Task 4: Enrich Kasmina Telemetry Events

**Files:**
- Modify: `src/esper/kasmina/slot.py:499` (germinate)
- Modify: `src/esper/kasmina/slot.py:543` (fossilize)
- Modify: `src/esper/kasmina/slot.py:553-560` (cull)
- Test: `tests/test_kasmina_telemetry.py`

**Step 1: Write the failing test**

Create `tests/test_kasmina_telemetry.py`:

```python
"""Tests for enriched Kasmina telemetry events."""

import pytest
import torch

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.kasmina.slot import SeedSlot
from esper.kasmina.blueprints import BlueprintCatalog


class TestEnrichedTelemetry:
    """Tests for enriched telemetry event data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.events: list[TelemetryEvent] = []

        def capture_event(event: TelemetryEvent):
            self.events.append(event)

        self.slot = SeedSlot(
            slot_id="test_slot",
            channels=64,
            device="cpu",
            on_telemetry=capture_event,
            fast_mode=False,
        )

    def test_germinate_emits_blueprint_id(self):
        """Germination event includes blueprint_id."""
        self.slot.germinate("depthwise", "seed_001")

        assert len(self.events) == 1
        event = self.events[0]
        assert event.event_type == TelemetryEventType.SEED_GERMINATED
        assert event.data["blueprint_id"] == "depthwise"
        assert event.data["seed_id"] == "seed_001"
        assert "params" in event.data
        assert event.data["params"] > 0

    def test_fossilize_emits_improvement(self):
        """Fossilization event includes improvement and params."""
        self.slot.germinate("depthwise", "seed_001")
        self.events.clear()

        # Simulate training improvement
        self.slot.state.metrics.initial_val_accuracy = 70.0
        self.slot.state.metrics.current_val_accuracy = 75.0

        # Advance through stages to FOSSILIZED
        self.slot.state.stage = SeedStage.BLENDING
        self.slot.advance_stage(SeedStage.FOSSILIZED)

        # Find fossilization event
        foss_events = [e for e in self.events
                       if e.event_type == TelemetryEventType.SEED_FOSSILIZED]
        assert len(foss_events) == 1

        event = foss_events[0]
        assert event.data["blueprint_id"] == "depthwise"
        assert event.data["improvement"] == 5.0  # 75 - 70
        assert "params_added" in event.data

    def test_cull_emits_improvement(self):
        """Cull event includes improvement (churn metric)."""
        self.slot.germinate("attention", "seed_002")
        self.events.clear()

        # Simulate negative improvement
        self.slot.state.metrics.initial_val_accuracy = 70.0
        self.slot.state.metrics.current_val_accuracy = 69.5

        self.slot.cull("no_improvement")

        cull_events = [e for e in self.events
                       if e.event_type == TelemetryEventType.SEED_CULLED]
        assert len(cull_events) == 1

        event = cull_events[0]
        assert event.data["blueprint_id"] == "attention"
        assert event.data["improvement"] == -0.5
        assert event.data["reason"] == "no_improvement"


# Need this import for the test
from esper.leyline import SeedStage
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_kasmina_telemetry.py -v`
Expected: FAIL - events don't contain `blueprint_id` or `improvement`

**Step 3: Update slot.py**

Edit `src/esper/kasmina/slot.py`:

**Line 499** - Replace:
```python
        self._emit_telemetry(TelemetryEventType.SEED_GERMINATED)
```
With:
```python
        self._emit_telemetry(
            TelemetryEventType.SEED_GERMINATED,
            data={
                "blueprint_id": blueprint_id,
                "seed_id": seed_id,
                "params": sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
            }
        )
```

**Line 543** - Replace:
```python
                    self._emit_telemetry(TelemetryEventType.SEED_FOSSILIZED)
```
With:
```python
                    self._emit_telemetry(
                        TelemetryEventType.SEED_FOSSILIZED,
                        data={
                            "blueprint_id": self.state.blueprint_id,
                            "seed_id": self.state.seed_id,
                            "improvement": self.state.metrics.total_improvement,
                            "params_added": sum(
                                p.numel() for p in self.seed.parameters() if p.requires_grad
                            ),
                        }
                    )
```

**Lines 555-560** - Replace:
```python
        if self.state:
            self.state.transition(SeedStage.CULLED)
            self._emit_telemetry(
                TelemetryEventType.SEED_CULLED,
                data={"reason": reason}
            )
```
With:
```python
        if self.state:
            # Capture metrics before transition clears state
            improvement = self.state.metrics.total_improvement
            blueprint_id = self.state.blueprint_id
            seed_id = self.state.seed_id

            self.state.transition(SeedStage.CULLED)
            self._emit_telemetry(
                TelemetryEventType.SEED_CULLED,
                data={
                    "reason": reason,
                    "blueprint_id": blueprint_id,
                    "seed_id": seed_id,
                    "improvement": improvement,
                }
            )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_kasmina_telemetry.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_kasmina_telemetry.py
git commit -m "feat(kasmina): enrich telemetry events with blueprint_id and improvement"
```

---

## Task 5: Wire Analytics into Vectorized Training

**Files:**
- Modify: `src/esper/simic/vectorized.py`

**Step 1: Add imports**

At top of file (after existing imports around line 39), add:

```python
from esper.leyline import TelemetryEvent
from esper.nissa import NissaHub, BlueprintAnalytics
```

**Step 2: Create analytics in train_ppo_vectorized**

After the agent creation block (around line 206), add:

```python
    # ==========================================================================
    # Blueprint Analytics Setup
    # ==========================================================================
    analytics = BlueprintAnalytics()
    hub = NissaHub()
    hub.add_backend(analytics)

    def make_telemetry_callback(env_idx: int):
        """Create callback that injects env_id before emitting to hub."""
        def callback(event: TelemetryEvent):
            event.data["env_id"] = env_idx
            hub.emit(event)
        return callback
```

**Step 3: Wire callback into create_env_state**

Modify `create_env_state` function (around line 211) to accept and wire the callback:

Find this block:
```python
    def create_env_state(env_idx: int, base_seed: int, trainloader, testloader) -> ParallelEnvState:
        """Create environment state with CUDA stream, reusing pre-created DataLoaders."""
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(env_device)
```

After `model = create_model(env_device)`, add:

```python
        # Wire telemetry callback with env_id injection
        model.seed_slot.on_telemetry = make_telemetry_callback(env_idx)
        model.seed_slot.fast_mode = False  # Enable telemetry

        # Set host_params baseline for scoreboard
        analytics._get_scoreboard(env_idx).host_params = sum(
            p.numel() for p in model.host.parameters() if p.requires_grad
        )
```

**Step 4: Add periodic summary printing**

Find the episode completion print block (search for `print(f"Episode {episode"` around line 540-550).

After the episode summary print, add:

```python
        # Print analytics summary every 5 episodes
        if (episode + 1) % 5 == 0 and len(analytics.stats) > 0:
            print()
            print(analytics.summary_table())
            for env_idx in range(n_envs):
                if env_idx in analytics.scoreboards:
                    print(analytics.scoreboard_table(env_idx))
            print()
```

**Step 5: Include analytics in return value**

Find the return statement at end of function. Change from:

```python
    return agent, history
```

To:

```python
    # Add analytics to final history entry
    if history:
        history[-1]["blueprint_analytics"] = analytics.snapshot()

    return agent, history
```

**Step 6: Verify manually**

Run: `PYTHONPATH=src .venv/bin/python -c "from esper.simic.vectorized import train_ppo_vectorized; print('Import OK')"`
Expected: `Import OK`

**Step 7: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): wire BlueprintAnalytics into vectorized training"
```

---

## Task 6: Integration Test

**Files:**
- Create: `tests/integration/test_blueprint_analytics.py`

**Step 1: Write integration test**

```python
"""Integration test for blueprint analytics in training loop."""

import pytest
import torch

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import NissaHub, BlueprintAnalytics
from esper.tolaria import create_model


class TestBlueprintAnalyticsIntegration:
    """Integration tests for analytics with real models."""

    def test_full_lifecycle_tracking(self):
        """Track a complete germinate -> fossilize cycle."""
        analytics = BlueprintAnalytics()
        hub = NissaHub()
        hub.add_backend(analytics)

        # Create model with telemetry callback
        model = create_model("cpu")

        def callback(event: TelemetryEvent):
            event.data["env_id"] = 0
            hub.emit(event)

        model.seed_slot.on_telemetry = callback
        model.seed_slot.fast_mode = False

        # Set host params baseline
        analytics._get_scoreboard(0).host_params = sum(
            p.numel() for p in model.host.parameters() if p.requires_grad
        )

        # Germinate
        model.germinate_seed("depthwise", "test_seed")

        assert analytics.stats["depthwise"].germinated == 1
        assert analytics.scoreboards[0].live_blueprint == "depthwise"

        # Simulate improvement and fossilize
        model.seed_slot.state.metrics.initial_val_accuracy = 70.0
        model.seed_slot.state.metrics.current_val_accuracy = 75.0

        # Force to BLENDING stage for fossilization
        from esper.leyline import SeedStage
        model.seed_slot.state.stage = SeedStage.BLENDING
        model.seed_slot.advance_stage(SeedStage.FOSSILIZED)

        assert analytics.stats["depthwise"].fossilized == 1
        assert analytics.stats["depthwise"].acc_deltas == [5.0]
        assert analytics.scoreboards[0].total_fossilized == 1
        assert analytics.scoreboards[0].params_added > 0

    def test_summary_tables_format(self):
        """Summary tables are readable and complete."""
        analytics = BlueprintAnalytics()

        # Add test data
        analytics.stats["depthwise"].germinated = 20
        analytics.stats["depthwise"].fossilized = 12
        analytics.stats["depthwise"].culled = 8
        analytics.stats["depthwise"].acc_deltas = [2.0] * 12

        analytics.stats["attention"].germinated = 10
        analytics.stats["attention"].fossilized = 1
        analytics.stats["attention"].culled = 9
        analytics.stats["attention"].churns = [-0.5] * 9

        analytics.scoreboards[0] = SeedScoreboard(
            total_fossilized=13,
            params_added=150000,
            host_params=1000000,
        )
        analytics.scoreboards[0].fossilized_by_blueprint["depthwise"] = 12
        analytics.scoreboards[0].fossilized_by_blueprint["attention"] = 1

        # Check summary table
        summary = analytics.summary_table()
        assert "depthwise" in summary
        assert "attention" in summary
        assert "60.0%" in summary  # depthwise rate

        # Check scoreboard
        scoreboard = analytics.scoreboard_table(0)
        assert "13" in scoreboard  # total fossilized
        assert "150.0K" in scoreboard  # params
        assert "15.0%" in scoreboard  # of host


from esper.nissa.analytics import SeedScoreboard
```

**Step 2: Run integration test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_blueprint_analytics.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/integration/test_blueprint_analytics.py
git commit -m "test: add integration tests for blueprint analytics"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Final commit if any cleanup needed**

---

## Summary

| Task | Files | Description |
|------|-------|-------------|
| 1 | `nissa/analytics.py`, `test_nissa_analytics.py` | BlueprintStats & SeedScoreboard dataclasses |
| 2 | `nissa/analytics.py`, `test_nissa_analytics.py` | BlueprintAnalytics OutputBackend |
| 3 | `nissa/__init__.py` | Export analytics module |
| 4 | `kasmina/slot.py`, `test_kasmina_telemetry.py` | Enrich telemetry events |
| 5 | `simic/vectorized.py` | Wire analytics into training |
| 6 | `test_blueprint_analytics.py` | Integration tests |
| 7 | - | Full test suite verification |

**Total estimated time:** 45-60 minutes

**Key verification command:**
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_nissa_analytics.py tests/test_kasmina_telemetry.py tests/integration/test_blueprint_analytics.py -v
```
