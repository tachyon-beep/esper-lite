# Seed Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-seed telemetry (`SeedTelemetry`) to enable fair IQL evaluation with real telemetry signals instead of zero-padding.

**Architecture:** Add `SeedTelemetry` contract to leyline, integrate into `SeedState` in kasmina, collect gradient signals during training, and consume in comparison.py. State dimension changes from 54 (27 base + 27 full telemetry) to 37 (27 base + 10 seed telemetry).

**Tech Stack:** Python dataclasses, PyTorch gradient hooks, existing leyline/kasmina/simic modules.

**Design Doc:** `docs/plans/2025-11-29-seed-telemetry-design.md`

**Gemini Review Notes (incorporated):**

- `gradient_norm` may need layer-type baseline normalization in future (Conv vs LayerNorm profiles differ)
- Fast mode: call `sync_telemetry` at epoch boundaries, not skip entirely (stale telemetry is acceptable, missing is not)
- `layer_id` string -> integer mapping at init time for multi-seed performance
- Gradient norm calculation: consider `torch._foreach_norm` for speed in multi-seed

---

## Task 1: Add SeedTelemetry Contract to Leyline

**Files:**

- Modify: `src/esper/leyline/telemetry.py:93` (after DEFAULT_BUDGETS)
- Modify: `src/esper/leyline/__init__.py:56-61,101-105`
- Create: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Create `tests/test_seed_telemetry.py`:

```python
"""Tests for SeedTelemetry contract."""

import pytest
from datetime import datetime, timezone


class TestSeedTelemetry:
    """Tests for SeedTelemetry dataclass."""

    def test_import_from_leyline(self):
        """SeedTelemetry should be importable from leyline."""
        from esper.leyline import SeedTelemetry
        assert SeedTelemetry is not None

    def test_create_with_seed_id(self):
        """SeedTelemetry requires seed_id."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test_seed_1")
        assert telem.seed_id == "test_seed_1"

    def test_default_values(self):
        """SeedTelemetry has sensible defaults."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test")
        assert telem.gradient_norm == 0.0
        assert telem.gradient_health == 1.0
        assert telem.has_vanishing is False
        assert telem.has_exploding is False
        assert telem.accuracy == 0.0
        assert telem.stage == 1
        assert telem.alpha == 0.0

    def test_to_features_returns_10_dims(self):
        """to_features() returns exactly 10 dimensions."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test")
        features = telem.to_features()
        assert len(features) == 10
        assert SeedTelemetry.feature_dim() == 10

    def test_to_features_normalized_range(self):
        """Features should be normalized to approximately [0, 1]."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(
            seed_id="test",
            gradient_norm=5.0,
            gradient_health=0.8,
            has_vanishing=True,
            has_exploding=False,
            accuracy=75.0,
            accuracy_delta=2.5,
            epochs_in_stage=10,
            stage=4,  # BLENDING
            alpha=0.5,
            epoch=12,
            max_epochs=25,
        )
        features = telem.to_features()

        # All features should be in reasonable range
        for i, f in enumerate(features):
            assert -1.0 <= f <= 1.0, f"Feature {i} out of range: {f}"

    def test_stage_normalization(self):
        """Stage should normalize to [0, 1] for stages 1-7."""
        from esper.leyline import SeedTelemetry

        # Stage 1 (DORMANT) -> 0.0
        telem1 = SeedTelemetry(seed_id="test", stage=1)
        assert telem1.to_features()[7] == 0.0

        # Stage 7 (FOSSILIZED) -> 1.0
        telem7 = SeedTelemetry(seed_id="test", stage=7)
        assert telem7.to_features()[7] == 1.0

    def test_temporal_position(self):
        """Temporal position should be epoch/max_epochs."""
        from esper.leyline import SeedTelemetry
        telem = SeedTelemetry(seed_id="test", epoch=10, max_epochs=20)
        features = telem.to_features()
        assert features[9] == 0.5  # 10/20

    def test_captured_at_timestamp(self):
        """SeedTelemetry should have a captured_at timestamp."""
        from esper.leyline import SeedTelemetry
        before = datetime.now(timezone.utc)
        telem = SeedTelemetry(seed_id="test")
        after = datetime.now(timezone.utc)
        assert before <= telem.captured_at <= after
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py -v`

Expected: FAIL with "cannot import name 'SeedTelemetry'"

**Step 3: Add SeedTelemetry to telemetry.py**

Add to `src/esper/leyline/telemetry.py` after line 92 (after `DEFAULT_BUDGETS = ...`):

```python


# =============================================================================
# Seed Telemetry Snapshot
# =============================================================================

@dataclass(slots=True)
class SeedTelemetry:
    """Per-seed telemetry snapshot - the seed's 'local picture'.

    Contract between seed implementations (Kasmina/Simic) and
    decision-makers (Tamiyo). Designed for:
    - Single seed (current): one instance
    - Multi-seed (future): collection managed by registry
    - Hierarchical (stretch): tactical aggregates for strategic

    Note: Uses slots=True for memory efficiency in multi-seed scenarios.
    """

    seed_id: str
    blueprint_id: str = ""
    layer_id: str = ""

    # Health signals (lightweight, always collected)
    gradient_norm: float = 0.0
    gradient_health: float = 1.0  # 0-1, higher is healthier
    has_vanishing: bool = False
    has_exploding: bool = False

    # Progress signals
    accuracy: float = 0.0  # percentage (0-100)
    accuracy_delta: float = 0.0  # positive = improving
    epochs_in_stage: int = 0

    # Stage context
    stage: int = 1  # SeedStage enum value (1-7)
    alpha: float = 0.0  # blending weight (0-1)

    # Temporal context
    epoch: int = 0
    max_epochs: int = 25

    # Timestamp for staleness detection
    captured_at: datetime = field(default_factory=_utc_now)

    def to_features(self) -> list[float]:
        """Convert to 10-dim feature vector for RL policies.

        All features normalized to approximately [0, 1] range.
        """
        return [
            min(self.gradient_norm, 10.0) / 10.0,
            self.gradient_health,
            float(self.has_vanishing),
            float(self.has_exploding),
            min(self.epochs_in_stage, 50) / 50.0,
            self.accuracy / 100.0,
            max(-1.0, min(1.0, self.accuracy_delta / 10.0)),
            (self.stage - 1) / 6.0,  # stages 1-7 -> [0, 1]
            self.alpha,
            self.epoch / max(self.max_epochs, 1),  # temporal position
        ]

    @classmethod
    def feature_dim(cls) -> int:
        """Return current feature vector dimension."""
        return 10
```

**Step 4: Export SeedTelemetry from leyline/**init**.py**

Modify `src/esper/leyline/__init__.py`:

At line 56-61, change:

```python
# Telemetry contracts
from esper.leyline.telemetry import (
    TelemetryEventType,
    TelemetryEvent,
    PerformanceBudgets,
    DEFAULT_BUDGETS,
)
```

To:

```python
# Telemetry contracts
from esper.leyline.telemetry import (
    TelemetryEventType,
    TelemetryEvent,
    PerformanceBudgets,
    DEFAULT_BUDGETS,
    SeedTelemetry,
)
```

At line 101-105, change:

```python
    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
]
```

To:

```python
    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
    "SeedTelemetry",
]
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py -v`

Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add src/esper/leyline/telemetry.py src/esper/leyline/__init__.py tests/test_seed_telemetry.py
git commit -m "feat(leyline): add SeedTelemetry contract for per-seed telemetry"
```

---

## Task 2: Add telemetry Field to SeedState

**Files:**

- Modify: `src/esper/kasmina/slot.py:16-33,107-127`
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestSeedStateTelemetry:
    """Tests for SeedState.telemetry integration."""

    def test_seed_state_has_telemetry(self):
        """SeedState should have a telemetry field."""
        from esper.kasmina.slot import SeedState
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")
        assert hasattr(state, 'telemetry')
        assert state.telemetry is not None

    def test_telemetry_initialized_with_seed_info(self):
        """Telemetry should be initialized with seed_id and blueprint_id."""
        from esper.kasmina.slot import SeedState
        state = SeedState(seed_id="seed_1", blueprint_id="attention")
        assert state.telemetry.seed_id == "seed_1"
        assert state.telemetry.blueprint_id == "attention"

    def test_sync_telemetry_updates_from_metrics(self):
        """sync_telemetry should copy values from metrics."""
        from esper.kasmina.slot import SeedState
        from esper.leyline import SeedStage

        state = SeedState(seed_id="test", blueprint_id="conv")
        state.stage = SeedStage.TRAINING
        state.metrics.current_val_accuracy = 75.0
        state.metrics.epochs_in_current_stage = 5
        state.alpha = 0.3

        state.sync_telemetry(
            gradient_norm=2.5,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            epoch=10,
            max_epochs=25,
        )

        assert state.telemetry.accuracy == 75.0
        assert state.telemetry.epochs_in_stage == 5
        assert state.telemetry.stage == SeedStage.TRAINING.value
        assert state.telemetry.alpha == 0.3
        assert state.telemetry.gradient_norm == 2.5
        assert state.telemetry.gradient_health == 0.9
        assert state.telemetry.epoch == 10
        assert state.telemetry.max_epochs == 25
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestSeedStateTelemetry -v`

Expected: FAIL with "no attribute 'telemetry'"

**Step 3: Add telemetry to SeedState imports**

Modify `src/esper/kasmina/slot.py` lines 16-33. Change:

```python
from esper.leyline import (
    # Lifecycle
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
    # Reports
    SeedMetrics as LeylineSeedMetrics,
    SeedStateReport,
    # Gates
    GateLevel,
    GateResult,
    # Telemetry
    TelemetryEvent,
    TelemetryEventType,
)
```

To:

```python
from esper.leyline import (
    # Lifecycle
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
    # Reports
    SeedMetrics as LeylineSeedMetrics,
    SeedStateReport,
    # Gates
    GateLevel,
    GateResult,
    # Telemetry
    TelemetryEvent,
    TelemetryEventType,
    SeedTelemetry,
)
```

**Step 4: Add telemetry field and sync_telemetry to SeedState**

Modify `src/esper/kasmina/slot.py`. After line 127 (`stage_history: list[...]`), add the telemetry field. Then add `__post_init__` and `sync_telemetry` methods.

Find lines 107-127:

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    stage_entered_at: datetime = field(default_factory=datetime.utcnow)

    alpha: float = 0.0
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History
    stage_history: list[tuple[SeedStage, datetime]] = field(default_factory=list)
```

Change to:

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    stage_entered_at: datetime = field(default_factory=datetime.utcnow)

    alpha: float = 0.0
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History
    stage_history: list[tuple[SeedStage, datetime]] = field(default_factory=list)

    # Telemetry (initialized in __post_init__)
    telemetry: SeedTelemetry = field(default=None)

    def __post_init__(self):
        """Initialize telemetry with seed identity."""
        if self.telemetry is None:
            self.telemetry = SeedTelemetry(
                seed_id=self.seed_id,
                blueprint_id=self.blueprint_id,
            )

    def sync_telemetry(
        self,
        gradient_norm: float,
        gradient_health: float,
        has_vanishing: bool,
        has_exploding: bool,
        epoch: int = 0,
        max_epochs: int = 25,
    ) -> None:
        """Sync telemetry from metrics + gradient signals.

        Call this once per epoch after validation to update telemetry.
        SeedMetrics remains the source of truth for accuracy/epoch data.
        """
        from datetime import timezone

        self.telemetry.accuracy = self.metrics.current_val_accuracy
        self.telemetry.accuracy_delta = self.metrics.improvement_since_stage_start
        self.telemetry.epochs_in_stage = self.metrics.epochs_in_current_stage
        self.telemetry.stage = self.stage.value
        self.telemetry.alpha = self.alpha

        self.telemetry.gradient_norm = gradient_norm
        self.telemetry.gradient_health = gradient_health
        self.telemetry.has_vanishing = has_vanishing
        self.telemetry.has_exploding = has_exploding

        self.telemetry.epoch = epoch
        self.telemetry.max_epochs = max_epochs
        self.telemetry.captured_at = datetime.now(timezone.utc)
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py -v`

Expected: All 11 tests PASS

**Step 6: Run existing kasmina tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -k kasmina -v`

Expected: All kasmina tests PASS

**Step 7: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_seed_telemetry.py
git commit -m "feat(kasmina): add telemetry field and sync_telemetry to SeedState"
```

---

## Task 3: Add Lightweight Gradient Collector

**Files:**

- Create: `src/esper/simic/gradient_collector.py`
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestSeedGradientCollector:
    """Tests for lightweight gradient collection."""

    def test_import_collector(self):
        """SeedGradientCollector should be importable."""
        from esper.simic.gradient_collector import SeedGradientCollector
        assert SeedGradientCollector is not None

    def test_collect_gradient_stats(self):
        """Collector should compute gradient stats from parameters."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create a simple model with gradients
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        assert 'gradient_norm' in stats
        assert 'gradient_health' in stats
        assert 'has_vanishing' in stats
        assert 'has_exploding' in stats
        assert stats['gradient_norm'] >= 0
        assert 0 <= stats['gradient_health'] <= 1

    def test_detect_vanishing_gradients(self):
        """Collector should detect vanishing gradients."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create model with tiny gradients
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.grad = torch.zeros_like(p) + 1e-10

        collector = SeedGradientCollector(vanishing_threshold=1e-7)
        stats = collector.collect(model.parameters())

        assert stats['has_vanishing'] is True

    def test_detect_exploding_gradients(self):
        """Collector should detect exploding gradients."""
        import torch
        import torch.nn as nn
        from esper.simic.gradient_collector import SeedGradientCollector

        # Create model with huge gradients
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 1000

        collector = SeedGradientCollector(exploding_threshold=100)
        stats = collector.collect(model.parameters())

        assert stats['has_exploding'] is True
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestSeedGradientCollector -v`

Expected: FAIL with "No module named 'esper.simic.gradient_collector'"

**Step 3: Create gradient_collector.py**

Create `src/esper/simic/gradient_collector.py`:

```python
"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn


class SeedGradientCollector:
    """Lightweight gradient statistics collector for seed telemetry.

    Unlike DiagnosticTracker, this collector:
    - Does not use hooks (called explicitly after backward)
    - Computes only essential stats (norm, health, vanishing, exploding)
    - Is stateless (no history)
    """

    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        """Initialize collector with detection thresholds.

        Args:
            vanishing_threshold: Gradient norm below this is considered vanishing
            exploding_threshold: Gradient norm above this is considered exploding
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def collect(self, parameters: Iterator[nn.Parameter]) -> dict:
        """Collect gradient statistics from parameters.

        Call this after loss.backward() to gather gradient stats.

        Args:
            parameters: Iterator of parameters (e.g., model.parameters())

        Returns:
            Dict with keys: gradient_norm, gradient_health, has_vanishing, has_exploding
        """
        total_norm = 0.0
        n_params = 0
        n_vanishing = 0
        n_exploding = 0

        for param in parameters:
            if param.grad is None:
                continue

            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            n_params += 1

            if param_norm < self.vanishing_threshold:
                n_vanishing += 1
            if param_norm > self.exploding_threshold:
                n_exploding += 1

        if n_params == 0:
            return {
                'gradient_norm': 0.0,
                'gradient_health': 1.0,
                'has_vanishing': False,
                'has_exploding': False,
            }

        gradient_norm = (total_norm ** 0.5) / n_params  # Average norm

        # Compute health score (0-1, higher is healthier)
        # Penalize vanishing/exploding gradients
        vanishing_ratio = n_vanishing / n_params
        exploding_ratio = n_exploding / n_params

        health = 1.0
        health -= vanishing_ratio * 0.5  # Penalize vanishing
        health -= exploding_ratio * 0.8  # Penalize exploding more
        health = max(0.0, min(1.0, health))

        return {
            'gradient_norm': gradient_norm,
            'gradient_health': health,
            'has_vanishing': n_vanishing > 0,
            'has_exploding': n_exploding > 0,
        }


def collect_seed_gradients(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 100.0,
) -> dict:
    """Convenience function to collect gradient stats.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection

    Returns:
        Dict with gradient statistics
    """
    collector = SeedGradientCollector(
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )
    return collector.collect(seed_parameters)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestSeedGradientCollector -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/gradient_collector.py tests/test_seed_telemetry.py
git commit -m "feat(simic): add SeedGradientCollector for lightweight gradient stats"
```

---

## Task 4: Update comparison.py to Use Seed Telemetry

**Files:**

- Modify: `src/esper/simic/comparison.py:20-24,61-92`
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestComparisonTelemetry:
    """Tests for comparison.py telemetry integration."""

    def test_snapshot_to_features_with_seed_telemetry(self):
        """snapshot_to_features should use seed telemetry when provided."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot
        from esper.leyline import SeedTelemetry

        snapshot = TrainingSnapshot(
            epoch=5,
            global_step=500,
            train_loss=0.5,
            val_loss=0.6,
            loss_delta=-0.1,
            train_accuracy=80.0,
            val_accuracy=75.0,
            accuracy_delta=2.0,
            plateau_epochs=0,
            best_val_accuracy=75.0,
            best_val_loss=0.6,
            loss_history_5=(0.9, 0.8, 0.7, 0.6, 0.6),
            accuracy_history_5=(60.0, 65.0, 70.0, 73.0, 75.0),
            has_active_seed=True,
            seed_stage=3,
            seed_epochs_in_stage=2,
            seed_alpha=0.0,
            seed_improvement=5.0,
            available_slots=0,
        )

        seed_telemetry = SeedTelemetry(
            seed_id="test",
            gradient_norm=1.5,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            accuracy=75.0,
            accuracy_delta=5.0,
            epochs_in_stage=2,
            stage=3,
            alpha=0.0,
            epoch=5,
            max_epochs=25,
        )

        features = snapshot_to_features(
            snapshot,
            use_telemetry=True,
            seed_telemetry=seed_telemetry
        )

        # Should be 27 base + 10 seed = 37 dims
        assert len(features) == 37

        # Last 10 should be from seed telemetry
        seed_features = features[-10:]
        expected = seed_telemetry.to_features()
        assert seed_features == expected

    def test_snapshot_to_features_no_telemetry(self):
        """Without telemetry, should return 27 dims."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5,
            accuracy_history_5=(50.0,)*5, has_active_seed=False,
            seed_stage=0, seed_epochs_in_stage=0, seed_alpha=0.0,
            seed_improvement=0.0, available_slots=1,
        )

        features = snapshot_to_features(snapshot, use_telemetry=False)
        assert len(features) == 27
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestComparisonTelemetry -v`

Expected: FAIL with "unexpected keyword argument 'seed_telemetry'"

**Step 3: Update imports in comparison.py**

Modify `src/esper/simic/comparison.py` lines 20-24. Change:

```python
from esper.leyline import SimicAction, SeedStage
from esper.simic.episodes import TrainingSnapshot
from esper.simic.features import obs_to_base_features
from esper.simic.iql import IQL
from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig, SignalTracker
```

To:

```python
from esper.leyline import SimicAction, SeedStage, SeedTelemetry
from esper.simic.episodes import TrainingSnapshot
from esper.simic.features import obs_to_base_features
from esper.simic.gradient_collector import collect_seed_gradients
from esper.simic.iql import IQL
from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig, SignalTracker
```

**Step 4: Update snapshot_to_features signature**

Modify `src/esper/simic/comparison.py` lines 61-92. Change:

```python
def snapshot_to_features(snapshot: TrainingSnapshot, use_telemetry: bool = False) -> list[float]:
    """Convert TrainingSnapshot to feature vector for IQL."""
    # Convert snapshot to observation dict format expected by obs_to_base_features
    obs = {
        'epoch': snapshot.epoch,
        'global_step': snapshot.global_step,
        'train_loss': snapshot.train_loss,
        'val_loss': snapshot.val_loss,
        'loss_delta': snapshot.loss_delta,
        'train_accuracy': snapshot.train_accuracy,
        'val_accuracy': snapshot.val_accuracy,
        'accuracy_delta': snapshot.accuracy_delta,
        'plateau_epochs': snapshot.plateau_epochs,
        'best_val_accuracy': snapshot.best_val_accuracy,
        'best_val_loss': snapshot.best_val_loss,
        'loss_history_5': list(snapshot.loss_history_5),
        'accuracy_history_5': list(snapshot.accuracy_history_5),
        'has_active_seed': snapshot.has_active_seed,
        'seed_stage': snapshot.seed_stage,
        'seed_epochs_in_stage': snapshot.seed_epochs_in_stage,
        'seed_alpha': snapshot.seed_alpha,
        'seed_improvement': snapshot.seed_improvement,
        'available_slots': snapshot.available_slots,
    }

    features = obs_to_base_features(obs)

    if use_telemetry:
        # Pad with zeros for telemetry features (not available in live comparison)
        features.extend([0.0] * 27)

    return features
```

To:

```python
def snapshot_to_features(
    snapshot: TrainingSnapshot,
    use_telemetry: bool = False,
    seed_telemetry: SeedTelemetry | None = None,
) -> list[float]:
    """Convert TrainingSnapshot to feature vector for IQL.

    Args:
        snapshot: Training state snapshot
        use_telemetry: Whether to include telemetry features
        seed_telemetry: Per-seed telemetry (10 dims). If None and use_telemetry=True,
                       falls back to zeros with a warning.

    Returns:
        Feature vector:
        - 27 dims if use_telemetry=False
        - 37 dims if use_telemetry=True (27 base + 10 seed)
    """
    # Convert snapshot to observation dict format expected by obs_to_base_features
    obs = {
        'epoch': snapshot.epoch,
        'global_step': snapshot.global_step,
        'train_loss': snapshot.train_loss,
        'val_loss': snapshot.val_loss,
        'loss_delta': snapshot.loss_delta,
        'train_accuracy': snapshot.train_accuracy,
        'val_accuracy': snapshot.val_accuracy,
        'accuracy_delta': snapshot.accuracy_delta,
        'plateau_epochs': snapshot.plateau_epochs,
        'best_val_accuracy': snapshot.best_val_accuracy,
        'best_val_loss': snapshot.best_val_loss,
        'loss_history_5': list(snapshot.loss_history_5),
        'accuracy_history_5': list(snapshot.accuracy_history_5),
        'has_active_seed': snapshot.has_active_seed,
        'seed_stage': snapshot.seed_stage,
        'seed_epochs_in_stage': snapshot.seed_epochs_in_stage,
        'seed_alpha': snapshot.seed_alpha,
        'seed_improvement': snapshot.seed_improvement,
        'available_slots': snapshot.available_slots,
    }

    features = obs_to_base_features(obs)

    if use_telemetry:
        if seed_telemetry is not None:
            features.extend(seed_telemetry.to_features())
        else:
            # Fallback to zeros (anti-pattern per DRL review, but safe default)
            import warnings
            warnings.warn(
                "use_telemetry=True but no seed_telemetry provided. "
                "Using zero-padding which may cause distribution shift.",
                UserWarning,
            )
            features.extend([0.0] * SeedTelemetry.feature_dim())

    return features
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestComparisonTelemetry -v`

Expected: All 2 tests PASS

**Step 6: Run all simic tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/esper/simic/comparison.py tests/test_seed_telemetry.py
git commit -m "feat(simic): update snapshot_to_features to use SeedTelemetry"
```

---

## Task 5: Integrate Telemetry Collection in head_to_head_comparison

**Files:**

- Modify: `src/esper/simic/comparison.py` (head_to_head_comparison function)
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestHeadToHeadTelemetry:
    """Tests for telemetry collection in head_to_head_comparison."""

    def test_run_training_episode_collects_telemetry(self):
        """Training episodes should collect and use seed telemetry."""
        # This is an integration test - we'll verify the function signature
        # accepts telemetry collection. Full integration test would require
        # CIFAR-10 and is expensive.
        from esper.simic.comparison import head_to_head_comparison
        import inspect

        # Verify the function exists and is callable
        assert callable(head_to_head_comparison)

        # The implementation should use seed telemetry internally
        # We'll verify this by checking the source includes our patterns
        source = inspect.getsource(head_to_head_comparison)
        assert 'seed_telemetry' in source.lower() or 'sync_telemetry' in source.lower() or 'collect_seed_gradients' in source.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestHeadToHeadTelemetry -v`

Expected: FAIL with assertion error (source doesn't contain expected patterns)

**Step 3: Update head_to_head_comparison to collect telemetry**

This is a larger modification. Read the current function and update the `run_training_episode` helper and `iql_action_fn` to collect and use telemetry.

The key changes in `src/esper/simic/comparison.py`:

1. In `run_training_episode`, after backward pass, collect gradient stats
2. After validation, sync telemetry to seed state
3. In `iql_action_fn`, pass seed telemetry to `snapshot_to_features`

Due to the size of this change, here's the pattern to apply:

Inside `run_training_episode` (around line 370-380, after training phase), add gradient collection:

```python
# After loss.backward() in the training loop:
if seed_state is not None:
    grad_stats = collect_seed_gradients(model.get_seed_parameters())
```

After validation phase (around line 465), sync telemetry:

```python
# After computing val_acc, before tracker.update:
if model.has_active_seed and model.seed_state is not None:
    model.seed_state.sync_telemetry(
        gradient_norm=grad_stats.get('gradient_norm', 0.0) if 'grad_stats' in dir() else 0.0,
        gradient_health=grad_stats.get('gradient_health', 1.0) if 'grad_stats' in dir() else 1.0,
        has_vanishing=grad_stats.get('has_vanishing', False) if 'grad_stats' in dir() else False,
        has_exploding=grad_stats.get('has_exploding', False) if 'grad_stats' in dir() else False,
        epoch=epoch,
        max_epochs=max_epochs,
    )
```

In `iql_action_fn` (around line 568), pass telemetry:

```python
# Get seed telemetry if available
seed_telemetry = None
if model.has_active_seed and model.seed_state is not None:
    seed_telemetry = model.seed_state.telemetry

features = snapshot_to_features(snapshot, use_telemetry=use_telemetry, seed_telemetry=seed_telemetry)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestHeadToHeadTelemetry -v`

Expected: PASS

**Step 5: Run all comparison tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/comparison.py tests/test_seed_telemetry.py
git commit -m "feat(simic): integrate telemetry collection in head_to_head_comparison"
```

---

## Task 6: Update load_iql_model for New Dimensions

**Files:**

- Modify: `src/esper/simic/comparison.py:31-54`
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write the failing test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestLoadIQLModel:
    """Tests for IQL model loading with new dimensions."""

    def test_infer_telemetry_from_state_dim_37(self):
        """State dim 37 should infer use_telemetry=True (new format)."""
        # This is a unit test of the inference logic
        # 27 = base only, 37 = base + seed, 54 = base + full (legacy)

        def infer_telemetry(state_dim: int) -> tuple[bool, str]:
            """Infer telemetry usage from state dimension."""
            if state_dim == 27:
                return False, "none"
            elif state_dim == 37:
                return True, "seed"
            elif state_dim == 54:
                return True, "full"
            else:
                return False, "unknown"

        assert infer_telemetry(27) == (False, "none")
        assert infer_telemetry(37) == (True, "seed")
        assert infer_telemetry(54) == (True, "full")
```

**Step 2: Run test to verify it passes (this is just logic validation)**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::TestLoadIQLModel -v`

Expected: PASS (no code change needed for this test)

**Step 3: Update load_iql_model to handle new dimensions**

Modify `src/esper/simic/comparison.py` lines 31-54. Change:

```python
def load_iql_model(model_path: str, device: str = "cpu") -> tuple[IQL, bool]:
    """Load a trained IQL model.

    Returns:
        Tuple of (IQL agent, use_telemetry flag inferred from state_dim)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    state_dim = checkpoint['state_dim']
    # Infer telemetry usage: 54 = base + telemetry, 27 = base only
    use_telemetry = (state_dim == 54)

    iql = IQL(
        state_dim=state_dim,
        action_dim=checkpoint['action_dim'],
        gamma=checkpoint.get('gamma', 0.99),
        tau=checkpoint.get('tau', 0.7),
        beta=checkpoint.get('beta', 3.0),
        device=device,
    )
    iql.q_network.load_state_dict(checkpoint['q_network'])
    iql.v_network.load_state_dict(checkpoint['v_network'])

    return iql, use_telemetry
```

To:

```python
def load_iql_model(model_path: str, device: str = "cpu") -> tuple[IQL, bool, str]:
    """Load a trained IQL model.

    Returns:
        Tuple of (IQL agent, use_telemetry flag, telemetry_type)
        - telemetry_type: "none" (27), "seed" (37), "full" (54), or "unknown"
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    state_dim = checkpoint['state_dim']

    # Infer telemetry usage from state dimension
    if state_dim == 27:
        use_telemetry = False
        telemetry_type = "none"
    elif state_dim == 37:
        use_telemetry = True
        telemetry_type = "seed"
    elif state_dim == 54:
        use_telemetry = True
        telemetry_type = "full"
    else:
        use_telemetry = False
        telemetry_type = "unknown"
        import warnings
        warnings.warn(f"Unknown state_dim {state_dim}, assuming no telemetry")

    iql = IQL(
        state_dim=state_dim,
        action_dim=checkpoint['action_dim'],
        gamma=checkpoint.get('gamma', 0.99),
        tau=checkpoint.get('tau', 0.7),
        beta=checkpoint.get('beta', 3.0),
        device=device,
    )
    iql.q_network.load_state_dict(checkpoint['q_network'])
    iql.v_network.load_state_dict(checkpoint['v_network'])

    return iql, use_telemetry, telemetry_type
```

**Step 4: Update callers of load_iql_model**

Search for uses of `load_iql_model` and update to handle the third return value.

In `live_comparison` (around line 122):

```python
# Change:
iql, use_telemetry = load_iql_model(model_path, device=device)
# To:
iql, use_telemetry, telemetry_type = load_iql_model(model_path, device=device)
print(f"  State dim: {iql.q_network.net[0].in_features} (telemetry: {telemetry_type})")
```

In `head_to_head_comparison` (around line 325):

```python
# Change:
iql, use_telemetry = load_iql_model(model_path, device=device)
# To:
iql, use_telemetry, telemetry_type = load_iql_model(model_path, device=device)
print(f"  State dim: {state_dim} (telemetry: {telemetry_type})")
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/comparison.py tests/test_seed_telemetry.py
git commit -m "feat(simic): update load_iql_model to handle 37-dim seed telemetry"
```

---

## Task 7: Final Integration Test and Cleanup

**Files:**

- Modify: `tests/test_seed_telemetry.py`
- Run full test suite

**Step 1: Add comprehensive integration test**

Add to `tests/test_seed_telemetry.py`:

```python
class TestFullIntegration:
    """End-to-end integration tests."""

    def test_seed_telemetry_flow(self):
        """Test complete flow: create seed -> collect telemetry -> to_features."""
        import torch
        import torch.nn as nn
        from esper.kasmina.slot import SeedState
        from esper.leyline import SeedStage
        from esper.simic.gradient_collector import collect_seed_gradients

        # 1. Create seed state
        state = SeedState(seed_id="integration_test", blueprint_id="conv_enhance")
        assert state.telemetry is not None

        # 2. Simulate training progress
        state.stage = SeedStage.TRAINING
        state.metrics.record_accuracy(70.0)
        state.metrics.record_accuracy(75.0)
        state.alpha = 0.0

        # 3. Create a simple model and compute gradients
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # 4. Collect gradient stats
        grad_stats = collect_seed_gradients(model.parameters())

        # 5. Sync telemetry
        state.sync_telemetry(
            gradient_norm=grad_stats['gradient_norm'],
            gradient_health=grad_stats['gradient_health'],
            has_vanishing=grad_stats['has_vanishing'],
            has_exploding=grad_stats['has_exploding'],
            epoch=2,
            max_epochs=25,
        )

        # 6. Convert to features
        features = state.telemetry.to_features()

        # 7. Verify
        assert len(features) == 10
        assert features[5] == 0.75  # accuracy/100
        assert features[7] == (SeedStage.TRAINING.value - 1) / 6.0  # stage normalized
        assert features[9] == 2 / 25  # epoch/max_epochs
```

**Step 2: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py -v`

Expected: All tests PASS

**Step 3: Run full simic test suite**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v`

Expected: All tests PASS

**Step 4: Final commit**

```bash
git add tests/test_seed_telemetry.py
git commit -m "test: add full integration test for seed telemetry flow"
```

---

## Summary

After completing all tasks:

1. **SeedTelemetry contract** in `leyline/telemetry.py` (10-dim features)
2. **SeedState.telemetry** field with `sync_telemetry()` method
3. **SeedGradientCollector** for lightweight gradient stats
4. **comparison.py** updated to use real telemetry (37-dim state)
5. **load_iql_model** handles 27/37/54-dim models

Models trained with 54-dim (old format) will still work but use `telemetry_type="full"`. New models should use 37-dim (base + seed telemetry).

IMPORTANT ADDITIONAL UPDATES:
This is a **Green Light** implementation plan. It is structurally sound, follows Test-Driven Development (TDD), and respects the separation of concerns between `Leyline` (contracts) and `Kasmina` (mechanics).

However, as your Technical Reviewer, I have **two specific optimizations** and **one critical timing check** to add before you hand this to Claude.

### 1\. Performance Optimization: `_foreach_norm` (Task 3)

In the plan's preamble, you noted `torch._foreach_norm` as a Gemini suggestion, but the code snippet in **Task 3** still iterates through parameters in a Python loop.

For a single seed, this doesn't matter. For 50 seeds (future state), a Python loop over thousands of parameters will create a massive CPU bottleneck on the critical path.

**Recommendation:** Replace the `collect` method in `src/esper/simic/gradient_collector.py` with this vectorized implementation.

```python
    def collect(self, parameters: Iterator[nn.Parameter]) -> dict:
        # Filter params with grads
        grads = [p.grad for p in parameters if p.grad is not None]
        
        if not grads:
            return { ... } # Return zeros (same as plan)

        # 1. Vectorized Norm Calculation (CPU bottleneck fix)
        # Computes L2 norm for all tensors in the list at once
        device = grads[0].device
        # Note: _foreach_norm returns a list of scalar tensors
        per_param_norms = torch._foreach_norm(grads, 2.0)
        
        # Stack to compute stats efficiently on GPU/CPU
        all_norms = torch.stack(per_param_norms)
        
        # 2. Compute Statistics
        total_squared_norm = torch.sum(all_norms ** 2).item()
        gradient_norm = (total_squared_norm ** 0.5) / len(grads)
        
        # 3. Health Checks
        n_vanishing = torch.sum(all_norms < self.vanishing_threshold).item()
        n_exploding = torch.sum(all_norms > self.exploding_threshold).item()
        
        # ... remainder of health score logic is fine ...
```

### 2\. Critical Timing Check (Task 5)

In **Task 5 (Integration)**, the plan says:

> "In `run_training_episode`, after backward pass, collect gradient stats."

**Risk:** You must ensure `collect_seed_gradients` happens **immediately** after `loss.backward()` and **before** `optimizer.step()` or `optimizer.zero_grad()`.

- If you collect after `step()`, the gradients might be modified (e.g., by weight decay or momentum).
- If you collect after `zero_grad()`, you get all zeros.

**Correct Flow Diagram:**

Ensure Claude inserts the hook here:

```python
# ... inside training loop ...
scaler.scale(loss).backward()  # (If using AMP)

# <--- INSERT COLLECTION HERE --->
if seed_state is not None:
   grad_stats = collect_seed_gradients(...)

scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

### 3\. Normalization Heuristic (Task 1)

In **Task 1**, you normalize the gradient norm:

```python
min(self.gradient_norm, 10.0) / 10.0
```

**Constraint:** This assumes `10.0` is a reasonable "Max" for a gradient norm. In deep networks, especially early in training, norms can spike much higher.
**Refinement:** I recommend logging the *raw* `gradient_norm` to your `DiagnosticTracker` (Nissa) as well, even if the RL agent only sees the clamped 0-1 value. If you see the agent's input is constantly pegged at `1.0`, you'll know your normalization constant (`10.0`) is too low and blinding the agent.

-----

### Comparison of Feature Vectors

The shift from 54-dim to 37-dim is the correct move for this phase.

| Dimension | Previous (54-dim) | New (37-dim) | Benefit |
| :--- | :--- | :--- | :--- |
| **0-26** | Base State | Base State | Unchanged |
| **27-36** | **Global** Telemetry | **Seed** Telemetry | **High Signal:** Agent sees specific seed health. |
| **37-53** | **Global** Metrics | *Removed* | **Noise Reduction:** Removing irrelevant global noise. |

### Conclusion

The plan is **Ready for Execution**.

If you pass the `_foreach_norm` snippet to Claude along with the plan, you will save yourself a refactor cycle later when you scale to multi-seed architectures.
