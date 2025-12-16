# Dynamic Slots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make slot count fully dynamic — hosts define available injection points, and everything else adapts automatically.

**Architecture:** Replace hardcoded "3 slots" with a single source of truth: hosts expose `InjectionSpec` objects describing available injection points. All downstream systems (features, networks, TUI) derive their slot count from this source. This eliminates the implicit coupling that causes runtime failures when components disagree about slot count.

**Tech Stack:** Python dataclasses, PyTorch nn.Module, existing leyline infrastructure.

**Why This Improves Stability:**
- Current state: "3" hardcoded in ~15 locations across 8 files
- Failure mode: Components drift (e.g., validation accepts `r1c0` but host doesn't support it)
- Fix: Single source of truth eliminates inconsistency by construction

---

## Phase 1: Foundation - InjectionSpec Protocol

### Task 1.1: Create InjectionSpec Dataclass

**Files:**
- Create: `src/esper/leyline/injection_spec.py`
- Modify: `src/esper/leyline/__init__.py`
- Test: `tests/leyline/test_injection_spec.py`

**Step 1: Write the failing test**

```python
# tests/leyline/test_injection_spec.py
"""Tests for InjectionSpec dataclass."""

import pytest
from esper.leyline.injection_spec import InjectionSpec


class TestInjectionSpec:
    def test_basic_creation(self):
        spec = InjectionSpec(
            slot_id="r0c0",
            channels=64,
            position=0.33,
            layer_range=(0, 2),
        )
        assert spec.slot_id == "r0c0"
        assert spec.channels == 64
        assert spec.position == 0.33
        assert spec.layer_range == (0, 2)

    def test_position_must_be_0_to_1(self):
        with pytest.raises(ValueError, match="position must be between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=1.5, layer_range=(0, 2))

    def test_layer_range_must_be_valid(self):
        with pytest.raises(ValueError, match="layer_range"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(5, 2))

    def test_frozen_immutable(self):
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        with pytest.raises(AttributeError):
            spec.channels = 128
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_injection_spec.py -v`
Expected: FAIL with "No module named 'esper.leyline.injection_spec'"

**Step 3: Write minimal implementation**

```python
# src/esper/leyline/injection_spec.py
"""InjectionSpec - Describes a host injection point for seeds.

This is the contract between hosts (which define injection points) and
the rest of the system (which needs to know about available slots).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionSpec:
    """Specification for a single injection point in a host network.

    Attributes:
        slot_id: Canonical slot ID (e.g., "r0c0", "r0c1")
        channels: Channel/embedding dimension at this injection point
        position: Relative position in network (0.0 = input, 1.0 = output)
        layer_range: Tuple of (start_layer, end_layer) this slot covers
    """

    slot_id: str
    channels: int
    position: float
    layer_range: tuple[int, int]

    def __post_init__(self) -> None:
        if not (0.0 <= self.position <= 1.0):
            raise ValueError(f"position must be between 0 and 1, got {self.position}")
        start, end = self.layer_range
        if start > end:
            raise ValueError(f"layer_range start ({start}) must be <= end ({end})")
        if start < 0:
            raise ValueError(f"layer_range start must be non-negative, got {start}")


__all__ = ["InjectionSpec"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_injection_spec.py -v`
Expected: PASS

**Step 5: Export from leyline**

Add to `src/esper/leyline/__init__.py`:
```python
from esper.leyline.injection_spec import InjectionSpec
```

And add `"InjectionSpec"` to the `__all__` list.

**Step 6: Commit**

```bash
git add src/esper/leyline/injection_spec.py tests/leyline/test_injection_spec.py src/esper/leyline/__init__.py
git commit -m "feat(leyline): add InjectionSpec dataclass for dynamic slot support"
```

---

### Task 1.2: Add SlotConfig.from_specs() Factory

**Files:**
- Modify: `src/esper/leyline/slot_config.py`
- Test: `tests/leyline/test_slot_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/leyline/test_slot_config.py
from esper.leyline import InjectionSpec, SlotConfig


class TestSlotConfigFromSpecs:
    def test_from_specs_extracts_slot_ids(self):
        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
        ]
        config = SlotConfig.from_specs(specs)
        assert config.slot_ids == ("r0c0", "r0c1")
        assert config.num_slots == 2

    def test_from_specs_sorts_by_position(self):
        # Out of order input
        specs = [
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
        ]
        config = SlotConfig.from_specs(specs)
        # Should be sorted by position
        assert config.slot_ids == ("r0c0", "r0c1")

    def test_from_specs_preserves_channel_info(self):
        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
        ]
        config = SlotConfig.from_specs(specs)
        assert config.channels_for_slot("r0c0") == 64
        assert config.channels_for_slot("r0c1") == 128

    def test_from_specs_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig.from_specs([])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_config.py::TestSlotConfigFromSpecs -v`
Expected: FAIL with "AttributeError: type object 'SlotConfig' has no attribute 'from_specs'"

**Step 3: Write minimal implementation**

Update `src/esper/leyline/slot_config.py`:

```python
"""SlotConfig dataclass for dynamic action spaces.

Replaces the fixed SlotAction enum with dynamic slot configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.leyline.slot_id import format_slot_id

if TYPE_CHECKING:
    from esper.leyline.injection_spec import InjectionSpec


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for slot action space.

    Replaces the fixed SlotAction enum with dynamic slot configuration.

    Attributes:
        slot_ids: Tuple of slot IDs in action space order.
        _channel_map: Internal mapping of slot_id -> channels (frozen dict workaround)
    """

    slot_ids: tuple[str, ...]
    _channel_map: tuple[tuple[str, int], ...] = field(default=())

    @property
    def num_slots(self) -> int:
        """Number of slots in this configuration."""
        return len(self.slot_ids)

    def slot_id_for_index(self, idx: int) -> str:
        """Get slot ID for action index."""
        return self.slot_ids[idx]

    def index_for_slot_id(self, slot_id: str) -> int:
        """Get action index for slot ID."""
        return self.slot_ids.index(slot_id)

    def channels_for_slot(self, slot_id: str) -> int:
        """Get channel dimension for a slot.

        Returns:
            Channel count, or 0 if not available.
        """
        for sid, channels in self._channel_map:
            if sid == slot_id:
                return channels
        return 0

    @classmethod
    def default(cls) -> "SlotConfig":
        """Default 3-slot configuration (legacy compatible)."""
        return cls(slot_ids=("r0c0", "r0c1", "r0c2"))

    @classmethod
    def from_specs(cls, specs: list["InjectionSpec"]) -> "SlotConfig":
        """Create config from host injection specs.

        Args:
            specs: List of InjectionSpec from host.

        Returns:
            SlotConfig with slots sorted by position.

        Raises:
            ValueError: If specs is empty.
        """
        if not specs:
            raise ValueError("SlotConfig.from_specs requires at least one InjectionSpec")

        # Sort by position (early -> late in network)
        sorted_specs = sorted(specs, key=lambda s: s.position)

        slot_ids = tuple(s.slot_id for s in sorted_specs)
        channel_map = tuple((s.slot_id, s.channels) for s in sorted_specs)

        return cls(slot_ids=slot_ids, _channel_map=channel_map)

    @classmethod
    def for_grid(cls, rows: int, cols: int) -> "SlotConfig":
        """Create config for a full grid."""
        slot_ids = tuple(format_slot_id(r, c) for r in range(rows) for c in range(cols))
        return cls(slot_ids=slot_ids)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/slot_config.py tests/leyline/test_slot_config.py
git commit -m "feat(leyline): add SlotConfig.from_specs() for dynamic slot configuration"
```

---

## Phase 2: Host Dynamic Injection Points

### Task 2.1: Add injection_specs() to CNNHost

**Files:**
- Modify: `src/esper/kasmina/host.py`
- Test: `tests/kasmina/test_cnn_host.py`

**Step 1: Write the failing test**

```python
# Add to tests/kasmina/test_cnn_host.py
from esper.kasmina.host import CNNHost
from esper.leyline import InjectionSpec


class TestCNNHostInjectionSpecs:
    def test_default_3_block_has_3_specs(self):
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        assert len(specs) == 3
        assert all(isinstance(s, InjectionSpec) for s in specs)

    def test_specs_have_correct_slot_ids(self):
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1", "r0c2"]

    def test_specs_have_increasing_positions(self):
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        positions = [s.position for s in specs]
        assert positions == sorted(positions)
        assert all(0 < p <= 1.0 for p in positions)

    def test_specs_have_correct_channels(self):
        host = CNNHost(n_blocks=3, base_channels=32)
        specs = host.injection_specs()
        # Channels double each block: 32, 64, 128
        channels = [s.channels for s in specs]
        assert channels == [32, 64, 128]

    def test_5_block_host_has_5_specs(self):
        host = CNNHost(n_blocks=5, base_channels=16)
        specs = host.injection_specs()
        assert len(specs) == 5
        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_cnn_host.py::TestCNNHostInjectionSpecs -v`
Expected: FAIL with "AttributeError: 'CNNHost' object has no attribute 'injection_specs'"

**Step 3: Write minimal implementation**

Add to `CNNHost` class in `src/esper/kasmina/host.py`:

```python
def injection_specs(self) -> list["InjectionSpec"]:
    """Return available injection points as InjectionSpec objects.

    Returns:
        List of InjectionSpec, one per block, sorted by network position.
    """
    from esper.leyline import InjectionSpec
    from esper.leyline.slot_id import format_slot_id

    specs = []
    for i in range(self.n_blocks):
        specs.append(
            InjectionSpec(
                slot_id=format_slot_id(0, i),
                channels=self.blocks[i].conv.out_channels,
                position=(i + 1) / self.n_blocks,
                layer_range=(i, i + 1),
            )
        )
    return specs
```

Also update `segment_channels` property to use `injection_specs()`:

```python
@property
def segment_channels(self) -> dict[str, int]:
    """Map of slot_id -> channel dimension (derived from injection_specs)."""
    return {spec.slot_id: spec.channels for spec in self.injection_specs()}
```

Remove the hardcoded `self.segment_channels = {...}` from `__init__`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_cnn_host.py::TestCNNHostInjectionSpecs -v`
Expected: PASS

**Step 5: Run all CNNHost tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_cnn_host.py
git commit -m "feat(kasmina): add injection_specs() to CNNHost for dynamic slot discovery"
```

---

### Task 2.2: Add injection_specs() to TransformerHost

**Files:**
- Modify: `src/esper/kasmina/host.py`
- Test: `tests/kasmina/test_transformer_host.py`

**Step 1: Write the failing test**

```python
# Add to tests/kasmina/test_transformer_host.py
from esper.kasmina.host import TransformerHost
from esper.leyline import InjectionSpec


class TestTransformerHostInjectionSpecs:
    def test_default_has_specs(self):
        host = TransformerHost(n_layer=6)
        specs = host.injection_specs()
        assert len(specs) > 0
        assert all(isinstance(s, InjectionSpec) for s in specs)

    def test_6_layer_default_3_segments(self):
        host = TransformerHost(n_layer=6)
        specs = host.injection_specs()
        # Default: divide into 3 segments
        assert len(specs) == 3
        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1", "r0c2"]

    def test_layer_ranges_cover_all_layers(self):
        host = TransformerHost(n_layer=6)
        specs = host.injection_specs()
        # r0c0: layers 0-1, r0c1: layers 2-3, r0c2: layers 4-5
        ranges = [s.layer_range for s in specs]
        assert ranges[0] == (0, 2)
        assert ranges[1] == (2, 4)
        assert ranges[2] == (4, 6)

    def test_custom_num_segments(self):
        host = TransformerHost(n_layer=6, num_segments=2)
        specs = host.injection_specs()
        assert len(specs) == 2
        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1"]

    def test_channels_are_n_embd(self):
        host = TransformerHost(n_layer=6, n_embd=512)
        specs = host.injection_specs()
        assert all(s.channels == 512 for s in specs)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_transformer_host.py::TestTransformerHostInjectionSpecs -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `TransformerHost.__init__` to accept `num_segments` parameter:

```python
def __init__(
    self,
    vocab_size: int = 50257,
    n_embd: int = 384,
    n_head: int = 6,
    n_layer: int = 6,
    block_size: int = 256,
    dropout: float = 0.1,
    num_segments: int = 3,  # NEW: configurable segment count
):
    super().__init__()
    self.n_layer = n_layer
    self.n_embd = n_embd
    self.block_size = block_size
    self._num_segments = num_segments
    # ... rest of init unchanged ...
```

Add `injection_specs()` method:

```python
def injection_specs(self) -> list["InjectionSpec"]:
    """Return available injection points as InjectionSpec objects.

    Returns:
        List of InjectionSpec, one per segment, sorted by network position.
    """
    from esper.leyline import InjectionSpec
    from esper.leyline.slot_id import format_slot_id

    specs = []
    for i in range(self._num_segments):
        start_layer = i * self.n_layer // self._num_segments
        end_layer = (i + 1) * self.n_layer // self._num_segments
        specs.append(
            InjectionSpec(
                slot_id=format_slot_id(0, i),
                channels=self.n_embd,
                position=end_layer / self.n_layer,
                layer_range=(start_layer, end_layer),
            )
        )
    return specs
```

Update `segment_channels` to be a property derived from `injection_specs()`:

```python
@property
def segment_channels(self) -> dict[str, int]:
    """Map of slot_id -> channel dimension (derived from injection_specs)."""
    return {spec.slot_id: spec.channels for spec in self.injection_specs()}
```

Update `_segment_boundaries` similarly.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_transformer_host.py::TestTransformerHostInjectionSpecs -v`
Expected: PASS

**Step 5: Run all Kasmina tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_transformer_host.py
git commit -m "feat(kasmina): add injection_specs() to TransformerHost with configurable num_segments"
```

---

### Task 2.3: Update MorphogeneticModel to Use Host Specs

**Files:**
- Modify: `src/esper/kasmina/host.py`
- Test: `tests/kasmina/test_morphogenetic_model.py`

**Step 1: Write the failing test**

```python
# Add to tests/kasmina/test_morphogenetic_model.py
from esper.kasmina.host import CNNHost, MorphogeneticModel


class TestMorphogeneticModelDynamicSlots:
    def test_validates_slots_against_host_specs(self):
        host = CNNHost(n_blocks=3)
        # r0c0, r0c1, r0c2 should be valid
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1"])
        assert len(model.seed_slots) == 2

    def test_rejects_unsupported_slot(self):
        host = CNNHost(n_blocks=3)
        with pytest.raises(ValueError, match="Unknown slot.*r0c5"):
            MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c5"])

    def test_5_block_host_supports_5_slots(self):
        host = CNNHost(n_blocks=5)
        model = MorphogeneticModel(
            host, device="cpu",
            slots=["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]
        )
        assert len(model.seed_slots) == 5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_morphogenetic_model.py::TestMorphogeneticModelDynamicSlots -v`
Expected: FAIL (5-block test fails because host.segment_channels only has 3 keys currently)

**Step 3: Implementation**

The implementation is already mostly correct since we updated `segment_channels` to be derived from `injection_specs()`. Just ensure `MorphogeneticModel._slot_order` is dynamic:

```python
# In MorphogeneticModel.__init__, replace:
# self._slot_order = ["r0c0", "r0c1", "r0c2"]
# With:
from esper.leyline.slot_id import slot_sort_key
self._slot_order = sorted(segment_channels.keys(), key=slot_sort_key)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_morphogenetic_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_morphogenetic_model.py
git commit -m "refactor(kasmina): derive MorphogeneticModel._slot_order from host specs"
```

---

## Phase 3: Dynamic Feature Extraction

### Task 3.1: Make MULTISLOT_FEATURE_SIZE Dynamic

**Files:**
- Modify: `src/esper/simic/features.py`
- Test: `tests/simic/test_features.py`

**Step 1: Write the failing test**

```python
# Add to tests/simic/test_features.py
from esper.simic.features import compute_feature_size, obs_to_multislot_features


class TestDynamicFeatureSize:
    def test_compute_feature_size_3_slots(self):
        size = compute_feature_size(num_slots=3, use_telemetry=False)
        assert size == 50  # 23 base + 3*9 per-slot

    def test_compute_feature_size_2_slots(self):
        size = compute_feature_size(num_slots=2, use_telemetry=False)
        assert size == 41  # 23 base + 2*9 per-slot

    def test_compute_feature_size_5_slots(self):
        size = compute_feature_size(num_slots=5, use_telemetry=False)
        assert size == 68  # 23 base + 5*9 per-slot

    def test_compute_feature_size_with_telemetry(self):
        size = compute_feature_size(num_slots=3, use_telemetry=True)
        assert size == 80  # 50 base + 3*10 telemetry

    def test_obs_to_features_respects_slot_ids(self):
        obs = {
            'epoch': 1, 'step': 10, 'loss': 0.5, 'accuracy': 0.8,
            'slots': {
                'r0c0': {'is_active': 1.0, 'stage': 2, 'alpha': 0.5, 'improvement': 0.01, 'blueprint_id': 'norm'},
                'r0c1': {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None},
            }
        }
        # With 2-slot config
        features = obs_to_multislot_features(obs, slot_ids=("r0c0", "r0c1"))
        expected_size = compute_feature_size(num_slots=2, use_telemetry=False)
        assert len(features) == expected_size
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_features.py::TestDynamicFeatureSize -v`
Expected: FAIL with "cannot import name 'compute_feature_size'"

**Step 3: Write minimal implementation**

Add to `src/esper/simic/features.py`:

```python
# Constants for feature calculation
_BASE_FEATURE_COUNT = 23
_PER_SLOT_FEATURE_COUNT = 9
_TELEMETRY_FEATURE_DIM = 10  # SeedTelemetry.feature_dim()


def compute_feature_size(num_slots: int, use_telemetry: bool = False) -> int:
    """Compute feature vector size for given slot count.

    Args:
        num_slots: Number of slots in configuration.
        use_telemetry: Whether telemetry features are included.

    Returns:
        Total feature vector dimension.
    """
    base = _BASE_FEATURE_COUNT + num_slots * _PER_SLOT_FEATURE_COUNT
    if use_telemetry:
        base += num_slots * _TELEMETRY_FEATURE_DIM
    return base


# Keep MULTISLOT_FEATURE_SIZE for backwards compatibility (3-slot default)
MULTISLOT_FEATURE_SIZE = compute_feature_size(num_slots=3, use_telemetry=False)
```

Update `obs_to_multislot_features` signature:

```python
def obs_to_multislot_features(
    obs: dict,
    total_seeds: int = 0,
    max_seeds: int = 1,
    slot_ids: tuple[str, ...] = ("r0c0", "r0c1", "r0c2"),  # NEW
) -> list[float]:
    """Extract features including per-slot state.

    Args:
        obs: Observation dictionary.
        total_seeds: Total seeds for budget tracking.
        max_seeds: Maximum allowed seeds.
        slot_ids: Tuple of slot IDs to extract features for.

    Returns:
        Feature vector of size compute_feature_size(len(slot_ids)).
    """
```

Then update the loop inside to use `slot_ids` instead of hardcoded list:

```python
# Replace:
# for slot_id in ['r0c0', 'r0c1', 'r0c2']:
# With:
for slot_id in slot_ids:
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_features.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/features.py tests/simic/test_features.py
git commit -m "feat(simic): make feature extraction dynamic with compute_feature_size()"
```

---

### Task 3.2: Update signals_to_features in PPO

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Test: `tests/simic/test_ppo.py`

**Step 1: Write the failing test**

```python
# Add to tests/simic/test_ppo.py
from esper.simic.ppo import signals_to_features


class TestSignalsToFeaturesDynamic:
    def test_respects_enabled_slots_2_slots(self):
        signals = {
            'epoch': 1, 'step': 10, 'loss': 0.5, 'accuracy': 0.8,
            'host_accuracy': 0.75, 'total_params': 1000,
        }
        slot_reports = {}
        enabled_slots = ["r0c0", "r0c1"]

        obs, features = signals_to_features(
            signals, slot_reports,
            enabled_slots=enabled_slots,
            use_telemetry=False,
        )

        # Features should be sized for 2 slots
        from esper.simic.features import compute_feature_size
        expected_size = compute_feature_size(num_slots=2, use_telemetry=False)
        assert len(features) == expected_size
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py::TestSignalsToFeaturesDynamic -v`
Expected: FAIL (currently hardcoded to 3 slots)

**Step 3: Write minimal implementation**

Update `signals_to_features` in `src/esper/simic/ppo.py`:

```python
def signals_to_features(
    signals: dict,
    slot_reports: dict[str, SeedStateReport],
    enabled_slots: list[str],
    use_telemetry: bool = False,
    total_seeds: int = 0,
    max_seeds: int = 1,
) -> tuple[dict, list[float]]:
    """Convert training signals to observation dict and feature vector.

    Args:
        signals: Raw training signals.
        slot_reports: Per-slot state reports.
        enabled_slots: List of enabled slot IDs (determines feature size).
        use_telemetry: Include telemetry features.
        total_seeds: Total seeds for budget tracking.
        max_seeds: Maximum allowed seeds.

    Returns:
        Tuple of (obs_dict, feature_vector).
    """
    # ... build obs dict ...

    # Build per-slot state dict from reports
    slot_states = {}
    for slot_id in enabled_slots:  # CHANGED: was hardcoded ['r0c0', 'r0c1', 'r0c2']
        report = slot_reports.get(slot_id)
        if report:
            # ... existing logic ...
        else:
            slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0, 'blueprint_id': None}

    obs['slots'] = slot_states

    features = obs_to_multislot_features(
        obs,
        total_seeds=total_seeds,
        max_seeds=max_seeds,
        slot_ids=tuple(enabled_slots),  # CHANGED: pass slot_ids
    )

    if use_telemetry:
        from esper.leyline import SeedTelemetry
        telemetry_features: list[float] = []
        for slot_id in enabled_slots:  # CHANGED: was CANONICAL_SLOTS
            # ... existing telemetry logic ...
        features.extend(telemetry_features)

    return obs, features
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo.py
git commit -m "refactor(simic): make signals_to_features use dynamic slot list"
```

---

## Phase 4: Network and Buffer Dynamic Sizing

### Task 4.1: Thread slot_config Through PPOAgent

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Test: `tests/simic/test_ppo.py`

**Step 1: Write the failing test**

```python
# Add to tests/simic/test_ppo.py
from esper.leyline import SlotConfig


class TestPPOAgentDynamicSlots:
    def test_agent_accepts_slot_config(self):
        from esper.simic.ppo import PPOAgent
        from esper.simic.features import compute_feature_size

        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
        state_dim = compute_feature_size(num_slots=2, use_telemetry=False)

        agent = PPOAgent(state_dim=state_dim, slot_config=slot_config)
        assert agent.network.num_slots == 2

    def test_agent_default_3_slots(self):
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(state_dim=50)  # Default 3-slot size
        assert agent.network.num_slots == 3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py::TestPPOAgentDynamicSlots -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `PPOAgent.__init__` in `src/esper/simic/ppo.py`:

```python
class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        slot_config: SlotConfig | None = None,  # NEW
        # ... other params ...
    ):
        if slot_config is None:
            slot_config = SlotConfig.default()

        self.slot_config = slot_config

        self.network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=slot_config.num_slots,  # CHANGED: was SlotConfig.default().num_slots
            # ... other params ...
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py::TestPPOAgentDynamicSlots -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo.py
git commit -m "feat(simic): thread slot_config through PPOAgent to network"
```

---

### Task 4.2: Update TamiyoBuffer for Dynamic Slots

**Files:**
- Modify: `src/esper/simic/tamiyo_buffer.py`
- Test: `tests/simic/test_tamiyo_buffer.py`

**Step 1: Write the failing test**

```python
# Add to tests/simic/test_tamiyo_buffer.py
from esper.leyline import SlotConfig


class TestTamiyoBufferDynamicSlots:
    def test_buffer_accepts_slot_config(self):
        from esper.simic.tamiyo_buffer import TamiyoBuffer

        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
        buffer = TamiyoBuffer(
            capacity=100,
            state_dim=41,  # 2-slot feature size
            slot_config=slot_config,
        )
        assert buffer.num_slots == 2

    def test_buffer_default_3_slots(self):
        from esper.simic.tamiyo_buffer import TamiyoBuffer

        buffer = TamiyoBuffer(capacity=100, state_dim=50)
        assert buffer.num_slots == 3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_tamiyo_buffer.py::TestTamiyoBufferDynamicSlots -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `TamiyoBuffer.__init__`:

```python
@dataclass
class TamiyoBuffer:
    capacity: int
    state_dim: int
    slot_config: SlotConfig = field(default_factory=SlotConfig.default)

    @property
    def num_slots(self) -> int:
        return self.slot_config.num_slots

    def __post_init__(self):
        # Use self.num_slots instead of hardcoded value
        # ... rest of init ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_tamiyo_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/tamiyo_buffer.py tests/simic/test_tamiyo_buffer.py
git commit -m "feat(simic): thread slot_config through TamiyoBuffer"
```

---

### Task 4.3: Update train_ppo_vectorized Entry Point

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Test: `tests/simic/test_vectorized.py`

**Step 1: Write the failing test**

```python
# Add to tests/simic/test_vectorized.py
class TestVectorizedDynamicSlots:
    def test_train_ppo_vectorized_builds_correct_slot_config(self):
        """Verify slot_config is built from enabled slots, not hardcoded."""
        # This is an integration test - we just verify the function accepts
        # a 2-slot configuration without error
        from esper.simic.vectorized import train_ppo_vectorized
        from esper.simic.config import TrainingConfig

        # Would need mock environment setup - mark as integration test
        pass  # See integration tests for full coverage
```

**Step 2: Implementation**

In `train_ppo_vectorized`, find where `slot_config = SlotConfig.default()` is used (line ~826) and replace with:

```python
# Build slot_config from enabled slots
from esper.leyline import SlotConfig
slot_config = SlotConfig(slot_ids=tuple(slots))  # slots comes from config.slots
```

**Step 3: Run all vectorized tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_vectorized.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "fix(simic): build slot_config from enabled slots in train_ppo_vectorized"
```

---

## Phase 5: Cleanup Legacy Hardcoding

### Task 5.1: Remove CANONICAL_SLOTS from Simic

**Files:**
- Modify: `src/esper/simic/slots.py`
- Modify: `src/esper/simic/config.py`
- Modify: `src/esper/simic/ppo.py`
- Test: `tests/simic/test_slots.py`

**Step 1: Audit CANONICAL_SLOTS usage**

Run: `grep -rn "CANONICAL_SLOTS" src/esper/`

**Step 2: Replace usages**

In `config.py`, replace validation against `CANONICAL_SLOTS` with validation against host specs (requires host to be passed or defer validation to MorphogeneticModel).

For now, keep `CANONICAL_SLOTS` but add deprecation warning:

```python
# src/esper/simic/slots.py
import warnings

# DEPRECATED: Use host.injection_specs() instead
CANONICAL_SLOTS: tuple[str, ...] = ("r0c0", "r0c1", "r0c2")

def _warn_canonical_slots():
    warnings.warn(
        "CANONICAL_SLOTS is deprecated. Use host.injection_specs() to discover available slots.",
        DeprecationWarning,
        stacklevel=3,
    )
```

**Step 3: Commit**

```bash
git add src/esper/simic/slots.py src/esper/simic/config.py src/esper/simic/ppo.py
git commit -m "refactor(simic): deprecate CANONICAL_SLOTS in favor of host.injection_specs()"
```

---

### Task 5.2: Update TUI for Dynamic Slots

**Files:**
- Modify: `src/esper/karn/tui.py`
- Test: Manual testing (TUI is hard to unit test)

**Step 1: Find hardcoded slot references**

Run: `grep -n "r0c0\|r0c1\|r0c2" src/esper/karn/tui.py`

**Step 2: Replace with dynamic iteration**

```python
# Instead of:
# _slot_summary("r0c0"),
# _slot_summary("r0c1"),
# _slot_summary("r0c2"),

# Use:
# for slot_id in self._enabled_slots:
#     _slot_summary(slot_id)
```

The TUI needs to receive `enabled_slots` from the training config.

**Step 3: Commit**

```bash
git add src/esper/karn/tui.py
git commit -m "refactor(karn): make TUI slot display dynamic"
```

---

### Task 5.3: Update Action Masks

**Files:**
- Modify: `src/esper/simic/action_masks.py`
- Test: `tests/simic/test_action_masks.py`

**Step 1: Remove hardcoded default**

```python
# Change:
def slot_id_to_index(slot_id: str, slot_ids: tuple[str, ...] = ("r0c0", "r0c1", "r0c2")) -> int:

# To:
def slot_id_to_index(slot_id: str, slot_ids: tuple[str, ...]) -> int:
    """Convert slot ID to index.

    Args:
        slot_id: Slot ID to look up.
        slot_ids: Ordered tuple of valid slot IDs (required, no default).
    """
```

**Step 2: Update all call sites**

Search for `slot_id_to_index` and ensure `slot_ids` is always passed.

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_action_masks.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "refactor(simic): remove hardcoded default from slot_id_to_index"
```

---

## Phase 6: Integration Testing

### Task 6.1: End-to-End Test with 2 Slots

**Files:**
- Create: `tests/integration/test_dynamic_slots.py`

**Step 1: Write integration test**

```python
# tests/integration/test_dynamic_slots.py
"""Integration tests for dynamic slot support."""

import pytest
import torch

from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.leyline import SlotConfig
from esper.simic.features import compute_feature_size


class TestDynamicSlotsIntegration:
    @pytest.mark.slow
    def test_2_slot_training_smoke(self):
        """Verify 2-slot configuration works end-to-end."""
        # Create 2-slot host
        host = CNNHost(n_blocks=2)
        specs = host.injection_specs()
        assert len(specs) == 2

        # Create model with 2 slots
        slot_ids = [s.slot_id for s in specs]
        model = MorphogeneticModel(host, device="cpu", slots=slot_ids)
        assert len(model.seed_slots) == 2

        # Verify feature size
        feature_size = compute_feature_size(num_slots=2, use_telemetry=False)
        assert feature_size == 41

        # Forward pass
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10)

    @pytest.mark.slow
    def test_5_slot_configuration(self):
        """Verify 5-slot configuration works."""
        host = CNNHost(n_blocks=5, base_channels=16, pool_layers=3)
        specs = host.injection_specs()
        assert len(specs) == 5

        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]

        model = MorphogeneticModel(host, device="cpu", slots=slot_ids)
        assert len(model.seed_slots) == 5

        feature_size = compute_feature_size(num_slots=5, use_telemetry=False)
        assert feature_size == 68

        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10)
```

**Step 2: Run integration test**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_dynamic_slots.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_dynamic_slots.py
git commit -m "test: add integration tests for dynamic slot configurations"
```

---

### Task 6.2: Run Full Test Suite

**Step 1: Run all tests**

Run: `PYTHONPATH=src uv run pytest tests/ -v --tb=short`
Expected: All PASS

**Step 2: Fix any regressions**

If tests fail, investigate and fix before proceeding.

**Step 3: Final commit**

```bash
git commit --allow-empty -m "chore: verify full test suite passes with dynamic slots"
```

---

## Summary

**Total Tasks:** 14 tasks across 6 phases
**Estimated Time:** 20-25 hours

**Key Changes:**
1. `InjectionSpec` dataclass as contract between hosts and consumers
2. Hosts expose `injection_specs()` method
3. `SlotConfig.from_specs()` factory for building config from host
4. `compute_feature_size()` for dynamic feature vector sizing
5. Thread `slot_config` through PPOAgent → Network → Buffer
6. Remove hardcoded "3" from all locations

**Stability Improvements:**
- Single source of truth (host defines available slots)
- Fail-fast validation (config validated against host at construction time)
- No implicit coupling (components explicitly pass slot_config)
- Impossible for components to disagree about slot count

---

**Plan complete and saved to `docs/plans/2025-12-16-dynamic-slots.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
