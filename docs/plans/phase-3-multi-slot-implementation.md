# Multi-Slot Seed Architecture Implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-slot architecture with multi-slot morphogenetic plane supporting factored action space

**Architecture:** Multiple SeedSlots at different network depths (early/mid/late), each with independent lifecycle. Factored multi-head policy samples (slot, blueprint, blend_method, lifecycle_op) jointly. Single-slot mode is just multi-slot with 1 slot.

**Tech Stack:** PyTorch, PPO, existing esper modules (kasmina, simic, leyline)

**Design Doc:** `docs/plans/multi-seed-architecture.md`

---

## Task 1: Add NoopSeed Blueprint

**Files:**
- Modify: `src/esper/kasmina/blueprints.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_blueprints.py
def test_noop_seed_is_identity():
    """NoopSeed should pass through input unchanged."""
    from esper.kasmina.blueprints import BlueprintCatalog
    import torch

    seed = BlueprintCatalog.create_seed("noop", channels=64)
    x = torch.randn(2, 64, 8, 8)
    y = seed(x)

    assert torch.allclose(x, y), "NoopSeed should be identity"
    assert sum(p.numel() for p in seed.parameters()) == 0, "NoopSeed should have no params"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/kasmina/test_blueprints.py::test_noop_seed_is_identity -v`
Expected: FAIL with "Unknown blueprint: noop"

**Step 3: Write minimal implementation**

```python
# In src/esper/kasmina/blueprints.py, add after DepthwiseSeed class:

class NoopSeed(nn.Module):
    """Identity seed - placeholder before bursting."""

    blueprint_id = "noop"

    def __init__(self, channels: int):
        super().__init__()
        # No parameters - pure pass-through

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
```

And register in BlueprintCatalog._blueprints:
```python
_blueprints: dict[str, type[nn.Module]] = {
    "noop": NoopSeed,  # Add this line
    "conv_enhance": ConvEnhanceSeed,
    ...
}
```

And add to __all__:
```python
__all__ = [
    "NoopSeed",  # Add this
    ...
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/kasmina/test_blueprints.py::test_noop_seed_is_identity -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add NoopSeed identity blueprint"
```

---

## Task 2: Add Blending Algorithm Library

**Files:**
- Create: `src/esper/kasmina/blending.py`
- Test: `tests/kasmina/test_blending.py`

**Step 1: Write the failing tests**

```python
# tests/kasmina/test_blending.py
import pytest
import torch


def test_linear_blend_schedule():
    """Linear blend should ramp alpha from 0 to 1 over steps."""
    from esper.kasmina.blending import LinearBlend

    blend = LinearBlend(total_steps=10)

    assert blend.get_alpha(0) == 0.0
    assert blend.get_alpha(5) == 0.5
    assert blend.get_alpha(10) == 1.0
    assert blend.get_alpha(15) == 1.0  # Clamp at 1.0


def test_sigmoid_blend_schedule():
    """Sigmoid blend should have smooth S-curve."""
    from esper.kasmina.blending import SigmoidBlend

    blend = SigmoidBlend(total_steps=10)

    # Sigmoid properties: starts slow, ends slow, fast in middle
    alpha_0 = blend.get_alpha(0)
    alpha_5 = blend.get_alpha(5)
    alpha_10 = blend.get_alpha(10)

    assert alpha_0 < 0.1, "Sigmoid should start near 0"
    assert 0.4 < alpha_5 < 0.6, "Sigmoid midpoint should be ~0.5"
    assert alpha_10 > 0.9, "Sigmoid should end near 1"


def test_gated_blend_schedule():
    """Gated blend should use learned gate."""
    from esper.kasmina.blending import GatedBlend

    blend = GatedBlend(channels=64)

    # Should have learnable parameters
    assert sum(p.numel() for p in blend.parameters()) > 0

    # Should produce valid alpha tensor
    x = torch.randn(2, 64, 8, 8)
    alpha = blend.get_alpha_tensor(x)
    assert alpha.shape == (2, 64, 1, 1) or alpha.shape == (2, 1, 1, 1)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_blend_registry():
    """BlendCatalog should list and create blends."""
    from esper.kasmina.blending import BlendCatalog

    available = BlendCatalog.list_algorithms()
    assert "linear" in available
    assert "sigmoid" in available
    assert "gated" in available

    blend = BlendCatalog.create("linear", total_steps=10)
    assert blend.get_alpha(5) == 0.5
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/kasmina/test_blending.py -v`
Expected: FAIL with "No module named 'esper.kasmina.blending'"

**Step 3: Write implementation**

```python
# src/esper/kasmina/blending.py
"""Kasmina Blending Algorithms - Tamiyo's blending library.

Each algorithm defines how a seed's influence ramps from 0 to 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn


class BlendAlgorithm(ABC):
    """Base class for blending algorithms."""

    algorithm_id: str = "base"

    @abstractmethod
    def get_alpha(self, step: int) -> float:
        """Get alpha value for a given step."""
        pass

    def get_alpha_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Get alpha as tensor (for gated blends). Default: scalar."""
        raise NotImplementedError("Use get_alpha() for non-gated blends")


class LinearBlend(BlendAlgorithm):
    """Linear ramp from 0 to 1 over total_steps."""

    algorithm_id = "linear"

    def __init__(self, total_steps: int = 5):
        self.total_steps = max(1, total_steps)

    def get_alpha(self, step: int) -> float:
        return min(1.0, max(0.0, step / self.total_steps))


class SigmoidBlend(BlendAlgorithm):
    """Sigmoid curve for smooth transitions."""

    algorithm_id = "sigmoid"

    def __init__(self, total_steps: int = 10, steepness: float = 1.0):
        self.total_steps = max(1, total_steps)
        self.steepness = steepness

    def get_alpha(self, step: int) -> float:
        # Map step to [-6, 6] range for sigmoid
        x = (step / self.total_steps - 0.5) * 12 * self.steepness
        return 1.0 / (1.0 + math.exp(-x))


class GatedBlend(BlendAlgorithm, nn.Module):
    """Learned gating mechanism for adaptive blending."""

    algorithm_id = "gated"

    def __init__(self, channels: int):
        BlendAlgorithm.__init__(self)
        nn.Module.__init__(self)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        self._step = 0

    def get_alpha(self, step: int) -> float:
        self._step = step
        return 0.5  # Default; actual alpha comes from get_alpha_tensor

    def get_alpha_tensor(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(x)  # (B, 1)
        return alpha.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting


class BlendCatalog:
    """Registry of blending algorithms."""

    _algorithms: dict[str, type] = {
        "linear": LinearBlend,
        "sigmoid": SigmoidBlend,
        "gated": GatedBlend,
    }

    @classmethod
    def list_algorithms(cls) -> list[str]:
        return list(cls._algorithms.keys())

    @classmethod
    def create(cls, algorithm_id: str, **kwargs) -> BlendAlgorithm:
        if algorithm_id not in cls._algorithms:
            raise ValueError(f"Unknown blend algorithm: {algorithm_id}")
        return cls._algorithms[algorithm_id](**kwargs)


__all__ = [
    "BlendAlgorithm",
    "LinearBlend",
    "SigmoidBlend",
    "GatedBlend",
    "BlendCatalog",
]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/kasmina/test_blending.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blending.py tests/kasmina/test_blending.py
git commit -m "feat(kasmina): add blending algorithm library (linear, sigmoid, gated)"
```

---

## Task 3: Extend HostCNN with Segment Boundaries

**Files:**
- Modify: `src/esper/kasmina/host.py`
- Test: `tests/kasmina/test_host.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_host.py
def test_host_segment_channels():
    """HostCNN should expose channel counts at each segment boundary."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()

    # Should expose channels at each injection point
    assert hasattr(host, 'segment_channels')
    assert host.segment_channels == {
        "early": 32,   # After block1
        "mid": 64,     # After block2
        "late": 128,   # After block3
    }


def test_host_forward_segments():
    """HostCNN should support segmented forward pass."""
    from esper.kasmina.host import HostCNN
    import torch

    host = HostCNN()
    x = torch.randn(2, 3, 32, 32)

    # Forward to each segment
    x_early = host.forward_to_segment("early", x)
    assert x_early.shape == (2, 32, 16, 16)  # After block1 + pool

    x_mid = host.forward_to_segment("mid", x)
    assert x_mid.shape == (2, 64, 8, 8)  # After block2 + pool

    x_late = host.forward_to_segment("late", x)
    assert x_late.shape == (2, 128, 4, 4)  # After block3 + pool

    # Forward from segment to output
    out = host.forward_from_segment("late", x_late)
    assert out.shape == (2, 10)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/kasmina/test_host.py::test_host_segment_channels -v`
Expected: FAIL with "has no attribute 'segment_channels'"

**Step 3: Write implementation**

Update HostCNN in `src/esper/kasmina/host.py`:

```python
class HostCNN(nn.Module):
    """Host CNN with multiple injection points."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Segment channel counts for multi-slot support
        self.segment_channels = {
            "early": 32,   # After block1
            "mid": 64,     # After block2
            "late": 128,   # After block3
        }

        # Legacy single-slot compatibility
        self.injection_channels = 64  # After block2

    def forward_to_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        """Forward through network up to and including a segment."""
        x = self.pool(self.block1(x))
        if segment == "early":
            return x

        x = self.pool(self.block2(x))
        if segment == "mid":
            return x

        x = self.pool(self.block3(x))
        return x  # "late"

    def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        """Forward from a segment to output."""
        if segment == "early":
            x = self.pool(self.block2(x))
            x = self.pool(self.block3(x))
        elif segment == "mid":
            x = self.pool(self.block3(x))
        # "late" - already at final features

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    # Keep legacy methods for backwards compatibility during migration
    def forward_to_injection(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_to_segment("mid", x)

    def forward_from_injection(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_segment("mid", x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_to_segment("late", x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/kasmina/test_host.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_host.py
git commit -m "feat(kasmina): add segment boundaries to HostCNN for multi-slot support"
```

---

## Task 4: Multi-Slot MorphogeneticModel

**Files:**
- Modify: `src/esper/kasmina/host.py`
- Test: `tests/kasmina/test_host.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_host.py
def test_multislot_model_creation():
    """MorphogeneticModel should support multiple slots."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel

    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    assert len(model.seed_slots) == 3
    assert "early" in model.seed_slots
    assert "mid" in model.seed_slots
    assert "late" in model.seed_slots

    # Each slot should have correct channels
    assert model.seed_slots["early"].channels == 32
    assert model.seed_slots["mid"].channels == 64
    assert model.seed_slots["late"].channels == 128


def test_multislot_forward_pass():
    """Multi-slot model forward should pass through all slots."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel
    import torch

    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_multislot_germinate_specific_slot():
    """Should germinate seed in specific slot."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel

    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Germinate in mid slot
    model.germinate_seed("conv_enhance", "test_seed", slot="mid")

    assert model.seed_slots["mid"].is_active
    assert not model.seed_slots["early"].is_active
    assert not model.seed_slots["late"].is_active


def test_single_slot_is_default():
    """Single slot mode (backwards compat) should still work."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel

    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu")  # No slots arg

    # Should default to single "mid" slot for backwards compat
    assert len(model.seed_slots) == 1
    assert "mid" in model.seed_slots
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/kasmina/test_host.py::test_multislot_model_creation -v`
Expected: FAIL with "unexpected keyword argument 'slots'"

**Step 3: Write implementation**

Update MorphogeneticModel in `src/esper/kasmina/host.py`:

```python
class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slots - the morphogenetic plane.

    Kasmina behaves like many independent cells, but is implemented
    as a single morphogenetic plane for scalability.
    """

    def __init__(
        self,
        host: HostCNN,
        device: str = "cpu",
        slots: list[str] | None = None,
    ):
        super().__init__()
        self.host = host
        self._device = device

        # Default to single mid slot for backwards compatibility
        if slots is None:
            slots = ["mid"]

        # Create seed slots dict
        self.seed_slots: dict[str, SeedSlot] = {}
        for slot_id in slots:
            if slot_id not in host.segment_channels:
                raise ValueError(f"Unknown slot: {slot_id}. "
                               f"Available: {list(host.segment_channels.keys())}")
            self.seed_slots[slot_id] = SeedSlot(
                slot_id=slot_id,
                channels=host.segment_channels[slot_id],
                device=device,
            )

        # Track slot order for forward pass
        self._slot_order = ["early", "mid", "late"]
        self._active_slots = [s for s in self._slot_order if s in self.seed_slots]

        # Isolation monitor
        self.isolation_monitor = GradientIsolationMonitor()

        # Legacy property for single-slot code
        self._legacy_single_slot = len(slots) == 1

    @property
    def seed_slot(self) -> SeedSlot:
        """Legacy property for single-slot access."""
        if self._legacy_single_slot:
            return next(iter(self.seed_slots.values()))
        raise AttributeError("Use seed_slots dict for multi-slot access")

    def to(self, *args, **kwargs):
        """Override to() to propagate device to all slots."""
        result = super().to(*args, **kwargs)

        try:
            new_device = next(self.parameters()).device
        except StopIteration:
            return result

        for slot in self.seed_slots.values():
            slot.device = new_device
            if slot.seed is not None:
                slot.seed = slot.seed.to(new_device)

        self._device = str(new_device)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through host with all active slots."""
        # Process through each segment with its slot
        for slot_id in self._active_slots:
            x = self.host.forward_to_segment(slot_id, x)
            x = self.seed_slots[slot_id].forward(x)

        # Final classification
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.host.classifier(x)

    def germinate_seed(
        self,
        blueprint_id: str,
        seed_id: str,
        slot: str | None = None,
        blend_algorithm: str = "linear",
    ) -> None:
        """Germinate a new seed in a specific slot."""
        if slot is None:
            slot = "mid"  # Default for backwards compat

        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}")

        self.seed_slots[slot].germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )

    def cull_seed(self, slot: str | None = None) -> None:
        """Cull the seed in a specific slot."""
        if slot is None:
            slot = "mid"
        self.seed_slots[slot].cull()

    def get_seed_parameters(self, slot: str | None = None):
        """Get seed parameters from specific slot or all slots."""
        if slot:
            return self.seed_slots[slot].get_parameters()
        # Return all seed params from all slots
        for s in self.seed_slots.values():
            yield from s.get_parameters()

    def get_host_parameters(self):
        return self.host.parameters()

    @property
    def has_active_seed(self) -> bool:
        """Check if any slot has an active seed."""
        return any(s.is_active for s in self.seed_slots.values())

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has active seed."""
        return self.seed_slots[slot].is_active

    @property
    def seed_state(self):
        """Legacy property - returns first active seed state."""
        for slot in self.seed_slots.values():
            if slot.is_active:
                return slot.state
        return None

    def get_slot_states(self) -> dict[str, any]:
        """Get state of all slots."""
        return {
            slot_id: slot.state
            for slot_id, slot in self.seed_slots.items()
        }

    @property
    def active_seed_params(self) -> int:
        """Total trainable params across all active seeds."""
        return sum(s.active_seed_params for s in self.seed_slots.values())
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/kasmina/test_host.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_host.py
git commit -m "feat(kasmina): multi-slot MorphogeneticModel with per-slot lifecycle"
```

---

## Task 5: Factored Action Space

**Files:**
- Create: `src/esper/leyline/factored_actions.py`
- Test: `tests/leyline/test_factored_actions.py`

**Step 1: Write the failing test**

```python
# tests/leyline/test_factored_actions.py
def test_slot_action_enum():
    """SlotAction should enumerate slot choices."""
    from esper.leyline.factored_actions import SlotAction

    assert SlotAction.EARLY.value == 0
    assert SlotAction.MID.value == 1
    assert SlotAction.LATE.value == 2
    assert len(SlotAction) == 3


def test_blueprint_action_enum():
    """BlueprintAction should enumerate blueprint choices."""
    from esper.leyline.factored_actions import BlueprintAction

    assert BlueprintAction.NOOP.value == 0
    assert BlueprintAction.CONV_ENHANCE.value == 1
    assert len(BlueprintAction) >= 5  # noop + 4 blueprints


def test_blend_action_enum():
    """BlendAction should enumerate blending algorithm choices."""
    from esper.leyline.factored_actions import BlendAction

    assert BlendAction.LINEAR.value == 0
    assert BlendAction.SIGMOID.value == 1
    assert BlendAction.GATED.value == 2


def test_lifecycle_op_enum():
    """LifecycleOp should enumerate lifecycle operations."""
    from esper.leyline.factored_actions import LifecycleOp

    assert LifecycleOp.WAIT.value == 0
    assert LifecycleOp.GERMINATE.value == 1
    assert LifecycleOp.ADVANCE.value == 2
    assert LifecycleOp.CULL.value == 3


def test_factored_action_composition():
    """FactoredAction should compose slot, blueprint, blend, op."""
    from esper.leyline.factored_actions import (
        FactoredAction, SlotAction, BlueprintAction, BlendAction, LifecycleOp
    )

    action = FactoredAction(
        slot=SlotAction.MID,
        blueprint=BlueprintAction.CONV_ENHANCE,
        blend=BlendAction.LINEAR,
        op=LifecycleOp.GERMINATE,
    )

    assert action.slot == SlotAction.MID
    assert action.blueprint == BlueprintAction.CONV_ENHANCE
    assert action.is_germinate
    assert action.blueprint_id == "conv_enhance"
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/leyline/test_factored_actions.py -v`
Expected: FAIL with "No module named 'esper.leyline.factored_actions'"

**Step 3: Write implementation**

```python
# src/esper/leyline/factored_actions.py
"""Factored Action Space for Multi-Slot Control.

The action space is factored into:
- SlotAction: which slot to target
- BlueprintAction: what blueprint to germinate
- BlendAction: which blending algorithm to use
- LifecycleOp: what operation to perform
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SlotAction(IntEnum):
    """Target slot selection."""
    EARLY = 0
    MID = 1
    LATE = 2

    def to_slot_id(self) -> str:
        return ["early", "mid", "late"][self.value]


class BlueprintAction(IntEnum):
    """Blueprint selection for germination."""
    NOOP = 0
    CONV_ENHANCE = 1
    ATTENTION = 2
    NORM = 3
    DEPTHWISE = 4

    def to_blueprint_id(self) -> str | None:
        mapping = {
            0: "noop",
            1: "conv_enhance",
            2: "attention",
            3: "norm",
            4: "depthwise",
        }
        return mapping.get(self.value)


class BlendAction(IntEnum):
    """Blending algorithm selection."""
    LINEAR = 0
    SIGMOID = 1
    GATED = 2

    def to_algorithm_id(self) -> str:
        return ["linear", "sigmoid", "gated"][self.value]


class LifecycleOp(IntEnum):
    """Lifecycle operation."""
    WAIT = 0
    GERMINATE = 1
    ADVANCE = 2
    CULL = 3


@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Composed action from factored components."""
    slot: SlotAction
    blueprint: BlueprintAction
    blend: BlendAction
    op: LifecycleOp

    @property
    def is_germinate(self) -> bool:
        return self.op == LifecycleOp.GERMINATE

    @property
    def is_advance(self) -> bool:
        return self.op == LifecycleOp.ADVANCE

    @property
    def is_cull(self) -> bool:
        return self.op == LifecycleOp.CULL

    @property
    def is_wait(self) -> bool:
        return self.op == LifecycleOp.WAIT

    @property
    def slot_id(self) -> str:
        return self.slot.to_slot_id()

    @property
    def blueprint_id(self) -> str | None:
        return self.blueprint.to_blueprint_id()

    @property
    def blend_algorithm_id(self) -> str:
        return self.blend.to_algorithm_id()

    @classmethod
    def from_indices(
        cls,
        slot_idx: int,
        blueprint_idx: int,
        blend_idx: int,
        op_idx: int,
    ) -> "FactoredAction":
        return cls(
            slot=SlotAction(slot_idx),
            blueprint=BlueprintAction(blueprint_idx),
            blend=BlendAction(blend_idx),
            op=LifecycleOp(op_idx),
        )


# Dimension sizes for policy network
NUM_SLOTS = len(SlotAction)
NUM_BLUEPRINTS = len(BlueprintAction)
NUM_BLENDS = len(BlendAction)
NUM_OPS = len(LifecycleOp)


__all__ = [
    "SlotAction",
    "BlueprintAction",
    "BlendAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_SLOTS",
    "NUM_BLUEPRINTS",
    "NUM_BLENDS",
    "NUM_OPS",
]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/leyline/test_factored_actions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/factored_actions.py tests/leyline/test_factored_actions.py
git commit -m "feat(leyline): add factored action space (slot, blueprint, blend, op)"
```

---

## Task 6: Factored Multi-Head Policy Network

**Files:**
- Create: `src/esper/simic/factored_network.py`
- Test: `tests/simic/test_factored_network.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_factored_network.py
import torch


def test_factored_actor_critic_forward():
    """FactoredActorCritic should output distributions for each head."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(
        state_dim=30,  # Extended obs with per-slot features
        num_slots=3,
        num_blueprints=5,
        num_blends=3,
        num_ops=4,
    )

    obs = torch.randn(4, 30)  # Batch of 4
    dists, value = net(obs)

    assert "slot" in dists
    assert "blueprint" in dists
    assert "blend" in dists
    assert "op" in dists

    # Each should be a Categorical distribution
    assert dists["slot"].probs.shape == (4, 3)
    assert dists["blueprint"].probs.shape == (4, 5)
    assert dists["blend"].probs.shape == (4, 3)
    assert dists["op"].probs.shape == (4, 4)

    assert value.shape == (4,)


def test_factored_actor_critic_sample():
    """Should sample actions from all heads."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)
    actions, log_probs, values = net.get_action_batch(obs)

    assert actions["slot"].shape == (4,)
    assert actions["blueprint"].shape == (4,)
    assert actions["blend"].shape == (4,)
    assert actions["op"].shape == (4,)

    # Log probs should be summed across heads
    assert log_probs.shape == (4,)
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/simic/test_factored_network.py -v`
Expected: FAIL with "No module named 'esper.simic.factored_network'"

**Step 3: Write implementation**

```python
# src/esper/simic/factored_network.py
"""Factored Multi-Head Policy Network for Multi-Slot Control.

The policy outputs separate distributions for each action dimension,
then samples them jointly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical


class FactoredActorCritic(nn.Module):
    """Actor-Critic with factored action heads.

    Instead of one output for all actions, we have separate heads:
    - slot_head: which slot to target
    - blueprint_head: which blueprint to germinate
    - blend_head: which blending algorithm
    - op_head: which lifecycle operation
    """

    def __init__(
        self,
        state_dim: int,
        num_slots: int = 3,
        num_blueprints: int = 5,
        num_blends: int = 3,
        num_ops: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.num_blueprints = num_blueprints
        self.num_blends = num_blends
        self.num_ops = num_ops

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Factored action heads
        head_hidden = hidden_dim // 2
        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_slots),
        )
        self.blueprint_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blueprints),
        )
        self.blend_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blends),
        )
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_ops),
        )

        # Critic head (single value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers
        for head in [self.slot_head, self.blueprint_head, self.blend_head, self.op_head]:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, Categorical], torch.Tensor]:
        """Forward pass returning distributions for each head and value.

        Args:
            state: Observation tensor (batch, state_dim)
            masks: Optional dict of action masks per head

        Returns:
            dists: Dict of Categorical distributions
            value: Value estimates (batch,)
        """
        features = self.shared(state)

        # Get logits from each head
        slot_logits = self.slot_head(features)
        blueprint_logits = self.blueprint_head(features)
        blend_logits = self.blend_head(features)
        op_logits = self.op_head(features)

        # Apply masks if provided (set invalid actions to -inf)
        if masks:
            if "slot" in masks:
                slot_logits = slot_logits.masked_fill(~masks["slot"], float("-inf"))
            if "blueprint" in masks:
                blueprint_logits = blueprint_logits.masked_fill(~masks["blueprint"], float("-inf"))
            if "blend" in masks:
                blend_logits = blend_logits.masked_fill(~masks["blend"], float("-inf"))
            if "op" in masks:
                op_logits = op_logits.masked_fill(~masks["op"], float("-inf"))

        dists = {
            "slot": Categorical(logits=slot_logits),
            "blueprint": Categorical(logits=blueprint_logits),
            "blend": Categorical(logits=blend_logits),
            "op": Categorical(logits=op_logits),
        }

        value = self.critic(features).squeeze(-1)
        return dists, value

    def get_action_batch(
        self,
        states: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample actions from all heads.

        Returns:
            actions: Dict of action indices per head
            log_probs: Sum of log probs across heads
            values: Value estimates
        """
        with torch.no_grad():
            dists, values = self.forward(states, masks)

            actions = {}
            log_probs_list = []

            for key, dist in dists.items():
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
                actions[key] = action
                log_probs_list.append(dist.log_prob(action))

            # Sum log probs across heads (joint probability)
            log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)

            return actions, log_probs, values

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: Sum of log probs across heads
            values: Value estimates
            entropy: Sum of entropies across heads
        """
        dists, values = self.forward(states, masks)

        log_probs_list = []
        entropy_list = []

        for key, dist in dists.items():
            log_probs_list.append(dist.log_prob(actions[key]))
            entropy_list.append(dist.entropy())

        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)

        return log_probs, values, entropy


__all__ = ["FactoredActorCritic"]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_factored_network.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/factored_network.py tests/simic/test_factored_network.py
git commit -m "feat(simic): add FactoredActorCritic with multi-head policy"
```

---

## Task 7: Per-Slot Observation Features

**Files:**
- Modify: `src/esper/simic/features.py`
- Test: `tests/simic/test_features.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_features.py
def test_multislot_features():
    """obs_to_multislot_features should include per-slot state."""
    from esper.simic.features import obs_to_multislot_features

    obs = {
        # Base features
        'epoch': 10,
        'global_step': 100,
        'train_loss': 0.5,
        'val_loss': 0.6,
        'loss_delta': -0.1,
        'train_accuracy': 70.0,
        'val_accuracy': 68.0,
        'accuracy_delta': 0.5,
        'plateau_epochs': 2,
        'best_val_accuracy': 70.0,
        'best_val_loss': 0.5,
        'loss_history_5': [0.6, 0.55, 0.5, 0.52, 0.5],
        'accuracy_history_5': [65.0, 66.0, 67.0, 68.0, 68.0],

        # Per-slot features
        'slots': {
            'early': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
            'mid': {'is_active': True, 'stage': 3, 'alpha': 0.5, 'improvement': 2.5},
            'late': {'is_active': False, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0},
        },
    }

    features = obs_to_multislot_features(obs)

    # Base features (22) + per-slot (3 slots * 4 features) = 34
    assert len(features) == 34

    # Check per-slot features are included
    # After base features, we have slot features
    slot_start = 22
    # early slot: is_active=0, stage=0, alpha=0, improvement=0
    assert features[slot_start:slot_start+4] == [0.0, 0.0, 0.0, 0.0]
    # mid slot: is_active=1, stage=3, alpha=0.5, improvement=2.5
    assert features[slot_start+4:slot_start+8] == [1.0, 3.0, 0.5, 2.5]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_features.py::test_multislot_features -v`
Expected: FAIL with "cannot import name 'obs_to_multislot_features'"

**Step 3: Write implementation**

Add to `src/esper/simic/features.py`:

```python
# Add this constant at top
MULTISLOT_FEATURE_SIZE = 34  # 22 base + 3*4 slot features


def obs_to_multislot_features(obs: dict) -> list[float]:
    """Extract features including per-slot state (34 dims).

    Base features (22 dims) - same as V1 minus legacy seed state:
    - Timing: epoch, global_step (2)
    - Losses: train_loss, val_loss, loss_delta (3)
    - Accuracies: train_accuracy, val_accuracy, accuracy_delta (3)
    - Trends: plateau_epochs, best_val_accuracy, best_val_loss (3)
    - History: loss_history_5 (5), accuracy_history_5 (5)
    - Capacity: total_params (1) [new]

    Per-slot features (4 dims each, 3 slots = 12 dims):
    - is_active, stage, alpha, improvement

    This keeps each slot's local state visible, while still giving Tamiyo
    a single flat observation vector that standard PPO implementations
    can consume without custom architecture changes.

    Args:
        obs: Observation dictionary with 'slots' key

    Returns:
        List of 34 floats
    """
    # Base features (22 dims)
    base = [
        float(obs['epoch']),
        float(obs['global_step']),
        safe(obs['train_loss'], 10.0),
        safe(obs['val_loss'], 10.0),
        safe(obs['loss_delta'], 0.0),
        obs['train_accuracy'],
        obs['val_accuracy'],
        safe(obs['accuracy_delta'], 0.0),
        float(obs['plateau_epochs']),
        obs['best_val_accuracy'],
        safe(obs['best_val_loss'], 10.0),
        *[safe(v, 10.0) for v in obs['loss_history_5']],
        *obs['accuracy_history_5'],
    ]

    # Per-slot features (4 dims per slot, 3 slots)
    slot_features = []
    for slot_id in ['early', 'mid', 'late']:
        slot = obs.get('slots', {}).get(slot_id, {})
        slot_features.extend([
            float(slot.get('is_active', 0)),
            float(slot.get('stage', 0)),
            float(slot.get('alpha', 0.0)),
            float(slot.get('improvement', 0.0)),
        ])

    return base + slot_features
```

Also add to __all__:
```python
__all__ = [
    "safe",
    "obs_to_base_features",
    "obs_to_multislot_features",
    "MULTISLOT_FEATURE_SIZE",
]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_features.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/features.py tests/simic/test_features.py
git commit -m "feat(simic): add obs_to_multislot_features with per-slot state"
```

---

## Task 8: Per-Slot Reward Shaping

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Test: `tests/simic/test_rewards.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_rewards.py
def test_multislot_reward():
    """compute_multislot_reward should shape rewards per-slot."""
    from esper.simic.rewards import compute_multislot_reward
    from esper.leyline.factored_actions import LifecycleOp, SlotAction

    # Germinate in empty slot should get bonus
    reward = compute_multislot_reward(
        op=LifecycleOp.GERMINATE,
        slot=SlotAction.MID,
        acc_delta=0.5,
        val_acc=65.0,
        slot_states={
            "early": None,
            "mid": None,  # Empty slot
            "late": None,
        },
        epoch=5,
        max_epochs=25,
    )

    assert reward > 0, "Germinating in empty slot should be positive"


def test_multislot_reward_germinate_occupied():
    """Germinating in occupied slot should be penalized."""
    from esper.simic.rewards import compute_multislot_reward, SeedInfo
    from esper.leyline.factored_actions import LifecycleOp, SlotAction
    from esper.leyline import SeedStage

    reward = compute_multislot_reward(
        op=LifecycleOp.GERMINATE,
        slot=SlotAction.MID,
        acc_delta=0.5,
        val_acc=65.0,
        slot_states={
            "early": None,
            "mid": SeedInfo(stage=SeedStage.TRAINING.value, improvement_since_stage_start=1.0, epochs_in_stage=5),
            "late": None,
        },
        epoch=5,
        max_epochs=25,
    )

    assert reward < 0, "Germinating in occupied slot should be negative"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_rewards.py::test_multislot_reward -v`
Expected: FAIL with "cannot import name 'compute_multislot_reward'"

**Step 3: Write implementation**

Add to `src/esper/simic/rewards.py`:

```python
def compute_multislot_reward(
    op: "LifecycleOp",
    slot: "SlotAction",
    acc_delta: float,
    val_acc: float,
    slot_states: dict[str, SeedInfo | None],
    epoch: int,
    max_epochs: int,
    config: RewardConfig | None = None,
) -> float:
    """Compute reward for multi-slot action.

    The reward remains primarily global (accuracy deltas are shared),
    but we add light slot-specific shaping so Tamiyo has a clearer
    signal about good/bad interventions per location.

    Args:
        op: Lifecycle operation (WAIT, GERMINATE, ADVANCE, CULL)
        slot: Target slot
        acc_delta: Accuracy improvement this step
        val_acc: Current validation accuracy
        slot_states: Dict mapping slot_id to SeedInfo or None
        epoch: Current epoch
        max_epochs: Maximum epochs
        config: Reward configuration

    Returns:
        Shaped reward value
    """
    from esper.leyline.factored_actions import LifecycleOp, SlotAction

    if config is None:
        config = _DEFAULT_CONFIG

    slot_id = slot.to_slot_id() if hasattr(slot, 'to_slot_id') else str(slot)
    seed_info = slot_states.get(slot_id)

    reward = 0.0

    # Base: accuracy improvement (global)
    reward += acc_delta * config.acc_delta_weight

    # Lifecycle stage rewards (for target slot)
    if seed_info is not None:
        stage = seed_info.stage
        improvement = seed_info.improvement_since_stage_start

        if stage == STAGE_TRAINING:
            reward += config.training_bonus
            if improvement > 0:
                reward += improvement * config.stage_improvement_weight
        elif stage == STAGE_BLENDING:
            reward += config.blending_bonus
        elif stage == STAGE_FOSSILIZED:
            reward += config.fossilized_bonus

    # Operation-specific shaping
    if op == LifecycleOp.GERMINATE:
        if seed_info is None:
            # Bonus for germinating empty slot
            reward += config.germinate_no_seed_bonus
            if epoch < max_epochs * config.early_epoch_fraction:
                reward += config.germinate_early_bonus
        else:
            # Penalty for germinating occupied slot
            reward += config.germinate_with_seed_penalty

    elif op == LifecycleOp.ADVANCE:
        if seed_info is None:
            reward += config.advance_no_seed_penalty
        else:
            stage = seed_info.stage
            improvement = seed_info.improvement_since_stage_start
            if stage == STAGE_TRAINING:
                if improvement > 0:
                    reward += config.advance_good_bonus
                else:
                    reward += config.advance_premature_penalty
            elif stage == STAGE_BLENDING:
                reward += config.advance_blending_bonus

    elif op == LifecycleOp.CULL:
        if seed_info is None:
            reward += config.cull_no_seed_penalty
        else:
            improvement = seed_info.improvement_since_stage_start
            if improvement < config.cull_failing_threshold:
                reward += config.cull_failing_bonus
            elif improvement < 0:
                reward += config.cull_acceptable_bonus
            else:
                reward += config.cull_promising_penalty

    elif op == LifecycleOp.WAIT:
        reward += _wait_shaping(seed_info, acc_delta, config)

    # Terminal bonus
    if epoch == max_epochs:
        reward += val_acc * config.terminal_acc_weight

    return reward
```

Also add to __all__:
```python
__all__ = [
    ...
    "compute_multislot_reward",
]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_rewards.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_rewards.py
git commit -m "feat(simic): add compute_multislot_reward for per-slot shaping"
```

---

## Task 9: Action Masking

**Files:**
- Create: `src/esper/simic/action_masks.py`
- Test: `tests/simic/test_action_masks.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_action_masks.py
import torch


def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not ADVANCE/CULL."""
    from esper.simic.action_masks import compute_action_masks

    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    masks = compute_action_masks(slot_states)

    # All slots should be valid targets
    assert masks["slot"].all()

    # WAIT and GERMINATE should be valid, ADVANCE and CULL should not
    # op order: WAIT=0, GERMINATE=1, ADVANCE=2, CULL=3
    assert masks["op"][0] == True   # WAIT
    assert masks["op"][1] == True   # GERMINATE
    assert masks["op"][2] == False  # ADVANCE (no seed to advance)
    assert masks["op"][3] == False  # CULL (no seed to cull)


def test_compute_action_masks_active_slot():
    """Active slot should allow ADVANCE/CULL, not GERMINATE."""
    from esper.simic.action_masks import compute_action_masks
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    slot_states = {
        "early": None,
        "mid": SeedInfo(stage=SeedStage.TRAINING.value, improvement_since_stage_start=1.0, epochs_in_stage=5),
        "late": None,
    }

    masks = compute_action_masks(slot_states, target_slot="mid")

    # GERMINATE should be invalid for occupied slot
    assert masks["op"][1] == False  # GERMINATE

    # ADVANCE and CULL should be valid
    assert masks["op"][2] == True   # ADVANCE
    assert masks["op"][3] == True   # CULL
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/simic/test_action_masks.py -v`
Expected: FAIL with "No module named 'esper.simic.action_masks'"

**Step 3: Write implementation**

```python
# src/esper/simic/action_masks.py
"""Action Masking for Multi-Slot Control.

Not all action combinations are valid. This module computes masks
for each action head based on current slot states.
"""

from __future__ import annotations

import torch

from esper.simic.rewards import SeedInfo
from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS


def compute_action_masks(
    slot_states: dict[str, SeedInfo | None],
    target_slot: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Args:
        slot_states: Dict mapping slot_id to SeedInfo or None
        target_slot: If provided, compute op mask for this specific slot

    Returns:
        Dict of boolean tensors for each action head
    """
    masks = {}

    # Slot mask: all slots are always valid targets
    masks["slot"] = torch.ones(NUM_SLOTS, dtype=torch.bool)

    # Blueprint mask: all blueprints valid (masked by op later)
    masks["blueprint"] = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)

    # Blend mask: all blends valid (masked by op later)
    masks["blend"] = torch.ones(NUM_BLENDS, dtype=torch.bool)

    # Op mask: depends on slot state
    # WAIT=0 always valid, GERMINATE=1, ADVANCE=2, CULL=3
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool)
    op_mask[0] = True  # WAIT always valid

    if target_slot:
        seed_info = slot_states.get(target_slot)
    else:
        # If no target slot, check if ANY slot has a seed
        seed_info = None
        for s in slot_states.values():
            if s is not None:
                seed_info = s
                break

    if seed_info is None:
        # No active seed: can GERMINATE, cannot ADVANCE/CULL
        op_mask[1] = True   # GERMINATE
        op_mask[2] = False  # ADVANCE
        op_mask[3] = False  # CULL
    else:
        # Has active seed: cannot GERMINATE, can ADVANCE/CULL
        op_mask[1] = False  # GERMINATE
        op_mask[2] = True   # ADVANCE
        op_mask[3] = True   # CULL

    masks["op"] = op_mask

    return masks


def compute_batch_masks(
    batch_slot_states: list[dict[str, SeedInfo | None]],
    target_slots: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        target_slots: Optional list of target slots per env

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head
    """
    batch_size = len(batch_slot_states)

    slot_masks = torch.ones(batch_size, NUM_SLOTS, dtype=torch.bool)
    blueprint_masks = torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool)
    blend_masks = torch.ones(batch_size, NUM_BLENDS, dtype=torch.bool)
    op_masks = torch.zeros(batch_size, NUM_OPS, dtype=torch.bool)
    op_masks[:, 0] = True  # WAIT always valid

    for i, slot_states in enumerate(batch_slot_states):
        target_slot = target_slots[i] if target_slots else None

        if target_slot:
            seed_info = slot_states.get(target_slot)
        else:
            seed_info = next((s for s in slot_states.values() if s is not None), None)

        if seed_info is None:
            op_masks[i, 1] = True   # GERMINATE
        else:
            op_masks[i, 2] = True   # ADVANCE
            op_masks[i, 3] = True   # CULL

    return {
        "slot": slot_masks,
        "blueprint": blueprint_masks,
        "blend": blend_masks,
        "op": op_masks,
    }


__all__ = ["compute_action_masks", "compute_batch_masks"]
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_action_masks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "feat(simic): add action masking for multi-slot control"
```

---

## Task 10: Integration Test - Multi-Slot Training Step

**Files:**
- Test: `tests/integration/test_multislot_training.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_multislot_training.py
import torch


def test_multislot_forward_backward():
    """Full forward-backward pass with multi-slot model."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.features import MULTISLOT_FEATURE_SIZE

    # Create multi-slot model
    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Create factored policy
    policy = FactoredActorCritic(
        state_dim=MULTISLOT_FEATURE_SIZE,
        num_slots=3,
        num_blueprints=5,
        num_blends=3,
        num_ops=4,
    )

    # Germinate a seed in mid slot
    model.germinate_seed("conv_enhance", "test_seed", slot="mid")

    # Forward pass through model
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

    # Policy forward
    obs = torch.randn(2, MULTISLOT_FEATURE_SIZE)
    actions, log_probs, values = policy.get_action_batch(obs)

    assert "slot" in actions
    assert "op" in actions
    assert log_probs.shape == (2,)
    assert values.shape == (2,)


def test_multislot_germinate_cull_cycle():
    """Should be able to germinate and cull in different slots."""
    from esper.kasmina.host import HostCNN, MorphogeneticModel

    host = HostCNN()
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Germinate in early
    model.germinate_seed("attention", "seed_early", slot="early")
    assert model.has_active_seed_in_slot("early")
    assert not model.has_active_seed_in_slot("mid")

    # Germinate in late
    model.germinate_seed("norm", "seed_late", slot="late")
    assert model.has_active_seed_in_slot("early")
    assert model.has_active_seed_in_slot("late")

    # Cull early
    model.cull_seed(slot="early")
    assert not model.has_active_seed_in_slot("early")
    assert model.has_active_seed_in_slot("late")

    # Model still works
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
```

**Step 2: Run test**

Run: `PYTHONPATH=src pytest tests/integration/test_multislot_training.py -v`
Expected: PASS (after all prior tasks complete)

**Step 3: Commit**

```bash
git add tests/integration/test_multislot_training.py
git commit -m "test: add multi-slot integration tests"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | NoopSeed blueprint | `kasmina/blueprints.py` |
| 2 | Blending library | `kasmina/blending.py` |
| 3 | HostCNN segments | `kasmina/host.py` |
| 4 | Multi-slot MorphogeneticModel | `kasmina/host.py` |
| 5 | Factored actions | `leyline/factored_actions.py` |
| 6 | Factored network | `simic/factored_network.py` |
| 7 | Multi-slot features | `simic/features.py` |
| 8 | Multi-slot rewards | `simic/rewards.py` |
| 9 | Action masking | `simic/action_masks.py` |
| 10 | Integration tests | `tests/integration/` |

After completing these tasks, the vectorized training loop (`simic/vectorized.py`) needs to be updated to use the new factored action space. That's a larger refactor that should be done as a follow-up.
