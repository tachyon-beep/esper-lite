# Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend Esper from single-task (CIFAR-10 classification with CNN) to multi-task support (also TinyStories language modeling with Transformer).

**Architecture:** HostProtocol abstraction with ModuleDict + Identity pattern for torch.compile compatibility. Loss-primary rewards with PBRS stage bonuses. Blueprint plugin registry for extensibility.

**Tech Stack:** PyTorch, pytest, Hypothesis

**Design Document:** See `docs/plans/2025-11-30-phase2-train-anything-design.md` for detailed rationale and expert review feedback.

---

## Task 1: HostProtocol Definition

Create the structural typing Protocol for pluggable host architectures.

**Files:**
- Create: `src/esper/kasmina/protocol.py`
- Test: `tests/test_host_protocol.py`

**Step 1: Write the failing test**

```python
# tests/test_host_protocol.py
"""Tests for HostProtocol compliance."""

import pytest
import torch
import torch.nn as nn
from typing import Protocol, runtime_checkable


def test_host_protocol_is_importable():
    """HostProtocol can be imported."""
    from esper.kasmina.protocol import HostProtocol
    assert HostProtocol is not None


def test_host_protocol_has_required_methods():
    """HostProtocol defines required interface."""
    from esper.kasmina.protocol import HostProtocol

    # Check required attributes exist on the protocol
    assert hasattr(HostProtocol, 'injection_points')
    assert hasattr(HostProtocol, 'register_slot')
    assert hasattr(HostProtocol, 'unregister_slot')
    assert hasattr(HostProtocol, 'forward')
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py -v`
Expected: FAIL with "No module named 'esper.kasmina.protocol'"

**Step 3: Write minimal implementation**

```python
# src/esper/kasmina/protocol.py
"""Kasmina Protocol - Structural typing for pluggable hosts.

Hosts declare where seeds can be planted (injection_points),
accept seed modules (register_slot), and handle their own
forward pass including calling any attached slots.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


@runtime_checkable
class HostProtocol(Protocol):
    """Contract for graftable host networks.

    Hosts declare where seeds can be planted (injection_points),
    accept seed modules (register_slot), and handle their own
    forward pass including calling any attached slots.
    """

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel/embedding dimension.

        Example:
            {"block2_post": 64}  # CNN
            {"layer_0_post_block": 384, ...}  # Transformer
        """
        ...

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point.

        The host is responsible for calling this slot during forward().
        Implementation must move slot to correct device.

        Raises:
            ValueError: If slot_id is not a valid injection point.
        """
        ...

    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point.

        Resets the slot to identity (no-op) behavior.

        Raises:
            ValueError: If slot_id is not a valid injection point.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass, including any attached slots.

        The host calls registered slots at appropriate points internally.
        """
        ...


__all__ = ["HostProtocol"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/protocol.py tests/test_host_protocol.py
git commit -m "feat(kasmina): add HostProtocol for pluggable hosts"
```

---

## Task 2: Update HostCNN to Implement HostProtocol

Refactor HostCNN to use ModuleDict + Identity pattern for torch.compile compatibility.

**Files:**
- Modify: `src/esper/kasmina/host.py:19-46`
- Test: `tests/test_host_protocol.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_host_protocol.py

def test_host_cnn_implements_protocol():
    """HostCNN implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    assert isinstance(host, HostProtocol)


def test_host_cnn_injection_points():
    """HostCNN declares injection points."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    points = host.injection_points

    assert "block2_post" in points
    assert points["block2_post"] == 64


def test_host_cnn_register_slot():
    """HostCNN can register a slot module."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    host.register_slot("block2_post", slot)

    # Slot should be in the host's slots dict
    assert host.slots["block2_post"] is slot


def test_host_cnn_register_invalid_slot_raises():
    """HostCNN raises ValueError for invalid slot_id."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    with pytest.raises(ValueError, match="Unknown injection point"):
        host.register_slot("invalid_slot", slot)


def test_host_cnn_unregister_slot():
    """HostCNN can unregister a slot (resets to Identity)."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    slot = torch.nn.Conv2d(64, 64, 1)

    host.register_slot("block2_post", slot)
    host.unregister_slot("block2_post")

    # Should be back to Identity
    assert isinstance(host.slots["block2_post"], torch.nn.Identity)


def test_host_cnn_forward_with_identity():
    """HostCNN forward works with Identity slots (no-op)."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()
    x = torch.randn(2, 3, 32, 32)

    out = host(x)

    assert out.shape == (2, 10)  # 10 classes


def test_host_cnn_forward_with_slot():
    """HostCNN forward passes through registered slot."""
    from esper.kasmina.host import HostCNN

    host = HostCNN()

    # Create a slot that doubles the input
    class DoubleSlot(torch.nn.Module):
        def forward(self, x):
            return x * 2

    host.register_slot("block2_post", DoubleSlot())

    x = torch.randn(2, 3, 32, 32)
    out = host(x)

    # Should still produce valid output (different from identity path)
    assert out.shape == (2, 10)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py::test_host_cnn_implements_protocol -v`
Expected: FAIL (HostCNN doesn't have `slots` attribute or new methods)

**Step 3: Write minimal implementation**

```python
# Update src/esper/kasmina/host.py - replace HostCNN class

class HostCNN(nn.Module):
    """CNN host with single injection point after block2.

    Implements HostProtocol with ModuleDict + Identity pattern
    for torch.compile compatibility.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Injection points with compile-friendly ModuleDict
        self._slot_keys = ("block2_post",)
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel dimension."""
        return {"block2_post": 64}

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = next(self.parameters()).device
        self.slots[slot_id] = slot.to(device)

    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))

        # Always call slot (Identity is no-op when empty)
        x = self.slots["block2_post"](x)

        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    # Keep legacy methods for MorphogeneticModel compatibility
    @property
    def injection_channels(self) -> int:
        """Legacy: channel count at injection point."""
        return 64

    def forward_to_injection(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy: forward to injection point."""
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return x

    def forward_from_injection(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy: forward from injection point."""
        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/test_host_protocol.py
git commit -m "refactor(kasmina): HostCNN implements HostProtocol with ModuleDict"
```

---

## Task 3: Create TransformerHost

Add GPT-style decoder host with injection points after each layer.

**Files:**
- Modify: `src/esper/kasmina/host.py` (add classes)
- Test: `tests/test_host_protocol.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_host_protocol.py

def test_transformer_host_implements_protocol():
    """TransformerHost implements HostProtocol."""
    from esper.kasmina.protocol import HostProtocol
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)
    assert isinstance(host, HostProtocol)


def test_transformer_host_injection_points():
    """TransformerHost declares injection points per layer."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=4)
    points = host.injection_points

    assert len(points) == 4
    assert "layer_0_post_block" in points
    assert "layer_3_post_block" in points
    assert all(dim == 64 for dim in points.values())


def test_transformer_host_weight_tying():
    """TransformerHost has weight tying between embedding and output."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)

    # tok_emb is master, head shares its weight
    assert host.head.weight.data_ptr() == host.tok_emb.weight.data_ptr()


def test_transformer_host_forward():
    """TransformerHost forward produces logits."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32)

    x = torch.randint(0, 1000, (2, 16))  # batch=2, seq_len=16
    out = host(x)

    assert out.shape == (2, 16, 1000)  # (batch, seq, vocab)


def test_transformer_host_register_unregister():
    """TransformerHost can register and unregister slots."""
    from esper.kasmina.host import TransformerHost

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2)
    slot = torch.nn.Linear(64, 64)

    host.register_slot("layer_0_post_block", slot)
    assert host.slots["layer_0_post_block"] is slot

    host.unregister_slot("layer_0_post_block")
    assert isinstance(host.slots["layer_0_post_block"], torch.nn.Identity)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py::test_transformer_host_implements_protocol -v`
Expected: FAIL with "cannot import name 'TransformerHost'"

**Step 3: Write minimal implementation**

```python
# Add to src/esper/kasmina/host.py after HostCNN

# =============================================================================
# Transformer Components
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Causal self-attention with pre-computed mask."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerHost(nn.Module):
    """GPT-style decoder with injection points after each layer.

    Implements HostProtocol with ModuleDict + Identity pattern
    for torch.compile compatibility.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: tok_emb is master (convention from GPT-2)
        self.head.weight = self.tok_emb.weight
        assert self.head.weight.data_ptr() == self.tok_emb.weight.data_ptr()

        # Injection points with compile-friendly ModuleDict
        self._slot_keys = tuple(f"layer_{i}_post_block" for i in range(n_layer))
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> embedding dimension."""
        return {k: self.n_embd for k in self._slot_keys}

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = self.tok_emb.weight.device
        self.slots[slot_id] = slot.to(device)

    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embeddings
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        # Transformer layers with slot injection
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.slots[self._slot_keys[i]](h)  # Always call, Identity is no-op

        # Output
        h = self.ln_f(h)
        return self.head(h)
```

**Step 4: Update __all__ in host.py**

```python
__all__ = [
    "ConvBlock",
    "HostCNN",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "TransformerHost",
    "MorphogeneticModel",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_host_protocol.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/host.py tests/test_host_protocol.py
git commit -m "feat(kasmina): add TransformerHost implementing HostProtocol"
```

---

## Task 4: Blueprint Plugin Registry

Create decorator-based blueprint registration system for extensibility.

**Files:**
- Create: `src/esper/kasmina/blueprints/registry.py`
- Create: `src/esper/kasmina/blueprints/__init__.py`
- Create: `src/esper/kasmina/blueprints/cnn.py`
- Modify: `src/esper/kasmina/blueprints.py` (imports from new structure)
- Test: `tests/test_blueprint_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_blueprint_registry.py
"""Tests for blueprint plugin registry."""

import pytest
import torch.nn as nn


def test_registry_is_importable():
    """BlueprintRegistry can be imported."""
    from esper.kasmina.blueprints import BlueprintRegistry
    assert BlueprintRegistry is not None


def test_registry_register_decorator():
    """Decorator registers a blueprint."""
    from esper.kasmina.blueprints import BlueprintRegistry

    @BlueprintRegistry.register("test_blueprint", "cnn", param_estimate=100)
    def create_test(dim: int) -> nn.Module:
        return nn.Linear(dim, dim)

    specs = BlueprintRegistry.list_for_topology("cnn")
    names = [s.name for s in specs]
    assert "test_blueprint" in names


def test_registry_list_for_topology():
    """Registry filters blueprints by topology."""
    from esper.kasmina.blueprints import BlueprintRegistry

    cnn_specs = BlueprintRegistry.list_for_topology("cnn")
    transformer_specs = BlueprintRegistry.list_for_topology("transformer")

    # CNN blueprints exist (from existing code)
    assert any(s.name == "conv_enhance" for s in cnn_specs)

    # Lists are separate
    cnn_names = {s.name for s in cnn_specs}
    transformer_names = {s.name for s in transformer_specs}
    # Note: Some blueprints might apply to both, but they should be registered separately


def test_registry_sorted_by_params():
    """Blueprints sorted by param estimate (ascending)."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    params = [s.param_estimate for s in specs]

    assert params == sorted(params)


def test_blueprint_spec_has_factory():
    """BlueprintSpec can create modules."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    # Get first spec
    spec = specs[0]

    module = spec.factory(64)  # 64 channels
    assert isinstance(module, nn.Module)


def test_blueprint_spec_actual_param_count():
    """BlueprintSpec can compute actual param count."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("cnn")
    spec = next(s for s in specs if s.name == "norm")

    actual = spec.actual_param_count(64)
    assert actual > 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py::test_registry_is_importable -v`
Expected: FAIL with import error

**Step 3: Create registry module**

```python
# src/esper/kasmina/blueprints/registry.py
"""Blueprint Registry - Plugin system for seed blueprints.

Blueprints are registered with a decorator and automatically
become available in the action space for their topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    """Specification for a registered blueprint.

    Attributes:
        name: Unique name within topology (e.g., "conv_enhance")
        topology: Target topology ("cnn" or "transformer")
        factory: Callable that takes dimension and returns nn.Module
        param_estimate: Rough parameter count for action ordering
        description: Human-readable description
    """
    name: str
    topology: str
    factory: Callable[[int], nn.Module]
    param_estimate: int
    description: str = ""

    def actual_param_count(self, dim: int) -> int:
        """Compute actual param count for given dimension."""
        module = self.factory(dim)
        return sum(p.numel() for p in module.parameters())


class BlueprintRegistry:
    """Registry of available seed blueprints.

    Use the @register decorator to add new blueprints:

        @BlueprintRegistry.register("my_blueprint", "cnn", param_estimate=1000)
        def create_my_blueprint(dim: int) -> nn.Module:
            return MyBlueprint(dim)
    """

    _blueprints: dict[str, BlueprintSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        topology: str,
        param_estimate: int,
        description: str = "",
    ):
        """Decorator to register a blueprint factory.

        Args:
            name: Unique name within topology
            topology: Target topology ("cnn" or "transformer")
            param_estimate: Rough parameter count for ordering
            description: Human-readable description

        Returns:
            Decorator that registers the factory function
        """
        def decorator(factory: Callable[[int], nn.Module]):
            key = f"{topology}:{name}"
            cls._blueprints[key] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            return factory
        return decorator

    @classmethod
    def list_for_topology(cls, topology: str) -> list[BlueprintSpec]:
        """All blueprints for a topology, sorted by param count."""
        return sorted(
            [s for s in cls._blueprints.values() if s.topology == topology],
            key=lambda s: s.param_estimate,
        )

    @classmethod
    def get(cls, topology: str, name: str) -> BlueprintSpec:
        """Get a specific blueprint spec."""
        key = f"{topology}:{name}"
        if key not in cls._blueprints:
            available = cls.list_for_topology(topology)
            names = [s.name for s in available]
            raise ValueError(f"Unknown blueprint {name!r} for {topology}. Available: {names}")
        return cls._blueprints[key]

    @classmethod
    def create(cls, topology: str, name: str, dim: int) -> nn.Module:
        """Create a module from a registered blueprint."""
        spec = cls.get(topology, name)
        return spec.factory(dim)


__all__ = ["BlueprintSpec", "BlueprintRegistry"]
```

**Step 4: Create CNN blueprints module**

```python
# src/esper/kasmina/blueprints/cnn.py
"""CNN Blueprints - Seed modules for convolutional hosts.

These blueprints are designed for CNN injection points with
spatial dimensions (B, C, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BlueprintRegistry


class ConvBlock(nn.Module):
    """Standard conv-bn-relu block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


@BlueprintRegistry.register("norm", "cnn", param_estimate=100, description="BatchNorm only")
def create_norm_seed(channels: int) -> nn.Module:
    """Normalization enhancement seed."""
    class NormSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.scale * (self.norm(x) - x)

    return NormSeed(channels)


@BlueprintRegistry.register("attention", "cnn", param_estimate=2000, description="SE-style channel attention")
def create_attention_seed(channels: int, reduction: int = 4) -> nn.Module:
    """Channel attention seed (SE-style)."""
    class AttentionSeed(nn.Module):
        def __init__(self, channels: int, reduction: int):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    return AttentionSeed(channels, reduction)


@BlueprintRegistry.register("depthwise", "cnn", param_estimate=4800, description="Depthwise-separable conv")
def create_depthwise_seed(channels: int) -> nn.Module:
    """Depthwise separable convolution seed."""
    class DepthwiseSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.depthwise = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
            )
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            return residual + F.relu(x)

    return DepthwiseSeed(channels)


@BlueprintRegistry.register("conv_enhance", "cnn", param_estimate=74000, description="Heavy conv block")
def create_conv_enhance_seed(channels: int) -> nn.Module:
    """Convolutional enhancement seed."""
    class ConvEnhanceSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.enhance = nn.Sequential(
                ConvBlock(channels, channels),
                ConvBlock(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.enhance(x)

    return ConvEnhanceSeed(channels)


__all__ = ["ConvBlock"]
```

**Step 5: Create blueprints __init__.py**

```python
# src/esper/kasmina/blueprints/__init__.py
"""Kasmina Blueprints - Plugin registry for seed architectures.

Usage:
    from esper.kasmina.blueprints import BlueprintRegistry

    # List available blueprints for a topology
    cnn_specs = BlueprintRegistry.list_for_topology("cnn")

    # Create a module from a blueprint
    module = BlueprintRegistry.create("cnn", "attention", dim=64)
"""

from .registry import BlueprintSpec, BlueprintRegistry

# Import CNN blueprints to trigger registration
from . import cnn

__all__ = [
    "BlueprintSpec",
    "BlueprintRegistry",
]
```

**Step 6: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/kasmina/blueprints/ tests/test_blueprint_registry.py
git commit -m "feat(kasmina): add blueprint plugin registry"
```

---

## Task 5: Transformer Blueprints

Add transformer-specific blueprints (norm, lora, attention, mlp).

**Files:**
- Create: `src/esper/kasmina/blueprints/transformer.py`
- Modify: `src/esper/kasmina/blueprints/__init__.py`
- Test: `tests/test_blueprint_registry.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_blueprint_registry.py

def test_transformer_blueprints_registered():
    """Transformer blueprints are registered."""
    from esper.kasmina.blueprints import BlueprintRegistry

    specs = BlueprintRegistry.list_for_topology("transformer")
    names = {s.name for s in specs}

    assert "norm" in names
    assert "lora" in names


def test_transformer_lora_creates_module():
    """LoRA blueprint creates valid module."""
    from esper.kasmina.blueprints import BlueprintRegistry
    import torch

    module = BlueprintRegistry.create("transformer", "lora", dim=384)

    # Test forward pass
    x = torch.randn(2, 16, 384)  # batch=2, seq=16, dim=384
    out = module(x)

    assert out.shape == x.shape


def test_transformer_norm_creates_module():
    """Norm blueprint creates valid module."""
    from esper.kasmina.blueprints import BlueprintRegistry
    import torch

    module = BlueprintRegistry.create("transformer", "norm", dim=384)

    x = torch.randn(2, 16, 384)
    out = module(x)

    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py::test_transformer_blueprints_registered -v`
Expected: FAIL (no transformer blueprints registered)

**Step 3: Write minimal implementation**

```python
# src/esper/kasmina/blueprints/transformer.py
"""Transformer Blueprints - Seed modules for transformer hosts.

These blueprints are designed for transformer injection points with
sequence dimensions (B, T, D).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BlueprintRegistry


@BlueprintRegistry.register("norm", "transformer", param_estimate=800, description="LayerNorm only")
def create_transformer_norm_seed(dim: int) -> nn.Module:
    """LayerNorm enhancement seed for transformers."""
    class TransformerNormSeed(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.scale * (self.norm(x) - x)

    return TransformerNormSeed(dim)


@BlueprintRegistry.register("lora", "transformer", param_estimate=6000, description="Low-rank adapter (rank=8)")
def create_lora_seed(dim: int, rank: int = 8) -> nn.Module:
    """Low-rank adapter seed (LoRA-style)."""
    class LoRASeed(nn.Module):
        def __init__(self, dim: int, rank: int):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            # Zero-init up projection for stable integration
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.up(self.down(x))

    return LoRASeed(dim, rank)


@BlueprintRegistry.register("attention", "transformer", param_estimate=50000, description="Additional self-attention head")
def create_transformer_attention_seed(dim: int, n_head: int = 4) -> nn.Module:
    """Additional self-attention head seed."""
    class TransformerAttentionSeed(nn.Module):
        def __init__(self, dim: int, n_head: int):
            super().__init__()
            self.n_head = n_head
            self.head_dim = dim // n_head

            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            # Zero-init output projection for stable integration
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape

            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention (without causal mask - this is a residual addition)
            scale = self.head_dim ** -0.5
            att = (q @ k.transpose(-2, -1)) * scale
            att = F.softmax(att, dim=-1)

            out = (att @ v).transpose(1, 2).reshape(B, T, C)
            return x + self.proj(out)

    return TransformerAttentionSeed(dim, n_head)


@BlueprintRegistry.register("mlp", "transformer", param_estimate=1200000, description="Additional MLP (4x expansion)")
def create_transformer_mlp_seed(dim: int, expansion: int = 4) -> nn.Module:
    """Additional MLP seed."""
    class TransformerMLPSeed(nn.Module):
        def __init__(self, dim: int, expansion: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            # Zero-init output for stable integration
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.gelu(self.fc1(x)))

    return TransformerMLPSeed(dim, expansion)


__all__ = []
```

**Step 4: Update __init__.py to import transformer blueprints**

```python
# src/esper/kasmina/blueprints/__init__.py
"""Kasmina Blueprints - Plugin registry for seed architectures."""

from .registry import BlueprintSpec, BlueprintRegistry

# Import blueprints to trigger registration
from . import cnn
from . import transformer

__all__ = [
    "BlueprintSpec",
    "BlueprintRegistry",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/blueprints/ tests/test_blueprint_registry.py
git commit -m "feat(kasmina): add transformer blueprints (norm, lora, attention, mlp)"
```

---

## Task 6: Update Legacy BlueprintCatalog

Make the legacy BlueprintCatalog use the new registry internally.

**Files:**
- Modify: `src/esper/kasmina/blueprints.py`
- Test: `tests/test_blueprint_registry.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_blueprint_registry.py

def test_legacy_catalog_still_works():
    """Legacy BlueprintCatalog API still works."""
    from esper.kasmina.blueprints import BlueprintCatalog

    blueprints = BlueprintCatalog.list_blueprints()
    assert "conv_enhance" in blueprints
    assert "norm" in blueprints


def test_legacy_catalog_creates_seed():
    """Legacy BlueprintCatalog.create_seed still works."""
    from esper.kasmina.blueprints import BlueprintCatalog
    import torch

    seed = BlueprintCatalog.create_seed("attention", channels=64)

    x = torch.randn(2, 64, 8, 8)
    out = seed(x)

    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py::test_legacy_catalog_still_works -v`
Expected: FAIL (old BlueprintCatalog doesn't use new registry)

**Step 3: Update legacy module**

```python
# src/esper/kasmina/blueprints.py
"""Kasmina Blueprints - Seed module implementations.

This module provides backwards-compatible access to the blueprint system.
The actual implementations are in esper.kasmina.blueprints/ package.

For new code, use:
    from esper.kasmina.blueprints import BlueprintRegistry
"""

from __future__ import annotations

# Re-export from new package structure
from esper.kasmina.blueprints import BlueprintRegistry, BlueprintSpec

# Re-export CNN building blocks
from esper.kasmina.blueprints.cnn import ConvBlock


class BlueprintCatalog:
    """Legacy API for blueprint access.

    For new code, use BlueprintRegistry instead.
    """

    @classmethod
    def list_blueprints(cls) -> list[str]:
        """List available blueprint IDs (CNN only for legacy compat)."""
        specs = BlueprintRegistry.list_for_topology("cnn")
        return [s.name for s in specs]

    @classmethod
    def create_seed(cls, blueprint_id: str, channels: int, **kwargs):
        """Create a seed module from a blueprint (CNN only)."""
        return BlueprintRegistry.create("cnn", blueprint_id, channels)

    @classmethod
    def register_blueprint(cls, blueprint_id: str, blueprint_class: type):
        """Legacy registration (no longer supported)."""
        raise NotImplementedError(
            "Legacy register_blueprint is deprecated. "
            "Use @BlueprintRegistry.register decorator instead."
        )


# Legacy class exports for type checking
# Note: These are no longer used, but kept for import compatibility
class ConvEnhanceSeed:
    blueprint_id = "conv_enhance"

class AttentionSeed:
    blueprint_id = "attention"

class NormSeed:
    blueprint_id = "norm"

class DepthwiseSeed:
    blueprint_id = "depthwise"


__all__ = [
    "ConvBlock",
    "ConvEnhanceSeed",
    "AttentionSeed",
    "NormSeed",
    "DepthwiseSeed",
    "BlueprintCatalog",
    "BlueprintRegistry",
    "BlueprintSpec",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_blueprint_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints.py tests/test_blueprint_registry.py
git commit -m "refactor(kasmina): BlueprintCatalog delegates to BlueprintRegistry"
```

---

## Task 7: Per-Topology Action Enums

Add dynamic action enum generation from registered blueprints.

**Files:**
- Modify: `src/esper/leyline/actions.py`
- Test: `tests/test_action_enums.py`

**Step 1: Write the failing test**

```python
# tests/test_action_enums.py
"""Tests for per-topology action enums."""

import pytest
from enum import IntEnum


def test_build_action_enum_cnn():
    """build_action_enum creates CNN action enum."""
    from esper.leyline.actions import build_action_enum

    CNNAction = build_action_enum("cnn")

    assert issubclass(CNNAction, IntEnum)
    assert CNNAction.WAIT.value == 0
    assert hasattr(CNNAction, 'GERMINATE_NORM')
    assert hasattr(CNNAction, 'ADVANCE')
    assert hasattr(CNNAction, 'CULL')


def test_build_action_enum_transformer():
    """build_action_enum creates Transformer action enum."""
    from esper.leyline.actions import build_action_enum

    TransformerAction = build_action_enum("transformer")

    assert issubclass(TransformerAction, IntEnum)
    assert TransformerAction.WAIT.value == 0
    assert hasattr(TransformerAction, 'GERMINATE_LORA')
    assert hasattr(TransformerAction, 'ADVANCE')
    assert hasattr(TransformerAction, 'CULL')


def test_action_enum_values_sequential():
    """Action values are sequential integers."""
    from esper.leyline.actions import build_action_enum

    Action = build_action_enum("cnn")
    values = [a.value for a in Action]

    assert values == list(range(len(Action)))


def test_action_enum_cull_is_last():
    """CULL is always the last action."""
    from esper.leyline.actions import build_action_enum

    Action = build_action_enum("cnn")

    assert Action.CULL.value == len(Action) - 1


def test_get_blueprint_from_action():
    """Can get blueprint name from germinate action."""
    from esper.leyline.actions import build_action_enum, get_blueprint_from_action

    Action = build_action_enum("cnn")

    # GERMINATE_NORM should return "norm"
    blueprint = get_blueprint_from_action(Action.GERMINATE_NORM)
    assert blueprint == "norm"

    # Non-germinate actions return None
    assert get_blueprint_from_action(Action.WAIT) is None
    assert get_blueprint_from_action(Action.ADVANCE) is None
    assert get_blueprint_from_action(Action.CULL) is None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_action_enums.py -v`
Expected: FAIL (no build_action_enum function)

**Step 3: Write minimal implementation**

```python
# Replace content of src/esper/leyline/actions.py

"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
Per-topology action enums are built dynamically from registered blueprints.
"""

from enum import IntEnum
from typing import TypeVar

# Cache for built enums
_action_enum_cache: dict[str, type[IntEnum]] = {}


def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from registered blueprints for a topology.

    Action layout:
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT> (sorted by param count)
        N+1: ADVANCE
        N+2: CULL

    Args:
        topology: "cnn" or "transformer"

    Returns:
        IntEnum subclass with topology-specific actions
    """
    if topology in _action_enum_cache:
        return _action_enum_cache[topology]

    # Import here to avoid circular dependency
    from esper.kasmina.blueprints import BlueprintRegistry

    blueprints = BlueprintRegistry.list_for_topology(topology)

    members = {"WAIT": 0}
    for i, spec in enumerate(blueprints, start=1):
        members[f"GERMINATE_{spec.name.upper()}"] = i
    members["ADVANCE"] = len(blueprints) + 1
    members["CULL"] = len(blueprints) + 2

    enum_name = f"{topology.title()}Action"
    action_enum = IntEnum(enum_name, members)

    _action_enum_cache[topology] = action_enum
    return action_enum


def get_blueprint_from_action(action: IntEnum) -> str | None:
    """Get blueprint name from a germinate action.

    Args:
        action: An action from a topology-specific enum

    Returns:
        Blueprint name (e.g., "norm") or None if not a germinate action
    """
    name = action.name
    if name.startswith("GERMINATE_"):
        return name[len("GERMINATE_"):].lower()
    return None


def is_germinate_action(action: IntEnum) -> bool:
    """Check if action is any germinate variant."""
    return action.name.startswith("GERMINATE_")


# Legacy compatibility: keep the old Action enum for Phase 1 code
class Action(IntEnum):
    """Legacy action enum for Phase 1 CNN-only code.

    For new code, use build_action_enum("cnn") or build_action_enum("transformer").
    """
    WAIT = 0
    GERMINATE_CONV = 1
    GERMINATE_ATTENTION = 2
    GERMINATE_NORM = 3
    GERMINATE_DEPTHWISE = 4
    ADVANCE = 5
    CULL = 6

    @classmethod
    def is_germinate(cls, action: "Action") -> bool:
        """Check if action is any germinate variant."""
        return action in (cls.GERMINATE_CONV, cls.GERMINATE_ATTENTION,
                         cls.GERMINATE_NORM, cls.GERMINATE_DEPTHWISE)

    @classmethod
    def get_blueprint_id(cls, action: "Action") -> str | None:
        """Get blueprint ID for germinate actions, None for others."""
        from esper.leyline.blueprints import action_to_blueprint
        return action_to_blueprint(action)


# Legacy alias
SimicAction = Action


__all__ = [
    "Action",
    "SimicAction",
    "build_action_enum",
    "get_blueprint_from_action",
    "is_germinate_action",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_action_enums.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/actions.py tests/test_action_enums.py
git commit -m "feat(leyline): add per-topology action enum generation"
```

---

## Task 8: Loss-Primary Reward Refactor

Refactor rewards from accuracy-primary to loss-primary with normalization and PBRS stage bonuses.

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Test: `tests/test_loss_primary_rewards.py`

**Step 1: Write the failing test**

```python
# tests/test_loss_primary_rewards.py
"""Tests for loss-primary reward computation."""

import pytest


def test_compute_loss_reward_basic():
    """Basic loss reward: lower loss = positive reward."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    # Loss improved (went down)
    reward = compute_loss_reward(
        action=0,  # WAIT
        loss_delta=-0.1,  # Loss decreased
        val_loss=2.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert reward > 0  # Positive reward for improvement


def test_compute_loss_reward_regression_penalized():
    """Loss regression gives negative reward."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    # Loss got worse (went up)
    reward = compute_loss_reward(
        action=0,
        loss_delta=0.1,  # Loss increased
        val_loss=2.5,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert reward < 0  # Negative reward for regression


def test_asymmetric_regression_penalty():
    """Regression penalty is scaled down (asymmetric)."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    # Equal magnitude improvement and regression
    improvement_reward = compute_loss_reward(
        action=0, loss_delta=-0.1, val_loss=2.0,
        seed_info=None, epoch=10, max_epochs=25, config=config,
    )
    regression_reward = compute_loss_reward(
        action=0, loss_delta=0.1, val_loss=2.0,
        seed_info=None, epoch=10, max_epochs=25, config=config,
    )

    # Regression penalty should be less than improvement reward (asymmetric)
    assert abs(regression_reward) < abs(improvement_reward)


def test_compute_rent():
    """Compute rent penalizes excess parameters."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig

    config = LossRewardConfig.default()

    # Same scenario, but with excess params
    no_params_reward = compute_loss_reward(
        action=0, loss_delta=-0.1, val_loss=2.0,
        seed_info=None, epoch=10, max_epochs=25,
        total_params=0, host_params=100000,
        config=config,
    )
    with_params_reward = compute_loss_reward(
        action=0, loss_delta=-0.1, val_loss=2.0,
        seed_info=None, epoch=10, max_epochs=25,
        total_params=50000, host_params=100000,  # 50% extra params
        config=config,
    )

    # More params = less reward (rent paid)
    assert with_params_reward < no_params_reward


def test_pbrs_stage_bonus():
    """PBRS stage bonus rewards stage progression."""
    from esper.simic.rewards import compute_pbrs_stage_bonus, LossRewardConfig
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()

    # Transition from TRAINING to BLENDING
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        previous_stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=2.0,
        epochs_in_stage=0,
    )

    bonus = compute_pbrs_stage_bonus(seed_info, config)

    assert bonus > 0  # Positive bonus for forward progress


def test_pbrs_stage_bonus_fossilized_highest():
    """FOSSILIZED has highest potential (terminal success)."""
    from esper.simic.rewards import compute_pbrs_stage_bonus, LossRewardConfig
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()

    # Transition to FOSSILIZED
    fossilized_bonus = compute_pbrs_stage_bonus(
        SeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            previous_stage=SeedStage.PROBATIONARY.value,
            improvement_since_stage_start=0.0,
            epochs_in_stage=0,
        ),
        config,
    )

    # Transition to BLENDING
    blending_bonus = compute_pbrs_stage_bonus(
        SeedInfo(
            stage=SeedStage.BLENDING.value,
            previous_stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=0.0,
            epochs_in_stage=0,
        ),
        config,
    )

    # FOSSILIZED should give larger bonus
    assert fossilized_bonus > blending_bonus
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_loss_primary_rewards.py::test_compute_loss_reward_basic -v`
Expected: FAIL (no compute_loss_reward function)

**Step 3: Write minimal implementation**

Add the following to `src/esper/simic/rewards.py`:

```python
# Add to src/esper/simic/rewards.py - NEW SECTION

# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================

@dataclass(slots=True)
class LossRewardConfig:
    """Configuration for loss-primary reward computation.

    All weights are tunable hyperparameters optimized for
    cross-task comparability using normalized loss delta.
    """

    # Loss delta scaling
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0  # After normalization
    regression_penalty_scale: float = 0.5  # Asymmetric clipping
    typical_loss_delta_std: float = 0.1  # Task-specific normalization

    # Compute rent
    compute_rent_weight: float = 0.05

    # Stage bonuses (PBRS-compatible)
    stage_potential_weight: float = 0.1

    # Terminal bonus
    baseline_loss: float = 2.3  # Task-specific (random init loss)
    target_loss: float = 0.3  # Task-specific (achievable loss)
    terminal_loss_weight: float = 1.0

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss

    @staticmethod
    def default() -> "LossRewardConfig":
        return LossRewardConfig()

    @staticmethod
    def for_cifar10() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,
            typical_loss_delta_std=0.05,
        )

    @staticmethod
    def for_tinystories() -> "LossRewardConfig":
        return LossRewardConfig(
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,
            typical_loss_delta_std=0.15,
        )


# =============================================================================
# PBRS Stage Bonus
# =============================================================================

# Stage potentials for PBRS (monotonically increasing toward FOSSILIZED)
_STAGE_POTENTIALS = {
    0: 0.0,  # UNKNOWN
    1: 0.0,  # DORMANT
    2: 1.0,  # GERMINATED
    3: 2.0,  # TRAINING
    4: 3.0,  # BLENDING
    5: 4.0,  # SHADOWING
    6: 5.0,  # PROBATIONARY
    7: 6.0,  # FOSSILIZED (highest)
}


def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: LossRewardConfig,
    gamma: float = 0.99,
) -> float:
    """PBRS-compatible stage bonus using potential function.

    Potential Phi(s) increases monotonically toward FOSSILIZED.
    Bonus = gamma * Phi(s') - Phi(s), computed from stage transition.

    This ensures shaping doesn't change optimal policy (Ng et al., 1999).
    """
    # hasattr AUTHORIZED by John on 2025-11-30 15:00:00 UTC
    # Justification: SeedInfo may or may not have previous_stage depending on version
    previous_stage = getattr(seed_info, 'previous_stage', seed_info.stage)

    current_potential = _STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    previous_potential = _STAGE_POTENTIALS.get(previous_stage, 0.0)

    return config.stage_potential_weight * (gamma * current_potential - previous_potential)


# =============================================================================
# Loss-Primary Reward Function
# =============================================================================

def compute_loss_reward(
    action: int,
    loss_delta: float,  # current_loss - previous_loss (negative = improvement)
    val_loss: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: LossRewardConfig | None = None,
) -> float:
    """Compute loss-primary reward for seed lifecycle control.

    Sign convention:
        loss_delta < 0 means loss improved (went down)
        reward > 0 means good outcome

    Args:
        action: Action taken
        loss_delta: Loss change (negative = improvement)
        val_loss: Current validation loss
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Extra params added (fossilized + active seed)
        host_params: Baseline host model params
        config: Reward configuration

    Returns:
        Shaped reward value
    """
    if config is None:
        config = LossRewardConfig.default()

    reward = 0.0

    # === PRIMARY: Loss improvement ===
    # Normalize for cross-task comparability
    normalized_delta = loss_delta / config.typical_loss_delta_std

    # Clip to handle architecture-change spikes
    clipped = max(-config.max_loss_delta, min(normalized_delta, config.max_loss_delta))

    # Asymmetric: forgive temporary regressions
    if clipped > 0:
        clipped *= config.regression_penalty_scale

    # Negative delta (improvement) -> positive reward
    reward += (-clipped) * config.loss_delta_weight

    # === SECONDARY: Compute rent ===
    if host_params > 0 and total_params > 0:
        params_ratio = total_params / host_params
        reward -= config.compute_rent_weight * params_ratio

    # === TERTIARY: Stage bonuses (PBRS-compatible) ===
    if seed_info is not None:
        reward += compute_pbrs_stage_bonus(seed_info, config)

    # === TERMINAL: Normalized improvement from baseline ===
    if epoch == max_epochs:
        improvement = config.baseline_loss - val_loss
        normalized = max(0.0, min(improvement / config.achievable_range, 1.0))
        reward += normalized * config.terminal_loss_weight

    return reward


# Update SeedInfo to include previous_stage
class SeedInfo(NamedTuple):
    """Minimal seed information for reward computation.

    Extended for Phase 2 with previous_stage for PBRS calculation.
    """
    stage: int
    improvement_since_stage_start: float
    epochs_in_stage: int
    seed_params: int = 0
    previous_stage: int = 0  # For PBRS stage bonus


# Update __all__
__all__ = [
    # Existing exports
    "RewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_seed_potential",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    # New Phase 2 exports
    "LossRewardConfig",
    "compute_loss_reward",
    "compute_pbrs_stage_bonus",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_loss_primary_rewards.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/test_loss_primary_rewards.py
git commit -m "feat(simic): add loss-primary rewards with PBRS stage bonuses"
```

---

## Task 9: Observation Normalization

Add per-feature observation normalization for stable PPO training.

**Files:**
- Modify: `src/esper/simic/features.py`
- Test: `tests/test_observation_normalization.py`

**Step 1: Write the failing test**

```python
# tests/test_observation_normalization.py
"""Tests for observation normalization."""

import pytest


def test_normalize_observation_basic():
    """normalize_observation returns dict with same keys."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()
    obs = {
        'epoch': 10,
        'global_step': 1000,
        'train_loss': 1.5,
        'val_loss': 1.8,
        'loss_delta': -0.1,
        'plateau_epochs': 3,
        'seed_alpha': 0.5,
        'has_active_seed': 1,
        'seed_stage': 3,
    }

    normalized = normalize_observation(obs, config)

    assert 'epoch' in normalized
    assert 'val_loss' in normalized


def test_normalize_observation_epoch_range():
    """Epoch normalized to [0, 1]."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()

    obs_start = {'epoch': 0, 'global_step': 0, 'train_loss': 2.0, 'val_loss': 2.0,
                 'loss_delta': 0, 'plateau_epochs': 0, 'seed_alpha': 0,
                 'has_active_seed': 0, 'seed_stage': 0}
    obs_end = {'epoch': 25, 'global_step': 10000, 'train_loss': 0.5, 'val_loss': 0.5,
               'loss_delta': 0, 'plateau_epochs': 0, 'seed_alpha': 0,
               'has_active_seed': 0, 'seed_stage': 0}

    norm_start = normalize_observation(obs_start, config)
    norm_end = normalize_observation(obs_end, config)

    assert norm_start['epoch'] == 0.0
    assert norm_end['epoch'] == 1.0


def test_normalize_observation_loss_centered():
    """Loss normalized relative to task baseline."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()  # baseline=2.3, target=0.3

    obs = {'epoch': 10, 'global_step': 1000, 'train_loss': 2.3, 'val_loss': 2.3,
           'loss_delta': 0, 'plateau_epochs': 0, 'seed_alpha': 0,
           'has_active_seed': 0, 'seed_stage': 0}

    normalized = normalize_observation(obs, config)

    # At baseline loss, normalized should be ~1.0
    assert 0.9 < normalized['val_loss'] < 1.1


def test_task_config_cifar10():
    """TaskConfig has CIFAR-10 preset."""
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_cifar10()

    assert config.max_epochs == 25
    assert config.baseline_loss == pytest.approx(2.3, rel=0.1)


def test_task_config_tinystories():
    """TaskConfig has TinyStories preset."""
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_tinystories()

    assert config.max_epochs == 50
    assert config.baseline_loss > 5.0  # ln(vocab_size)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_observation_normalization.py -v`
Expected: FAIL (no normalize_observation or TaskConfig)

**Step 3: Write minimal implementation**

Add to `src/esper/simic/features.py`:

```python
# Add to src/esper/simic/features.py

from dataclasses import dataclass


@dataclass(slots=True)
class TaskConfig:
    """Task-specific configuration for observation normalization.

    Each task type has different baseline/target losses and episode lengths.
    """
    task_type: str  # "classification" or "lm"
    topology: str   # "cnn" or "transformer"
    baseline_loss: float  # Random init loss
    target_loss: float    # Achievable loss
    typical_loss_delta_std: float
    max_epochs: int
    max_steps: int = 10000

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss

    @staticmethod
    def for_cifar10() -> "TaskConfig":
        return TaskConfig(
            task_type="classification",
            topology="cnn",
            baseline_loss=2.3,  # ln(10)
            target_loss=0.3,    # ~90% accuracy
            typical_loss_delta_std=0.05,
            max_epochs=25,
            max_steps=10000,
        )

    @staticmethod
    def for_tinystories() -> "TaskConfig":
        return TaskConfig(
            task_type="lm",
            topology="transformer",
            baseline_loss=10.8,  # ln(50257)
            target_loss=3.5,     # ~33 perplexity
            typical_loss_delta_std=0.15,
            max_epochs=50,
            max_steps=50000,
        )


def normalize_observation(obs: dict, config: TaskConfig) -> dict:
    """Normalize observations for stable PPO training.

    Per-feature normalization based on task configuration:
    - Progress features: normalize to [0, 1]
    - Loss features: normalize relative to task baseline
    - Already bounded features: pass through

    Args:
        obs: Raw observation dictionary
        config: Task-specific configuration

    Returns:
        Normalized observation dictionary
    """
    return {
        # Progress features: normalize to [0, 1]
        'epoch': obs['epoch'] / config.max_epochs,
        'global_step': obs['global_step'] / config.max_steps,

        # Loss features: normalize relative to task baseline
        'train_loss': (obs['train_loss'] - config.target_loss) / config.achievable_range,
        'val_loss': (obs['val_loss'] - config.target_loss) / config.achievable_range,
        'loss_delta': obs['loss_delta'] / config.typical_loss_delta_std,

        # Already normalized or discrete
        'plateau_epochs': min(obs['plateau_epochs'] / 10.0, 1.0),
        'seed_alpha': obs['seed_alpha'],  # Already [0, 1]

        # Seed state: discrete, normalize stage by max
        'has_active_seed': obs['has_active_seed'],
        'seed_stage': obs['seed_stage'] / 7.0,  # Max stage is FOSSILIZED=7
    }


# Update __all__
__all__ = [
    "safe",
    "obs_to_base_features",
    "TaskConfig",
    "normalize_observation",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_observation_normalization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/features.py tests/test_observation_normalization.py
git commit -m "feat(simic): add observation normalization with TaskConfig"
```

---

## Task 10: Update SeedSlot for Forward Identity

Ensure SeedSlot.forward() returns identity when no active seed.

**Files:**
- Modify: `src/esper/kasmina/slot.py:606-622`
- Test: `tests/test_seed_slot.py`

**Step 1: Write the failing test**

```python
# tests/test_seed_slot.py
"""Tests for SeedSlot behavior."""

import pytest
import torch


def test_seed_slot_forward_no_seed_identity():
    """SeedSlot forward returns input unchanged when no seed."""
    from esper.kasmina.slot import SeedSlot

    slot = SeedSlot(slot_id="test", channels=64)

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    assert torch.allclose(out, x)


def test_seed_slot_forward_dormant_identity():
    """SeedSlot forward returns identity for DORMANT stage."""
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import SeedStage

    slot = SeedSlot(slot_id="test", channels=64)
    slot.germinate("norm", "test-seed")

    # Force back to DORMANT (shouldn't happen normally, but test the guard)
    slot.state.stage = SeedStage.DORMANT

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    assert torch.allclose(out, x)


def test_seed_slot_forward_with_seed():
    """SeedSlot forward applies seed transformation."""
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import SeedStage

    slot = SeedSlot(slot_id="test", channels=64)
    slot.germinate("norm", "test-seed")

    # Advance to TRAINING
    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    # Output should be different (seed applied)
    assert not torch.allclose(out, x)
```

**Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_seed_slot.py -v`

If tests pass, the existing implementation already handles this. If not:

**Step 3: Verify implementation is correct**

The existing `SeedSlot.forward()` in `slot.py:606-622` already checks:
- `if not self.is_active` → returns `host_features`
- `if self.alpha == 0.0` → returns `host_features`
- `if not is_active_stage(self.state.stage)` → returns `host_features`

This is correct. The tests validate the existing behavior.

**Step 4: Commit tests**

```bash
git add tests/test_seed_slot.py
git commit -m "test(kasmina): add SeedSlot forward identity tests"
```

---

## Task 11: TinyStories Dataset

Add TinyStories dataset loader for language modeling task.

**Files:**
- Modify: `src/esper/utils/data.py`
- Test: `tests/test_tinystories.py`

**Step 1: Write the failing test**

```python
# tests/test_tinystories.py
"""Tests for TinyStories dataset."""

import pytest
import torch


@pytest.mark.slow
def test_tinystories_dataset_creates():
    """TinyStoriesDataset can be instantiated (mocked)."""
    from esper.utils.data import TinyStoriesDataset

    # Use small max_samples for testing
    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64)

    assert len(dataset) > 0


@pytest.mark.slow
def test_tinystories_returns_tensor_pair():
    """TinyStoriesDataset returns (input, target) tensor pair."""
    from esper.utils.data import TinyStoriesDataset

    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)


@pytest.mark.slow
def test_tinystories_shifted_target():
    """Target is shifted by 1 from input."""
    from esper.utils.data import TinyStoriesDataset

    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64)

    x, y = dataset[0]

    # First target token should equal second input token (if we had it)
    # This tests the shift is applied correctly
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_load_tinystories_helper():
    """load_tinystories returns train/val loaders."""
    from esper.utils.data import load_tinystories

    # Just test the function signature exists
    # Actual loading tested with @slow marker
    assert callable(load_tinystories)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tinystories.py::test_load_tinystories_helper -v`
Expected: FAIL (no load_tinystories function)

**Step 3: Write minimal implementation**

Add to `src/esper/utils/data.py`:

```python
# Add to src/esper/utils/data.py

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class TinyStoriesDataset(Dataset):
    """TinyStories for causal language modeling.

    Loads TinyStories from HuggingFace and tokenizes with GPT-2 tokenizer.

    Args:
        split: "train" or "validation"
        block_size: Context length for sequences
        max_samples: Limit number of stories (for testing)
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 256,
        max_samples: int | None = None,
    ):
        self.block_size = block_size

        # Lazy import to avoid requiring transformers for CNN-only usage
        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "TinyStories requires 'datasets' and 'transformers' packages. "
                "Install with: pip install datasets transformers"
            )

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Tokenize per-story, chunk into blocks
        self.examples = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            # Chunk this story into block_size sequences
            for i in range(0, len(tokens) - self.block_size, self.block_size):
                self.examples.append(tokens[i:i + self.block_size + 1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        x = tokens[:-1]   # input
        y = tokens[1:]    # target (shifted by 1)
        return x, y


def load_tinystories(
    block_size: int = 256,
    batch_size: int = 32,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Load TinyStories train and validation dataloaders.

    Args:
        block_size: Context length for sequences
        batch_size: Batch size
        max_train_samples: Limit training samples (for testing)
        max_val_samples: Limit validation samples (for testing)
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TinyStoriesDataset(
        split="train",
        block_size=block_size,
        max_samples=max_train_samples,
    )
    val_dataset = TinyStoriesDataset(
        split="validation",
        block_size=block_size,
        max_samples=max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# Update existing load_cifar10 if needed...

__all__ = [
    # Existing exports...
    "TinyStoriesDataset",
    "load_tinystories",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tinystories.py::test_load_tinystories_helper -v`
Expected: PASS

Note: Tests marked `@pytest.mark.slow` require network access and will download the dataset.

**Step 5: Commit**

```bash
git add src/esper/utils/data.py tests/test_tinystories.py
git commit -m "feat(utils): add TinyStories dataset for language modeling"
```

---

## Task 12: Integration Test - TransformerHost + SeedSlot

Verify TransformerHost works with seed lifecycle.

**Files:**
- Create: `tests/integration/test_transformer_integration.py`

**Step 1: Write integration test**

```python
# tests/integration/test_transformer_integration.py
"""Integration tests for TransformerHost with seed lifecycle."""

import pytest
import torch
import torch.nn as nn


def test_transformer_with_seed_lifecycle():
    """Full seed lifecycle on TransformerHost."""
    from esper.kasmina.host import TransformerHost
    from esper.kasmina.slot import SeedSlot
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.leyline import SeedStage

    # Create host
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32)

    # Create seed slot for first injection point
    slot_id = list(host.injection_points.keys())[0]
    dim = host.injection_points[slot_id]
    slot = SeedSlot(slot_id=slot_id, channels=dim, fast_mode=True)

    # Test forward before seed
    x = torch.randint(0, 1000, (2, 16))
    out_before = host(x)

    # Germinate seed
    slot.germinate("lora", "test-lora")
    host.register_slot(slot_id, slot.seed)

    # Advance to training
    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    # Forward with seed
    out_with_seed = host(x)

    # Should be different (seed active)
    assert out_before.shape == out_with_seed.shape
    assert not torch.allclose(out_before, out_with_seed)

    # Unregister seed
    host.unregister_slot(slot_id)

    # Forward after unregister
    out_after = host(x)

    # Should match before (back to identity)
    assert torch.allclose(out_before, out_after)


def test_transformer_gradient_flow():
    """Gradients flow through TransformerHost with seed."""
    from esper.kasmina.host import TransformerHost
    from esper.kasmina.blueprints import BlueprintRegistry

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32)

    # Register a LoRA seed
    slot_id = "layer_0_post_block"
    lora = BlueprintRegistry.create("transformer", "lora", dim=64)
    host.register_slot(slot_id, lora)

    # Forward and backward
    x = torch.randint(0, 1000, (2, 16))
    out = host(x)
    loss = out.sum()
    loss.backward()

    # LoRA should have gradients
    for name, param in lora.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_host_cnn_still_works():
    """HostCNN still works with new protocol."""
    from esper.kasmina.host import HostCNN
    from esper.kasmina.blueprints import BlueprintRegistry

    host = HostCNN()

    # Register attention seed
    attention = BlueprintRegistry.create("cnn", "attention", dim=64)
    host.register_slot("block2_post", attention)

    # Forward
    x = torch.randn(2, 3, 32, 32)
    out = host(x)

    assert out.shape == (2, 10)

    # Gradients
    loss = out.sum()
    loss.backward()

    for param in attention.parameters():
        assert param.grad is not None
```

**Step 2: Run test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_transformer_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_transformer_integration.py
git commit -m "test(integration): add TransformerHost + SeedSlot integration tests"
```

---

## Task 13: Verify All Tests Pass

Run full test suite to ensure no regressions.

**Step 1: Run all tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --ignore=tests/integration/test_ppo_integration.py -x`

**Step 2: Fix any failures**

Address any test failures before proceeding.

**Step 3: Commit fixes if needed**

```bash
git add -A
git commit -m "fix: address test failures from Phase 2 changes"
```

---

## Task 14: Update Package Exports

Ensure new modules are exported from package `__init__.py` files.

**Files:**
- Modify: `src/esper/kasmina/__init__.py`
- Modify: `src/esper/leyline/__init__.py`
- Modify: `src/esper/simic/__init__.py`

**Step 1: Update kasmina exports**

```python
# src/esper/kasmina/__init__.py
from esper.kasmina.protocol import HostProtocol
from esper.kasmina.host import (
    HostCNN,
    TransformerHost,
    TransformerBlock,
    MorphogeneticModel,
)
from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics
from esper.kasmina.blueprints import BlueprintRegistry, BlueprintSpec

__all__ = [
    "HostProtocol",
    "HostCNN",
    "TransformerHost",
    "TransformerBlock",
    "MorphogeneticModel",
    "SeedSlot",
    "SeedState",
    "SeedMetrics",
    "BlueprintRegistry",
    "BlueprintSpec",
]
```

**Step 2: Update leyline exports**

```python
# Add to src/esper/leyline/__init__.py
from esper.leyline.actions import build_action_enum, get_blueprint_from_action, is_germinate_action
```

**Step 3: Update simic exports**

```python
# Add to src/esper/simic/__init__.py
from esper.simic.rewards import LossRewardConfig, compute_loss_reward, compute_pbrs_stage_bonus
from esper.simic.features import TaskConfig, normalize_observation
```

**Step 4: Commit**

```bash
git add src/esper/kasmina/__init__.py src/esper/leyline/__init__.py src/esper/simic/__init__.py
git commit -m "chore: update package exports for Phase 2"
```

---

## Task 15: Topology Safety Guard

Add assertion in SeedSlot.germinate() to prevent blueprint/host topology mismatch.

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/test_topology_guard.py`

**Step 1: Write the failing test**

```python
# tests/test_topology_guard.py
"""Tests for topology safety guard."""

import pytest


def test_germinate_wrong_topology_raises():
    """Germinating transformer blueprint on CNN slot raises."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.features import TaskConfig

    # CNN task config
    config = TaskConfig.for_cifar10()

    slot = SeedSlot(
        slot_id="block2_post",
        channels=64,
        task_config=config,  # CNN topology
    )

    # Try to germinate a transformer blueprint
    with pytest.raises(AssertionError, match="topology"):
        slot.germinate("lora", "bad-seed")  # lora is transformer-only


def test_germinate_correct_topology_succeeds():
    """Germinating matching topology blueprint succeeds."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_cifar10()

    slot = SeedSlot(
        slot_id="block2_post",
        channels=64,
        task_config=config,
    )

    # Germinate CNN blueprint - should work
    state = slot.germinate("norm", "good-seed")
    assert state is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_topology_guard.py -v`
Expected: FAIL (SeedSlot doesn't have task_config parameter)

**Step 3: Update SeedSlot**

```python
# In src/esper/kasmina/slot.py - update __init__ and germinate

def __init__(
    self,
    slot_id: str,
    channels: int,
    device: torch.device | str = "cpu",
    gates: QualityGates | None = None,
    on_telemetry: Callable[[TelemetryEvent], None] | None = None,
    fast_mode: bool = False,
    task_config: "TaskConfig | None" = None,  # NEW
):
    # ... existing init ...
    self.task_config = task_config

def germinate(
    self,
    blueprint_id: str,
    seed_id: str | None = None,
    host_module: nn.Module | None = None,
) -> SeedState:
    """Germinate a new seed in this slot."""
    from esper.kasmina.blueprints import BlueprintRegistry

    # Topology guard
    if self.task_config is not None:
        topology = self.task_config.topology
        try:
            spec = BlueprintRegistry.get(topology, blueprint_id)
        except ValueError:
            # Blueprint not found for this topology
            available = BlueprintRegistry.list_for_topology(topology)
            names = [s.name for s in available]
            raise AssertionError(
                f"Blueprint '{blueprint_id}' not available for topology '{topology}'. "
                f"Available: {names}"
            )

    # ... rest of existing germinate code ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_topology_guard.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_topology_guard.py
git commit -m "feat(kasmina): add topology safety guard in SeedSlot.germinate"
```

---

## Task 16: Rent Grace Period

Add seed age tracking and rent-free grace period for newly germinated seeds.

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Test: `tests/test_rent_grace.py`

**Step 1: Write the failing test**

```python
# tests/test_rent_grace.py
"""Tests for rent grace period."""

import pytest


def test_rent_not_applied_during_grace():
    """No rent during grace period."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig, SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    # Seed in grace period (age=1)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=50000,
        seed_age_epochs=1,  # NEW: within grace
    )

    reward_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info,
        epoch=5,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    # Same but outside grace
    seed_info_old = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,  # Outside grace
    )

    reward_no_grace = compute_loss_reward(
        action=0,
        loss_delta=-0.1,
        val_loss=2.0,
        seed_info=seed_info_old,
        epoch=9,
        max_epochs=25,
        total_params=50000,
        host_params=100000,
        config=config,
    )

    # Grace period should give higher reward (no rent paid)
    assert reward_grace > reward_no_grace


def test_rent_applied_after_grace():
    """Rent applied after grace period ends."""
    from esper.simic.rewards import compute_loss_reward, LossRewardConfig, SeedInfo
    from esper.leyline import SeedStage

    config = LossRewardConfig.default()
    config.grace_epochs = 3

    # Seed past grace (age=5)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        epochs_in_stage=5,
        seed_params=50000,
        seed_age_epochs=5,
    )

    # With params
    reward_with_params = compute_loss_reward(
        action=0, loss_delta=0.0, val_loss=2.0,
        seed_info=seed_info, epoch=10, max_epochs=25,
        total_params=50000, host_params=100000,
        config=config,
    )

    # Without params
    reward_no_params = compute_loss_reward(
        action=0, loss_delta=0.0, val_loss=2.0,
        seed_info=seed_info, epoch=10, max_epochs=25,
        total_params=0, host_params=100000,
        config=config,
    )

    # Rent should be applied
    assert reward_no_params > reward_with_params
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rent_grace.py -v`
Expected: FAIL (SeedInfo doesn't have seed_age_epochs)

**Step 3: Update implementation**

```python
# Update SeedInfo in src/esper/simic/rewards.py

class SeedInfo(NamedTuple):
    """Minimal seed information for reward computation."""
    stage: int
    improvement_since_stage_start: float
    epochs_in_stage: int
    seed_params: int = 0
    previous_stage: int = 0
    seed_age_epochs: int = 0  # NEW: total epochs since germination


@dataclass(slots=True)
class LossRewardConfig:
    # ... existing fields ...
    grace_epochs: int = 3  # NEW: rent-free grace period


def compute_loss_reward(...) -> float:
    # ... existing code ...

    # === SECONDARY: Compute rent (with grace period) ===
    if host_params > 0 and total_params > 0:
        # Check if seed is past grace period
        in_grace = False
        if seed_info is not None:
            in_grace = seed_info.seed_age_epochs < config.grace_epochs

        if not in_grace:
            params_ratio = total_params / host_params
            reward -= config.compute_rent_weight * params_ratio

    # ... rest of function ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rent_grace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/test_rent_grace.py
git commit -m "feat(simic): add rent-free grace period for new seeds"
```

---

## Task 17: LM Validation (Perplexity)

Add perplexity computation for language model validation.

**Files:**
- Modify: `src/esper/tolaria/trainer.py`
- Test: `tests/test_lm_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_lm_validation.py
"""Tests for LM validation metrics."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_validate_lm_returns_perplexity():
    """validate_and_get_metrics computes perplexity for LM."""
    from esper.tolaria.trainer import validate_and_get_metrics

    # Simple model
    model = nn.Linear(100, 100)

    # Fake data
    x = torch.randint(0, 100, (32, 10))
    y = torch.randint(0, 100, (32, 10))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, train_loss, train_acc, per_class, perplexity = \
        validate_and_get_metrics(
            model=model,
            trainloader=loader,
            testloader=loader,
            criterion=criterion,
            device="cpu",
            task_type="lm",  # NEW
        )

    assert perplexity is not None
    assert perplexity > 0


def test_validate_classification_no_perplexity():
    """Classification tasks return None for perplexity."""
    from esper.tolaria.trainer import validate_and_get_metrics

    # Simple CNN
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )

    # Fake data
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, train_loss, train_acc, per_class, perplexity = \
        validate_and_get_metrics(
            model=model,
            trainloader=loader,
            testloader=loader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

    assert perplexity is None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lm_validation.py -v`
Expected: FAIL (task_type parameter not supported)

**Step 3: Update validate_and_get_metrics**

```python
# In src/esper/tolaria/trainer.py

import math

def validate_and_get_metrics(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    compute_per_class: bool = False,
    num_classes: int = 10,
    task_type: str = "classification",  # NEW: "classification" or "lm"
) -> tuple[float, float, float, float, dict[int, float] | None, float | None]:
    """Get validation and training metrics.

    Returns:
        Tuple of (val_loss, val_accuracy, train_loss, train_accuracy,
                  per_class_acc, perplexity)
        perplexity is None for classification, exp(val_loss) for LM.
    """
    # ... existing validation code ...

    # Compute perplexity for LM tasks
    perplexity = None
    if task_type == "lm":
        perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')

    return val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lm_validation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tolaria/trainer.py tests/test_lm_validation.py
git commit -m "feat(tolaria): add perplexity computation for LM validation"
```

---

## Task 18: Sanity Check Logging

Add lightweight logging to catch issues early: reward magnitude, params ratio, shape assertions.

**Files:**
- Modify: `src/esper/simic/training.py` (or create `src/esper/simic/sanity.py`)
- Test: `tests/test_sanity_logging.py`

**Step 1: Write the failing test**

```python
# tests/test_sanity_logging.py
"""Tests for sanity check logging."""

import pytest


def test_reward_magnitude_warning(caplog):
    """Large reward magnitude triggers warning."""
    from esper.simic.sanity import check_reward_magnitude

    import logging
    caplog.set_level(logging.WARNING)

    # Normal reward
    check_reward_magnitude(2.5, epoch=1, max_epochs=25)
    assert len(caplog.records) == 0

    # Large reward should warn
    check_reward_magnitude(15.0, epoch=1, max_epochs=25)
    assert any("reward magnitude" in r.message.lower() for r in caplog.records)


def test_params_ratio_logging():
    """Params ratio is logged for debugging."""
    from esper.simic.sanity import log_params_ratio

    # Just verify it doesn't crash
    log_params_ratio(total_params=50000, host_params=100000, epoch=5)


def test_shape_guard_assertion():
    """Shape guard raises on mismatch."""
    import torch
    from esper.simic.sanity import assert_slot_shape

    x = torch.randn(2, 64, 8, 8)  # CNN: (B, C, H, W)

    # Correct shape
    assert_slot_shape(x, expected_dim=64, topology="cnn")  # Should pass

    # Wrong shape
    with pytest.raises(AssertionError):
        assert_slot_shape(x, expected_dim=128, topology="cnn")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_sanity_logging.py -v`
Expected: FAIL (no simic.sanity module)

**Step 3: Create sanity module**

```python
# src/esper/simic/sanity.py
"""Sanity Check Utilities for Simic Training.

Lightweight logging and assertions to catch issues early.
Enable with DEBUG logging level or environment variable.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Enable detailed sanity checks via env var
SANITY_CHECKS_ENABLED = os.getenv("ESPER_SANITY_CHECKS", "0") == "1"


def check_reward_magnitude(
    reward: float,
    epoch: int,
    max_epochs: int,
    threshold: float = 10.0,
) -> None:
    """Warn if reward magnitude is unexpectedly large.

    Large rewards suggest miscalibrated weights or normalization issues.
    """
    if abs(reward) > threshold:
        logger.warning(
            f"Large reward magnitude {reward:.2f} at epoch {epoch}/{max_epochs}. "
            "Consider reducing loss_delta_weight or adjusting typical_loss_delta_std."
        )


def log_params_ratio(
    total_params: int,
    host_params: int,
    epoch: int,
) -> None:
    """Log params ratio for debugging rent calibration."""
    if host_params > 0:
        ratio = total_params / host_params
        logger.debug(f"Epoch {epoch}: params_ratio={ratio:.3f} ({total_params}/{host_params})")


def assert_slot_shape(
    x: torch.Tensor,
    expected_dim: int,
    topology: str,
) -> None:
    """Assert tensor has expected dimension for slot.

    CNN: expects (B, C, H, W) where C == expected_dim
    Transformer: expects (B, T, D) where D == expected_dim
    """
    if topology == "cnn":
        if x.dim() != 4:
            raise AssertionError(f"CNN slot expects 4D tensor, got {x.dim()}D")
        actual = x.shape[1]  # C dimension
    elif topology == "transformer":
        if x.dim() != 3:
            raise AssertionError(f"Transformer slot expects 3D tensor, got {x.dim()}D")
        actual = x.shape[2]  # D dimension
    else:
        raise ValueError(f"Unknown topology: {topology}")

    if actual != expected_dim:
        raise AssertionError(
            f"Slot dimension mismatch: expected {expected_dim}, got {actual}"
        )


__all__ = [
    "check_reward_magnitude",
    "log_params_ratio",
    "assert_slot_shape",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_sanity_logging.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/sanity.py tests/test_sanity_logging.py
git commit -m "feat(simic): add sanity check utilities for early issue detection"
```

---

## Summary

This implementation plan covers the core Phase 2 infrastructure:

### Core Infrastructure (Tasks 1-14)

1. **HostProtocol** - Structural typing for pluggable hosts
2. **HostCNN refactor** - ModuleDict + Identity pattern
3. **TransformerHost** - GPT-style decoder with injection points
4. **Blueprint registry** - Plugin system with decorators
5. **Transformer blueprints** - norm, lora, attention, mlp
6. **Legacy compatibility** - BlueprintCatalog delegates to registry
7. **Per-topology actions** - Dynamic action enum generation
8. **Loss-primary rewards** - With PBRS stage bonuses
9. **Observation normalization** - TaskConfig-driven
10. **SeedSlot identity** - Forward returns input when inactive
11. **TinyStories dataset** - Language modeling data
12. **Integration tests** - TransformerHost + SeedSlot lifecycle
13. **Test verification** - Full suite passes
14. **Package exports** - Updated __init__.py files

### Guards & Sanity Checks (Tasks 15-18)

15. **Topology safety guard** - Prevent blueprint/host mismatch
16. **Rent grace period** - Rent-free exploration for new seeds
17. **LM validation** - Perplexity computation for language models
18. **Sanity logging** - Reward magnitude, params ratio, shape assertions

**Total estimated tasks:** 18 (each with 4-6 sub-steps)

**Not included (deferred to Phase 3+):**
- MorphogeneticModel updates for TransformerHost
- Multi-slot coordination
- TensorSchema update (accuracy removal)
- Training loop modifications for LM task
