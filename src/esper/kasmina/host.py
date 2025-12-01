"""Kasmina Host - The graftable host network.

The MorphogeneticModel is the host network that accepts seed grafts.
It manages the injection points where seeds can be attached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import SeedStage
from esper.kasmina.slot import SeedSlot
from esper.kasmina.isolation import GradientIsolationMonitor
from esper.kasmina.blueprints.cnn import ConvBlock  # Reuse shared building block


class HostCNN(nn.Module):
    """CNN host with single injection point after block2."""

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

    # Legacy methods for MorphogeneticModel compatibility
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
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
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
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
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
    """GPT-style decoder with injection points after each layer."""

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

        self.layers = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: tok_emb is master
        self.head.weight = self.tok_emb.weight

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


# =============================================================================
# Morphogenetic Model
# =============================================================================

class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slot."""

    def __init__(self, host: HostCNN, device: str = "cpu"):
        super().__init__()
        self.host = host
        self._device = device

        # Single seed slot at injection point
        self.seed_slot = SeedSlot(
            slot_id="injection_point",
            channels=host.injection_channels,
            device=device,
        )

        # Isolation monitor
        self.isolation_monitor = GradientIsolationMonitor()

    def to(self, *args, **kwargs):
        """Override to() to propagate device change to SeedSlot.

        SeedSlot is not an nn.Module, so PyTorch's recursive to() doesn't
        reach it. We manually propagate the device change to ensure the
        slot creates new seeds on the correct device.
        """
        result = super().to(*args, **kwargs)

        # Determine the new device from model parameters
        try:
            new_device = next(self.parameters()).device
        except StopIteration:
            return result  # No parameters, nothing to update

        # Propagate to seed slot
        self.seed_slot.device = new_device
        self._device = str(new_device)

        # Move existing seed if present
        if self.seed_slot.seed is not None:
            self.seed_slot.seed = self.seed_slot.seed.to(new_device)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.host.forward_to_injection(x)
        features = self.seed_slot.forward(features)
        return self.host.forward_from_injection(features)

    def germinate_seed(self, blueprint_id: str, seed_id: str) -> None:
        """Germinate a new seed."""
        state = self.seed_slot.germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )

    def cull_seed(self) -> None:
        """Cull the current seed."""
        self.seed_slot.cull()

    def get_seed_parameters(self):
        return self.seed_slot.get_parameters()

    def get_host_parameters(self):
        return self.host.parameters()

    @property
    def has_active_seed(self) -> bool:
        return self.seed_slot.is_active

    @property
    def seed_state(self):
        return self.seed_slot.state

    @property
    def active_seed_params(self) -> int:
        """Return trainable params of active seed."""
        return self.seed_slot.active_seed_params


__all__ = [
    "ConvBlock",
    "HostCNN",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "TransformerHost",
    "MorphogeneticModel",
]
