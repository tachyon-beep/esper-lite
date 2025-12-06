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
from esper.kasmina.blueprints.cnn import ConvBlock  # Reuse shared building block


class CNNHost(nn.Module):
    """CNN host with dynamic blocks and injection points after each block (except the first).

    Mirrors TransformerHost's pattern: a ModuleList of blocks, a ModuleDict of slots keyed
    by block index, and a simple looped forward that applies slots as identities when unused.
    """

    def __init__(self, num_classes: int = 10, n_blocks: int = 3, base_channels: int = 32):
        super().__init__()
        if n_blocks < 2:
            raise ValueError("CNNHost requires at least 2 blocks to expose an injection point")

        self.n_blocks = n_blocks
        self.base_channels = base_channels

        # Build blocks with doubling channels each stage
        blocks: list[nn.Module] = []
        in_c = 3
        for i in range(n_blocks):
            out_c = base_channels * (2 ** i)
            blocks.append(ConvBlock(in_c, out_c))
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2, 2)

        # Slots after each block except the first (aligns with previous block2_post default)
        self._slot_indices = tuple(range(1, n_blocks))
        # Keep legacy-friendly naming (block2_post) while allowing multiple slots
        self._slot_keys = tuple(f"block{idx + 1}_post" for idx in self._slot_indices)
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

        # Classifier maps final channels → logits
        self.classifier = nn.Linear(in_c, num_classes)

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel dimension."""
        return {k: self.blocks[idx].conv.out_channels for k, idx in zip(self._slot_keys, self._slot_indices)}

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
        slot_idx = 0
        for idx, block in enumerate(self.blocks):
            x = self.pool(block(x))
            # Use pre-computed _slot_indices instead of string formatting
            if idx in self._slot_indices:
                x = self.slots[self._slot_keys[slot_idx]](x)
                slot_idx += 1

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


# =============================================================================
# Transformer Components
# =============================================================================


class CausalSelfAttention(nn.Module):
    """Causal self-attention using scaled_dot_product_attention for Flash Attention support."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout_p = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        # Note: attn_dropout handled by SDPA internally via dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use scaled_dot_product_attention for Flash Attention support
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,  # Handles causal masking automatically
        )
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
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

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
    """Model with Kasmina seed slot registered into host injection points."""

    def __init__(
        self,
        host: nn.Module,
        device: str = "cpu",
        slot_id: str | None = None,
        task_config=None,
        fast_mode: bool = False,
    ):
        super().__init__()
        self.host = host
        self._device = device
        self.task_config = task_config

        # Host must expose a concrete injection_points mapping for seed slots.
        try:
            injection_points = host.injection_points
        except AttributeError as exc:
            raise ValueError("Host must expose injection_points mapping for seed slots") from exc
        if not injection_points:
            raise ValueError("Host injection_points mapping must not be empty")

        self.slot_id = slot_id or next(iter(injection_points))
        if self.slot_id not in injection_points:
            raise ValueError(
                f"Slot '{self.slot_id}' not found in host injection points: {list(injection_points.keys())}"
            )

        channels = injection_points[self.slot_id]
        self.seed_slot = SeedSlot(
            slot_id=self.slot_id,
            channels=channels,
            device=device,
            task_config=task_config,
            fast_mode=fast_mode,
        )

        # Host must implement register_slot(slot_id, module).
        try:
            register_slot = self.host.register_slot
        except AttributeError as exc:
            raise ValueError("Host must implement register_slot(slot_id, module)") from exc

        # Move host to device BEFORE registering slot. register_slot() queries
        # next(self.parameters()).device to place the slot, so host must already
        # be on the target device to avoid corrupting SeedSlot.device to CPU.
        self.host = self.host.to(device)
        register_slot(self.slot_id, self.seed_slot)

    def to(self, *args, **kwargs):
        """Override to() to update device tracking after transfer.

        Note: super().to() already moves all registered submodules including
        seed_slot and its seed. We only update our device tracking string.

        Implementation note (PyTorch Expert review): Query device from parameters
        AFTER super().to() completes rather than parsing args. This is simpler,
        correct, and follows PyTorch conventions - query state after mutation
        rather than trying to parse the complex .to() signature which accepts
        device, dtype, tensor, memory_format, and non_blocking in various forms.
        """
        result = super().to(*args, **kwargs)

        # Query actual device from parameters (canonical source of truth)
        try:
            actual_device = next(self.parameters()).device
        except StopIteration:
            # No parameters - keep existing device tracking
            return result

        # Update tracking (seed already moved by super().to())
        self.seed_slot.device = actual_device
        self._device = str(actual_device)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.host(x)

    def germinate_seed(self, blueprint_id: str, seed_id: str) -> None:
        """Germinate a new seed."""
        self.seed_slot.germinate(
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
        """Return host backbone parameters only (exclude seed slots)."""
        for name, param in self.host.named_parameters():
            if "slots" in name:
                continue
            yield param

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
    "CNNHost",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "TransformerHost",
    "MorphogeneticModel",
]
