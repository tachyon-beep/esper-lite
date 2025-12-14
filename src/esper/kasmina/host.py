"""Kasmina Host - The graftable host network.

The MorphogeneticModel is the host network that accepts seed grafts.
It manages the injection points where seeds can be attached.
"""

from __future__ import annotations

from typing_extensions import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import SeedStage, is_terminal_stage
from esper.kasmina.slot import SeedSlot
from esper.kasmina.blueprints.cnn import ConvBlock  # Reuse shared building block


class CNNHost(nn.Module):
    """CNN host with dynamic blocks and injection points after each block (except the first).

    Mirrors TransformerHost's pattern: a ModuleList of blocks, a ModuleDict of slots keyed
    by block index, and a simple looped forward that applies slots as identities when unused.

    Args:
        num_classes: Number of output classes (default 10 for CIFAR-10)
        n_blocks: Number of conv blocks (default 3, minimum 2)
        base_channels: Initial channel count, doubles each block (default 32)
        pool_layers: Number of blocks that apply max pooling (default: all blocks).
            For CIFAR-10 (32x32), max 5 pool layers (32→16→8→4→2→1).
            Extra blocks after pool_layers add depth without reducing spatial size.
        memory_format: Memory layout for conv operations (default channels_last).
            channels_last provides 10-20% speedup on Ampere/Hopper GPUs with Tensor Cores.
            Use contiguous_format for older GPUs or debugging.
    """

    def __init__(
        self,
        num_classes: int = 10,
        n_blocks: int = 3,
        base_channels: int = 32,
        pool_layers: int | None = None,
        memory_format: torch.memory_format = torch.channels_last,
    ):
        super().__init__()
        if n_blocks < 2:
            raise ValueError("CNNHost requires at least 2 blocks to expose an injection point")

        self.n_blocks = n_blocks
        self.base_channels = base_channels
        self._memory_format = memory_format
        # Default: pool on all layers (original behavior)
        # For deep networks on small images, limit pooling to avoid 0x0 spatial
        self._pool_layers = pool_layers if pool_layers is not None else n_blocks

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

        # Segment channel counts for multi-slot support
        # Map named segments to their channel dimensions at injection points
        # Requires at least 3 blocks for full early/mid/late segment support
        if n_blocks >= 3:
            self.segment_channels = {
                "early": self.blocks[0].conv.out_channels,    # After block1 (32 by default)
                "mid": self.blocks[1].conv.out_channels,      # After block2 (64 by default)
                "late": self.blocks[2].conv.out_channels,     # After block3 (128 by default)
            }
        else:
            # Fallback for shallow networks - only expose available segments
            self.segment_channels = {
                f"block{i}": self.blocks[i].conv.out_channels for i in range(n_blocks)
            }

    @property
    @override
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel dimension."""
        return {k: self.blocks[idx].conv.out_channels for k, idx in zip(self._slot_keys, self._slot_indices)}

    @override
    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = next(self.parameters()).device
        self.slots[slot_id] = slot.to(device)

    @override
    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    # NOTE: forward_to_segment() is intentionally duplicated between CNNHost
    # and TransformerHost. While the structure is similar, the details differ:
    # - CNN: handles pool_layers, uses spatial features (B, C, H, W)
    # - Transformer: handles sequence features (B, T, n_embd), no pooling
    # Extracting to shared base would require topology-specific conditionals,
    # reducing clarity. Duplication is acceptable given distinct semantics.

    def forward_to_segment(
        self,
        segment: str,
        x: torch.Tensor,
        from_segment: str | None = None
    ) -> torch.Tensor:
        """Forward from one segment boundary to another.

        Args:
            segment: Target segment (e.g., "early", "mid", "late")
            x: Raw input if from_segment is None, else features at from_segment boundary
            from_segment: Starting point (None = network input)

        Returns:
            Features at segment boundary
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")
        if from_segment is not None and from_segment not in self.segment_channels:
            raise ValueError(f"Unknown from_segment: {from_segment}. Available: {list(self.segment_channels.keys())}")

        # Convert to channels_last ONCE before processing for Tensor Core optimization
        if self._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        # Map segment names to block indices (0-indexed)
        segment_to_block = {
            "early": 0,  # After block 0 (block1)
            "mid": 1,    # After block 1 (block2)
            "late": 2,   # After block 2 (block3)
        }
        target_block = segment_to_block[segment]
        start_block = 0 if from_segment is None else segment_to_block[from_segment] + 1

        # Map block index to registered slot key for slot application
        slot_by_idx = dict(zip(self._slot_indices, self._slot_keys))

        # Forward through blocks in range [start_block, target_block]
        for idx in range(start_block, target_block + 1):
            x = self.blocks[idx](x)
            # Only pool on first pool_layers blocks
            if idx < self._pool_layers:
                x = self.pool(x)
            slot_key = slot_by_idx.get(idx)
            if slot_key:
                x = self.slots[slot_key](x)

        return x

    def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        """Forward from a segment to output.

        Args:
            segment: Starting segment name ("early", "mid", or "late")
            x: Feature map at segment boundary (should already be channels_last if from forward_to_segment)

        Returns:
            Classification logits
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

        # Ensure channels_last format for Tensor Core optimization (safe even if already in format)
        if self._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        # Map segment names to block indices
        segment_to_block = {
            "early": 0,
            "mid": 1,
            "late": 2,
        }
        start_block = segment_to_block[segment]

        slot_by_idx = dict(zip(self._slot_indices, self._slot_keys))

        # Forward through remaining blocks
        for idx in range(start_block + 1, self.n_blocks):
            x = self.blocks[idx](x)
            # Only pool on first pool_layers blocks
            if idx < self._pool_layers:
                x = self.pool(x)
            slot_key = slot_by_idx.get(idx)
            if slot_key:
                x = self.slots[slot_key](x)

        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to channels_last ONCE before processing for Tensor Core optimization
        # Conv2d, BatchNorm2d, MaxPool2d all preserve channels_last format
        if self._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        slot_idx = 0
        for idx, block in enumerate(self.blocks):
            x = block(x)
            # Only pool on first pool_layers blocks (avoids 0x0 spatial on deep nets)
            if idx < self._pool_layers:
                x = self.pool(x)
            # Use pre-computed _slot_indices instead of string formatting
            if idx in self._slot_indices:
                x = self.slots[self._slot_keys[slot_idx]](x)
                slot_idx += 1

        # flatten() handles memory format conversion automatically (returns contiguous)
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

        # Segment channel counts for multi-slot support
        # For transformers, all segments have the same embedding dimension
        # Segments map to layer ranges: early (0-1), mid (2-3), late (4-5)
        self.segment_channels = {
            "early": n_embd,
            "mid": n_embd,
            "late": n_embd,
        }

        # Layer range boundaries for segments (layer index where segment ENDS)
        # For n_layer=6: early=0-1, mid=2-3, late=4-5
        third = n_layer // 3
        self._segment_boundaries = {
            "early": third,           # Layers 0 to third-1
            "mid": 2 * third,         # Layers third to 2*third-1
            "late": n_layer,          # Layers 2*third to n_layer-1
        }

    @property
    @override
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> embedding dimension."""
        return {k: self.n_embd for k in self._slot_keys}

    @override
    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = self.tok_emb.weight.device
        self.slots[slot_id] = slot.to(device)

    @override
    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point."""
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    # NOTE: forward_to_segment() is intentionally duplicated between CNNHost
    # and TransformerHost. While the structure is similar, the details differ:
    # - CNN: handles pool_layers, uses spatial features (B, C, H, W)
    # - Transformer: handles sequence features (B, T, n_embd), no pooling
    # Extracting to shared base would require topology-specific conditionals,
    # reducing clarity. Duplication is acceptable given distinct semantics.

    def forward_to_segment(
        self,
        segment: str,
        x: torch.Tensor,
        from_segment: str | None = None
    ) -> torch.Tensor:
        """Forward from one segment boundary to another.

        Args:
            segment: Target segment (e.g., "early", "mid", "late")
            x: Raw input if from_segment is None (B, T token indices),
               else features at from_segment boundary (B, T, n_embd)
            from_segment: Starting point (None = network input)

        Returns:
            Hidden states at segment boundary (B, T, n_embd)
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")
        if from_segment is not None and from_segment not in self.segment_channels:
            raise ValueError(f"Unknown from_segment: {from_segment}. Available: {list(self.segment_channels.keys())}")

        # If starting from network input, apply embeddings
        if from_segment is None:
            B, T = x.shape
            if T > self.block_size:
                raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
            pos = torch.arange(T, device=x.device)
            h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
            start_layer = 0
        else:
            # x is already embedded features
            h = x
            start_layer = self._segment_boundaries[from_segment]

        # Forward through layers in range [start_layer, end_layer)
        end_layer = self._segment_boundaries[segment]
        for i in range(start_layer, end_layer):
            h = self.layers[i](h)
            h = self.slots[self._slot_keys[i]](h)

        return h

    def forward_from_segment(self, segment: str, h: torch.Tensor) -> torch.Tensor:
        """Forward from a segment boundary to output logits.

        Args:
            segment: Starting segment name ("early", "mid", or "late")
            h: Hidden states at segment boundary (B, T, n_embd)

        Returns:
            Output logits (B, T, vocab_size)
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

        # Forward through remaining layers
        start_layer = self._segment_boundaries[segment]
        for i in range(start_layer, self.n_layer):
            h = self.layers[i](h)
            h = self.slots[self._slot_keys[i]](h)

        # Output
        h = self.ln_f(h)
        return self.head(h)

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
    """Model with Kasmina seed slots registered into host injection points.

    Multi-slot architecture for managing multiple concurrent seeds at different
    network segments (early/mid/late).
    """

    def __init__(
        self,
        host: nn.Module,
        device: str = "cpu",
        *,
        slots: list[str],
        task_config=None,
        fast_mode: bool = False,
    ):
        super().__init__()
        self.host = host
        self._device = device
        self.task_config = task_config

        # Host must expose segment_channels for multi-slot support
        segment_channels = host.segment_channels

        # Create seed slots as ModuleDict for proper submodule registration
        slots_dict = {}
        for slot_name in slots:
            if slot_name not in segment_channels:
                raise ValueError(
                    f"Unknown slot: {slot_name}. Available: {list(segment_channels.keys())}"
                )
            slots_dict[slot_name] = SeedSlot(
                slot_id=slot_name,
                channels=segment_channels[slot_name],
                device=device,
                task_config=task_config,
                fast_mode=fast_mode,
            )
        self.seed_slots = nn.ModuleDict(slots_dict)

        # Track slot order for forward pass
        self._slot_order = ["early", "mid", "late"]
        self._active_slots = [s for s in self._slot_order if s in self.seed_slots]

        # Move host to device
        self.host = self.host.to(device)

    def to(self, *args, **kwargs):
        """Override to() to update device tracking after transfer."""
        result = super().to(*args, **kwargs)

        # Query actual device from parameters (canonical source of truth)
        try:
            actual_device = next(self.parameters()).device
        except StopIteration:
            return result

        # Update tracking for all slots
        for slot in self.seed_slots.values():
            slot.device = actual_device
        self._device = str(actual_device)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through host with all active slots.

        Processes sequentially through network segments, applying slot
        transformations at each segment boundary.
        """
        # If no active slots, use host's forward directly
        if not self._active_slots:
            return self.host(x)

        # Process through active slots
        prev_segment = None
        for slot_id in self._active_slots:
            x = self.host.forward_to_segment(slot_id, x, from_segment=prev_segment)
            x = self.seed_slots[slot_id].forward(x)
            prev_segment = slot_id
        return self.host.forward_from_segment(prev_segment, x)

    def germinate_seed(
        self,
        blueprint_id: str,
        seed_id: str,
        *,
        slot: str,
        blend_algorithm_id: str = "sigmoid",
    ) -> None:
        """Germinate a new seed in a specific slot.

        Args:
            blueprint_id: Blueprint to instantiate (e.g., "norm", "attention")
            seed_id: Unique identifier for the seed
            slot: Target slot ("early", "mid", "late")
            blend_algorithm_id: Blending algorithm ("linear", "sigmoid", "gated")
        """
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")

        self.seed_slots[slot].germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
            blend_algorithm_id=blend_algorithm_id,
        )

    def cull_seed(self, *, slot: str) -> None:
        """Cull the seed in a specific slot."""
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")
        self.seed_slots[slot].cull()

    def get_seed_parameters(self, slot: str | None = None):
        """Get seed parameters from specific slot or all slots."""
        if slot:
            # Must use 'yield from' not 'return' - function with yield is a generator,
            # and 'return' in a generator doesn't return a value, it raises StopIteration
            yield from self.seed_slots[slot].get_parameters()
        else:
            for s in self.seed_slots.values():
                yield from s.get_parameters()

    def get_host_parameters(self):
        """Return host backbone parameters only (exclude seed slots)."""
        for name, param in self.host.named_parameters():
            if "slots" in name:
                continue
            yield param

    @property
    def has_active_seed(self) -> bool:
        """Check if any slot has an active seed."""
        return any(s.is_active for s in self.seed_slots.values())

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has active seed."""
        return self.seed_slots[slot].is_active

    def get_slot_states(self) -> dict:
        """Get state of all slots."""
        return {
            slot_id: slot.state
            for slot_id, slot in self.seed_slots.items()
        }

    @property
    def active_seed_params(self) -> int:
        """Total trainable params across all active seeds."""
        return sum(s.active_seed_params for s in self.seed_slots.values())

    @property
    def total_params(self) -> int:
        """Total trainable params (host + active seeds)."""
        host_params = sum(p.numel() for p in self.host.parameters() if p.requires_grad)
        return host_params + self.active_seed_params

    def count_active_seeds(self) -> int:
        """Count seeds currently active (not fossilized or terminal).

        Uses is_terminal_stage to exclude FOSSILIZED and failure stages,
        preventing double-counting in total_seeds().
        """
        return sum(
            1 for s in self.seed_slots.values()
            if s.is_active and s.state and not is_terminal_stage(s.state.stage)
        )

    def count_fossilized_seeds(self) -> int:
        """Count fossilized seeds across all slots."""
        return sum(
            1 for s in self.seed_slots.values()
            if s.state and s.state.stage == SeedStage.FOSSILIZED
        )

    def total_seeds(self) -> int:
        """Count all seeds (active + fossilized)."""
        return self.count_active_seeds() + self.count_fossilized_seeds()

__all__ = [
    "ConvBlock",
    "CNNHost",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "TransformerHost",
    "MorphogeneticModel",
]
