"""Kasmina Host - The graftable host network.

The MorphogeneticModel is the host network that accepts seed grafts.
It manages the injection points where seeds can be attached.
"""

from __future__ import annotations

from typing import Any, Generator, TYPE_CHECKING, cast

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import AlphaAlgorithm, SeedStage, is_terminal_stage
from esper.kasmina.slot import QualityGates, SeedSlot
from esper.kasmina.blueprints.cnn import ConvBlock  # Reuse shared building block

if TYPE_CHECKING:
    from esper.leyline import InjectionSpec, SeedStateReport
    from esper.tamiyo.policy.features import TaskConfig
    from esper.kasmina.protocol import HostProtocol


class CNNHost(nn.Module):
    """CNN backbone with segment routing for external slot attachment.

    Provides segment boundaries for MorphogeneticModel to attach SeedSlots.
    The host itself performs no slot application - it routes activations
    between segment boundaries.

    Architecture Immutability:
        Following standard nn.Module conventions, the network topology
        (blocks, layers, channel dimensions) is fixed after __init__.
        Properties derived from architecture (segment_channels, _segment_to_block)
        are cached and remain valid for the module's lifetime.

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
        kernel_size: int = 3,
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
            blocks.append(ConvBlock(in_c, out_c, kernel_size=kernel_size))
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2, 2)

        # Classifier maps final channels → logits
        self.classifier = nn.Linear(in_c, num_classes)

    def injection_specs(self) -> list["InjectionSpec"]:
        """Return segment boundaries as InjectionSpec objects.

        Returns:
            List of InjectionSpec, one per block, sorted by network position.
        """
        from esper.leyline import InjectionSpec
        from esper.leyline.slot_id import format_slot_id

        specs = []
        for i in range(self.n_blocks):
            block = self.blocks[i]
            assert isinstance(block, ConvBlock)  # Guaranteed by __init__
            specs.append(
                InjectionSpec(
                    slot_id=format_slot_id(0, i),
                    channels=block.conv.out_channels,
                    position=(i + 1) / self.n_blocks,
                    layer_range=(i, i + 1),
                )
            )
        return specs

    @functools.cached_property
    def segment_channels(self) -> dict[str, int]:
        """Slot ID to channel dimension mapping (cached; architecture-derived)."""
        return {spec.slot_id: spec.channels for spec in self.injection_specs()}

    @functools.cached_property
    def _segment_to_block(self) -> dict[str, int]:
        """Slot ID to block index mapping (cached; architecture-derived)."""
        return {spec.slot_id: spec.layer_range[0] for spec in self.injection_specs()}

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel dimension. Alias for segment_channels."""
        return self.segment_channels

    @property
    def topology(self) -> str:
        """Return 'cnn' for this CNN backbone."""
        return "cnn"

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
            segment: Target segment (e.g., "r0c0", "r0c1", "r0c2")
            x: Raw input if from_segment is None, else features at from_segment boundary
            from_segment: Starting point (None = network input)

        Returns:
            Features at segment boundary
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")
        if from_segment is not None and from_segment not in self.segment_channels:
            raise ValueError(f"Unknown from_segment: {from_segment}. Available: {list(self.segment_channels.keys())}")

        # Only convert at entry point (avoid redundant conversion in chained calls)
        if from_segment is None and self._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        # Use cached mapping (avoid per-call dict rebuilding)
        target_block = self._segment_to_block[segment]
        start_block = 0 if from_segment is None else self._segment_to_block[from_segment] + 1

        # Forward through blocks in range [start_block, target_block]
        for idx in range(start_block, target_block + 1):
            x = self.blocks[idx](x)
            if idx < self._pool_layers:
                x = self.pool(x)

        return x

    def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        """Forward from a segment to output.

        Args:
            segment: Starting segment ID ("r0c0", "r0c1", or "r0c2")
            x: Feature map at segment boundary (already in correct memory format)

        Returns:
            Classification logits
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

        # No memory format conversion needed - tensor already converted by forward_to_segment
        # Use cached mapping
        start_block = self._segment_to_block[segment]

        # Forward through remaining blocks
        for idx in range(start_block + 1, self.n_blocks):
            x = self.blocks[idx](x)
            if idx < self._pool_layers:
                x = self.pool(x)

        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN backbone (no slot application)."""
        # Convert to channels_last ONCE before processing for Tensor Core optimization
        # PERF NOTE: This is idempotent and cheap if tensor is already channels_last.
        # For maximum throughput, configure data loaders to produce channels_last tensors
        # directly (e.g., transforms.ConvertImageDtype() + memory_format arg).
        if self._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            # Only pool on first pool_layers blocks (avoids 0x0 spatial on deep nets)
            if idx < self._pool_layers:
                x = self.pool(x)

        # flatten() handles memory format conversion automatically (returns contiguous)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)  # type: ignore[no-any-return]


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

        return self.resid_dropout(self.c_proj(y))  # type: ignore[no-any-return]


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
        return self.dropout(x)  # type: ignore[no-any-return]


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
    """Transformer backbone with segment routing for external slot attachment.

    Provides segment boundaries for MorphogeneticModel to attach SeedSlots.
    The host itself performs no slot application - it routes hidden states
    between segment boundaries.

    Architecture Immutability:
        Following standard nn.Module conventions, the network topology
        (layers, embedding dimensions, segment boundaries) is fixed after __init__.
        Properties derived from architecture (segment_channels) are cached and
        remain valid for the module's lifetime.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
        num_segments: int = 3,
    ):
        super().__init__()
        if n_layer % num_segments != 0:
            raise ValueError(
                f"n_layer ({n_layer}) must be divisible by num_segments ({num_segments})"
            )
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.num_segments = num_segments

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Pre-allocate position indices buffer to avoid per-forward allocation
        # persistent=False excludes from state_dict (reconstructed from block_size)
        self.pos_indices: torch.Tensor
        self.register_buffer('pos_indices', torch.arange(block_size), persistent=False)

        self.layers = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: tok_emb is master
        self.head.weight = self.tok_emb.weight

        # Compute layer range boundaries for segments
        layers_per_segment = n_layer // num_segments
        self._segment_boundaries = {}
        for i in range(num_segments):
            from esper.leyline.slot_id import format_slot_id
            slot_id = format_slot_id(0, i)
            end_layer = (i + 1) * layers_per_segment
            self._segment_boundaries[slot_id] = end_layer

    def injection_specs(self) -> list["InjectionSpec"]:
        """Return available injection points as InjectionSpec objects.

        Returns:
            List of InjectionSpec, one per segment, sorted by network position.
        """
        from esper.leyline import InjectionSpec
        from esper.leyline.slot_id import format_slot_id

        specs = []
        layers_per_segment = self.n_layer // self.num_segments
        for i in range(self.num_segments):
            start_layer = i * layers_per_segment
            end_layer = (i + 1) * layers_per_segment
            specs.append(
                InjectionSpec(
                    slot_id=format_slot_id(0, i),
                    channels=self.n_embd,
                    position=(i + 1) / self.num_segments,
                    layer_range=(start_layer, end_layer),
                )
            )
        return specs

    @functools.cached_property
    def segment_channels(self) -> dict[str, int]:
        """Slot ID to embedding dimension mapping (cached; architecture-derived)."""
        return {spec.slot_id: spec.channels for spec in self.injection_specs()}

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> embedding dimension. Alias for segment_channels."""
        return self.segment_channels

    @property
    def topology(self) -> str:
        """Return 'transformer' for this transformer backbone."""
        return "transformer"

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
            segment: Target segment (e.g., "r0c0", "r0c1", "r0c2")
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
            # Use pre-allocated buffer to avoid per-forward allocation
            h = self.drop(self.tok_emb(x) + self.pos_emb(self.pos_indices[:T]))
            start_layer = 0
        else:
            h = x
            start_layer = self._segment_boundaries[from_segment]

        # Forward through layers in range [start_layer, end_layer)
        end_layer = self._segment_boundaries[segment]
        for i in range(start_layer, end_layer):
            h = self.layers[i](h)

        return h  # type: ignore[no-any-return]

    def forward_from_segment(self, segment: str, h: torch.Tensor) -> torch.Tensor:
        """Forward from a segment boundary to output logits.

        Args:
            segment: Starting segment ID ("r0c0", "r0c1", or "r0c2")
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

        # Output
        h = self.ln_f(h)
        return self.head(h)  # type: ignore[no-any-return]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer backbone (no slot application)."""
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # Embeddings - use pre-allocated buffer to avoid per-forward allocation
        h = self.drop(self.tok_emb(x) + self.pos_emb(self.pos_indices[:T]))

        # Transformer layers (no slot application)
        for layer in self.layers:
            h = layer(h)

        # Output
        h = self.ln_f(h)
        return self.head(h)  # type: ignore[no-any-return]


# =============================================================================
# Morphogenetic Model
# =============================================================================


class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slots registered into host injection points.

    Multi-slot architecture for managing multiple concurrent seeds at different
    network segments using canonical slot IDs (r0c0, r0c1, r0c2).
    """

    def __init__(
        self,
        host: "HostProtocol",
        device: str = "cpu",
        *,
        slots: list[str],
        task_config: "TaskConfig | None" = None,
        fast_mode: bool = False,
        permissive_gates: bool = False,
    ):
        super().__init__()
        self.host: HostProtocol = host
        self._device = torch.device(device) if isinstance(device, str) else device
        self.task_config = task_config
        self.permissive_gates = permissive_gates

        # Host must expose segment_channels for multi-slot support
        segment_channels = host.segment_channels

        # Create quality gates (permissive mode lets Tamiyo learn thresholds via rewards)
        gates = QualityGates(permissive=permissive_gates)

        # Create seed slots as ModuleDict for proper submodule registration
        slots_dict: dict[str, SeedSlot] = {}
        for slot_id in slots:
            if slot_id not in segment_channels:
                raise ValueError(
                    f"Unknown slot: {slot_id}. Available: {list(segment_channels.keys())}"
                )
            slots_dict[slot_id] = SeedSlot(
                slot_id=slot_id,
                channels=segment_channels[slot_id],
                device=device,
                gates=gates,
                task_config=task_config,
                fast_mode=fast_mode,
            )
        self.seed_slots: nn.ModuleDict = nn.ModuleDict(slots_dict)

        # Track slot order for forward pass (derived from host's injection_specs)
        self._slot_order = [spec.slot_id for spec in host.injection_specs()]
        self._active_slots = [s for s in self._slot_order if s in self.seed_slots]

        # Move host to device
        # Host must be an nn.Module (all HostProtocol implementers are nn.Module)
        cast(nn.Module, self.host).to(device)

    def to(self, *args: Any, **kwargs: Any) -> "MorphogeneticModel":
        """Override to() to update device tracking after transfer."""
        result = super().to(*args, **kwargs)

        # Query actual device from parameters (canonical source of truth)
        try:
            actual_device = next(self.parameters()).device
        except StopIteration:
            return result

        # Update tracking for all slots
        for slot_module in self.seed_slots.values():
            cast(SeedSlot, slot_module).device = actual_device
        # Store as torch.device for type consistency with slot.device
        self._device = actual_device

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through host with all active slots.

        Processes sequentially through network segments, applying slot
        transformations at each segment boundary.
        """
        # If no active slots, use host's forward directly
        if not self._active_slots:
            return self.host.forward(x)

        # Process through active slots
        prev_segment = None
        for slot_id in self._active_slots:
            x = self.host.forward_to_segment(slot_id, x, from_segment=prev_segment)
            # Use __call__ to preserve hooks/wrappers (e.g., torch.compile, profilers).
            slot_obj: SeedSlot = self.seed_slots[slot_id]  # type: ignore[assignment]
            x = slot_obj(x)
            prev_segment = slot_id
        assert prev_segment is not None, (
            "prev_segment unexpectedly None after processing _active_slots. "
            "This indicates a control flow bug - the early return should handle empty slots."
        )
        return self.host.forward_from_segment(prev_segment, x)

    def _get_expected_alpha_shape(self, batch_size: int) -> tuple[int, ...]:
        """Return expected shape for alpha_override tensors based on host topology.

        Args:
            batch_size: Total batch size (K * B for fused forward).

        Returns:
            Expected shape: (batch_size, 1, 1, 1) for CNN, (batch_size, 1, 1) for transformer.
        """
        if self.host.topology == "cnn":
            return (batch_size, 1, 1, 1)
        else:  # transformer
            return (batch_size, 1, 1)

    def fused_forward(self, x: torch.Tensor, alpha_overrides: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fused forward pass for multiple alpha configurations (Zero-Sync Validation).

        Args:
            x: Expanded input tensor of shape [K * B, C, H, W] (CNN) or [K * B, T, C] (transformer).
            alpha_overrides: Dict mapping slot_id -> tensor of shape [K * B, 1, 1, 1] (CNN)
                or [K * B, 1, 1] (transformer) for broadcasting.

        Returns:
            Output logits for all configurations [K * B, num_classes].

        Raises:
            ValueError: If alpha_overrides contains keys not in seed_slots.
            AssertionError: If alpha_override shape doesn't match expected topology shape.
        """
        if not self._active_slots:
            return self.host.forward(x)

        # Validate all alpha_overrides keys exist in seed_slots (fail-fast on typos)
        unknown_keys = set(alpha_overrides) - set(self.seed_slots)
        if unknown_keys:
            raise ValueError(
                f"Unknown alpha_overrides keys: {sorted(unknown_keys)}. "
                f"Available slots: {sorted(self.seed_slots.keys())}"
            )

        # Validate alpha_override shapes early (fail-fast on shape mismatches)
        expected_shape = self._get_expected_alpha_shape(x.shape[0])
        topology = self.host.topology
        for slot_id, alpha in alpha_overrides.items():
            assert alpha.shape == expected_shape, (
                f"alpha_override for slot '{slot_id}' has shape {tuple(alpha.shape)}, "
                f"expected {expected_shape} for {topology} topology. "
                f"Shape mismatch causes silent broadcasting errors or wrong alpha application."
            )

        prev_segment = None
        for slot_id in self._active_slots:
            x = self.host.forward_to_segment(slot_id, x, from_segment=prev_segment)
            # Call slot with alpha_override tensor
            override = alpha_overrides.get(slot_id)
            slot_obj: SeedSlot = self.seed_slots[slot_id]  # type: ignore[assignment]
            x = slot_obj(x, alpha_override=override)
            prev_segment = slot_id
        assert prev_segment is not None, (
            "prev_segment unexpectedly None after processing _active_slots. "
            "This indicates a control flow bug - the early return should handle empty slots."
        )
        return self.host.forward_from_segment(prev_segment, x)

    def germinate_seed(
        self,
        blueprint_id: str,
        seed_id: str,
        *,
        slot: str,
        blend_algorithm_id: str = "sigmoid",
        blend_tempo_epochs: int = 5,
        alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD,
        alpha_target: float | None = None,
    ) -> None:
        """Germinate a new seed in a specific slot.

        Args:
            blueprint_id: Blueprint to instantiate (e.g., "norm", "attention")
            seed_id: Unique identifier for the seed
            slot: Target slot ("r0c0", "r0c1", "r0c2")
            blend_algorithm_id: Blending algorithm ("linear", "sigmoid", "gated")
            blend_tempo_epochs: Number of epochs for blending (3, 5, or 8)
            alpha_algorithm: Blend operator / gating mode (ADD, MULTIPLY, or GATE)
            alpha_target: Initial blend target (defaults to full amplitude)
        """
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")

        slot_obj: SeedSlot = self.seed_slots[slot]  # type: ignore[assignment]
        slot_obj.germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=cast(nn.Module, self.host),
            blend_algorithm_id=blend_algorithm_id,
            blend_tempo_epochs=blend_tempo_epochs,
            alpha_algorithm=alpha_algorithm,
            alpha_target=alpha_target,
        )

    def prune_seed(self, *, slot: str) -> None:
        """Prune the seed in a specific slot (immediate removal)."""
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")
        slot_obj: SeedSlot = self.seed_slots[slot]  # type: ignore[assignment]
        slot_obj.prune()

    def get_seed_parameters(self, slot: str | None = None) -> Generator[torch.nn.Parameter, None, None]:
        """Get seed parameters from specific slot or all slots."""
        if slot:
            # Must use 'yield from' not 'return' - function with yield is a generator,
            # and 'return' in a generator doesn't return a value, it raises StopIteration
            yield from cast(SeedSlot, self.seed_slots[slot]).get_parameters()
        else:
            for slot_module in self.seed_slots.values():
                yield from cast(SeedSlot, slot_module).get_parameters()

    def get_host_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        """Return host backbone parameters only (exclude seed slots)."""
        for name, param in cast(nn.Module, self.host).named_parameters():
            if "slots" in name:
                continue
            yield param

    @property
    def has_active_seed(self) -> bool:
        """Check if any slot has an active seed."""
        return any(
            cast(SeedSlot, slot).is_active
            for slot in self.seed_slots.values()
        )

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has active seed."""
        slot_obj: SeedSlot = self.seed_slots[slot]  # type: ignore[assignment]
        return slot_obj.is_active

    def get_slot_reports(self) -> dict[str, SeedStateReport]:
        """Return per-slot SeedStateReport for all slots (active or not).

        Slots without an active state will not appear in the dict.
        """
        reports: dict[str, SeedStateReport] = {}
        for slot_id, slot_module in self.seed_slots.items():
            slot = cast(SeedSlot, slot_module)
            if slot.state is None:
                continue
            reports[slot_id] = slot.state.to_report()
        return reports

    @property
    def active_seed_params(self) -> int:
        """Total trainable params across all active seeds."""
        return sum(
            cast(SeedSlot, slot).active_seed_params
            for slot in self.seed_slots.values()
        )

    @property
    def total_params(self) -> int:
        """Total trainable params (host + active seeds).

        Uses set() to deduplicate parameters, which is necessary for
        TransformerHost with weight tying (tok_emb shares weights with head).
        Without deduplication, tied weights would be counted twice.
        """
        # Deduplicate to handle weight tying in TransformerHost
        host_params = sum(p.numel() for p in set(cast(nn.Module, self.host).parameters()) if p.requires_grad)
        return host_params + self.active_seed_params

    def count_active_seeds(self) -> int:
        """Count seeds currently active (not fossilized or terminal).

        Uses is_terminal_stage to exclude FOSSILIZED and failure stages,
        preventing double-counting in total_seeds().
        """
        return sum(
            1 for slot_module in self.seed_slots.values()
            if (slot := cast(SeedSlot, slot_module)).is_active
            and slot.state
            and not is_terminal_stage(slot.state.stage)
        )

    def count_fossilized_seeds(self) -> int:
        """Count fossilized seeds across all slots."""
        return sum(
            1 for slot_module in self.seed_slots.values()
            if (slot := cast(SeedSlot, slot_module)).state
            and slot.state.stage == SeedStage.FOSSILIZED
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
