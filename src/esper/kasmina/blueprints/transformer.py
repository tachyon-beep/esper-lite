"""Transformer Blueprints - Seed modules for transformer hosts."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .registry import BlueprintRegistry


@BlueprintRegistry.register("norm", "transformer", param_estimate=800, description="LayerNorm only")
def create_transformer_norm_seed(dim: int, **kwargs: Any) -> nn.Module:
    """LayerNorm enhancement seed for transformers."""
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/norm: {sorted(kwargs)}")

    class TransformerNormSeed(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.scale = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Bound scale to [-1, 1] via tanh to prevent gradient explosion
            return x + torch.tanh(self.scale) * (self.norm(x) - x)  # type: ignore[no-any-return]

    return TransformerNormSeed(dim)


@BlueprintRegistry.register("lora", "transformer", param_estimate=6000, description="Low-rank adapter (rank=8)")
def create_lora_seed(dim: int, rank: int = 8, **kwargs: Any) -> nn.Module:
    """Low-rank adapter seed (LoRA-style)."""
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/lora: {sorted(kwargs)}")

    class LoRASeed(nn.Module):
        def __init__(self, dim: int, rank: int):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.up(self.down(x))  # type: ignore[no-any-return]

    return LoRASeed(dim, rank)


@BlueprintRegistry.register(
    "lora_large", "transformer", param_estimate=25000,
    description="Large low-rank adapter (rank=32) - more expressive than standard LoRA"
)
def create_lora_large_seed(dim: int, rank: int = 32, **kwargs: Any) -> nn.Module:
    """Large low-rank adapter seed - more expressive than standard LoRA.

    Uses rank=32 instead of rank=8, providing 4× more capacity for
    adaptation while still being parameter-efficient.

    For dim=384 with rank=32:
    - Down: 384 * 32 = 12,288 params
    - Up: 32 * 384 = 12,288 params
    - Total: 24,576 params
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/lora_large: {sorted(kwargs)}")

    class LoRALargeSeed(nn.Module):
        def __init__(self, dim: int, rank: int):
            super().__init__()
            self.down = nn.Linear(dim, rank, bias=False)
            self.up = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.up(self.down(x))  # type: ignore[no-any-return]

    return LoRALargeSeed(dim, rank)


@BlueprintRegistry.register(
    "attention", "transformer", param_estimate=591000,
    description="Additional self-attention head (~591k params for dim=384)"
)
def create_transformer_attention_seed(
    dim: int, n_head: int = 4, **kwargs: Any
) -> nn.Module:
    """Additional self-attention head seed.

    For dim=384 with n_head=4:
    - QKV projection: 384 * (3 * 384) + 1152 = 443,520 params
    - Output projection: 384 * 384 + 384 = 147,840 params
    - Total: 591,360 params
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/attention: {sorted(kwargs)}")
    if dim % n_head != 0:
        raise ValueError(f"Transformer attention seed requires dim % n_head == 0, got dim={dim}, n_head={n_head}")

    class TransformerAttentionSeed(nn.Module):
        def __init__(self, dim: int, n_head: int):
            super().__init__()
            self.n_head = n_head
            self.head_dim = dim // n_head

            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape

            qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Use SDPA with causal mask to match host's autoregressive attention
            # (PyTorch Expert review 2025-12-09: enables Flash Attention causal kernel)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            out = out.transpose(1, 2).reshape(b, t, c)
            return x + self.proj(out)  # type: ignore[no-any-return]

    return TransformerAttentionSeed(dim, n_head)


@BlueprintRegistry.register(
    "mlp_small", "transformer", param_estimate=591000,
    description="Small MLP (2x expansion) - same tier as attention (~591k params)"
)
def create_transformer_mlp_small_seed(
    dim: int, expansion: int = 2, checkpoint: bool = False, **kwargs: Any
) -> nn.Module:
    """Small MLP seed with 2× expansion - half the params of full 4× MLP.

    Uses 2× expansion instead of 4×, halving parameters while still
    providing meaningful non-linear transformation capacity.

    For dim=384 with expansion=2:
    - fc1: 384 * 768 + 768 = 295,680 params
    - fc2: 768 * 384 + 384 = 295,296 params
    - Total: 590,976 params (vs 1.18M for 4× expansion)
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/mlp_small: {sorted(kwargs)}")

    class TransformerMLPSmallSeed(nn.Module):
        def __init__(self, dim: int, expansion: int, use_checkpoint: bool):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.use_checkpoint = use_checkpoint
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))  # type: ignore[no-any-return]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_checkpoint and self.training and x.requires_grad:
                return x + torch_checkpoint(self._mlp_forward, x, use_reentrant=False)  # type: ignore[no-any-return]
            return x + self._mlp_forward(x)

    return TransformerMLPSmallSeed(dim, expansion, checkpoint)


@BlueprintRegistry.register(
    "mlp", "transformer", param_estimate=1200000, description="Additional MLP (4x expansion)"
)
def create_transformer_mlp_seed(
    dim: int, expansion: int = 4, checkpoint: bool = False, **kwargs: Any
) -> nn.Module:
    """Additional MLP seed with optional activation checkpointing."""
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/mlp: {sorted(kwargs)}")

    class TransformerMLPSeed(nn.Module):
        def __init__(self, dim: int, expansion: int, use_checkpoint: bool):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.use_checkpoint = use_checkpoint
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))  # type: ignore[no-any-return]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_checkpoint and self.training and x.requires_grad:
                return x + torch_checkpoint(self._mlp_forward, x, use_reentrant=False)  # type: ignore[no-any-return]
            return x + self._mlp_forward(x)

    return TransformerMLPSeed(dim, expansion, checkpoint)


@BlueprintRegistry.register(
    "flex_attention", "transformer", param_estimate=591000,
    description="FlexAttention with causal mask (~591k params for dim=384)"
)
def create_flex_attention_seed(dim: int, n_head: int = 4, **kwargs: Any) -> nn.Module:
    """FlexAttention seed with customizable attention patterns.

    Uses PyTorch's FlexAttention API for block-sparse attention patterns.
    Same parameter count as standard attention seed.

    For dim=384 with n_head=4:
    - QKV projection: 384 * (3 * 384) + 1152 = 443,520 params
    - Output projection: 384 * 384 + 384 = 147,840 params
    - Total: 591,360 params
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/flex_attention: {sorted(kwargs)}")
    if dim % n_head != 0:
        raise ValueError(
            f"FlexAttention seed requires dim % n_head == 0, got dim={dim}, n_head={n_head}"
        )

    class FlexAttentionSeed(nn.Module):
        """Flexible attention with block-sparse patterns.

        Warning:
            The block_mask cache uses OrderedDict operations that cause graph breaks
            under torch.compile. Consider pre-computing masks or using
            mode="reduce-overhead" if compilation is critical.
        """

        _MAX_CACHE_SIZE = 8  # LRU-style bound on cached masks

        def __init__(self, dim: int, n_head: int):
            super().__init__()
            self.n_head = n_head
            self.head_dim = dim // n_head

            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

            # LRU cache for block masks: (seq_len, device_str, dtype_str) -> BlockMask
            # PERF NOTE: Cache lookup in forward() creates a function call boundary that
            # prevents torch.compile from fusing block_mask creation with attention.
            # This is acceptable because:
            #   1. Block mask creation is expensive (~10ms) - caching is essential
            #   2. The actual flex_attention kernel is fused internally
            #   3. For fixed sequence lengths, the cache hit rate is 100% after warmup
            # If profiling shows this as a bottleneck for known fixed seq_lens, consider
            # pre-computing common masks in __init__:
            #   self._precomputed_masks = {64: create_block_mask(...), 128: ..., etc}
            self._block_mask_cache: OrderedDict[tuple[int, str, str], object] = OrderedDict()

        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            """Clear block mask cache on device/dtype transfer to prevent stale masks."""
            self._block_mask_cache.clear()
            return super()._apply(fn, recurse)  # type: ignore[no-untyped-call]

        @torch.compiler.disable  # type: ignore[untyped-decorator]  # B3-PT-03: public API
        def _get_causal_block_mask(
            self, seq_len: int, device: torch.device, dtype: torch.dtype
        ) -> Any:
            """Get or create cached causal block mask with LRU eviction.

            Note: OrderedDict cache operations cause graph breaks under torch.compile
            (move_to_end, popitem, dict mutations). Disabled from Dynamo tracing.
            """
            # Use str(device) and str(dtype) for reliable dict equality
            key = (seq_len, str(device), str(dtype))
            if key in self._block_mask_cache:
                # Cache hit: mark as recently used
                self._block_mask_cache.move_to_end(key)
            else:
                # Cache miss: evict LRU entry if full
                if len(self._block_mask_cache) >= self._MAX_CACHE_SIZE:
                    self._block_mask_cache.popitem(last=False)

                # create_block_mask uses boolean mask function (no score param)
                def causal(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
                    return q_idx >= kv_idx

                self._block_mask_cache[key] = create_block_mask(
                    causal,
                    B=None,  # Batch-independent
                    H=None,  # Head-independent
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )
            return self._block_mask_cache[key]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape

            qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Use cached block mask for efficient causal attention
            block_mask = self._get_causal_block_mask(t, x.device, x.dtype)
            # B3-PT-02: flex_attention returns Tensor in PyTorch 2.5+
            # Removed isinstance check that caused torch.compile graph break
            out = flex_attention(q, k, v, block_mask=block_mask)
            # Type narrowing: flex_attention returns Tensor when return_lse=False (default)
            assert isinstance(out, torch.Tensor)

            out = out.transpose(1, 2).reshape(b, t, c)
            return x + self.proj(out)  # type: ignore[no-any-return]

    return FlexAttentionSeed(dim, n_head)


@BlueprintRegistry.register("noop", "transformer", param_estimate=0, description="Identity seed")
def create_transformer_noop_seed(dim: int, **kwargs: Any) -> nn.Module:
    """No-op blueprint for transformer (identity function).

    Used when the agent selects NOOP blueprint for a transformer host.
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for transformer/noop: {sorted(kwargs)}")
    return nn.Identity()


__all__ = []
