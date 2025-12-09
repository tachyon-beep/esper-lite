"""Transformer Blueprints - Seed modules for transformer hosts."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .registry import BlueprintRegistry


# Check for FlexAttention availability (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX_ATTENTION = True
except ImportError:
    _HAS_FLEX_ATTENTION = False


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
            nn.init.zeros_(self.up.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.up(self.down(x))

    return LoRASeed(dim, rank)


@BlueprintRegistry.register(
    "attention", "transformer", param_estimate=50000, description="Additional self-attention head"
)
def create_transformer_attention_seed(dim: int, n_head: int = 4) -> nn.Module:
    """Additional self-attention head seed."""

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

            # Use SDPA for automatic Flash Attention optimization
            out = F.scaled_dot_product_attention(q, k, v)

            out = out.transpose(1, 2).reshape(b, t, c)
            return x + self.proj(out)

    return TransformerAttentionSeed(dim, n_head)


@BlueprintRegistry.register(
    "mlp", "transformer", param_estimate=1200000, description="Additional MLP (4x expansion)"
)
def create_transformer_mlp_seed(dim: int, expansion: int = 4, checkpoint: bool = False) -> nn.Module:
    """Additional MLP seed with optional activation checkpointing."""

    class TransformerMLPSeed(nn.Module):
        def __init__(self, dim: int, expansion: int, use_checkpoint: bool):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.use_checkpoint = use_checkpoint
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_checkpoint and self.training and x.requires_grad:
                return x + torch_checkpoint(self._mlp_forward, x, use_reentrant=False)
            return x + self._mlp_forward(x)

    return TransformerMLPSeed(dim, expansion, checkpoint)


# FlexAttention blueprint - conditionally registered
if _HAS_FLEX_ATTENTION:
    @BlueprintRegistry.register(
        "flex_attention", "transformer", param_estimate=55000,
        description="FlexAttention with causal mask (PyTorch 2.5+)"
    )
    def create_flex_attention_seed(dim: int, n_head: int = 4) -> nn.Module:
        """FlexAttention seed with customizable attention patterns.

        Uses block mask caching for improved performance on repeated sequence lengths.
        Cache keys include (seq_len, device_str, dtype) for correctness across
        devices and mixed-precision scenarios.
        """

        class FlexAttentionSeed(nn.Module):
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
                self._block_mask_cache: OrderedDict[tuple[int, str, str], object] = OrderedDict()

            def _apply(self, fn):
                """Clear block mask cache on device/dtype transfer to prevent stale masks."""
                self._block_mask_cache.clear()
                return super()._apply(fn)

            def _get_causal_block_mask(
                self, seq_len: int, device: torch.device, dtype: torch.dtype
            ):
                """Get or create cached causal block mask with LRU eviction."""
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
                    def causal(b, h, q_idx, kv_idx):
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
                out = flex_attention(q, k, v, block_mask=block_mask)

                out = out.transpose(1, 2).reshape(b, t, c)
                return x + self.proj(out)

        return FlexAttentionSeed(dim, n_head)


__all__ = []
