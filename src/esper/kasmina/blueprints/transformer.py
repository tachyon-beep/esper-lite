"""Transformer Blueprints - Seed modules for transformer hosts."""

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
def create_transformer_mlp_seed(dim: int, expansion: int = 4) -> nn.Module:
    """Additional MLP seed."""

    class TransformerMLPSeed(nn.Module):
        def __init__(self, dim: int, expansion: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.gelu(self.fc1(x)))

    return TransformerMLPSeed(dim, expansion)


__all__ = []
