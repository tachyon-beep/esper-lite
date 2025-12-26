"""Seed blueprint initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class FinalLayerRef:
    module: nn.Module
    name: str


def find_final_affine_layer(module: nn.Module) -> FinalLayerRef | None:
    """Find the last affine layer (Linear/Conv*) in a module.

    This is used for identity-like initialization of residual seeds where the
    last affine layer controls the seed's output delta.
    """
    final: FinalLayerRef | None = None
    for name, child in module.named_modules():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            final = FinalLayerRef(module=child, name=name)
    return final


def zero_init_final_layer(module: nn.Module, *, allow_missing: bool = False) -> None:
    """Zero-initialize the module's final affine layer.

    This is an explicit Phase 3 pre-implementation risk reduction tool:
    - For residual seeds (`seed(x) = x + delta(x)`), zeroing the final affine
      makes `delta(x) ≈ 0` at birth, so blend operators start near identity.

    Args:
        module: Seed module to initialize.
        allow_missing: If True, modules with no affine layers are left untouched.
            If False, missing affine layers raises ValueError.
    """
    found = find_final_affine_layer(module)
    if found is None:
        if allow_missing:
            return
        raise ValueError(f"No affine layer (Linear/Conv*) found in module {type(module).__name__}")

    layer = found.module
    if layer.weight is None:
        raise ValueError(f"Final affine layer {found.name!r} has no weight tensor")

    nn.init.zeros_(layer.weight)  # type: ignore[arg-type]
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)  # type: ignore[arg-type]


__all__ = [
    "FinalLayerRef",
    "find_final_affine_layer",
    "zero_init_final_layer",
]
