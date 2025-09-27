"""Multi-seed gradient aggregation utilities for Tolaria.

This module provides aggregation schemes and optional PCGrad conflict
projection. Integration with the training loop is deferred until seeds
provide per-seed gradients.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Tuple

import torch

_PCGRAD_EPS = 1e-12


def grads_to_flat(params: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    flats: list[torch.Tensor] = []
    shapes: list[torch.Size] = []
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    for p in params:
        if not isinstance(p, torch.Tensor):
            continue
        shapes.append(p.shape)
        if device is None:
            device = p.device
        elif p.device != device:
            raise RuntimeError("gradient tensors must share the same device")
        if dtype is None:
            dtype = p.dtype
        elif p.dtype != dtype:
            raise RuntimeError("gradient tensors must share the same dtype")
        flats.append(p.reshape(-1))

    if not flats:
        raise RuntimeError("no gradients available for aggregation")

    flat = torch.cat(flats)
    if device is not None and flat.device != device:
        flat = flat.to(device=device)
    if dtype is not None and flat.dtype != dtype:
        flat = flat.to(dtype=dtype)

    return flat, shapes


def flat_to_grads(flat: torch.Tensor, shapes: List[torch.Size]) -> list[torch.Tensor]:
    if flat.ndim != 1:
        flat = flat.reshape(-1)

    element_counts: list[int] = []
    for shp in shapes:
        count = 1
        for dim in shp:
            count *= int(dim)
        element_counts.append(count)

    expected = sum(element_counts)
    if flat.numel() != expected:
        raise RuntimeError(
            f"flat gradient has {flat.numel()} elements but {expected} were expected"
        )

    grads: list[torch.Tensor] = []
    offset = 0
    for shp, count in zip(shapes, element_counts):
        chunk = flat.narrow(0, offset, count)
        grads.append(chunk.reshape(shp))
        offset += count

    return grads


def pcgrad(grad_i: torch.Tensor, grad_j: torch.Tensor, *, epsilon: float = _PCGRAD_EPS) -> torch.Tensor:
    """Project ``grad_i`` to reduce conflict with ``grad_j`` following PCGrad."""

    dot = torch.dot(grad_i, grad_j)
    if dot >= 0:
        return grad_i

    denom = torch.dot(grad_j, grad_j)
    if torch.abs(denom) <= epsilon:
        return grad_i

    proj = dot / (denom + epsilon) * grad_j
    return grad_i - proj


def aggregate_mean(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.stack(grads, dim=0), dim=0)


def _build_weight_tensor(grads: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not weights:
        raise RuntimeError("weighted aggregation requires non-empty weights")
    if len(weights) != len(grads):
        raise RuntimeError("weights and gradients must have matching lengths")

    base = grads[0]
    dtype = base.dtype
    device = base.device
    if not all(g.dtype == dtype for g in grads[1:]):
        raise RuntimeError("all gradients must share dtype for weighted aggregation")
    if not all(g.device == device for g in grads[1:]):
        raise RuntimeError("all gradients must share device for weighted aggregation")

    weight_tensor = torch.tensor(weights, dtype=dtype, device=device)
    total = torch.sum(weight_tensor)
    if torch.isnan(total) or torch.isinf(total) or torch.abs(total) <= _PCGRAD_EPS:
        raise RuntimeError("invalid weight normalisation")
    return weight_tensor / total


def _reshape_weights(weights: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if grad.dim() == 1:
        return weights.view(-1, 1)
    shape = [-1] + [1 for _ in range(grad.dim())]
    return weights.view(*shape)


def aggregate_weighted(grads: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not all(g.shape == grads[0].shape for g in grads[1:]):
        raise RuntimeError("all gradients must share shape for weighted aggregation")

    norm_weights = _build_weight_tensor(grads, weights)
    stacked = torch.stack(grads, dim=0)
    scaled = stacked * _reshape_weights(norm_weights, grads[0])
    return torch.sum(scaled, dim=0)


def aggregate_sum(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.sum(torch.stack(grads, dim=0), dim=0)


def combine_flat_grads(
    flats: list[torch.Tensor], *, use_pcgrad: bool = True, weights: list[float] | None = None
) -> tuple[torch.Tensor, int]:
    """Combine flat gradients, returning the merged flat tensor and conflicts count.

    Conflicts count is the number of negative dot-product encounters used to
    trigger PCGrad projections.
    """
    if not flats:
        raise RuntimeError("no gradients available for aggregation")

    conflicts = 0
    if use_pcgrad and len(flats) >= 2:
        adjusted = [grad.clone() for grad in flats]
        order = list(range(len(adjusted)))
        random.shuffle(order)

        conflict_pairs: set[tuple[int, int]] = set()
        for idx in order:
            grad_i = adjusted[idx]
            for jdx in order:
                if jdx == idx:
                    continue
                grad_j = adjusted[jdx]
                dot = torch.dot(grad_i, grad_j)
                if dot < 0:
                    conflict_pairs.add((min(idx, jdx), max(idx, jdx)))
                    grad_i = pcgrad(grad_i, grad_j)
            adjusted[idx] = grad_i

        conflicts = len(conflict_pairs)
        flats = adjusted
    if weights is None:
        return aggregate_sum(flats), conflicts
    return aggregate_weighted(flats, weights), conflicts


__all__ = [
    "pcgrad",
    "aggregate_mean",
    "aggregate_weighted",
    "aggregate_sum",
    "grads_to_flat",
    "flat_to_grads",
    "combine_flat_grads",
]
