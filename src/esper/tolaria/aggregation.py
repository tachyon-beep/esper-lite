"""Multi-seed gradient aggregation utilities for Tolaria.

This module provides aggregation schemes and optional PCGrad conflict
projection. Integration with the training loop is deferred until seeds
provide per-seed gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch


@dataclass(slots=True)
class AggregationResult:
    conflicts_projected: int = 0


def grads_to_flat(params: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    flats: list[torch.Tensor] = []
    shapes: list[torch.Size] = []
    for p in params:
        g = p if isinstance(p, torch.Tensor) else None
        if g is None:
            continue
        shapes.append(g.shape)
        flats.append(g.reshape(-1))
    if not flats:
        return torch.tensor([]), shapes
    return torch.cat(flats), shapes


def flat_to_grads(flat: torch.Tensor, shapes: List[torch.Size]) -> list[torch.Tensor]:
    grads: list[torch.Tensor] = []
    offset = 0
    for shp in shapes:
        n = int(torch.tensor(shp).prod().item()) if shp else 1
        chunk = flat[offset : offset + n]
        grads.append(chunk.reshape(shp))
        offset += n
    return grads


def pcgrad(grad_i: torch.Tensor, grad_j: torch.Tensor) -> torch.Tensor:
    """Project grad_i to reduce conflict with grad_j as in PCGrad.

    If dot(gi, gj) < 0, project gi onto the orthogonal plane of gj.
    """
    dot = torch.dot(grad_i, grad_j)
    if dot < 0:
        proj = dot / (torch.norm(grad_j) ** 2 + 1e-12) * grad_j
        return grad_i - proj
    return grad_i


def aggregate_mean(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.stack(grads, dim=0), dim=0)


def aggregate_weighted(grads: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=grads[0].dtype, device=grads[0].device)
    g = torch.stack(grads, dim=0)
    return torch.sum(g * w.view(-1, 1), dim=0) / (torch.sum(w) + 1e-12)


def aggregate_sum(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.sum(torch.stack(grads, dim=0), dim=0)


def combine_flat_grads(
    flats: list[torch.Tensor], *, use_pcgrad: bool = True, weights: list[float] | None = None
) -> tuple[torch.Tensor, int]:
    """Combine flat gradients, returning the merged flat tensor and conflicts count.

    Conflicts count is the number of negative dot-product encounters used to
    trigger PCGrad projections.
    """
    conflicts = 0
    if use_pcgrad and len(flats) >= 2:
        adjusted = flats[:]
        base = adjusted[0]
        for i in range(1, len(adjusted)):
            dot = torch.dot(base, adjusted[i])
            if dot < 0:
                conflicts += 1
            base = pcgrad(base, adjusted[i])
        adjusted[0] = base
        flats = adjusted
    if weights is None:
        return aggregate_sum(flats), conflicts
    return aggregate_weighted(flats, weights), conflicts


__all__ = [
    "AggregationResult",
    "pcgrad",
    "aggregate_mean",
    "aggregate_weighted",
    "aggregate_sum",
    "grads_to_flat",
    "flat_to_grads",
    "combine_flat_grads",
]
