"""Type definitions for Simic agent module.

TypedDicts provide type safety for dictionary returns from PPO and network functions.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class GradientStats(TypedDict):
    """Gradient statistics from PPO update."""

    grad_norm: float
    max_grad: float
    min_grad: float


class PPOUpdateMetrics(TypedDict):
    """Metrics from a single PPO update step."""

    policy_loss: list[float]
    value_loss: list[float]
    entropy_loss: list[float]
    total_loss: list[float]
    approx_kl: list[float]
    clip_fraction: list[float]
    explained_variance: float
    gradient_stats: GradientStats | None
    # Per-head entropy (P3-1) - for Task 8
    head_entropies: dict[str, list[float]]


class HeadLogProbs(TypedDict):
    """Per-head log probabilities from factored policy."""

    slot: torch.Tensor
    blueprint: torch.Tensor
    blend: torch.Tensor
    op: torch.Tensor


class HeadEntropies(TypedDict):
    """Per-head entropies from factored policy."""

    slot: torch.Tensor
    blueprint: torch.Tensor
    blend: torch.Tensor
    op: torch.Tensor


class ActionDict(TypedDict):
    """Factored action dictionary."""

    slot: int
    blueprint: int
    blend: int
    op: int


__all__ = [
    "GradientStats",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
