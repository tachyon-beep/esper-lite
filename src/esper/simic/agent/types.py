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


class HeadGradientNorms(TypedDict):
    """Per-head gradient norms from factored policy (P4-6).

    Tracks gradient norm per action head to diagnose if one head
    dominates learning. Complements per-head entropy tracking.
    """

    slot: float
    blueprint: float
    style: float
    tempo: float
    alpha_target: float
    alpha_speed: float
    alpha_curve: float
    op: float
    value: float


class PPOUpdateMetrics(TypedDict, total=False):
    """Metrics from a single PPO update step.

    Note: total=False makes all keys optional since update() may return
    empty dict when buffer is empty, or subset of keys in some cases.
    """

    policy_loss: list[float]
    value_loss: list[float]
    entropy_loss: list[float]
    total_loss: list[float]
    approx_kl: list[float]
    clip_fraction: list[float]
    explained_variance: float
    gradient_stats: GradientStats | None
    # Per-head entropy (P3-1) - for exploring exploration collapse
    head_entropies: dict[str, list[float]]
    # Per-head gradient norms (P4-6) - for diagnosing head dominance
    head_grad_norms: dict[str, list[float]]
    # Additional metrics that may be present
    entropy: list[float]
    ratio_mean: list[float]
    ratio_max: list[float]
    ratio_min: list[float]
    ratio_diagnostic: dict


class HeadLogProbs(TypedDict):
    """Per-head log probabilities from factored policy."""

    slot: torch.Tensor
    blueprint: torch.Tensor
    style: torch.Tensor
    tempo: torch.Tensor
    alpha_target: torch.Tensor
    alpha_speed: torch.Tensor
    alpha_curve: torch.Tensor
    op: torch.Tensor


class HeadEntropies(TypedDict):
    """Per-head entropies from factored policy."""

    slot: torch.Tensor
    blueprint: torch.Tensor
    style: torch.Tensor
    tempo: torch.Tensor
    alpha_target: torch.Tensor
    alpha_speed: torch.Tensor
    alpha_curve: torch.Tensor
    op: torch.Tensor


class ActionDict(TypedDict):
    """Factored action dictionary."""

    slot: int
    blueprint: int
    style: int
    tempo: int
    alpha_target: int
    alpha_speed: int
    alpha_curve: int
    op: int


__all__ = [
    "GradientStats",
    "HeadGradientNorms",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
