"""Type definitions for Simic agent module.

TypedDicts provide type safety for dictionary returns from PPO and network functions.
"""

from __future__ import annotations

from typing import Any, TypedDict

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

    Important: PPOAgent.update() aggregates metrics across epochs before
    returning, so scalar metrics are float (not list[float]). Only
    head_entropies and head_grad_norms retain per-epoch structure.

    Finiteness Gate Contract:
    - ppo_update_performed: True if at least one epoch completed successfully
    - finiteness_gate_skip_count: Number of epochs skipped due to non-finite values
    - When all epochs skip: ppo_update_performed=False, other metrics are NaN
    - Callers should check ppo_update_performed before using other metrics
    """

    # Update status (finiteness gate contract)
    ppo_update_performed: bool  # True if at least one epoch completed
    finiteness_gate_skip_count: int  # Number of epochs skipped due to NaN/Inf

    # Scalar metrics (aggregated across epochs)
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    entropy: float
    ratio_mean: float
    ratio_max: float
    ratio_min: float
    ratio_std: float  # Standard deviation of importance sampling ratio
    early_stop_epoch: int | None  # None when early stopping didn't occur
    pre_clip_grad_norm: float  # Gradient norm before clipping (for telemetry)
    # Log prob extremes (NaN predictor)
    log_prob_min: float
    log_prob_max: float
    # Per-head ratio max (for detecting per-head ratio explosion)
    head_slot_ratio_max: float
    head_blueprint_ratio_max: float
    head_style_ratio_max: float
    head_tempo_ratio_max: float
    head_alpha_target_ratio_max: float
    head_alpha_speed_ratio_max: float
    head_alpha_curve_ratio_max: float
    head_op_ratio_max: float
    joint_ratio_max: float
    # Structured metrics
    gradient_stats: GradientStats | None
    head_entropies: dict[str, list[float]]  # Per-head, per-epoch
    head_grad_norms: dict[str, list[float]]  # Per-head, per-epoch
    ratio_diagnostic: dict[str, Any]
    # Q-values (Policy V2 op-conditioned critic)
    q_germinate: float
    q_advance: float
    q_fossilize: float
    q_prune: float
    q_wait: float
    q_set_alpha: float
    q_variance: float
    q_spread: float
    # Per-head NaN/Inf detection (for indicator lights)
    head_nan_detected: dict[str, bool]
    head_inf_detected: dict[str, bool]
    # LSTM hidden state health (TELE-340)
    lstm_h_norm: float | None
    lstm_c_norm: float | None
    lstm_h_max: float | None
    lstm_c_max: float | None
    lstm_has_nan: bool | None
    lstm_has_inf: bool | None


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
