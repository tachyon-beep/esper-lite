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


class FinitenessGateFailure(TypedDict):
    """Diagnostics for an epoch skipped by the finiteness gate."""

    epoch: int
    sources: list[str]


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
    finiteness_gate_failures: list[FinitenessGateFailure]  # One entry per skipped epoch

    # Scalar metrics (aggregated across epochs)
    policy_loss: float
    value_loss: float
    entropy_loss: float
    entropy_floor_penalty: float  # Per-head entropy floor penalty (for calibration debugging)
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
    # Value function metrics (TELE-220 to TELE-228)
    v_return_correlation: float
    td_error_mean: float
    td_error_std: float
    bellman_error: float
    return_p10: float
    return_p50: float
    return_p90: float
    return_variance: float
    return_skewness: float
    # CUDA memory metrics (infrastructure monitoring)
    cuda_memory_allocated_gb: float
    cuda_memory_reserved_gb: float
    cuda_memory_peak_gb: float
    cuda_memory_fragmentation: float
    throughput_step_time_ms_sum: float
    throughput_dataloader_wait_ms_sum: float
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
    conditional_head_entropies: dict[str, list[float]]  # Entropy when head is causally relevant
    head_grad_norms: dict[str, list[float]]  # Per-head, per-epoch
    ratio_diagnostic: dict[str, Any]
    # Q-values (Policy V2 op-conditioned critic)
    op_q_values: tuple[float, ...]
    op_valid_mask: tuple[bool, ...]
    q_variance: float
    q_spread: float
    # Per-head NaN/Inf detection (for indicator lights)
    head_nan_detected: dict[str, bool]
    head_inf_detected: dict[str, bool]
    # LSTM hidden state health (TELE-340)
    lstm_h_l2_total: float | None
    lstm_c_l2_total: float | None
    lstm_h_rms: float | None
    lstm_c_rms: float | None
    lstm_h_env_rms_mean: float | None
    lstm_h_env_rms_max: float | None
    lstm_c_env_rms_mean: float | None
    lstm_c_env_rms_max: float | None
    lstm_h_max: float | None
    lstm_c_max: float | None
    lstm_has_nan: bool | None
    lstm_has_inf: bool | None
    # D5: Slot Saturation Diagnostics
    # Track forced WAIT steps to understand PPO stability under slot saturation
    forced_step_ratio: float  # Fraction of timesteps with forced decisions (no agency)
    usable_actor_timesteps: int  # Count of timesteps where agent had real choice
    decision_density: float  # Fraction with agency (1 - forced_step_ratio), higher = healthier
    advantage_std_floored: bool  # True if advantage std was clamped to floor (degenerate batch)
    pre_norm_advantage_std: float  # Raw std before normalization (for diagnostics)

    # Auxiliary contribution supervision metrics (Phase 4.1)
    # DRL Expert: Monitor for prediction collapse and quality
    aux_contribution_loss: float  # Raw auxiliary MSE loss value
    effective_aux_coef: float  # Current warmup-scaled coefficient
    aux_pred_variance: float  # Prediction variance - warn if < 0.01 (collapse)
    aux_explained_variance: float  # Should increase over training
    aux_pred_target_correlation: float  # Should be > 0.5 eventually


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
    "FinitenessGateFailure",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
