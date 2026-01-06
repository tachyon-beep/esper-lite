"""PPO update math helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class RatioMetrics:
    """Computed ratio/KL metrics for a PPO update epoch."""

    per_head_ratios: dict[str, torch.Tensor]
    joint_ratio: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    clip_fraction_positive: torch.Tensor
    clip_fraction_negative: torch.Tensor
    ratio_stats: torch.Tensor
    early_stop: torch.Tensor
    per_head_ratio_max: dict[str, torch.Tensor]
    joint_ratio_max: torch.Tensor


@dataclass(slots=True)
class LossMetrics:
    """Loss components for a PPO update epoch."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass(slots=True)
class PPOUpdateResult:
    """Combined update math results for a PPO epoch."""

    ratio_metrics: RatioMetrics
    loss_metrics: LossMetrics | None


def compute_ratio_metrics(
    *,
    log_probs: dict[str, torch.Tensor],
    old_log_probs: dict[str, torch.Tensor],
    head_masks: dict[str, torch.Tensor],
    clip_ratio: float,
    target_kl: float | None,
    head_names: tuple[str, ...],
    total_timesteps: torch.Tensor,
) -> RatioMetrics:
    """Compute PPO ratio statistics, KL, and clip metrics."""
    per_head_ratios: dict[str, torch.Tensor] = {}
    log_ratios_for_joint: dict[str, torch.Tensor] = {}
    for key in head_names:
        log_ratio = log_probs[key] - old_log_probs[key]
        log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
        log_ratios_for_joint[key] = log_ratio_clamped
        per_head_ratios[key] = torch.exp(log_ratio_clamped)

    with torch.inference_mode():
        per_head_ratio_max = {key: per_head_ratios[key].max() for key in head_names}

        stacked_log_ratios = torch.stack([log_ratios_for_joint[k] for k in head_names])
        joint_log_ratio = stacked_log_ratios.sum(dim=0)
        joint_log_ratio_clamped = torch.clamp(joint_log_ratio, min=-30.0, max=30.0)
        joint_ratio = torch.exp(joint_log_ratio_clamped)
        joint_ratio_max = joint_ratio.max()

        weighted_kl_sum = torch.tensor(0.0, device=joint_ratio.device)
        total_weight = torch.tensor(0.0, device=joint_ratio.device)
        for key in head_names:
            mask = head_masks[key]
            log_ratio_clamped = log_ratios_for_joint[key]
            kl_per_step = (torch.exp(log_ratio_clamped) - 1) - log_ratio_clamped
            n_valid = mask.sum().float().clamp(min=1)
            head_kl = (kl_per_step * mask).sum() / n_valid
            causal_weight = n_valid / total_timesteps
            weighted_kl_sum = weighted_kl_sum + causal_weight * head_kl
            total_weight = total_weight + causal_weight
        approx_kl = weighted_kl_sum / total_weight.clamp(min=1e-8)

        clip_fraction_t = ((joint_ratio - 1.0).abs() > clip_ratio).float().mean()
        clip_pos = (joint_ratio > 1.0 + clip_ratio).float().mean()
        clip_neg = (joint_ratio < 1.0 - clip_ratio).float().mean()

        if target_kl is None:
            early_stop = torch.tensor(False, device=approx_kl.device)
        else:
            early_stop = approx_kl > (1.5 * target_kl)

        ratio_stats = torch.stack([
            joint_ratio.mean(),
            joint_ratio.max(),
            joint_ratio.min(),
            joint_ratio.std(),
        ])

    return RatioMetrics(
        per_head_ratios=per_head_ratios,
        joint_ratio=joint_ratio,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction_t,
        clip_fraction_positive=clip_pos,
        clip_fraction_negative=clip_neg,
        ratio_stats=ratio_stats,
        early_stop=early_stop,
        per_head_ratio_max=per_head_ratio_max,
        joint_ratio_max=joint_ratio_max,
    )


def compute_losses(
    *,
    per_head_ratios: dict[str, torch.Tensor],
    per_head_advantages: dict[str, torch.Tensor],
    head_masks: dict[str, torch.Tensor],
    values: torch.Tensor,
    normalized_returns: torch.Tensor,
    old_values: torch.Tensor,
    entropy: dict[str, torch.Tensor],
    entropy_coef_per_head: dict[str, float],
    entropy_coef: float,
    clip_ratio: float,
    clip_value: bool,
    value_clip: float,
    value_coef: float,
    head_names: tuple[str, ...],
) -> LossMetrics:
    """Compute PPO policy/value/entropy losses for a single epoch."""
    policy_loss: torch.Tensor = torch.tensor(0.0, device=values.device)
    for key in head_names:
        ratio = per_head_ratios[key]
        adv = per_head_advantages[key]
        mask = head_masks[key]

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        clipped_surr = torch.min(surr1, surr2)

        n_valid = mask.sum().clamp(min=1)
        head_loss = -(clipped_surr * mask).sum() / n_valid
        policy_loss = policy_loss + head_loss

    if clip_value:
        values_clipped = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
        value_loss_unclipped = (values - normalized_returns) ** 2
        value_loss_clipped = (values_clipped - normalized_returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = F.mse_loss(values, normalized_returns)

    entropy_loss: torch.Tensor = torch.tensor(0.0, device=values.device)
    for key in entropy:
        head_coef = entropy_coef_per_head[key]
        mask = head_masks[key]
        n_valid = mask.sum().clamp(min=1)
        masked_ent = (entropy[key] * mask).sum() / n_valid
        entropy_loss = entropy_loss - head_coef * masked_ent

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    return LossMetrics(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        total_loss=total_loss,
    )


__all__ = [
    "RatioMetrics",
    "LossMetrics",
    "PPOUpdateResult",
    "compute_ratio_metrics",
    "compute_losses",
]
