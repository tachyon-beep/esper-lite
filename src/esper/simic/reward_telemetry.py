"""Reward Telemetry Dataclasses.

Captures per-component breakdown of reward computation
for diagnosing reward hacking and tuning reward weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardComponentsTelemetry:
    """Breakdown of reward components for debugging.

    Each field represents one component of the total reward.
    All components should sum to total_reward.
    """

    # Base signal (legacy shaped reward)
    base_acc_delta: float = 0.0

    # Contribution-primary signal
    seed_contribution: float | None = None
    bounded_attribution: float | None = None
    progress_since_germination: float | None = None
    attribution_discount: float = 1.0  # Sigmoid discount for negative total_improvement

    # Penalties
    compute_rent: float = 0.0
    blending_warning: float = 0.0  # Escalating penalty for negative trajectory during BLENDING

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0

    # Context (for debugging) - DRL Expert recommended fields
    action_name: str = ""
    action_success: bool = True
    seed_stage: int | None = None
    epoch: int = 0
    val_acc: float = 0.0
    acc_at_germination: float | None = None
    host_baseline_acc: float | None = None  # Counterfactual baseline
    growth_ratio: float = 0.0  # total_params / host_params

    # Total
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field.

        Uses explicit dict construction instead of asdict() for 3-5x performance
        improvement in hot path (PyTorch Expert recommendation).
        """
        return {
            "base_acc_delta": self.base_acc_delta,
            "seed_contribution": self.seed_contribution,
            "bounded_attribution": self.bounded_attribution,
            "progress_since_germination": self.progress_since_germination,
            "attribution_discount": self.attribution_discount,
            "compute_rent": self.compute_rent,
            "blending_warning": self.blending_warning,
            "stage_bonus": self.stage_bonus,
            "pbrs_bonus": self.pbrs_bonus,
            "action_shaping": self.action_shaping,
            "terminal_bonus": self.terminal_bonus,
            "action_name": self.action_name,
            "action_success": self.action_success,
            "seed_stage": self.seed_stage,
            "epoch": self.epoch,
            "val_acc": self.val_acc,
            "acc_at_germination": self.acc_at_germination,
            "host_baseline_acc": self.host_baseline_acc,
            "growth_ratio": self.growth_ratio,
            "total_reward": self.total_reward,
        }


__all__ = ["RewardComponentsTelemetry"]
