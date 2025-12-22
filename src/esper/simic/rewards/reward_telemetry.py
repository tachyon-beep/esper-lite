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
    ratio_penalty: float = 0.0  # Penalty for high contribution with low/negative improvement (ransomware)

    # Penalties
    compute_rent: float = 0.0
    alpha_shock: float = 0.0  # Convex penalty on alpha deltas (Phase 5)
    blending_warning: float = 0.0  # Escalating penalty for negative trajectory during BLENDING
    holding_warning: float = 0.0  # Escalating penalty for WAITing in HOLDING

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0  # Terminal bonus from fossilized seed count
    num_fossilized_seeds: int = 0  # Total fossilized seeds for debugging
    num_contributing_fossilized: int = 0  # Seeds with total_improvement >= MIN_FOSSILIZE_CONTRIBUTION

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

    @property
    def shaped_reward_ratio(self) -> float:
        """Fraction of total reward from shaping terms.

        High values (> 0.5) suggest potential reward hacking - the agent may
        be optimizing for shaping bonuses rather than actual value creation.

        Returns:
            Ratio of |shaped terms| / |total reward|, or 0.0 if total is negligible.

        Note:
            M1: Uses 1e-8 threshold for zero-guard because reward magnitudes
            in this system are typically O(0.1) to O(10). Rewards below 1e-8
            indicate either (a) true zero or (b) near-perfect cancellation of
            positive/negative terms - in either case, the ratio is meaningless.
        """
        # M1: Guard against division by zero/near-zero
        # 1e-8 is well below minimum meaningful reward magnitude (~0.01)
        if abs(self.total_reward) < 1e-8:
            return 0.0
        shaped = self.stage_bonus + self.pbrs_bonus + self.action_shaping
        return abs(shaped) / abs(self.total_reward)

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
            "ratio_penalty": self.ratio_penalty,
            "compute_rent": self.compute_rent,
            "alpha_shock": self.alpha_shock,
            "blending_warning": self.blending_warning,
            "holding_warning": self.holding_warning,
            "stage_bonus": self.stage_bonus,
            "pbrs_bonus": self.pbrs_bonus,
            "action_shaping": self.action_shaping,
            "terminal_bonus": self.terminal_bonus,
            "fossilize_terminal_bonus": self.fossilize_terminal_bonus,
            "num_fossilized_seeds": self.num_fossilized_seeds,
            "num_contributing_fossilized": self.num_contributing_fossilized,
            "action_name": self.action_name,
            "action_success": self.action_success,
            "seed_stage": self.seed_stage,
            "epoch": self.epoch,
            "val_acc": self.val_acc,
            "acc_at_germination": self.acc_at_germination,
            "host_baseline_acc": self.host_baseline_acc,
            "growth_ratio": self.growth_ratio,
            "total_reward": self.total_reward,
            "shaped_reward_ratio": self.shaped_reward_ratio,
        }


__all__ = ["RewardComponentsTelemetry"]
