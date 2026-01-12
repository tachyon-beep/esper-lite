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
    stable_val_acc: float | None = None  # Stable accuracy used for escrow/progress gating
    attribution_discount: float = 1.0  # Sigmoid discount for negative total_improvement
    ratio_penalty: float = 0.0  # Penalty for high contribution with low/negative improvement (ransomware)

    # Escrow attribution (RewardMode.ESCROW)
    escrow_credit_prev: float = 0.0
    escrow_credit_target: float = 0.0
    escrow_delta: float = 0.0
    escrow_credit_next: float = 0.0
    escrow_forfeit: float = 0.0  # Terminal clawback for non-fossilized escrow credit

    # Penalties
    compute_rent: float = 0.0
    alpha_shock: float = 0.0  # Convex penalty on alpha deltas (Phase 5)
    blending_warning: float = 0.0  # Escalating penalty for negative trajectory during BLENDING
    holding_warning: float = 0.0  # Escalating penalty for WAITing in HOLDING

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    synergy_bonus: float = 0.0  # Scaffolding bonus for positive interactions
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0  # Terminal bonus from fossilized seed count
    hindsight_credit: float = 0.0  # Scaffold contribution credit applied at fossilization
    num_fossilized_seeds: int = 0  # Total fossilized seeds for debugging
    num_contributing_fossilized: int = 0  # Seeds with total_improvement >= MIN_FOSSILIZE_CONTRIBUTION

    # D2: Capacity Economics (slot saturation prevention)
    occupancy_rent: float = 0.0  # Per-epoch cost for seeds above free_slots threshold
    fossilized_rent: float = 0.0  # Per-epoch maintenance cost for fossilized seeds
    first_germinate_bonus: float = 0.0  # One-time bonus for first germination (breaks "do nothing" symmetry)
    n_active_seeds: int = 0  # Count of active seeds (for diagnostics)

    # D3: Anti-Timing-Gaming (early germination discount)
    timing_discount: float = 1.0  # Discount factor for early germination [discount_floor, 1.0]

    # Drip reward (BASIC_PLUS mode post-fossilization accountability)
    drip_this_epoch: float = 0.0  # Sum of drip payouts from all fossilized seeds this epoch

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

        PyTorch Expert Review 2025-12-26: Added all shaping terms to computation.
        Previous version only included 4 of 10 shaping terms.
        """
        # M1: Guard against division by zero/near-zero
        # 1e-8 is well below minimum meaningful reward magnitude (~0.01)
        if abs(self.total_reward) < 1e-8:
            return 0.0
        # All shaping terms (bonuses and penalties that are not primary signal)
        shaped = (
            # Bonuses
            self.stage_bonus
            + self.pbrs_bonus
            + self.synergy_bonus
            + self.action_shaping
            + self.terminal_bonus
            + self.fossilize_terminal_bonus
            + self.hindsight_credit
            # Penalties (these shape behavior, so include in total)
            + self.compute_rent
            + self.alpha_shock
            + self.blending_warning
            + self.holding_warning
            + self.ratio_penalty
            + self.escrow_forfeit
            # D2: Capacity economics (shaping terms)
            - self.occupancy_rent  # Already negative in reward, store as positive
            - self.fossilized_rent  # Already negative in reward, store as positive
            + self.first_germinate_bonus
        )
        return abs(shaped) / abs(self.total_reward)

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        """Convert to dict for TelemetryEvent data field.

        Uses explicit dict construction instead of asdict() for 3-5x performance
        improvement in hot path (PyTorch Expert recommendation).
        """
        return {
            "base_acc_delta": self.base_acc_delta,
            "seed_contribution": self.seed_contribution,
            "bounded_attribution": self.bounded_attribution,
            "progress_since_germination": self.progress_since_germination,
            "stable_val_acc": self.stable_val_acc,
            "attribution_discount": self.attribution_discount,
            "ratio_penalty": self.ratio_penalty,
            "escrow_credit_prev": self.escrow_credit_prev,
            "escrow_credit_target": self.escrow_credit_target,
            "escrow_delta": self.escrow_delta,
            "escrow_credit_next": self.escrow_credit_next,
            "escrow_forfeit": self.escrow_forfeit,
            "compute_rent": self.compute_rent,
            "alpha_shock": self.alpha_shock,
            "blending_warning": self.blending_warning,
            "holding_warning": self.holding_warning,
            "stage_bonus": self.stage_bonus,
            "pbrs_bonus": self.pbrs_bonus,
            "synergy_bonus": self.synergy_bonus,
            "action_shaping": self.action_shaping,
            "terminal_bonus": self.terminal_bonus,
            "fossilize_terminal_bonus": self.fossilize_terminal_bonus,
            "hindsight_credit": self.hindsight_credit,
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
            # D2: Capacity economics
            "occupancy_rent": self.occupancy_rent,
            "fossilized_rent": self.fossilized_rent,
            "first_germinate_bonus": self.first_germinate_bonus,
            "n_active_seeds": self.n_active_seeds,
            # D3: Anti-timing-gaming
            "timing_discount": self.timing_discount,
            # Drip reward (BASIC_PLUS mode)
            "drip_this_epoch": self.drip_this_epoch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int | str | bool | None]) -> "RewardComponentsTelemetry":
        """Reconstruct from dict (inverse of to_dict).

        Args:
            data: Dictionary from to_dict() or JSON deserialization.

        Returns:
            RewardComponentsTelemetry instance.

        Note:
            shaped_reward_ratio is a computed property and is ignored during
            reconstruction (it will be recalculated from the component values).
        """
        return cls(
            base_acc_delta=float(data["base_acc_delta"]),  # type: ignore[arg-type]
            seed_contribution=float(data["seed_contribution"]) if data["seed_contribution"] is not None else None,
            bounded_attribution=float(data["bounded_attribution"]) if data["bounded_attribution"] is not None else None,
            progress_since_germination=float(data["progress_since_germination"]) if data["progress_since_germination"] is not None else None,
            stable_val_acc=float(data["stable_val_acc"]) if data["stable_val_acc"] is not None else None,
            attribution_discount=float(data["attribution_discount"]),  # type: ignore[arg-type]
            ratio_penalty=float(data["ratio_penalty"]),  # type: ignore[arg-type]
            escrow_credit_prev=float(data["escrow_credit_prev"]),  # type: ignore[arg-type]
            escrow_credit_target=float(data["escrow_credit_target"]),  # type: ignore[arg-type]
            escrow_delta=float(data["escrow_delta"]),  # type: ignore[arg-type]
            escrow_credit_next=float(data["escrow_credit_next"]),  # type: ignore[arg-type]
            escrow_forfeit=float(data["escrow_forfeit"]),  # type: ignore[arg-type]
            compute_rent=float(data["compute_rent"]),  # type: ignore[arg-type]
            alpha_shock=float(data["alpha_shock"]),  # type: ignore[arg-type]
            blending_warning=float(data["blending_warning"]),  # type: ignore[arg-type]
            holding_warning=float(data["holding_warning"]),  # type: ignore[arg-type]
            stage_bonus=float(data["stage_bonus"]),  # type: ignore[arg-type]
            pbrs_bonus=float(data["pbrs_bonus"]),  # type: ignore[arg-type]
            synergy_bonus=float(data["synergy_bonus"]),  # type: ignore[arg-type]
            action_shaping=float(data["action_shaping"]),  # type: ignore[arg-type]
            terminal_bonus=float(data["terminal_bonus"]),  # type: ignore[arg-type]
            fossilize_terminal_bonus=float(data["fossilize_terminal_bonus"]),  # type: ignore[arg-type]
            hindsight_credit=float(data["hindsight_credit"]),  # type: ignore[arg-type]
            num_fossilized_seeds=int(data["num_fossilized_seeds"]),  # type: ignore[arg-type]
            num_contributing_fossilized=int(data["num_contributing_fossilized"]),  # type: ignore[arg-type]
            action_name=str(data["action_name"]),
            action_success=bool(data["action_success"]),
            seed_stage=int(data["seed_stage"]) if data["seed_stage"] is not None else None,
            epoch=int(data["epoch"]),  # type: ignore[arg-type]
            val_acc=float(data["val_acc"]),  # type: ignore[arg-type]
            acc_at_germination=float(data["acc_at_germination"]) if data["acc_at_germination"] is not None else None,
            host_baseline_acc=float(data["host_baseline_acc"]) if data["host_baseline_acc"] is not None else None,
            growth_ratio=float(data["growth_ratio"]),  # type: ignore[arg-type]
            total_reward=float(data["total_reward"]),  # type: ignore[arg-type]
            # D2: Capacity economics
            occupancy_rent=float(data.get("occupancy_rent", 0.0)),  # type: ignore[arg-type]
            fossilized_rent=float(data.get("fossilized_rent", 0.0)),  # type: ignore[arg-type]
            first_germinate_bonus=float(data.get("first_germinate_bonus", 0.0)),  # type: ignore[arg-type]
            n_active_seeds=int(data.get("n_active_seeds", 0)),  # type: ignore[arg-type]
            # D3: Anti-timing-gaming
            timing_discount=float(data.get("timing_discount", 1.0)),  # type: ignore[arg-type]
            # Drip reward (BASIC_PLUS mode)
            drip_this_epoch=float(data.get("drip_this_epoch", 0.0)),  # type: ignore[arg-type]
        )


__all__ = ["RewardComponentsTelemetry"]
