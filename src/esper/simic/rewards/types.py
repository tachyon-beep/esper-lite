from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from esper.leyline import SeedStage

if TYPE_CHECKING:
    from esper.leyline import LifecycleOp, LossRewardConfig
    from esper.simic.rewards.contribution import ContributionRewardConfig


class SeedInfo(NamedTuple):
    """Minimal seed information for reward computation.

    Designed to avoid importing heavy classes in the hot path.
    Stage values match SeedStage IntEnum:
        0=UNKNOWN, 1=DORMANT, 2=GERMINATED, 3=TRAINING,
        4=BLENDING, 6=HOLDING, 7=FOSSILIZED, etc. (5 skipped)
    """

    stage: int  # SeedStage.value
    improvement_since_stage_start: float
    total_improvement: float  # Since germination (for G5 gate alignment)
    epochs_in_stage: int
    seed_params: int = 0  # Trainable params of active seed
    previous_stage: int = 0  # For PBRS stage bonus calculation
    previous_epochs_in_stage: int = 0  # Epochs in previous stage at transition (for PBRS telescoping)
    seed_age_epochs: int = 0  # Total epochs since germination (for rent grace)
    # Scaffolding support (Phase 3.1)
    interaction_sum: float = 0.0  # Total synergy with other seeds
    boost_received: float = 0.0  # Strongest single interaction

    @staticmethod
    def from_seed_state(seed_state: Any, seed_params: int = 0) -> "SeedInfo | None":
        """Convert from kasmina.SeedState to SeedInfo.

        Args:
            seed_state: The seed state from kasmina, or None
            seed_params: Trainable parameter count of the active seed module

        Returns:
            SeedInfo or None if no seed state
        """
        if seed_state is None:
            return None
        metrics = seed_state.metrics
        improvement = 0.0
        total_improvement = 0.0
        seed_age = 0
        interaction_sum = 0.0
        boost_received = 0.0
        if metrics:
            improvement = metrics.current_val_accuracy - metrics.accuracy_at_stage_start
            total_improvement = metrics.total_improvement
            seed_age = metrics.epochs_total
            interaction_sum = metrics.interaction_sum
            boost_received = metrics.boost_received
        return SeedInfo(
            stage=seed_state.stage.value,
            improvement_since_stage_start=improvement,
            total_improvement=total_improvement,
            epochs_in_stage=seed_state.epochs_in_stage,
            seed_params=seed_params,
            previous_stage=seed_state.previous_stage.value,
            previous_epochs_in_stage=seed_state.previous_epochs_in_stage,
            seed_age_epochs=seed_age,
            interaction_sum=interaction_sum,
            boost_received=boost_received,
        )


# Stage constants from leyline contract
STAGE_GERMINATED = SeedStage.GERMINATED.value
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
STAGE_HOLDING = SeedStage.HOLDING.value


@dataclass(slots=True)
class ContributionRewardInputs:
    action: LifecycleOp
    seed_contribution: float | None
    val_acc: float
    seed_info: SeedInfo | None
    epoch: int
    max_epochs: int
    total_params: int
    host_params: int
    acc_at_germination: float | None
    acc_delta: float
    committed_val_acc: float | None = None
    fossilized_seed_params: int = 0
    num_fossilized_seeds: int = 0
    num_contributing_fossilized: int = 0
    config: ContributionRewardConfig | None = None
    return_components: bool = False
    effective_seed_params: float | None = None
    alpha_delta_sq_sum: float = 0.0
    stable_val_acc: float | None = None
    escrow_credit_prev: float = 0.0
    slot_id: str | None = None
    seed_id: str | None = None


@dataclass(slots=True)
class LossRewardInputs:
    action: LifecycleOp
    loss_delta: float
    val_loss: float
    seed_info: SeedInfo | None
    epoch: int
    max_epochs: int
    total_params: int = 0
    host_params: int = 1
    config: LossRewardConfig | None = None


__all__ = [
    "ContributionRewardInputs",
    "LossRewardInputs",
    "SeedInfo",
    "STAGE_GERMINATED",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_HOLDING",
]
