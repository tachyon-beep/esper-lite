from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.leyline import LifecycleOp, LossRewardConfig
    from esper.simic.rewards.rewards import ContributionRewardConfig, SeedInfo


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
]
