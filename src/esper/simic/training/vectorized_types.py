from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from esper.leyline import AlphaAlgorithm, LifecycleOp

if TYPE_CHECKING:
    from esper.leyline.episode_outcome import EpisodeOutcome
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry


@dataclass(slots=True)
class ActionSpec:
    slot_idx: int = 0
    blueprint_idx: int = 0
    style_idx: int = 0
    tempo_idx: int = 0
    alpha_target_idx: int = 0
    alpha_speed_idx: int = 0
    alpha_curve_idx: int = 0
    op_idx: int = 0
    target_slot: str = ""
    slot_is_enabled: bool = False
    action_valid_for_reward: bool = False
    action_for_reward: LifecycleOp = LifecycleOp.WAIT
    blend_algorithm_id: str = ""
    alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD
    alpha_target: float = 0.0


@dataclass(slots=True)
class ActionMaskFlags:
    op_masked: bool = False
    slot_masked: bool = False
    blueprint_masked: bool = False
    style_masked: bool = False
    tempo_masked: bool = False
    alpha_target_masked: bool = False
    alpha_speed_masked: bool = False
    alpha_curve_masked: bool = False


@dataclass(slots=True)
class ActionOutcome:
    action_success: bool = False
    action_name: str = ""
    reward_raw: float = 0.0
    reward_normalized: float = 0.0
    reward_components: RewardComponentsTelemetry | None = None
    rollback_occurred: bool = False
    truncated: bool = False
    episode_reward: float | None = None
    final_accuracy: float | None = None
    episode_outcome: EpisodeOutcome | None = None


@dataclass(slots=True)
class RewardSummaryAccumulator:
    bounded_attribution: float = 0.0
    compute_rent: float = 0.0
    alpha_shock: float = 0.0
    hindsight_credit: float = 0.0
    total_reward: float = 0.0
    count: int = 0
    scaffold_count: int = 0
    scaffold_delay_total: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bounded_attribution": self.bounded_attribution,
            "compute_rent": self.compute_rent,
            "alpha_shock": self.alpha_shock,
            "hindsight_credit": self.hindsight_credit,
            "total_reward": self.total_reward,
            "count": self.count,
            "scaffold_count": self.scaffold_count,
            "scaffold_delay_total": self.scaffold_delay_total,
        }


@dataclass(slots=True)
class EpisodeRecord:
    env_id: int
    episode_reward: float
    final_accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "env_id": self.env_id,
            "episode_reward": self.episode_reward,
            "final_accuracy": self.final_accuracy,
        }


@dataclass(slots=True)
class BatchSummary:
    batch: int
    episodes: int
    avg_accuracy: float
    rolling_avg_accuracy: float
    metrics: dict[str, Any]
    reward_summary: list[RewardSummaryAccumulator]
    episode_history: list[EpisodeRecord]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "batch": self.batch,
            "episodes": self.episodes,
            "avg_accuracy": self.avg_accuracy,
            "rolling_avg_accuracy": self.rolling_avg_accuracy,
        }
        data.update(self.metrics)
        data["reward_summary"] = [
            summary.to_dict() for summary in self.reward_summary
        ]
        data["episode_history"] = [
            record.to_dict() for record in self.episode_history
        ]
        return data


__all__ = [
    "ActionSpec",
    "ActionMaskFlags",
    "ActionOutcome",
    "RewardSummaryAccumulator",
    "EpisodeRecord",
    "BatchSummary",
]
