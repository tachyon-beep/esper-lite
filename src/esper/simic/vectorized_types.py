from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from esper.leyline import AlphaAlgorithm, LifecycleOp

if TYPE_CHECKING:
    from esper.leyline.episode_outcome import EpisodeOutcome
    from esper.simic.rewards import ContributionRewardInputs, LossRewardInputs
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
class EnvStepRecord:
    """Per-environment, per-step mutable workspace for the vectorized training loop.

    Pre-allocated ONCE per episode (not per epoch). Holds the same pre-allocated
    ActionSpec/ActionOutcome/ActionMaskFlags objects by reference — they are
    mutated in place each step (identical to current action_execution.py:560-574).
    RewardSummaryAccumulator is held by reference for the same reason.

    Per-step output fields (reward_components, episode_reward, final_accuracy,
    episode_outcome, rollback_occurred, truncated) are reset to sentinel by
    execute_actions before each step (I16).

    Distinct from ParallelEnvState — that owns durable per-episode state
    (model, optimizers, episode_rewards). This record owns ephemeral per-step
    outputs consumed across phase boundaries within one epoch.

    RESUME SEAM (HEALTH_REPORT open item): ContributionRewardInputs and
    LossRewardInputs held here are epoch-hot-path objects, not GPU tensors.
    The del step_records at batch end is for Python object release, not GPU
    segment release (GPU segments are released via env_state synchronize at
    vectorized_trainer.py:2366-2368).
    """
    env_idx: int
    # Composed objects held by reference (same pre-allocated instances as before)
    action_spec: "ActionSpec"
    action_outcome: "ActionOutcome"
    mask_flags: "ActionMaskFlags"
    reward_summary: "RewardSummaryAccumulator"  # also passed as reward_summary_accum kwarg
    # Pre-allocated reward-input objects (mutated in place, not reallocated per epoch)
    contribution_reward_inputs: "ContributionRewardInputs"
    loss_reward_inputs: "LossRewardInputs"
    # Per-episode accumulators (zeroed at episode start via reset_episode)
    rollback_occurred: bool = False      # was: env_rollback_occurred[env_idx]
    env_final_acc: float = 0.0           # was: env_final_accs[env_idx]
    env_total_reward: float = 0.0        # was: env_total_rewards[env_idx]

    def reset_episode(self) -> None:
        """Reset episode-level fields. Called once per episode start."""
        self.rollback_occurred = False
        self.env_final_acc = 0.0
        self.env_total_reward = 0.0


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
    topology_manifests: list[dict[str, Any]] = field(default_factory=list)

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
        data["topology_manifests"] = self.topology_manifests
        return data


__all__ = [
    "ActionSpec",
    "ActionMaskFlags",
    "ActionOutcome",
    "RewardSummaryAccumulator",
    "EpisodeRecord",
    "BatchSummary",
    "EnvStepRecord",
]
