"""Leyline Episode Outcome - Multi-objective training outcome for Pareto analysis.

EpisodeOutcome captures the key metrics for comparing training runs across
multiple objectives (accuracy, efficiency, stability). It's used by:
- simic: Vectorized training emits outcomes
- karn: Stores and analyzes outcomes for Pareto frontiers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EpisodeOutcome:
    """Multi-objective outcome for Pareto analysis.

    Captures the key metrics we're optimizing:
    - final_accuracy: Task performance (higher = better)
    - param_ratio: Parameter efficiency (lower = better)
    - stability_score: Training stability (higher = better)
    """

    env_id: int
    episode_idx: int
    final_accuracy: float
    param_ratio: float  # total_params / host_params
    num_fossilized: int
    num_contributing_fossilized: int  # Seeds that contributed to learning
    episode_reward: float  # Total reward for the episode
    stability_score: float  # 1 - variance(recent_losses)
    reward_mode: str  # "shaped", "simplified", etc.
    timestamp: datetime = field(default_factory=_utc_now)

    def dominates(self, other: "EpisodeOutcome") -> bool:
        """Pareto dominance check.

        Returns True if self dominates other (better or equal on all objectives,
        strictly better on at least one).

        Objectives (higher is better):
        - final_accuracy
        - stability_score

        Objectives (lower is better):
        - param_ratio
        """
        # Check: self >= other on all objectives
        geq_accuracy = self.final_accuracy >= other.final_accuracy
        geq_stability = self.stability_score >= other.stability_score
        leq_ratio = self.param_ratio <= other.param_ratio

        all_geq = geq_accuracy and geq_stability and leq_ratio

        # Check: self > other on at least one objective
        gt_accuracy = self.final_accuracy > other.final_accuracy
        gt_stability = self.stability_score > other.stability_score
        lt_ratio = self.param_ratio < other.param_ratio

        any_gt = gt_accuracy or gt_stability or lt_ratio

        return all_geq and any_gt

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "env_id": self.env_id,
            "episode_idx": self.episode_idx,
            "reward_mode": self.reward_mode,
            "final_accuracy": self.final_accuracy,
            "param_ratio": self.param_ratio,
            "stability_score": self.stability_score,
            "num_fossilized": self.num_fossilized,
            "num_contributing_fossilized": self.num_contributing_fossilized,
            "episode_reward": self.episode_reward,
            "timestamp": self.timestamp.isoformat(),
        }


__all__ = ["EpisodeOutcome"]
