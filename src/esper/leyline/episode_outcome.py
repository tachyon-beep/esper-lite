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


# Worst-acceptable param_ratio for Pareto hypervolume reference points.
# param_ratio is the growth multiple total/host (1.0 = no growth). The reference
# (worst-case) point sits at 2.0 = the model has doubled in size; episodes whose
# growth meets or exceeds this contribute nothing to the dominated hypervolume.
MAX_PARAM_RATIO_REF: float = 2.0


@dataclass(frozen=True)
class EpisodeOutcome:
    """Multi-objective outcome for Pareto analysis.

    Captures the key metrics we're optimizing:
    - final_accuracy: Task performance (higher = better)
    - param_ratio: Parameter growth ratio (lower = better)
    - stability_score: Training stability (higher = better)
    """

    env_id: int
    episode_idx: int
    final_accuracy: float
    # param_ratio = total_params / host_params. This is the single canonical
    # semantic for the field across producer (simic), contract (leyline), Karn
    # views/SQL, Pareto analysis, and the proof packet. It is a growth multiple,
    # NOT an overage:
    #   1.0 = no growth (total == host)
    #   1.2 = 20% growth (total is 1.2x the host)
    # "minimize" therefore means "prefer less parameter growth"; accuracy ROI is
    # final_accuracy / param_ratio (accuracy per unit of grown model size).
    param_ratio: float
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
        - param_ratio (growth multiple total/host; 1.0 = no growth)
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


__all__ = ["EpisodeOutcome", "MAX_PARAM_RATIO_REF"]
