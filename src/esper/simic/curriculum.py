"""Blueprint curriculum with UCB1 exploration."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CurriculumStats:
    """UCB1 statistics for a single blueprint in curriculum selection.

    Note: This is distinct from nissa.analytics.BlueprintStats which tracks
    fossilization/culling analytics. This class tracks UCB1 bandit statistics.
    """
    trials: int = 0
    successes: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.trials if self.trials > 0 else 0.0


class BlueprintCurriculum:
    """UCB1-based curriculum for blueprint selection.

    Balances exploration (trying new blueprints) with exploitation
    (using blueprints that have worked well).

    UCB1 formula: mean_reward + c * sqrt(log(total_trials) / blueprint_trials)

    Complexity penalty is applied at reward recording time (not UCB calculation)
    to preserve UCB1's optimism principle and regret guarantees.

    Note: Adjusted rewards can be negative (range [-complexity_penalty, 1.0])
    when complexity penalty is applied to low raw rewards.
    """

    # Theoretical optimal exploration coefficient for [0,1] rewards (Auer et al., 2002)
    SQRT_2 = math.sqrt(2)

    def __init__(
        self,
        blueprints: list[str],
        complexity: list[int],
        exploration_weight: float | None = None,
        complexity_penalty: float = 0.1,
        reward_range: tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize UCB1 curriculum.

        Args:
            blueprints: List of blueprint names
            complexity: List of complexity scores (same length as blueprints)
            exploration_weight: UCB exploration coefficient. If None, uses sqrt(2) scaled
                by reward range (theoretically optimal for bounded rewards).
            complexity_penalty: Weight for complexity penalty applied to rewards.
                Higher values favor simpler blueprints more strongly.
            reward_range: Expected (min, max) reward range for normalization.
        """
        if len(blueprints) != len(complexity):
            raise ValueError("blueprints and complexity must have same length")

        self.blueprints = blueprints
        self.complexity = dict(zip(blueprints, complexity))
        self.complexity_penalty = complexity_penalty
        self._reward_min, self._reward_max = reward_range

        # Default exploration weight: sqrt(2) (theoretically optimal for [0,1] rewards)
        # Scale by effective reward range after complexity adjustment:
        # - Input rewards normalized to [0, 1]
        # - Complexity penalty extends range to [-complexity_penalty, 1.0]
        # - Effective range width = 1.0 + complexity_penalty
        input_reward_scale = self._reward_max - self._reward_min
        effective_reward_scale = input_reward_scale * (1.0 + complexity_penalty)
        self.exploration_weight = (
            exploration_weight if exploration_weight is not None
            else self.SQRT_2 * effective_reward_scale
        )

        self._stats: dict[str, CurriculumStats] = {
            name: CurriculumStats() for name in blueprints
        }
        self._total_trials = 0

        # Normalize complexity to [0, 1] range
        max_complexity = max(complexity) if complexity else 1
        self._normalized_complexity = {
            name: c / max(max_complexity, 1) for name, c in self.complexity.items()
        }

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] range and clip."""
        # Use epsilon comparison to avoid division by near-zero
        if abs(self._reward_max - self._reward_min) < 1e-8:
            return 0.5
        normalized = (reward - self._reward_min) / (self._reward_max - self._reward_min)
        return max(0.0, min(1.0, normalized))

    def record_outcome(self, blueprint: str, success: bool, reward: float) -> None:
        """Record outcome of a blueprint trial.

        Complexity penalty is applied here to the reward signal, preserving
        UCB1's optimism principle in the score calculation.
        """
        if blueprint not in self._stats:
            raise ValueError(f"Unknown blueprint: {blueprint}")

        # Normalize reward to [0, 1]
        normalized_reward = self._normalize_reward(reward)

        # Apply complexity penalty to reward (not UCB score)
        # This preserves optimism in the face of uncertainty
        complexity_adjusted_reward = (
            normalized_reward
            - self.complexity_penalty * self._normalized_complexity[blueprint]
        )

        stats = self._stats[blueprint]
        stats.trials += 1
        stats.total_reward += complexity_adjusted_reward
        if success:
            stats.successes += 1
        self._total_trials += 1

    def get_ucb_scores(self) -> dict[str, float | None]:
        """Compute UCB scores for all blueprints.

        Returns None for unexplored blueprints (they're selected first
        in the initialization phase, no fake score needed).
        """
        scores: dict[str, float | None] = {}

        for name in self.blueprints:
            stats = self._stats[name]

            if stats.trials == 0:
                # Unexplored: no score (initialization phase handles these)
                scores[name] = None
            else:
                # Standard UCB1 formula (no complexity penalty - it's in the reward)
                mean = stats.mean_reward
                exploration = self.exploration_weight * math.sqrt(
                    math.log(self._total_trials) / stats.trials
                )
                scores[name] = mean + exploration

        return scores

    def select_blueprint(self) -> str:
        """Select blueprint using UCB1 algorithm.

        Phase 1 (Initialization): Unexplored blueprints are selected first.
        Phase 2 (UCB1): Select blueprint with highest UCB score.
        """
        # Phase 1: Try all blueprints at least once
        unexplored = [name for name in self.blueprints if self._stats[name].trials == 0]
        if unexplored:
            # Return first unexplored (deterministic for reproducibility)
            return unexplored[0]

        # Phase 2: Standard UCB1 selection
        scores = self.get_ucb_scores()
        # Filter to explored only (scores are not None)
        explored_scores = {k: v for k, v in scores.items() if v is not None}
        return max(explored_scores, key=lambda k: explored_scores[k])

    def get_stats(self, blueprint: str) -> dict:
        """Get statistics for a blueprint.

        Note: mean_reward reflects complexity-adjusted rewards, not raw rewards.
        """
        stats = self._stats[blueprint]
        return {
            "trials": stats.trials,
            "successes": stats.successes,
            "mean_reward": stats.mean_reward,
        }


__all__ = ["BlueprintCurriculum", "CurriculumStats"]
