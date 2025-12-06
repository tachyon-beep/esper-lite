"""Blueprint curriculum with UCB1 exploration."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BlueprintStats:
    """Statistics for a single blueprint."""
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

    Additionally applies a complexity penalty to favor simpler blueprints
    initially, following curriculum learning principles.
    """

    def __init__(
        self,
        blueprints: list[str],
        complexity: list[int],
        exploration_weight: float = 2.0,
        complexity_penalty: float = 0.1,
    ):
        if len(blueprints) != len(complexity):
            raise ValueError("blueprints and complexity must have same length")

        self.blueprints = blueprints
        self.complexity = dict(zip(blueprints, complexity))
        self.exploration_weight = exploration_weight
        self.complexity_penalty = complexity_penalty

        self._stats: dict[str, BlueprintStats] = {
            name: BlueprintStats() for name in blueprints
        }
        self._total_trials = 0

        # Normalize complexity to [0, 1] range
        max_complexity = max(complexity)
        self._normalized_complexity = {
            name: c / max_complexity for name, c in self.complexity.items()
        }

    def record_outcome(self, blueprint: str, success: bool, reward: float) -> None:
        """Record outcome of a blueprint trial."""
        if blueprint not in self._stats:
            raise ValueError(f"Unknown blueprint: {blueprint}")

        stats = self._stats[blueprint]
        stats.trials += 1
        stats.total_reward += reward
        if success:
            stats.successes += 1
        self._total_trials += 1

    def get_ucb_scores(self) -> dict[str, float]:
        """Compute UCB scores for all blueprints."""
        scores = {}

        for name in self.blueprints:
            stats = self._stats[name]

            if stats.trials == 0:
                # Unexplored: high exploration bonus, complexity penalty
                exploration = self.exploration_weight * 2.0  # Extra bonus for unexplored
                complexity_term = self.complexity_penalty * self._normalized_complexity[name]
                scores[name] = exploration - complexity_term
            else:
                # UCB1 formula
                mean = stats.mean_reward
                exploration = self.exploration_weight * math.sqrt(
                    math.log(self._total_trials + 1) / stats.trials
                )
                complexity_term = self.complexity_penalty * self._normalized_complexity[name]
                scores[name] = mean + exploration - complexity_term

        return scores

    def select_blueprint(self) -> str:
        """Select blueprint with highest UCB score."""
        scores = self.get_ucb_scores()
        return max(scores, key=scores.get)

    def get_stats(self, blueprint: str) -> dict:
        """Get statistics for a blueprint."""
        stats = self._stats[blueprint]
        return {
            "trials": stats.trials,
            "successes": stats.successes,
            "mean_reward": stats.mean_reward,
        }


__all__ = ["BlueprintCurriculum", "BlueprintStats"]
