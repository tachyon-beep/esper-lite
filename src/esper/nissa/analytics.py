"""Nissa Analytics - Blueprint performance aggregation.

Aggregates telemetry events into strategic dashboards:
- BlueprintStats: Per-blueprint fossilization rates and accuracy metrics
- SeedScoreboard: Per-environment cumulative seed tracking
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Callable


# =============================================================================
# Compute Cost Multipliers
# =============================================================================

BLUEPRINT_COMPUTE_MULTIPLIERS: dict[str, float] = {
    "depthwise": 1.08,      # Cheap - depthwise separable
    "conv_enhance": 1.15,   # Moderate - adds conv layers
    "norm": 1.02,           # Minimal - just normalization
    "attention": 1.35,      # Expensive - O(n²) attention
}


def compute_cost_for_blueprint(blueprint_id: str) -> float:
    """Return compute multiplier for a blueprint type."""
    return BLUEPRINT_COMPUTE_MULTIPLIERS.get(blueprint_id, 1.1)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BlueprintStats:
    """Performance statistics for a single blueprint type."""

    germinated: int = 0
    fossilized: int = 0
    culled: int = 0
    acc_deltas: list[float] = field(default_factory=list)
    churns: list[float] = field(default_factory=list)

    @property
    def mean_acc_delta(self) -> float:
        """Mean accuracy improvement at terminal state."""
        return sum(self.acc_deltas) / len(self.acc_deltas) if self.acc_deltas else 0.0

    @property
    def mean_churn(self) -> float:
        """Mean accuracy change on cull (usually negative)."""
        return sum(self.churns) / len(self.churns) if self.churns else 0.0

    @property
    def fossilization_rate(self) -> float:
        """Percentage of terminal seeds that fossilized (not culled)."""
        total = self.fossilized + self.culled
        return (self.fossilized / total * 100) if total > 0 else 0.0


@dataclass
class SeedScoreboard:
    """Cumulative seed tracking for an environment."""

    total_germinated: int = 0
    total_fossilized: int = 0
    total_culled: int = 0
    fossilized_by_blueprint: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    live_blueprint: str | None = None
    params_added: int = 0
    host_params: int = 0

    @property
    def compute_cost(self) -> float:
        """Estimated compute cost relative to baseline (1.0)."""
        cost = 1.0
        for bp_id, count in self.fossilized_by_blueprint.items():
            cost += (compute_cost_for_blueprint(bp_id) - 1.0) * count
        return cost

    @property
    def params_percentage(self) -> float:
        """Params added as percentage of host."""
        return (self.params_added / self.host_params * 100) if self.host_params > 0 else 0.0


__all__ = [
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
    "BlueprintStats",
    "SeedScoreboard",
]
