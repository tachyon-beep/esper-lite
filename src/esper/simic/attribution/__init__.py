"""Simic Attribution - Counterfactual analysis for seed contribution.

Computes Shapley values and marginal contributions to understand
which seeds are helping vs hurting the host model.
"""

from esper.simic.attribution.counterfactual import (
    CounterfactualEngine,
    CounterfactualConfig,
    CounterfactualMatrix,
    ShapleyEstimate,
)
from esper.simic.attribution.counterfactual_helper import (
    CounterfactualHelper,
    ContributionResult,
    compute_simple_ablation,
)

__all__ = [
    "CounterfactualEngine",
    "CounterfactualConfig",
    "CounterfactualMatrix",
    "ShapleyEstimate",
    "CounterfactualHelper",
    "ContributionResult",
    "compute_simple_ablation",
]
