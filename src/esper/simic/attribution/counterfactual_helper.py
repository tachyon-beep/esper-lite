"""Simic Counterfactual Helper - Bridge between CounterfactualEngine and training loop.

Provides a simple interface for computing counterfactual attribution during training.
This is a thin wrapper that handles the integration concerns.

Usage:
    from esper.simic.attribution import CounterfactualHelper

    helper = CounterfactualHelper()
    contributions = helper.compute_contributions(
        slot_ids=["r0c0", "r0c1"],
        evaluate_fn=my_evaluate_fn,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from esper.simic.attribution.counterfactual import (
    CounterfactualEngine,
    CounterfactualConfig,
    CounterfactualMatrix,
)
from esper.nissa import get_hub

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


@dataclass
class ContributionResult:
    """Result of counterfactual contribution computation."""

    slot_id: str
    contribution: float  # Marginal contribution
    shapley_mean: float = 0.0  # Shapley value (if computed)
    shapley_std: float = 0.0  # Shapley uncertainty
    is_significant: bool = True  # Statistically significant?


class CounterfactualHelper:
    """Helper for computing counterfactual contributions during training.

    Wraps CounterfactualEngine with training-loop-friendly interface.

    Telemetry:
        When emit_events=True (default), the underlying CounterfactualEngine
        emits ANALYTICS_SNAPSHOT events with kind="shapley_computed" containing
        Shapley values for each slot. This happens during compute_contributions()
        when Shapley values are calculated.
    """

    def __init__(
        self,
        strategy: str = "auto",
        shapley_samples: int = 20,
        emit_events: bool = True,
    ):
        """Initialize helper.

        Args:
            strategy: "auto", "full_factorial", "shapley", or "ablation_only"
            shapley_samples: Number of permutation samples for Shapley
            emit_events: Whether to emit telemetry events
        """
        config = CounterfactualConfig(
            strategy=strategy,
            shapley_samples=shapley_samples,
        )
        # Convert emit_events bool to callback
        emit_callback = None
        if emit_events:
            hub = get_hub()
            if hub is not None:
                emit_callback = hub.emit
        self.engine = CounterfactualEngine(config, emit_callback=emit_callback)
        self._last_matrix: CounterfactualMatrix | None = None

    def compute_contributions(
        self,
        slot_ids: list[str],
        evaluate_fn: Callable[[dict[str, float]], tuple[float, float]],
        epoch: int | None = None,
    ) -> dict[str, ContributionResult]:
        """Compute counterfactual contributions for all slots.

        Args:
            slot_ids: List of slot IDs to evaluate
            evaluate_fn: Function(alpha_settings) -> (loss, accuracy)
            epoch: Current epoch (for telemetry)

        Returns:
            Dict mapping slot_id to ContributionResult
        """
        if not slot_ids:
            return {}

        # Compute full matrix
        matrix = self.engine.compute_matrix(slot_ids, evaluate_fn)
        self._last_matrix = matrix

        results: dict[str, ContributionResult] = {}

        # Get marginal contributions
        for slot_id in slot_ids:
            contribution = matrix.marginal_contribution(slot_id)
            results[slot_id] = ContributionResult(
                slot_id=slot_id,
                contribution=contribution,
            )

        # Add Shapley values if we have enough configs
        if len(matrix.configs) > len(slot_ids) + 1:
            shapley_values = self.engine.compute_shapley_values(matrix)
            for slot_id, estimate in shapley_values.items():
                if slot_id in results:
                    results[slot_id].shapley_mean = estimate.mean
                    results[slot_id].shapley_std = estimate.std
                    results[slot_id].is_significant = estimate.is_significant()

        # Note: Telemetry is emitted by CounterfactualEngine.compute_shapley_values()
        # when emit_events=True (default). See ANALYTICS_SNAPSHOT events.

        _logger.debug(
            f"Counterfactual computed: {len(matrix.configs)} configs, "
            f"{matrix.compute_time_seconds:.2f}s"
        )

        return results

    def get_interaction_terms(self) -> dict[tuple[str, str], float]:
        """Get interaction terms from last computation.

        Returns:
            Dict mapping (slot_a, slot_b) to interaction strength
        """
        if not self._last_matrix:
            return {}

        interactions = self.engine.compute_interaction_terms(self._last_matrix)
        return {pair: term.interaction for pair, term in interactions.items()}

    @property
    def last_matrix(self) -> CounterfactualMatrix | None:
        """Get the last computed counterfactual matrix."""
        return self._last_matrix


def compute_simple_ablation(
    slot_ids: list[str],
    full_accuracy: float,
    per_slot_accuracy: dict[str, float],
) -> dict[str, float]:
    """Compute simple ablation-based contributions.

    This is a lightweight alternative when full counterfactual
    computation is too expensive. Uses pre-computed per-slot
    accuracies from the training loop.

    Args:
        slot_ids: List of slot IDs
        full_accuracy: Accuracy with all slots enabled
        per_slot_accuracy: Dict of accuracy with each slot disabled

    Returns:
        Dict mapping slot_id to contribution (removal cost)
    """
    contributions = {}
    for slot_id in slot_ids:
        if slot_id in per_slot_accuracy:
            # Removal cost = full - without_slot
            # Positive means slot is helping
            contributions[slot_id] = full_accuracy - per_slot_accuracy[slot_id]
        else:
            contributions[slot_id] = 0.0
    return contributions
