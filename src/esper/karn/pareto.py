"""Pareto frontier analysis for multi-objective reward evaluation.

Provides tools for extracting non-dominated outcomes and computing
hypervolume indicators for tracking training progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.store import EpisodeOutcome


def extract_pareto_frontier(outcomes: list["EpisodeOutcome"]) -> list["EpisodeOutcome"]:
    """Extract non-dominated outcomes from a list.

    An outcome is non-dominated (Pareto optimal) if no other outcome
    is strictly better in all objectives.

    Objectives considered:
    - final_accuracy: maximize
    - param_ratio: minimize
    - stability_score: maximize

    Args:
        outcomes: List of episode outcomes to analyze

    Returns:
        List of non-dominated outcomes (the Pareto frontier)
    """
    if not outcomes:
        return []

    frontier = []
    for candidate in outcomes:
        is_dominated = False
        for other in outcomes:
            if other is candidate:
                continue
            if other.dominates(candidate):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate)

    return frontier


def compute_hypervolume_2d(
    frontier: list["EpisodeOutcome"],
    ref_point: tuple[float, float],
) -> float:
    """Compute 2D hypervolume for accuracy vs param_ratio.

    Uses sweep-line algorithm: sort by accuracy descending, sweep from
    high accuracy to low, accumulating rectangular areas.

    Args:
        frontier: List of Pareto-optimal outcomes
        ref_point: (min_accuracy, max_param_ratio) - worst acceptable values

    Returns:
        Hypervolume (area dominated by frontier)
    """
    if not frontier:
        return 0.0

    ref_acc, ref_param = ref_point

    # Extract and sort by accuracy descending
    points = sorted(
        [(o.final_accuracy, o.param_ratio) for o in frontier],
        key=lambda p: -p[0]
    )

    hv = 0.0
    current_param = ref_param  # Start at worst param_ratio

    for acc, param in points:
        if param < current_param:
            # This point extends the dominated region
            hv += acc * (current_param - param)
            current_param = param

    return hv


__all__ = ["extract_pareto_frontier", "compute_hypervolume_2d"]
