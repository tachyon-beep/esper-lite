"""Hypothesis strategies for Tamiyo property tests.

This module provides reusable strategies for testing:
- HeuristicPolicyConfig configurations
- Mock seed states at various lifecycle stages
- Mock training signals
- Loss/accuracy sequences with realistic patterns
- Decision sequences for stateful testing

Usage:
    from tests.tamiyo.strategies import (
        tamiyo_configs,
        mock_seed_states,
        mock_training_signals,
        loss_sequences,
        accuracy_sequences,
        stabilization_scenarios,
    )

    @given(config=tamiyo_configs(), signals=mock_training_signals())
    def test_something(self, config, signals):
        ...
"""

from tests.tamiyo.strategies.decision_strategies import (
    tamiyo_configs,
    mock_seed_states,
    mock_seed_states_at_stage,
    mock_training_signals,
    decision_sequences,
    germination_contexts,
    fossilization_contexts,
    cull_contexts,
    embargo_contexts,
)
from tests.tamiyo.strategies.tracker_strategies import (
    loss_sequences,
    accuracy_sequences,
    stabilization_scenarios,
    plateau_sequences,
    diverging_sequences,
)

__all__ = [
    # Decision strategies
    "tamiyo_configs",
    "mock_seed_states",
    "mock_seed_states_at_stage",
    "mock_training_signals",
    "decision_sequences",
    "germination_contexts",
    "fossilization_contexts",
    "cull_contexts",
    "embargo_contexts",
    # Tracker strategies
    "loss_sequences",
    "accuracy_sequences",
    "stabilization_scenarios",
    "plateau_sequences",
    "diverging_sequences",
]
