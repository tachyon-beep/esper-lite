"""Tamiyo test configuration and shared fixtures.

This module provides:
1. Hypothesis profiles for different testing contexts
2. Shared fixtures for Tamiyo components
3. Mock objects for external dependencies

Usage:
    # In test files
    def test_something(heuristic_policy, signal_tracker):
        decision = heuristic_policy.decide(signals, seeds)
        ...

Hypothesis Profiles:
    HYPOTHESIS_PROFILE=tamiyo_dev pytest tests/tamiyo/     # Fast dev iteration
    HYPOTHESIS_PROFILE=tamiyo_ci pytest tests/tamiyo/      # CI (more examples)
    HYPOTHESIS_PROFILE=tamiyo_thorough pytest tests/tamiyo/  # Deep exploration
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings, Verbosity, Phase

from esper.leyline import DEFAULT_BLUEPRINT_PENALTY_DECAY

if TYPE_CHECKING:
    from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig
    from esper.tamiyo.tracker import SignalTracker


# =============================================================================
# Hypothesis Profiles
# =============================================================================

# Development profile: fast iteration
settings.register_profile(
    "tamiyo_dev",
    max_examples=20,
    deadline=500,  # 500ms per example
    suppress_health_check=[],
)

# CI profile: thorough but time-bounded
settings.register_profile(
    "tamiyo_ci",
    max_examples=200,
    deadline=None,  # No deadline in CI
    suppress_health_check=[],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Thorough profile: comprehensive exploration
settings.register_profile(
    "tamiyo_thorough",
    max_examples=1000,
    deadline=None,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

# Debug profile: minimal examples, verbose output
settings.register_profile(
    "tamiyo_debug",
    max_examples=5,
    deadline=None,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # Skip shrinking
)

# Load profile based on environment
_profile = os.getenv("HYPOTHESIS_PROFILE", "tamiyo_dev")
if _profile.startswith("tamiyo_"):
    settings.load_profile(_profile)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config() -> "HeuristicPolicyConfig":
    """Default HeuristicPolicyConfig for testing."""
    from esper.tamiyo.heuristic import HeuristicPolicyConfig
    return HeuristicPolicyConfig()


@pytest.fixture
def strict_config() -> "HeuristicPolicyConfig":
    """Strict configuration with low tolerances (quick decisions)."""
    from esper.tamiyo.heuristic import HeuristicPolicyConfig
    return HeuristicPolicyConfig(
        plateau_epochs_to_germinate=1,
        min_epochs_before_germinate=0,
        cull_after_epochs_without_improvement=1,
        cull_if_accuracy_drops_by=0.5,
        min_improvement_to_fossilize=0.0,
        embargo_epochs_after_cull=0,
        blueprint_rotation=["conv_light"],
        blueprint_penalty_on_cull=1.0,
        blueprint_penalty_decay=DEFAULT_BLUEPRINT_PENALTY_DECAY,  # Uses leyline default
        blueprint_penalty_threshold=5.0,
    )


@pytest.fixture
def lenient_config() -> "HeuristicPolicyConfig":
    """Lenient configuration with high tolerances (patient decisions)."""
    from esper.tamiyo.heuristic import HeuristicPolicyConfig
    return HeuristicPolicyConfig(
        plateau_epochs_to_germinate=10,
        min_epochs_before_germinate=20,
        cull_after_epochs_without_improvement=10,
        cull_if_accuracy_drops_by=5.0,
        min_improvement_to_fossilize=1.0,
        embargo_epochs_after_cull=10,
        blueprint_rotation=["conv_light", "conv_heavy", "attention", "norm", "depthwise"],
        blueprint_penalty_on_cull=3.0,
        blueprint_penalty_decay=0.7,
        blueprint_penalty_threshold=10.0,
    )


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def heuristic_policy() -> "HeuristicTamiyo":
    """Fresh HeuristicTamiyo instance for testing."""
    from esper.tamiyo.heuristic import HeuristicTamiyo
    return HeuristicTamiyo(topology="cnn")


@pytest.fixture
def heuristic_policy_with_config(default_config) -> "HeuristicTamiyo":
    """HeuristicTamiyo with explicit default config."""
    from esper.tamiyo.heuristic import HeuristicTamiyo
    return HeuristicTamiyo(config=default_config, topology="cnn")


@pytest.fixture
def signal_tracker() -> "SignalTracker":
    """Fresh SignalTracker instance for testing."""
    from esper.tamiyo.tracker import SignalTracker
    return SignalTracker()


@pytest.fixture
def signal_tracker_strict() -> "SignalTracker":
    """SignalTracker with strict stabilization settings."""
    from esper.tamiyo.tracker import SignalTracker
    return SignalTracker(
        stabilization_threshold=0.01,  # 1% - stricter
        stabilization_epochs=5,         # More epochs required
    )


@pytest.fixture
def signal_tracker_lenient() -> "SignalTracker":
    """SignalTracker with lenient stabilization settings."""
    from esper.tamiyo.tracker import SignalTracker
    return SignalTracker(
        stabilization_threshold=0.10,  # 10% - more lenient
        stabilization_epochs=2,         # Fewer epochs required
    )


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_nissa_hub():
    """Mock Nissa hub that captures emitted events.

    Usage:
        def test_telemetry(mock_nissa_hub):
            # ... trigger some telemetry emission ...
            assert len(mock_nissa_hub) == 1
            assert mock_nissa_hub[0].event_type == TelemetryEventType.TAMIYO_INITIATED
    """
    events = []

    class MockHub:
        def emit(self, event):
            events.append(event)

        def subscribe(self, *args, **kwargs):
            pass  # No-op for tests

    from esper.nissa import output
    original = output._global_hub

    output._global_hub = MockHub()

    yield events

    output._global_hub = original


@pytest.fixture
def mock_seed_factory():
    """Factory for creating mock seed states with specific properties.

    Usage:
        def test_something(mock_seed_factory):
            seed = mock_seed_factory(stage=SeedStage.HOLDING, improvement=5.0)
            ...
    """
    from esper.leyline import SeedStage

    class MockSeedMetrics:
        def __init__(self, improvement, total, counterfactual):
            self.improvement_since_stage_start = improvement
            self.total_improvement = total
            self.counterfactual_contribution = counterfactual

    class MockSeedState:
        def __init__(
            self,
            seed_id="test_seed",
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            alpha=0.0,
            blueprint_id="conv_light",
            improvement=0.0,
            total=0.0,
            counterfactual=None,
        ):
            self.seed_id = seed_id
            self.stage = stage
            self.epochs_in_stage = epochs_in_stage
            self.alpha = alpha
            self.blueprint_id = blueprint_id
            self.metrics = MockSeedMetrics(improvement, total, counterfactual)

    def factory(**kwargs):
        return MockSeedState(**kwargs)

    return factory


@pytest.fixture
def mock_signals_factory():
    """Factory for creating mock training signals with specific properties.

    Usage:
        def test_something(mock_signals_factory):
            signals = mock_signals_factory(epoch=10, plateau_epochs=5, host_stabilized=1)
            ...
    """
    class MockTrainingMetrics:
        def __init__(
            self,
            epoch=0,
            plateau_epochs=0,
            host_stabilized=0,
            accuracy_delta=0.0,
        ):
            self.epoch = epoch
            self.plateau_epochs = plateau_epochs
            self.host_stabilized = host_stabilized
            self.accuracy_delta = accuracy_delta

    class MockTrainingSignals:
        def __init__(self, **kwargs):
            self.metrics = MockTrainingMetrics(**kwargs)

    def factory(**kwargs):
        return MockTrainingSignals(**kwargs)

    return factory


# =============================================================================
# Parametrization Helpers
# =============================================================================

# Stage parameters for parametrized tests
ACTIVE_STAGES = pytest.param(
    ["GERMINATED", "TRAINING", "BLENDING", "HOLDING"],
    id="active_stages",
)

TERMINAL_STAGES = pytest.param(
    ["FOSSILIZED", "PRUNED"],
    id="terminal_stages",
)

# Blueprint rotation options
BLUEPRINT_SETS = [
    pytest.param(["conv_light"], id="single_blueprint"),
    pytest.param(["conv_light", "conv_heavy"], id="two_blueprints"),
    pytest.param(["conv_light", "conv_heavy", "attention"], id="three_blueprints"),
    pytest.param(
        ["conv_light", "conv_heavy", "attention", "norm", "depthwise"],
        id="full_rotation",
    ),
]
