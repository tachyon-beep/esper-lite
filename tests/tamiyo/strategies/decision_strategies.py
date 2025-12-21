"""Hypothesis strategies for Tamiyo decision-making property tests.

These strategies generate valid inputs for HeuristicTamiyo.decide() and related
functions, enabling property-based testing across the full input space.

Strategy Categories:
1. Configuration strategies (tamiyo_configs)
2. Input generation (mock_seed_states, mock_training_signals)
3. Scenario-specific (germination_contexts, fossilization_contexts, etc.)
4. Sequences for stateful testing (decision_sequences)
"""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import composite

from esper.leyline import SeedStage
from esper.tamiyo.heuristic import HeuristicPolicyConfig

# Import shared strategies from tests.strategies
from tests.strategies import bounded_floats


# =============================================================================
# Stage Categories for Strategy Generation
# =============================================================================

# All active (non-terminal) stages where decisions can affect seeds
ACTIVE_STAGES = [
    SeedStage.GERMINATED,
    SeedStage.TRAINING,
    SeedStage.BLENDING,
    SeedStage.HOLDING,
]

# Early stages (before counterfactual is available)
PRE_BLENDING_STAGES = [
    SeedStage.GERMINATED,
    SeedStage.TRAINING,
]

# Stages where counterfactual contribution is available
BLENDING_PLUS_STAGES = [
    SeedStage.BLENDING,
    SeedStage.HOLDING,
]


# =============================================================================
# Mock Classes for Testing
# =============================================================================

class MockSeedMetrics:
    """Mock seed metrics for testing."""

    def __init__(
        self,
        improvement: float,
        total: float,
        counterfactual: float | None,
    ):
        self.improvement_since_stage_start = improvement
        self.total_improvement = total
        self.counterfactual_contribution = counterfactual


class MockSeedState:
    """Mock seed state for testing.

    Matches the SeedState interface expected by HeuristicTamiyo.decide().
    """

    def __init__(
        self,
        seed_id: str,
        stage: SeedStage,
        epochs_in_stage: int,
        alpha: float,
        blueprint_id: str,
        improvement: float,
        total: float,
        counterfactual: float | None,
    ):
        self.seed_id = seed_id
        self.stage = stage
        self.epochs_in_stage = epochs_in_stage
        self.alpha = alpha
        self.blueprint_id = blueprint_id
        self.metrics = MockSeedMetrics(improvement, total, counterfactual)


class MockTrainingMetrics:
    """Mock training metrics for testing."""

    def __init__(
        self,
        epoch: int,
        plateau_epochs: int,
        host_stabilized: int,
        accuracy_delta: float,
    ):
        self.epoch = epoch
        self.plateau_epochs = plateau_epochs
        self.host_stabilized = host_stabilized
        self.accuracy_delta = accuracy_delta


class MockTrainingSignals:
    """Mock training signals for testing.

    Matches the TrainingSignals interface expected by HeuristicTamiyo.decide().
    """

    def __init__(self, metrics: MockTrainingMetrics):
        self.metrics = metrics


# =============================================================================
# Configuration Strategies
# =============================================================================

@composite
def tamiyo_configs(draw) -> HeuristicPolicyConfig:
    """Generate valid HeuristicPolicyConfig instances.

    Returns configurations with sensible random values covering:
    - Germination triggers (plateau epochs, min epochs)
    - Culling thresholds (epochs without improvement, accuracy drop)
    - Fossilization threshold
    - Embargo duration
    - Blueprint rotation and penalty settings
    """
    return HeuristicPolicyConfig(
        plateau_epochs_to_germinate=draw(st.integers(min_value=1, max_value=10)),
        min_epochs_before_germinate=draw(st.integers(min_value=1, max_value=20)),
        prune_after_epochs_without_improvement=draw(st.integers(min_value=1, max_value=10)),
        prune_if_accuracy_drops_by=draw(bounded_floats(0.5, 5.0)),
        min_improvement_to_fossilize=draw(bounded_floats(-1.0, 2.0)),
        embargo_epochs_after_prune=draw(st.integers(min_value=0, max_value=10)),
        blueprint_rotation=["conv_light", "conv_heavy", "attention"],
        blueprint_penalty_on_prune=draw(bounded_floats(0.5, 5.0)),
        blueprint_penalty_decay=draw(bounded_floats(0.1, 0.9)),
        blueprint_penalty_threshold=draw(bounded_floats(1.0, 10.0)),
    )


@composite
def strict_configs(draw) -> HeuristicPolicyConfig:
    """Generate strict configurations (low tolerances, quick decisions).

    Useful for testing edge cases where policy should act quickly.
    """
    return HeuristicPolicyConfig(
        plateau_epochs_to_germinate=1,
        min_epochs_before_germinate=draw(st.integers(min_value=0, max_value=3)),
        prune_after_epochs_without_improvement=1,
        prune_if_accuracy_drops_by=0.5,
        min_improvement_to_fossilize=0.0,
        embargo_epochs_after_prune=draw(st.integers(min_value=0, max_value=2)),
        blueprint_rotation=["conv_light"],
        blueprint_penalty_on_prune=1.0,
        blueprint_penalty_decay=0.5,
        blueprint_penalty_threshold=5.0,
    )


@composite
def lenient_configs(draw) -> HeuristicPolicyConfig:
    """Generate lenient configurations (high tolerances, slow decisions).

    Useful for testing scenarios where policy waits longer before acting.
    """
    return HeuristicPolicyConfig(
        plateau_epochs_to_germinate=draw(st.integers(min_value=5, max_value=10)),
        min_epochs_before_germinate=draw(st.integers(min_value=10, max_value=30)),
        prune_after_epochs_without_improvement=draw(st.integers(min_value=5, max_value=15)),
        prune_if_accuracy_drops_by=draw(bounded_floats(3.0, 10.0)),
        min_improvement_to_fossilize=draw(bounded_floats(0.5, 3.0)),
        embargo_epochs_after_prune=draw(st.integers(min_value=5, max_value=15)),
        blueprint_rotation=["conv_light", "conv_heavy", "attention", "norm", "depthwise"],
        blueprint_penalty_on_prune=draw(bounded_floats(1.0, 5.0)),
        blueprint_penalty_decay=draw(bounded_floats(0.3, 0.7)),
        blueprint_penalty_threshold=draw(bounded_floats(5.0, 15.0)),
    )


# =============================================================================
# Seed State Strategies
# =============================================================================

@composite
def mock_seed_states(draw, stage: SeedStage | None = None) -> MockSeedState:
    """Generate mock seed states for decision testing.

    Args:
        stage: If provided, fixes the stage to this value.
               Otherwise draws from all active stages.
    """
    if stage is None:
        stage = draw(st.sampled_from(ACTIVE_STAGES))

    # Counterfactual only available in BLENDING+ stages
    counterfactual = None
    if stage in BLENDING_PLUS_STAGES:
        counterfactual = draw(st.one_of(
            st.none(),
            bounded_floats(-5.0, 10.0)
        ))

    # Generate epochs in stage (sensible for the stage)
    if stage == SeedStage.GERMINATED:
        epochs = draw(st.integers(min_value=0, max_value=3))
    elif stage == SeedStage.TRAINING:
        epochs = draw(st.integers(min_value=0, max_value=15))
    elif stage == SeedStage.BLENDING:
        epochs = draw(st.integers(min_value=0, max_value=20))
    else:  # HOLDING
        epochs = draw(st.integers(min_value=0, max_value=10))

    # Alpha based on stage (increases through lifecycle)
    if stage == SeedStage.GERMINATED:
        alpha = 0.0
    elif stage == SeedStage.TRAINING:
        alpha = 0.0
    elif stage == SeedStage.BLENDING:
        alpha = draw(bounded_floats(0.0, 1.0))
    else:  # HOLDING
        alpha = 1.0

    return MockSeedState(
        seed_id=draw(st.text(
            min_size=1,
            max_size=16,
            alphabet=st.characters(whitelist_categories=('L', 'N'))
        )),
        stage=stage,
        epochs_in_stage=epochs,
        alpha=alpha,
        blueprint_id=draw(st.sampled_from(["conv_light", "conv_heavy", "attention"])),
        improvement=draw(bounded_floats(-10.0, 10.0)),
        total=draw(bounded_floats(-10.0, 10.0)),
        counterfactual=counterfactual,
    )


@composite
def mock_seed_states_at_stage(draw, stage: SeedStage) -> MockSeedState:
    """Generate mock seed states fixed at a specific stage."""
    return draw(mock_seed_states(stage=stage))


@composite
def failing_seed_states(draw) -> MockSeedState:
    """Generate seeds that are failing (negative improvement)."""
    stage = draw(st.sampled_from([SeedStage.TRAINING, SeedStage.BLENDING]))

    counterfactual = None
    if stage == SeedStage.BLENDING:
        counterfactual = draw(bounded_floats(-5.0, -0.5))

    return MockSeedState(
        seed_id=draw(st.text(min_size=1, max_size=8, alphabet="abcdefgh")),
        stage=stage,
        epochs_in_stage=draw(st.integers(min_value=5, max_value=15)),  # Long enough to trigger cull
        alpha=0.5 if stage == SeedStage.BLENDING else 0.0,
        blueprint_id=draw(st.sampled_from(["conv_light", "conv_heavy", "attention"])),
        improvement=draw(bounded_floats(-10.0, -3.0)),  # Clearly failing
        total=draw(bounded_floats(-10.0, -2.0)),
        counterfactual=counterfactual,
    )


@composite
def succeeding_seed_states(draw) -> MockSeedState:
    """Generate seeds that are succeeding (positive improvement)."""
    stage = draw(st.sampled_from([SeedStage.TRAINING, SeedStage.BLENDING, SeedStage.HOLDING]))

    counterfactual = None
    if stage in BLENDING_PLUS_STAGES:
        counterfactual = draw(bounded_floats(0.5, 10.0))

    return MockSeedState(
        seed_id=draw(st.text(min_size=1, max_size=8, alphabet="abcdefgh")),
        stage=stage,
        epochs_in_stage=draw(st.integers(min_value=1, max_value=10)),
        alpha=1.0 if stage == SeedStage.HOLDING else (
            draw(bounded_floats(0.3, 1.0)) if stage == SeedStage.BLENDING else 0.0
        ),
        blueprint_id=draw(st.sampled_from(["conv_light", "conv_heavy", "attention"])),
        improvement=draw(bounded_floats(0.5, 10.0)),  # Clearly succeeding
        total=draw(bounded_floats(1.0, 10.0)),
        counterfactual=counterfactual,
    )


@composite
def probationary_seed_states(draw, with_counterfactual: bool = True) -> MockSeedState:
    """Generate HOLDING seeds for fossilization testing.

    Args:
        with_counterfactual: Whether to include counterfactual contribution.
    """
    counterfactual = None
    if with_counterfactual:
        counterfactual = draw(bounded_floats(-5.0, 10.0))

    return MockSeedState(
        seed_id=draw(st.text(min_size=1, max_size=8, alphabet="abcdefgh")),
        stage=SeedStage.HOLDING,
        epochs_in_stage=draw(st.integers(min_value=1, max_value=10)),
        alpha=1.0,
        blueprint_id=draw(st.sampled_from(["conv_light", "conv_heavy", "attention"])),
        improvement=draw(bounded_floats(-5.0, 10.0)),
        total=draw(bounded_floats(-5.0, 10.0)),
        counterfactual=counterfactual,
    )


# =============================================================================
# Training Signals Strategies
# =============================================================================

@composite
def mock_training_signals(
    draw,
    epoch: int | None = None,
    stabilized: bool | None = None,
) -> MockTrainingSignals:
    """Generate mock training signals for decision testing.

    Args:
        epoch: If provided, fixes the epoch to this value.
        stabilized: If provided, fixes the stabilization status.
    """
    _epoch = epoch if epoch is not None else draw(st.integers(min_value=0, max_value=1000))
    _stabilized = stabilized if stabilized is not None else draw(st.booleans())

    return MockTrainingSignals(MockTrainingMetrics(
        epoch=_epoch,
        plateau_epochs=draw(st.integers(min_value=0, max_value=20)),
        host_stabilized=1 if _stabilized else 0,
        accuracy_delta=draw(bounded_floats(-10.0, 10.0)),
    ))


@composite
def stabilized_signals(draw) -> MockTrainingSignals:
    """Generate signals where host is stabilized (germination allowed)."""
    return draw(mock_training_signals(stabilized=True))


@composite
def unstabilized_signals(draw) -> MockTrainingSignals:
    """Generate signals where host is not stabilized (germination blocked)."""
    return draw(mock_training_signals(stabilized=False))


@composite
def plateau_signals(draw, min_plateau: int = 3) -> MockTrainingSignals:
    """Generate signals indicating a training plateau."""
    return MockTrainingSignals(MockTrainingMetrics(
        epoch=draw(st.integers(min_value=10, max_value=100)),
        plateau_epochs=draw(st.integers(min_value=min_plateau, max_value=20)),
        host_stabilized=1,
        accuracy_delta=draw(bounded_floats(-0.5, 0.5)),  # Low improvement
    ))


@composite
def early_training_signals(draw, max_epoch: int = 5) -> MockTrainingSignals:
    """Generate signals from early training (before min_epochs_before_germinate)."""
    return MockTrainingSignals(MockTrainingMetrics(
        epoch=draw(st.integers(min_value=0, max_value=max_epoch)),
        plateau_epochs=draw(st.integers(min_value=0, max_value=5)),
        host_stabilized=draw(st.integers(min_value=0, max_value=1)),
        accuracy_delta=draw(bounded_floats(-5.0, 10.0)),
    ))


# =============================================================================
# Context Strategies (for specific decision scenarios)
# =============================================================================

@composite
def germination_contexts(draw) -> tuple[MockTrainingSignals, list]:
    """Generate contexts where germination might trigger.

    Returns:
        Tuple of (signals, active_seeds) where germination conditions may be met.
    """
    signals = MockTrainingSignals(MockTrainingMetrics(
        epoch=draw(st.integers(min_value=10, max_value=100)),
        plateau_epochs=draw(st.integers(min_value=3, max_value=15)),
        host_stabilized=1,
        accuracy_delta=draw(bounded_floats(-0.5, 0.5)),
    ))
    return signals, []  # No active seeds


@composite
def fossilization_contexts(draw) -> tuple[MockTrainingSignals, list[MockSeedState]]:
    """Generate contexts where fossilization might trigger.

    Returns:
        Tuple of (signals, active_seeds) with HOLDING seed that has positive contribution.
    """
    signals = draw(stabilized_signals())
    seed = draw(probationary_seed_states(with_counterfactual=True))
    # Ensure positive contribution for fossilization
    seed.metrics.counterfactual_contribution = draw(bounded_floats(0.5, 10.0))
    return signals, [seed]


@composite
def cull_contexts(draw) -> tuple[MockTrainingSignals, list[MockSeedState]]:
    """Generate contexts where culling might trigger.

    Returns:
        Tuple of (signals, active_seeds) with failing seed.
    """
    signals = draw(stabilized_signals())
    seed = draw(failing_seed_states())
    return signals, [seed]


@composite
def embargo_contexts(draw, embargo_epochs: int = 5) -> tuple[MockTrainingSignals, int]:
    """Generate contexts during embargo period.

    Returns:
        Tuple of (signals, last_cull_epoch) where current epoch is within embargo.
    """
    last_cull = draw(st.integers(min_value=0, max_value=50))
    offset = draw(st.integers(min_value=0, max_value=embargo_epochs - 1))
    current_epoch = last_cull + offset

    signals = MockTrainingSignals(MockTrainingMetrics(
        epoch=current_epoch,
        plateau_epochs=draw(st.integers(min_value=5, max_value=15)),  # Would otherwise trigger
        host_stabilized=1,
        accuracy_delta=draw(bounded_floats(-0.5, 0.5)),
    ))
    return signals, last_cull


# =============================================================================
# Sequence Strategies (for stateful testing)
# =============================================================================

@composite
def decision_sequences(draw, min_length: int = 3, max_length: int = 20) -> list[tuple]:
    """Generate sequences of (signals, seeds) pairs for stateful testing.

    Returns:
        List of (MockTrainingSignals, list[MockSeedState]) tuples.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    sequence = []

    for epoch in range(length):
        signals = draw(mock_training_signals(epoch=epoch))
        # 50% chance of having an active seed
        if draw(st.booleans()):
            seeds = [draw(mock_seed_states())]
        else:
            seeds = []
        sequence.append((signals, seeds))

    return sequence
