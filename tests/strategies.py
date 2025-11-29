"""Hypothesis strategies for esper-lite property-based testing.

This module provides strategies for generating random but valid instances
of esper-lite types. These strategies are used throughout the property-based
test suite to generate hundreds of test cases automatically.

Usage:
    from tests.strategies import training_snapshots, seed_telemetries
    from hypothesis import given

    @given(training_snapshots())
    def test_my_property(snapshot):
        # Test code here
        pass

Design principles:
1. **Performance**: Use pytorch_tensors() for torch.Tensor, not lists
2. **Validity**: All generated instances must satisfy type invariants
3. **Coverage**: Strategies should cover edge cases (empty, zero, max values)
4. **Reproducibility**: Use seeds for deterministic generation
"""

from __future__ import annotations

import torch
from datetime import datetime, timezone
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

# =============================================================================
# Low-Level Primitives
# =============================================================================

@st.composite
def pytorch_tensors(
    draw,
    shape: tuple[int, ...],
    min_value: float = -1e6,
    max_value: float = 1e6,
    dtype: torch.dtype = torch.float32,
):
    """Generate PyTorch tensors with specific shapes.

    This is MUCH faster than generating lists and converting.
    Use this for all neural network testing.

    Args:
        draw: Hypothesis draw function
        shape: Tensor shape (e.g., (32, 27) for batch_size=32, features=27)
        min_value: Minimum value for elements
        max_value: Maximum value for elements
        dtype: PyTorch dtype

    Returns:
        torch.Tensor of specified shape

    Example:
        @given(pytorch_tensors(shape=(32, 27)))
        def test_forward_pass(batch):
            output = model(batch)
            assert output.shape[0] == 32
    """
    # Generate numpy array first (Hypothesis is optimized for this)
    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    np_array = draw(
        arrays(
            dtype=np_dtype,
            shape=shape,
            elements=st.floats(
                min_value=min_value,
                max_value=max_value,
                width=32 if dtype == torch.float32 else 64,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    return torch.from_numpy(np_array)


@st.composite
def bounded_floats(draw, min_value: float, max_value: float):
    """Generate floats in [min_value, max_value] without NaN/Inf.

    Use this instead of st.floats() to avoid NaN/Inf edge cases
    that crash RL algorithms.
    """
    return draw(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def normalized_floats(draw):
    """Generate floats roughly in [-3, 3] (z-score range).

    Use for testing normalized features.
    """
    return draw(bounded_floats(-3.0, 3.0))


@st.composite
def probabilities(draw):
    """Generate probabilities in [0, 1]."""
    return draw(bounded_floats(0.0, 1.0))


@st.composite
def accuracies(draw):
    """Generate accuracy percentages in [0, 100]."""
    return draw(bounded_floats(0.0, 100.0))


# =============================================================================
# Leyline Contracts
# =============================================================================

@st.composite
def seed_stages(draw):
    """Generate valid SeedStage enum values (1-7).

    Values:
        1=DORMANT, 2=GERMINATED, 3=TRAINING, 4=BLENDING,
        5=SHADOWING, 6=PROBATIONARY, 7=FOSSILIZED

    Note: SeedStage is an IntEnum, so we return integers directly.
    """
    return draw(st.integers(min_value=1, max_value=7))


@st.composite
def seed_telemetries(draw, seed_id: str | None = None):
    """Generate random but valid SeedTelemetry instances.

    Args:
        draw: Hypothesis draw function
        seed_id: Optional seed ID (generates random if None)

    Returns:
        SeedTelemetry instance

    Example:
        @given(seed_telemetries())
        def test_telemetry_features(telemetry):
            features = telemetry.to_features()
            assert len(features) == 10
    """
    from esper.leyline import SeedTelemetry

    return SeedTelemetry(
        seed_id=seed_id or draw(st.text(min_size=1, max_size=16)),
        blueprint_id=draw(st.sampled_from(["conv_enhance", "attention", "norm", "depthwise"])),
        layer_id=draw(st.text(min_size=0, max_size=32)),
        # Health signals
        gradient_norm=draw(bounded_floats(0.0, 100.0)),
        gradient_health=draw(probabilities()),
        has_vanishing=draw(st.booleans()),
        has_exploding=draw(st.booleans()),
        # Progress signals
        accuracy=draw(accuracies()),
        accuracy_delta=draw(bounded_floats(-10.0, 10.0)),
        epochs_in_stage=draw(st.integers(min_value=0, max_value=100)),
        # Stage context
        stage=draw(seed_stages()),
        alpha=draw(probabilities()),
        # Temporal context
        epoch=draw(st.integers(min_value=0, max_value=1000)),
        max_epochs=draw(st.integers(min_value=1, max_value=1000)),
    )


@st.composite
def training_metrics(draw):
    """Generate TrainingMetrics instances.

    Ensures consistency: val_accuracy >= 0, train_loss >= 0, etc.
    """
    from esper.leyline import TrainingMetrics

    epoch = draw(st.integers(min_value=0, max_value=1000))
    train_loss = draw(bounded_floats(0.0, 10.0))
    val_loss = draw(bounded_floats(0.0, 10.0))
    val_accuracy = draw(accuracies())

    return TrainingMetrics(
        epoch=epoch,
        global_step=draw(st.integers(min_value=0, max_value=1000000)),
        train_loss=train_loss,
        val_loss=val_loss,
        loss_delta=draw(bounded_floats(-5.0, 5.0)),
        train_accuracy=draw(accuracies()),
        val_accuracy=val_accuracy,
        accuracy_delta=draw(bounded_floats(-10.0, 10.0)),
        plateau_epochs=draw(st.integers(min_value=0, max_value=50)),
        best_val_accuracy=draw(bounded_floats(val_accuracy, 100.0)),  # best >= current
        best_val_loss=draw(bounded_floats(0.0, val_loss)),  # best_val_loss <= current (lower is better)
        grad_norm_host=draw(bounded_floats(0.0, 100.0)),
        grad_norm_seed=draw(bounded_floats(0.0, 100.0)),
    )


@st.composite
def training_signals(draw, has_active_seed: bool | None = None):
    """Generate TrainingSignals instances.

    Args:
        has_active_seed: Force has_active_seed value (None = random)

    Returns:
        TrainingSignals instance with valid state
    """
    from esper.leyline import TrainingSignals

    signals = TrainingSignals()

    # Set metrics
    signals.metrics = draw(training_metrics())

    # Set active seeds
    if has_active_seed is None:
        has_active_seed = draw(st.booleans())

    if has_active_seed:
        seed_id = draw(st.text(min_size=1, max_size=16))
        signals.active_seeds.append(seed_id)

    # Set available slots
    signals.available_slots = draw(st.integers(min_value=0, max_value=10))

    return signals


@st.composite
def training_snapshots(draw, has_active_seed: bool | None = None):
    """Generate TrainingSnapshot instances.

    This is the core state representation for RL algorithms.

    Args:
        has_active_seed: Force has_active_seed value (None = random)

    Returns:
        TrainingSnapshot with consistent state

    Example:
        @given(training_snapshots(has_active_seed=True))
        def test_snapshot_with_seed(snapshot):
            assert snapshot.has_active_seed is True

    IMPORTANT: Active seeds cannot be DORMANT (stage 1). They must be
    GERMINATED (2) or higher. This prevents the "Active but Dormant" bug.
    """
    from esper.simic import TrainingSnapshot

    if has_active_seed is None:
        has_active_seed = draw(st.booleans())

    epoch = draw(st.integers(min_value=0, max_value=1000))
    val_accuracy = draw(accuracies())

    snapshot = TrainingSnapshot(
        epoch=epoch,
        global_step=draw(st.integers(min_value=0, max_value=1000000)),
        train_loss=draw(bounded_floats(0.0, 10.0)),
        val_loss=draw(bounded_floats(0.0, 10.0)),
        val_accuracy=val_accuracy,
        best_val_accuracy=draw(bounded_floats(val_accuracy, 100.0)),
        plateau_epochs=draw(st.integers(min_value=0, max_value=50)),
        loss_history_5=draw(
            st.tuples(
                *[bounded_floats(0.0, 10.0) for _ in range(5)]
            )
        ),
        accuracy_history_5=draw(
            st.tuples(
                *[accuracies() for _ in range(5)]
            )
        ),
        has_active_seed=has_active_seed,
        # FIX: Active seeds must be GERMINATED (2+), not DORMANT (1)
        seed_stage=draw(st.integers(min_value=2, max_value=7)) if has_active_seed else 0,
        seed_alpha=draw(probabilities()) if has_active_seed else 0.0,
    )

    return snapshot


# =============================================================================
# Simic RL Types
# =============================================================================

@st.composite
def action_values(draw):
    """Generate valid action values (0-6).

    Actions:
        0=WAIT, 1=GERMINATE_CONV, 2=GERMINATE_ATTENTION,
        3=GERMINATE_NORM, 4=GERMINATE_DEPTHWISE,
        5=ADVANCE, 6=CULL
    """
    return draw(st.integers(min_value=0, max_value=6))


@st.composite
def reward_configs(draw):
    """Generate RewardConfig instances with valid hyperparameters.

    All weights are randomized within reasonable bounds.
    """
    from esper.simic.rewards import RewardConfig

    return RewardConfig(
        acc_delta_weight=draw(bounded_floats(0.0, 2.0)),
        training_bonus=draw(bounded_floats(0.0, 1.0)),
        blending_bonus=draw(bounded_floats(0.0, 1.0)),
        fossilized_bonus=draw(bounded_floats(0.0, 2.0)),
        stage_improvement_weight=draw(bounded_floats(0.0, 0.5)),
    )


@st.composite
def seed_infos(draw):
    """Generate SeedInfo instances.

    Used for reward computation testing.
    """
    from esper.simic.rewards import SeedInfo

    return SeedInfo(
        stage=draw(seed_stages()),
        improvement_since_stage_start=draw(bounded_floats(-10.0, 10.0)),
        epochs_in_stage=draw(st.integers(min_value=0, max_value=100)),
    )


# =============================================================================
# Neural Network Types
# =============================================================================

@st.composite
def simple_network_configs(draw, state_dim: int = 27, action_dim: int = 7):
    """Generate configurations for simple neural networks.

    Returns dict with network hyperparameters.
    """
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": draw(st.sampled_from([64, 128, 256])),
        "num_layers": draw(st.integers(min_value=1, max_value=3)),
        "activation": draw(st.sampled_from(["relu", "tanh"])),
    }


# =============================================================================
# Composite Strategies for Integration Tests
# =============================================================================

@st.composite
def full_episodes(draw, num_steps: int | None = None):
    """Generate complete Episode instances with consistent state transitions.

    Args:
        num_steps: Number of steps (None = random 1-10)

    Returns:
        Episode with DecisionPoints and StepOutcomes
    """
    from esper.simic.episodes import Episode, DecisionPoint, StepOutcome, ActionTaken
    from esper.leyline import Action

    if num_steps is None:
        num_steps = draw(st.integers(min_value=1, max_value=10))

    episode = Episode(episode_id=draw(st.text(min_size=1, max_size=32)))

    for step in range(num_steps):
        # Generate consistent state transition
        snapshot_before = draw(training_snapshots())

        action = draw(st.sampled_from(list(Action)))
        action_taken = ActionTaken(
            action=action,
            confidence=draw(probabilities()),
        )

        reward = draw(bounded_floats(-5.0, 5.0))

        # Generate outcome with actual fields
        outcome = StepOutcome(
            accuracy_after=draw(accuracies()),
            accuracy_change=draw(bounded_floats(-10.0, 10.0)),
            loss_after=draw(bounded_floats(0.0, 10.0)),
            loss_change=draw(bounded_floats(-5.0, 5.0)),
            seed_still_alive=draw(st.booleans()),
            seed_stage_after=draw(st.integers(min_value=0, max_value=7)),
            reward=reward,
        )

        decision_point = DecisionPoint(
            observation=snapshot_before,
            action=action_taken,
            outcome=outcome,
        )

        episode.decisions.append(decision_point)

    return episode


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primitives
    "pytorch_tensors",
    "bounded_floats",
    "normalized_floats",
    "probabilities",
    "accuracies",
    # Leyline
    "seed_stages",
    "seed_telemetries",
    "training_metrics",
    "training_signals",
    "training_snapshots",
    # Simic
    "action_values",
    "reward_configs",
    "seed_infos",
    # Networks
    "simple_network_configs",
    # Composite
    "full_episodes",
]
