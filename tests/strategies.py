"""Hypothesis strategies for esper-lite property-based testing.

This module provides strategies for generating random but valid instances
of esper-lite types. These strategies are used throughout the property-based
test suite to generate hundreds of test cases automatically.

Usage:
    from tests.strategies import training_signals, seed_telemetries
    from hypothesis import given

    @given(training_signals())
    def test_my_property(signals):
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
from enum import IntEnum
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
        shape: Tensor shape (e.g., (32, 30) for batch_size=32, features=28)
        min_value: Minimum value for elements
        max_value: Maximum value for elements
        dtype: PyTorch dtype

    Returns:
        torch.Tensor of specified shape

    Example:
        @given(pytorch_tensors(shape=(32, 30)))
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
        5=SHADOWING (deprecated/reserved), 6=PROBATIONARY, 7=FOSSILIZED

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
        blueprint_id=draw(st.sampled_from(["conv_heavy", "attention", "norm", "depthwise"])),
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


# =============================================================================
# Simic RL Types
# =============================================================================

@st.composite
def action_members(draw):
    """Generate action enum members with a variable number of germinate actions."""
    germinate_count = draw(st.integers(min_value=1, max_value=6))
    members = {"WAIT": 0}
    for i in range(1, germinate_count + 1):
        members[f"GERMINATE_{i}"] = i
    members["FOSSILIZE"] = germinate_count + 1
    members["CULL"] = germinate_count + 2
    ActionEnum = IntEnum("TestAction", members)
    return draw(st.sampled_from(list(ActionEnum)))


@st.composite
def seed_infos(draw):
    """Generate SeedInfo instances.

    Used for reward computation testing. Generates realistic combinations
    of epochs_in_stage and previous_epochs_in_stage for PBRS testing.
    """
    from esper.simic.rewards import SeedInfo

    improvement = draw(bounded_floats(-10.0, 10.0))
    epochs = draw(st.integers(min_value=0, max_value=100))

    # For transition cases (epochs_in_stage == 0), generate realistic previous counts
    # For non-transition cases, previous_epochs_in_stage is irrelevant (defaults to 0)
    if epochs == 0:
        previous_epochs = draw(st.integers(min_value=0, max_value=100))
        previous_stage = draw(seed_stages())
    else:
        previous_epochs = 0
        previous_stage = 0

    return SeedInfo(
        stage=draw(seed_stages()),
        improvement_since_stage_start=improvement,
        total_improvement=draw(bounded_floats(-10.0, 10.0)),
        epochs_in_stage=epochs,
        previous_stage=previous_stage,
        previous_epochs_in_stage=previous_epochs,
    )


# =============================================================================
# Neural Network Types
# =============================================================================

@st.composite
def simple_network_configs(draw, state_dim: int = 28, action_dim: int = 7):
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
    # Simic
    "action_members",
    "seed_infos",
    # Networks
    "simple_network_configs",
]
