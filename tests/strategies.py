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
# Slot Configuration Types
# =============================================================================

@st.composite
def slot_configs(draw, min_slots: int = 1, max_slots: int = 10):
    """Generate valid SlotConfig instances.

    Args:
        draw: Hypothesis draw function
        min_slots: Minimum number of slots
        max_slots: Maximum number of slots

    Returns:
        SlotConfig with valid slot IDs in row-major order

    Example:
        @given(slot_configs())
        def test_slot_config_property(config):
            assert config.num_slots >= 1
    """
    from esper.leyline.slot_config import SlotConfig

    n_slots = draw(st.integers(min_value=min_slots, max_value=max_slots))
    # Calculate grid dimensions that produce at least n_slots
    # Use a simple heuristic: prefer wider grids
    rows = draw(st.integers(min_value=1, max_value=max(1, n_slots)))
    cols = (n_slots + rows - 1) // rows  # Ceiling division

    return SlotConfig.for_grid(rows, cols)


@st.composite
def injection_specs(draw, slot_id: str | None = None):
    """Generate valid InjectionSpec instances.

    Args:
        draw: Hypothesis draw function
        slot_id: Optional specific slot ID (generates valid one if None)

    Returns:
        InjectionSpec with valid invariants

    Example:
        @given(injection_specs())
        def test_injection_spec_property(spec):
            assert spec.channels > 0
    """
    from esper.leyline.injection_spec import InjectionSpec

    if slot_id is None:
        row = draw(st.integers(min_value=0, max_value=9))
        col = draw(st.integers(min_value=0, max_value=9))
        slot_id = f"r{row}c{col}"

    channels = draw(st.integers(min_value=1, max_value=1024))
    position = draw(bounded_floats(0.0, 1.0))
    start_layer = draw(st.integers(min_value=0, max_value=99))
    end_layer = draw(st.integers(min_value=start_layer, max_value=100))

    return InjectionSpec(
        slot_id=slot_id,
        channels=channels,
        position=position,
        layer_range=(start_layer, end_layer),
    )


@st.composite
def slot_states_for_config(draw, slot_config):
    """Generate slot states dict matching a SlotConfig.

    Args:
        draw: Hypothesis draw function
        slot_config: SlotConfig defining available slots

    Returns:
        Dict mapping slot_id to MaskSeedInfo or None (empty slot)

    Example:
        @given(slot_configs())
        def test_with_states(config):
            states = slot_states_for_config(config).example()
    """
    from esper.simic.control import MaskSeedInfo
    from esper.leyline import SeedStage

    states: dict[str, MaskSeedInfo | None] = {}

    for slot_id in slot_config.slot_ids:
        # Randomly decide if slot is occupied
        if draw(st.booleans()):
            # Generate a seed in this slot
            stage = draw(st.sampled_from([s.value for s in SeedStage if s != SeedStage.DORMANT]))
            age = draw(st.integers(min_value=0, max_value=100))
            states[slot_id] = MaskSeedInfo(stage=stage, seed_age_epochs=age)
        else:
            states[slot_id] = None

    return states


@st.composite
def enabled_slots_for_config(draw, slot_config):
    """Generate a subset of enabled slots from a SlotConfig.

    Args:
        draw: Hypothesis draw function
        slot_config: SlotConfig defining available slots

    Returns:
        List of enabled slot IDs (non-empty subset)

    Example:
        @given(slot_configs())
        def test_with_enabled(config):
            enabled = enabled_slots_for_config(config).example()
            assert len(enabled) >= 1
    """
    # Must enable at least one slot
    all_slots = list(slot_config.slot_ids)
    # Sample 1 to all slots
    n_enabled = draw(st.integers(min_value=1, max_value=len(all_slots)))
    return draw(st.lists(st.sampled_from(all_slots), min_size=n_enabled, max_size=n_enabled, unique=True))


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
# Kasmina Types
# =============================================================================

@st.composite
def seed_stages_enum(draw, exclude_terminal: bool = False, exclude_failure: bool = False):
    """Generate valid SeedStage enum values.

    Args:
        draw: Hypothesis draw function
        exclude_terminal: If True, exclude FOSSILIZED (terminal success state)
        exclude_failure: If True, exclude CULLED, EMBARGOED, RESETTING (failure states)

    Returns:
        SeedStage enum value

    Example:
        @given(seed_stages_enum())
        def test_stage_property(stage):
            assert isinstance(stage, SeedStage)
    """
    from esper.leyline import SeedStage
    from esper.leyline.stages import is_terminal_stage, is_failure_stage

    stages = list(SeedStage)
    # Remove UNKNOWN which is not a valid lifecycle stage
    stages = [s for s in stages if s != SeedStage.UNKNOWN]

    if exclude_terminal:
        stages = [s for s in stages if not is_terminal_stage(s)]
    if exclude_failure:
        stages = [s for s in stages if not is_failure_stage(s)]

    return draw(st.sampled_from(stages))


@st.composite
def alpha_values(draw, include_boundaries: bool = True):
    """Generate valid alpha values in [0, 1].

    Alpha values control the blending between host and seed outputs.
    0.0 = pure host, 1.0 = pure seed.

    Args:
        draw: Hypothesis draw function
        include_boundaries: If True, occasionally generate exactly 0.0 and 1.0

    Returns:
        float in [0, 1]

    Example:
        @given(alpha_values())
        def test_alpha_property(alpha):
            assert 0.0 <= alpha <= 1.0
    """
    if include_boundaries:
        # Occasionally generate exact boundary values for edge case testing
        boundary_choice = draw(st.integers(min_value=0, max_value=99))
        if boundary_choice < 5:  # 5% chance of 0.0
            return 0.0
        elif boundary_choice < 10:  # 5% chance of 1.0
            return 1.0

    return draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def channel_dimensions(draw, min_channels: int = 1, max_channels: int = 512):
    """Generate valid channel counts for neural network layers.

    Args:
        draw: Hypothesis draw function
        min_channels: Minimum channel count (default 1)
        max_channels: Maximum channel count (default 512)

    Returns:
        int representing channel count

    Example:
        @given(channel_dimensions())
        def test_channel_property(channels):
            assert 1 <= channels <= 512
    """
    return draw(st.integers(min_value=min_channels, max_value=max_channels))


@st.composite
def seed_metrics_kasmina(draw):
    """Generate valid SeedMetrics instances for Kasmina.

    Ensures consistency: best_val_accuracy >= current_val_accuracy, etc.

    Returns:
        SeedMetrics instance with valid invariants

    Example:
        @given(seed_metrics_kasmina())
        def test_metrics_property(metrics):
            assert metrics.best_val_accuracy >= metrics.current_val_accuracy
    """
    from esper.kasmina.slot import SeedMetrics

    epochs_total = draw(st.integers(min_value=0, max_value=100))
    epochs_in_stage = draw(st.integers(min_value=0, max_value=max(1, epochs_total)))

    initial_acc = draw(bounded_floats(0.0, 100.0))
    current_acc = draw(bounded_floats(0.0, 100.0))
    best_acc = draw(bounded_floats(max(initial_acc, current_acc), 100.0))

    metrics = SeedMetrics()
    metrics.epochs_total = epochs_total
    metrics.epochs_in_current_stage = epochs_in_stage
    metrics.initial_val_accuracy = initial_acc
    metrics.current_val_accuracy = current_acc
    metrics.best_val_accuracy = best_acc
    metrics.accuracy_at_stage_start = draw(bounded_floats(0.0, current_acc))
    metrics.gradient_norm_avg = draw(bounded_floats(0.0, 100.0))
    metrics.current_alpha = draw(alpha_values())
    metrics.alpha_ramp_step = draw(st.integers(min_value=0, max_value=1000))
    metrics.seed_gradient_norm_ratio = draw(bounded_floats(0.0, 10.0))

    return metrics


@st.composite
def seed_states_kasmina(draw, stage: "SeedStage | None" = None):
    """Generate valid SeedState instances for Kasmina.

    Args:
        draw: Hypothesis draw function
        stage: Optional specific stage (generates random if None)

    Returns:
        SeedState instance with valid invariants

    Example:
        @given(seed_states_kasmina())
        def test_state_property(state):
            assert state.alpha >= 0.0
    """
    from esper.kasmina.slot import SeedState, SeedMetrics
    from esper.leyline import SeedStage

    if stage is None:
        stage = draw(seed_stages_enum())

    seed_id = draw(st.text(alphabet="abcdefghijklmnop", min_size=4, max_size=8))
    blueprint_id = draw(st.sampled_from(["conv_heavy", "attention", "norm", "depthwise", "noop"]))
    slot_id = f"r{draw(st.integers(0, 4))}c{draw(st.integers(0, 4))}"

    # Alpha should be 0 for early stages, ramping up for BLENDING
    if stage in (SeedStage.DORMANT, SeedStage.GERMINATED, SeedStage.TRAINING):
        alpha = 0.0
    elif stage == SeedStage.BLENDING:
        alpha = draw(alpha_values())
    else:  # PROBATIONARY, FOSSILIZED
        alpha = 1.0

    state = SeedState(
        seed_id=seed_id,
        blueprint_id=blueprint_id,
        slot_id=slot_id,
        stage=stage,
        alpha=alpha,
    )
    state.metrics = draw(seed_metrics_kasmina())

    return state


@st.composite
def blending_schedules(draw, algorithm: str | None = None):
    """Generate valid blending schedule parameters.

    Args:
        draw: Hypothesis draw function
        algorithm: Optional specific algorithm ("linear", "sigmoid", "gated")

    Returns:
        Dict with algorithm name and parameters

    Example:
        @given(blending_schedules())
        def test_schedule_property(schedule):
            assert schedule["algorithm"] in ["linear", "sigmoid", "gated"]
    """
    if algorithm is None:
        algorithm = draw(st.sampled_from(["linear", "sigmoid", "gated"]))

    total_steps = draw(st.integers(min_value=10, max_value=1000))

    if algorithm == "linear":
        return {
            "algorithm": "linear",
            "total_steps": total_steps,
        }
    elif algorithm == "sigmoid":
        steepness = draw(bounded_floats(0.1, 10.0))
        midpoint = draw(bounded_floats(0.2, 0.8))
        return {
            "algorithm": "sigmoid",
            "total_steps": total_steps,
            "steepness": steepness,
            "midpoint": midpoint,
        }
    else:  # gated
        hidden_dim = draw(st.sampled_from([16, 32, 64]))
        return {
            "algorithm": "gated",
            "total_steps": total_steps,
            "hidden_dim": hidden_dim,
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
    # Slot Configuration
    "slot_configs",
    "injection_specs",
    "slot_states_for_config",
    "enabled_slots_for_config",
    # Kasmina
    "seed_stages_enum",
    "alpha_values",
    "channel_dimensions",
    "seed_metrics_kasmina",
    "seed_states_kasmina",
    "blending_schedules",
    # Networks
    "simple_network_configs",
]
