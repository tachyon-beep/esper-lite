"""Hypothesis strategies for SignalTracker property tests.

These strategies generate realistic loss/accuracy sequences and scenarios
for testing the SignalTracker's stabilization detection and signal computation.

Strategy Categories:
1. Sequence generators (loss_sequences, accuracy_sequences)
2. Pattern generators (plateau_sequences, diverging_sequences)
3. Scenario generators (stabilization_scenarios)
"""

from __future__ import annotations

from hypothesis import strategies as st
from hypothesis.strategies import composite

# Import shared strategies
from tests.strategies import bounded_floats


# =============================================================================
# Sequence Strategies
# =============================================================================

@composite
def loss_sequences(
    draw,
    min_length: int = 2,
    max_length: int = 20,
    pattern: str = "decay",
) -> list[float]:
    """Generate realistic loss sequences.

    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        pattern: One of "decay" (typical), "plateau", "diverging", "noisy"

    Returns:
        List of loss values simulating training progression.

    Loss sequences for training typically:
    - Start high (1.0-5.0)
    - Generally decrease (with noise)
    - Never go negative
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    start_loss = draw(bounded_floats(1.0, 5.0))

    losses = [start_loss]

    for _ in range(length - 1):
        if pattern == "decay":
            # Typical training: 80% improvement, 20% regression
            if draw(st.floats(min_value=0.0, max_value=1.0)) < 0.8:
                factor = draw(bounded_floats(0.90, 1.0))  # 0-10% improvement
            else:
                factor = draw(bounded_floats(1.0, 1.2))   # 0-20% regression

        elif pattern == "plateau":
            # Plateau: very small changes either direction
            factor = draw(bounded_floats(0.98, 1.02))  # +-2% change

        elif pattern == "diverging":
            # Diverging: loss increases
            factor = draw(bounded_floats(1.05, 1.3))  # 5-30% increase

        elif pattern == "noisy":
            # Noisy: large random fluctuations
            factor = draw(bounded_floats(0.7, 1.3))

        else:  # default to decay
            factor = draw(bounded_floats(0.90, 1.0))

        new_loss = max(0.001, losses[-1] * factor)  # Never go to 0
        losses.append(new_loss)

    return losses


@composite
def stable_loss_sequences(
    draw,
    min_length: int = 5,
    max_length: int = 15,
    threshold: float = 0.03,
) -> list[float]:
    """Generate loss sequences that should trigger stabilization.

    Each epoch has < threshold relative improvement, ensuring
    consecutive stable epochs.

    Args:
        threshold: Maximum relative improvement per epoch (default 3%)

    Note: The stabilization check uses STRICT less-than (< threshold),
    so we must generate improvements strictly below threshold. We use
    0.8 * threshold as our maximum to provide margin.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    start_loss = draw(bounded_floats(1.0, 3.0))

    losses = [start_loss]

    for _ in range(length - 1):
        # Stay strictly below threshold with margin
        # Max improvement: threshold * 0.8 (e.g., 2.4% for 3% threshold)
        # Min improvement: threshold * 0.3 (e.g., 0.9% for 3% threshold)
        max_factor = 1.0 - (threshold * 0.3)  # e.g., 0.991 for 3% threshold
        min_factor = 1.0 - (threshold * 0.8)  # e.g., 0.976 for 3% threshold

        factor = draw(bounded_floats(min_factor, max_factor))
        new_loss = max(0.001, losses[-1] * factor)
        losses.append(new_loss)

    return losses


@composite
def explosive_loss_sequences(
    draw,
    min_length: int = 5,
    max_length: int = 15,
    threshold: float = 0.03,
) -> list[float]:
    """Generate loss sequences with explosive improvement (> threshold).

    These sequences should NOT trigger stabilization.

    Args:
        threshold: Minimum relative improvement per epoch
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    start_loss = draw(bounded_floats(2.0, 5.0))  # Start higher for room to improve

    losses = [start_loss]

    for _ in range(length - 1):
        # Exceed threshold (double it to be clearly explosive)
        min_factor = 1.0 - (threshold * 3)  # e.g., 0.91 for 3% threshold
        max_factor = 1.0 - threshold         # e.g., 0.97 for 3% threshold

        factor = draw(bounded_floats(min_factor, max_factor))
        new_loss = max(0.001, losses[-1] * factor)
        losses.append(new_loss)

    return losses


@composite
def accuracy_sequences(
    draw,
    min_length: int = 2,
    max_length: int = 20,
    start_accuracy: float | None = None,
) -> list[float]:
    """Generate accuracy sequences (typically increasing, 0-100%).

    Args:
        start_accuracy: Starting accuracy. If None, randomly chosen.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    start = start_accuracy if start_accuracy is not None else draw(bounded_floats(10.0, 50.0))

    accuracies = [start]

    for _ in range(length - 1):
        # 80% chance of improvement
        if draw(st.floats(min_value=0.0, max_value=1.0)) < 0.8:
            delta = draw(bounded_floats(0.0, 2.0))  # 0-2% improvement
        else:
            delta = draw(bounded_floats(-1.0, 0.0))  # 0-1% regression

        new_acc = max(0.0, min(100.0, accuracies[-1] + delta))
        accuracies.append(new_acc)

    return accuracies


@composite
def loss_accuracy_pairs(
    draw,
    min_length: int = 5,
    max_length: int = 20,
) -> list[tuple[float, float]]:
    """Generate paired loss and accuracy sequences.

    Loss and accuracy are inversely correlated (as loss decreases, accuracy increases).
    """
    losses = draw(loss_sequences(min_length=min_length, max_length=max_length))

    pairs = []
    for loss in losses:
        # Accuracy loosely inversely correlated with loss
        # Higher loss -> lower accuracy, with some noise
        base_accuracy = max(0.0, 100.0 - loss * 20)  # Rough inverse
        noise = draw(bounded_floats(-5.0, 5.0))
        accuracy = max(0.0, min(100.0, base_accuracy + noise))
        pairs.append((loss, accuracy))

    return pairs


# =============================================================================
# Pattern-Specific Strategies
# =============================================================================

@composite
def plateau_sequences(
    draw,
    min_length: int = 5,
    max_length: int = 15,
    plateau_start: int | None = None,
) -> tuple[list[float], int]:
    """Generate loss sequences with a plateau region.

    Returns:
        Tuple of (losses, plateau_start_index) where plateau begins.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    _plateau_start = plateau_start if plateau_start is not None else draw(
        st.integers(min_value=2, max_value=length - 3)
    )

    start_loss = draw(bounded_floats(2.0, 5.0))
    losses = [start_loss]

    for i in range(length - 1):
        if i < _plateau_start:
            # Normal decay before plateau
            factor = draw(bounded_floats(0.85, 0.95))
        else:
            # Plateau: minimal change
            factor = draw(bounded_floats(0.995, 1.005))

        new_loss = max(0.001, losses[-1] * factor)
        losses.append(new_loss)

    return losses, _plateau_start


@composite
def diverging_sequences(
    draw,
    min_length: int = 5,
    max_length: int = 15,
    diverge_start: int | None = None,
) -> tuple[list[float], int]:
    """Generate loss sequences that diverge (loss increases).

    Returns:
        Tuple of (losses, diverge_start_index) where divergence begins.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    _diverge_start = diverge_start if diverge_start is not None else draw(
        st.integers(min_value=2, max_value=length - 3)
    )

    start_loss = draw(bounded_floats(1.0, 3.0))
    losses = [start_loss]

    for i in range(length - 1):
        if i < _diverge_start:
            # Normal decay before divergence
            factor = draw(bounded_floats(0.90, 0.98))
        else:
            # Divergence: loss increases
            factor = draw(bounded_floats(1.1, 1.5))

        new_loss = losses[-1] * factor
        losses.append(new_loss)

    return losses, _diverge_start


@composite
def spike_sequences(
    draw,
    min_length: int = 8,
    max_length: int = 20,
) -> tuple[list[float], list[int]]:
    """Generate loss sequences with occasional spikes.

    Returns:
        Tuple of (losses, spike_indices) where spikes occurred.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    num_spikes = draw(st.integers(min_value=1, max_value=3))

    spike_indices = sorted(draw(st.lists(
        st.integers(min_value=2, max_value=length - 2),
        min_size=num_spikes,
        max_size=num_spikes,
        unique=True,
    )))

    start_loss = draw(bounded_floats(2.0, 4.0))
    losses = [start_loss]

    for i in range(length - 1):
        if i in spike_indices:
            # Spike: large sudden increase
            factor = draw(bounded_floats(1.5, 3.0))
        elif i > 0 and (i - 1) in spike_indices:
            # Recovery after spike
            factor = draw(bounded_floats(0.3, 0.6))
        else:
            # Normal decay
            factor = draw(bounded_floats(0.92, 0.98))

        new_loss = max(0.001, losses[-1] * factor)
        losses.append(new_loss)

    return losses, spike_indices


# =============================================================================
# Scenario Strategies (for specific test scenarios)
# =============================================================================

@composite
def stabilization_scenarios(
    draw,
    stabilization_epochs: int = 3,
    threshold: float = 0.03,
) -> dict:
    """Generate complete scenarios for testing stabilization detection.

    Returns:
        Dict with:
        - losses: Loss sequence
        - should_stabilize: Whether stabilization should trigger
        - stabilize_at_epoch: Expected epoch of stabilization (or None)
    """
    # Decide if this scenario should stabilize
    should_stabilize = draw(st.booleans())

    if should_stabilize:
        # Generate stable losses that should trigger stabilization
        # Need baseline + stabilization_epochs stable epochs
        min_length = stabilization_epochs + 2
        losses = draw(stable_loss_sequences(
            min_length=min_length,
            max_length=min_length + 5,
            threshold=threshold,
        ))
        # Stabilization triggers after seeing stabilization_epochs stable epochs
        # First epoch can't count (no previous loss), so it's epoch=stabilization_epochs
        stabilize_at = stabilization_epochs

    else:
        # Generate explosive losses that should NOT stabilize
        losses = draw(explosive_loss_sequences(
            min_length=stabilization_epochs + 2,
            max_length=stabilization_epochs + 10,
            threshold=threshold,
        ))
        stabilize_at = None

    return {
        "losses": losses,
        "should_stabilize": should_stabilize,
        "stabilize_at_epoch": stabilize_at,
        "threshold": threshold,
        "required_epochs": stabilization_epochs,
    }


@composite
def tracker_update_sequences(
    draw,
    min_length: int = 5,
    max_length: int = 30,
) -> list[dict]:
    """Generate sequences of tracker.update() call parameters.

    Returns:
        List of dicts with update() kwargs.
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    pairs = draw(loss_accuracy_pairs(min_length=length, max_length=length))

    updates = []
    for epoch, (loss, accuracy) in enumerate(pairs):
        updates.append({
            "epoch": epoch,
            "global_step": epoch * 100,
            "train_loss": loss * draw(bounded_floats(0.9, 1.1)),  # Train loss similar to val
            "train_accuracy": accuracy * draw(bounded_floats(0.95, 1.05)),
            "val_loss": loss,
            "val_accuracy": accuracy,
            "active_seeds": [],
        })

    return updates


@composite
def edge_case_scenarios(draw) -> dict:
    """Generate edge case scenarios for tracker testing.

    Returns various edge cases:
    - Zero loss
    - Very small loss
    - Large loss
    - Perfect accuracy
    - Zero accuracy
    """
    scenario_type = draw(st.sampled_from([
        "zero_loss",
        "tiny_loss",
        "huge_loss",
        "perfect_accuracy",
        "zero_accuracy",
        "identical_epochs",
    ]))

    if scenario_type == "zero_loss":
        return {
            "type": "zero_loss",
            "val_loss": 0.0001,  # Near-zero
            "val_accuracy": 99.9,
        }
    elif scenario_type == "tiny_loss":
        return {
            "type": "tiny_loss",
            "val_loss": draw(bounded_floats(0.0001, 0.001)),
            "val_accuracy": draw(bounded_floats(95.0, 99.9)),
        }
    elif scenario_type == "huge_loss":
        return {
            "type": "huge_loss",
            "val_loss": draw(bounded_floats(100.0, 10000.0)),
            "val_accuracy": draw(bounded_floats(0.0, 10.0)),
        }
    elif scenario_type == "perfect_accuracy":
        return {
            "type": "perfect_accuracy",
            "val_loss": draw(bounded_floats(0.001, 0.1)),
            "val_accuracy": 100.0,
        }
    elif scenario_type == "zero_accuracy":
        return {
            "type": "zero_accuracy",
            "val_loss": draw(bounded_floats(5.0, 20.0)),
            "val_accuracy": 0.0,
        }
    else:  # identical_epochs
        loss = draw(bounded_floats(0.5, 2.0))
        acc = draw(bounded_floats(40.0, 80.0))
        return {
            "type": "identical_epochs",
            "val_loss": loss,
            "val_accuracy": acc,
            "repeat_count": draw(st.integers(min_value=5, max_value=15)),
        }
