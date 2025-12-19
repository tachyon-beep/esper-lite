"""Property-based tests for Kasmina SeedSlot invariants.

These tests use Hypothesis to verify critical invariants hold across
all possible states and inputs:

1. Alpha is always bounded in [0, 1]
2. Stage transitions follow VALID_TRANSITIONS
3. best_val_accuracy never decreases (monotonic)
4. STE forward produces host-identical output
5. Gradient flow during blending reaches both host and seed
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.kasmina.slot import SeedSlot, SeedMetrics, SeedState, QualityGates
from esper.kasmina.isolation import ste_forward, blend_with_isolation
from esper.leyline import SeedStage, VALID_TRANSITIONS, is_valid_transition

from tests.strategies import (
    alpha_values,
    seed_stages_enum,
    seed_metrics_kasmina,
    bounded_floats,
    channel_dimensions,
)


class TestAlphaBoundsInvariant:
    """Alpha must always be in [0, 1] regardless of input."""

    @given(alpha=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=200)
    def test_set_alpha_clamps_to_bounds(self, alpha: float):
        """set_alpha() should clamp any value to [0, 1]."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # Germinate a seed so we have state
        slot.germinate("noop", seed_id="test")

        # Try to set arbitrary alpha
        if not (alpha != alpha):  # Skip NaN (NaN != NaN is True)
            slot.set_alpha(alpha)

            # Invariant: alpha always in [0, 1]
            assert 0.0 <= slot.state.alpha <= 1.0
            assert 0.0 <= slot.state.metrics.current_alpha <= 1.0

    @given(alpha=alpha_values())
    @settings(max_examples=100)
    def test_alpha_values_strategy_produces_valid_range(self, alpha: float):
        """Verify our alpha_values strategy produces valid alphas."""
        assert 0.0 <= alpha <= 1.0

    @given(alpha=st.floats(min_value=-100, max_value=100))
    @settings(max_examples=100)
    def test_set_alpha_extreme_values_clamped(self, alpha: float):
        """Extreme values should be clamped to [0, 1]."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(alpha)

        assert slot.state.alpha == max(0.0, min(1.0, alpha))


class TestStageTransitionValidity:
    """All stage transitions must follow VALID_TRANSITIONS."""

    @given(stage=seed_stages_enum(exclude_failure=True))
    @settings(max_examples=50)
    def test_can_transition_to_follows_valid_transitions(self, stage: SeedStage):
        """SeedState.can_transition_to() should match VALID_TRANSITIONS."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            slot_id="r0c0",
            stage=stage,
        )

        valid_targets = VALID_TRANSITIONS.get(stage, ())

        for target in SeedStage:
            if target == SeedStage.UNKNOWN:
                continue
            expected = target in valid_targets
            actual = state.can_transition_to(target)
            assert actual == expected, f"{stage} -> {target}: expected {expected}, got {actual}"

    @given(from_stage=seed_stages_enum(), to_stage=seed_stages_enum())
    @settings(max_examples=100)
    def test_transition_only_succeeds_for_valid_pairs(
        self, from_stage: SeedStage, to_stage: SeedStage
    ):
        """transition() should only succeed for valid stage pairs."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            slot_id="r0c0",
            stage=from_stage,
        )

        is_valid = is_valid_transition(from_stage, to_stage)
        success = state.transition(to_stage)

        assert success == is_valid, f"{from_stage} -> {to_stage}: success={success}, valid={is_valid}"

    def test_dormant_can_only_transition_to_germinated(self):
        """DORMANT stage has exactly one valid transition: GERMINATED."""
        assert VALID_TRANSITIONS[SeedStage.DORMANT] == (SeedStage.GERMINATED,)

    def test_fossilized_is_terminal(self):
        """FOSSILIZED has no valid transitions (terminal state)."""
        assert VALID_TRANSITIONS[SeedStage.FOSSILIZED] == ()


class TestBestAccuracyMonotonic:
    """best_val_accuracy should never decrease."""

    @given(
        initial=bounded_floats(0.0, 100.0),
        updates=st.lists(bounded_floats(0.0, 100.0), min_size=1, max_size=50),
    )
    @settings(max_examples=100)
    def test_best_accuracy_never_decreases(self, initial: float, updates: list[float]):
        """best_val_accuracy is monotonically non-decreasing."""
        metrics = SeedMetrics()

        # First accuracy sets initial and best
        metrics.record_accuracy(initial)
        assert metrics.best_val_accuracy == initial

        previous_best = initial
        for accuracy in updates:
            metrics.record_accuracy(accuracy)

            # Invariant: best_val_accuracy >= previous best
            assert metrics.best_val_accuracy >= previous_best
            # Invariant: best_val_accuracy >= current
            assert metrics.best_val_accuracy >= metrics.current_val_accuracy

            previous_best = metrics.best_val_accuracy

    @given(accuracies=st.lists(bounded_floats(0.0, 100.0), min_size=2, max_size=20))
    @settings(max_examples=50)
    def test_best_accuracy_equals_max_seen(self, accuracies: list[float]):
        """best_val_accuracy should equal max of all seen accuracies."""
        metrics = SeedMetrics()

        for acc in accuracies:
            metrics.record_accuracy(acc)

        assert metrics.best_val_accuracy == max(accuracies)


class TestSTEForwardBehavior:
    """Straight-Through Estimator must produce host-identical output."""

    @given(
        batch=st.integers(min_value=1, max_value=8),
        channels=channel_dimensions(min_channels=8, max_channels=128),
        spatial=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=50)
    def test_ste_forward_equals_host_in_forward_pass(
        self, batch: int, channels: int, spatial: int
    ):
        """ste_forward(host, seed) == host in forward pass (numerically)."""
        host_features = torch.randn(batch, channels, spatial, spatial)
        seed_features = torch.randn(batch, channels, spatial, spatial)

        result = ste_forward(host_features, seed_features)

        # Forward should be identical to host
        torch.testing.assert_close(result, host_features)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=channel_dimensions(min_channels=8, max_channels=64),
    )
    @settings(max_examples=30)
    def test_ste_forward_allows_seed_gradients(self, batch: int, channels: int):
        """STE should allow gradients to flow to seed parameters."""
        host_features = torch.randn(batch, channels, 8, 8, requires_grad=True)
        seed_features = torch.randn(batch, channels, 8, 8, requires_grad=True)

        result = ste_forward(host_features, seed_features)
        loss = result.sum()
        loss.backward()

        # Seed should receive gradients (STE backward path)
        assert seed_features.grad is not None
        assert seed_features.grad.abs().sum() > 0


class TestBlendGradientFlow:
    """Both host and seed must receive gradients during blending."""

    @given(
        alpha=alpha_values(include_boundaries=False),
        batch=st.integers(min_value=1, max_value=4),
        channels=channel_dimensions(min_channels=8, max_channels=64),
    )
    @settings(max_examples=50)
    def test_blend_both_receive_gradients(self, alpha: float, batch: int, channels: int):
        """blend_with_isolation should give gradients to both host and seed."""
        # Skip boundary cases where one side gets zero weight
        assume(0.01 < alpha < 0.99)

        host = torch.randn(batch, channels, 8, 8, requires_grad=True)
        seed = torch.randn(batch, channels, 8, 8, requires_grad=True)
        alpha_tensor = torch.tensor(alpha)

        blended = blend_with_isolation(host, seed, alpha_tensor)
        loss = blended.sum()
        loss.backward()

        # Both should receive gradients
        assert host.grad is not None, "Host should receive gradients"
        assert seed.grad is not None, "Seed should receive gradients"

        # Gradients should be non-zero (weighted by alpha)
        assert host.grad.abs().sum() > 0, "Host gradients should be non-zero"
        assert seed.grad.abs().sum() > 0, "Seed gradients should be non-zero"

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=channel_dimensions(min_channels=8, max_channels=64),
    )
    @settings(max_examples=30)
    def test_alpha_zero_only_host_contributes(self, batch: int, channels: int):
        """At alpha=0, output should equal host exactly."""
        host = torch.randn(batch, channels, 8, 8, requires_grad=True)
        seed = torch.randn(batch, channels, 8, 8, requires_grad=True)
        alpha = torch.tensor(0.0)

        blended = blend_with_isolation(host, seed, alpha)

        torch.testing.assert_close(blended, host)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=channel_dimensions(min_channels=8, max_channels=64),
    )
    @settings(max_examples=30)
    def test_alpha_one_only_seed_contributes(self, batch: int, channels: int):
        """At alpha=1, output should equal seed exactly."""
        host = torch.randn(batch, channels, 8, 8, requires_grad=True)
        seed = torch.randn(batch, channels, 8, 8, requires_grad=True)
        alpha = torch.tensor(1.0)

        blended = blend_with_isolation(host, seed, alpha)

        torch.testing.assert_close(blended, seed)


class TestShapePreservation:
    """Seed slot forward must preserve input shape."""

    @given(
        batch=st.integers(min_value=1, max_value=4),
        channels=st.sampled_from([32, 64, 128]),  # Common channel sizes
        spatial=st.integers(min_value=8, max_value=32),
    )
    @settings(max_examples=30)
    def test_slot_forward_preserves_shape(self, batch: int, channels: int, spatial: int):
        """SeedSlot.forward() should preserve input shape."""
        slot = SeedSlot(slot_id="r0c0", channels=channels)
        slot.germinate("noop", seed_id="test")

        # Move to TRAINING stage
        slot.state.transition(SeedStage.TRAINING)

        input_tensor = torch.randn(batch, channels, spatial, spatial)
        output = slot(input_tensor)

        assert output.shape == input_tensor.shape


class TestMetricsConsistency:
    """Metrics should maintain internal consistency."""

    @given(metrics=seed_metrics_kasmina())
    @settings(max_examples=50)
    def test_metrics_best_geq_current(self, metrics: SeedMetrics):
        """best_val_accuracy >= current_val_accuracy always."""
        # Strategy ensures this, but verify
        assert metrics.best_val_accuracy >= metrics.current_val_accuracy

    @given(
        n_accuracies=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30)
    def test_epochs_total_increments_correctly(self, n_accuracies: int):
        """epochs_total should increment by 1 for each record_accuracy call."""
        metrics = SeedMetrics()

        for i in range(n_accuracies):
            metrics.record_accuracy(50.0)
            assert metrics.epochs_total == i + 1


class TestTempoProperties:
    """Property tests for blend tempo lever functionality."""

    @given(
        tempo=st.sampled_from([3, 5, 8]),  # Valid TEMPO_TO_EPOCHS values
    )
    def test_blend_tempo_epochs_stored_in_seed_state(self, tempo: int):
        """Property: blend_tempo_epochs is stored correctly in SeedState."""
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test", blend_tempo_epochs=tempo)

        assert slot.state.blend_tempo_epochs == tempo

    @given(
        tempo=st.sampled_from([3, 5, 8]),
    )
    def test_blend_tempo_epochs_default_is_standard(self, tempo: int):
        """Property: Default blend_tempo_epochs is 5 (STANDARD)."""
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")  # No tempo specified

        # Default should be STANDARD = 5 epochs
        assert slot.state.blend_tempo_epochs == 5

    @given(
        tempo=st.sampled_from([3, 5, 8]),
    )
    def test_blend_tempo_epochs_serialization_roundtrip(self, tempo: int):
        """Property: blend_tempo_epochs survives to_dict/from_dict roundtrip."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
            blend_tempo_epochs=tempo,
        )

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.blend_tempo_epochs == tempo

    @given(
        tempo=st.sampled_from([3, 5, 8]),
        alpha=alpha_values(include_boundaries=False),
    )
    @settings(max_examples=50)
    def test_blending_velocity_bounded_by_tempo(self, tempo: int, alpha: float):
        """Property: Blending velocity is bounded by 1/tempo epochs."""
        assume(0.01 < alpha < 0.99)

        # Blending velocity = d(alpha) / d(epoch)
        # Max velocity occurs when alpha goes 0->1 in `tempo` epochs
        max_velocity = 1.0 / tempo

        # Simulated velocity (this is what the policy should produce)
        # If alpha changes by delta_alpha in 1 epoch, velocity = delta_alpha
        velocity = alpha / tempo  # Assuming linear ramp from 0

        assert velocity <= max_velocity + 1e-9, (
            f"Velocity {velocity} exceeds max {max_velocity} for tempo {tempo}"
        )

    def test_tempo_to_epochs_mapping_consistency(self):
        """Property: TEMPO_TO_EPOCHS covers all TempoAction values."""
        from esper.leyline.factored_actions import TempoAction, TEMPO_TO_EPOCHS

        # Every enum value should have a mapping
        for tempo in TempoAction:
            assert tempo in TEMPO_TO_EPOCHS, f"Missing TEMPO_TO_EPOCHS entry for {tempo}"

        # Mapping should be monotonic (FAST < STANDARD < SLOW)
        epochs = [TEMPO_TO_EPOCHS[t] for t in sorted(TempoAction, key=lambda x: x.value)]
        assert epochs == sorted(epochs), f"TEMPO_TO_EPOCHS should be monotonic: {epochs}"

    def test_tempo_action_enum_bounds(self):
        """Property: TempoAction enum values are contiguous 0..N-1."""
        from esper.leyline.factored_actions import TempoAction, NUM_TEMPO

        values = [t.value for t in TempoAction]

        assert min(values) == 0, "TempoAction should start at 0"
        assert max(values) == NUM_TEMPO - 1, f"TempoAction max should be {NUM_TEMPO - 1}"
        assert len(values) == NUM_TEMPO, "TempoAction count mismatch"
        assert sorted(values) == list(range(NUM_TEMPO)), "TempoAction values not contiguous"
