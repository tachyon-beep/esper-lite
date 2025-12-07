"""Tests for TolariaGovernor - the fail-safe watchdog mechanism."""

import math
import pytest
import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestTolariaGovernor:
    """Test suite for TolariaGovernor watchdog functionality."""

    def test_import(self):
        """Test that TolariaGovernor can be imported."""
        from esper.tolaria import TolariaGovernor, GovernorReport
        assert TolariaGovernor is not None
        assert GovernorReport is not None

    def test_initialization(self):
        """Test Governor initializes with correct defaults."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        assert gov.model is model
        assert gov.sensitivity == 6.0  # Conservative default
        assert gov.multiplier == 3.0   # Loss must be 3x average
        assert gov.absolute_threshold == 10.0  # Minimum loss threshold
        assert gov.death_penalty == 10.0
        assert gov.min_panics_before_rollback == 2  # Require consecutive
        assert len(gov.loss_history) == 0
        assert gov.last_good_state is not None
        assert 'linear.weight' in gov.last_good_state
        assert gov.consecutive_panics == 0

    def test_custom_parameters(self):
        """Test Governor accepts custom parameters."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(
            model,
            sensitivity=5.0,
            multiplier=5.0,
            absolute_threshold=50.0,
            death_penalty=20.0,
            history_window=30,
            min_panics_before_rollback=3,
        )

        assert gov.sensitivity == 5.0
        assert gov.multiplier == 5.0
        assert gov.absolute_threshold == 50.0
        assert gov.death_penalty == 20.0
        assert gov.loss_history.maxlen == 30
        assert gov.min_panics_before_rollback == 3

    def test_snapshot_saves_state(self):
        """Test that snapshot() saves model state."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Auto-snapshot is taken on init, and manual snapshot refreshes it
        initial_snapshot = gov.last_good_state

        model.linear.weight.data.add_(1.0)
        gov.snapshot()
        assert gov.last_good_state is not None
        assert gov.last_good_state is not initial_snapshot
        assert isinstance(gov.last_good_state, dict)

        # Snapshot should contain model weights
        assert 'linear.weight' in gov.last_good_state
        assert 'linear.bias' in gov.last_good_state

    def test_snapshot_is_independent_copy(self):
        """Test that snapshot creates independent copy of weights."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Save original weights
        original_weight = model.linear.weight.data.clone()

        # Take snapshot
        gov.snapshot()

        # Modify model weights
        model.linear.weight.data.fill_(999.0)

        # Snapshot should still have original values
        snapshot_weight = gov.last_good_state['linear.weight']
        assert torch.allclose(snapshot_weight, original_weight)
        assert not torch.allclose(snapshot_weight, model.linear.weight.data)

    def test_check_vital_signs_detects_nan(self):
        """Test that NaN loss triggers panic immediately."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # NaN should trigger panic even with no history
        assert gov.check_vital_signs(float('nan')) is True

    def test_check_vital_signs_detects_inf(self):
        """Test that Inf loss triggers panic immediately."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Inf should trigger panic even with no history
        assert gov.check_vital_signs(float('inf')) is True
        assert gov.check_vital_signs(float('-inf')) is True

    def test_check_vital_signs_detects_lobotomy(self):
        """Test that sudden jump to random-guess loss triggers panic."""
        from esper.tolaria import TolariaGovernor
        import math

        model = DummyModel()
        # random_guess_loss = ln(10) ≈ 2.3 for CIFAR-10 (10 classes)
        gov = TolariaGovernor(model, random_guess_loss=math.log(10))

        # Build healthy history (loss ~0.8, well below random guess)
        for _ in range(15):
            gov.check_vital_signs(0.8)

        # Sudden jump to exactly random guessing (ln(10) ≈ 2.302)
        # This is the "lobotomy signature" - model outputs uniform probs
        is_panic = gov.check_vital_signs(math.log(10))
        assert is_panic is True, "Should detect lobotomy (jump to random guess loss)"

    def test_check_vital_signs_no_lobotomy_if_already_bad(self):
        """Test that lobotomy detection only fires if we were doing well."""
        from esper.tolaria import TolariaGovernor
        import math

        model = DummyModel()
        gov = TolariaGovernor(model, random_guess_loss=math.log(10))

        # Build history where we were already doing poorly (loss ~2.0)
        for _ in range(15):
            gov.check_vital_signs(2.0)

        # Jump to random guess loss - but we weren't doing well before
        # So this shouldn't trigger lobotomy detection
        is_panic = gov.check_vital_signs(math.log(10))
        assert is_panic is False, "Should not trigger if we weren't doing well"

    def test_check_vital_signs_no_panic_with_insufficient_history(self):
        """Test that no panic occurs with < 10 samples."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # First 9 samples should not panic (building history)
        for i in range(9):
            assert gov.check_vital_signs(1.0 + i * 0.1) is False

        assert len(gov.loss_history) == 9

    def test_check_vital_signs_no_panic_on_normal_loss(self):
        """Test that normal loss values don't trigger panic."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        # Build stable history
        for i in range(10):
            loss = 1.0 + (i * 0.01)  # Slowly increasing
            assert gov.check_vital_signs(loss) is False

        assert len(gov.loss_history) == 10
        assert gov.consecutive_panics == 0

    def test_check_vital_signs_panics_on_catastrophic_spike(self):
        """Test that truly catastrophic spike triggers panic after consecutive anomalies."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        # Use min_panics=1 to test single-spike behavior
        gov = TolariaGovernor(model, sensitivity=3.0, multiplier=2.0,
                              absolute_threshold=5.0, min_panics_before_rollback=1)

        # Build stable history (mean ~1.0, low std)
        for i in range(15):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Catastrophic spike (>5.0 absolute, >2x mean, >3 sigma) should panic
        assert gov.check_vital_signs(50.0) is True

    def test_check_vital_signs_requires_all_conditions(self):
        """Test that panic requires ALL thresholds: absolute, multiplier, and statistical."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        # Set absolute_threshold=20 so we can test values below it
        gov = TolariaGovernor(model, sensitivity=3.0, multiplier=3.0,
                              absolute_threshold=20.0, min_panics_before_rollback=1)

        # Build history with mean=2.0, std~0
        for i in range(15):
            gov.check_vital_signs(2.0)

        # Loss of 15.0 is 7.5x mean but < absolute_threshold (20.0)
        # Should NOT panic even though it exceeds statistical and multiplier
        result = gov.check_vital_signs(15.0)
        assert result is False

        # Loss of 25.0 exceeds all thresholds but requires consecutive panics
        # (default is 2, we set to 1 for this test)
        result2 = gov.check_vital_signs(25.0)
        assert result2 is True

    def test_check_vital_signs_resets_consecutive_panics_on_normal(self):
        """Test that consecutive_panics resets when vitals are good."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Build initial history so statistical detection is active
        for i in range(12):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Simulate a panic scenario manually (after history built)
        gov.consecutive_panics = 5
        gov._pending_panic = True

        # Add a normal value - should reset consecutive panics
        gov.check_vital_signs(1.15)

        # Should have reset because the value is normal
        assert gov.consecutive_panics == 0
        assert gov._pending_panic is False

    def test_execute_rollback_restores_weights(self):
        """Test that rollback restores model to snapshot state."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Save original weights
        original_weight = model.linear.weight.data.clone()
        gov.snapshot()

        # Corrupt model
        model.linear.weight.data.fill_(999.0)

        # Build some history for report
        for i in range(5):
            gov.loss_history.append(1.0)

        # Execute rollback
        report = gov.execute_rollback()

        # Weights should be restored
        assert torch.allclose(model.linear.weight.data, original_weight)

    def test_execute_rollback_uses_initial_snapshot(self):
        """Rollback should work immediately using the auto snapshot."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Corrupt model right away (before any manual snapshot)
        original_weight = model.linear.weight.data.clone()
        model.linear.weight.data.fill_(123.0)

        report = gov.execute_rollback()

        # Weights restored from initial snapshot and report indicates rollback
        assert torch.allclose(model.linear.weight.data, original_weight)
        assert report.rollback_occurred is True

    def test_execute_rollback_returns_report(self):
        """Test that rollback returns proper GovernorReport."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        gov.snapshot()

        # Build history
        for i in range(5):
            gov.loss_history.append(1.0 + (i * 0.1))

        report = gov.execute_rollback()

        assert report.reason == "Structural Collapse"
        assert report.rollback_occurred is True
        assert report.consecutive_panics == 0  # Reset after rollback
        assert report.loss_threshold > 0.0
        assert math.isnan(report.loss_at_panic)

    def test_execute_rollback_resets_consecutive_panics_each_time(self):
        """Test that each rollback resets panic counter (not increments)."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        gov.snapshot()

        # Build minimal history
        for i in range(5):
            gov.loss_history.append(1.0)

        # First rollback - should reset to 0
        report1 = gov.execute_rollback()
        assert gov.consecutive_panics == 0

        # Manually set panics to simulate another panic event
        gov.consecutive_panics = 2

        # Second rollback - should reset to 0 again
        report2 = gov.execute_rollback()
        assert gov.consecutive_panics == 0

    def test_get_punishment_reward(self):
        """Test that punishment reward matches death penalty."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, death_penalty=15.0)

        assert gov.get_punishment_reward() == -15.0

    def test_reset_clears_all_state(self):
        """Test that reset() clears all governor state."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Build up state
        gov.snapshot()
        for i in range(10):
            gov.loss_history.append(1.0 + i)
        gov.consecutive_panics = 5

        # Reset
        gov.reset()

        # Everything should be cleared and a fresh snapshot captured
        assert len(gov.loss_history) == 0
        assert gov.last_good_state is not None
        assert torch.allclose(gov.last_good_state['linear.weight'], model.linear.weight.data)
        assert gov.consecutive_panics == 0

    def test_integration_scenario_stable_training(self):
        """Integration test: Stable training with no panics."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)  # Use conservative defaults

        # Simulate 30 stable epochs
        for epoch in range(30):
            # Take snapshot every 5 epochs
            if epoch % 5 == 0:
                gov.snapshot()

            # Normal loss with small variance
            loss = 1.0 + (epoch * 0.01) + (0.05 if epoch % 2 else 0.0)

            # Should never panic - normal training noise is expected
            is_panic = gov.check_vital_signs(loss)
            assert is_panic is False

        assert gov.consecutive_panics == 0

    def test_integration_scenario_panic_and_recovery(self):
        """Integration test: NaN triggers immediate rollback."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)  # Conservative defaults

        # Save initial weights
        initial_weight = model.linear.weight.data.clone()

        # Stable training
        gov.snapshot()
        for i in range(15):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Corrupt model
        model.linear.weight.data.fill_(999.0)

        # NaN causes immediate panic (the nuclear option)
        is_panic = gov.check_vital_signs(float('nan'))
        assert is_panic is True

        # Execute rollback
        report = gov.execute_rollback()
        assert report.rollback_occurred is True

        # Model should be restored
        assert torch.allclose(model.linear.weight.data, initial_weight)

        # Can continue training after rollback
        punishment_reward = gov.get_punishment_reward()
        assert punishment_reward < 0.0

    def test_governor_report_dataclass(self):
        """Test GovernorReport structure."""
        from esper.tolaria import GovernorReport

        report = GovernorReport(
            reason="Test Panic",
            loss_at_panic=999.0,
            loss_threshold=10.0,
            consecutive_panics=3,
            rollback_occurred=True,
        )

        assert report.reason == "Test Panic"
        assert report.loss_at_panic == 999.0
        assert report.loss_threshold == 10.0
        assert report.consecutive_panics == 3

    def test_execute_rollback_clears_live_seeds(self):
        """Test that rollback clears live (non-fossilized) seeds from slots.

        This implements Option B semantics:
        - Restore host + fossilized seeds
        - Discard experimental (non-fossilized) seeds
        """
        from esper.tolaria import TolariaGovernor
        from esper.kasmina import MorphogeneticModel, CNNHost

        # Create a real MorphogeneticModel with seed slot
        host = CNNHost()
        model = MorphogeneticModel(host, device="cpu")
        gov = TolariaGovernor(model)

        # Take snapshot
        gov.snapshot()

        # Germinate a seed (simulates live/experimental seed)
        model.seed_slot.germinate("conv_heavy", "test_seed")
        assert model.seed_slot.is_active
        assert model.seed_slot.state is not None

        # Build minimal history
        for i in range(5):
            gov.loss_history.append(1.0)

        # Execute rollback
        report = gov.execute_rollback()

        # Seed slot should be cleared (experimental seeds discarded)
        assert not model.seed_slot.is_active
        assert model.seed_slot.seed is None
        assert model.seed_slot.state is None
        assert report.rollback_occurred is True

    def test_execute_rollback_resets_consecutive_panics(self):
        """Test that rollback resets consecutive_panics to allow fresh start."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Build history for statistical detection
        for i in range(15):
            gov.check_vital_signs(1.0)

        # Simulate panic buildup (2 consecutive anomalies trigger rollback)
        gov.consecutive_panics = 2
        gov._panic_loss = 50.0

        # Execute rollback
        report = gov.execute_rollback()
        assert report.rollback_occurred is True

        # After rollback, consecutive_panics should reset to 0
        # This allows training to recover without escalating panic detection
        assert gov.consecutive_panics == 0, (
            f"consecutive_panics should be 0 after rollback, got {gov.consecutive_panics}"
        )

    def test_snapshot_does_not_track_gradients(self):
        """Test that snapshot() does not create tensors that track gradients."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Ensure model has gradients enabled
        model.linear.weight.requires_grad_(True)

        gov.snapshot()

        # Snapshot tensors should NOT require gradients
        for key, value in gov.last_good_state.items():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad, (
                    f"Snapshot tensor '{key}' should not require gradients"
                )
