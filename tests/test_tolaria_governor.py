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
        assert gov.sensitivity == 3.0
        assert gov.death_penalty == 10.0
        assert len(gov.loss_history) == 0
        assert gov.last_good_state is None
        assert gov.consecutive_panics == 0

    def test_custom_parameters(self):
        """Test Governor accepts custom parameters."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(
            model,
            sensitivity=5.0,
            death_penalty=20.0,
            history_window=20,
        )

        assert gov.sensitivity == 5.0
        assert gov.death_penalty == 20.0
        assert gov.loss_history.maxlen == 20

    def test_snapshot_saves_state(self):
        """Test that snapshot() saves model state."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Initially no snapshot
        assert gov.last_good_state is None

        # Take snapshot
        gov.snapshot()
        assert gov.last_good_state is not None
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

    def test_check_vital_signs_no_panic_with_insufficient_history(self):
        """Test that no panic occurs with < 5 samples."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # First 4 samples should not panic (building history)
        assert gov.check_vital_signs(1.0) is False
        assert gov.check_vital_signs(1.1) is False
        assert gov.check_vital_signs(1.2) is False
        assert gov.check_vital_signs(1.3) is False

        assert len(gov.loss_history) == 4

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

    def test_check_vital_signs_panics_on_spike(self):
        """Test that statistical spike triggers panic."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        # Build stable history (mean ~1.0, low std)
        for i in range(10):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Huge spike should trigger panic
        # With mean=1.045 and low std, a loss of 10.0 should panic
        assert gov.check_vital_signs(10.0) is True

    def test_check_vital_signs_requires_both_conditions(self):
        """Test that panic requires BOTH threshold AND 20% above mean."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        # Build history with mean=10.0, std=0
        for i in range(10):
            gov.check_vital_signs(10.0)

        # Loss of 11.5 is < 20% above mean (12.0), so no panic
        # Even though it exceeds mean + 3*std (10.0 + 0)
        result = gov.check_vital_signs(11.5)

        # This should NOT panic because 11.5 < 10.0 * 1.2 = 12.0
        assert result is False

    def test_check_vital_signs_resets_consecutive_panics_on_normal(self):
        """Test that consecutive_panics resets when vitals are good."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # Set up some panic history
        gov.consecutive_panics = 5

        # Build normal history
        for i in range(10):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Should have reset
        assert gov.consecutive_panics == 0

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

    def test_execute_rollback_without_snapshot_raises(self):
        """Test that rollback fails if no snapshot exists."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        # No snapshot taken
        with pytest.raises(RuntimeError, match="panic before first snapshot"):
            gov.execute_rollback()

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
        assert report.consecutive_panics == 1
        assert report.loss_threshold > 0.0
        assert math.isnan(report.loss_at_panic)

    def test_execute_rollback_increments_consecutive_panics(self):
        """Test that multiple rollbacks increment panic counter."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        gov.snapshot()

        # Build minimal history
        for i in range(5):
            gov.loss_history.append(1.0)

        # First rollback
        report1 = gov.execute_rollback()
        assert report1.consecutive_panics == 1

        # Second rollback
        report2 = gov.execute_rollback()
        assert report2.consecutive_panics == 2

        # Third rollback
        report3 = gov.execute_rollback()
        assert report3.consecutive_panics == 3

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

        # Everything should be cleared
        assert len(gov.loss_history) == 0
        assert gov.last_good_state is None
        assert gov.consecutive_panics == 0

    def test_integration_scenario_stable_training(self):
        """Integration test: Stable training with no panics."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        # Simulate 20 stable epochs
        for epoch in range(20):
            # Take snapshot every 5 epochs
            if epoch % 5 == 0:
                gov.snapshot()

            # Normal loss with small variance
            loss = 1.0 + (epoch * 0.01) + (0.05 if epoch % 2 else 0.0)

            # Should never panic
            is_panic = gov.check_vital_signs(loss)
            assert is_panic is False

        assert gov.consecutive_panics == 0

    def test_integration_scenario_panic_and_recovery(self):
        """Integration test: Panic triggers rollback, then recovery."""
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model, sensitivity=3.0)

        # Save initial weights
        initial_weight = model.linear.weight.data.clone()

        # Stable training
        gov.snapshot()
        for i in range(10):
            gov.check_vital_signs(1.0 + (i * 0.01))

        # Corrupt model
        model.linear.weight.data.fill_(999.0)

        # Loss spike causes panic
        is_panic = gov.check_vital_signs(100.0)
        assert is_panic is True

        # Execute rollback
        report = gov.execute_rollback()
        assert report.rollback_occurred is True
        assert report.consecutive_panics == 1

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
        assert report.rollback_occurred is True
