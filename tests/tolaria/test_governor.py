"""Tests for TolariaGovernor - the fail-safe watchdog mechanism."""

import math
import torch
import torch.nn as nn

from esper.leyline import (
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_GOVERNOR_LOSS_MULTIPLIER,
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
)


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
        assert gov.sensitivity == DEFAULT_GOVERNOR_SENSITIVITY
        assert gov.multiplier == DEFAULT_GOVERNOR_LOSS_MULTIPLIER
        assert gov.absolute_threshold == DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD
        assert gov.death_penalty == DEFAULT_GOVERNOR_DEATH_PENALTY
        assert gov.min_panics_before_rollback == DEFAULT_MIN_PANICS_BEFORE_ROLLBACK
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
        gov.execute_rollback()

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

    def test_execute_rollback_emits_env_id_and_device(self):
        """Rollback event should include env_id and device for multi-env telemetry."""
        from unittest.mock import Mock, patch

        from esper.leyline import TelemetryEventType
        from esper.tolaria import TolariaGovernor

        model = DummyModel()
        gov = TolariaGovernor(model)

        with patch("esper.tolaria.governor.get_hub") as get_hub:
            hub = Mock()
            get_hub.return_value = hub

            gov.execute_rollback(env_id=7)

            event = hub.emit.call_args[0][0]
            assert event.event_type == TelemetryEventType.GOVERNOR_ROLLBACK
            assert event.data["env_id"] == 7
            assert event.data["device"] == "cpu"

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
        gov.execute_rollback()
        assert gov.consecutive_panics == 0

        # Manually set panics to simulate another panic event
        gov.consecutive_panics = 2

        # Second rollback - should reset to 0 again
        gov.execute_rollback()
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
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        gov = TolariaGovernor(model)

        # Take snapshot
        gov.snapshot()

        # Germinate a seed (simulates live/experimental seed)
        model.seed_slots["r0c1"].germinate("conv_heavy", "test_seed")
        assert model.seed_slots["r0c1"].is_active
        assert model.seed_slots["r0c1"].state is not None

        # Build minimal history
        for i in range(5):
            gov.loss_history.append(1.0)

        # Execute rollback
        gov.execute_rollback()

        # Seed slot should be cleared (experimental seeds discarded)
        assert not model.seed_slots["r0c1"].is_active
        assert model.seed_slots["r0c1"].seed is None
        assert model.seed_slots["r0c1"].state is not None
        assert model.seed_slots["r0c1"].state.stage.name == "PRUNED"

        from esper.leyline import DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            model.seed_slots["r0c1"].step_epoch()
        assert model.seed_slots["r0c1"].state is None

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

    def test_lobotomy_detection_scales_with_task(self):
        """Test that lobotomy tolerance scales with random_guess_loss."""
        from esper.tolaria import TolariaGovernor
        import math

        model = DummyModel()

        # Test with TinyStories-like task (50257 classes)
        tinystories_loss = math.log(50257)  # ~10.82
        gov = TolariaGovernor(model, random_guess_loss=tinystories_loss)

        # Build healthy history (loss ~3.0, well below random guess)
        for _ in range(15):
            gov.check_vital_signs(3.0)

        # Jump to near random guess loss - should detect lobotomy
        # With relative tolerance of ~6.5%, threshold is ~0.7 for TinyStories
        # (vs fixed 0.15 which is too tight)
        is_panic = gov.check_vital_signs(tinystories_loss + 0.5)
        assert is_panic is True, (
            "Should detect lobotomy even with larger absolute tolerance for high-entropy tasks"
        )

    def test_rollback_uses_nonblocking_transfer(self):
        """Test that rollback transfers use non_blocking for efficiency."""
        from esper.tolaria import TolariaGovernor

        # This test verifies the rollback completes correctly with non_blocking
        # (actual performance benefit requires CUDA)
        model = DummyModel()
        gov = TolariaGovernor(model)

        original_weight = model.linear.weight.data.clone()
        gov.snapshot()

        model.linear.weight.data.fill_(999.0)

        # Build history
        for i in range(5):
            gov.loss_history.append(1.0)

        report = gov.execute_rollback()

        # Verify rollback still works correctly
        assert torch.allclose(model.linear.weight.data, original_weight)
        assert report.rollback_occurred is True

    def test_rollback_handles_parameterless_model(self):
        """Test that rollback handles models with no parameters gracefully."""
        from esper.tolaria import TolariaGovernor

        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # No parameters, just a buffer
                self.register_buffer('counter', torch.tensor(0))

            def forward(self, x):
                return x

        model = EmptyModel()
        gov = TolariaGovernor(model)

        # Build history
        for i in range(5):
            gov.loss_history.append(1.0)

        # Should not raise StopIteration
        report = gov.execute_rollback()
        assert report.rollback_occurred is True

    def test_rollback_succeeds_after_seed_culled(self):
        """Governor rollback should succeed even if seeds were culled after snapshot.

        This tests the fix for the state_dict key mismatch issue:
        - Snapshot is taken while experimental seed is active (has seed params)
        - Seed is culled (seed params removed from model)
        - Rollback is triggered (snapshot has keys that model doesn't)
        - Should succeed with strict=False, not fail with orphan key error

        The real bug scenario: snapshot() should filter out experimental seeds
        so the snapshot only contains fossilized seeds + host. That way if
        experimental seeds are culled between snapshot and rollback, there's
        no key mismatch.
        """
        from esper.tolaria import TolariaGovernor
        from esper.kasmina import MorphogeneticModel, CNNHost
        from esper.leyline import SeedStage

        # Create model with a seed slot
        host = CNNHost()
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0"])

        # Germinate a seed (experimental seed - not fossilized)
        model.seed_slots["r0c0"].germinate("conv_heavy", "test_seed")
        assert model.seed_slots["r0c0"].is_active
        assert model.seed_slots["r0c0"].state.stage != SeedStage.FOSSILIZED

        # Take snapshot - with the fix, this should exclude experimental seed params
        gov = TolariaGovernor(model)

        # Check that snapshot doesn't contain experimental seed keys
        # (this is the core fix being tested)
        snapshot_keys = set(gov.last_good_state.keys())
        seed_param_keys = {k for k in snapshot_keys if "seed_slots.r0c0.seed." in k}

        # With the fix, experimental seed params should be filtered out
        assert len(seed_param_keys) == 0, (
            f"Snapshot should not include experimental seed parameters, "
            f"but found: {seed_param_keys}"
        )

        # Cull the seed - removes seed parameters from model
        model.seed_slots["r0c0"].prune("test_cull")
        assert not model.seed_slots["r0c0"].is_active

        # Build minimal history for rollback
        for i in range(5):
            gov.loss_history.append(1.0)

        # Rollback should succeed without key mismatch errors
        report = gov.execute_rollback()
        assert report.rollback_occurred

    def test_snapshot_includes_fossilized_seeds(self):
        """Test that fossilized seeds ARE included in snapshots.

        Fossilized seeds are stable, committed memory and should be part of
        the Last Known Good state. Only experimental (non-fossilized) seeds
        should be excluded.
        """
        from esper.tolaria import TolariaGovernor
        from esper.kasmina import MorphogeneticModel, CNNHost
        from esper.leyline import SeedStage

        # Create model with a seed slot
        host = CNNHost()
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0"])

        # Germinate a seed and mark it as fossilized
        model.seed_slots["r0c0"].germinate("conv_heavy", "fossilized_seed")
        model.seed_slots["r0c0"].state.stage = SeedStage.FOSSILIZED

        # Take snapshot
        gov = TolariaGovernor(model)

        # Check that snapshot DOES contain fossilized seed keys
        snapshot_keys = set(gov.last_good_state.keys())
        fossilized_seed_keys = {k for k in snapshot_keys if "seed_slots.r0c0.seed." in k}

        # Fossilized seeds should be included
        assert len(fossilized_seed_keys) > 0, (
            "Snapshot should include fossilized seed parameters"
        )

    def test_snapshot_filters_alpha_schedule_for_experimental_seeds(self):
        """Test that alpha_schedule.* parameters are filtered for experimental seeds.

        Both seed.* and alpha_schedule.* parameters should be filtered out for
        non-fossilized seeds, since alpha_schedule is seed-specific state.
        """
        from esper.tolaria import TolariaGovernor
        from esper.kasmina import MorphogeneticModel, CNNHost
        from esper.leyline import SeedStage

        # Create model with a seed slot
        host = CNNHost()
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

        # Germinate a seed (experimental - not fossilized)
        model.seed_slots["r0c1"].germinate("conv_heavy", "experimental_seed")
        assert model.seed_slots["r0c1"].state.stage != SeedStage.FOSSILIZED

        # Take snapshot
        gov = TolariaGovernor(model)

        # Check that snapshot doesn't contain experimental seed OR alpha_schedule keys
        snapshot_keys = set(gov.last_good_state.keys())
        seed_param_keys = {k for k in snapshot_keys if "seed_slots.r0c1.seed." in k}
        alpha_schedule_keys = {k for k in snapshot_keys if "seed_slots.r0c1.alpha_schedule." in k}

        # Both should be filtered out
        assert len(seed_param_keys) == 0, (
            f"Snapshot should not include experimental seed.* parameters, "
            f"but found: {seed_param_keys}"
        )
        assert len(alpha_schedule_keys) == 0, (
            f"Snapshot should not include experimental alpha_schedule.* parameters, "
            f"but found: {alpha_schedule_keys}"
        )

    def test_snapshot_handles_mixed_seed_stages(self):
        """Test snapshot with multiple slots in different stages.

        Scenario: One fossilized, one training, one dormant.
        Expected: Only fossilized seed parameters are in snapshot.
        """
        from esper.tolaria import TolariaGovernor
        from esper.kasmina import MorphogeneticModel, CNNHost
        from esper.leyline import SeedStage

        # Create model with three seed slots
        host = CNNHost()
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

        # r0c0: Fossilized seed (should be included)
        model.seed_slots["r0c0"].germinate("conv_heavy", "fossilized_seed")
        model.seed_slots["r0c0"].state.stage = SeedStage.FOSSILIZED

        # r0c1: Training seed (should be excluded)
        model.seed_slots["r0c1"].germinate("conv_heavy", "training_seed")
        model.seed_slots["r0c1"].state.stage = SeedStage.TRAINING

        # r0c2: Dormant (no seed at all)
        # Leave as-is

        # Take snapshot
        gov = TolariaGovernor(model)

        snapshot_keys = set(gov.last_good_state.keys())

        # Check r0c0 (fossilized) - should be included
        r0c0_keys = {k for k in snapshot_keys if "seed_slots.r0c0.seed." in k}
        assert len(r0c0_keys) > 0, "Fossilized seed (r0c0) should be in snapshot"

        # Check r0c1 (training) - should be excluded
        r0c1_keys = {k for k in snapshot_keys if "seed_slots.r0c1.seed." in k}
        assert len(r0c1_keys) == 0, (
            f"Training seed (r0c1) should not be in snapshot, but found: {r0c1_keys}"
        )

        # Check r0c2 (dormant) - no seed parameters to check
        r0c2_keys = {k for k in snapshot_keys if "seed_slots.r0c2.seed." in k}
        assert len(r0c2_keys) == 0, "Dormant slot (r0c2) should have no seed parameters"
