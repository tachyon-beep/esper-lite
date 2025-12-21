"""Test step_epoch() lifecycle transitions.

Covers all stage transitions in the seed lifecycle state machine:
- GERMINATED → TRAINING
- TRAINING → BLENDING (with dwell time)
- BLENDING → HOLDING (with blending steps)
- HOLDING → FOSSILIZED (positive counterfactual)
- HOLDING → PRUNED (negative counterfactual or timeout)
"""

import torch.nn as nn

from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics, QualityGates, GateResult, GateLevel
from esper.leyline import SeedStage, DEFAULT_MAX_PROBATION_EPOCHS


class MockGates(QualityGates):
    """Gates that can be configured to pass or fail specific transitions."""

    def __init__(self):
        super().__init__()
        self._gate_results: dict[SeedStage, bool] = {}

    def set_gate_result(self, target_stage: SeedStage, passed: bool) -> None:
        """Configure whether gate to target_stage passes or fails."""
        self._gate_results[target_stage] = passed

    def check_gate(self, state: SeedState, target_stage: SeedStage) -> GateResult:
        """Return configured result for target stage."""
        passed = self._gate_results.get(target_stage, True)
        gate_level = {
            SeedStage.GERMINATED: GateLevel.G0,
            SeedStage.TRAINING: GateLevel.G1,
            SeedStage.BLENDING: GateLevel.G2,
            SeedStage.HOLDING: GateLevel.G3,
            SeedStage.FOSSILIZED: GateLevel.G5,
        }.get(target_stage, GateLevel.G0)

        return GateResult(
            gate=gate_level,
            passed=passed,
            checks_passed=["mock_pass"] if passed else [],
            checks_failed=[] if passed else ["mock_fail"],
        )


def create_test_slot(gates: QualityGates | None = None) -> SeedSlot:
    """Create a SeedSlot for testing with minimal setup."""
    return SeedSlot(
        slot_id="test_slot",
        channels=64,
        device="cpu",
        gates=gates or MockGates(),
        fast_mode=True,  # Disable telemetry/isolation for simpler tests
    )


def setup_state_at_stage(slot: SeedSlot, stage: SeedStage) -> None:
    """Manually set up slot state at a specific stage for testing."""
    slot.state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_blueprint",
        slot_id=slot.slot_id,
        stage=stage,
    )
    slot.state.metrics = SeedMetrics()
    # Create a minimal mock seed for is_active checks
    slot.seed = nn.Identity()


class TestStepEpochGerminatedToTraining:
    """Test GERMINATED → TRAINING transition."""

    def test_germinated_advances_to_training_when_gate_passes(self):
        """GERMINATED should immediately advance to TRAINING when G1 passes."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.TRAINING, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.GERMINATED)

        slot.step_epoch()

        assert slot.state.stage == SeedStage.TRAINING
        assert slot.isolate_gradients is True  # Incubator isolation enabled

    def test_germinated_stays_when_gate_fails(self):
        """GERMINATED should not advance if G1 fails."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.TRAINING, False)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.GERMINATED)

        slot.step_epoch()

        assert slot.state.stage == SeedStage.GERMINATED


class TestStepEpochTrainingToBlending:
    """Test TRAINING → BLENDING transition."""

    def test_training_stays_during_dwell(self):
        """TRAINING should not advance before dwell period completes."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.BLENDING, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.TRAINING)
        slot.state.metrics.epochs_in_current_stage = 0  # Below default dwell of 1

        slot.step_epoch()

        assert slot.state.stage == SeedStage.TRAINING

    def test_training_advances_to_blending_after_dwell(self):
        """TRAINING should advance to BLENDING after dwell when G2 passes.

        Note: Without task_config, topology defaults to "cnn", which keeps
        isolate_gradients=True after BLENDING (CNNs don't co-adapt).
        """
        gates = MockGates()
        gates.set_gate_result(SeedStage.BLENDING, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.TRAINING)
        slot.state.metrics.epochs_in_current_stage = 1  # Meets default dwell

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING
        # CNN topology (default): isolation kept to prevent co-adaptation
        assert slot.isolate_gradients is True

    def test_training_advances_to_blending_transformer_disables_isolation(self):
        """Transformer topology should disable isolation at BLENDING for co-adaptation."""
        from esper.tamiyo.policy.features import TaskConfig

        gates = MockGates()
        gates.set_gate_result(SeedStage.BLENDING, True)
        # Create slot with transformer task_config
        task_config = TaskConfig(
            task_type="lm",
            topology="transformer",
            baseline_loss=10.0,
            target_loss=1.0,
            typical_loss_delta_std=0.1,
            max_epochs=25,
        )
        slot = SeedSlot(
            slot_id="test_slot",
            channels=64,
            device="cpu",
            gates=gates,
            fast_mode=True,
            task_config=task_config,
        )
        setup_state_at_stage(slot, SeedStage.TRAINING)
        # Dwell = max(1, int(25 * 0.1)) = 2, so we need epochs >= 2
        slot.state.metrics.epochs_in_current_stage = 2

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING
        # Transformer topology: isolation disabled for co-adaptation
        assert slot.isolate_gradients is False

    def test_training_stays_when_gate_fails(self):
        """TRAINING should not advance if G2 fails."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.BLENDING, False)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.TRAINING)
        slot.state.metrics.epochs_in_current_stage = 10  # Well past dwell

        slot.step_epoch()

        assert slot.state.stage == SeedStage.TRAINING


class TestStepEpochBlendingToHolding:
    """Test BLENDING → HOLDING transition."""

    def test_blending_increments_steps(self):
        """BLENDING should increment alpha controller steps each epoch."""
        slot = create_test_slot()
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.alpha_controller.retarget(alpha_target=1.0, alpha_steps_total=5)

        slot.step_epoch()

        assert slot.state.alpha_controller.alpha_steps_done == 1

    def test_blending_stays_until_steps_complete(self):
        """BLENDING should not advance until all blending steps complete."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.HOLDING, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.alpha_controller.retarget(alpha_target=1.0, alpha_steps_total=5)
        for _ in range(2):
            slot.state.alpha_controller.step()
        slot.set_alpha(slot.state.alpha_controller.alpha)

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING
        assert slot.state.alpha_controller.alpha_steps_done == 3

    def test_blending_advances_to_holding_when_complete(self):
        """BLENDING should advance to HOLDING when steps complete and G3 passes."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.HOLDING, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.alpha_controller.retarget(alpha_target=1.0, alpha_steps_total=5)
        for _ in range(4):
            slot.state.alpha_controller.step()
        slot.set_alpha(slot.state.alpha_controller.alpha)

        slot.step_epoch()

        assert slot.state.stage == SeedStage.HOLDING

    def test_blending_stays_when_gate_fails(self):
        """BLENDING should not advance if G3 fails even when steps complete."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.HOLDING, False)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.BLENDING)
        slot.state.alpha_controller.retarget(alpha_target=1.0, alpha_steps_total=5)
        for _ in range(4):
            slot.state.alpha_controller.step()
        slot.set_alpha(slot.state.alpha_controller.alpha)

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING


class TestStepEpochHoldingOutcomes:
    """Test HOLDING → FOSSILIZED or PRUNED transitions."""

    def test_holding_stays_with_positive_counterfactual(self):
        """HOLDING should STAY (not auto-fossilize) when counterfactual is positive.

        Fossilization requires explicit FOSSILIZE action from Tamiyo, not auto-advance.
        (DRL Expert review 2025-12-10: auto-fossilize violated credit assignment)
        """
        gates = MockGates()
        gates.set_gate_result(SeedStage.FOSSILIZED, True)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = 2.5  # Positive contribution

        slot.step_epoch()

        # Should stay in HOLDING - fossilization requires explicit action
        assert slot.state.stage == SeedStage.HOLDING

    def test_holding_prunes_with_negative_counterfactual(self):
        """HOLDING should prune immediately when counterfactual is negative."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.FOSSILIZED, False)  # G5 fails
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = -1.0  # Negative = hurts performance

        slot.step_epoch()

        # cull() sets state to None after transitioning to PRUNED
        assert slot.state is None
        assert slot.seed is None  # Seed also cleared

    def test_holding_prunes_with_zero_counterfactual(self):
        """HOLDING should prune when counterfactual is exactly zero."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.FOSSILIZED, False)  # G5 must fail to reach <= 0 check
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = 0.0  # No benefit

        slot.step_epoch()

        # cull() sets state to None after transitioning to PRUNED
        assert slot.state is None
        assert slot.seed is None

    def test_holding_prunes_on_timeout(self):
        """HOLDING should prune when timeout is reached without counterfactual."""
        slot = create_test_slot()
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = None  # Not yet evaluated
        slot.state.metrics.epochs_in_current_stage = DEFAULT_MAX_PROBATION_EPOCHS

        slot.step_epoch()

        # cull() sets state to None after transitioning to PRUNED
        assert slot.state is None
        assert slot.seed is None

    def test_holding_waits_for_counterfactual_before_timeout(self):
        """HOLDING should wait for counterfactual evaluation before timeout."""
        slot = create_test_slot()
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = None  # Not yet evaluated
        slot.state.metrics.epochs_in_current_stage = 2  # Below timeout

        slot.step_epoch()

        assert slot.state.stage == SeedStage.HOLDING

    def test_holding_stays_when_g5_fails_with_positive_counterfactual(self):
        """HOLDING should stay if G5 fails even with positive counterfactual."""
        gates = MockGates()
        gates.set_gate_result(SeedStage.FOSSILIZED, False)
        slot = create_test_slot(gates)
        setup_state_at_stage(slot, SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = 2.5  # Positive but G5 fails
        slot.state.metrics.epochs_in_current_stage = 0  # Not timed out

        slot.step_epoch()

        # Should stay in HOLDING (G5 failed, not timed out, positive contribution)
        assert slot.state.stage == SeedStage.HOLDING


class TestStepEpochNoState:
    """Test step_epoch() edge cases."""

    def test_step_epoch_no_op_without_state(self):
        """step_epoch() should be no-op when no seed is active."""
        slot = create_test_slot()
        slot.state = None

        # Should not raise
        slot.step_epoch()

        assert slot.state is None
