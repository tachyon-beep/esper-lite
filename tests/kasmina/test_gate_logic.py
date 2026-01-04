"""Gate logic tests for Kasmina QualityGates.

Tests verify correct behavior of individual quality gates:
- G0: Basic sanity (seed_id, blueprint_id present)
- G1: Training readiness (stage == GERMINATED)
- G2: Blending readiness (improvement, gradient ratio, seed readiness)
- G3: Holding readiness (alpha complete, blending epochs)
- G5: Fossilization readiness (counterfactual, health)
"""


from esper.kasmina.slot import SeedState, QualityGates
from esper.leyline import (
    SeedStage,
    GateLevel,
    DEFAULT_MIN_TRAINING_IMPROVEMENT,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
)
from esper.leyline.alpha import AlphaMode


class TestG0Gate:
    """Tests for G0 gate (germination readiness)."""

    def test_g0_passes_with_seed_and_blueprint(self):
        """G0 should pass when both seed_id and blueprint_id are present."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test_seed",
            blueprint_id="noop",
            stage=SeedStage.DORMANT,
        )

        result = gates.check_gate(state, SeedStage.GERMINATED)

        assert result.passed
        assert result.gate == GateLevel.G0
        assert "seed_id_present" in result.checks_passed
        assert "blueprint_id_present" in result.checks_passed

    def test_g0_fails_without_seed_id(self):
        """G0 should fail when seed_id is missing."""
        gates = QualityGates()
        state = SeedState(
            seed_id="",  # Empty
            blueprint_id="noop",
            stage=SeedStage.DORMANT,
        )

        result = gates.check_gate(state, SeedStage.GERMINATED)

        assert not result.passed
        assert "seed_id_missing" in result.checks_failed

    def test_g0_fails_without_blueprint_id(self):
        """G0 should fail when blueprint_id is missing."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test_seed",
            blueprint_id="",  # Empty
            stage=SeedStage.DORMANT,
        )

        result = gates.check_gate(state, SeedStage.GERMINATED)

        assert not result.passed
        assert "blueprint_id_missing" in result.checks_failed


class TestG1Gate:
    """Tests for G1 gate (training readiness)."""

    def test_g1_passes_when_germinated(self):
        """G1 should pass when stage is GERMINATED."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.GERMINATED,
        )

        result = gates.check_gate(state, SeedStage.TRAINING)

        assert result.passed
        assert result.gate == GateLevel.G1
        assert "germinated" in result.checks_passed

    def test_g1_fails_when_not_germinated(self):
        """G1 should fail when not in GERMINATED stage."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.DORMANT,
        )

        result = gates.check_gate(state, SeedStage.TRAINING)

        assert not result.passed
        assert "not_germinated" in result.checks_failed


class TestG2Gate:
    """Tests for G2 gate (blending readiness)."""

    def test_g2_passes_with_all_conditions_met(self):
        """G2 should pass when improvement, gradient, and readiness conditions are met."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Set improvement >= threshold
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(50.0 + DEFAULT_MIN_TRAINING_IMPROVEMENT + 1.0)

        # Set gradient ratio >= threshold
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Accumulate epochs for readiness
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert result.passed
        assert result.gate == GateLevel.G2

    def test_g2_fails_with_insufficient_improvement(self):
        """G2 should fail when improvement below threshold."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Insufficient improvement
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(50.0 + DEFAULT_MIN_TRAINING_IMPROVEMENT - 0.5)

        # Other conditions pass
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert any("improvement_insufficient" in check for check in result.checks_failed)

    def test_g2_fails_with_low_gradient_ratio(self):
        """G2 should fail when gradient ratio below threshold."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Good improvement
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)

        # Low gradient ratio
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD - 0.01

        # Enough epochs
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert any("gradient" in check for check in result.checks_failed)

    def test_g2_fails_without_seed_readiness(self):
        """G2 should fail when seed hasn't trained enough epochs."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Good improvement
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)

        # Good gradient ratio
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Not enough epochs (below min_blending_epochs)
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS - 1

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert "seed_not_ready" in result.checks_failed

    def test_g2_requires_all_three_conditions(self):
        """G2 requires improvement AND gradient ratio AND seed readiness."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Only improvement passes
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)  # 10% improvement
        state.metrics.seed_gradient_norm_ratio = 0.0
        state.metrics.epochs_in_current_stage = 0

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        # Should have multiple failures
        assert len(result.checks_failed) >= 2


class TestG3Gate:
    """Tests for G3 gate (holding readiness)."""

    def test_g3_passes_with_alpha_and_epochs(self):
        """G3 should pass when alpha target is reached and epochs are sufficient."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.BLENDING,
        )

        state.alpha = 1.0
        state.alpha_controller.alpha = state.alpha
        state.alpha_controller.alpha_target = 1.0
        state.alpha_controller.alpha_mode = AlphaMode.HOLD
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.HOLDING)

        assert result.passed
        assert result.gate == GateLevel.G3
        assert "alpha_target_full" in result.checks_passed
        assert "alpha_target_reached" in result.checks_passed
        assert "blending_complete" in result.checks_passed

    def test_g3_fails_with_low_alpha(self):
        """G3 should fail when alpha is not at the controller target."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.BLENDING,
        )

        state.alpha = 0.5
        state.alpha_controller.alpha = state.alpha
        state.alpha_controller.alpha_target = 1.0
        state.alpha_controller.alpha_mode = AlphaMode.UP
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.HOLDING)

        assert not result.passed
        assert any("alpha_not_at_target" in check for check in result.checks_failed)

    def test_g3_fails_with_insufficient_epochs(self):
        """G3 should fail when not enough epochs in blending stage."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.BLENDING,
        )

        state.alpha = 1.0
        state.alpha_controller.alpha = state.alpha
        state.alpha_controller.alpha_target = 1.0
        state.alpha_controller.alpha_mode = AlphaMode.HOLD
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS - 1

        result = gates.check_gate(state, SeedStage.HOLDING)

        assert not result.passed
        assert any("blending_incomplete" in check for check in result.checks_failed)


class TestG5Gate:
    """Tests for G5 gate (fossilization readiness)."""

    def test_g5_passes_with_counterfactual_and_health(self):
        """G5 should pass with sufficient counterfactual and healthy state."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.HOLDING,
        )

        state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION + 1.0
        state.is_healthy = True

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.passed
        assert result.gate == GateLevel.G5
        assert any("sufficient_contribution" in check for check in result.checks_passed)
        assert "healthy" in result.checks_passed

    def test_g5_fails_without_counterfactual(self):
        """G5 should fail when counterfactual is None."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.HOLDING,
        )

        state.metrics.counterfactual_contribution = None
        state.is_healthy = True

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert "counterfactual_not_available" in result.checks_failed

    def test_g5_fails_with_low_counterfactual(self):
        """G5 should fail when counterfactual below threshold."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.HOLDING,
        )

        state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION - 0.5
        state.is_healthy = True

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in check for check in result.checks_failed)

    def test_g5_fails_when_unhealthy(self):
        """G5 should fail when seed is not healthy."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.HOLDING,
        )

        state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION + 1.0
        state.is_healthy = False  # Unhealthy

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert "unhealthy" in result.checks_failed


class TestGateResultStructure:
    """Tests for GateResult data structure."""

    def test_gate_result_has_all_fields(self):
        """GateResult should have all expected fields."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.GERMINATED,
        )

        result = gates.check_gate(state, SeedStage.TRAINING)

        assert result.gate is not None
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)
        assert isinstance(result.checks_passed, list)
        assert isinstance(result.checks_failed, list)

    def test_gate_result_score_bounded(self):
        """GateResult score should be in [0, 1]."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Set conditions for G2
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert 0.0 <= result.score <= 1.0


class TestCustomGateThresholds:
    """Tests for custom gate threshold configuration."""

    def test_custom_training_improvement(self):
        """QualityGates should accept custom min_training_improvement."""
        gates = QualityGates(min_training_improvement=20.0)  # High threshold

        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Only 10% improvement (below 20% threshold)
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert any("improvement_insufficient" in check for check in result.checks_failed)

    def test_custom_blending_epochs(self):
        """QualityGates should accept custom min_blending_epochs."""
        gates = QualityGates(min_blending_epochs=10)  # Higher threshold

        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.BLENDING,
        )

        state.alpha = 1.0
        state.metrics.epochs_in_current_stage = 5  # Less than 10

        result = gates.check_gate(state, SeedStage.HOLDING)

        assert not result.passed

    def test_custom_gradient_ratio(self):
        """QualityGates should accept custom min_seed_gradient_ratio."""
        gates = QualityGates(min_seed_gradient_ratio=0.5)  # High threshold

        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)
        state.metrics.seed_gradient_norm_ratio = 0.3  # Below 0.5
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert any("gradient" in check for check in result.checks_failed)


class TestGateLevelMapping:
    """Tests for gate level to stage mapping."""

    def test_germinated_maps_to_g0(self):
        """GERMINATED target should map to G0."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.DORMANT)

        result = gates.check_gate(state, SeedStage.GERMINATED)

        assert result.gate == GateLevel.G0

    def test_training_maps_to_g1(self):
        """TRAINING target should map to G1."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.GERMINATED)

        result = gates.check_gate(state, SeedStage.TRAINING)

        assert result.gate == GateLevel.G1

    def test_blending_maps_to_g2(self):
        """BLENDING target should map to G2."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.TRAINING)

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert result.gate == GateLevel.G2

    def test_holding_maps_to_g3(self):
        """HOLDING target should map to G3."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.BLENDING)

        result = gates.check_gate(state, SeedStage.HOLDING)

        assert result.gate == GateLevel.G3

    def test_fossilized_maps_to_g5(self):
        """FOSSILIZED target should map to G5."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.HOLDING)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.gate == GateLevel.G5

    def test_unmapped_stage_raises_valueerror(self):
        """Unmapped stages must raise ValueError, not silently default to G0."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.DORMANT)

        # UNKNOWN is not in the gate mapping and should raise
        import pytest
        with pytest.raises(ValueError, match="No gate defined"):
            gates.check_gate(state, SeedStage.UNKNOWN)

    def test_failure_stage_raises_valueerror(self):
        """Failure stages (PRUNED, EMBARGOED, RESETTING) have no gate mappings."""
        gates = QualityGates()
        state = SeedState(seed_id="test", blueprint_id="noop", stage=SeedStage.TRAINING)

        import pytest
        for failure_stage in [SeedStage.PRUNED, SeedStage.EMBARGOED, SeedStage.RESETTING]:
            with pytest.raises(ValueError, match="No gate defined"):
                gates.check_gate(state, failure_stage)
