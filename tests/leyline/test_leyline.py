"""Tests for leyline.py - shared contracts."""

import pytest
from esper.leyline import (
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
    CommandType,
    RiskLevel,
    AdaptationCommand,
    TrainingMetrics,
    SeedMetrics,
    SeedStateReport,
    GateLevel,
    GateResult,
)


class TestSeedStage:
    """Tests for SeedStage enum and transitions."""

    def test_all_stages_have_transitions(self):
        """Every stage should have an entry in VALID_TRANSITIONS."""
        # Exclude terminal/special stages that have no outgoing transitions
        excluded_stages = {SeedStage.DORMANT, SeedStage.CULLED, SeedStage.FOSSILIZED}
        for stage in SeedStage:
            if stage not in excluded_stages:
                assert stage in VALID_TRANSITIONS, f"{stage} missing from VALID_TRANSITIONS"

    def test_dormant_can_only_germinate(self):
        """DORMANT should only transition to GERMINATED."""
        assert VALID_TRANSITIONS[SeedStage.DORMANT] == (SeedStage.GERMINATED,)

    def test_fossilized_is_terminal(self):
        """FOSSILIZED should have no valid transitions (terminal)."""
        assert VALID_TRANSITIONS[SeedStage.FOSSILIZED] == ()

    def test_valid_transition_helper(self):
        """Test is_valid_transition helper function."""
        assert is_valid_transition(SeedStage.DORMANT, SeedStage.GERMINATED)
        assert not is_valid_transition(SeedStage.DORMANT, SeedStage.TRAINING)
        assert not is_valid_transition(SeedStage.FOSSILIZED, SeedStage.DORMANT)

    def test_terminal_stage_helper(self):
        """Test is_terminal_stage helper function."""
        assert is_terminal_stage(SeedStage.FOSSILIZED)
        assert not is_terminal_stage(SeedStage.TRAINING)
        assert not is_terminal_stage(SeedStage.CULLED)

    def test_active_stage_helper(self):
        """Test is_active_stage helper function."""
        assert is_active_stage(SeedStage.TRAINING)
        assert is_active_stage(SeedStage.BLENDING)
        assert is_active_stage(SeedStage.FOSSILIZED)
        assert not is_active_stage(SeedStage.DORMANT)
        assert not is_active_stage(SeedStage.CULLED)

    def test_failure_stage_helper(self):
        """Test is_failure_stage helper function."""
        assert is_failure_stage(SeedStage.CULLED)
        assert is_failure_stage(SeedStage.EMBARGOED)
        assert is_failure_stage(SeedStage.RESETTING)
        assert not is_failure_stage(SeedStage.FOSSILIZED)
        assert not is_failure_stage(SeedStage.TRAINING)

    def test_lifecycle_path_is_valid(self):
        """Test that the happy path lifecycle is all valid transitions."""
        happy_path = [
            SeedStage.DORMANT,
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.PROBATIONARY,
            SeedStage.FOSSILIZED,
        ]
        for i in range(len(happy_path) - 1):
            assert is_valid_transition(happy_path[i], happy_path[i + 1]), \
                f"Transition {happy_path[i]} -> {happy_path[i + 1]} should be valid"

    def test_cull_path_from_training(self):
        """Test that TRAINING can transition to CULLED."""
        assert is_valid_transition(SeedStage.TRAINING, SeedStage.CULLED)

    def test_recovery_path_after_cull(self):
        """Test the recovery path: CULLED -> EMBARGOED -> RESETTING -> DORMANT."""
        assert is_valid_transition(SeedStage.CULLED, SeedStage.EMBARGOED)
        assert is_valid_transition(SeedStage.EMBARGOED, SeedStage.RESETTING)
        assert is_valid_transition(SeedStage.RESETTING, SeedStage.DORMANT)


class TestAdaptationCommand:
    """Tests for AdaptationCommand dataclass."""

    def test_default_values(self):
        """Test default values are sensible."""
        cmd = AdaptationCommand()
        assert cmd.command_type == CommandType.GERMINATE
        assert cmd.confidence == 1.0
        assert cmd.risk_level == RiskLevel.GREEN
        assert cmd.command_id  # Should have auto-generated ID

    def test_command_is_frozen(self):
        """AdaptationCommand should be immutable (frozen)."""
        cmd = AdaptationCommand()
        with pytest.raises(Exception):  # FrozenInstanceError
            cmd.confidence = 0.5

    def test_unique_command_ids(self):
        """Each command should have a unique ID."""
        cmd1 = AdaptationCommand()
        cmd2 = AdaptationCommand()
        assert cmd1.command_id != cmd2.command_id


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = TrainingMetrics()
        assert metrics.epoch == 0
        assert metrics.train_loss == 0.0
        assert metrics.best_val_loss == float('inf')

    def test_can_set_values(self):
        """Test setting values."""
        metrics = TrainingMetrics(
            epoch=10,
            val_accuracy=75.5,
            plateau_epochs=3,
        )
        assert metrics.epoch == 10
        assert metrics.val_accuracy == 75.5
        assert metrics.plateau_epochs == 3


class TestSeedStateReport:
    """Tests for SeedStateReport dataclass."""

    def test_default_values(self):
        """Test default values."""
        report = SeedStateReport()
        assert report.stage == SeedStage.UNKNOWN
        assert report.is_healthy is True
        assert report.seed_id == ""

    def test_with_metrics(self):
        """Test with nested metrics."""
        metrics = SeedMetrics(epochs_total=5, current_val_accuracy=70.0)
        report = SeedStateReport(
            seed_id="test_seed",
            stage=SeedStage.TRAINING,
            metrics=metrics,
        )
        assert report.seed_id == "test_seed"
        assert report.metrics.epochs_total == 5


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_passed_gate(self):
        """Test a passing gate result."""
        result = GateResult(
            gate=GateLevel.G0,
            passed=True,
            score=0.95,
            checks_passed=["sanity_check", "param_count"],
        )
        assert result.passed
        assert result.score == 0.95
        assert len(result.checks_passed) == 2

    def test_failed_gate(self):
        """Test a failing gate result."""
        result = GateResult(
            gate=GateLevel.G2,
            passed=False,
            score=0.3,
            checks_failed=["improvement_threshold"],
            message="Insufficient improvement",
        )
        assert not result.passed
        assert "improvement_threshold" in result.checks_failed
