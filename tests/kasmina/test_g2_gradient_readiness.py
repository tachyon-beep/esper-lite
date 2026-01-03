"""Test G2 gate gradient-based seed readiness."""

from esper.kasmina.slot import SeedState, SeedMetrics, QualityGates
from esper.leyline import SeedStage


class TestG2GradientReadiness:
    """Verify G2 uses seed gradient statistics."""

    def test_g2_checks_seed_gradient_norm_ratio(self):
        """G2 should check seed gradient norm relative to host."""
        gates = QualityGates()

        # Create state with good global improvement but low seed gradient activity
        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="test_slot",
            stage=SeedStage.TRAINING,
        )
        state.metrics = SeedMetrics()
        # Set base accuracy to create improvement
        state.metrics.accuracy_at_stage_start = 50.0
        state.metrics.current_val_accuracy = 52.0  # 2.0% improvement
        state.metrics.epochs_in_current_stage = 5
        state.metrics.seed_gradient_norm_ratio = 0.01  # Very low seed activity

        result = gates._check_g2(state)

        # Should fail due to low seed gradient activity
        assert "seed_gradient_low" in result.checks_failed or not result.passed

    def test_g2_passes_with_sufficient_gradient_ratio(self):
        """G2 should pass when seed gradient ratio is sufficient."""
        gates = QualityGates()

        # Create state with good global improvement AND good seed gradient activity
        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="test_slot",
            stage=SeedStage.TRAINING,
        )
        state.metrics = SeedMetrics()
        # Set base accuracy to create improvement
        state.metrics.accuracy_at_stage_start = 50.0
        state.metrics.current_val_accuracy = 52.0  # 2.0% improvement
        state.metrics.epochs_in_current_stage = 5
        state.metrics.seed_gradient_norm_ratio = 0.10  # Good seed activity (10%)

        result = gates._check_g2(state)

        # Should pass with good gradient ratio
        assert result.passed
        assert any("seed_gradient_active" in check for check in result.checks_passed)

    def test_g2_fails_with_low_gradient_despite_global_improvement(self):
        """G2 should fail if seed gradient is too low, even with global improvement."""
        gates = QualityGates()

        # Create state with excellent global improvement but negligible seed activity
        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="test_slot",
            stage=SeedStage.TRAINING,
        )
        state.metrics = SeedMetrics()
        # Set base accuracy to create excellent improvement
        state.metrics.accuracy_at_stage_start = 50.0
        state.metrics.current_val_accuracy = 60.0  # 10.0% improvement
        state.metrics.epochs_in_current_stage = 10
        state.metrics.seed_gradient_norm_ratio = 0.001  # Nearly zero seed activity

        result = gates._check_g2(state)

        # Should fail despite global improvement
        assert not result.passed
        assert any("seed_gradient_low" in check for check in result.checks_failed)

    def test_g2_fails_loudly_when_gradient_stats_never_measured(self):
        """G2 should explicitly report when gradient stats were never collected.

        This catches training loop coupling issues where the loop forgets to
        call capture_gradient_telemetry(). Without this check, the gate would
        silently fail with 'gradient_low_0.00' which is misleading.
        """
        gates = QualityGates()

        # Create state with good metrics but gradient ratio never set (None)
        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="test_slot",
            stage=SeedStage.TRAINING,
        )
        state.metrics = SeedMetrics()
        state.metrics.accuracy_at_stage_start = 50.0
        state.metrics.current_val_accuracy = 60.0  # Excellent improvement
        state.metrics.epochs_in_current_stage = 10
        # NOTE: seed_gradient_norm_ratio defaults to None (never measured)
        assert state.metrics.seed_gradient_norm_ratio is None

        result = gates._check_g2(state)

        # Should fail with explicit "never measured" message
        assert not result.passed
        assert "gradient_stats_never_measured" in result.checks_failed
