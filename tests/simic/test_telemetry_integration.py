"""Tests for telemetry integration in training loop.

Verifies that HealthMonitor and CounterfactualHelper are properly wired
to emit telemetry events via callback injection.
"""

from esper.karn.health import HealthMonitor
from esper.simic.attribution import CounterfactualHelper, ContributionResult
from esper.leyline import TelemetryEventType


class TestHealthMonitorIntegration:
    """Test HealthMonitor callback wiring."""

    def test_emits_memory_warning_via_callback(self):
        """Verify MEMORY_WARNING event emitted when GPU utilization exceeds threshold."""
        events = []
        monitor = HealthMonitor(emit_callback=events.append)

        # Trigger warning (95% > default 85% threshold)
        emitted = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        assert emitted is True
        assert len(events) == 1
        assert events[0].event_type == TelemetryEventType.MEMORY_WARNING
        assert events[0].data["gpu_utilization"] == 0.95
        assert events[0].data["gpu_allocated_gb"] == 10.0
        assert events[0].data["gpu_total_gb"] == 12.0

    def test_no_warning_below_threshold(self):
        """Verify no event emitted when GPU utilization is below threshold."""
        events = []
        monitor = HealthMonitor(emit_callback=events.append)

        # Below threshold (50% < default 85%)
        emitted = monitor._check_memory_and_warn(
            gpu_utilization=0.50,
            gpu_allocated_gb=6.0,
            gpu_total_gb=12.0,
        )

        assert emitted is False
        assert len(events) == 0

    def test_cooldown_prevents_spam(self):
        """Verify cooldown prevents repeated warnings."""
        events = []
        monitor = HealthMonitor(
            emit_callback=events.append,
            memory_warning_cooldown=10.0,  # 10 second cooldown
        )

        # First warning
        monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        # Second warning within cooldown - should be suppressed
        emitted = monitor._check_memory_and_warn(
            gpu_utilization=0.96,
            gpu_allocated_gb=10.5,
            gpu_total_gb=12.0,
        )

        assert emitted is False
        assert len(events) == 1  # Only first warning

    def test_no_callback_no_crash(self):
        """Verify no crash when emit_callback is None."""
        monitor = HealthMonitor(emit_callback=None)

        # Should not raise - returns True because threshold exceeded
        # (return value indicates threshold breach, not actual emission)
        emitted = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        # Returns True because threshold was exceeded (even without callback)
        assert emitted is True


class TestCounterfactualHelperIntegration:
    """Test CounterfactualHelper callback wiring."""

    def test_computes_contributions_without_crash(self):
        """Verify compute_contributions works without emit_callback."""
        helper = CounterfactualHelper(emit_callback=None)

        # Simple evaluate function
        def evaluate_fn(alphas: dict[str, float]) -> tuple[float, float]:
            # Return loss, accuracy - higher alpha means better accuracy
            enabled = sum(1 for a in alphas.values() if a > 0.5)
            return 0.5, enabled * 25.0  # 25% per slot

        contributions = helper.compute_contributions(
            slot_ids=["r0c0", "r0c1"],
            evaluate_fn=evaluate_fn,
        )

        assert "r0c0" in contributions
        assert "r0c1" in contributions
        assert isinstance(contributions["r0c0"], ContributionResult)
        assert isinstance(contributions["r0c1"], ContributionResult)

    def test_marginal_contribution_calculated(self):
        """Verify marginal contributions are calculated correctly."""
        helper = CounterfactualHelper(emit_callback=None)

        # Evaluate function where removing slot drops accuracy
        full_acc = 90.0
        without_r0c0 = 70.0
        without_r0c1 = 85.0

        def evaluate_fn(alphas: dict[str, float]) -> tuple[float, float]:
            if all(a > 0.5 for a in alphas.values()):
                return 0.5, full_acc
            elif alphas.get("r0c0", 1.0) < 0.5:
                return 0.6, without_r0c0
            elif alphas.get("r0c1", 1.0) < 0.5:
                return 0.55, without_r0c1
            else:
                return 0.7, 60.0

        contributions = helper.compute_contributions(
            slot_ids=["r0c0", "r0c1"],
            evaluate_fn=evaluate_fn,
        )

        # r0c0 contributes more (removing it drops acc by 20%)
        assert contributions["r0c0"].contribution > contributions["r0c1"].contribution

    def test_empty_slots_returns_empty(self):
        """Verify empty slot list returns empty dict."""
        helper = CounterfactualHelper(emit_callback=None)

        contributions = helper.compute_contributions(
            slot_ids=[],
            evaluate_fn=lambda x: (0.5, 50.0),
        )

        assert contributions == {}


class TestParallelEnvStateFields:
    """Test that ParallelEnvState has the required telemetry fields."""

    def test_has_health_monitor_field(self):
        """Verify ParallelEnvState has health_monitor field."""
        from esper.simic.training.parallel_env_state import ParallelEnvState
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ParallelEnvState)}
        assert "health_monitor" in fields

    def test_has_counterfactual_helper_field(self):
        """Verify ParallelEnvState has counterfactual_helper field."""
        from esper.simic.training.parallel_env_state import ParallelEnvState
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ParallelEnvState)}
        assert "counterfactual_helper" in fields

    def test_fields_default_to_none(self):
        """Verify new telemetry fields default to None."""
        from esper.simic.training.parallel_env_state import ParallelEnvState
        import dataclasses

        for f in dataclasses.fields(ParallelEnvState):
            if f.name == "health_monitor":
                assert f.default is None
            elif f.name == "counterfactual_helper":
                assert f.default is None
