"""Tests for value function statistics display in PPOHealthPanel.

These tests verify the value function divergence detection logic, including:
- Healthy values showing ok status
- Exploding values (10x initial spread) showing critical
- Collapsed values (constant) showing critical
- High coefficient of variation showing warning
"""

import pytest

from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot
from esper.karn.sanctum.widgets.tamiyo_brain_v2.ppo_health import PPOHealthPanel


class TestValueFunctionDisplay:
    """Test value function statistics display."""

    def test_healthy_values_show_ok(self) -> None:
        """Normal value range should show ok status."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=2.0,
                value_min=-3.0,
                value_max=15.0,
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "ok"

    def test_exploding_values_show_critical(self) -> None:
        """Values 10x initial spread should show critical."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=50.0,
                value_std=30.0,
                value_min=-50.0,
                value_max=150.0,  # 200 range vs initial 10 = 20x
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_collapsed_values_show_critical(self) -> None:
        """Constant values should show critical (collapsed)."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=0.001,
                value_min=4.999,
                value_max=5.001,
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_high_cov_shows_warning(self) -> None:
        """High coefficient of variation should show warning."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=12.0,  # CoV = 12/5 = 2.4 > 2.0
                value_min=-10.0,
                value_max=25.0,
                initial_value_spread=None,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "warning"

    def test_extreme_cov_shows_critical(self) -> None:
        """Extreme coefficient of variation (>3.0) should show critical."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=20.0,  # CoV = 20/5 = 4.0 > 3.0
                value_min=-20.0,
                value_max=35.0,
                initial_value_spread=None,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_warning_at_5x_spread(self) -> None:
        """Values 5x initial spread should show warning."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=25.0,
                value_std=10.0,
                value_min=-15.0,
                value_max=45.0,  # 60 range vs initial 10 = 6x
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "warning"

    def test_absolute_fallback_critical(self) -> None:
        """Absolute fallback: range >1000 or max >10000 is critical."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5000.0,
                value_std=2000.0,
                value_min=-500.0,
                value_max=12000.0,  # max > 10000
                initial_value_spread=None,  # No initial spread (warmup)
            ),
            current_batch=30,  # Still in warmup
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_absolute_fallback_warning(self) -> None:
        """Absolute fallback: range >500 or max >5000 is warning."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5200.0,
                value_std=150.0,
                value_min=4800.0,
                value_max=5500.0,  # max > 5000 but < 10000, range = 700 < 1000
                initial_value_spread=None,  # No initial spread (warmup)
            ),
            current_batch=30,  # Still in warmup
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "warning"

    def test_zero_mean_avoids_cov_division(self) -> None:
        """When mean is near zero, CoV check should be skipped."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=0.05,  # Near zero
                value_std=5.0,  # Would be huge CoV if not skipped
                value_min=-5.0,
                value_max=5.0,
                initial_value_spread=None,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        # Should not trigger CoV check, falls through to absolute check
        assert status == "ok"  # Range=10, max=5, both below thresholds

    def test_render_value_stats_shows_range_and_std(self) -> None:
        """_render_value_stats should show range and std deviation."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=2.5,
                value_min=-3.0,
                value_max=15.0,
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        output = panel._render_value_stats()
        output_str = str(output)

        assert "Value Range" in output_str
        assert "-3.0" in output_str  # min
        assert "15.0" in output_str  # max
        assert "s=" in output_str  # std indicator

    def test_render_value_stats_critical_shows_alert(self) -> None:
        """Critical value status should show alert indicator."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=50.0,
                value_std=30.0,
                value_min=-50.0,
                value_max=150.0,  # 200 range vs initial 10 = 20x -> critical
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        output = panel._render_value_stats()
        output_str = str(output)

        assert "!" in output_str  # Alert indicator for critical status
