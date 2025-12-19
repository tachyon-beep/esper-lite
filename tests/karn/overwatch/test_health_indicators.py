"""Tests for health indicator logic."""

from __future__ import annotations


class TestTrendArrow:
    """Tests for trend arrow rendering."""

    def test_trend_arrow_rising(self) -> None:
        """Positive trend shows up arrow."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(0.05) == "↑"

    def test_trend_arrow_falling(self) -> None:
        """Negative trend shows down arrow."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(-0.05) == "↓"

    def test_trend_arrow_stable(self) -> None:
        """Near-zero trend shows right arrow (stable)."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(0.001) == "→"
        assert trend_arrow(-0.001) == "→"
        assert trend_arrow(0.0) == "→"

    def test_trend_arrow_custom_threshold(self) -> None:
        """Custom threshold changes sensitivity."""
        from esper.karn.overwatch.display_state import trend_arrow

        # With threshold=0.1, a trend of 0.05 is stable
        assert trend_arrow(0.05, threshold=0.1) == "→"
        # With threshold=0.01, a trend of 0.05 is rising
        assert trend_arrow(0.05, threshold=0.01) == "↑"


class TestHealthLevel:
    """Tests for health level classification."""

    def test_kl_health_ok(self) -> None:
        """Low KL divergence is healthy."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.01) == "ok"
        assert kl_health(0.02) == "ok"

    def test_kl_health_warn(self) -> None:
        """Medium KL divergence is warning."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.03) == "warn"
        assert kl_health(0.04) == "warn"

    def test_kl_health_crit(self) -> None:
        """High KL divergence is critical."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.05) == "crit"
        assert kl_health(0.1) == "crit"

    def test_entropy_health_ok(self) -> None:
        """Good entropy range is healthy."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(1.5) == "ok"
        assert entropy_health(2.0) == "ok"

    def test_entropy_health_warn_low(self) -> None:
        """Low entropy (collapsed) is warning."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(0.3) == "warn"

    def test_entropy_health_crit_collapsed(self) -> None:
        """Very low entropy is critical (policy collapsed)."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(0.1) == "crit"

    def test_ev_health_ok(self) -> None:
        """High explained variance is healthy."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.8) == "ok"
        assert ev_health(0.95) == "ok"

    def test_ev_health_warn(self) -> None:
        """Medium explained variance is warning."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.5) == "warn"

    def test_ev_health_crit(self) -> None:
        """Low explained variance is critical."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.2) == "crit"
        assert ev_health(-0.1) == "crit"


class TestFormatRuntime:
    """Tests for runtime formatting."""

    def test_format_runtime_seconds(self) -> None:
        """Short runtime shows seconds."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(45.0) == "45s"

    def test_format_runtime_minutes(self) -> None:
        """Minutes shown with seconds."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(125.0) == "2m 5s"

    def test_format_runtime_hours(self) -> None:
        """Hours shown with minutes."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(3725.0) == "1h 2m"

    def test_format_runtime_zero(self) -> None:
        """Zero runtime shows 0s."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(0.0) == "0s"
