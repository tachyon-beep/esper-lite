"""Tests for sanctum formatting utilities.

These are pure functions with no side effects - ideal for unit testing.
"""

from __future__ import annotations

from esper.karn.sanctum.formatting import format_runtime, format_params


class TestFormatRuntime:
    """Tests for format_runtime function."""

    def test_zero_seconds_returns_dashes(self) -> None:
        """Zero or negative duration shows placeholder."""
        assert format_runtime(0) == "--"
        assert format_runtime(-1) == "--"
        assert format_runtime(-100) == "--"

    def test_seconds_only(self) -> None:
        """Under 60 seconds shows only seconds."""
        assert format_runtime(1) == "1s"
        assert format_runtime(45) == "45s"
        assert format_runtime(59) == "59s"

    def test_minutes_and_seconds(self) -> None:
        """Between 1-60 minutes shows minutes and seconds."""
        assert format_runtime(60) == "1m 0s"
        assert format_runtime(61) == "1m 1s"
        assert format_runtime(125) == "2m 5s"
        assert format_runtime(3599) == "59m 59s"

    def test_hours_and_minutes_default(self) -> None:
        """Over 60 minutes shows hours and minutes (no seconds by default)."""
        assert format_runtime(3600) == "1h 0m"
        assert format_runtime(3665) == "1h 1m"
        assert format_runtime(7325) == "2h 2m"

    def test_hours_with_seconds_flag(self) -> None:
        """include_seconds_in_hours flag adds seconds to hour display."""
        assert format_runtime(3665, include_seconds_in_hours=True) == "1h 1m 5s"
        assert format_runtime(3600, include_seconds_in_hours=True) == "1h 0m 0s"
        assert format_runtime(7325, include_seconds_in_hours=True) == "2h 2m 5s"

    def test_fractional_seconds_truncated(self) -> None:
        """Fractional seconds are truncated (not rounded)."""
        assert format_runtime(1.9) == "1s"
        assert format_runtime(59.99) == "59s"
        assert format_runtime(60.5) == "1m 0s"


class TestFormatParams:
    """Tests for format_params function."""

    def test_zero_params(self) -> None:
        """Zero parameters shown as literal '0'."""
        assert format_params(0) == "0"

    def test_small_numbers_no_suffix(self) -> None:
        """Numbers under 1000 shown without suffix."""
        assert format_params(1) == "1"
        assert format_params(500) == "500"
        assert format_params(999) == "999"

    def test_thousands_k_suffix(self) -> None:
        """Numbers 1000-999999 shown with K suffix."""
        assert format_params(1000) == "1.0K"
        assert format_params(1500) == "1.5K"
        assert format_params(150_000) == "150.0K"
        assert format_params(999_999) == "1000.0K"  # Just under 1M

    def test_millions_m_suffix(self) -> None:
        """Numbers >= 1M shown with M suffix."""
        assert format_params(1_000_000) == "1.0M"
        assert format_params(2_500_000) == "2.5M"
        assert format_params(100_000_000) == "100.0M"

    def test_custom_precision(self) -> None:
        """Precision parameter controls decimal places."""
        assert format_params(1_234_567, precision=0) == "1M"
        assert format_params(1_234_567, precision=2) == "1.23M"
        assert format_params(1_234_567, precision=3) == "1.235M"
        assert format_params(1_500, precision=0) == "2K"  # Rounds
        assert format_params(1_500, precision=2) == "1.50K"
