"""Tests for sanctum error types.

Verifies error classes store and expose expected attributes.
"""

from __future__ import annotations

import pytest

from esper.karn.sanctum.errors import SanctumTelemetryFatalError


class TestSanctumTelemetryFatalError:
    """Tests for SanctumTelemetryFatalError exception."""

    def test_inherits_from_runtime_error(self) -> None:
        """Error should be a RuntimeError subclass."""
        assert issubclass(SanctumTelemetryFatalError, RuntimeError)

    def test_message_stored(self) -> None:
        """Error message accessible via str() and args."""
        error = SanctumTelemetryFatalError("Test failure", "traceback here")
        assert str(error) == "Test failure"
        assert error.args == ("Test failure",)

    def test_traceback_attribute(self) -> None:
        """Traceback stored as separate attribute."""
        tb = "File 'test.py', line 1\n  raise ValueError"
        error = SanctumTelemetryFatalError("Boom", tb)
        assert error.traceback == tb

    def test_can_be_raised_and_caught(self) -> None:
        """Error can be raised and caught like normal exception."""
        with pytest.raises(SanctumTelemetryFatalError) as exc_info:
            raise SanctumTelemetryFatalError("Fatal error", "stack trace")

        assert "Fatal error" in str(exc_info.value)
        assert exc_info.value.traceback == "stack trace"

    def test_empty_traceback_allowed(self) -> None:
        """Empty traceback string is valid."""
        error = SanctumTelemetryFatalError("No trace", "")
        assert error.traceback == ""
