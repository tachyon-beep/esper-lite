"""Sanctum errors.

Sanctum is a developer diagnostics surface. When telemetry is malformed or
contract-breaking, we fail loud: the UI must stop and show the error.
"""

from __future__ import annotations


class SanctumTelemetryFatalError(RuntimeError):
    """Raised when Sanctum telemetry processing encounters a fatal error."""

    def __init__(self, message: str, traceback: str) -> None:
        super().__init__(message)
        self.traceback = traceback

