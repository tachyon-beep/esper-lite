"""Tamiyo-specific exception types."""

from __future__ import annotations


class TamiyoPersistenceError(RuntimeError):
    """Raised when Tamiyo detects corrupted or unwritable persistence state."""


class TamiyoTimeoutError(RuntimeError):
    """Raised when Tamiyo cannot complete a critical operation within its deadline."""


__all__ = ["TamiyoPersistenceError", "TamiyoTimeoutError"]
