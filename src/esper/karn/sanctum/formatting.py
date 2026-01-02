"""Shared formatting utilities for Sanctum TUI.

This module provides consistent formatting functions used across multiple widgets
and modals to eliminate duplication and ensure visual consistency.

Usage:
    from esper.karn.sanctum.formatting import format_runtime, format_params
"""

from __future__ import annotations


def format_runtime(seconds: float, include_seconds_in_hours: bool = False) -> str:
    """Format runtime duration consistently across all widgets.

    Args:
        seconds: Duration in seconds
        include_seconds_in_hours: If True, show "1h 23m 45s"; if False, show "1h 23m"
            Default is False to match RunHeader's compact display.

    Returns:
        Formatted string like "1h 23m", "5m 30s", or "45s"

    Examples:
        >>> format_runtime(3665)
        '1h 1m'
        >>> format_runtime(3665, include_seconds_in_hours=True)
        '1h 1m 5s'
        >>> format_runtime(125)
        '2m 5s'
        >>> format_runtime(45)
        '45s'
        >>> format_runtime(0)
        '--'
    """
    if seconds <= 0:
        return "--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        if include_seconds_in_hours:
            return f"{hours}h {minutes}m {secs}s"
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_params(params: int, precision: int = 1) -> str:
    """Format parameter count with K/M suffix.

    Args:
        params: Number of parameters
        precision: Decimal places for M/K suffix (default 1)

    Returns:
        Formatted string like "2.5M", "150.0K", or "500"

    Examples:
        >>> format_params(2_500_000)
        '2.5M'
        >>> format_params(150_000)
        '150.0K'
        >>> format_params(500)
        '500'
        >>> format_params(0)
        '0'
    """
    if params >= 1_000_000:
        return f"{params / 1_000_000:.{precision}f}M"
    elif params >= 1_000:
        return f"{params / 1_000:.{precision}f}K"
    return str(params)


__all__ = ["format_runtime", "format_params"]
