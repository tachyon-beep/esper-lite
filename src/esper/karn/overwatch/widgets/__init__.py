"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

# Lazy import - Textual may not be installed
try:
    from esper.karn.overwatch.widgets.help import HelpOverlay
except ImportError:
    HelpOverlay = None  # type: ignore[misc, assignment]

__all__ = [
    "HelpOverlay",
]
