"""Overwatch - Web-based training monitoring dashboard.

Overwatch provides a Vue 3 web interface for real-time training monitoring,
mirroring Sanctum TUI functionality with enhanced visualizations.

Usage:
    from esper.karn.overwatch import OverwatchBackend

    backend = OverwatchBackend(port=8080)
    hub.add_backend(backend)
"""

from esper.karn.overwatch.backend import OverwatchBackend

__all__ = ["OverwatchBackend"]
