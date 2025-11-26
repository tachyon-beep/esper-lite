"""Weatherlight supervisor package.

Exposes the async service runner that composes Esper subsystems as described
in `docs/prototype-delta/weatherlight/README.md`.
"""

from .service_runner import main, run_service

__all__ = ["run_service", "main"]
