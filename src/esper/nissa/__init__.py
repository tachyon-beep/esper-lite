"""Nissa observability stack helpers.

Consumes telemetry envelopes and exports Prometheus/Elasticsearch metrics as
outlined in `docs/design/detailed_design/10-nissa.md`.
"""

from .observability import NissaIngestor, NissaIngestorConfig

__all__ = ["NissaIngestor", "NissaIngestorConfig"]
