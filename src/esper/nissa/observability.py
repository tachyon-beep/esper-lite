"""Nissa telemetry ingestion scaffold.

Responsible for mapping telemetry envelopes into Prometheus and Elasticsearch in
alignment with `docs/design/detailed_design/10-nissa.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

from elasticsearch import Elasticsearch
from prometheus_client import CollectorRegistry, Counter

from esper.core import FieldReport, SystemStatePacket


@dataclass(slots=True)
class NissaIngestorConfig:
    prometheus_gateway: str
    elasticsearch_url: str


class NissaIngestor:
    """Stub telemetry ingestor for Prometheus and Elasticsearch."""

    def __init__(self, config: NissaIngestorConfig) -> None:
        self._config = config
        self._registry = CollectorRegistry()
        self._run_counter = Counter(
            "esper_training_runs",
            "Count of training runs processed",
            registry=self._registry,
        )
        self._es = Elasticsearch(hosts=[config.elasticsearch_url])

    def ingest_state(self, packet: SystemStatePacket) -> None:
        self._run_counter.inc()
        self._index_document("system_state", packet.model_dump())

    def ingest_field_report(self, report: FieldReport) -> None:
        self._index_document("field_report", report.model_dump())

    def metrics(self) -> dict[str, str]:
        """Return the Prometheus metrics exposition text."""

        return {"metrics": "# Prometheus metrics placeholder"}

    def _index_document(self, index: str, document: dict[str, object]) -> None:
        self._es.index(index=index, document=document)


__all__ = ["NissaIngestor", "NissaIngestorConfig"]
