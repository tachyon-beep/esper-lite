"""Nissa telemetry ingestion scaffold.

Responsible for mapping telemetry envelopes into Prometheus and Elasticsearch in
alignment with `docs/design/detailed_design/10-nissa.md`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch
from google.protobuf.json_format import MessageToDict
from prometheus_client import CollectorRegistry, Counter

from esper.core import FieldReport, SystemStatePacket
from esper.leyline import leyline_pb2

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


@dataclass(slots=True)
class NissaIngestorConfig:
    prometheus_gateway: str
    elasticsearch_url: str


class NissaIngestor:
    """Stub telemetry ingestor for Prometheus and Elasticsearch."""

    def __init__(
        self,
        config: NissaIngestorConfig,
        *,
        es_client: Elasticsearch | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or CollectorRegistry()
        self._run_counter = Counter(
            "esper_training_runs",
            "Count of training runs processed",
            registry=self._registry,
        )
        self._telemetry_counter = Counter(
            "esper_telemetry_packets_total",
            "Count of telemetry packets ingested",
            registry=self._registry,
            labelnames=("source",),
        )
        self._es = es_client or Elasticsearch(hosts=[config.elasticsearch_url])

    def ingest_state(self, packet: SystemStatePacket) -> None:
        self._run_counter.inc()
        self._index_document("system_state", packet.model_dump())

    def ingest_field_report(self, report: FieldReport) -> None:
        self._index_document("field_report", report.model_dump())

    def ingest_telemetry(self, packet: leyline_pb2.TelemetryPacket) -> None:
        self._telemetry_counter.labels(source=packet.source_subsystem).inc()
        self._index_document(
            "telemetry",
            MessageToDict(packet, preserving_proto_field_name=True),
        )

    def metrics(self) -> dict[str, str]:
        """Return the Prometheus metrics exposition text."""

        return {"metrics": "# Prometheus metrics placeholder"}

    def _index_document(self, index: str, document: dict[str, object]) -> None:
        self._es.index(index=index, document=document)

    def consume_packets(
        self,
        packets: Iterable[leyline_pb2.TelemetryPacket],
    ) -> None:
        """Convenience helper to ingest a batch of telemetry packets."""

        for packet in packets:
            self.ingest_telemetry(packet)

    async def consume_from_oona(
        self,
        client: OonaClient,
        *,
        stream: str | None = None,
        count: int = 10,
        block_ms: int = 1000,
    ) -> None:
        """Consume telemetry packets from Oona and ingest them."""

        async def handler(message: OonaMessage) -> None:
            packet = leyline_pb2.TelemetryPacket()
            packet.ParseFromString(message.payload)
            self.ingest_telemetry(packet)

        await client.consume(
            handler,
            stream=stream or client.telemetry_stream,
            count=count,
            block_ms=block_ms,
        )

    @property
    def registry(self) -> CollectorRegistry:
        """Expose the Prometheus registry for HTTP export."""

        return self._registry


__all__ = ["NissaIngestor", "NissaIngestorConfig"]
