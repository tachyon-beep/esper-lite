"""Nissa telemetry ingestion scaffold.

Responsible for mapping telemetry envelopes into Prometheus and Elasticsearch in
alignment with `docs/design/detailed_design/10-nissa.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch
from google.protobuf.json_format import MessageToDict
from prometheus_client import CollectorRegistry, Counter

from esper.core import FieldReport, SystemStatePacket
from esper.leyline import leyline_pb2
from esper.nissa.alerts import AlertEngine, AlertRouter, DEFAULT_ALERT_RULES, AlertEvent
from esper.nissa.slo import SLOTracker, SLOStatus

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
        self._alert_router = AlertRouter()
        self._alert_engine = AlertEngine(DEFAULT_ALERT_RULES, router=self._alert_router)
        self._slo_tracker = SLOTracker()
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
        self._state_counter = Counter(
            "esper_system_state_packets_total",
            "Count of system state packets ingested",
            registry=self._registry,
            labelnames=("phase",),
        )
        self._field_report_counter = Counter(
            "esper_field_reports_total",
            "Count of field reports ingested",
            registry=self._registry,
            labelnames=("outcome",),
        )
        self._simic_reward_counter = Counter(
            "esper_simic_training_reward_total",
            "Cumulative Simic training reward",
            registry=self._registry,
        )
        self._simic_iterations_counter = Counter(
            "esper_simic_training_iterations_total",
            "Number of Simic PPO iterations processed",
            registry=self._registry,
        )
        self._es = es_client or Elasticsearch(hosts=[config.elasticsearch_url])

    def ingest_state(self, packet: SystemStatePacket | Mapping[str, object]) -> None:
        """Ingest a system state packet from pydantic or dict payload."""

        if isinstance(packet, SystemStatePacket):
            phase = packet.phase.value
            document = packet.model_dump()
        else:
            document = dict(packet)
            raw_phase = document.get("phase")
            if raw_phase is None:
                raw_phase = document.get("source_subsystem", "unknown")
            phase = _normalise_enum_label(str(raw_phase), prefix="TRAINING_PHASE_")
        self._run_counter.inc()
        self._state_counter.labels(phase=phase).inc()
        self._index_document("system_state", document)

    def ingest_field_report(self, report: FieldReport | Mapping[str, object]) -> None:
        """Ingest a field report packet from pydantic or dict payload."""

        if isinstance(report, FieldReport):
            outcome = report.outcome.value
            document = report.model_dump()
        else:
            document = dict(report)
            raw_outcome = str(document.get("outcome", "unknown"))
            outcome = _normalise_enum_label(raw_outcome, prefix="FIELD_REPORT_OUTCOME_")
        self._field_report_counter.labels(outcome=outcome).inc()
        self._index_document("field_report", document)

    def ingest_telemetry(self, packet: leyline_pb2.TelemetryPacket) -> None:
        self._telemetry_counter.labels(source=packet.source_subsystem).inc()
        document = MessageToDict(packet, preserving_proto_field_name=True)
        metrics = {metric.name: metric.value for metric in packet.metrics}
        if metrics:
            self._alert_engine.evaluate(metrics, packet.source_subsystem)
            self._process_slo_metrics(metrics)

        if packet.source_subsystem == "simic":
            reward = metrics.get("simic.training.reward", 0.0)
            iterations = metrics.get("simic.training.iterations", 0.0)
            self._simic_reward_counter.inc(reward)
            self._simic_iterations_counter.inc(iterations if iterations else 1.0)
            self._index_document("simic_metrics", document)
        else:
            self._index_document("telemetry", document)

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
            if message.message_type == "telemetry":
                packet = leyline_pb2.TelemetryPacket()
                packet.ParseFromString(message.payload)
                self.ingest_telemetry(packet)
            elif message.message_type == "system_state":
                packet = leyline_pb2.SystemStatePacket()
                packet.ParseFromString(message.payload)
                payload = MessageToDict(packet, preserving_proto_field_name=True)
                self.ingest_state(payload)
            elif message.message_type == "field_report":
                field_report = leyline_pb2.FieldReport()
                field_report.ParseFromString(message.payload)
                payload = MessageToDict(field_report, preserving_proto_field_name=True)
                self.ingest_field_report(payload)

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

    @property
    def alert_engine(self) -> AlertEngine:
        return self._alert_engine

    @property
    def active_alerts(self) -> dict[str, AlertEvent]:
        return self._alert_engine.active_alerts

    @property
    def alert_events(self) -> list[AlertEvent]:
        return self._alert_router.events()

    @property
    def slo_tracker(self) -> SLOTracker:
        return self._slo_tracker

    def slo_summary(self) -> dict[str, SLOStatus]:
        return self._slo_tracker.summary()

    def _process_slo_metrics(self, metrics: dict[str, float]) -> None:
        objectives: dict[str, float] = {}
        actuals: dict[str, float] = {}
        for name, value in metrics.items():
            if not name.startswith("slo."):
                continue
            label = name[4:]
            if label.endswith("_objective"):
                key = label[:-10]
                objectives[key] = value
            elif label.endswith("_actual"):
                key = label[:-7]
                actuals[key] = value

        reference = datetime.now(tz=UTC)
        for key, actual in actuals.items():
            objective = objectives.get(key)
            if objective is None:
                continue
            self._slo_tracker.record(key, objective=objective, actual=actual, timestamp=reference)



__all__ = ["NissaIngestor", "NissaIngestorConfig"]


def _normalise_enum_label(raw: str, *, prefix: str) -> str:
    value = raw
    if raw.upper().startswith(prefix):
        value = raw[len(prefix) :]
    return value.lower()
