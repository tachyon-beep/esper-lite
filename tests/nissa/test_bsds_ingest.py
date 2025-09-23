from __future__ import annotations

from prometheus_client import CollectorRegistry

from esper.leyline import leyline_pb2
from esper.nissa import NissaIngestor, NissaIngestorConfig


class _ElasticsearchStub:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, dict]] = []

    def index(self, index: str, document: dict) -> None:
        self.indexed.append((index, document))


def _ingestor() -> NissaIngestor:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    return NissaIngestor(config, es_client=es, registry=registry)


def test_bsds_critical_alert_and_counter() -> None:
    ingest = _ingestor()

    pkt = leyline_pb2.TelemetryPacket(
        packet_id="bsds-critical",
        source_subsystem="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    ev = pkt.events.add()
    ev.description = "bsds_hazard_critical"
    ev.attributes["provenance"] = "URABRASK"

    ingest.ingest_telemetry(pkt)

    assert "tamiyo_bsds_critical" in ingest.active_alerts
    last_event = ingest.alert_events[-1]
    assert last_event.routes == ("pagerduty",)
    # Counter increments with provenance label
    value = ingest.registry.get_sample_value(
        "esper_tamiyo_bsds_hazard_critical_total", {"provenance": "urabrask"}
    )
    assert value == 1.0


def test_bsds_high_alert_after_three_packets() -> None:
    ingest = _ingestor()

    for i in range(3):
        pkt = leyline_pb2.TelemetryPacket(
            packet_id=f"bsds-high-{i}",
            source_subsystem="tamiyo",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        )
        ev = pkt.events.add()
        ev.description = "bsds_hazard_high"
        ingest.ingest_telemetry(pkt)

    assert "tamiyo_bsds_high" in ingest.active_alerts
    event = ingest.active_alerts["tamiyo_bsds_high"]
    assert event.routes == ("slack",)

