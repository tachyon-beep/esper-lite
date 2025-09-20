import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from esper.core import (
    AdaptationCommand,
    AdaptationDirective,
    FieldReport,
    FieldReportOutcome,
    SystemStatePacket,
    TrainingPhase,
)
from esper.core.telemetry import TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.nissa import NissaIngestor, NissaIngestorConfig
from esper.nissa.server import create_app
from esper.oona import OonaClient, StreamConfig


class _ElasticsearchStub:
    def __init__(self) -> None:
        self.indexed: list[tuple[str, dict]] = []

    def index(self, index: str, document: dict) -> None:
        self.indexed.append((index, document))


def test_ingest_telemetry_records_metrics_and_indexes() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    packet = build_telemetry_packet(
        packet_id="pkt-1",
        source="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("tamiyo.validation_loss", 0.42)],
    )

    ingest.ingest_telemetry(packet)

    state = SystemStatePacket(
        run_id="run-1",
        epoch_index=0,
        phase=TrainingPhase.EPOCH,
    )
    ingest.ingest_state(state)

    report = FieldReport(
        run_id="run-1",
        command=AdaptationCommand(
            run_id="run-1",
            epoch_index=0,
            directive=AdaptationDirective.NO_OP,
        ),
        outcome=FieldReportOutcome.SUCCESS,
        metrics_delta={"loss_delta": -0.1},
    )
    ingest.ingest_field_report(report)

    value = registry.get_sample_value(
        "esper_telemetry_packets_total", {"source": "tamiyo"}
    )
    assert value == 1.0
    state_value = registry.get_sample_value(
        "esper_system_state_packets_total", {"phase": TrainingPhase.EPOCH.value}
    )
    assert state_value == 1.0
    report_value = registry.get_sample_value(
        "esper_field_reports_total", {"outcome": FieldReportOutcome.SUCCESS.value}
    )
    assert report_value == 1.0
    assert es.indexed and es.indexed[0][0] == "telemetry"


def test_ingest_simic_metrics_updates_counters() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    packet = leyline_pb2.TelemetryPacket(
        packet_id="simic-1",
        source_subsystem="simic",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    metric_reward = packet.metrics.add()
    metric_reward.name = "simic.training.reward"
    metric_reward.value = 1.25
    metric_iter = packet.metrics.add()
    metric_iter.name = "simic.training.iterations"
    metric_iter.value = 3

    ingest.ingest_telemetry(packet)

    reward_total = registry.get_sample_value("esper_simic_training_reward_total")
    iterations_total = registry.get_sample_value("esper_simic_training_iterations_total")
    assert reward_total == 1.25
    assert iterations_total == 3.0
    assert es.indexed[-1][0] == "simic_metrics"


@pytest.mark.asyncio
async def test_consume_from_oona_ingests_packets() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    redis = FakeRedis()
    oona_config = StreamConfig(
        normal_stream="oona.normal",
        emergency_stream="oona.emergency",
        telemetry_stream="oona.telemetry",
        group="nissa-test",
    )
    client = OonaClient("redis://localhost", config=oona_config, redis_client=redis)
    await client.ensure_consumer_group()
    packet = build_telemetry_packet(
        packet_id="pkt-1",
        source="tolaria",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("tolaria.loss", 0.5)],
    )
    await client.publish_telemetry(packet)
    await client.ensure_consumer_group()
    await ingest.consume_from_oona(client)
    await client.close()

    value = registry.get_sample_value(
        "esper_telemetry_packets_total", {"source": "tolaria"}
    )
    assert value == 1.0


def test_metrics_endpoint_serves_prometheus() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)
    packet = build_telemetry_packet(
        packet_id="pkt-1",
        source="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("tamiyo.validation_loss", 0.42)],
    )
    ingest.ingest_telemetry(packet)

    app = create_app(ingest)
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "esper_telemetry_packets_total" in response.text

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    summary = client.get("/metrics/summary")
    assert summary.status_code == 200


def test_training_latency_alert_triggers_after_consecutive_breaches() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    for value in (19.0, 20.0, 21.0):
        packet = build_telemetry_packet(
            packet_id=f"lat-{value}",
            source="tolaria",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=[TelemetryMetric("tolaria.training.latency_ms", value)],
        )
        ingest.ingest_telemetry(packet)

    assert "training_latency_high" in ingest.active_alerts
    last_event = ingest.alert_events[-1]
    assert last_event.routes == ("pagerduty", "slack")


def test_oona_queue_depth_alert_requires_two_samples() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    packet_low = build_telemetry_packet(
        packet_id="queue-low",
        source="oona",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("oona.queue.depth", 3500.0)],
    )
    ingest.ingest_telemetry(packet_low)
    assert "oona_queue_depth" not in ingest.active_alerts

    packet_high = build_telemetry_packet(
        packet_id="queue-high-1",
        source="oona",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("oona.queue.depth", 4500.0)],
    )
    ingest.ingest_telemetry(packet_high)
    assert "oona_queue_depth" not in ingest.active_alerts

    packet_high_second = build_telemetry_packet(
        packet_id="queue-high-2",
        source="oona",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        metrics=[TelemetryMetric("oona.queue.depth", 4600.0)],
    )
    ingest.ingest_telemetry(packet_high_second)
    assert "oona_queue_depth" in ingest.active_alerts
    event = ingest.active_alerts["oona_queue_depth"]
    assert event.routes == ("slack",)


def test_slo_summary_reports_burn_rate() -> None:
    registry = CollectorRegistry()
    es = _ElasticsearchStub()
    config = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
    )
    ingest = NissaIngestor(config, es_client=es, registry=registry)

    for actual, objective in ((120.0, 100.0), (90.0, 100.0), (130.0, 100.0)):
        metrics = [
            TelemetryMetric("slo.latency_actual", actual),
            TelemetryMetric("slo.latency_objective", objective),
        ]
        packet = build_telemetry_packet(
            packet_id=f"slo-{actual}",
            source="tolaria",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
        )
        ingest.ingest_telemetry(packet)

    summary = ingest.slo_summary()
    assert "latency" in summary
    status = summary["latency"]
    assert status.total == 3
    assert status.violations == 2
    assert status.burn_rate == pytest.approx(2 / 3, rel=1e-3)
