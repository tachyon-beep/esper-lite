import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

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

    value = registry.get_sample_value(
        "esper_telemetry_packets_total", {"source": "tamiyo"}
    )
    assert value == 1.0
    assert es.indexed and es.indexed[0][0] == "telemetry"


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
