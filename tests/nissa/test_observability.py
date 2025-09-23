from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from esper.leyline import leyline_pb2
from esper.nissa.observability import NissaIngestor, NissaIngestorConfig
from esper.nissa.server import create_app


def _fake_es():
    return SimpleNamespace(index=lambda **kwargs: None)


def _packet_with_metric(name: str, value: float) -> leyline_pb2.TelemetryPacket:
    pkt = leyline_pb2.TelemetryPacket(
        packet_id="nissa-test",
        source_subsystem="tamiyo",
        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    )
    m = pkt.metrics.add()
    m.name = name
    m.value = value
    return pkt


def test_ingest_coverage_exposes_metrics() -> None:
    cfg = NissaIngestorConfig(prometheus_gateway="http://localhost:9091", elasticsearch_url="http://localhost:9200")
    ing = NissaIngestor(cfg, es_client=_fake_es())
    pkt = _packet_with_metric("tamiyo.gnn.feature_coverage", 0.55)
    ing.ingest_telemetry(pkt)
    data = generate_latest(ing.registry).decode("utf-8")
    assert "tamiyo_gnn_feature_coverage" in data
    assert "0.55" in data


def test_coverage_alert_fires_after_3_packets() -> None:
    cfg = NissaIngestorConfig(prometheus_gateway="http://localhost:9091", elasticsearch_url="http://localhost:9200")
    ing = NissaIngestor(cfg, es_client=_fake_es())
    for _ in range(3):
        pkt = _packet_with_metric("tamiyo.gnn.feature_coverage", 0.65)  # < 0.7 threshold
        ing.ingest_telemetry(pkt)
    alerts = ing.active_alerts
    assert "tamiyo_coverage_low" in alerts


def test_bsds_elevated_risk_alert() -> None:
    cfg = NissaIngestorConfig(prometheus_gateway="http://localhost:9091", elasticsearch_url="http://localhost:9200")
    ing = NissaIngestor(cfg, es_client=_fake_es())
    pkt = _packet_with_metric("tamiyo.blueprint.risk", 0.85)
    ing.ingest_telemetry(pkt)
    alerts = ing.active_alerts
    assert "tamiyo_bsds_elevated_risk" in alerts


def test_metrics_endpoint_serves_prometheus() -> None:
    cfg = NissaIngestorConfig(prometheus_gateway="http://localhost:9091", elasticsearch_url="http://localhost:9200")
    ing = NissaIngestor(cfg, es_client=_fake_es())
    pkt = _packet_with_metric("tamiyo.gnn.feature_coverage", 0.75)
    ing.ingest_telemetry(pkt)
    app = create_app(ing)
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "tamiyo_gnn_feature_coverage" in resp.text


def test_config_overrides_thresholds_and_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure stricter thresholds and custom whitelist
    cfg = NissaIngestorConfig(
        prometheus_gateway="http://localhost:9091",
        elasticsearch_url="http://localhost:9200",
        alerts_enabled=True,
        coverage_alert_threshold=0.9,
        coverage_alert_consecutive=2,
        bsds_elevated_risk_threshold=0.85,
        coverage_feature_keys=("custom.feature",),
    )
    ing = NissaIngestor(cfg, es_client=_fake_es())
    # Per-type coverage honored via whitelist
    pkt = _packet_with_metric("tamiyo.gnn.coverage.custom.feature", 0.4)
    ing.ingest_telemetry(pkt)
    data = generate_latest(ing.registry).decode("utf-8")
    assert 'tamiyo_gnn_feature_coverage_by_type{feature="custom.feature"}' in data
    # Elevated risk threshold used
    pkt2 = _packet_with_metric("tamiyo.blueprint.risk", 0.86)
    ing.ingest_telemetry(pkt2)
    alerts = ing.active_alerts
    assert "tamiyo_bsds_elevated_risk" in alerts
    # Coverage alert threshold stricter: 2 packets < 0.9 should fire
    for _ in range(2):
        ing.ingest_telemetry(_packet_with_metric("tamiyo.gnn.feature_coverage", 0.85))
    assert "tamiyo_coverage_low" in ing.active_alerts
