from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2


def test_leyline_enums_exist_and_are_usable() -> None:
    # Check several enums are present and usable
    assert leyline_pb2.SEED_STAGE_TRAINING > 0
    assert leyline_pb2.TELEMETRY_LEVEL_INFO >= 0
    assert leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS >= 0


def test_build_telemetry_packet_uses_leyline_contracts() -> None:
    metrics = [
        TelemetryMetric("demo.metric", 1.23, unit="count"),
    ]
    events = [TelemetryEvent("demo_event", level=leyline_pb2.TELEMETRY_LEVEL_INFO)]
    packet = build_telemetry_packet(
        packet_id="pkt-1",
        source="test",
        level=leyline_pb2.TELEMETRY_LEVEL_INFO,
        metrics=metrics,
        events=events,
        health_status=leyline_pb2.HEALTH_STATUS_HEALTHY,
        health_summary="ok",
    )
    assert packet.source_subsystem == "test"
    assert len(packet.metrics) == 1
    assert packet.events[0].description == "demo_event"
