"""P2-FRAGMETRIC telemetry wiring: AllocatorStatsPayload round-trip + emit + serialization.

Telemetry-sacred: the payload must round-trip without silent field drops, the emitter must
produce an ALLOCATOR_STATS event, and the serializer must write the UPPERCASE enum .name
(so the Karn raw_events query 'event_type=ALLOCATOR_STATS' matches).
"""

from esper.leyline import AllocatorStatsPayload, TelemetryEvent, TelemetryEventType
from esper.nissa.output import _telemetry_event_to_dict
from esper.simic.telemetry.emitters import VectorizedEmitter


def _payload() -> AllocatorStatsPayload:
    return AllocatorStatsPayload(
        batch_idx=7,
        device="cuda:1",
        allocated_bytes=1_000_000,
        reserved_bytes=11_000_000,
        fragmentation_bytes=10_000_000,
        num_alloc_retries=3,
        num_ooms=0,
    )


def test_payload_round_trip_no_field_drop():
    p = _payload()
    d = p.to_dict()
    # Every field present, no silent drop.
    assert set(d) == {
        "batch_idx", "device", "allocated_bytes", "reserved_bytes",
        "fragmentation_bytes", "num_alloc_retries", "num_ooms",
    }
    assert AllocatorStatsPayload.from_dict(d) == p


def test_event_serializes_uppercase_event_type():
    event = TelemetryEvent(event_type=TelemetryEventType.ALLOCATOR_STATS, data=_payload())
    out = _telemetry_event_to_dict(event)
    # The Karn raw_events query keys on this exact (UPPERCASE) string.
    assert out["event_type"] == "ALLOCATOR_STATS"
    assert out["data"]["fragmentation_bytes"] == 10_000_000
    assert out["data"]["num_alloc_retries"] == 3


def test_emitter_emits_allocator_stats():
    captured: list[TelemetryEvent] = []

    class _Hub:
        def emit(self, event: TelemetryEvent) -> None:
            captured.append(event)

    emitter = VectorizedEmitter(env_id=0, device="cuda:0", group_id="A", hub=_Hub())
    emitter.on_allocator_stats(
        batch_idx=2,
        device="cuda:1",  # MEASURED device, not the emitter's env device
        allocated_bytes=5,
        reserved_bytes=9,
        fragmentation_bytes=4,
        num_alloc_retries=1,
        num_ooms=0,
    )
    assert len(captured) == 1
    ev = captured[0]
    assert ev.event_type is TelemetryEventType.ALLOCATOR_STATS
    # Payload carries the measured device, NOT the emitter's env device.
    assert ev.data.device == "cuda:1"
    assert ev.data.batch_idx == 2
    assert ev.data.fragmentation_bytes == 4


def test_emitter_no_hub_is_noop():
    emitter = VectorizedEmitter(env_id=0, device="cuda:0", group_id="A", hub=None)
    # Must not raise when hub is absent.
    emitter.on_allocator_stats(
        batch_idx=0, device="cpu", allocated_bytes=0, reserved_bytes=0,
        fragmentation_bytes=0, num_alloc_retries=0, num_ooms=0,
    )
