"""FOSSILIZE_RATE_GUARD_TRIPPED telemetry contract (pre-A/B build WI-3).

Emitted exactly once, immediately before the G1 abort raises; the run's
shutdown `finally` (hub.close()) flushes it to disk, so the trip and its
context survive the abort (drl review MINOR-4).
"""

from esper.leyline import FossilizeRateGuardTrippedPayload
from esper.leyline.telemetry import TelemetryEventType


def test_event_type_exists():
    assert TelemetryEventType.FOSSILIZE_RATE_GUARD_TRIPPED is not None


def test_payload_round_trips_from_dict():
    payload = FossilizeRateGuardTrippedPayload(
        batch_idx=7,
        fossilize_count=14,
        trip_count_threshold=13,
        consecutive_batches=2,
        episodes_in_batch=12,
    )
    parsed = FossilizeRateGuardTrippedPayload.from_dict(
        {
            "batch_idx": 7,
            "fossilize_count": 14,
            "trip_count_threshold": 13,
            "consecutive_batches": 2,
            "episodes_in_batch": 12,
        }
    )
    assert parsed == payload
