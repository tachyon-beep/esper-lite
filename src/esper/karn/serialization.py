"""Karn Serialization - Shared event serialization for output backends.

Provides a single source of truth for converting TelemetryEvent-like
objects to JSON strings. Used by WebSocketOutput and DashboardServer.

Usage:
    from esper.karn.serialization import serialize_event

    json_str = serialize_event(event)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike


def serialize_event(event: "TelemetryEventLike") -> str:
    """Serialize a TelemetryEventLike object to JSON string.

    Handles:
    - Enum event_type → string name
    - datetime timestamp → ISO format string
    - All standard TelemetryEventLike protocol fields

    Args:
        event: Any object implementing TelemetryEventLike protocol

    Returns:
        JSON string representation of the event
    """
    # Extract event_type (handle both enum and string)
    # hasattr AUTHORIZED by John on 2025-12-17 15:00:00 UTC
    # Justification: Serialization - handle both enum and string event_type values
    event_type = event.event_type
    if hasattr(event_type, "name"):
        event_type = event_type.name

    # Extract timestamp (handle datetime objects)
    # hasattr AUTHORIZED by John on 2025-12-17 15:00:00 UTC
    # Justification: Serialization - safely handle datetime objects
    timestamp = event.timestamp
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()

    data = {
        "event_type": event_type,
        "timestamp": timestamp,
        "data": event.data,
        "epoch": event.epoch,
        "seed_id": event.seed_id,
        "slot_id": event.slot_id,
        "severity": event.severity,
        "message": event.message,
    }

    return json.dumps(data, default=str)
