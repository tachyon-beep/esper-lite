"""Karn Serialization - Shared event serialization for output backends.

Provides a single source of truth for converting TelemetryEvent-like
objects to JSON strings. Used by WebSocketOutput and DashboardServer.

Usage:
    from esper.karn.serialization import serialize_event

    json_str = serialize_event(event)
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike


def _payload_to_dict(obj: Any) -> Any:
    """Convert a typed payload to a JSON-serializable dict.

    Handles:
    - Dataclass instances → dict via dataclasses.asdict()
    - Enum values → their .name string
    - None, primitives → pass through
    - Already-dict → pass through
    """
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # It's a dataclass instance, convert to dict
        # Use a custom dict_factory to handle nested enums
        return dataclasses.asdict(
            obj,
            dict_factory=lambda items: {
                k: v.name if isinstance(v, Enum) else v for k, v in items
            },
        )
    if isinstance(obj, dict):
        return obj
    # For any other type (shouldn't happen with typed payloads), pass through
    return obj


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
        "data": _payload_to_dict(event.data),
        "epoch": event.epoch,
        "seed_id": event.seed_id,
        "slot_id": event.slot_id,
        "severity": event.severity,
        "message": event.message,
    }

    return json.dumps(data, default=str)
