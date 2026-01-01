"""Karn Serialization - Shared event serialization for output backends.

Provides a single source of truth for converting TelemetryEvent-like
objects to JSON strings. Used by WebSocketOutput and OverwatchBackend.

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


def _convert_value(value: Any) -> Any:
    """Convert a single value to JSON-serializable form.

    Handles:
    - Dataclasses with to_dict() → call to_dict() (includes @property fields)
    - Other dataclasses → recursive _convert_value on fields
    - Enum values → their .name string
    - Lists/tuples → recursive conversion
    - None, primitives → pass through
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Prefer to_dict() if available (includes @property fields like shaped_reward_ratio)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        # Fallback: convert fields recursively
        return {
            field.name: _convert_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_convert_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    return value


def _payload_to_dict(obj: Any) -> Any:
    """Convert a typed payload to a JSON-serializable dict.

    Handles:
    - Dataclass instances with to_dict() → call to_dict() (includes @property fields)
    - Other dataclass instances → recursive field conversion
    - Enum values → their .name string
    - None, primitives → pass through
    - Already-dict → pass through
    """
    return _convert_value(obj)


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
    timestamp_raw = event.timestamp
    if hasattr(timestamp_raw, "isoformat"):
        timestamp: str = timestamp_raw.isoformat()
    else:
        timestamp = str(timestamp_raw)

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
