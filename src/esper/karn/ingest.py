"""Karn ingest helpers.

Karn ingests telemetry from multiple sources:
- Live events (`TelemetryEvent.data`)
- JSONL exports (`TelemetryStore.export_jsonl`)

These helpers coerce untyped payloads into the shapes Karn expects, while
emitting warnings when data is malformed. They are intentionally conservative:
they do not invent missing fields or accept legacy aliases.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import logging

from esper.leyline import SeedStage

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def coerce_int(
    value: Any,
    *,
    field: str,
    default: int = 0,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        _logger.warning("Invalid %s=%r (bool); using default=%r", field, value, default)
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _logger.warning("Invalid %s=%r; using default=%r", field, value, default)
        return default
    if minimum is not None and parsed < minimum:
        _logger.warning("Out-of-range %s=%r (<%r); using default=%r", field, parsed, minimum, default)
        return default
    if maximum is not None and parsed > maximum:
        _logger.warning("Out-of-range %s=%r (>%r); using default=%r", field, parsed, maximum, default)
        return default
    return parsed


def coerce_float(
    value: Any,
    *,
    field: str,
    default: float = 0.0,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        _logger.warning("Invalid %s=%r (bool); using default=%r", field, value, default)
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _logger.warning("Invalid %s=%r; using default=%r", field, value, default)
        return default
    if parsed != parsed:
        _logger.warning("Invalid %s=%r (NaN); using default=%r", field, value, default)
        return default
    if parsed in (float("inf"), float("-inf")):
        _logger.warning("Invalid %s=%r (inf); using default=%r", field, value, default)
        return default
    return parsed


def coerce_str(
    value: Any,
    *,
    field: str,
    default: str = "",
) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    _logger.warning("Invalid %s=%r; using default=%r", field, value, default)
    return default


def coerce_bool_or_none(
    value: Any,
    *,
    field: str,
) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        _logger.warning("Coercing %s=%r (int) to bool", field, value)
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "t", "yes", "y", "1"):
            _logger.warning("Coercing %s=%r (str) to bool", field, value)
            return True
        if lowered in ("false", "f", "no", "n", "0"):
            _logger.warning("Coercing %s=%r (str) to bool", field, value)
            return False
    _logger.warning("Invalid %s=%r (expected bool); using None", field, value)
    return None


def coerce_float_or_none(
    value: Any,
    *,
    field: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _logger.warning("Invalid %s=%r (bool); using None", field, value)
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _logger.warning("Invalid %s=%r; using None", field, value)
        return None
    if parsed != parsed:
        _logger.warning("Invalid %s=%r (NaN); using None", field, value)
        return None
    if parsed in (float("inf"), float("-inf")):
        _logger.warning("Invalid %s=%r (inf); using None", field, value)
        return None
    return parsed


def coerce_int_or_none(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _logger.warning("Invalid %s=%r (bool); using None", field, value)
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _logger.warning("Invalid %s=%r; using None", field, value)
        return None
    if minimum is not None and parsed < minimum:
        _logger.warning("Out-of-range %s=%r (<%r); using None", field, parsed, minimum)
        return None
    if maximum is not None and parsed > maximum:
        _logger.warning("Out-of-range %s=%r (>%r); using None", field, parsed, maximum)
        return None
    return parsed


def coerce_str_or_none(
    value: Any,
    *,
    field: str,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    _logger.warning("Invalid %s=%r (expected str); using None", field, value)
    return None


def coerce_float_dict(
    value: Any,
    *,
    field: str,
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        _logger.warning("Invalid %s=%r (expected dict); using empty dict", field, value)
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            _logger.warning("Invalid %s key=%r (expected str); skipping", field, key)
            continue
        coerced = coerce_float(raw, field=f"{field}.{key}", default=0.0)
        result[key] = coerced
    return result


def coerce_datetime(
    value: Any,
    *,
    field: str,
    default: datetime | None = None,
) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            _logger.warning("Invalid %s=%r (expected ISO datetime); using default=%r", field, value, default)
            return default
    _logger.warning("Invalid %s=%r (expected datetime/str); using default=%r", field, value, default)
    return default


def coerce_path(
    value: Any,
    *,
    field: str,
) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    _logger.warning("Invalid %s=%r (expected path str); using None", field, value)
    return None


def coerce_seed_stage(
    value: Any,
    *,
    field: str,
    default: SeedStage = SeedStage.DORMANT,
) -> SeedStage:
    if value is None:
        return default
    if isinstance(value, SeedStage):
        return value
    if isinstance(value, bool):
        _logger.warning("Invalid %s=%r (bool); using default=%r", field, value, default)
        return default
    if isinstance(value, int):
        try:
            return SeedStage(value)
        except ValueError:
            _logger.warning("Invalid %s=%r (unknown stage int); using default=%r", field, value, default)
            return default
    if isinstance(value, str):
        try:
            return SeedStage[value]
        except KeyError:
            try:
                return SeedStage(int(value))
            except (TypeError, ValueError):
                _logger.warning("Invalid %s=%r (unknown stage); using default=%r", field, value, default)
                return default
    _logger.warning("Invalid %s=%r (unexpected type); using default=%r", field, value, default)
    return default


def filter_dataclass_kwargs(cls: type[T], raw: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Filter a dict down to fields accepted by dataclass `cls`.

    Unknown keys are ignored but logged to help detect schema drift.
    """
    from dataclasses import is_dataclass
    if not is_dataclass(cls):
        _logger.warning("filter_dataclass_kwargs called on non-dataclass %s", cls.__name__)
        return raw
    allowed = {field.name for field in fields(cls)}
    unknown = [key for key in raw.keys() if key not in allowed]
    if unknown:
        _logger.warning("Ignoring unknown %s fields for %s: %s", context, cls.__name__, sorted(unknown))
    return {key: raw[key] for key in raw.keys() if key in allowed}
