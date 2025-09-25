"""Lightweight process-local metrics for Urabrask (prototype).

These counters are exported via Weatherlight telemetry under the
`urabrask.*` namespace when available.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_COUNTERS: dict[str, float] = {
    "wal.append_errors_total": 0.0,
    "integrity_failures": 0.0,
}


def inc_wal_append_errors(delta: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS["wal.append_errors_total"] = _COUNTERS.get(
            "wal.append_errors_total", 0.0
        ) + float(delta)


def inc_integrity_failures(delta: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS["integrity_failures"] = _COUNTERS.get("integrity_failures", 0.0) + float(delta)


def snapshot() -> dict[str, float]:
    with _LOCK:
        return dict(_COUNTERS)


__all__ = [
    "inc_wal_append_errors",
    "inc_integrity_failures",
    "snapshot",
]
