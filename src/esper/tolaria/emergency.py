"""Emergency protocol controller for Tolaria.

Implements a four-level escalation model and emits telemetry-friendly
events. Broadcast integration is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, monotonic
from typing import Callable
import struct

try:
    from multiprocessing import shared_memory  # type: ignore
except Exception:  # pragma: no cover - platform dependent
    shared_memory = None  # type: ignore

import threading

from esper.core import TelemetryEvent
from esper.leyline import leyline_pb2


class Level:
    """Emergency level constants sourced from Leyline."""

    L1_NOTICE: int = leyline_pb2.EmergencyLevel.EMERGENCY_LEVEL_NOTICE
    L2_ELEVATED: int = leyline_pb2.EmergencyLevel.EMERGENCY_LEVEL_ELEVATED
    L3_CONSERVATIVE: int = leyline_pb2.EmergencyLevel.EMERGENCY_LEVEL_CONSERVATIVE
    L4_HALT: int = leyline_pb2.EmergencyLevel.EMERGENCY_LEVEL_HALT


@dataclass(slots=True)
class Escalation:
    level: int
    reason: str
    broadcasted: bool
    latency_ms: float


class EmergencyController:
    def __init__(self, *, bypass_cap_per_min: int = 60) -> None:
        self._level = Level.L1_NOTICE
        self._bypass_cap = max(1, bypass_cap_per_min)
        self._bypass_count = 0

    @property
    def level(self) -> int:
        return self._level

    def escalate(
        self,
        level: int,
        *,
        reason: str,
        broadcaster: Callable[[leyline_pb2.EmergencySignal], None] | None = None,
    ) -> Escalation:
        start = perf_counter()
        broadcasted = False
        if level > self._level:
            self._level = level
            if broadcaster is not None:
                try:
                    msg = leyline_pb2.EmergencySignal(
                        version=1,
                        level=int(level),
                        reason=reason,
                        origin="tolaria",
                    )
                    broadcaster(msg)
                    broadcasted = True
                except Exception:
                    broadcasted = False
        return Escalation(level=int(level), reason=reason, broadcasted=broadcasted, latency_ms=(perf_counter() - start) * 1000.0)

    def telemetry_events(self) -> list[TelemetryEvent]:
        return [
            TelemetryEvent(
                description="tolaria.emergency.level",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={"level": str(int(self._level))},
            )
        ]


class LocalEmergencySignal:
    """Process-local emergency signal used when shared memory is unavailable."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._level = leyline_pb2.EmergencyLevel.EMERGENCY_LEVEL_UNSPECIFIED
        self._reason = ""
        self._timestamp_ms: int | None = None

    def trigger(self, level: int, reason: str, *, monotonic_ms: int | None = None) -> None:
        self._level = int(level)
        self._reason = reason
        self._timestamp_ms = monotonic_ms
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        self._event.clear()

    def read_level(self) -> int:
        return int(self._level)

    def read_reason(self) -> str:
        return self._reason

    def read_timestamp_ms(self) -> int | None:
        return self._timestamp_ms


class SharedEmergencySignal:
    """Cross-process emergency signal backed by shared memory."""

    _SIZE = 64  # bytes: flag, level, timestamp, reason length, reason bytes

    def __init__(self, name: str, create: bool = True) -> None:
        if shared_memory is None:
            raise RuntimeError("shared_memory unavailable")
        self._name = name
        self._created = create
        if create:
            try:
                self._shm = shared_memory.SharedMemory(name=name, create=True, size=self._SIZE)
            except FileExistsError:
                self._shm = shared_memory.SharedMemory(name=name, create=False)
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        if create:
            self.clear()

    @classmethod
    def create(cls, name: str) -> "SharedEmergencySignal":
        return cls(name, create=True)

    @classmethod
    def attach(cls, name: str) -> "SharedEmergencySignal":
        return cls(name, create=False)

    def trigger(self, level: int, reason: str, *, monotonic_ms: int | None = None) -> None:
        buf = self._shm.buf
        buf[0] = 1
        buf[1] = max(0, min(255, int(level)))
        ts = int(monotonic_ms if monotonic_ms is not None else int(monotonic() * 1000.0))
        buf[2:10] = struct.pack("<Q", ts)
        reason_bytes = reason.encode("utf-8", "ignore")[: (self._SIZE - 11)]
        buf[10] = len(reason_bytes)
        if reason_bytes:
            buf[11 : 11 + len(reason_bytes)] = reason_bytes

    def clear(self) -> None:
        buf = self._shm.buf
        buf[0] = 0
        buf[1] = 0
        buf[2:10] = b"\x00" * 8
        buf[10] = 0
        buf[11:] = b"\x00" * (self._SIZE - 11)

    def is_set(self) -> bool:
        return int(self._shm.buf[0]) == 1

    def read_level(self) -> int:
        return int(self._shm.buf[1])

    def read_timestamp_ms(self) -> int | None:
        try:
            ts = struct.unpack("<Q", bytes(self._shm.buf[2:10]))[0]
            return int(ts) if ts > 0 else None
        except Exception:
            return None

    def read_reason(self) -> str:
        try:
            length = int(self._shm.buf[10])
            if length <= 0:
                return ""
            data = bytes(self._shm.buf[11 : 11 + length])
            return data.decode("utf-8", "ignore")
        except Exception:
            return ""

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        if self._created:
            try:
                self._shm.unlink()
            except Exception:
                pass


__all__ = [
    "EmergencyController",
    "Level",
    "Escalation",
    "SharedEmergencySignal",
    "LocalEmergencySignal",
]
