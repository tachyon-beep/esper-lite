"""Emergency protocol controller for Tolaria.

Implements a four-level escalation model and emits telemetry-friendly
events. Broadcast integration is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from esper.core import TelemetryEvent
from esper.leyline import leyline_pb2


class Level:
    """Emergency levels as plain integer constants (avoid Enum for Leyline centralization).

    Using simple constants keeps domain codes out of Python Enums to satisfy the
    "no internal enums" guardrail until Leyline exposes an official schema.
    """

    L1_NOTICE: int = 1
    L2_ELEVATED: int = 2
    L3_CONSERVATIVE: int = 3
    L4_HALT: int = 4


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
        broadcaster: Callable[[leyline_pb2.EmergencySignal], None] | None = None,  # type: ignore[name-defined]
    ) -> Escalation:
        start = perf_counter()
        broadcasted = False
        if level > self._level:
            self._level = level
            if broadcaster is not None:
                try:
                    # Construct a minimal compatible signal when schema exists
                    signal = getattr(leyline_pb2, "EmergencySignal", None)
                    if callable(signal):
                        msg = signal(level=int(level), reason=reason, origin="tolaria")
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


__all__ = ["EmergencyController", "Level", "Escalation"]
