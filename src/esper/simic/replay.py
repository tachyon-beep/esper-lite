"""Replay buffer scaffolding for Simic.

The buffer accumulates Tamiyo field reports for offline policy improvement as
specified in `docs/project/implementation_plan.md` (Slice 4) and
`docs/design/detailed_design/04-simic.md`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class FieldReportReplayBuffer:
    """Simple in-memory FIFO buffer for field reports."""

    capacity: int = 1024
    _buffer: deque[leyline_pb2.FieldReport] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_buffer", deque(maxlen=self.capacity))

    def add(self, report: leyline_pb2.FieldReport) -> None:
        self._buffer.append(report)

    def sample(self, count: int) -> Iterable[leyline_pb2.FieldReport]:
        """Yield at most `count` field reports."""

        returned = 0
        for report in list(self._buffer):
            if returned >= count:
                break
            returned += 1
            yield report

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


__all__ = ["FieldReportReplayBuffer"]
