"""Replay buffer scaffolding for Simic.

The buffer accumulates Tamiyo field reports for offline policy improvement as
specified in `docs/project/implementation_plan.md` (Slice 4) and
`docs/design/detailed_design/04-simic.md`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.leyline import leyline_pb2

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


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

    async def ingest_from_oona(
        self,
        client: OonaClient,
        *,
        stream: str | None = None,
        count: int = 50,
        block_ms: int = 1000,
    ) -> None:
        """Consume field reports from Oona and load them into the buffer."""

        async def handler(message: OonaMessage) -> None:
            report = leyline_pb2.FieldReport()
            report.ParseFromString(message.payload)
            self.add(report)

        await client.consume(
            handler,
            stream=stream or client.normal_stream,
            count=count,
            block_ms=block_ms,
        )


__all__ = ["FieldReportReplayBuffer"]
