"""Persistence helpers for Tamiyo field reports.

Implements the WAL-backed storage and retention policy described in
`docs/design/detailed_design/old/03-tamiyo.md` and the backlog item
`docs/project/backlog.md` TKT-401.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from esper.leyline import leyline_pb2


_LENGTH_STRUCT = struct.Struct("<I")


@dataclass(slots=True)
class FieldReportStoreConfig:
    """Configuration for the Tamiyo field report store."""

    path: Path
    retention: timedelta = timedelta(hours=24)


class FieldReportStore:
    """Durable append-only storage for Tamiyo field reports."""

    def __init__(self, config: FieldReportStoreConfig) -> None:
        self._config = config
        self._path = config.path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._reports: list[leyline_pb2.FieldReport] = list(self._load_from_disk())
        self.enforce_retention()

    @property
    def retention(self) -> timedelta:
        return self._config.retention

    @property
    def path(self) -> Path:
        return self._path

    def reports(self) -> list[leyline_pb2.FieldReport]:
        """Return a snapshot of the stored field reports."""

        return [report for report in self._reports]

    def append(self, report: leyline_pb2.FieldReport) -> None:
        """Append a field report to the WAL and refresh retention state."""

        payload = report.SerializeToString()
        with self._path.open("ab") as handle:
            handle.write(_LENGTH_STRUCT.pack(len(payload)))
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._reports.append(report)
        self.enforce_retention()

    def enforce_retention(self, *, now: datetime | None = None) -> None:
        """Drop reports that fall outside the retention window."""

        if not self._reports:
            return
        current_time = now or datetime.now(tz=UTC)
        cutoff = current_time - self._config.retention
        retained: list[leyline_pb2.FieldReport] = []
        for report in self._reports:
            if report.HasField("issued_at"):
                issued = report.issued_at.ToDatetime().replace(tzinfo=UTC)
                if issued.timestamp() == 0:
                    issued = current_time
            else:
                issued = current_time
            if issued >= cutoff:
                retained.append(report)
        if len(retained) == len(self._reports):
            return
        self._rewrite(retained)
        self._reports = retained

    def _rewrite(self, reports: Iterable[leyline_pb2.FieldReport]) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            for report in reports:
                payload = report.SerializeToString()
                handle.write(_LENGTH_STRUCT.pack(len(payload)))
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self._path)

    def _load_from_disk(self) -> Iterable[leyline_pb2.FieldReport]:
        if not self._path.exists():
            return []
        reports: list[leyline_pb2.FieldReport] = []
        with self._path.open("rb") as handle:
            while True:
                length_bytes = handle.read(_LENGTH_STRUCT.size)
                if not length_bytes:
                    break
                if len(length_bytes) != _LENGTH_STRUCT.size:
                    break
                (length,) = _LENGTH_STRUCT.unpack(length_bytes)
                payload = handle.read(length)
                if len(payload) != length:
                    break
                report = leyline_pb2.FieldReport()
                try:
                    report.ParseFromString(payload)
                except Exception:  # pragma: no cover - defensive guard
                    break
                reports.append(report)
        return reports


__all__ = ["FieldReportStore", "FieldReportStoreConfig"]
