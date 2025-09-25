"""SLO tracking helpers for Nissa."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(slots=True)
class SLOSample:
    metric: str
    objective: float
    actual: float
    timestamp: datetime

    @property
    def within_objective(self) -> bool:
        return self.actual <= self.objective


@dataclass(slots=True)
class SLOStatus:
    metric: str
    total: int
    violations: int
    burn_rate: float


@dataclass(slots=True)
class SLOConfig:
    window_hours: int = 24 * 30
    burn_alert_threshold: float = 0.4


class SLOTracker:
    """Maintains rolling SLO samples and computes burn rate."""

    def __init__(self, config: SLOConfig | None = None) -> None:
        self._config = config or SLOConfig()
        self._samples: dict[str, deque[SLOSample]] = {}

    @property
    def config(self) -> SLOConfig:
        return self._config

    def record(self, metric: str, *, objective: float, actual: float, timestamp: datetime | None = None) -> SLOStatus:
        ts = timestamp or datetime.now(tz=UTC)
        bucket = self._samples.setdefault(metric, deque())
        bucket.append(SLOSample(metric, objective, actual, ts))
        self._prune(metric, reference=ts)
        return self.status(metric)

    def status(self, metric: str) -> SLOStatus:
        samples = self._samples.get(metric, deque())
        total = len(samples)
        if not total:
            return SLOStatus(metric, 0, 0, 0.0)
        violations = sum(0 if sample.within_objective else 1 for sample in samples)
        burn = violations / total
        return SLOStatus(metric, total, violations, burn)

    def summary(self) -> dict[str, SLOStatus]:
        return {metric: self.status(metric) for metric in self._samples}

    def breached(self) -> dict[str, SLOStatus]:
        threshold = self._config.burn_alert_threshold
        return {metric: status for metric, status in self.summary().items() if status.burn_rate >= threshold}

    def _prune(self, metric: str, *, reference: datetime) -> None:
        window = timedelta(hours=self._config.window_hours)
        cutoff = reference - window
        bucket = self._samples.get(metric)
        if not bucket:
            return
        while bucket and bucket[0].timestamp < cutoff:
            bucket.popleft()


__all__ = ["SLOTracker", "SLOConfig", "SLOStatus", "SLOSample"]
