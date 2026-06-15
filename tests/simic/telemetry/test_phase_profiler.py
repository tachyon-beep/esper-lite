"""Tier-0 phase profiler: per-phase wall-clock + Python-CPU attribution.

The profiler accumulates two clocks per transaction phase:
  - wall  via perf_counter_ns  (real elapsed)
  - cpu   via thread_time_ns   (Python CPU = quantity (e), the dispatch-thread hold)
and at drain() reports per-phase wall_ms / python_cpu_ms / python_cpu_ratio.

Clocks are injected so timing is exact and non-flaky (no sleeps, no tolerances).
See docs/plans/concepts/2026-06-16-gil-throughput-profiler.md (Tier 0).
"""
from __future__ import annotations

from esper.leyline import PhaseProfileReport
from esper.simic.telemetry.phase_profiler import (
    NullProfiler,
    PhaseProfiler,
    phase_profiler,
)

_MS = 1_000_000  # ns per ms


class _FakeClock:
    """Returns scripted ns values on each call (enter, exit, enter, exit, ...)."""

    def __init__(self, values: list[int]) -> None:
        self._values = values
        self._i = 0

    def __call__(self) -> int:
        v = self._values[self._i]
        self._i += 1
        return v


def test_phase_records_wall_and_cpu_per_name() -> None:
    wall = _FakeClock([0, 2 * _MS])  # 2.0 ms elapsed
    cpu = _FakeClock([0, 1 * _MS])  # 1.0 ms Python CPU
    prof = PhaseProfiler(wall_clock=wall, cpu_clock=cpu)

    with prof.phase("reward"):
        pass

    report = prof.drain()
    assert isinstance(report, PhaseProfileReport)
    timing = report.phases["reward"]
    assert timing.wall_ms == 2.0
    assert timing.python_cpu_ms == 1.0
    assert timing.python_cpu_ratio == 0.5


def test_repeated_phase_accumulates() -> None:
    # reward entered twice: 2.0ms then 1.0ms wall -> 3.0ms total
    wall = _FakeClock([0, 2 * _MS, 5 * _MS, 6 * _MS])
    cpu = _FakeClock([0, 2 * _MS, 5 * _MS, 6 * _MS])
    prof = PhaseProfiler(wall_clock=wall, cpu_clock=cpu)

    with prof.phase("reward"):
        pass
    with prof.phase("reward"):
        pass

    report = prof.drain()
    assert report.phases["reward"].wall_ms == 3.0


def test_drain_resets_accumulators_each_epoch() -> None:
    wall = _FakeClock([0, 2 * _MS])
    cpu = _FakeClock([0, 1 * _MS])
    prof = PhaseProfiler(wall_clock=wall, cpu_clock=cpu)

    with prof.phase("reward"):
        pass
    prof.drain()

    # Second epoch with no phases entered -> empty report, not stale carry-over.
    report2 = prof.drain()
    assert report2.phases == {}


def test_drain_stamps_epoch_and_batch() -> None:
    prof = PhaseProfiler()
    report = prof.drain(epoch=9, batch_idx=4)
    assert report.epoch == 9
    assert report.batch_idx == 4


def test_zero_wall_phase_has_zero_ratio_not_division_error() -> None:
    wall = _FakeClock([100, 100])  # zero elapsed
    cpu = _FakeClock([100, 100])
    prof = PhaseProfiler(wall_clock=wall, cpu_clock=cpu)
    with prof.phase("rollback"):
        pass
    report = prof.drain()
    assert report.phases["rollback"].python_cpu_ratio == 0.0


def test_null_profiler_phase_is_noop_and_drain_returns_none() -> None:
    prof = NullProfiler()
    with prof.phase("reward"):  # must work as a context manager
        pass
    assert prof.drain() is None


def test_factory_disabled_yields_null_profiler() -> None:
    with phase_profiler(enabled=False) as prof:
        assert isinstance(prof, NullProfiler)


def test_factory_enabled_yields_real_profiler() -> None:
    with phase_profiler(enabled=True) as prof:
        assert isinstance(prof, PhaseProfiler)
