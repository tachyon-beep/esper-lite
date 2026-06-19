"""Tier-0 phase profiler for the vectorized PPO runtime.

Always-on, near-zero-overhead per-transaction-phase attribution. For each phase it
accumulates two clocks:

  - wall via ``time.perf_counter_ns`` (real elapsed)
  - Python CPU via ``time.thread_time_ns`` (dispatch-thread GIL-holding work)

At ``drain()`` (called once per epoch, after all CUDA streams are synchronized) it
emits a :class:`~esper.leyline.PhaseProfileReport` with per-phase ``wall_ms``,
``python_cpu_ms`` and ``python_cpu_ratio``.

Determinism contract (see docs/plans/concepts/2026-06-16-gil-throughput-profiler.md):
this tier touches no tensors, issues no CUDA call, and adds no host<->device sync.
The report is observation-only and flows one-way to telemetry. When disabled the
factory yields a :class:`NullProfiler` whose every call is a no-op, so a disabled run
is byte-identical to baseline.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator

from esper.leyline import PhaseProfileReport, PhaseTiming

_NS_PER_MS = 1_000_000


class PhaseProfiler:
    """Accumulates per-phase wall and Python-CPU time, drained once per epoch.

    Clocks are injected (defaulting to the real monotonic clocks) so the
    accumulation logic is exactly testable without sleeps or tolerances.
    """

    __slots__ = ("_wall", "_cpu", "_wall_ns", "_cpu_ns")

    def __init__(
        self,
        *,
        wall_clock: Callable[[], int] = time.perf_counter_ns,
        cpu_clock: Callable[[], int] = time.thread_time_ns,
    ) -> None:
        self._wall = wall_clock
        self._cpu = cpu_clock
        # Pre-existing accumulators; keys appear on first use (~6-8 per epoch).
        self._wall_ns: dict[str, int] = defaultdict(int)
        self._cpu_ns: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one transaction-phase span; repeated entries for ``name`` sum."""
        wall_start = self._wall()
        cpu_start = self._cpu()
        try:
            yield
        finally:
            self._wall_ns[name] += self._wall() - wall_start
            self._cpu_ns[name] += self._cpu() - cpu_start

    def drain(
        self, *, epoch: int | None = None, batch_idx: int | None = None
    ) -> PhaseProfileReport:
        """Build the epoch report from accumulated timings and reset for the next epoch."""
        phases: dict[str, PhaseTiming] = {}
        for name, wall_ns in self._wall_ns.items():
            wall_ms = wall_ns / _NS_PER_MS
            cpu_ms = self._cpu_ns[name] / _NS_PER_MS
            ratio = cpu_ms / wall_ms if wall_ms > 0.0 else 0.0
            phases[name] = PhaseTiming(
                wall_ms=wall_ms, python_cpu_ms=cpu_ms, python_cpu_ratio=ratio
            )
        self._wall_ns.clear()
        self._cpu_ns.clear()
        return PhaseProfileReport(phases=phases, epoch=epoch, batch_idx=batch_idx)


class NullProfiler:
    """No-op profiler used when the instrument is disabled (byte-identical runs)."""

    __slots__ = ()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        yield

    def drain(
        self, *, epoch: int | None = None, batch_idx: int | None = None
    ) -> None:
        return None


@contextmanager
def phase_profiler(
    *,
    enabled: bool = True,
    wall_clock: Callable[[], int] = time.perf_counter_ns,
    cpu_clock: Callable[[], int] = time.thread_time_ns,
) -> Iterator[PhaseProfiler | NullProfiler]:
    """Yield a phase-profiler handle (or a :class:`NullProfiler` when disabled).

    Entered once per run as a sibling of ``training_profiler``; the yielded handle is
    used for the whole run, its ``drain()`` called once per epoch.
    """
    if not enabled:
        yield NullProfiler()
        return
    yield PhaseProfiler(wall_clock=wall_clock, cpu_clock=cpu_clock)


__all__ = ["PhaseProfiler", "NullProfiler", "phase_profiler"]
