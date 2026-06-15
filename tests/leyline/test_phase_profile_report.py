"""Contract tests for the Tier-0 phase-profiler leyline types.

PhaseTiming / PhaseProfileReport are the observation-only payload the GIL/throughput
profiler emits per epoch (see docs/plans/concepts/2026-06-16-gil-throughput-profiler.md).
"""
from __future__ import annotations

from esper.leyline import PhaseProfileReport, PhaseTiming


def test_phase_timing_is_slots_dataclass() -> None:
    t = PhaseTiming(wall_ms=2.0, python_cpu_ms=1.0, python_cpu_ratio=0.5)
    assert not hasattr(t, "__dict__"), "slots=True dataclass must not carry __dict__"
    assert t.wall_ms == 2.0
    assert t.python_cpu_ms == 1.0
    assert t.python_cpu_ratio == 0.5


def test_phase_profile_report_holds_per_phase_timings() -> None:
    report = PhaseProfileReport(
        phases={"reward": PhaseTiming(wall_ms=2.0, python_cpu_ms=2.0, python_cpu_ratio=1.0)},
        epoch=7,
        batch_idx=3,
    )
    assert report.epoch == 7
    assert report.batch_idx == 3
    assert report.phases["reward"].python_cpu_ratio == 1.0


def test_phase_profile_report_epoch_batch_default_none() -> None:
    """epoch/batch_idx are stamped by the caller at drain; default to None."""
    report = PhaseProfileReport(phases={})
    assert report.epoch is None
    assert report.batch_idx is None


def test_total_wall_ms_sums_phases() -> None:
    report = PhaseProfileReport(
        phases={
            "rollout": PhaseTiming(wall_ms=3.0, python_cpu_ms=1.0, python_cpu_ratio=0.33),
            "reward": PhaseTiming(wall_ms=2.0, python_cpu_ms=2.0, python_cpu_ratio=1.0),
        }
    )
    assert report.total_wall_ms() == 5.0
