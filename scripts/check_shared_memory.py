#!/usr/bin/env python3
"""Smoke-test Tolaria shared-memory primitives."""

from esper.tolaria.rollback import SharedDeadlineSignal
from esper.tolaria.emergency import SharedEmergencySignal


def _check_deadline(name: str) -> None:
    sig = SharedDeadlineSignal.create(name)
    try:
        sig.trigger()
        assert sig.is_set(), "SharedDeadlineSignal trigger failed"
        sig.clear()
    finally:
        sig.close()
        sig.unlink()


def _check_emergency(name: str) -> None:
    sig = SharedEmergencySignal.create(name)
    try:
        sig.trigger(level=2, reason="diagnostic")
        assert sig.is_set(), "SharedEmergencySignal trigger failed"
        sig.clear()
    finally:
        sig.close()
        sig.unlink()


def main() -> int:
    _check_deadline("tolaria-diagnostic-deadline")
    _check_emergency("tolaria-diagnostic-emergency")
    print("Shared memory diagnostics OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
