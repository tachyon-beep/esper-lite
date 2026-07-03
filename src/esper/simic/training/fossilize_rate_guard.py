"""G1 fossilize-rate guard: the in-run runaway breaker for the ON arm.

Pre-A/B build WI-3 (gate criterion 3 / drl-review condition 3). Active only
at ``shapley_synergy_scale > 0`` — the OFF arm never constructs or consults
it, preserving bitwise identity. Division of labor (drl review MAJOR-2):
this guard catches only gross commit-spam RUNAWAY; k=3 concentration is
invisible to it by design, and correctness is adjudicated at scoring time by
the G3 efficiency floor + the Δcorr(reward, J) effect-size floor. Abort (not
credit-freeze) is deliberate: mutating reward semantics mid-run would turn
the arm into a mixture of two treatments and confound the A/B.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from esper.leyline import (
    FOSSILIZE_RATE_GUARD_CONSECUTIVE_BATCHES,
    FOSSILIZE_RATE_GUARD_TRIP_COUNT,
)

__all__ = ["FossilizeRateGuard"]


@dataclass
class FossilizeRateGuard:
    """Trips after N consecutive batches at/above the fossilize-count line."""

    trip_count: int = FOSSILIZE_RATE_GUARD_TRIP_COUNT
    consecutive_required: int = FOSSILIZE_RATE_GUARD_CONSECUTIVE_BATCHES
    _consecutive: int = field(default=0, init=False)

    def observe_batch(self, fossilize_count: int) -> bool:
        """Record one batch's fossilize count; True when the guard trips."""
        if fossilize_count >= self.trip_count:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.consecutive_required

    @property
    def consecutive(self) -> int:
        """Consecutive over-threshold batches so far (for the trip event)."""
        return self._consecutive
