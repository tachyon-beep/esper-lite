"""Canonical counterfactual-contribution state (Obs V4).

ONE typed source for a seed's counterfactual (leave-one-out) contribution and its
measurement *status*, consumed by the observation encoder and (future L2) the
settlement path. It replaces three independent ``None -> 0.0`` coercions that made a
*structurally unmeasured* counterfactual (a fossilized seed excluded from the ablation,
or a seed at birth before its first measurement) read as a confident **zero**.

Round-13 representation correction (both primes):
- ``NEVER_MEASURED`` must read as the UNKNOWN sentinel, NOT ``0.0`` (the original bug).
- A ``STALE``/``FROZEN_AT_FOSSILIZE`` value must *shrink toward the UNKNOWN sentinel as
  staleness rises* (interval-widening "we don't know") — NOT held at full magnitude
  forever (the mirror bug: a 70-epoch-old frozen value the net can read while ignoring
  the freshness), and NOT collapsed to ``0.0``.
- A true measured ``0.0`` (``FRESH`` with value 0.0) stays distinguishable from absence.

The four states are only jointly self-describing when paired with an explicit
observed/frozen/age mask in the observation (a bare ``-1`` sentinel in the value dim is
ambiguous with a true strong-negative that also clips to ``-1``). See
``esper.tamiyo.policy.features._encode_contribution_v4``.

Lifecycle identity: ``seed_generation_id`` increments on germinate/slot-reuse so a new
occupant of a slot id never inherits the previous occupant's cached ``last_valid`` (the
carry-forward artifact that caused the offline analysis over-reads — kept OUT of the live
policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CounterfactualStatus(Enum):
    """Measurement status of a seed's counterfactual (LOO) contribution."""

    NEVER_MEASURED = "never_measured"
    """Birth / pre-ablation: no counterfactual has ever been measured. UNKNOWN sentinel."""

    FRESH = "fresh"
    """Measured this epoch (includes a genuine measured ``0.0``)."""

    STALE = "stale"
    """Last valid value held; measurement is ageing and uncertainty is rising."""

    FROZEN_AT_FOSSILIZE = "frozen_at_fossilize"
    """Captured at the commit instant; the seed is excluded from the ablation thereafter."""


@dataclass(slots=True)
class ContributionState:
    """Canonical per-(env, slot) counterfactual-contribution state.

    Attributes:
        seed_generation_id: Monotonic lifecycle identity for the slot's current occupant.
            Increments on germinate/slot-reuse; the reset clears the cached value so a new
            seed never inherits a prior occupant's ``last_valid``.
        counterfactual_status: One of :class:`CounterfactualStatus`.
        last_valid_counterfactual_contribution: The most recent *measured* LOO contribution
            (percentage-point accuracy delta), held through STALE/FROZEN. ``None`` iff the
            status is ``NEVER_MEASURED``.
        epochs_since_counterfactual: Epochs since the value was last measured. ``0`` when
            ``FRESH`` (and at the fossilize commit instant); rises while ``STALE``/``FROZEN``.
        measured_this_epoch: True on the epoch a fresh ablation set the value.
    """

    seed_generation_id: int
    counterfactual_status: CounterfactualStatus
    last_valid_counterfactual_contribution: float | None
    epochs_since_counterfactual: int
    measured_this_epoch: bool

    @classmethod
    def never_measured(cls, seed_generation_id: int) -> "ContributionState":
        """State for a freshly germinated seed: no evidence yet (UNKNOWN)."""
        return cls(
            seed_generation_id=seed_generation_id,
            counterfactual_status=CounterfactualStatus.NEVER_MEASURED,
            last_valid_counterfactual_contribution=None,
            epochs_since_counterfactual=0,
            measured_this_epoch=False,
        )

    def record_measurement(self, value: float) -> None:
        """Record a fresh counterfactual ablation result.

        A genuine measured ``0.0`` is a FRESH measurement, distinct from absence.
        """
        self.counterfactual_status = CounterfactualStatus.FRESH
        self.last_valid_counterfactual_contribution = float(value)
        self.epochs_since_counterfactual = 0
        self.measured_this_epoch = True

    def advance_epoch(self) -> None:
        """Advance one epoch with NO fresh measurement (ageing).

        ``NEVER_MEASURED`` stays never-measured (there is nothing to grow stale from).
        ``FRESH`` becomes ``STALE``; ``FROZEN_AT_FOSSILIZE`` stays frozen but keeps ageing
        (its value shrinks toward the UNKNOWN sentinel in the observation).
        """
        self.measured_this_epoch = False
        if self.counterfactual_status == CounterfactualStatus.NEVER_MEASURED:
            return
        self.epochs_since_counterfactual += 1
        if self.counterfactual_status == CounterfactualStatus.FRESH:
            self.counterfactual_status = CounterfactualStatus.STALE

    def freeze_at_fossilize(self) -> None:
        """Freeze the last-valid contribution at the commit instant (staleness reset to 0).

        The permanent seed is excluded from the ablation thereafter (measuring host damage,
        not contribution), so its value is never re-measured; it ages from the commit anchor
        via :meth:`advance_epoch`, shrinking toward the UNKNOWN sentinel in the observation.
        ``last_valid`` is preserved (the pre-commit LOO); if it was never measured it stays
        ``None`` and the observation reads UNKNOWN.
        """
        self.counterfactual_status = CounterfactualStatus.FROZEN_AT_FOSSILIZE
        self.epochs_since_counterfactual = 0
        self.measured_this_epoch = False


__all__ = ["CounterfactualStatus", "ContributionState"]
