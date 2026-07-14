"""Fossil-settlement contracts (Phase 2, "make permanence visible").

Design provenance: docs/analysis/2026-07-14-permanence-visible-preregistration.md
§2 (rounds 15–21) and docs/plans/ready/2026-07-14-phase2-settlement-implementation.md.
The feature is DEFAULT-OFF: it activates only when a validated
``FossilSettlementConfig`` object is attached (the escrow fail-closed precedent) —
never via a bare string or bool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GateForm = Literal["raw", "lcb"]
# OWNER FORK F1 (PDR-0091 #4): what a safety-abort pays. Both implemented;
# the field is REQUIRED (no default) so activation forces an explicit choice.
AbortPaymentMode = Literal["settle_partial", "void"]


@dataclass(frozen=True, kw_only=True)
class FossilSettlementConfig:
    """The non-selectable settlement protocol's frozen parameters (pre-reg §2.2–§2.4)."""

    abort_payment_mode: AbortPaymentMode

    # Fixed-cadence audit boundaries: settle at the first multiple of w_settle
    # that is >= request + min_window (READ-1: the policy chooses WHETHER, never WHEN).
    w_settle: int = 10
    min_window: int = 5

    # q_settle: EWMA (span 5, adjust=False) over post-request measurements;
    # >= min_measurements valid measurements required, else extend to the next
    # boundary (the zero-slack rule, §2.3).
    ewma_span: int = 5
    min_measurements: int = 5

    # Boundary qualification (OWNER FORK F3: both forms implemented; the
    # near-cliff noise read decides — PDR-0092).
    q_threshold: float = 1.0
    gate_form: GateForm = "raw"
    lcb_z: float = 1.0

    def __post_init__(self) -> None:
        if self.min_window < 1 or self.w_settle < 1:
            raise ValueError("w_settle and min_window must be >= 1")
        if self.min_measurements < 1:
            raise ValueError("min_measurements must be >= 1")
        if self.ewma_span < 1:
            raise ValueError("ewma_span must be >= 1")


__all__ = ["AbortPaymentMode", "FossilSettlementConfig", "GateForm"]
