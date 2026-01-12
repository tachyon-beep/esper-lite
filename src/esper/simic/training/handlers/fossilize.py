"""FOSSILIZE operation handler.

Handles the permanent integration of seeds into the host network.
Fossilization is the successful terminal state for a seed - its weights
become part of the host model.

Fossilization:
1. Requires seed to be in HOLDING stage
2. Evaluates G5 gate (contribution threshold)
3. Permanently integrates seed weights into host
4. Computes hindsight credit for scaffolds that helped this seed
5. Updates episode counters and cleans up tracking state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    HINDSIGHT_CREDIT_WEIGHT,
    MAX_HINDSIGHT_CREDIT,
    SeedStage,
)
from esper.simic.rewards import compute_scaffold_hindsight_credit
from esper.simic.training.handlers.base import HandlerContext, HandlerResult

if TYPE_CHECKING:
    from esper.leyline import SlottedHostProtocol
    from esper.simic.rewards import SeedInfo


@dataclass(slots=True)
class HindsightCreditResult:
    """Result from hindsight credit computation.

    Attributes:
        total_credit: Total hindsight credit to add (capped).
        scaffold_count: Number of scaffolds that contributed.
        total_delay: Sum of delays for average calculation.
    """

    total_credit: float
    scaffold_count: int
    total_delay: int


def compute_hindsight_credit_for_beneficiary(
    ctx: HandlerContext,
    beneficiary_improvement: float,
) -> HindsightCreditResult:
    """Compute temporally-discounted hindsight credit for scaffolds.

    When a seed fossilizes successfully, scaffolds that previously boosted
    it receive delayed credit proportional to:
    - The boost they provided
    - The beneficiary's final improvement
    - Temporal discount (gamma^delay)

    Args:
        ctx: Handler context with scaffold ledger.
        beneficiary_improvement: The fossilizing seed's total improvement.

    Returns:
        HindsightCreditResult with credit and metrics.
    """
    if beneficiary_improvement <= 0:
        return HindsightCreditResult(
            total_credit=0.0,
            scaffold_count=0,
            total_delay=0,
        )

    current_epoch = ctx.epoch
    total_credit = 0.0
    scaffold_count = 0
    total_delay = 0

    # Find all scaffolds that boosted this beneficiary
    for scaffold_slot, boosts in ctx.env_state.scaffold_boost_ledger.items():
        for boost_given, beneficiary_slot, epoch_of_boost in boosts:
            if beneficiary_slot == ctx.slot_id and boost_given > 0:
                # Temporal discount: credit decays with distance
                delay = current_epoch - epoch_of_boost
                discount = DEFAULT_GAMMA**delay

                # Compute discounted hindsight credit
                raw_credit = compute_scaffold_hindsight_credit(
                    boost_given=boost_given,
                    beneficiary_improvement=beneficiary_improvement,
                    credit_weight=HINDSIGHT_CREDIT_WEIGHT,
                )
                total_credit += raw_credit * discount
                scaffold_count += 1
                total_delay += delay

    # Cap total credit to prevent runaway values
    total_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)

    return HindsightCreditResult(
        total_credit=total_credit,
        scaffold_count=scaffold_count,
        total_delay=total_delay,
    )


def clear_beneficiary_from_ledger(ctx: HandlerContext) -> None:
    """Remove fossilized beneficiary from all scaffold ledgers.

    Once a seed fossilizes, it can no longer receive boosts, so we clean
    up the ledger entries that reference it.

    Args:
        ctx: Handler context with scaffold ledger to clean.
    """
    for scaffold_slot in list(ctx.env_state.scaffold_boost_ledger.keys()):
        ctx.env_state.scaffold_boost_ledger[scaffold_slot] = [
            (b, ben, e)
            for (b, ben, e) in ctx.env_state.scaffold_boost_ledger[scaffold_slot]
            if ben != ctx.slot_id
        ]
        # Remove empty entries
        if not ctx.env_state.scaffold_boost_ledger[scaffold_slot]:
            del ctx.env_state.scaffold_boost_ledger[scaffold_slot]


def can_fossilize(ctx: HandlerContext) -> bool:
    """Check if FOSSILIZE operation is allowed in the given context.

    Preconditions:
    - The slot must have an active seed (seed_state is not None)
    - The seed must be in HOLDING stage

    Note: G5 gate evaluation (contribution threshold) happens during
    execution, not in this precondition check.

    Args:
        ctx: Handler context with slot and seed state.

    Returns:
        True if fossilize can be attempted.
    """
    if ctx.seed_state is None:
        return False

    return ctx.seed_state.stage == SeedStage.HOLDING


def execute_fossilize(
    ctx: HandlerContext,
    seed_info: "SeedInfo | None",
    fossilize_fn: Callable[["SlottedHostProtocol", str], bool],
) -> HandlerResult:
    """Execute the FOSSILIZE operation.

    Calls the external fossilize function (from ActionExecutionContext)
    which handles G5 gate evaluation and actual fossilization.

    Args:
        ctx: Handler context with environment and model state.
        seed_info: SeedInfo for the fossilizing seed (for contribution check).
        fossilize_fn: Callback to perform actual fossilization.

    Returns:
        HandlerResult with success=True if fossilization succeeded.
    """
    # Verify preconditions
    if ctx.seed_state is None:
        return HandlerResult(
            success=False,
            error="Cannot fossilize: no active seed in slot",
        )

    if ctx.seed_state.stage != SeedStage.HOLDING:
        return HandlerResult(
            success=False,
            error=f"Cannot fossilize: seed in {ctx.seed_state.stage.name}, must be HOLDING",
        )

    # Execute fossilization (handles G5 gate internally)
    success = fossilize_fn(ctx.model, ctx.slot_id)

    if not success:
        return HandlerResult(
            success=False,
            error="Fossilization failed (likely G5 gate not passed)",
        )

    # Update episode counters
    ctx.env_state.seeds_fossilized += 1
    ctx.env_state.fossilize_count += 1

    # Track contributing fossilized seeds
    is_contributing = (
        seed_info is not None
        and seed_info.total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
    )
    if is_contributing:
        ctx.env_state.contributing_fossilized += 1

    # Clean up germination tracking
    ctx.env_state.acc_at_germination.pop(ctx.slot_id, None)

    # Compute hindsight credit for scaffolds
    beneficiary_improvement = seed_info.total_improvement if seed_info else 0.0
    hindsight = compute_hindsight_credit_for_beneficiary(ctx, beneficiary_improvement)

    if hindsight.total_credit > 0:
        ctx.env_state.pending_hindsight_credit += hindsight.total_credit

    # Clear beneficiary from all scaffold ledgers
    clear_beneficiary_from_ledger(ctx)

    # Clean up seed optimizer (B8-DRL-02 fix)
    ctx.env_state.seed_optimizers.pop(ctx.slot_id, None)

    # Trigger governor snapshot (BUG FIX for rollback coherence)
    ctx.env_state.needs_governor_snapshot = True

    return HandlerResult(
        success=True,
        telemetry={
            "is_contributing": is_contributing,
            "total_improvement": seed_info.total_improvement if seed_info else 0.0,
            "hindsight_credit": hindsight.total_credit,
            "scaffold_count": hindsight.scaffold_count,
            "avg_scaffold_delay": (
                hindsight.total_delay / hindsight.scaffold_count
                if hindsight.scaffold_count > 0
                else 0.0
            ),
        },
    )


__all__ = [
    "HindsightCreditResult",
    "can_fossilize",
    "clear_beneficiary_from_ledger",
    "compute_hindsight_credit_for_beneficiary",
    "execute_fossilize",
]
