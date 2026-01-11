"""GERMINATE operation handler.

Handles the germination of new seeds in empty slots. This is the first
stage of the seed lifecycle, creating a new neural module that will
train on host errors.

Germination:
1. Creates a new seed module from the specified blueprint
2. Initializes seed state tracking (accuracy at germination, escrow credit)
3. Updates episode counters (seeds_created, germinate_count)
"""

from __future__ import annotations

from dataclasses import dataclass

from esper.leyline import (
    BLUEPRINT_IDS,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    TEMPO_TO_EPOCHS,
    TempoAction,
)
from esper.simic.training.handlers.base import HandlerContext, HandlerResult


@dataclass(slots=True)
class GerminateParams:
    """Parameters for the GERMINATE operation.

    Attributes:
        blueprint_idx: Index into BLUEPRINT_IDS for the seed type.
        style_idx: Index into germination styles (determines blend algorithm and alpha algorithm).
        tempo_idx: Index into TEMPO_TO_EPOCHS for blending tempo.
        alpha_target: Target alpha value for the seed.
    """

    blueprint_idx: int
    style_idx: int
    tempo_idx: int
    alpha_target: float


def can_germinate(ctx: HandlerContext) -> bool:
    """Check if germination is allowed in the given context.

    Preconditions:
    - The target slot must be enabled (exists in model.seed_slots)
    - The target slot must be empty (no active seed)

    Args:
        ctx: Handler context with slot and model state.

    Returns:
        True if germination can proceed.
    """
    # Slot must be empty (no active seed state)
    return ctx.slot.state is None


def execute_germinate(ctx: HandlerContext, params: GerminateParams) -> HandlerResult:
    """Execute the GERMINATE operation.

    Creates a new seed in the target slot using the specified blueprint
    and initializes all associated tracking state.

    Args:
        ctx: Handler context with environment and model state.
        params: Germination parameters (blueprint, style, tempo, alpha_target).

    Returns:
        HandlerResult with success=True if germination succeeded.
    """
    # Verify preconditions (should be checked by can_germinate, but defensive)
    if ctx.slot.state is not None:
        return HandlerResult(
            success=False,
            error="Cannot germinate: slot already has active seed",
        )

    # Resolve blueprint ID from index
    # Note: All blueprints are valid strings (including "noop" at index 0).
    # Action masking is responsible for preventing inappropriate selections.
    blueprint_id = BLUEPRINT_IDS[params.blueprint_idx]

    # Resolve style parameters
    blend_algorithm_id = STYLE_BLEND_IDS[params.style_idx]
    alpha_algorithm = STYLE_ALPHA_ALGORITHMS[params.style_idx]
    blend_tempo_epochs = TEMPO_TO_EPOCHS[TempoAction(params.tempo_idx)]

    # Generate unique seed ID
    seed_id = (
        f"ep{ctx.episodes_completed + ctx.env_idx}_env{ctx.env_idx}_"
        f"seed_{ctx.env_state.seeds_created}"
    )

    # Record accuracy at germination (for progress tracking)
    ctx.env_state.acc_at_germination[ctx.slot_id] = ctx.env_state.val_acc

    # Initialize escrow credit for this slot
    ctx.env_state.escrow_credit[ctx.slot_id] = 0.0

    # Create the seed in the model
    ctx.model.germinate_seed(
        blueprint_id,
        seed_id,
        slot=ctx.slot_id,
        blend_algorithm_id=blend_algorithm_id,
        blend_tempo_epochs=blend_tempo_epochs,
        alpha_algorithm=alpha_algorithm,
        alpha_target=params.alpha_target,
    )

    # Initialize Obs V3 slot tracking for the new seed
    ctx.env_state.init_obs_v3_slot_tracking(ctx.slot_id)

    # Update episode counters
    ctx.env_state.seeds_created += 1
    ctx.env_state.germinate_count += 1

    # Clear any stale optimizer for this slot
    ctx.env_state.seed_optimizers.pop(ctx.slot_id, None)

    return HandlerResult(
        success=True,
        telemetry={
            "blueprint_id": blueprint_id,
            "seed_id": seed_id,
            "blend_algorithm_id": blend_algorithm_id,
            "alpha_algorithm": alpha_algorithm.name if alpha_algorithm else None,
            "blend_tempo_epochs": blend_tempo_epochs,
            "alpha_target": params.alpha_target,
            "acc_at_germination": ctx.env_state.val_acc,
        },
    )


__all__ = [
    "GerminateParams",
    "can_germinate",
    "execute_germinate",
]
