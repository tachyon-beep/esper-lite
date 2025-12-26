"""Leyline Stages - Seed lifecycle stages and transitions.

Defines the state machine for seed development:
DORMANT -> GERMINATED -> TRAINING -> BLENDING -> HOLDING -> FOSSILIZED
                              |          |           |
                              v          v           v
                            PRUNED <- EMBARGOED <- RESETTING <- DORMANT
"""

from enum import IntEnum


class SeedStage(IntEnum):
    """Lifecycle stages for a seed.

    The full lifecycle represents a trust escalation model:

    DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► HOLDING
                    │              │            │            │
                    ▼              ▼            ▼            ▼
                 PRUNED ◄───────────────────────────────────────
                    │
                    ▼
               EMBARGOED ──► RESETTING ──► DORMANT (slot recycled)

    HOLDING ──► FOSSILIZED (terminal success)
         │
         ▼
      PRUNED (failure path)

    Stages explained:
    - DORMANT: Empty slot, waiting for a seed
    - GERMINATED: Seed attached, sanity checks passed, ready to train
    - TRAINING: Isolated training with gradient isolation from host
    - BLENDING: Alpha-managed grafting, gradually blending into host
    - HOLDING: Final validation period before permanent integration
    - FOSSILIZED: Permanently integrated into the model (terminal success)
    - PRUNED: Removed due to failure or poor performance
    - EMBARGOED: Cooldown period after removal to prevent thrashing
    - RESETTING: Cleanup before slot can be reused
    """

    UNKNOWN = 0
    DORMANT = 1
    GERMINATED = 2
    TRAINING = 3
    BLENDING = 4
    # Value 5 intentionally skipped (was SHADOWING, removed)
    HOLDING = 6
    FOSSILIZED = 7      # Terminal state (success)
    PRUNED = 8          # Failure state
    EMBARGOED = 9       # Post-removal cooldown
    RESETTING = 10      # Cleanup before reuse


# Valid transitions between stages
VALID_TRANSITIONS: dict[SeedStage, tuple[SeedStage, ...]] = {
    SeedStage.UNKNOWN: (SeedStage.DORMANT,),
    SeedStage.DORMANT: (SeedStage.GERMINATED,),
    SeedStage.GERMINATED: (SeedStage.TRAINING, SeedStage.PRUNED),
    SeedStage.TRAINING: (SeedStage.BLENDING, SeedStage.PRUNED),
    SeedStage.BLENDING: (SeedStage.HOLDING, SeedStage.PRUNED),
    # HOLDING is the full-amplitude (alpha≈1.0) decision point.
    # We also allow HOLDING -> BLENDING to support scheduled prune (phase 4),
    # but keep FOSSILIZED first so advance_stage() defaults to "finalize".
    SeedStage.HOLDING: (SeedStage.FOSSILIZED, SeedStage.BLENDING, SeedStage.PRUNED),
    SeedStage.FOSSILIZED: (),  # Terminal - no transitions out
    SeedStage.PRUNED: (SeedStage.EMBARGOED,),
    SeedStage.EMBARGOED: (SeedStage.RESETTING,),
    SeedStage.RESETTING: (SeedStage.DORMANT,),
}


def is_valid_transition(from_stage: SeedStage, to_stage: SeedStage) -> bool:
    """Check if a stage transition is valid."""
    return to_stage in VALID_TRANSITIONS.get(from_stage, ())


def is_terminal_stage(stage: SeedStage) -> bool:
    """Check if a stage is terminal (no further transitions)."""
    return stage in (SeedStage.FOSSILIZED,)


def is_active_stage(stage: SeedStage) -> bool:
    """Check if a stage represents an active seed (contributing to forward pass)."""
    return stage in (
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
        SeedStage.FOSSILIZED,
    )


def is_failure_stage(stage: SeedStage) -> bool:
    """Check if a stage represents a failed seed."""
    return stage in (SeedStage.PRUNED, SeedStage.EMBARGOED, SeedStage.RESETTING)
