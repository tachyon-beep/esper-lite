"""Leyline Stages - Seed lifecycle stages and transitions.

Defines the state machine for seed development:
DORMANT -> GERMINATED -> TRAINING -> BLENDING -> PROBATIONARY -> FOSSILIZED
                              |          |           |
                              v          v           v
                            CULLED <- EMBARGOED <- RESETTING <- DORMANT
"""

from enum import Enum, IntEnum, auto


class SeedStage(IntEnum):
    """Lifecycle stages for a seed.

    The full lifecycle represents a trust escalation model:

    DORMANT ──► GERMINATED ──► TRAINING ──► BLENDING ──► PROBATIONARY
                    │              │            │            │
                    ▼              ▼            ▼            ▼
                 CULLED ◄───────────────────────────────────────
                    │
                    ▼
               EMBARGOED ──► RESETTING ──► DORMANT (slot recycled)

    PROBATIONARY ──► FOSSILIZED (terminal success)
         │
         ▼
      CULLED (failure path)

    Stages explained:
    - DORMANT: Empty slot, waiting for a seed
    - GERMINATED: Seed attached, sanity checks passed, ready to train
    - TRAINING: Isolated training with gradient isolation from host
    - BLENDING: Alpha-managed grafting, gradually blending into host
    - PROBATIONARY: Final validation period before permanent integration
    - FOSSILIZED: Permanently integrated into the model (terminal success)
    - CULLED: Removed due to failure or poor performance
    - EMBARGOED: Cooldown period after culling to prevent thrashing
    - RESETTING: Cleanup before slot can be reused
    """

    UNKNOWN = 0
    DORMANT = 1
    GERMINATED = 2
    TRAINING = 3
    BLENDING = 4
    SHADOWING = 5       # DEPRECATED: No longer used in lifecycle (kept for serialization compat)
    PROBATIONARY = 6
    FOSSILIZED = 7      # Terminal state (success)
    CULLED = 8          # Failure state
    EMBARGOED = 9       # Post-cull cooldown
    RESETTING = 10      # Cleanup before reuse


# Valid transitions between stages
VALID_TRANSITIONS: dict[SeedStage, tuple[SeedStage, ...]] = {
    SeedStage.UNKNOWN: (SeedStage.DORMANT,),
    SeedStage.DORMANT: (SeedStage.GERMINATED,),
    SeedStage.GERMINATED: (SeedStage.TRAINING, SeedStage.CULLED),
    SeedStage.TRAINING: (SeedStage.BLENDING, SeedStage.CULLED),
    SeedStage.BLENDING: (SeedStage.PROBATIONARY, SeedStage.CULLED),
    SeedStage.PROBATIONARY: (SeedStage.FOSSILIZED, SeedStage.CULLED),
    SeedStage.FOSSILIZED: (),  # Terminal - no transitions out
    SeedStage.CULLED: (SeedStage.EMBARGOED,),
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
        SeedStage.PROBATIONARY,
        SeedStage.FOSSILIZED,
    )


def is_failure_stage(stage: SeedStage) -> bool:
    """Check if a stage represents a failed seed."""
    return stage in (SeedStage.CULLED, SeedStage.EMBARGOED, SeedStage.RESETTING)


class CommandType(Enum):
    """Types of commands Tamiyo can issue to Kasmina."""

    # Lifecycle commands
    GERMINATE = auto()          # Create a new seed
    ADVANCE_STAGE = auto()      # Move seed to next stage
    CULL = auto()               # Kill a seed

    # Parameter commands
    SET_ALPHA = auto()          # Set blending alpha directly
    SET_LEARNING_RATE = auto()  # Adjust seed learning rate

    # Control commands
    PAUSE_SEED = auto()         # Temporarily pause seed training
    RESUME_SEED = auto()        # Resume paused seed

    # Query commands
    REQUEST_STATE = auto()      # Request current seed state


class RiskLevel(IntEnum):
    """Risk assessment levels for commands."""
    GREEN = 1       # Safe, routine operation
    YELLOW = 2      # Caution, monitor closely
    ORANGE = 3      # Elevated risk, may need rollback
    RED = 4         # High risk, prepare for intervention
    CRITICAL = 5    # Emergency, immediate attention required
