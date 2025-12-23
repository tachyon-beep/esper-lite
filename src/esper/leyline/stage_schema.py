"""Stage Schema - Centralized stage encoding contract.

Defines the canonical mapping from SeedStage enum values to ML features.
All stage encoding must go through this schema to ensure:
1. Invalid/retired stage values (like 5) are rejected
2. Stage features are versioned for compatibility detection
3. One-hot encoding provides categorical representation

This is a Leyline contract - imported by Tamiyo, Simic, and telemetry.
"""

from __future__ import annotations

from esper.leyline.stages import SeedStage

# Schema version - increment when encoding changes
# Version 1: Initial one-hot encoding (replaces normalized scalar)
STAGE_SCHEMA_VERSION: int = 1

# Valid stages in lifecycle order (excludes reserved value 5)
# This tuple defines both the allowed values AND the one-hot index order
VALID_STAGES: tuple[SeedStage, ...] = (
    SeedStage.UNKNOWN,      # index 0
    SeedStage.DORMANT,      # index 1
    SeedStage.GERMINATED,   # index 2
    SeedStage.TRAINING,     # index 3
    SeedStage.BLENDING,     # index 4
    SeedStage.HOLDING,      # index 5
    SeedStage.FOSSILIZED,   # index 6
    SeedStage.PRUNED,       # index 7
    SeedStage.EMBARGOED,    # index 8
    SeedStage.RESETTING,    # index 9
)

# Number of valid stages (for one-hot sizing)
NUM_STAGES: int = len(VALID_STAGES)  # 10

# Stage enum value -> contiguous index (for one-hot encoding)
# Maps: 0->0, 1->1, 2->2, 3->3, 4->4, 6->5, 7->6, 8->7, 9->8, 10->9
STAGE_TO_INDEX: dict[int, int] = {
    stage.value: idx for idx, stage in enumerate(VALID_STAGES)
}

# Contiguous index -> Stage enum value (for decoding)
INDEX_TO_STAGE: dict[int, int] = {
    idx: stage.value for idx, stage in enumerate(VALID_STAGES)
}

# Valid stage values as frozenset (for O(1) validation)
VALID_STAGE_VALUES: frozenset[int] = frozenset(stage.value for stage in VALID_STAGES)

# Reserved/retired stage values that must be rejected
RESERVED_STAGE_VALUES: frozenset[int] = frozenset({5})  # Was SHADOWING


def stage_to_one_hot(stage_value: int) -> list[float]:
    """Convert stage value to one-hot encoding.

    Args:
        stage_value: SeedStage enum value (0-10, excluding 5)

    Returns:
        One-hot list of NUM_STAGES floats (10 dimensions)

    Raises:
        ValueError: If stage_value is not valid (including reserved value 5)

    Example:
        >>> stage_to_one_hot(SeedStage.TRAINING.value)  # value 3
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        >>> stage_to_one_hot(SeedStage.HOLDING.value)   # value 6
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    """
    if stage_value not in STAGE_TO_INDEX:
        raise ValueError(
            f"Invalid stage value {stage_value}; "
            f"valid values are {sorted(VALID_STAGE_VALUES)}"
        )
    one_hot = [0.0] * NUM_STAGES
    one_hot[STAGE_TO_INDEX[stage_value]] = 1.0
    return one_hot


def stage_to_index(stage_value: int) -> int:
    """Convert stage value to contiguous index.

    Args:
        stage_value: SeedStage enum value

    Returns:
        Contiguous index in range [0, NUM_STAGES-1]

    Raises:
        ValueError: If stage_value is not valid
    """
    if stage_value not in STAGE_TO_INDEX:
        raise ValueError(
            f"Invalid stage value {stage_value}; "
            f"valid values are {sorted(VALID_STAGE_VALUES)}"
        )
    return STAGE_TO_INDEX[stage_value]


def validate_stage_value(stage_value: int, *, context: str = "") -> None:
    """Validate a stage value at an untrusted boundary.

    Args:
        stage_value: Value to validate
        context: Optional context for error message (e.g., "SeedTelemetry.stage")

    Raises:
        ValueError: If stage_value is not valid
    """
    if stage_value in RESERVED_STAGE_VALUES:
        ctx = f"{context}: " if context else ""
        raise ValueError(
            f"{ctx}Stage value {stage_value} is reserved/retired; "
            f"valid values are {sorted(VALID_STAGE_VALUES)}"
        )
    if stage_value not in VALID_STAGE_VALUES:
        ctx = f"{context}: " if context else ""
        raise ValueError(
            f"{ctx}Invalid stage value {stage_value}; "
            f"valid values are {sorted(VALID_STAGE_VALUES)}"
        )
