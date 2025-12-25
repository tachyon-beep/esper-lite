"""Type contracts for observation fields.

These TypedDict definitions document the structure and contracts of observation dictionaries
used throughout Esper. They provide IDE hints and documentation but are NOT used for runtime
validation.

IMPORTANT: These are documentation-only contracts. Do NOT add runtime validation based on these
types. They exist solely to improve IDE support and clarify the contracts between components.
"""

from typing import TypedDict


class SeedObservationFields(TypedDict, total=True):
    """Required fields for seed potential calculation (compute_seed_potential).

    These fields are accessed via direct key lookup - missing keys raise KeyError.
    Used by: simic.ppo.features.compute_seed_potential()

    Fields:
        has_active_seed: Binary indicator (0 or 1) for whether a seed is present
        seed_stage: SeedStage enum value (1-7) representing current lifecycle stage
        seed_epochs_in_stage: Number of epochs spent in the current stage
    """

    has_active_seed: int  # 0 or 1
    seed_stage: int  # SeedStage enum value (1-7)
    seed_epochs_in_stage: int  # Epochs spent in current stage


class SlotObservationFields(TypedDict, total=False):
    """Slot-level observation fields for feature extraction.

    All fields are optional with sensible defaults for inactive slots.
    Used by: simic.ppo.features for per-slot feature extraction.

    Default values (from features.py P3 Audit comment):
    - is_active: 0.0
    - stage: 0 (inactive)
    - alpha: 0.0
    - alpha_target: alpha value (0.0 for inactive)
    - alpha_mode: AlphaMode.HOLD.value
    - alpha_steps_total: 0
    - alpha_steps_done: 0
    - time_to_target: 0.0
    - alpha_velocity: 0.0
    - alpha_algorithm: 0
    - improvement: 0.0
    - contribution_velocity: 0.0
    - interaction_sum: 0.0
    - boost_received: 0.0
    - upstream_alpha_sum: 0.0
    - downstream_alpha_sum: 0.0
    - blend_tempo_epochs: 5
    - blueprint_id: None

    Fields:
        is_active: Normalized float (0.0 or 1.0) indicating slot activity
        stage: SeedStage enum value or 0 for inactive
        alpha: Current blend coefficient [0.0, 1.0]
        alpha_target: Target alpha value for ramping
        alpha_mode: AlphaMode enum value (HOLD, RAMP, etc.)
        alpha_steps_total: Total steps in alpha ramp schedule
        alpha_steps_done: Steps completed in current alpha ramp
        time_to_target: Normalized steps remaining to alpha_target
        alpha_velocity: Rate of alpha change per epoch
        alpha_algorithm: AlphaAlgorithm enum value (LINEAR, EXPONENTIAL, etc.)
        improvement: Seed performance delta vs host baseline
        contribution_velocity: Rate of improvement change
        interaction_sum: Sum of interaction coefficients with other slots
        boost_received: Upstream boost signal strength
        upstream_alpha_sum: Sum of alpha values for upstream dependencies
        downstream_alpha_sum: Sum of alpha values for downstream dependents
        blend_tempo_epochs: Number of epochs for blending schedule
        blueprint_id: Optional blueprint identifier (None for no blueprint)
    """

    is_active: float  # Default: 0.0
    stage: int  # Default: 0 (inactive)
    alpha: float  # Default: 0.0
    alpha_target: float  # Default: alpha value
    alpha_mode: int  # Default: AlphaMode.HOLD.value
    alpha_steps_total: int  # Default: 0
    alpha_steps_done: int  # Default: 0
    time_to_target: float  # Default: 0.0
    alpha_velocity: float  # Default: 0.0
    alpha_algorithm: int  # Default: 0
    improvement: float  # Default: 0.0
    contribution_velocity: float  # Default: 0.0
    interaction_sum: float  # Default: 0.0
    boost_received: float  # Default: 0.0
    upstream_alpha_sum: float  # Default: 0.0
    downstream_alpha_sum: float  # Default: 0.0
    blend_tempo_epochs: int  # Default: 5
    blueprint_id: int | None  # Default: None
