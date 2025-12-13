"""Action Masking for Multi-Slot Control.

Only masks PHYSICALLY IMPOSSIBLE actions:
- SLOT: only enabled slots (from --slots arg) are selectable
- GERMINATE: blocked if ALL enabled slots occupied OR at seed limit
- FOSSILIZE: blocked if NO enabled slot has a PROBATIONARY seed
- CULL: blocked if NO enabled slot has a cullable seed with age >= MIN_CULL_AGE
- WAIT: always valid
- BLUEPRINT: NOOP always blocked (0 trainable parameters)

Does NOT mask timing heuristics (epoch, plateau, stabilization).
Tamiyo learns optimal timing from counterfactual reward signals.

Multi-slot execution: The sampled slot determines which slot is targeted.
The op mask is computed optimistically (valid if ANY enabled slot allows it).
Invalid slot+op combinations are rejected at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from esper.leyline import SeedStage, MIN_CULL_AGE
from esper.leyline.factored_actions import (
    BlueprintAction,
    LifecycleOp,
    SlotAction,
    NUM_SLOTS,
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
)

if TYPE_CHECKING:
    from esper.kasmina.host import MorphogeneticModel

# Mapping from slot ID string to SlotAction index
_SLOT_ID_TO_INDEX: dict[str, int] = {
    "early": SlotAction.EARLY.value,
    "mid": SlotAction.MID.value,
    "late": SlotAction.LATE.value,
}

# Stage sets for validation
_FOSSILIZABLE_STAGES = frozenset({
    SeedStage.PROBATIONARY.value,
})

# Stages from which a seed can be culled
# Derived as: active stages that have CULLED in VALID_TRANSITIONS
# Equivalently: set(active_stages) - {FOSSILIZED} (terminal success)
# See stages.py VALID_TRANSITIONS for authoritative source
_CULLABLE_STAGES = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.PROBATIONARY.value,
    # NOT FOSSILIZED - terminal success, no outgoing transitions
})


@dataclass(frozen=True, slots=True)
class MaskSeedInfo:
    """Minimal seed info for action masking only.

    Uses int for stage (not enum) for torch.compile safety.
    """

    stage: int  # SeedStage.value
    seed_age_epochs: int


def build_slot_states(
    model: MorphogeneticModel,
    slots: list[str],
) -> dict[str, MaskSeedInfo | None]:
    """Build slot_states dict for action masking from model state.

    Args:
        model: The morphogenetic model
        slots: List of slot IDs to check

    Returns:
        Dict mapping slot_id to MaskSeedInfo or None if slot is empty
    """
    slot_states: dict[str, MaskSeedInfo | None] = {}
    for slot_id in slots:
        seed_slot = model.seed_slots[slot_id]
        # state is None when slot has no seed, or DORMANT when seed is inactive
        if seed_slot.state is None or seed_slot.state.stage == SeedStage.DORMANT:
            slot_states[slot_id] = None
        else:
            slot_states[slot_id] = MaskSeedInfo(
                stage=seed_slot.state.stage.value,
                seed_age_epochs=seed_slot.state.metrics.epochs_total,
            )
    return slot_states


def compute_action_masks(
    slot_states: dict[str, MaskSeedInfo | None],
    enabled_slots: list[str],
    total_seeds: int = 0,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Only masks PHYSICALLY IMPOSSIBLE actions. Does not mask timing heuristics.

    Args:
        slot_states: Dict mapping slot_id to MaskSeedInfo or None
        enabled_slots: List of slot IDs that are enabled (from --slots arg)
        total_seeds: Total number of active seeds across all slots
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors for each action head:
        - "slot": [NUM_SLOTS] - which slots can be targeted (only enabled slots)
        - "blueprint": [NUM_BLUEPRINTS] - which blueprints can be used
        - "blend": [NUM_BLENDS] - which blend methods can be used
        - "op": [NUM_OPS] - which operations are valid (ANY enabled slot)
    """
    device = device or torch.device("cpu")

    # Slot mask: only enabled slots are selectable
    slot_mask = torch.zeros(NUM_SLOTS, dtype=torch.bool, device=device)
    for slot_id in enabled_slots:
        if slot_id in _SLOT_ID_TO_INDEX:
            slot_mask[_SLOT_ID_TO_INDEX[slot_id]] = True

    # Blueprint mask: disable zero-parameter blueprints (can't train them)
    # NOOP is a placeholder seed with no trainable parameters
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool, device=device)
    blueprint_mask[BlueprintAction.NOOP] = False

    # Blend mask: all blend methods valid (network learns preferences)
    blend_mask = torch.ones(NUM_BLENDS, dtype=torch.bool, device=device)

    # Op mask: depends on slot states across ALL enabled slots
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool, device=device)
    op_mask[LifecycleOp.WAIT] = True  # WAIT always valid

    # Check slot states for enabled slots only
    has_empty_enabled_slot = any(
        slot_states.get(slot_id) is None
        for slot_id in enabled_slots
    )

    # GERMINATE: valid if ANY enabled slot is empty AND under seed limit
    if has_empty_enabled_slot:
        seed_limit_reached = max_seeds > 0 and total_seeds >= max_seeds
        if not seed_limit_reached:
            op_mask[LifecycleOp.GERMINATE] = True

    # FOSSILIZE/CULL: valid if ANY enabled slot has a valid state
    # (optimistic masking - network learns slot+op associations)
    for slot_id in enabled_slots:
        seed_info = slot_states.get(slot_id)
        if seed_info is not None:
            stage = seed_info.stage
            age = seed_info.seed_age_epochs

            # FOSSILIZE: only from PROBATIONARY
            if stage in _FOSSILIZABLE_STAGES:
                op_mask[LifecycleOp.FOSSILIZE] = True

            # CULL: only from cullable stages AND if seed age >= MIN_CULL_AGE
            if stage in _CULLABLE_STAGES and age >= MIN_CULL_AGE:
                op_mask[LifecycleOp.CULL] = True

    return {
        "slot": slot_mask,
        "blueprint": blueprint_mask,
        "blend": blend_mask,
        "op": op_mask,
    }


def compute_batch_masks(
    batch_slot_states: list[dict[str, MaskSeedInfo | None]],
    enabled_slots: list[str],
    total_seeds_list: list[int] | None = None,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Delegates to compute_action_masks for each env, then stacks results.
    This ensures single source of truth for masking logic.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        enabled_slots: List of enabled slot IDs (same for all envs, from --slots arg)
        total_seeds_list: List of total seeds per env (None = all 0)
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head
    """
    device = device or torch.device("cpu")

    # Delegate to compute_action_masks for each env
    masks_list = [
        compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled_slots,
            total_seeds=total_seeds_list[i] if total_seeds_list else 0,
            max_seeds=max_seeds,
            device=device,
        )
        for i, slot_states in enumerate(batch_slot_states)
    ]

    # Stack into batch tensors
    return {
        key: torch.stack([m[key] for m in masks_list])
        for key in masks_list[0]
    }


def slot_id_to_index(slot_id: str) -> int:
    """Convert slot ID string to SlotAction index.

    Args:
        slot_id: Slot name ("early", "mid", "late")

    Returns:
        Corresponding SlotAction index (0, 1, 2)

    Raises:
        KeyError: If slot_id is not a valid slot name
    """
    return _SLOT_ID_TO_INDEX[slot_id]


__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
]
