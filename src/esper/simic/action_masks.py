"""Action Masking for Multi-Slot Control.

Only masks PHYSICALLY IMPOSSIBLE actions:
- GERMINATE: blocked if slot occupied OR at seed limit
- FOSSILIZE: blocked if not PROBATIONARY
- CULL: blocked if no seed OR seed_age < MIN_CULL_AGE
- WAIT: always valid

Does NOT mask timing heuristics (epoch, plateau, stabilization).
Tamiyo learns optimal timing from counterfactual reward signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from esper.leyline import SeedStage, MIN_CULL_AGE
from esper.leyline.factored_actions import (
    LifecycleOp,
    NUM_SLOTS,
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
)

if TYPE_CHECKING:
    from esper.kasmina.host import MorphogeneticModel

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
        if seed_slot.state.stage == SeedStage.DORMANT:
            slot_states[slot_id] = None
        else:
            slot_states[slot_id] = MaskSeedInfo(
                stage=seed_slot.state.stage.value,
                seed_age_epochs=seed_slot.state.metrics.epochs_total,
            )
    return slot_states


def compute_action_masks(
    slot_states: dict[str, MaskSeedInfo | None],
    target_slot: str,
    total_seeds: int = 0,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Only masks PHYSICALLY IMPOSSIBLE actions. Does not mask timing heuristics.

    Args:
        slot_states: Dict mapping slot_id to MaskSeedInfo or None
        target_slot: Slot ID to evaluate FOSSILIZE/CULL against (REQUIRED)
        total_seeds: Total number of active seeds across all slots
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors for each action head:
        - "slot": [NUM_SLOTS] - which slots can be targeted
        - "blueprint": [NUM_BLUEPRINTS] - which blueprints can be used
        - "blend": [NUM_BLENDS] - which blend methods can be used
        - "op": [NUM_OPS] - which operations are valid
    """
    device = device or torch.device("cpu")

    # Slot mask: all slots always selectable
    slot_mask = torch.ones(NUM_SLOTS, dtype=torch.bool, device=device)

    # Blueprint/blend: always all valid (network learns preferences)
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool, device=device)
    blend_mask = torch.ones(NUM_BLENDS, dtype=torch.bool, device=device)

    # Op mask: depends on slot states
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool, device=device)
    op_mask[LifecycleOp.WAIT] = True  # WAIT always valid

    # Check slot states
    has_empty_slot = any(info is None for info in slot_states.values())

    # Get target slot's seed info for FOSSILIZE/CULL decisions
    target_seed_info = slot_states.get(target_slot)

    # GERMINATE: valid if empty slot exists AND under seed limit
    if has_empty_slot:
        seed_limit_reached = max_seeds > 0 and total_seeds >= max_seeds
        if not seed_limit_reached:
            op_mask[LifecycleOp.GERMINATE] = True

    # FOSSILIZE/CULL: valid based on TARGET seed state
    if target_seed_info is not None:
        stage = target_seed_info.stage
        age = target_seed_info.seed_age_epochs

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
    total_seeds_list: list[int] | None = None,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        total_seeds_list: List of total seeds per env (None = all 0)
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head
    """
    device = device or torch.device("cpu")
    batch_size = len(batch_slot_states)

    slot_masks = torch.ones(batch_size, NUM_SLOTS, dtype=torch.bool, device=device)
    blueprint_masks = torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool, device=device)
    blend_masks = torch.ones(batch_size, NUM_BLENDS, dtype=torch.bool, device=device)
    op_masks = torch.zeros(batch_size, NUM_OPS, dtype=torch.bool, device=device)
    op_masks[:, LifecycleOp.WAIT] = True  # WAIT always valid

    for i, slot_states in enumerate(batch_slot_states):
        env_total_seeds = total_seeds_list[i] if total_seeds_list else 0

        has_empty_slot = False
        active_seed_info: MaskSeedInfo | None = None

        for info in slot_states.values():
            if info is None:
                has_empty_slot = True
            elif active_seed_info is None:
                active_seed_info = info

        # GERMINATE
        if has_empty_slot:
            seed_limit_reached = max_seeds > 0 and env_total_seeds >= max_seeds
            if not seed_limit_reached:
                op_masks[i, LifecycleOp.GERMINATE] = True

        # FOSSILIZE/CULL
        if active_seed_info is not None:
            stage = active_seed_info.stage
            age = active_seed_info.seed_age_epochs

            if stage in _FOSSILIZABLE_STAGES:
                op_masks[i, LifecycleOp.FOSSILIZE] = True

            if age >= MIN_CULL_AGE:
                op_masks[i, LifecycleOp.CULL] = True

    return {
        "slot": slot_masks,
        "blueprint": blueprint_masks,
        "blend": blend_masks,
        "op": op_masks,
    }


__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "MIN_CULL_AGE",
]
