"""Action Masking for Multi-Slot Control.

Not all action combinations are valid. This module computes masks
for each action head based on current slot states.

Key masking rules:
- Can't GERMINATE in occupied slot
- Can't ADVANCE/CULL in empty slot
- Blueprint/blend only matter when GERMINATE
- WAIT is always valid
"""

from __future__ import annotations

import torch

from esper.simic.rewards import SeedInfo
from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS


def compute_action_masks(
    slot_states: dict[str, SeedInfo | None],
    target_slot: str | None = None,
    total_seeds: int = 0,
    max_seeds: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Args:
        slot_states: Dict mapping slot_id to SeedInfo or None
        target_slot: If provided, compute op mask for this specific slot
        total_seeds: Total number of seeds germinated so far
        max_seeds: Maximum allowed seeds per epoch

    Returns:
        Dict of boolean tensors for each action head
    """
    masks = {}

    # Slot mask: all slots are always valid targets
    masks["slot"] = torch.ones(NUM_SLOTS, dtype=torch.bool)

    # Blueprint mask: all blueprints valid (masked by op later)
    masks["blueprint"] = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)

    # Blend mask: all blends valid (masked by op later)
    masks["blend"] = torch.ones(NUM_BLENDS, dtype=torch.bool)

    # Op mask: depends on slot state
    # WAIT=0 always valid, GERMINATE=1, ADVANCE=2, CULL=3
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool)
    op_mask[0] = True  # WAIT always valid

    if target_slot:
        seed_info = slot_states.get(target_slot)
    else:
        # If no target slot, check if ANY slot has a seed
        seed_info = None
        for s in slot_states.values():
            if s is not None:
                seed_info = s
                break

    if seed_info is None:
        # No active seed: can GERMINATE, cannot ADVANCE/CULL
        op_mask[1] = True   # GERMINATE
        op_mask[2] = False  # ADVANCE
        op_mask[3] = False  # CULL
    else:
        # Has active seed: cannot GERMINATE, can ADVANCE/CULL
        op_mask[1] = False  # GERMINATE
        op_mask[2] = True   # ADVANCE
        op_mask[3] = True   # CULL

    # Mask GERMINATE when at hard limit
    if total_seeds >= max_seeds:
        op_mask[1] = False  # GERMINATE

    masks["op"] = op_mask

    return masks


def compute_batch_masks(
    batch_slot_states: list[dict[str, SeedInfo | None]],
    target_slots: list[str] | None = None,
    total_seeds: list[int] | None = None,
    max_seeds: int = 1,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        target_slots: Optional list of target slots per env
        total_seeds: Optional list of total seeds per env
        max_seeds: Maximum allowed seeds per epoch

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head
    """
    batch_size = len(batch_slot_states)

    slot_masks = torch.ones(batch_size, NUM_SLOTS, dtype=torch.bool)
    blueprint_masks = torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool)
    blend_masks = torch.ones(batch_size, NUM_BLENDS, dtype=torch.bool)
    op_masks = torch.zeros(batch_size, NUM_OPS, dtype=torch.bool)
    op_masks[:, 0] = True  # WAIT always valid

    for i, slot_states in enumerate(batch_slot_states):
        target_slot = target_slots[i] if target_slots else None
        env_total_seeds = total_seeds[i] if total_seeds else 0

        if target_slot:
            seed_info = slot_states.get(target_slot)
        else:
            seed_info = next((s for s in slot_states.values() if s is not None), None)

        if seed_info is None:
            op_masks[i, 1] = True   # GERMINATE
        else:
            op_masks[i, 2] = True   # ADVANCE
            op_masks[i, 3] = True   # CULL

        # Mask GERMINATE when at hard limit
        if env_total_seeds >= max_seeds:
            op_masks[i, 1] = False  # GERMINATE

    return {
        "slot": slot_masks,
        "blueprint": blueprint_masks,
        "blend": blend_masks,
        "op": op_masks,
    }


__all__ = ["compute_action_masks", "compute_batch_masks"]
