"""Per-head advantage computation with causal masking.

Causal structure for Tamiyo's factored action space:

    DECISION TREE AT EACH EPOCH:

    op_head decides: [WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE, ADVANCE]
        |
        +-- WAIT: No other heads matter
        |
        +-- GERMINATE:
        |   +-- slot_head: WHERE to place seed
        |   +-- blueprint_head: WHAT architecture
        |   +-- style_head: HOW to germinate (blend + alpha algorithm)
        |   +-- alpha_target_head: TARGET amplitude for initial blend
        |
        +-- FOSSILIZE:
        |   +-- slot_head: WHICH seed to fossilize (target_slot)
        |
        +-- SET_ALPHA_TARGET:
        |   +-- slot_head: WHICH seed to retarget
        |   +-- style_head: WHICH alpha algorithm to use
        |   +-- alpha_target_head: TARGET alpha
        |   +-- alpha_speed_head: SPEED of schedule
        |   +-- alpha_curve_head: CURVE of schedule
        |
        +-- PRUNE:
            +-- slot_head: WHICH seed to remove (target_slot)
            +-- alpha_speed_head: SPEED of schedule
            +-- alpha_curve_head: CURVE of schedule
        |
        +-- ADVANCE:
            +-- slot_head: WHICH seed to advance (target_slot)

When computing advantages, we mask out heads that had no causal effect
on the outcome. This reduces gradient noise significantly.
"""

from __future__ import annotations

import torch

from esper.leyline.causal_masks import compute_causal_masks


def compute_per_head_advantages(
    base_advantages: torch.Tensor,
    op_actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute advantages with causal masking per head.

    Args:
        base_advantages: GAE advantages [batch] or [batch, seq]
        op_actions: Operation actions [batch] or [batch, seq] (LifecycleOp values)

    Returns:
        Dict with per-head advantages, causally masked.
    """
    # B4-DRL-01: Use single source of truth for causal masks
    masks = compute_causal_masks(op_actions)

    # Apply causal masks to advantages
    # op head: always gets advantage (always causally relevant)
    # M8: No clone needed - we're not modifying the tensor, just returning it.
    # Other heads use multiplication which creates new tensors anyway.
    return {
        "op": base_advantages,
        "slot": base_advantages * masks["slot"].float(),
        "blueprint": base_advantages * masks["blueprint"].float(),
        "style": base_advantages * masks["style"].float(),
        "tempo": base_advantages * masks["tempo"].float(),
        "alpha_target": base_advantages * masks["alpha_target"].float(),
        "alpha_speed": base_advantages * masks["alpha_speed"].float(),
        "alpha_curve": base_advantages * masks["alpha_curve"].float(),
    }


__all__ = ["compute_per_head_advantages"]
