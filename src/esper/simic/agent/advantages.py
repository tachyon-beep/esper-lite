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
        |   +-- tempo_head: WHEN to germinate (timestep selection)
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
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute advantages with causal masking per head.

    Args:
        base_advantages: GAE advantages [batch] or [batch, seq]
        op_actions: Operation actions [batch] or [batch, seq] (LifecycleOp values)

    Returns:
        Tuple of (per_head_advantages, masks):
        - per_head_advantages: Dict with per-head advantages, causally masked
        - masks: Dict with boolean causal masks (for reuse in KL/entropy/loss)

    Note:
        Boolean masks are multiplied directly (no .float() conversion) to
        preserve base_advantages.dtype under AMP (float16/bfloat16).
    """
    # B4-DRL-01: Use single source of truth for causal masks
    masks = compute_causal_masks(op_actions)

    # Apply causal masks to advantages
    # op head: always gets advantage (always causally relevant)
    # M8: No clone needed for op_advantages. The caller (PPO.update) only reads
    # these values for loss computation - it never modifies them in-place.
    # All arithmetic operations (advantage * ratio, etc.) create new tensors.
    # If this contract changes, we'd need to clone to prevent mutation of
    # base_advantages. Other heads use multiplication which creates new tensors.
    #
    # PERF: Multiply by bool directly - PyTorch fuses the bool→float conversion
    # into the multiplication kernel, preserving base_advantages.dtype.
    per_head_advantages = {
        "op": base_advantages,
        "slot": base_advantages * masks["slot"],
        "blueprint": base_advantages * masks["blueprint"],
        "style": base_advantages * masks["style"],
        "tempo": base_advantages * masks["tempo"],
        "alpha_target": base_advantages * masks["alpha_target"],
        "alpha_speed": base_advantages * masks["alpha_speed"],
        "alpha_curve": base_advantages * masks["alpha_curve"],
    }
    return per_head_advantages, masks


__all__ = ["compute_per_head_advantages"]
