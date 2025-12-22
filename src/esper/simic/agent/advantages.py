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
        |   +-- blend_head: HOW to blend
        |   +-- alpha_target_head: TARGET amplitude for initial blend
        |   +-- alpha_algorithm_head: BLEND composition / gating mode
        |
        +-- FOSSILIZE:
        |   +-- slot_head: WHICH seed to fossilize (target_slot)
        |
        +-- SET_ALPHA_TARGET:
        |   +-- slot_head: WHICH seed to retarget
        |   +-- alpha_target_head: TARGET alpha
        |   +-- alpha_speed_head: SPEED of schedule
        |   +-- alpha_curve_head: CURVE of schedule
        |   +-- alpha_algorithm_head: BLEND composition / gating mode
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

from esper.leyline.factored_actions import LifecycleOp


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
    # Create causal masks based on op type
    is_wait = op_actions == LifecycleOp.WAIT
    is_germinate = op_actions == LifecycleOp.GERMINATE
    is_set_alpha = op_actions == LifecycleOp.SET_ALPHA_TARGET
    is_prune = op_actions == LifecycleOp.PRUNE

    # op head: always gets advantage (always causally relevant)
    # M8: No clone needed - we're not modifying the tensor, just returning it.
    # Other heads use multiplication which creates new tensors anyway.
    op_advantages = base_advantages

    # slot head: relevant for GERMINATE, FOSSILIZE, PRUNE, ADVANCE (not WAIT)
    slot_mask = ~is_wait
    slot_advantages = base_advantages * slot_mask.float()

    # blueprint head: only relevant for GERMINATE
    blueprint_mask = is_germinate
    blueprint_advantages = base_advantages * blueprint_mask.float()

    # blend head: only relevant for GERMINATE
    blend_mask = is_germinate
    blend_advantages = base_advantages * blend_mask.float()

    # tempo head: only relevant for GERMINATE (same as blueprint/blend)
    tempo_mask = is_germinate
    tempo_advantages = base_advantages * tempo_mask.float()

    # alpha_target head: relevant for GERMINATE and SET_ALPHA_TARGET
    alpha_target_mask = is_set_alpha | is_germinate
    alpha_target_advantages = base_advantages * alpha_target_mask.float()

    # alpha_speed/alpha_curve: relevant for SET_ALPHA_TARGET and PRUNE
    alpha_speed_mask = is_set_alpha | is_prune
    alpha_speed_advantages = base_advantages * alpha_speed_mask.float()
    alpha_curve_mask = is_set_alpha | is_prune
    alpha_curve_advantages = base_advantages * alpha_curve_mask.float()

    # alpha_algorithm head: relevant for GERMINATE and SET_ALPHA_TARGET
    alpha_algorithm_mask = is_set_alpha | is_germinate
    alpha_algorithm_advantages = base_advantages * alpha_algorithm_mask.float()

    return {
        "op": op_advantages,
        "slot": slot_advantages,
        "blueprint": blueprint_advantages,
        "blend": blend_advantages,
        "tempo": tempo_advantages,
        "alpha_target": alpha_target_advantages,
        "alpha_speed": alpha_speed_advantages,
        "alpha_curve": alpha_curve_advantages,
        "alpha_algorithm": alpha_algorithm_advantages,
    }


__all__ = ["compute_per_head_advantages"]
