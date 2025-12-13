"""Per-head advantage computation with causal masking.

Causal structure for Tamiyo's factored action space:

    DECISION TREE AT EACH EPOCH:

    op_head decides: [WAIT, GERMINATE, FOSSILIZE, CULL]
        |
        +-- WAIT: No other heads matter
        |
        +-- GERMINATE:
        |   +-- slot_head: WHERE to place seed
        |   +-- blueprint_head: WHAT architecture
        |   +-- blend_head: HOW to blend
        |
        +-- FOSSILIZE:
        |   +-- slot_head: WHICH seed to fossilize (target_slot)
        |
        +-- CULL:
            +-- slot_head: WHICH seed to remove (target_slot)

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

    # op head: always gets advantage (always causally relevant)
    op_advantages = base_advantages.clone()

    # slot head: relevant for GERMINATE, FOSSILIZE, CULL (not WAIT)
    slot_mask = ~is_wait
    slot_advantages = base_advantages * slot_mask.float()

    # blueprint head: only relevant for GERMINATE
    blueprint_mask = is_germinate
    blueprint_advantages = base_advantages * blueprint_mask.float()

    # blend head: only relevant for GERMINATE
    blend_mask = is_germinate
    blend_advantages = base_advantages * blend_mask.float()

    return {
        "op": op_advantages,
        "slot": slot_advantages,
        "blueprint": blueprint_advantages,
        "blend": blend_advantages,
    }


def compute_per_head_policy_loss(
    per_head_advantages: dict[str, torch.Tensor],
    per_head_ratios: dict[str, torch.Tensor],
    clip_epsilon: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute clipped PPO loss per head.

    Args:
        per_head_advantages: Dict of masked advantages per head
        per_head_ratios: Dict of probability ratios per head
        clip_epsilon: PPO clipping parameter

    Returns:
        Dict of policy losses per head (to be summed)
    """
    losses = {}

    for key in ["op", "slot", "blueprint", "blend"]:
        ratio = per_head_ratios[key]
        advantages = per_head_advantages[key]

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        losses[key] = -torch.min(surr1, surr2)

    return losses


__all__ = ["compute_per_head_advantages", "compute_per_head_policy_loss"]
