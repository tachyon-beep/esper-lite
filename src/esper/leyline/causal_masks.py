"""Causal relevance masks for Tamiyo's factored action space.

This module defines the SINGLE SOURCE OF TRUTH for which action heads are
causally relevant for each LifecycleOp. These masks are used by:

1. advantages.py - Mask GAE advantages per head (credit assignment)
2. ppo.py - Mask KL divergence, policy loss, and entropy bonus

CRITICAL: If you add a new LifecycleOp or action head, update this module
and both call sites will automatically use the correct masks.

Causal Structure:
    WAIT:           Only op head matters
    GERMINATE:      slot, blueprint, style, tempo, alpha_target
    SET_ALPHA:      slot, style, alpha_target, alpha_speed, alpha_curve
    PRUNE:          slot, alpha_speed, alpha_curve
    FOSSILIZE:      slot only (implicit via ~is_wait)
    ADVANCE:        slot only (implicit via ~is_wait)
"""

from __future__ import annotations

import torch

from esper.leyline.factored_actions import LifecycleOp


def compute_causal_masks(op_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute causal relevance masks for each action head.

    Args:
        op_actions: Tensor of LifecycleOp values [batch] or [batch, seq]

    Returns:
        Dict mapping head names to boolean masks indicating causal relevance.
        True = head's action affected the outcome for this timestep.

    Example:
        >>> masks = compute_causal_masks(op_actions)
        >>> # For advantage masking (bool multiplication preserves dtype):
        >>> masked_adv = advantages * masks["blueprint"]
        >>> # For masked mean:
        >>> head_loss = (loss * masks["slot"]).sum() / masks["slot"].sum()
    """
    is_wait = op_actions == LifecycleOp.WAIT
    is_germinate = op_actions == LifecycleOp.GERMINATE
    is_set_alpha = op_actions == LifecycleOp.SET_ALPHA_TARGET
    is_prune = op_actions == LifecycleOp.PRUNE

    # alpha_speed and alpha_curve share identical masks (SET_ALPHA_TARGET + PRUNE)
    # Share the reference to avoid redundant tensor allocation
    alpha_schedule_mask = is_set_alpha | is_prune

    return {
        "op": torch.ones_like(is_wait),           # Always causally relevant
        "slot": ~is_wait,                          # All non-WAIT ops
        "blueprint": is_germinate,                 # GERMINATE only
        "style": is_germinate | is_set_alpha,      # GERMINATE + SET_ALPHA_TARGET
        "tempo": is_germinate,                     # GERMINATE only
        "alpha_target": is_germinate | is_set_alpha,
        "alpha_speed": alpha_schedule_mask,        # SET_ALPHA_TARGET + PRUNE
        "alpha_curve": alpha_schedule_mask,        # SET_ALPHA_TARGET + PRUNE
    }


__all__ = ["compute_causal_masks"]
