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

from esper.leyline import ADVANTAGE_STD_FLOOR, MIN_HEAD_NORM_COUNT
from esper.leyline.causal_masks import compute_causal_masks

# Head order is fixed by compute_causal_masks (op first, then the causally-gated
# heads). We materialise it here so per-head advantage stats are emitted in a
# stable order regardless of dict-iteration details.
_HEAD_ORDER: tuple[str, ...] = (
    "op",
    "slot",
    "blueprint",
    "style",
    "tempo",
    "alpha_target",
    "alpha_speed",
    "alpha_curve",
)


def compute_per_head_advantages(
    base_advantages: torch.Tensor,
    op_actions: torch.Tensor,
    *,
    per_head_normalize: bool = False,
    min_head_norm_count: int = MIN_HEAD_NORM_COUNT,
    std_floor: float = ADVANTAGE_STD_FLOOR,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    """Compute advantages with causal masking per head.

    ``base_advantages`` is the GLOBALLY-normalized GAE advantage (one mean/std over
    every valid step, dominated by the always-active ``op`` head). Under a shared
    critic V(s) the advantage is the SAME scalar for all heads on a given step, so a
    single global scale denominates every head in op's variance units — a sparse head
    (blueprint/tempo active ~18% of steps, style/alpha_* ~22%) inherits op's scale and
    its genuine signal is attenuated. ``per_head_normalize`` re-standardizes each
    head's advantage over its OWN causally-active subset, decoupling the per-head
    gradient scale from op's dominance.

    Args:
        base_advantages: GAE advantages [batch] or [batch, seq], already globally
            normalized by ``TamiyoRolloutBuffer.normalize_advantages``.
        op_actions: Operation actions [batch] or [batch, seq] (LifecycleOp values).
        per_head_normalize: When True, restandardize each head over its active subset
            (gated by ``min_head_norm_count``). When False (default) the historical
            global-then-mask behaviour is preserved EXACTLY.
        min_head_norm_count: A head with fewer causally-active steps than this falls
            back to the global scale — a per-head std over a handful of samples is pure
            noise, and that is exactly the gradient-amplification the global D4 floor
            guards against (here at per-head granularity). See ``MIN_HEAD_NORM_COUNT``.
        std_floor: Per-head std clamp (mirrors the global ``ADVANTAGE_STD_FLOOR``) so a
            within-subset variance collapse cannot amplify noise to unit variance.

    Returns:
        Tuple of (per_head_advantages, masks, stats):
        - per_head_advantages: Dict with per-head advantages, causally masked.
        - masks: Dict with boolean causal masks (for reuse in KL/entropy/loss).
        - stats: Dict head -> {pre_norm_std, post_norm_std, n_active, fellback}, where
          ``pre_norm_std`` is the head's advantage std over its active subset BEFORE any
          per-head renormalization (the observability signal for the
          exploration→return-variance loop) and ``fellback`` records whether the
          low-count guard sent this head back to the global scale. Stats are ALWAYS
          computed, even when ``per_head_normalize`` is False, so the loop is observable
          before the behaviour change is enabled.

    Note:
        Boolean masks are multiplied directly (no .float() conversion) to preserve
        base_advantages.dtype under AMP (float16/bfloat16). Per-head statistics are
        computed in float32 to avoid fp16 variance underflow on the sparse subsets.
    """
    # B4-DRL-01: Use single source of truth for causal masks
    masks = compute_causal_masks(op_actions)

    # Per-head stats are computed in fp32: std() over an 18-22% fp16 subset of
    # near-zero-mean values is variance-underflow-prone (squared deviations underflow
    # to 0 before the reduction). Detach — these are diagnostics / scale factors, not
    # a gradient path.
    base_fp32 = base_advantages.detach().to(torch.float32)

    stats: dict[str, dict[str, float]] = {}
    per_head_advantages: dict[str, torch.Tensor] = {}

    for head in _HEAD_ORDER:
        mask = masks[head]
        active = base_fp32[mask.bool()]
        n_active = int(active.numel())

        if n_active >= 2:
            head_mean = active.mean()
            head_std = active.std(correction=0)
        else:
            # Single- or zero-active-step head: std is undefined/0. Never divide by it.
            head_mean = base_fp32.new_zeros(())
            head_std = base_fp32.new_zeros(())

        renormalized = False
        if per_head_normalize and n_active >= min_head_norm_count:
            # Standardize over THIS head's active subset, with the same floor
            # philosophy as the global D4 clamp, then re-apply the boolean mask last
            # to preserve the AMP-safe zeroing of causally-irrelevant steps.
            effective_std = head_std.clamp(min=std_floor)
            normed = (base_fp32 - head_mean) / (effective_std + 1e-8)
            head_adv = normed.to(base_advantages.dtype) * mask
            renormalized = True
        elif head == "op":
            # M8: op is always causally relevant (mask all-True); hand back the base
            # tensor directly (no allocation). Caller only ever reads it.
            head_adv = base_advantages
        else:
            # Historical path: global-normalized advantage, masked to active steps.
            head_adv = base_advantages * mask

        if renormalized and n_active >= 2:
            post_std = float(
                head_adv.detach().to(torch.float32)[mask.bool()].std(correction=0).item()
            )
        else:
            # No renormalization: post-norm std equals the (global-scale) pre-norm std.
            post_std = float(head_std.item())

        per_head_advantages[head] = head_adv
        stats[head] = {
            "pre_norm_std": float(head_std.item()),
            "post_norm_std": post_std,
            "n_active": float(n_active),
            "fellback": float(per_head_normalize and not renormalized),
            "normalized": float(renormalized),
        }

    return per_head_advantages, masks, stats


__all__ = ["compute_per_head_advantages"]
