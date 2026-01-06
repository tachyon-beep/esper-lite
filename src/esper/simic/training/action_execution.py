from __future__ import annotations

from typing import Any

from esper.leyline import (
    ALPHA_TARGET_VALUES,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    AlphaMode,
    LifecycleOp,
    SeedStage,
    SlotConfig,
    SlottedHostProtocol,
    OP_ADVANCE,
    OP_FOSSILIZE,
    OP_GERMINATE,
    OP_PRUNE,
    OP_SET_ALPHA_TARGET,
    MIN_PRUNE_AGE,
)

from .vectorized_helpers import _resolve_target_slot


def parse_sampled_action(
    env_idx: int,
    op_idx: int,
    slot_idx: int,
    style_idx: int,
    alpha_target_idx: int,
    slots: list[str],
    slot_config: SlotConfig,
    model: SlottedHostProtocol,
) -> tuple[str, bool, Any, Any, bool, LifecycleOp, str, Any, float]:
    """Consolidate action derived values and validation logic (Deduplication)."""
    target_slot, slot_is_enabled = _resolve_target_slot(
        slot_idx, enabled_slots=slots, slot_config=slot_config
    )

    slot_state = model.seed_slots[target_slot].state if slot_is_enabled else None
    seed_state = (
        slot_state if slot_is_enabled and model.has_active_seed_in_slot(target_slot) else None
    )

    blend_algorithm_id = STYLE_BLEND_IDS[style_idx]
    alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx]
    alpha_target = ALPHA_TARGET_VALUES[alpha_target_idx]

    action_valid_for_reward = True
    if not slot_is_enabled:
        action_valid_for_reward = False
    elif op_idx == OP_GERMINATE:
        action_valid_for_reward = slot_state is None
    elif op_idx == OP_FOSSILIZE:
        action_valid_for_reward = (
            seed_state is not None and seed_state.stage == SeedStage.HOLDING
        )
    elif op_idx == OP_PRUNE:
        action_valid_for_reward = (
            seed_state is not None
            and seed_state.stage in (
                SeedStage.GERMINATED,
                SeedStage.TRAINING,
                SeedStage.BLENDING,
                SeedStage.HOLDING,
            )
            and seed_state.alpha_controller.alpha_mode == AlphaMode.HOLD
            and seed_state.can_transition_to(SeedStage.PRUNED)
            and seed_state.metrics is not None
            and seed_state.metrics.epochs_total >= MIN_PRUNE_AGE
        )
    elif op_idx == OP_SET_ALPHA_TARGET:
        action_valid_for_reward = (
            seed_state is not None
            and seed_state.alpha_controller.alpha_mode == AlphaMode.HOLD
            and seed_state.stage in (SeedStage.BLENDING, SeedStage.HOLDING)
        )
    elif op_idx == OP_ADVANCE:
        action_valid_for_reward = seed_state is not None and seed_state.stage in (
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
        )

    action_for_reward = LifecycleOp(op_idx) if action_valid_for_reward else LifecycleOp.WAIT

    return (
        target_slot,
        slot_is_enabled,
        slot_state,
        seed_state,
        action_valid_for_reward,
        action_for_reward,
        blend_algorithm_id,
        alpha_algorithm,
        alpha_target,
    )
