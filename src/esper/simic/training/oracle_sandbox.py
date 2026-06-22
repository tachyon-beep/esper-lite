"""Oracle sandbox lifecycle action helpers.

The oracle sandbox runs declared factored actions through Kasmina's public
lifecycle APIs without invoking PPO sampling. Later proof steps add schedule,
telemetry, and packet surfaces on top of this source-owned helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from esper.leyline import (
    AlphaMode,
    FactoredAction,
    LifecycleOp,
    SeedSlotProtocol,
    SeedStage,
    SlottedHostProtocol,
    SlotConfig,
    TEMPO_TO_EPOCHS,
)


@dataclass(frozen=True, slots=True)
class OracleActionResult:
    """Outcome from applying one oracle-selected factored action."""

    success: bool
    seed_counter: int
    target_slot: str
    op: LifecycleOp


def _resolve_oracle_target_slot(
    slot_idx: int,
    *,
    enabled_slots: list[str],
    slot_config: SlotConfig,
) -> tuple[str, bool]:
    try:
        slot_id = slot_config.slot_id_for_index(slot_idx)
    except IndexError:
        return enabled_slots[0], False
    return slot_id, slot_id in enabled_slots


def apply_oracle_factored_action(
    model: SlottedHostProtocol,
    action: FactoredAction,
    *,
    slot_config: SlotConfig,
    enabled_slots: list[str],
    seed_counter: int,
) -> OracleActionResult:
    """Apply one oracle-selected factored action through lifecycle APIs."""
    target_slot, slot_is_enabled = _resolve_oracle_target_slot(
        action.slot_idx,
        enabled_slots=enabled_slots,
        slot_config=slot_config,
    )
    if not slot_is_enabled:
        return OracleActionResult(False, seed_counter, target_slot, action.op)

    if action.op == LifecycleOp.GERMINATE:
        tempo_epochs = TEMPO_TO_EPOCHS[action.tempo]
        seed_id = f"oracle_seed_{seed_counter}"
        model.germinate_seed(
            action.blueprint.to_blueprint_id(),
            seed_id,
            slot=target_slot,
            blend_algorithm_id=action.blend_algorithm_id,
            blend_tempo_epochs=tempo_epochs,
            alpha_algorithm=action.alpha_algorithm_value,
            alpha_target=action.alpha_target_value,
        )
        return OracleActionResult(True, seed_counter + 1, target_slot, action.op)

    if action.op == LifecycleOp.FOSSILIZE:
        slot = cast(SeedSlotProtocol, model.seed_slots[target_slot])
        gate = slot.advance_stage(SeedStage.FOSSILIZED)
        return OracleActionResult(gate.passed, seed_counter, target_slot, action.op)

    if action.op == LifecycleOp.PRUNE:
        if model.has_active_seed_in_slot(target_slot):
            slot = cast(SeedSlotProtocol, model.seed_slots[target_slot])
            slot_state = slot.state
            if (
                slot_state is None
                or slot_state.alpha_controller.alpha_mode != AlphaMode.HOLD
                or not slot_state.can_transition_to(SeedStage.PRUNED)
            ):
                return OracleActionResult(False, seed_counter, target_slot, action.op)
            speed_steps = action.alpha_speed_steps
            curve = action.alpha_curve_value
            scheduled = slot.schedule_prune(
                steps=speed_steps,
                curve=curve,
                initiator="oracle",
            )
            return OracleActionResult(scheduled, seed_counter, target_slot, action.op)
        return OracleActionResult(False, seed_counter, target_slot, action.op)

    if action.op == LifecycleOp.SET_ALPHA_TARGET:
        if model.has_active_seed_in_slot(target_slot):
            slot = cast(SeedSlotProtocol, model.seed_slots[target_slot])
            updated = slot.set_alpha_target(
                alpha_target=action.alpha_target_value,
                steps=action.alpha_speed_steps,
                curve=action.alpha_curve_value,
                alpha_algorithm=action.alpha_algorithm_value,
                initiator="oracle",
            )
            return OracleActionResult(updated, seed_counter, target_slot, action.op)
        return OracleActionResult(False, seed_counter, target_slot, action.op)

    return OracleActionResult(True, seed_counter, target_slot, action.op)


__all__ = [
    "OracleActionResult",
    "apply_oracle_factored_action",
]
