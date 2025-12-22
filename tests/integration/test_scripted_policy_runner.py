"""Scripted policy smoke test for factored action wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel
from esper.leyline.factored_actions import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    TempoAction,
    TEMPO_TO_EPOCHS,
)
from esper.leyline.slot_config import SlotConfig
from esper.leyline.alpha import AlphaMode
from esper.leyline.stages import SeedStage
from esper.simic.training.vectorized import _resolve_target_slot
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks


class ScriptedHost(nn.Module):
    """Minimal CNN host model for scripted policy smoke tests."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
        self._slots: dict[str, nn.Module] = {}
        self.segment_channels = {"r0c1": 16}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        if "r0c1" in self._slots and self._slots["r0c1"] is not None:
            x = self._slots["r0c1"](x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @property
    def injection_points(self) -> dict[str, int]:
        return {"r0c1": 16}

    def injection_specs(self):
        from esper.leyline import InjectionSpec
        return [
            InjectionSpec(
                slot_id="r0c1",
                channels=16,
                position=0.5,
                layer_range=(0, 1),
            )
        ]

    def forward_to_segment(
        self,
        segment: str,
        x: torch.Tensor,
        from_segment: str | None = None,
    ) -> torch.Tensor:
        if segment == "r0c1":
            return torch.relu(self.conv1(x))
        raise ValueError(f"Unknown segment: {segment}")

    def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        if segment == "r0c1":
            if "r0c1" in self._slots and self._slots["r0c1"] is not None:
                x = self._slots["r0c1"](x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
        raise ValueError(f"Unknown segment: {segment}")


def _apply_factored_action(
    model: MorphogeneticModel,
    action: FactoredAction,
    *,
    slot_config: SlotConfig,
    enabled_slots: list[str],
    seed_counter: int,
) -> tuple[bool, int]:
    target_slot, slot_is_enabled = _resolve_target_slot(
        action.slot_idx,
        enabled_slots=enabled_slots,
        slot_config=slot_config,
    )
    if not slot_is_enabled:
        return False, seed_counter

    if action.op == LifecycleOp.GERMINATE:
        tempo_epochs = TEMPO_TO_EPOCHS[action.tempo]
        seed_id = f"scripted_seed_{seed_counter}"
        model.germinate_seed(
            action.blueprint_id or "norm",
            seed_id,
            slot=target_slot,
            blend_algorithm_id=action.blend_algorithm_id,
            blend_tempo_epochs=tempo_epochs,
            alpha_algorithm=action.alpha_algorithm_value,
            alpha_target=action.alpha_target_value,
        )
        return True, seed_counter + 1

    if action.op == LifecycleOp.FOSSILIZE:
        gate = model.seed_slots[target_slot].advance_stage(SeedStage.FOSSILIZED)
        return gate.passed, seed_counter

    if action.op == LifecycleOp.PRUNE:
        if model.has_active_seed_in_slot(target_slot):
            slot = model.seed_slots[target_slot]
            slot_state = slot.state
            if (
                slot_state is None
                or slot_state.alpha_controller.alpha_mode != AlphaMode.HOLD
                or not slot_state.can_transition_to(SeedStage.PRUNED)
            ):
                return False, seed_counter
            speed_steps = action.alpha_speed_steps
            curve = action.alpha_curve_value
            if speed_steps <= 0:
                pruned = slot.prune(reason="policy_prune", initiator="scripted")
                return pruned, seed_counter
            scheduled = slot.schedule_prune(
                steps=speed_steps,
                curve=curve,
                initiator="scripted",
            )
            return scheduled, seed_counter
        return False, seed_counter

    if action.op == LifecycleOp.SET_ALPHA_TARGET:
        if model.has_active_seed_in_slot(target_slot):
            updated = model.seed_slots[target_slot].set_alpha_target(
                alpha_target=action.alpha_target_value,
                steps=action.alpha_speed_steps,
                curve=action.alpha_curve_value,
                alpha_algorithm=action.alpha_algorithm_value,
                initiator="scripted",
            )
            return updated, seed_counter
        return False, seed_counter

    return True, seed_counter


def test_scripted_policy_runner_smoke() -> None:
    """Scripted action sequence should execute without PPO training."""
    model = MorphogeneticModel(host=ScriptedHost(), slots=["r0c1"], device="cpu")
    slot_config = SlotConfig(slot_ids=("r0c1",))
    enabled_slots = ["r0c1"]

    actions = [
        FactoredAction(
            slot_idx=0,
            blueprint=BlueprintAction.NORM,
            style=GerminationStyle.SIGMOID_ADD,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.FULL,
            alpha_speed=AlphaSpeedAction.MEDIUM,
            alpha_curve=AlphaCurveAction.LINEAR,
            op=LifecycleOp.GERMINATE,
        ),
        FactoredAction(
            slot_idx=0,
            blueprint=BlueprintAction.NOOP,
            style=GerminationStyle.SIGMOID_ADD,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.SEVENTY,
            alpha_speed=AlphaSpeedAction.MEDIUM,
            alpha_curve=AlphaCurveAction.COSINE,
            op=LifecycleOp.WAIT,
        ),
        FactoredAction(
            slot_idx=0,
            blueprint=BlueprintAction.NOOP,
            style=GerminationStyle.SIGMOID_ADD,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.FULL,
            alpha_speed=AlphaSpeedAction.INSTANT,
            alpha_curve=AlphaCurveAction.COSINE,
            op=LifecycleOp.SET_ALPHA_TARGET,
        ),
        FactoredAction(
            slot_idx=0,
            blueprint=BlueprintAction.NOOP,
            style=GerminationStyle.SIGMOID_ADD,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.FULL,
            alpha_speed=AlphaSpeedAction.SLOW,
            alpha_curve=AlphaCurveAction.SIGMOID,
            op=LifecycleOp.PRUNE,
        ),
    ]

    seed_counter = 0
    for action in actions:
        slot_reports = model.get_slot_reports()
        slot_states = build_slot_states(slot_reports, enabled_slots)
        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled_slots,
            slot_config=slot_config,
        )
        assert masks["slot"][action.slot_idx].item() is True
        assert masks["op"][action.op.value].item() is True
        if action.op == LifecycleOp.GERMINATE:
            assert masks["blueprint"][action.blueprint.value].item() is True
            assert masks["style"][action.style.value].item() is True
            assert masks["tempo"][action.tempo.value].item() is True
        if action.op in (LifecycleOp.SET_ALPHA_TARGET, LifecycleOp.PRUNE):
            assert masks["alpha_target"][action.alpha_target.value].item() is True
            assert masks["alpha_speed"][action.alpha_speed.value].item() is True
            assert masks["alpha_curve"][action.alpha_curve.value].item() is True

        success, seed_counter = _apply_factored_action(
            model,
            action,
            slot_config=slot_config,
            enabled_slots=enabled_slots,
            seed_counter=seed_counter,
        )
        assert success

        if action.op == LifecycleOp.GERMINATE:
            slot = model.seed_slots["r0c1"]
            slot.state.metrics.record_accuracy(0.0)
            gate_result = slot.advance_stage(SeedStage.TRAINING)
            assert gate_result.passed
            # Force BLENDING + HOLD to exercise SET_ALPHA_TARGET.
            slot.state.transition(SeedStage.BLENDING)
            slot._on_enter_stage(SeedStage.BLENDING, SeedStage.TRAINING)
            slot.set_alpha(0.7)
            slot.state.alpha_controller.alpha_target = 0.7
            slot.state.alpha_controller.alpha_mode = AlphaMode.HOLD
        elif action.op == LifecycleOp.PRUNE and action.alpha_speed_steps > 0:
            # Scheduled prune completes on subsequent step_epoch ticks.
            slot = model.seed_slots["r0c1"]
            for _ in range(action.alpha_speed_steps + 1):
                slot.step_epoch()

    assert not model.has_active_seed_in_slot("r0c1")
