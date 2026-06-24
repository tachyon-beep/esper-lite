"""Scripted policy smoke test for factored action wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel
from esper.leyline import (
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    MIN_PRUNE_AGE,
    SeedStage,
    SlotConfig,
    TempoAction,
)
from esper.simic.training.oracle_sandbox import (
    OracleActionResult,
    apply_oracle_factored_action,
)
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

    @property
    def topology(self) -> str:
        return "cnn"

    def execution_order(self) -> list[str]:
        return ["r0c1"]

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

        result = apply_oracle_factored_action(
            model,
            action,
            slot_config=slot_config,
            enabled_slots=enabled_slots,
            seed_counter=seed_counter,
        )
        assert isinstance(result, OracleActionResult)
        assert result.success
        seed_counter = result.seed_counter

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
        elif action.op == LifecycleOp.SET_ALPHA_TARGET:
            slot = model.seed_slots["r0c1"]
            slot.state.metrics.epochs_total = MIN_PRUNE_AGE
            slot.state.alpha_controller.alpha_mode = AlphaMode.HOLD
        elif action.op == LifecycleOp.PRUNE and action.alpha_speed_steps > 0:
            # Scheduled prune completes on subsequent step_epoch ticks.
            slot = model.seed_slots["r0c1"]
            for _ in range(action.alpha_speed_steps + 1):
                slot.step_epoch()

    assert not model.has_active_seed_in_slot("r0c1")
