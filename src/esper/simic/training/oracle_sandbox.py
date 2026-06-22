"""Oracle sandbox lifecycle action helpers.

The oracle sandbox runs declared factored actions through Kasmina's public
lifecycle APIs without invoking PPO sampling. Later proof steps add schedule,
telemetry, and packet surfaces on top of this source-owned helper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel
from esper.leyline import (
    AlphaMode,
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    InjectionSpec,
    LifecycleOp,
    SeedSlotProtocol,
    SeedStage,
    SlottedHostProtocol,
    SlotConfig,
    TEMPO_TO_EPOCHS,
    TempoAction,
)

ORACLE_CI_SLOT_ID = "r0c1"
ORACLE_CI_FIXTURE_ID = "oracle-ci-cnn-v1"
ORACLE_PROOF_PROFILE = "oracle-sandbox"


@dataclass(frozen=True, slots=True)
class OracleActionResult:
    """Outcome from applying one oracle-selected factored action."""

    success: bool
    seed_counter: int
    target_slot: str
    op: LifecycleOp


@dataclass(frozen=True, slots=True)
class OracleScheduleStep:
    """One declared oracle action and its expected execution outcome."""

    step_id: str
    action: FactoredAction
    expect_success: bool = True

    def __post_init__(self) -> None:
        if self.step_id == "":
            raise ValueError("oracle schedule step_id must be non-empty")
        if self.action.slot_idx < 0:
            raise ValueError("oracle schedule action slot_idx must be non-negative")

    def to_manifest(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "slot_idx": self.action.slot_idx,
            "blueprint": self.action.blueprint.name,
            "style": self.action.style.name,
            "tempo": self.action.tempo.name,
            "alpha_target": self.action.alpha_target.name,
            "alpha_speed": self.action.alpha_speed.name,
            "alpha_curve": self.action.alpha_curve.name,
            "op": self.action.op.name,
            "expect_success": self.expect_success,
        }


@dataclass(frozen=True, slots=True)
class OracleSchedule:
    """Deterministic oracle action schedule with stable provenance identity."""

    name: str
    steps: tuple[OracleScheduleStep, ...]
    fixture_id: str = ORACLE_CI_FIXTURE_ID
    proof_profile: str = ORACLE_PROOF_PROFILE

    def __post_init__(self) -> None:
        if self.name == "":
            raise ValueError("oracle schedule name must be non-empty")
        if len(self.steps) == 0:
            raise ValueError("oracle schedule requires at least one step")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("oracle schedule contains duplicate step_id values")

    @property
    def identity(self) -> str:
        payload = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.proof_profile}:{self.name}:{digest}"

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "proof_profile": self.proof_profile,
            "name": self.name,
            "fixture_id": self.fixture_id,
            "steps": tuple(step.to_manifest() for step in self.steps),
        }


@dataclass(frozen=True, slots=True)
class OracleSandboxFixture:
    """Tiny deterministic CPU-first fixture for oracle sandbox schedules."""

    fixture_id: str
    model: MorphogeneticModel
    slot_config: SlotConfig
    enabled_slots: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor


class OracleSandboxHost(nn.Module):
    """Minimal CNN host used by the oracle CI fixture."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
        self._slots: dict[str, nn.Module] = {}
        self.segment_channels = {ORACLE_CI_SLOT_ID: 16}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        if ORACLE_CI_SLOT_ID in self._slots and self._slots[ORACLE_CI_SLOT_ID] is not None:
            x = cast(torch.Tensor, self._slots[ORACLE_CI_SLOT_ID](x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return cast(torch.Tensor, self.fc(x))

    @property
    def injection_points(self) -> dict[str, int]:
        return {ORACLE_CI_SLOT_ID: 16}

    @property
    def topology(self) -> str:
        return "cnn"

    def execution_order(self) -> list[str]:
        return [ORACLE_CI_SLOT_ID]

    def injection_specs(self) -> list[InjectionSpec]:
        return [
            InjectionSpec(
                slot_id=ORACLE_CI_SLOT_ID,
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
        if segment == ORACLE_CI_SLOT_ID:
            return torch.relu(self.conv1(x))
        raise ValueError(f"Unknown segment: {segment}")

    def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        if segment == ORACLE_CI_SLOT_ID:
            if ORACLE_CI_SLOT_ID in self._slots and self._slots[ORACLE_CI_SLOT_ID] is not None:
                x = cast(torch.Tensor, self._slots[ORACLE_CI_SLOT_ID](x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return cast(torch.Tensor, self.fc(x))
        raise ValueError(f"Unknown segment: {segment}")


def _oracle_action(
    *,
    op: LifecycleOp,
    blueprint: BlueprintAction = BlueprintAction.NOOP,
    style: GerminationStyle = GerminationStyle.SIGMOID_ADD,
    tempo: TempoAction = TempoAction.STANDARD,
    alpha_target: AlphaTargetAction = AlphaTargetAction.FULL,
    alpha_speed: AlphaSpeedAction = AlphaSpeedAction.INSTANT,
    alpha_curve: AlphaCurveAction = AlphaCurveAction.LINEAR,
) -> FactoredAction:
    return FactoredAction(
        slot_idx=0,
        blueprint=blueprint,
        style=style,
        tempo=tempo,
        alpha_target=alpha_target,
        alpha_speed=alpha_speed,
        alpha_curve=alpha_curve,
        op=op,
    )


def build_default_oracle_schedule() -> OracleSchedule:
    """Build the stable CI lifecycle schedule for the oracle sandbox."""
    return OracleSchedule(
        name="oracle-ci-lifecycle-v1",
        steps=(
            OracleScheduleStep(
                step_id="germinate_norm",
                action=_oracle_action(
                    op=LifecycleOp.GERMINATE,
                    blueprint=BlueprintAction.NORM,
                    alpha_speed=AlphaSpeedAction.MEDIUM,
                ),
            ),
            OracleScheduleStep(
                step_id="advance_to_training",
                action=_oracle_action(op=LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="advance_to_blending",
                action=_oracle_action(op=LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="set_alpha_full",
                action=_oracle_action(
                    op=LifecycleOp.SET_ALPHA_TARGET,
                    alpha_target=AlphaTargetAction.FULL,
                    alpha_speed=AlphaSpeedAction.INSTANT,
                    alpha_curve=AlphaCurveAction.COSINE,
                ),
            ),
            OracleScheduleStep(
                step_id="advance_to_holding",
                action=_oracle_action(op=LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="fossilize_seed",
                action=_oracle_action(op=LifecycleOp.FOSSILIZE),
            ),
            OracleScheduleStep(
                step_id="reject_prune_after_fossilize",
                action=_oracle_action(
                    op=LifecycleOp.PRUNE,
                    alpha_speed=AlphaSpeedAction.SLOW,
                    alpha_curve=AlphaCurveAction.SIGMOID,
                ),
                expect_success=False,
            ),
        ),
    )


def build_oracle_ci_fixture(
    *,
    device: torch.device | str = "cpu",
) -> OracleSandboxFixture:
    """Build the deterministic CPU-first host/input fixture for oracle schedules."""
    device_obj = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        host = OracleSandboxHost()
    model = MorphogeneticModel(
        host=host,
        slots=[ORACLE_CI_SLOT_ID],
        device=str(device_obj),
    )
    inputs = torch.linspace(
        0.0,
        1.0,
        steps=2 * 3 * 8 * 8,
        device=device_obj,
    ).reshape(2, 3, 8, 8)
    targets = torch.tensor([0, 1], device=device_obj)
    return OracleSandboxFixture(
        fixture_id=ORACLE_CI_FIXTURE_ID,
        model=model,
        slot_config=SlotConfig(slot_ids=(ORACLE_CI_SLOT_ID,)),
        enabled_slots=(ORACLE_CI_SLOT_ID,),
        inputs=inputs,
        targets=targets,
    )


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
    "ORACLE_CI_FIXTURE_ID",
    "ORACLE_CI_SLOT_ID",
    "ORACLE_PROOF_PROFILE",
    "OracleActionResult",
    "OracleSandboxFixture",
    "OracleSandboxHost",
    "OracleSchedule",
    "OracleScheduleStep",
    "apply_oracle_factored_action",
    "build_default_oracle_schedule",
    "build_oracle_ci_fixture",
]
