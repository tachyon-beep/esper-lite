"""Oracle sandbox lifecycle action helpers.

The oracle sandbox runs declared factored actions through Kasmina's public
lifecycle APIs without invoking PPO sampling. Later proof steps add schedule,
telemetry, and packet surfaces on top of this source-owned helper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel
from esper.leyline import (
    AlphaMode,
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    DEFAULT_MIN_BLENDING_EPOCHS,
    EpisodeOutcomePayload,
    FactoredAction,
    GerminationStyle,
    InjectionSpec,
    LifecycleOp,
    MorphologyCausalLogPayload,
    MorphologyCausalLogPhase,
    SeedSlotProtocol,
    SeedStage,
    SlottedHostProtocol,
    SlotConfig,
    TelemetryEvent,
    TelemetryEventType,
    TempoAction,
    TrainingStartedPayload,
)
from esper.nissa.output import DirectoryOutput
from esper.simic.rewards import SeedInfo
from esper.simic.training.handlers import (
    AlphaTargetParams,
    GerminateParams,
    HandlerContext,
    PruneParams,
    get_handler,
)
from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.simic.training.vectorized import _resolve_target_slot
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks
from esper.tolaria import TolariaGovernor

ORACLE_CI_SLOT_ID = "r0c1"
ORACLE_CI_FIXTURE_ID = "oracle-ci-cnn-v1"
ORACLE_PROOF_PROFILE = "oracle-sandbox"


@dataclass(frozen=True, slots=True)
class OracleStepResult:
    """Trace row for one oracle schedule step."""

    step_id: str
    op: LifecycleOp
    expected_success: bool
    success: bool
    stage_before: str
    stage_after: str
    mask_checked: bool
    governor_checked: bool
    governor_approved: bool | None
    handler_called: bool
    blocked_by: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OracleRunResult:
    """Result for a complete oracle schedule execution."""

    schedule_identity: str
    success: bool
    steps: tuple[OracleStepResult, ...]
    telemetry_dir: str | None = None


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
    device: torch.device


class OracleSandboxHost(nn.Module):
    """Minimal CNN host used by the oracle CI fixture."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
        self.segment_channels = {ORACLE_CI_SLOT_ID: 16}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone only: the MorphogeneticModel applies seeds via seed_slots, the host never
        # does (matching CNNHost, whose forward is "no slot application").
        x = torch.relu(self.conv1(x))
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
        # Tail only: the seed has already been applied by the model at the segment boundary.
        if segment == ORACLE_CI_SLOT_ID:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return cast(torch.Tensor, self.fc(x))
        raise ValueError(f"Unknown segment: {segment}")


class _OracleSignalTracker:
    def reset(self) -> None:
        return


def _oracle_schedule_hash(schedule: OracleSchedule) -> str:
    return schedule.identity[-16:]


def _oracle_action_id(
    *,
    schedule_step: OracleScheduleStep,
    target_slot: str,
    epoch: int,
) -> str:
    return (
        f"oracle-b0-e{epoch}-env0-{target_slot}-op"
        f"{schedule_step.action.op.value}-{schedule_step.step_id}"
    )


def _oracle_observation_hash(schedule_step: OracleScheduleStep) -> str:
    payload = json.dumps(
        schedule_step.to_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"obs-{digest}"


def _oracle_rng_seed(action_id: str) -> int:
    return int(hashlib.sha256(action_id.encode("utf-8")).hexdigest()[:16], 16)


def _emit_oracle_training_started(
    *,
    telemetry_output: DirectoryOutput,
    schedule: OracleSchedule,
    fixture: OracleSandboxFixture,
    host_params: int,
    max_epochs: int,
) -> None:
    device = fixture.device.type
    telemetry_output.emit(TelemetryEvent(
        event_id=f"{schedule.identity}:training-start",
        event_type=TelemetryEventType.TRAINING_STARTED,
        group_id=ORACLE_PROOF_PROFILE,
        message="Oracle sandbox training started",
        data=TrainingStartedPayload(
            n_envs=1,
            max_epochs=max_epochs,
            max_batches=1,
            task=fixture.fixture_id,
            host_params=host_params,
            slot_ids=fixture.enabled_slots,
            seed=0,
            n_episodes=1,
            lr=0.0,
            clip_ratio=0.0,
            entropy_coef=0.0,
            param_budget=host_params,
            policy_device=device,
            env_devices=(device,),
            reward_mode="oracle",
            episode_id=schedule.identity,
            proof_profile=ORACLE_PROOF_PROFILE,
            proof_baseline_mode=ORACLE_PROOF_PROFILE,
            proof_baseline_pair_id=schedule.identity,
            proof_baseline_lifecycle_policy="oracle_schedule",
            proof_baseline_schedule_id=schedule.identity,
            proof_baseline_schedule_hash=_oracle_schedule_hash(schedule),
            proof_baseline_schedule_version=1,
            proof_baseline_schedule_action_count=len(schedule.steps),
        ),
    ))


def _emit_oracle_causal_event(
    *,
    telemetry_output: DirectoryOutput,
    schedule_step: OracleScheduleStep,
    target_slot: str,
    epoch: int,
    phase: MorphologyCausalLogPhase,
    governor_approved: bool | None,
    governor_reason: str | None,
    governor_blocked_factor: str | None,
    linked_event_id: str | None,
    message: str,
) -> None:
    action = schedule_step.action
    action_id = _oracle_action_id(
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
    )
    mutation_id = f"{action_id}-mutation"
    blueprint_id = (
        action.blueprint.to_blueprint_id()
        if action.op == LifecycleOp.GERMINATE
        else None
    )
    telemetry_output.emit(TelemetryEvent(
        event_id=f"{action_id}:{phase}",
        event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
        slot_id=target_slot,
        epoch=epoch,
        group_id=ORACLE_PROOF_PROFILE,
        message=message,
        severity="debug",
        data=MorphologyCausalLogPayload(
            phase=phase,
            env_id=0,
            slot_id=target_slot,
            operation=action.op.name,
            action_id=action_id,
            proposal_id=f"{action_id}-proposal",
            verdict_id=f"{action_id}-verdict",
            mutation_id=mutation_id,
            observation_hash=_oracle_observation_hash(schedule_step),
            rng_stream="oracle_sandbox.schedule",
            rng_seed=_oracle_rng_seed(action_id),
            topology="cnn",
            blueprint_id=blueprint_id,
            governor_approved=governor_approved,
            governor_reason=governor_reason,
            governor_blocked_factor=governor_blocked_factor,
            watch_window_evidence=None,
            linked_event_id=linked_event_id,
        ),
    ))


def _emit_oracle_step_telemetry(
    *,
    telemetry_output: DirectoryOutput,
    schedule_step: OracleScheduleStep,
    step_result: OracleStepResult,
    target_slot: str,
    epoch: int,
) -> None:
    action_id = _oracle_action_id(
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
    )
    mutation_id = f"{action_id}-mutation"
    blocked_factor = step_result.blocked_by
    governor_approved = step_result.governor_approved
    governor_reason = (
        "approved"
        if governor_approved is True
        else step_result.detail or blocked_factor
    )
    if blocked_factor == "mask":
        governor_approved = False
        governor_reason = "action masked"

    _emit_oracle_causal_event(
        telemetry_output=telemetry_output,
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
        phase="proposal",
        governor_approved=None,
        governor_reason=None,
        governor_blocked_factor=None,
        linked_event_id=None,
        message="Oracle morphology proposal",
    )
    _emit_oracle_causal_event(
        telemetry_output=telemetry_output,
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
        phase="verdict",
        governor_approved=governor_approved,
        governor_reason=governor_reason,
        governor_blocked_factor=blocked_factor,
        linked_event_id=f"{action_id}-verdict",
        message="Oracle morphology verdict",
    )
    if not step_result.handler_called:
        _emit_oracle_causal_event(
            telemetry_output=telemetry_output,
            schedule_step=schedule_step,
            target_slot=target_slot,
            epoch=epoch,
            phase="audit",
            governor_approved=governor_approved,
            governor_reason=governor_reason,
            governor_blocked_factor=blocked_factor,
            linked_event_id=f"{action_id}-verdict",
            message="Oracle morphology audit",
        )
        return

    _emit_oracle_causal_event(
        telemetry_output=telemetry_output,
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
        phase="mutation",
        governor_approved=True,
        governor_reason="approved",
        governor_blocked_factor=None,
        linked_event_id=mutation_id,
        message="Oracle morphology mutation dispatch",
    )
    _emit_oracle_causal_event(
        telemetry_output=telemetry_output,
        schedule_step=schedule_step,
        target_slot=target_slot,
        epoch=epoch,
        phase="dispatch",
        governor_approved=True,
        governor_reason="approved",
        governor_blocked_factor=None,
        linked_event_id=mutation_id,
        message="Oracle morphology mutation dispatched",
    )
    if step_result.success:
        terminal_phase: MorphologyCausalLogPhase = (
            "fossilization"
            if schedule_step.action.op == LifecycleOp.FOSSILIZE
            else "commit"
        )
        _emit_oracle_causal_event(
            telemetry_output=telemetry_output,
            schedule_step=schedule_step,
            target_slot=target_slot,
            epoch=epoch,
            phase=terminal_phase,
            governor_approved=True,
            governor_reason="approved",
            governor_blocked_factor=None,
            linked_event_id=mutation_id,
            message="Oracle morphology dispatch committed",
        )


def _emit_oracle_policy_evidence(
    *,
    telemetry_output: DirectoryOutput,
    schedule: OracleSchedule,
) -> None:
    action_id = f"oracle-policy-{_oracle_schedule_hash(schedule)}"
    telemetry_output.emit(TelemetryEvent(
        event_id=f"{action_id}:audit",
        event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
        slot_id=ORACLE_CI_SLOT_ID,
        epoch=0,
        group_id=ORACLE_PROOF_PROFILE,
        message="Oracle policy evidence",
        severity="debug",
        data=MorphologyCausalLogPayload(
            phase="audit",
            env_id=0,
            slot_id=ORACLE_CI_SLOT_ID,
            operation="ORACLE_POLICY",
            action_id=action_id,
            proposal_id=f"{action_id}-proposal",
            verdict_id=f"{action_id}-verdict",
            mutation_id=f"{action_id}-mutation",
            observation_hash=f"obs-{_oracle_schedule_hash(schedule)}",
            rng_stream="oracle_sandbox.policy",
            rng_seed=_oracle_rng_seed(action_id),
            topology="cnn",
            blueprint_id=None,
            governor_approved=True,
            governor_reason="schedule_declared",
            governor_blocked_factor=None,
            watch_window_evidence=None,
            linked_event_id=schedule.identity,
        ),
    ))


def _emit_oracle_episode_outcome(
    *,
    telemetry_output: DirectoryOutput,
    schedule: OracleSchedule,
    fixture: OracleSandboxFixture,
    step_results: tuple[OracleStepResult, ...],
    host_params: int,
    val_accuracy: float,
    run_success: bool,
    num_fossilized: int,
    num_contributing_fossilized: int,
) -> None:
    successful_steps = tuple(step for step in step_results if step.success)
    telemetry_output.emit(TelemetryEvent(
        event_id=f"{schedule.identity}:episode-outcome",
        event_type=TelemetryEventType.EPISODE_OUTCOME,
        epoch=len(step_results),
        group_id=ORACLE_PROOF_PROFILE,
        message="Oracle sandbox episode outcome",
        data=EpisodeOutcomePayload(
            env_id=0,
            episode_idx=0,
            final_accuracy=val_accuracy,
            param_ratio=fixture.model.total_params / max(1, host_params),
            num_fossilized=num_fossilized,
            num_contributing_fossilized=num_contributing_fossilized,
            episode_reward=float(len(successful_steps)),
            stability_score=1.0 if run_success else 0.0,
            reward_mode="oracle",
            episode_length=len(step_results),
            outcome_type="success" if run_success else "blocked",
            germinate_count=sum(
                1 for step in successful_steps if step.op == LifecycleOp.GERMINATE
            ),
            prune_count=sum(
                1 for step in successful_steps if step.op == LifecycleOp.PRUNE
            ),
            fossilize_count=sum(
                1 for step in successful_steps if step.op == LifecycleOp.FOSSILIZE
            ),
        ),
    ))


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
    return OracleSandboxFixture(
        fixture_id=ORACLE_CI_FIXTURE_ID,
        model=model,
        slot_config=SlotConfig(slot_ids=(ORACLE_CI_SLOT_ID,)),
        enabled_slots=(ORACLE_CI_SLOT_ID,),
        device=device_obj,
    )


def _slot_for(model: SlottedHostProtocol, slot_id: str) -> SeedSlotProtocol:
    return cast(SeedSlotProtocol, model.seed_slots[slot_id])


def _slot_stage_name(model: SlottedHostProtocol, slot_id: str) -> str:
    slot = _slot_for(model, slot_id)
    state = slot.state
    if state is None:
        return "DORMANT"
    return state.stage.name


def _settle_oracle_step_evidence(
    *,
    fixture: OracleSandboxFixture,
    target_slot: str,
    val_accuracy: float,
    max_epochs: int,
) -> None:
    slot = _slot_for(fixture.model, target_slot)
    state = slot.state
    if state is None:
        return

    if state.stage == SeedStage.TRAINING:
        state.metrics.record_accuracy(val_accuracy - 1.0)
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS - 1):
            state.metrics.record_accuracy(val_accuracy)
        state.metrics.seed_gradient_norm_ratio = 1.0
        state.metrics.counterfactual_contribution = max(
            state.metrics.counterfactual_contribution or 0.0,
            2.0,
        )
        state.sync_telemetry(
            gradient_norm=1.0,
            gradient_health=1.0,
            has_vanishing=False,
            has_exploding=False,
            epoch=state.metrics.epochs_total,
            max_epochs=max_epochs,
        )

    if state.stage == SeedStage.BLENDING:
        while (
            state.stage == SeedStage.BLENDING
            and state.alpha_controller.alpha_mode != AlphaMode.HOLD
        ):
            state.metrics.record_accuracy(val_accuracy)
            state.sync_telemetry(
                gradient_norm=1.0,
                gradient_health=1.0,
                has_vanishing=False,
                has_exploding=False,
                epoch=state.metrics.epochs_total,
                max_epochs=max_epochs,
            )
            slot.step_epoch()

    if state.stage == SeedStage.BLENDING:
        while state.metrics.epochs_in_current_stage < DEFAULT_MIN_BLENDING_EPOCHS:
            state.metrics.record_accuracy(val_accuracy)
            state.sync_telemetry(
                gradient_norm=1.0,
                gradient_health=1.0,
                has_vanishing=False,
                has_exploding=False,
                epoch=state.metrics.epochs_total,
                max_epochs=max_epochs,
            )

    if state.stage == SeedStage.HOLDING:
        state.metrics.counterfactual_contribution = 2.0
        state.metrics.record_accuracy(val_accuracy)
        state.sync_telemetry(
            gradient_norm=1.0,
            gradient_health=1.0,
            has_vanishing=False,
            has_exploding=False,
            epoch=state.metrics.epochs_total,
            max_epochs=max_epochs,
        )


def _action_allowed_by_masks(
    *,
    action: FactoredAction,
    masks: dict[str, torch.Tensor],
    slot_is_enabled: bool,
) -> bool:
    if not slot_is_enabled:
        return False
    if not bool(masks["slot"][action.slot_idx]):
        return False
    if not bool(masks["op"][action.op.value]):
        return False
    if not bool(masks["slot_by_op"][action.op.value, action.slot_idx]):
        return False
    if action.op == LifecycleOp.GERMINATE:
        return (
            bool(masks["blueprint"][action.blueprint.value])
            and bool(masks["style"][action.style.value])
            and bool(masks["tempo"][action.tempo.value])
        )
    if action.op in (LifecycleOp.SET_ALPHA_TARGET, LifecycleOp.PRUNE):
        return (
            bool(masks["style"][action.style.value])
            and bool(masks["alpha_target"][action.alpha_target.value])
            and bool(masks["alpha_speed"][action.alpha_speed.value])
            and bool(masks["alpha_curve"][action.alpha_curve.value])
        )
    return True


def _build_oracle_env_state(
    *,
    fixture: OracleSandboxFixture,
    val_loss: float,
    val_accuracy: float,
) -> ParallelEnvState:
    host_optimizer = torch.optim.SGD(fixture.model.parameters(), lr=0.0)
    governor = TolariaGovernor(
        model=fixture.model,
        min_panics_before_rollback=1,
    )
    env_state = ParallelEnvState(
        model=fixture.model,
        host_optimizer=host_optimizer,
        signal_tracker=cast(Any, _OracleSignalTracker()),
        governor=governor,
        env_device=fixture.device.type,
    )
    env_state.val_loss = val_loss
    env_state.val_acc = val_accuracy
    env_state.committed_val_acc = val_accuracy
    env_state.escrow_credit = {slot_id: 0.0 for slot_id in fixture.enabled_slots}
    env_state.prev_slot_alphas = {slot_id: 0.0 for slot_id in fixture.enabled_slots}
    env_state.prev_slot_params = {slot_id: 0 for slot_id in fixture.enabled_slots}
    return env_state


def _oracle_seed_info(
    *,
    slot: SeedSlotProtocol,
) -> SeedInfo | None:
    seed_state = slot.state
    contribution = (
        seed_state.metrics.counterfactual_contribution
        if seed_state is not None
        else None
    )
    return SeedInfo.from_seed_state(
        seed_state,
        slot.active_seed_params,
        counterfactual_total_improvement=contribution,
    )


def _oracle_fossilize_active_seed(model: SlottedHostProtocol, slot_id: str) -> bool:
    if not model.has_active_seed_in_slot(slot_id):
        return False
    slot = _slot_for(model, slot_id)
    seed_state = slot.state
    if seed_state is None:
        return False
    if seed_state.stage != SeedStage.HOLDING:
        return False
    gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
    if gate_result.passed:
        slot.set_alpha(1.0)
        return True
    return False


def _execute_oracle_handler(
    *,
    action: FactoredAction,
    env_state: ParallelEnvState,
    slot: SeedSlotProtocol,
    target_slot: str,
    epoch: int,
    max_epochs: int,
) -> bool:
    handler_ctx = HandlerContext(
        env_idx=0,
        slot_id=target_slot,
        env_state=env_state,
        model=env_state.model,
        slot=slot,
        seed_state=slot.state,
        epoch=epoch,
        max_epochs=max_epochs,
        episodes_completed=0,
    )
    handler = get_handler(action.op.value)
    if action.op == LifecycleOp.GERMINATE:
        result = handler(
            handler_ctx,
            GerminateParams(
                blueprint_idx=action.blueprint.value,
                style_idx=action.style.value,
                tempo_idx=action.tempo.value,
                alpha_target=action.alpha_target_value,
            ),
        )
        return bool(result.success)
    if action.op == LifecycleOp.FOSSILIZE:
        result = handler(
            handler_ctx,
            _oracle_seed_info(slot=slot),
            _oracle_fossilize_active_seed,
        )
        return bool(result.success)
    if action.op == LifecycleOp.PRUNE:
        result = handler(
            handler_ctx,
            PruneParams(
                alpha_speed_idx=action.alpha_speed.value,
                alpha_curve_idx=action.alpha_curve.value,
            ),
            _oracle_seed_info(slot=slot),
        )
        return bool(result.success)
    if action.op == LifecycleOp.SET_ALPHA_TARGET:
        result = handler(
            handler_ctx,
            AlphaTargetParams(
                alpha_target_idx=action.alpha_target.value,
                alpha_speed_idx=action.alpha_speed.value,
                alpha_curve_idx=action.alpha_curve.value,
                style_idx=action.style.value,
            ),
        )
        return bool(result.success)
    result = handler(handler_ctx)
    return bool(result.success)


def run_oracle_schedule(
    *,
    schedule: OracleSchedule,
    fixture: OracleSandboxFixture,
    val_loss: float = 0.5,
    val_accuracy: float = 2.0,
    max_epochs: int = 32,
    telemetry_dir: Path | str | None = None,
) -> OracleRunResult:
    """Run an oracle schedule through masks, governor preflight, and handlers."""
    telemetry_output = (
        DirectoryOutput(telemetry_dir, buffer_size=1)
        if telemetry_dir is not None
        else None
    )
    output_dir = str(telemetry_output.output_dir) if telemetry_output is not None else None
    host_params = fixture.model.total_params - fixture.model.active_seed_params
    env_state = _build_oracle_env_state(
        fixture=fixture,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )
    step_results: list[OracleStepResult] = []
    if telemetry_output is not None:
        _emit_oracle_training_started(
            telemetry_output=telemetry_output,
            schedule=schedule,
            fixture=fixture,
            host_params=host_params,
            max_epochs=max_epochs,
        )

    try:
        for epoch, schedule_step in enumerate(schedule.steps, start=1):
            action = schedule_step.action
            target_slot, slot_is_enabled = _resolve_target_slot(
                action.slot_idx,
                enabled_slots=list(fixture.enabled_slots),
                slot_config=fixture.slot_config,
            )
            if not slot_is_enabled:
                stage_before = "DORMANT"
            else:
                stage_before = _slot_stage_name(fixture.model, target_slot)

            slot_reports = fixture.model.get_slot_reports()
            slot_states = build_slot_states(slot_reports, list(fixture.enabled_slots))
            masks = compute_action_masks(
                slot_states=slot_states,
                enabled_slots=list(fixture.enabled_slots),
                slot_config=fixture.slot_config,
            )
            mask_allowed = _action_allowed_by_masks(
                action=action,
                masks=masks,
                slot_is_enabled=slot_is_enabled,
            )
            if not mask_allowed:
                step_result = OracleStepResult(
                    step_id=schedule_step.step_id,
                    op=action.op,
                    expected_success=schedule_step.expect_success,
                    success=False,
                    stage_before=stage_before,
                    stage_after=(
                        _slot_stage_name(fixture.model, target_slot)
                        if slot_is_enabled
                        else "DORMANT"
                    ),
                    mask_checked=True,
                    governor_checked=False,
                    governor_approved=None,
                    handler_called=False,
                    blocked_by="mask",
                    detail="action masked",
                )
                step_results.append(step_result)
                if telemetry_output is not None:
                    _emit_oracle_step_telemetry(
                        telemetry_output=telemetry_output,
                        schedule_step=schedule_step,
                        step_result=step_result,
                        target_slot=target_slot,
                        epoch=epoch,
                    )
                if schedule_step.expect_success:
                    break
                continue

            governor_checked = action.op != LifecycleOp.WAIT
            governor_approved: bool | None = None
            if governor_checked:
                slot = _slot_for(fixture.model, target_slot)
                seed_state = slot.state
                preflight = env_state.governor.preflight_lifecycle_mutation(
                    operation=action.op,
                    slot_id=target_slot,
                    blueprint_id=(
                        action.blueprint.to_blueprint_id()
                        if action.op == LifecycleOp.GERMINATE
                        else None
                    ),
                    alpha_target=(
                        action.alpha_target_value
                        if action.op == LifecycleOp.SET_ALPHA_TARGET
                        else None
                    ),
                    alpha_speed_steps=(
                        action.alpha_speed_steps
                        if action.op in (LifecycleOp.SET_ALPHA_TARGET, LifecycleOp.PRUNE)
                        else None
                    ),
                    alpha_curve=(
                        action.alpha_curve.name
                        if action.op in (LifecycleOp.SET_ALPHA_TARGET, LifecycleOp.PRUNE)
                        else None
                    ),
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    seed_stage=seed_state.stage if seed_state is not None else None,
                    total_params=fixture.model.total_params,
                    effective_seed_params=float(slot.active_seed_params),
                    max_seeds=1,
                    active_seed_count=fixture.model.total_seeds(),
                    cooldown_epochs_remaining=0,
                    event_id=f"{schedule.identity}:{schedule_step.step_id}",
                )
                governor_approved = preflight.approved
                if not preflight.approved:
                    step_result = OracleStepResult(
                        step_id=schedule_step.step_id,
                        op=action.op,
                        expected_success=schedule_step.expect_success,
                        success=False,
                        stage_before=stage_before,
                        stage_after=_slot_stage_name(fixture.model, target_slot),
                        mask_checked=True,
                        governor_checked=True,
                        governor_approved=False,
                        handler_called=False,
                        blocked_by="governor",
                        detail=preflight.reason,
                    )
                    step_results.append(step_result)
                    if telemetry_output is not None:
                        _emit_oracle_step_telemetry(
                            telemetry_output=telemetry_output,
                            schedule_step=schedule_step,
                            step_result=step_result,
                            target_slot=target_slot,
                            epoch=epoch,
                        )
                    if schedule_step.expect_success:
                        break
                    continue

            slot = _slot_for(fixture.model, target_slot)
            handler_success = _execute_oracle_handler(
                action=action,
                env_state=env_state,
                slot=slot,
                target_slot=target_slot,
                epoch=epoch,
                max_epochs=max_epochs,
            )
            stage_after = _slot_stage_name(fixture.model, target_slot)
            step_result = OracleStepResult(
                step_id=schedule_step.step_id,
                op=action.op,
                expected_success=schedule_step.expect_success,
                success=handler_success,
                stage_before=stage_before,
                stage_after=stage_after,
                mask_checked=True,
                governor_checked=governor_checked,
                governor_approved=governor_approved,
                handler_called=True,
                blocked_by=None if handler_success else "handler",
            )
            step_results.append(step_result)
            if handler_success:
                _settle_oracle_step_evidence(
                    fixture=fixture,
                    target_slot=target_slot,
                    val_accuracy=val_accuracy,
                    max_epochs=max_epochs,
                )
            if telemetry_output is not None:
                _emit_oracle_step_telemetry(
                    telemetry_output=telemetry_output,
                    schedule_step=schedule_step,
                    step_result=step_result,
                    target_slot=target_slot,
                    epoch=epoch,
                )
            if handler_success != schedule_step.expect_success:
                break

        step_results_tuple = tuple(step_results)
        run_success = all(
            step.success == step.expected_success for step in step_results_tuple
        ) and len(step_results_tuple) == len(schedule.steps)
        if telemetry_output is not None:
            _emit_oracle_policy_evidence(
                telemetry_output=telemetry_output,
                schedule=schedule,
            )
            _emit_oracle_episode_outcome(
                telemetry_output=telemetry_output,
                schedule=schedule,
                fixture=fixture,
                step_results=step_results_tuple,
                host_params=host_params,
                val_accuracy=val_accuracy,
                run_success=run_success,
                num_fossilized=env_state.seeds_fossilized,
                num_contributing_fossilized=env_state.contributing_fossilized,
            )
        return OracleRunResult(
            schedule_identity=schedule.identity,
            success=run_success,
            steps=step_results_tuple,
            telemetry_dir=output_dir,
        )
    finally:
        if telemetry_output is not None:
            telemetry_output.close()


__all__ = [
    "ORACLE_CI_FIXTURE_ID",
    "ORACLE_CI_SLOT_ID",
    "ORACLE_PROOF_PROFILE",
    "OracleRunResult",
    "OracleSandboxFixture",
    "OracleSandboxHost",
    "OracleSchedule",
    "OracleScheduleStep",
    "OracleStepResult",
    "build_default_oracle_schedule",
    "build_oracle_ci_fixture",
    "run_oracle_schedule",
]
