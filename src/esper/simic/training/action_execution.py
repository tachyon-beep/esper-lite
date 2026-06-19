from __future__ import annotations

import hashlib
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import numpy as np
import torch

from esper.leyline import (
    ALPHA_SPEED_TO_STEPS,
    ALPHA_TARGET_VALUES,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    BLUEPRINT_IDS,
    EPISODE_SUCCESS_THRESHOLD,
    EpisodeOutcome,
    EpisodeOutcomePayload,
    HEAD_NAMES,
    HeadTelemetry,
    LifecycleMutationCausalContext,
    LifecycleOp,
    MorphologyCausalLogPhase,
    MIN_PRUNE_AGE,
    MorphologyCausalLogPayload,
    OP_ADVANCE,
    OP_FOSSILIZE,
    OP_GERMINATE,
    OP_NAMES,
    OP_PRUNE,
    OP_SET_ALPHA_TARGET,
    OP_WAIT,
    SeedSlotProtocol,
    SeedStateProtocol,
    SeedStage,
    SlottedHostProtocol,
    SlotConfig,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    TelemetryEvent,
    TelemetryEventType,
)
from esper.simic.rewards import (
    compute_reward,
    compute_loss_reward,
    ContributionRewardConfig,
    RewardFamily,
    RewardMode,
    SeedInfo,
    STAGE_POTENTIALS,
)
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks

from .helpers import compute_rent_and_shock_inputs
from .handlers import (
    AlphaTargetParams,
    GerminateParams,
    HandlerContext,
    PruneParams,
    get_handler,
)
from .parallel_env_state import ParallelEnvState
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    EnvStepRecord,
    EpisodeRecord,
    RewardSummaryAccumulator,
)

if TYPE_CHECKING:
    from esper.leyline.reports import SeedStateReport
    from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
    from esper.simic.rewards.contribution import FossilizedSeedDripState
    from esper.simic.telemetry.emitters import VectorizedEmitter
    from esper.simic.control import RewardNormalizer
    from esper.simic.agent import PPOAgent
    from esper.leyline.signals import TrainingSignals
    from esper.runtime.tasks import TaskSpec
    from esper.nissa import BlueprintAnalytics


@dataclass(slots=True, frozen=True)
class FreshContributionTargets:
    """Counterfactual contribution targets plus proof freshness status."""

    has_fresh_contribution: bool
    contribution_targets: torch.Tensor | None
    contribution_mask: torch.Tensor | None
    stale_slots: tuple[str, ...] = ()
    missing_tracking_slots: tuple[str, ...] = ()


def build_fresh_contribution_targets(
    *,
    env_state: ParallelEnvState,
    baseline_accs: dict[str, float],
    slot_config: SlotConfig,
    device: torch.device | str,
) -> FreshContributionTargets:
    """Build rollout contribution targets only from current counterfactual evidence."""
    if not baseline_accs:
        return FreshContributionTargets(
            has_fresh_contribution=False,
            contribution_targets=None,
            contribution_mask=None,
        )

    stale_slots: list[str] = []
    missing_tracking_slots: list[str] = []
    for slot_id in baseline_accs:
        if slot_id not in env_state.epochs_since_counterfactual:
            missing_tracking_slots.append(slot_id)
            continue
        if env_state.epochs_since_counterfactual[slot_id] != 0:
            stale_slots.append(slot_id)

    if stale_slots or missing_tracking_slots:
        return FreshContributionTargets(
            has_fresh_contribution=False,
            contribution_targets=None,
            contribution_mask=None,
            stale_slots=tuple(stale_slots),
            missing_tracking_slots=tuple(missing_tracking_slots),
        )

    num_slots = slot_config.num_slots
    contribution_targets = torch.zeros(num_slots, device=device)
    contribution_mask = torch.zeros(num_slots, dtype=torch.bool, device=device)

    for slot_id, baseline_acc in baseline_accs.items():
        slot_idx = slot_config.index_for_slot_id(slot_id)
        contribution = env_state.val_acc - baseline_acc
        contribution_targets[slot_idx] = contribution
        contribution_mask[slot_idx] = True

    return FreshContributionTargets(
        has_fresh_contribution=True,
        contribution_targets=contribution_targets,
        contribution_mask=contribution_mask,
    )


def _build_decision_head_telemetry(
    *,
    head_confidences_cpu: np.ndarray | None,
    head_entropies_cpu: np.ndarray | None,
    env_idx: int,
) -> HeadTelemetry | None:
    """Build per-head decision telemetry only when entropy evidence is present."""
    if head_confidences_cpu is None or head_entropies_cpu is None:
        return None
    return HeadTelemetry(
        op_confidence=float(head_confidences_cpu[0, env_idx]),
        slot_confidence=float(head_confidences_cpu[1, env_idx]),
        blueprint_confidence=float(head_confidences_cpu[2, env_idx]),
        style_confidence=float(head_confidences_cpu[3, env_idx]),
        tempo_confidence=float(head_confidences_cpu[4, env_idx]),
        alpha_target_confidence=float(head_confidences_cpu[5, env_idx]),
        alpha_speed_confidence=float(head_confidences_cpu[6, env_idx]),
        curve_confidence=float(head_confidences_cpu[7, env_idx]),
        op_entropy=float(head_entropies_cpu[0, env_idx]),
        slot_entropy=float(head_entropies_cpu[1, env_idx]),
        blueprint_entropy=float(head_entropies_cpu[2, env_idx]),
        style_entropy=float(head_entropies_cpu[3, env_idx]),
        tempo_entropy=float(head_entropies_cpu[4, env_idx]),
        alpha_target_entropy=float(head_entropies_cpu[5, env_idx]),
        alpha_speed_entropy=float(head_entropies_cpu[6, env_idx]),
        curve_entropy=float(head_entropies_cpu[7, env_idx]),
    )


_HEAD_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(HEAD_NAMES)}
_HEAD_SLOT_IDX = _HEAD_NAME_TO_IDX["slot"]
_HEAD_BLUEPRINT_IDX = _HEAD_NAME_TO_IDX["blueprint"]
_HEAD_STYLE_IDX = _HEAD_NAME_TO_IDX["style"]
_HEAD_TEMPO_IDX = _HEAD_NAME_TO_IDX["tempo"]
_HEAD_ALPHA_TARGET_IDX = _HEAD_NAME_TO_IDX["alpha_target"]
_HEAD_ALPHA_SPEED_IDX = _HEAD_NAME_TO_IDX["alpha_speed"]
_HEAD_ALPHA_CURVE_IDX = _HEAD_NAME_TO_IDX["alpha_curve"]
_HEAD_OP_IDX = _HEAD_NAME_TO_IDX["op"]

_logger = logging.getLogger(__name__)


def _reward_components_required_for_state_transport(reward_mode: RewardMode) -> bool:
    """Reward modes that persist state via RewardComponentsTelemetry."""
    return reward_mode in (RewardMode.ESCROW, RewardMode.BASIC_PLUS)


class ResolveTargetSlot(Protocol):
    def __call__(
        self,
        slot_idx: int,
        *,
        enabled_slots: list[str],
        slot_config: SlotConfig,
    ) -> tuple[str, bool]: ...


def _parse_sampled_action(
    env_idx: int,
    op_idx: int,
    slot_idx: int,
    style_idx: int,
    alpha_target_idx: int,
    slots: list[str],
    slot_config: SlotConfig,
    model: SlottedHostProtocol,
    resolve_target_slot: ResolveTargetSlot,
    action_spec: ActionSpec,
) -> tuple[Any, Any]:
    """Consolidate action derived values and validation logic (Deduplication)."""
    # Use the SAMPLED slot as target (multi-slot support)
    target_slot, slot_is_enabled = resolve_target_slot(
        slot_idx,
        enabled_slots=slots,
        slot_config=slot_config,
    )

    slot_state = model.seed_slots[target_slot].state if slot_is_enabled else None
    seed_state = (
        slot_state
        if slot_is_enabled and model.has_active_seed_in_slot(target_slot)
        else None
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
            # BUG-020 fix: enforce MIN_PRUNE_AGE to match masking invariant
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

    action_for_reward = (
        LifecycleOp(op_idx) if action_valid_for_reward else LifecycleOp.WAIT
    )

    action_spec.slot_idx = slot_idx
    action_spec.style_idx = style_idx
    action_spec.alpha_target_idx = alpha_target_idx
    action_spec.op_idx = op_idx
    action_spec.target_slot = target_slot
    action_spec.slot_is_enabled = slot_is_enabled
    action_spec.action_valid_for_reward = action_valid_for_reward
    action_spec.action_for_reward = action_for_reward
    action_spec.blend_algorithm_id = blend_algorithm_id
    action_spec.alpha_algorithm = alpha_algorithm
    action_spec.alpha_target = alpha_target

    return slot_state, seed_state


def classify_episode_outcome(val_acc: float) -> str:
    """Classify episode outcome using percent-scale accuracy."""
    if val_acc >= EPISODE_SUCCESS_THRESHOLD:
        return "success"
    return "timeout"


@dataclass(frozen=True)
class ActionExecutionContext:
    slots: list[str]
    ordered_slots: list[str]
    slot_config: SlotConfig
    task_spec: TaskSpec
    env_reward_configs: list[ContributionRewardConfig]
    reward_family_enum: RewardFamily
    reward_config: ContributionRewardConfig
    loss_reward_config: Any
    reward_normalizer: RewardNormalizer
    telemetry_config: Any
    ops_telemetry_enabled: bool
    disable_advance: bool
    effective_max_seeds: int
    max_epochs: int
    num_train_batches: int
    device: str
    analytics: BlueprintAnalytics
    emitters: list[VectorizedEmitter]
    agent: PPOAgent
    fossilize_active_seed: Callable[[Any, str], bool]
    resolve_target_slot: ResolveTargetSlot
    host_params_baseline: int


@dataclass
class ActionExecutionResult:
    truncated_bootstrap_targets: list[tuple[int, int]]
    post_action_signals: list[TrainingSignals]
    post_action_slot_reports: list[dict[str, SeedStateReport]]
    post_action_masks: list[dict[str, torch.Tensor]]


def _build_lifecycle_mutation_context(
    *,
    batch_idx: int,
    epoch: int,
    env_idx: int,
    op_action: int,
    target_slot: str,
    observation_hash: str,
    topology: str,
) -> LifecycleMutationCausalContext:
    """Create stable morphology proposal/verdict/mutation identity for one action."""
    base = _build_lifecycle_action_id(
        batch_idx=batch_idx,
        epoch=epoch,
        env_idx=env_idx,
        target_slot=target_slot,
        op_action=op_action,
    )
    digest = hashlib.sha256(base.encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return LifecycleMutationCausalContext(
        action_id=base,
        proposal_id=f"{base}-proposal",
        verdict_id=f"{base}-verdict",
        mutation_id=f"{base}-mutation",
        observation_hash=observation_hash,
        rng_stream=f"simic.lifecycle.env{env_idx}",
        rng_seed=rng_seed,
        topology=topology,
        slot_id=target_slot,
        operation=LifecycleOp(op_action).name,
    )


def _build_lifecycle_action_id(
    *,
    batch_idx: int,
    epoch: int,
    env_idx: int,
    target_slot: str,
    op_action: int,
) -> str:
    """Create the stable action identity shared by buffer and causal log rows."""
    return f"morph-b{batch_idx}-e{epoch}-env{env_idx}-{target_slot}-op{op_action}"


def _observation_hash(observation: torch.Tensor) -> str:
    """Return a stable content hash for the observation used by a lifecycle action."""
    obs_cpu = observation.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(obs_cpu.numpy().tobytes()).hexdigest()[:16]
    return f"obs-{digest}"


def _build_morphology_causal_log_payload(
    *,
    phase: MorphologyCausalLogPhase,
    context: LifecycleMutationCausalContext,
    env_idx: int,
    blueprint_id: str | None,
    governor_approved: bool | None = None,
    governor_reason: str | None = None,
    governor_blocked_factor: str | None = None,
    watch_window_evidence: float | None = None,
    linked_event_id: str | None = None,
) -> MorphologyCausalLogPayload:
    return MorphologyCausalLogPayload(
        phase=phase,
        env_id=env_idx,
        slot_id=context.slot_id,
        operation=context.operation,
        action_id=context.action_id,
        proposal_id=context.proposal_id,
        verdict_id=context.verdict_id,
        mutation_id=context.mutation_id,
        observation_hash=context.observation_hash,
        rng_stream=context.rng_stream,
        rng_seed=context.rng_seed,
        topology=context.topology,
        blueprint_id=blueprint_id,
        governor_approved=governor_approved,
        governor_reason=governor_reason,
        governor_blocked_factor=governor_blocked_factor,
        watch_window_evidence=watch_window_evidence,
        linked_event_id=linked_event_id,
    )


def _build_rollback_causal_log_payload(
    *,
    batch_idx: int,
    epoch: int,
    env_idx: int,
    op_action: int,
    target_slot: str,
    topology: str,
    observation_hash: str,
    reason: str,
    loss_at_panic: float | None,
    blueprint_id: str | None = None,
) -> MorphologyCausalLogPayload:
    context = _build_lifecycle_mutation_context(
        batch_idx=batch_idx,
        epoch=epoch,
        env_idx=env_idx,
        op_action=op_action,
        target_slot=target_slot,
        observation_hash=observation_hash,
        topology=topology,
    )
    return _build_morphology_causal_log_payload(
        phase="rollback",
        context=context,
        env_idx=env_idx,
        blueprint_id=blueprint_id,
        governor_approved=False,
        governor_reason=reason,
        governor_blocked_factor="rollback",
        watch_window_evidence=loss_at_panic,
        linked_event_id=context.mutation_id,
    )


def execute_actions(
    *,
    context: ActionExecutionContext,
    env_states: list[ParallelEnvState],
    step_records: list[EnvStepRecord],
    actions_np: np.ndarray,
    values: list[float],
    all_signals: list[TrainingSignals],
    all_slot_reports: list[dict[str, SeedStateReport]],
    states_batch_normalized: torch.Tensor,
    blueprint_indices_batch: torch.Tensor,
    pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]],
    head_log_probs: dict[str, torch.Tensor],
    masks_batch: dict[str, torch.Tensor],
    head_confidences_cpu: np.ndarray | None,
    head_entropies_cpu: np.ndarray | None,
    op_probs_cpu: np.ndarray | None,
    masked_np: np.ndarray | None,
    baseline_accs: list[dict[str, Any]],
    all_disabled_accs: dict[int, float],
    governor_panic_envs: list[int],
    reward_summary_accum: list[RewardSummaryAccumulator],
    episode_history: list[EpisodeRecord],
    episode_outcomes: list[EpisodeOutcome],
    step_obs_stats: Any | None,
    epoch: int,
    episodes_completed: int,
    batch_idx: int,
    proof_controlled_step: bool = False,
) -> ActionExecutionResult:
    slots = context.slots
    slot_config = context.slot_config
    reward_family_enum = context.reward_family_enum
    env_reward_configs = context.env_reward_configs
    reward_config = context.reward_config
    loss_reward_config = context.loss_reward_config
    telemetry_config = context.telemetry_config
    ops_telemetry_enabled = context.ops_telemetry_enabled
    ordered_slots = context.ordered_slots
    effective_max_seeds = context.effective_max_seeds
    task_spec = context.task_spec
    device = context.device
    emitters = context.emitters
    agent = context.agent
    analytics = context.analytics
    reward_normalizer = context.reward_normalizer
    max_epochs = context.max_epochs
    num_train_batches = context.num_train_batches
    host_params_baseline = context.host_params_baseline

    truncated_bootstrap_targets: list[tuple[int, int]] = []
    all_post_action_signals: list[TrainingSignals] = []
    all_post_action_slot_reports: list[dict[str, SeedStateReport]] = []
    all_post_action_masks: list[dict[str, torch.Tensor]] = []

    # Cache per-head tensors for buffer writes (avoid dict lookups in hot loop)
    slot_log_probs_batch = head_log_probs["slot"]
    blueprint_log_probs_batch = head_log_probs["blueprint"]
    style_log_probs_batch = head_log_probs["style"]
    tempo_log_probs_batch = head_log_probs["tempo"]
    alpha_target_log_probs_batch = head_log_probs["alpha_target"]
    alpha_speed_log_probs_batch = head_log_probs["alpha_speed"]
    alpha_curve_log_probs_batch = head_log_probs["alpha_curve"]
    op_log_probs_batch = head_log_probs["op"]

    slot_by_op_masks_batch = masks_batch["slot_by_op"]
    blueprint_masks_batch = masks_batch["blueprint"]
    style_masks_batch = masks_batch["style"]
    tempo_masks_batch = masks_batch["tempo"]
    alpha_target_masks_batch = masks_batch["alpha_target"]
    alpha_speed_masks_batch = masks_batch["alpha_speed"]
    alpha_curve_masks_batch = masks_batch["alpha_curve"]
    op_masks_batch = masks_batch["op"]

    # D5: Vectorized forced step detection (single GPU sync for all envs)
    # A step is "forced" if only 1 valid op AND that op is WAIT.
    # ChatGPT Pro review 2025-01-08: Move detection OUTSIDE env loop to avoid
    # N separate GPU→CPU syncs from .item() calls in hot path.
    num_valid_ops_batch = op_masks_batch.sum(dim=1)  # [num_envs]
    wait_valid_batch = op_masks_batch[:, OP_WAIT]  # [num_envs]
    forced_batch = (num_valid_ops_batch == 1) & wait_valid_batch  # [num_envs]
    forced_batch_cpu = forced_batch.tolist()  # Single GPU→CPU sync

    for env_idx, env_state in enumerate(env_states):
        model = env_state.model
        signals = all_signals[env_idx]
        value = values[env_idx]

        # Parse sampled action indices and derive values (Deduplication)
        slot_action = int(actions_np[_HEAD_SLOT_IDX, env_idx])
        blueprint_action = int(actions_np[_HEAD_BLUEPRINT_IDX, env_idx])
        style_action = int(actions_np[_HEAD_STYLE_IDX, env_idx])
        tempo_action = int(actions_np[_HEAD_TEMPO_IDX, env_idx])
        alpha_target_action = int(actions_np[_HEAD_ALPHA_TARGET_IDX, env_idx])
        alpha_speed_action = int(actions_np[_HEAD_ALPHA_SPEED_IDX, env_idx])
        alpha_curve_action = int(actions_np[_HEAD_ALPHA_CURVE_IDX, env_idx])
        op_action = int(actions_np[_HEAD_OP_IDX, env_idx])

        record = step_records[env_idx]
        action_spec = record.action_spec
        action_spec.slot_idx = slot_action
        action_spec.blueprint_idx = blueprint_action
        action_spec.style_idx = style_action
        action_spec.tempo_idx = tempo_action
        action_spec.alpha_target_idx = alpha_target_action
        action_spec.alpha_speed_idx = alpha_speed_action
        action_spec.alpha_curve_idx = alpha_curve_action
        action_spec.op_idx = op_action

        action_outcome = record.action_outcome
        action_outcome.reward_components = None
        action_outcome.episode_reward = None
        action_outcome.final_accuracy = None
        action_outcome.episode_outcome = None
        action_outcome.action_name = OP_NAMES[op_action]

        slot_state, seed_state = _parse_sampled_action(
            env_idx,
            op_action,
            slot_action,
            style_action,
            alpha_target_action,
            slots,
            slot_config,
            model,
            context.resolve_target_slot,
            action_spec,
        )
        target_slot = action_spec.target_slot
        slot_is_enabled = action_spec.slot_is_enabled
        action_for_reward = action_spec.action_for_reward
        alpha_target = action_spec.alpha_target
        action_id = _build_lifecycle_action_id(
            batch_idx=batch_idx,
            epoch=epoch,
            env_idx=env_idx,
            target_slot=target_slot,
            op_action=op_action,
        )

        action_success = False

        # Governor rollback
        if env_idx in governor_panic_envs:
            # Capture panic info BEFORE rollback (execute_rollback resets these).
            # Used only for the joinable morphology causal-log rows below; the
            # authoritative GOVERNOR_ROLLBACK panic record is emitted by the governor.
            panic_reason = env_state.governor._panic_reason
            panic_loss = env_state.governor._panic_loss
            rollback_raw_penalty = env_state.governor.get_punishment_reward()
            rollback_normalized_penalty = float(
                max(
                    -reward_normalizer.clip,
                    min(reward_normalizer.clip, rollback_raw_penalty),
                )
            )
            rollback_triggering_action_id = (
                agent.buffer.last_action_id(env_idx)
                if agent.buffer.step_counts[env_idx] > 0
                else None
            )
            rollback_severity = abs(rollback_raw_penalty)
            rollback_watch_window_evidence = rollback_severity

            # Stream safety: rollback mutates model tensors; ensure it runs on the
            # per-env CUDA stream to avoid default-stream leakage and races.
            rollback_ctx = (
                torch.cuda.stream(env_state.stream)
                if env_state.stream
                else nullcontext()
            )
            with rollback_ctx:
                env_state.governor.execute_rollback(
                    env_id=env_idx,
                    triggering_action_id=rollback_triggering_action_id,
                    raw_penalty=rollback_raw_penalty,
                    normalized_penalty=rollback_normalized_penalty,
                    rollback_severity=rollback_severity,
                    watch_window_evidence=rollback_watch_window_evidence,
                )
            record.rollback_occurred = True

            # CRITICAL: Clear optimizer momentum after rollback.
            # PyTorch's load_state_dict() copies weights IN-PLACE, so
            # Parameter objects retain their identity (same id()). The
            # optimizer's state dict is keyed by Parameter objects, so
            # momentum/variance buffers SURVIVE the rollback. Without
            # clearing, SGD momentum continues pushing toward the
            # diverged state that caused the panic, risking immediate
            # re-divergence. See B1-PT-01 correction notes.
            env_state.host_optimizer.state.clear()
            for seed_opt in env_state.seed_optimizers.values():
                seed_opt.state.clear()

            # KTS-003: The GOVERNOR_ROLLBACK telemetry event is emitted by the
            # SINGLE authoritative source — TolariaGovernor.execute_rollback() — which
            # carries the complete panic context (loss_at_panic, loss_threshold,
            # consecutive_panics, panic_reason, any state_dict key mismatches, and
            # Simic's rollback attribution context).
            # Simic must NOT emit a second, partial GOVERNOR_ROLLBACK here; the governor
            # is the panic authority. Simic only records the joinable morphology causal
            # log rows below (which describe the *aborted lifecycle action*, not the
            # panic itself).
            rollback_blueprint_id = (
                BLUEPRINT_IDS[blueprint_action] if op_action == OP_GERMINATE else None
            )
            rollback_payload = _build_rollback_causal_log_payload(
                batch_idx=batch_idx,
                epoch=epoch,
                env_idx=env_idx,
                op_action=op_action,
                target_slot=target_slot,
                topology=task_spec.topology if task_spec.topology is not None else "unknown",
                observation_hash=_observation_hash(states_batch_normalized[env_idx]),
                reason=panic_reason or "unknown",
                loss_at_panic=panic_loss,
                blueprint_id=rollback_blueprint_id,
            )
            rollback_phases: tuple[tuple[MorphologyCausalLogPhase, str], ...] = (
                ("rollback", "Morphology rollback causal log"),
                ("cooldown", "Morphology rollback cooldown"),
                ("audit", "Morphology rollback audit"),
            )
            for rollback_phase, message in rollback_phases:
                emitters[env_idx].emit(TelemetryEvent(
                    event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                    data=replace(rollback_payload, phase=rollback_phase),
                    severity="warning",
                    message=message,
                ))

            action_outcome.rollback_occurred = True
            action_outcome.action_success = False
            action_outcome.reward_raw = 0.0
            action_outcome.reward_normalized = 0.0
            action_outcome.truncated = False
            env_state.last_action_success = False
            env_state.last_action_op = OP_WAIT

            # Credit assignment: the panic was detected from the CURRENT val_loss
            # (a consequence of the PRIOR executed transition); this step's sampled
            # action never executed (rollback ran instead). We must NOT add a buffer
            # row for that unexecuted action — doing so would make handle_rollbacks'
            # mark_terminal_with_penalty (which targets the last transition) blame
            # an innocent action and train PPO to avoid it. Instead we end the
            # episode here so the PRIOR executed transition becomes the terminal
            # row that receives the death penalty. If this env had no prior
            # transition (first-step panic), the penalty is correctly dropped
            # (mark_terminal_with_penalty returns False — nothing to attribute).
            agent.buffer.end_episode(env_id=env_idx)
            continue

        action_outcome.rollback_occurred = False
        # P1-a: clear the DOWNSTREAM-read flag on the pooled EnvStepRecord too.
        # step_records are pre-allocated once per batch and reused across epochs;
        # the per-epoch LSTM reset and the death-penalty / truncated-bootstrap
        # handling read record.rollback_occurred, NOT action_outcome's copy. A
        # stale True from a prior epoch's panic must not leak into this healthy
        # transition.
        record.rollback_occurred = False

        scoreboard = analytics._get_scoreboard(env_idx)
        host_params = scoreboard.host_params
        effective_seed_params, alpha_delta_sq_sum = compute_rent_and_shock_inputs(
            model=model,
            slot_ids=slots,
            host_params=host_params,
            host_params_floor=env_reward_configs[env_idx].rent_host_params_floor,
            base_slot_rent_ratio=env_reward_configs[env_idx].base_slot_rent_ratio,
            prev_slot_alphas=env_state.prev_slot_alphas,
            prev_slot_params=env_state.prev_slot_params,
        )

        mutation_allowed = True
        morphology_context: LifecycleMutationCausalContext | None = None
        morphology_blueprint_id: str | None = None
        if slot_is_enabled and op_action in (
            OP_GERMINATE,
            OP_FOSSILIZE,
            OP_PRUNE,
            OP_SET_ALPHA_TARGET,
            OP_ADVANCE,
        ):
            morphology_context = _build_lifecycle_mutation_context(
                batch_idx=batch_idx,
                epoch=epoch,
                env_idx=env_idx,
                op_action=op_action,
                target_slot=target_slot,
                observation_hash=_observation_hash(states_batch_normalized[env_idx]),
                topology=task_spec.topology if task_spec.topology is not None else "unknown",
            )
            preflight_alpha_speed_steps = None
            preflight_alpha_curve = None
            if op_action in (OP_PRUNE, OP_SET_ALPHA_TARGET):
                preflight_alpha_speed_steps = ALPHA_SPEED_TO_STEPS[
                    AlphaSpeedAction(alpha_speed_action)
                ]
                preflight_alpha_curve = AlphaCurveAction(alpha_curve_action).name
            preflight_blueprint_id = (
                BLUEPRINT_IDS[blueprint_action] if op_action == OP_GERMINATE else None
            )
            morphology_blueprint_id = preflight_blueprint_id
            emitters[env_idx].emit(TelemetryEvent(
                event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                data=_build_morphology_causal_log_payload(
                    phase="proposal",
                    context=morphology_context,
                    env_idx=env_idx,
                    blueprint_id=preflight_blueprint_id,
                ),
                severity="debug",
                message="Morphology proposal",
            ))
            preflight_verdict = env_state.governor.preflight_lifecycle_mutation(
                operation=LifecycleOp(op_action),
                slot_id=target_slot,
                blueprint_id=preflight_blueprint_id,
                alpha_target=alpha_target if op_action == OP_SET_ALPHA_TARGET else None,
                alpha_speed_steps=preflight_alpha_speed_steps,
                alpha_curve=preflight_alpha_curve,
                val_loss=env_state.val_loss,
                val_accuracy=env_state.val_acc,
                seed_stage=seed_state.stage if seed_state is not None else None,
                total_params=model.total_params,
                effective_seed_params=effective_seed_params,
                max_seeds=effective_max_seeds,
                active_seed_count=model.total_seeds(),
                cooldown_epochs_remaining=0,
                event_id=morphology_context.proposal_id,
            )
            emitters[env_idx].emit(TelemetryEvent(
                event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                data=_build_morphology_causal_log_payload(
                    phase="verdict",
                    context=morphology_context,
                    env_idx=env_idx,
                    blueprint_id=preflight_blueprint_id,
                    governor_approved=preflight_verdict.approved,
                    governor_reason=preflight_verdict.reason,
                    governor_blocked_factor=preflight_verdict.blocked_factor,
                ),
                severity="debug" if preflight_verdict.approved else "warning",
                message="Morphology governor verdict",
            ))
            if not preflight_verdict.approved:
                action_spec.action_valid_for_reward = False
                action_spec.action_for_reward = LifecycleOp.WAIT
                action_for_reward = LifecycleOp.WAIT
                mutation_allowed = False

        # Use op name for action counting only after rollback has ceded this step.
        env_state.action_counts[action_spec.action_for_reward.name] = (
            env_state.action_counts[action_spec.action_for_reward.name] + 1
        )

        # Compute reward
        seed_contribution = None
        if target_slot in baseline_accs[env_idx]:
            seed_contribution = (
                env_state.val_acc - baseline_accs[env_idx][target_slot]
            )

        emit_reward_components_event = (
            telemetry_config is not None and telemetry_config.should_collect("debug")
        )
        # Match ops_telemetry_enabled logic: default to True when no config
        # This ensures reward_components are computed when on_last_action is called
        collect_reward_summary = (
            telemetry_config is None
            or telemetry_config.should_collect("ops_normal")
        )

        seed_params_for_slot = (
            cast(SeedSlotProtocol, model.seed_slots[target_slot]).active_seed_params
            if slot_is_enabled
            else 0
        )
        # Clean counterfactual for the anti-gaming / fossilization gates: the total
        # seed-attributable improvement = val_acc(all seeds on) - val_acc(all seeds off).
        # Fall back to the host-only baseline (min over the per-slot leave-one-out accs)
        # when the all-disabled ablation wasn't measured for this env.
        env_all_off_acc = all_disabled_accs.get(env_idx)
        if env_all_off_acc is None and baseline_accs[env_idx]:
            env_all_off_acc = min(baseline_accs[env_idx].values())
        counterfactual_total_improvement = (
            env_state.val_acc - env_all_off_acc if env_all_off_acc is not None else None
        )
        seed_info = SeedInfo.from_seed_state(
            seed_state,
            seed_params_for_slot,
            counterfactual_total_improvement=counterfactual_total_improvement,
        )

        # Initialize reward_components to None (only populated for CONTRIBUTION family)
        reward_components: RewardComponentsTelemetry | None = None

        if reward_family_enum == RewardFamily.CONTRIBUTION:
            stable_val_acc = None
            if env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW:
                window = env_reward_configs[env_idx].escrow_stable_window
                if window <= 0:
                    raise ValueError(
                        f"escrow_stable_window must be positive, got {window}"
                    )
                acc_history = signals.accuracy_history
                if not acc_history:
                    raise RuntimeError(
                        "ESCROW stable accuracy requested before any accuracy history exists"
                    )
                k = window if window <= len(acc_history) else len(acc_history)
                stable_val_acc = min(acc_history[-k:])
            escrow_credit_prev = env_state.escrow_credit[target_slot]
            fossilized_seed_params = 0
            # D2: Count active seeds (non-fossilized seeds with state)
            # Active seeds are in GERMINATED, TRAINING, BLENDING, or HOLDING stages.
            # PRUNED seeds are excluded: once Tamiyo orders a prune, the seed is "dead"
            # even if alpha hasn't decayed to 0 yet. Penalizing the decay period would
            # create perverse incentives to delay pruning until the last moment.
            n_active_seeds = 0
            for slot_id in slots:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                slot_seed_state = slot_obj.state
                if slot_seed_state is None or slot_seed_state.metrics is None:
                    continue
                if slot_seed_state.stage == SeedStage.FOSSILIZED:
                    fossilized_seed_params += int(
                        slot_seed_state.metrics.seed_param_count
                    )
                    if slot_obj.alpha_schedule is not None:
                        fossilized_seed_params += sum(
                            p.numel() for p in slot_obj.alpha_schedule.parameters()
                        )
                elif slot_seed_state.stage == SeedStage.PRUNED:
                    # PRUNED seeds don't count as active - Tamiyo already decided their fate
                    pass
                else:
                    # Non-fossilized, non-pruned seed with state = active seed
                    n_active_seeds += 1
            acc_at_germination = (
                env_state.acc_at_germination[target_slot]
                if target_slot in env_state.acc_at_germination
                else None
            )
            seed_id = seed_state.seed_id if seed_state is not None else None
            force_reward_components = _reward_components_required_for_state_transport(
                env_reward_configs[env_idx].reward_mode
            )
            return_components = (
                emit_reward_components_event
                or collect_reward_summary
                or force_reward_components
            )
            reward_inputs = record.contribution_reward_inputs
            reward_inputs.action = action_for_reward
            reward_inputs.seed_contribution = seed_contribution
            reward_inputs.val_acc = env_state.val_acc
            reward_inputs.seed_info = seed_info
            reward_inputs.epoch = epoch
            reward_inputs.max_epochs = max_epochs
            reward_inputs.total_params = model.total_params
            reward_inputs.host_params = host_params
            reward_inputs.acc_at_germination = acc_at_germination
            reward_inputs.acc_delta = signals.metrics.accuracy_delta
            reward_inputs.committed_val_acc = env_state.committed_val_acc
            reward_inputs.fossilized_seed_params = fossilized_seed_params
            reward_inputs.num_fossilized_seeds = env_state.seeds_fossilized
            reward_inputs.num_contributing_fossilized = (
                env_state.contributing_fossilized
            )
            reward_inputs.config = env_reward_configs[env_idx]
            reward_inputs.return_components = return_components
            reward_inputs.effective_seed_params = effective_seed_params
            reward_inputs.alpha_delta_sq_sum = alpha_delta_sq_sum
            reward_inputs.stable_val_acc = stable_val_acc
            reward_inputs.escrow_credit_prev = escrow_credit_prev
            reward_inputs.slot_id = target_slot
            reward_inputs.seed_id = seed_id
            # D2: Capacity Economics (slot saturation prevention)
            reward_inputs.n_active_seeds = n_active_seeds

            # Drip reward parameters (BASIC_PLUS mode: post-fossilization accountability)
            # Pass existing drip states and build contributions from counterfactual baselines
            reward_inputs.fossilized_drip_states = env_state.fossilized_drip_states
            if env_state.fossilized_drip_states:
                # Build fossilized_contributions from current counterfactual measurements
                # Each drip state's seed_id maps to its current contribution
                fossilized_contributions: dict[str, float] = {}
                for drip_state in env_state.fossilized_drip_states:
                    slot_id = drip_state.slot_id
                    if slot_id in baseline_accs[env_idx]:
                        # contribution = accuracy drop when seed is disabled
                        contribution = env_state.val_acc - baseline_accs[env_idx][slot_id]
                        fossilized_contributions[drip_state.seed_id] = contribution
                reward_inputs.fossilized_contributions = fossilized_contributions
            else:
                reward_inputs.fossilized_contributions = None

            reward_result = compute_reward(reward_inputs)
            # new_drip_state is only present when reward_mode == BASIC_PLUS; None otherwise.
            new_drip_state: "FossilizedSeedDripState | None" = None
            if return_components:
                reward_tuple = cast(tuple[float, Any, ...], reward_result)
                reward, reward_components = reward_tuple[0], reward_tuple[1]
                if len(reward_tuple) == 3:
                    new_drip_state = reward_tuple[2]
                if target_slot in baseline_accs[env_idx]:
                    reward_components.host_baseline_acc = baseline_accs[env_idx][
                        target_slot
                    ]
                if (
                    env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW
                    and reward_components is not None
                ):
                    env_state.escrow_credit[target_slot] = (
                        reward_components.escrow_credit_next
                    )
            else:
                reward = cast(float, reward_result)
        else:
            loss_inputs = record.loss_reward_inputs
            loss_inputs.action = action_for_reward
            loss_inputs.loss_delta = signals.metrics.loss_delta
            loss_inputs.val_loss = env_state.val_loss
            loss_inputs.seed_info = seed_info
            loss_inputs.epoch = epoch
            loss_inputs.max_epochs = max_epochs
            loss_inputs.total_params = model.total_params
            loss_inputs.host_params = host_params
            loss_inputs.config = loss_reward_config
            reward = compute_loss_reward(loss_inputs)

        if env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW and epoch == max_epochs:
            assert reward_components is not None, (
                "RewardMode.ESCROW requires return_components=True"
            )
            escrow_forfeit = 0.0
            for slot_id in slots:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                slot_seed_state = slot_obj.state
                if slot_seed_state is None:
                    continue
                # Only forfeit escrow for seeds that are still "in play" (not terminal states).
                # FOSSILIZED: Successful integration, escrow is earned
                # PRUNED: Tamiyo already ordered removal, escrow was clawed back at prune time
                # Penalizing PRUNED seeds during alpha decay would unfairly punish timely pruning.
                if slot_seed_state.stage not in (SeedStage.FOSSILIZED, SeedStage.PRUNED):
                    escrow_forfeit += env_state.escrow_credit[slot_id]
            if escrow_forfeit != 0.0:
                reward -= escrow_forfeit
                reward_components.escrow_forfeit = -escrow_forfeit

        # Germination deposit clawback: seeds that never reached BLENDING must
        # repay the one-time PBRS germination bonus. This encourages completing
        # the scaffolding loop (GERMINATE → BLEND) instead of farming last-minute
        # germinations that never contribute.
        if (
            reward_family_enum == RewardFamily.CONTRIBUTION
            and epoch == max_epochs
            and env_reward_configs[env_idx].reward_mode
            in (RewardMode.SHAPED, RewardMode.ESCROW)
            and not env_reward_configs[env_idx].disable_pbrs
        ):
            phi_germinated = STAGE_POTENTIALS[SeedStage.GERMINATED]
            germination_bonus = (
                env_reward_configs[env_idx].pbrs_weight
                * (env_reward_configs[env_idx].gamma * phi_germinated)
            )
            germination_forfeit = 0.0
            for slot_id in slots:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                slot_seed_state = slot_obj.state
                if slot_seed_state is None:
                    continue
                if slot_seed_state.stage not in (
                    SeedStage.GERMINATED,
                    SeedStage.TRAINING,
                ):
                    continue
                seed_age_epochs = slot_seed_state.metrics.epochs_total
                discount = env_reward_configs[env_idx].gamma**seed_age_epochs
                if discount <= 0.0:
                    raise ValueError(
                        "Invalid gamma discount for germination clawback: "
                        f"gamma={env_reward_configs[env_idx].gamma} seed_age_epochs={seed_age_epochs}"
                    )
                # Discount-corrected refund: ensures the terminal clawback cancels
                # the earlier germination bonus in discounted return space.
                germination_forfeit += germination_bonus / discount
            if germination_forfeit != 0.0:
                reward -= germination_forfeit
                if reward_components is not None:
                    reward_components.action_shaping -= germination_forfeit

        reward += env_state.pending_auto_prune_penalty
        env_state.pending_auto_prune_penalty = 0.0

        # Add any pending hindsight credit BEFORE normalization
        # (DRL Specialist review: credit should go through normalizer for scale consistency)
        hindsight_credit_applied = 0.0
        if env_state.pending_hindsight_credit > 0:
            hindsight_credit_applied = env_state.pending_hindsight_credit
            reward += hindsight_credit_applied
            env_state.pending_hindsight_credit = 0.0
            # Populate RewardComponentsTelemetry for shaped_reward_ratio calculation
            if collect_reward_summary and reward_components is not None:
                reward_components.hindsight_credit = hindsight_credit_applied

        if reward_components is not None:
            reward_components.total_reward = reward
        action_outcome.reward_raw = reward
        action_outcome.reward_components = reward_components

        # Normalize reward for PPO stability (P1-6 fix)
        normalized_reward = reward_normalizer.update_and_normalize(reward)
        action_outcome.reward_normalized = normalized_reward
        # B11-CR-03 fix: Store RAW rewards for telemetry interpretability
        # PPO buffer uses normalized_reward (for training stability)
        # Telemetry uses raw reward (for cross-run comparability)
        env_state.episode_rewards.append(reward)

        if collect_reward_summary and reward_components is not None:
            summary = reward_summary_accum[env_idx]
            summary.total_reward += reward
            if reward_components.bounded_attribution is not None:
                summary.bounded_attribution += reward_components.bounded_attribution
            summary.compute_rent += reward_components.compute_rent
            summary.alpha_shock += reward_components.alpha_shock
            summary.hindsight_credit += hindsight_credit_applied
            summary.count += 1

        # Execute action
        # Stream safety: lifecycle ops can create/move CUDA tensors (germination
        # validation probes, module moves, etc). Run them on env_state.stream.
        lifecycle_ctx = (
            torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
        )
        with lifecycle_ctx:
            if slot_is_enabled and mutation_allowed:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[target_slot])
                if morphology_context is not None:
                    slot_obj.set_pending_morphology_context(morphology_context)
                    emitters[env_idx].emit(TelemetryEvent(
                        event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                        data=_build_morphology_causal_log_payload(
                            phase="mutation",
                            context=morphology_context,
                            env_idx=env_idx,
                            blueprint_id=morphology_blueprint_id,
                            governor_approved=True,
                            governor_reason="approved",
                            linked_event_id=morphology_context.mutation_id,
                        ),
                        severity="debug",
                        message="Morphology mutation dispatch",
                    ))
                handler_ctx = HandlerContext(
                    env_idx=env_idx,
                    slot_id=target_slot,
                    env_state=env_state,
                    model=model,
                    slot=slot_obj,
                    seed_state=seed_state,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    episodes_completed=episodes_completed,
                )
                try:
                    handler = get_handler(op_action)
                    if op_action == OP_GERMINATE:
                        handler_result = handler(
                            handler_ctx,
                            GerminateParams(
                                blueprint_idx=blueprint_action,
                                style_idx=style_action,
                                tempo_idx=tempo_action,
                                alpha_target=alpha_target,
                            ),
                        )
                    elif op_action == OP_FOSSILIZE:
                        handler_result = handler(
                            handler_ctx,
                            seed_info,
                            context.fossilize_active_seed,
                        )
                        if handler_result.success:
                            if collect_reward_summary:
                                summary = reward_summary_accum[env_idx]
                                summary.scaffold_count += int(
                                    handler_result.telemetry["scaffold_count"]
                                )
                                summary.scaffold_delay_total += (
                                    handler_result.telemetry["avg_scaffold_delay"]
                                    * handler_result.telemetry["scaffold_count"]
                                )
                            # Collect drip state for BASIC_PLUS mode post-fossilization
                            # accountability. new_drip_state is returned as the third element
                            # of the compute_reward() tuple when action=FOSSILIZE; collect it
                            # only after fossilization succeeds.
                            if new_drip_state is not None:
                                env_state.fossilized_drip_states.append(new_drip_state)
                    elif op_action == OP_PRUNE:
                        handler_result = handler(
                            handler_ctx,
                            PruneParams(
                                alpha_speed_idx=alpha_speed_action,
                                alpha_curve_idx=alpha_curve_action,
                            ),
                            seed_info,
                        )
                    elif op_action == OP_SET_ALPHA_TARGET:
                        handler_result = handler(
                            handler_ctx,
                            AlphaTargetParams(
                                alpha_target_idx=alpha_target_action,
                                alpha_speed_idx=alpha_speed_action,
                                alpha_curve_idx=alpha_curve_action,
                                style_idx=style_action,
                            ),
                        )
                    elif op_action in (OP_ADVANCE, OP_WAIT):
                        handler_result = handler(handler_ctx)
                    else:
                        raise ValueError(f"Unsupported lifecycle operation: {op_action}")
                finally:
                    slot_obj.clear_pending_morphology_context()
                action_success = handler_result.success
                if morphology_context is not None:
                    # KTS-002: These rows describe the mutation DISPATCH on this step.
                    # No genuine delayed post-mutation measurement exists here —
                    # env_state.val_loss is the PRE-mutation loss for this step, so it
                    # MUST NOT be attached as watch/audit evidence. We therefore emit a
                    # "dispatch" phase (the mutation was handed to the handler) and the
                    # terminal commit/fossilization outcome WITHOUT watch_window_evidence,
                    # rather than mislabelling pre-mutation loss as post-mutation evidence.
                    # A real watch window would require deferring evidence to a later
                    # epoch's validation pass (future work), not a same-step placeholder.
                    emitters[env_idx].emit(TelemetryEvent(
                        event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                        data=_build_morphology_causal_log_payload(
                            phase="dispatch",
                            context=morphology_context,
                            env_idx=env_idx,
                            blueprint_id=morphology_blueprint_id,
                            governor_approved=True,
                            governor_reason="approved",
                            linked_event_id=morphology_context.mutation_id,
                        ),
                        severity="debug",
                        message="Morphology mutation dispatched",
                    ))
                    # Only emit a terminal commit/fossilization row when the
                    # handler actually mutated. A declined handler (failed ADVANCE
                    # gate, FOSSILIZE G5 check) returns success=False after governor
                    # approval; emitting phase='commit' there would mark a no-op as a
                    # committed mutation and corrupt the proof/audit trail. The
                    # 'dispatch' row already records the attempt; no terminal row
                    # follows a declined handler.
                    if handler_result.success:
                        terminal_phase: MorphologyCausalLogPhase = (
                            "fossilization"
                            if op_action == OP_FOSSILIZE
                            else "commit"
                        )
                        emitters[env_idx].emit(TelemetryEvent(
                            event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                            data=_build_morphology_causal_log_payload(
                                phase=terminal_phase,
                                context=morphology_context,
                                env_idx=env_idx,
                                blueprint_id=morphology_blueprint_id,
                                governor_approved=True,
                                governor_reason="approved",
                                linked_event_id=morphology_context.mutation_id,
                            ),
                            severity="debug",
                            message="Morphology dispatch committed",
                        ))
            elif op_action == OP_WAIT:
                action_success = True

        if action_success:
            env_state.successful_action_counts[action_for_reward.name] = (
                env_state.successful_action_counts.get(action_for_reward.name, 0) + 1
            )
        action_outcome.action_success = action_success
        if reward_components is not None:
            reward_components.action_success = action_success

        # Obs V3: Update action feedback state for next timestep's feature extraction
        env_state.last_action_success = action_success
        env_state.last_action_op = op_action

        # Obs V3: Update gradient health history for LSTM trend detection
        # Use slot reports from current timestep (before action) as "prev" for next timestep
        slot_reports_for_env = all_slot_reports[env_idx]
        for slot_id in ordered_slots:
            if slot_id in slot_reports_for_env:
                report = slot_reports_for_env[slot_id]
                if report.telemetry is not None:
                    health_val = report.telemetry.gradient_health
                    # Unmeasured gradient health (None) stays ABSENT from the
                    # tracking dict so it encodes as UNKNOWN (Obs V3 sentinel),
                    # never a fabricated measured value (KTS-001/TPD-003). Only a
                    # measured value is stored; NaN/inf is still a real bug that
                    # would poison observation features and crash get_action().
                    if health_val is not None:
                        if not math.isfinite(health_val):
                            raise ValueError(
                                f"NaN/inf gradient_health from telemetry for slot {slot_id}: "
                                f"{health_val}. Check materialize_grad_stats() or sync_telemetry()."
                            )
                        env_state.gradient_health_prev[slot_id] = health_val

                # Obs V3: Increment epochs since last counterfactual measurement
                # This is reset to 0 when counterfactual_contribution is updated (see line ~2191)
                if slot_id in env_state.epochs_since_counterfactual:
                    env_state.epochs_since_counterfactual[slot_id] += 1
                else:
                    # Initialize tracking for new slots
                    env_state.epochs_since_counterfactual[slot_id] = 0

        # Consolidate telemetry via emitter
        if ops_telemetry_enabled and masked_np is not None:
            masked_flags = record.mask_flags
            masked_flags.op_masked = bool(masked_np[_HEAD_OP_IDX, env_idx])
            masked_flags.slot_masked = bool(masked_np[_HEAD_SLOT_IDX, env_idx])
            masked_flags.blueprint_masked = bool(
                masked_np[_HEAD_BLUEPRINT_IDX, env_idx]
            )
            masked_flags.style_masked = bool(masked_np[_HEAD_STYLE_IDX, env_idx])
            masked_flags.tempo_masked = bool(masked_np[_HEAD_TEMPO_IDX, env_idx])
            masked_flags.alpha_target_masked = bool(
                masked_np[_HEAD_ALPHA_TARGET_IDX, env_idx]
            )
            masked_flags.alpha_speed_masked = bool(
                masked_np[_HEAD_ALPHA_SPEED_IDX, env_idx]
            )
            masked_flags.alpha_curve_masked = bool(
                masked_np[_HEAD_ALPHA_CURVE_IDX, env_idx]
            )

            post_slot_obj = cast(SeedSlotProtocol, model.seed_slots[target_slot])
            post_slot_state = post_slot_obj.state
            active_algo = (
                post_slot_state.alpha_algorithm.name if post_slot_state else None
            )
            slot_reports_for_decision = all_slot_reports[env_idx]
            decision_slot_states: dict[str, str] = {}
            for slot_id in ordered_slots:
                if slot_id not in slot_reports_for_decision:
                    decision_slot_states[slot_id] = "Empty"
                    continue
                slot_report = slot_reports_for_decision[slot_id]
                stage_label = slot_report.stage.name.title()
                decision_slot_states[slot_id] = (
                    f"{stage_label} {slot_report.metrics.total_improvement:.0f}%"
                )

            # Compute action_confidence, alternatives, and decision_entropy from op_logits
            # PolicyBundle returns op_logits for telemetry when available
            # PERF: op_probs_cpu is pre-computed BEFORE the env loop to avoid N GPU syncs
            action_confidence = None
            alternatives: list[tuple[str, float]] | None = None
            decision_entropy = None
            if op_probs_cpu is not None and env_idx < op_probs_cpu.shape[0]:
                probs_cpu = op_probs_cpu[env_idx]
                chosen_op = op_action

                # action_confidence = P(chosen_op)
                if 0 <= chosen_op < len(probs_cpu):
                    action_confidence = float(probs_cpu[chosen_op])

                # alternatives = top-2 ops excluding chosen
                ranked = sorted(
                    enumerate(probs_cpu), key=lambda x: x[1], reverse=True
                )
                alternatives = [
                    (OP_NAMES[op_idx], float(prob))
                    for op_idx, prob in ranked
                    if op_idx != chosen_op
                ][:2]

                # decision_entropy = -sum(p * log(p)) for op head
                entropy_sum = 0.0
                for p in probs_cpu:
                    if p > 1e-8:  # Avoid log(0)
                        entropy_sum -= p * math.log(p)
                decision_entropy = entropy_sum

            head_telem = _build_decision_head_telemetry(
                head_confidences_cpu=head_confidences_cpu,
                head_entropies_cpu=head_entropies_cpu,
                env_idx=env_idx,
            )

            emitters[env_idx].on_last_action(
                epoch=epoch,
                action_spec=action_spec,
                masked=masked_flags,
                outcome=action_outcome,
                active_alpha_algorithm=active_algo,
                value_estimate=value,
                host_accuracy=env_state.val_acc,
                slot_states=decision_slot_states,
                action_confidence=action_confidence,
                alternatives=alternatives,
                decision_entropy=decision_entropy,
                head_telemetry=head_telem,
                # TELE-OBS: Only pass for env 0 to avoid redundant data (batch-level stat)
                observation_stats=step_obs_stats if env_idx == 0 else None,
            )

        # Store transition directly into rollout buffer.
        done = epoch == max_epochs
        truncated = done
        action_outcome.truncated = truncated
        effective_op_action = int(action_for_reward)

        # D5: Detect forced step from pre-computed array (no GPU sync in loop)
        # When all slots are occupied and no operations are valid, the action space
        # collapses to WAIT-only. These timesteps should be excluded from actor loss
        # (no agency, no gradient) but still included in GAE/LSTM unrolling.
        forced_step = proof_controlled_step or forced_batch_cpu[env_idx]

        # Phase 2.2: Build contribution targets from counterfactual ablation
        # baseline_accs[env_idx][slot_id] = accuracy with that slot disabled
        # contribution = val_acc - baseline_acc (positive = slot helps)
        # DRL Expert: Only set has_fresh_contribution=True when counterfactual measured
        env_baseline_accs = baseline_accs[env_idx]
        fresh_contribution = build_fresh_contribution_targets(
            env_state=env_state,
            baseline_accs=env_baseline_accs,
            slot_config=slot_config,
            device=device,
        )
        has_fresh_contribution = fresh_contribution.has_fresh_contribution
        contribution_targets_tensor = fresh_contribution.contribution_targets
        contribution_mask_tensor = fresh_contribution.contribution_mask

        step_idx = agent.buffer.step_counts[env_idx]
        agent.buffer.add(
            env_id=env_idx,
            state=states_batch_normalized[env_idx].detach(),
            blueprint_indices=blueprint_indices_batch[env_idx].detach(),
            slot_action=slot_action,
            blueprint_action=blueprint_action,
            style_action=style_action,
            tempo_action=tempo_action,
            alpha_target_action=alpha_target_action,
            alpha_speed_action=alpha_speed_action,
            alpha_curve_action=alpha_curve_action,
            op_action=op_action,
            effective_op_action=effective_op_action,
            slot_log_prob=slot_log_probs_batch[env_idx],
            blueprint_log_prob=blueprint_log_probs_batch[env_idx],
            style_log_prob=style_log_probs_batch[env_idx],
            tempo_log_prob=tempo_log_probs_batch[env_idx],
            alpha_target_log_prob=alpha_target_log_probs_batch[env_idx],
            alpha_speed_log_prob=alpha_speed_log_probs_batch[env_idx],
            alpha_curve_log_prob=alpha_curve_log_probs_batch[env_idx],
            op_log_prob=op_log_probs_batch[env_idx],
            value=value,
            reward=normalized_reward,
            done=done,
            slot_mask=slot_by_op_masks_batch[env_idx, op_action],
            blueprint_mask=blueprint_masks_batch[env_idx],
            style_mask=style_masks_batch[env_idx],
            tempo_mask=tempo_masks_batch[env_idx],
            alpha_target_mask=alpha_target_masks_batch[env_idx],
            alpha_speed_mask=alpha_speed_masks_batch[env_idx],
            alpha_curve_mask=alpha_curve_masks_batch[env_idx],
            op_mask=op_masks_batch[env_idx],
            hidden_h=pre_step_hiddens[env_idx][0].detach(),
            hidden_c=pre_step_hiddens[env_idx][1].detach(),
            truncated=truncated,
            bootstrap_value=0.0,
            forced_step=forced_step,
            action_id=action_id,
            # Phase 2.2: Counterfactual contribution supervision targets
            contribution_targets=contribution_targets_tensor,
            contribution_mask=contribution_mask_tensor,
            has_fresh_contribution=has_fresh_contribution,
        )
        if truncated:
            truncated_bootstrap_targets.append((env_idx, step_idx))

        if done:
            agent.buffer.end_episode(env_id=env_idx)
            # NOTE: Do NOT reset batched_lstm_hidden here. The bootstrap value computation
            # (after the epoch loop) requires the carried episode hidden state to correctly
            # estimate V(s_{t+1}) for truncated episodes. Resetting to initial_hidden() would
            # bias the GAE computation by computing V(s_{t+1}) with a "memory-wiped" agent.
            # The next rollout will initialize fresh hidden states anyway (line 1747).

        # Mechanical lifecycle advance
        for slot_id in slots:
            slot_for_step = cast(SeedSlotProtocol, model.seed_slots[slot_id])

            # Advance lifecycle (may set auto_pruned if scheduled prune completes)
            slot_for_step.step_epoch()

            # Check auto-prune flag AFTER step_epoch to catch both:
            # - Governor prunes (set flag outside step_epoch, caught on next check)
            # - Scheduled prune completions (set flag inside step_epoch, caught immediately)
            if slot_for_step.state and slot_for_step.state.metrics.auto_pruned:
                env_state.pending_auto_prune_penalty += reward_config.auto_prune_penalty
                # Clear one-shot flag after reading
                slot_for_step.state.metrics.auto_pruned = False

            if not model.has_active_seed_in_slot(slot_id):
                env_state.seed_optimizers.pop(slot_id, None)
                env_state.acc_at_germination.pop(slot_id, None)
                env_state.gradient_ratio_ema.pop(slot_id, None)
                env_state.escrow_credit[slot_id] = 0.0
            if slot_for_step.state is None:
                env_state.clear_obs_v3_slot_tracking(slot_id)

        # Fix BUG-022: Collect bootstrap state AFTER mechanical advance
        if truncated:
            all_post_action_signals.append(
                env_state.signal_tracker.peek(
                    epoch=epoch,
                    global_step=epoch * num_train_batches,
                    train_loss=env_state.train_loss,
                    train_accuracy=env_state.train_acc,
                    val_loss=env_state.val_loss,
                    val_accuracy=env_state.val_acc,
                    active_seeds=cast(
                        list[SeedStateProtocol],
                        [
                            s.state
                            for s in model.seed_slots.values()
                            if s.is_active and s.state
                        ],
                    ),
                    available_slots=sum(
                        1 for s in model.seed_slots.values() if s.state is None
                    ),
                )
            )
            all_post_action_slot_reports.append(model.get_slot_reports())
            all_post_action_masks.append(
                compute_action_masks(
                    slot_states=build_slot_states(
                        all_post_action_slot_reports[-1], ordered_slots
                    ),
                    enabled_slots=ordered_slots,
                    total_seeds=model.total_seeds(),
                    max_seeds=effective_max_seeds,
                    slot_config=slot_config,
                    device=torch.device(device),
                    topology=task_spec.topology,
                    disable_advance=context.disable_advance,
                )
            )

        if epoch == max_epochs:
            record.env_final_acc = env_state.val_acc
            record.env_total_reward = sum(env_state.episode_rewards)

            # Track episode completion for A/B testing
            episode_history.append(
                EpisodeRecord(
                    env_id=env_idx,
                    episode_reward=record.env_total_reward,
                    final_accuracy=record.env_final_acc,
                )
            )
            action_outcome.episode_reward = record.env_total_reward
            action_outcome.final_accuracy = record.env_final_acc

            # Compute stability score from reward variance
            recent_ep_rewards = (
                env_state.episode_rewards[-20:]
                if len(env_state.episode_rewards) >= 20
                else env_state.episode_rewards
            )
            if len(recent_ep_rewards) > 1:
                reward_var = float(np.var(recent_ep_rewards))
                stability = 1.0 / (1.0 + reward_var)
            else:
                stability = 1.0  # Default if insufficient data

            # Create EpisodeOutcome for Pareto analysis
            episode_outcome = EpisodeOutcome(
                env_id=env_idx,
                episode_idx=episodes_completed + env_idx,
                final_accuracy=env_state.val_acc,
                param_ratio=model.total_params / max(1, host_params_baseline),
                num_fossilized=env_state.seeds_fossilized,
                num_contributing_fossilized=env_state.contributing_fossilized,
                episode_reward=record.env_total_reward,
                stability_score=stability,
                reward_mode=env_reward_configs[env_idx].reward_mode.value,
            )
            episode_outcomes.append(episode_outcome)
            action_outcome.episode_outcome = episode_outcome

            # Emit EPISODE_OUTCOME telemetry for Pareto analysis
            # B11-CR-04 fix: Skip emission for rollback episodes (will emit corrected outcome later)
            if env_state.telemetry_cb and not record.rollback_occurred:
                # TELE-610: Classify episode outcome (percent-scale accuracy).
                outcome_type = classify_episode_outcome(env_state.val_acc)

                env_state.telemetry_cb(
                    TelemetryEvent(
                        event_type=TelemetryEventType.EPISODE_OUTCOME,
                        epoch=episodes_completed + env_idx,
                        data=EpisodeOutcomePayload(
                            env_id=env_idx,
                            episode_idx=episode_outcome.episode_idx,
                            final_accuracy=episode_outcome.final_accuracy,
                            param_ratio=episode_outcome.param_ratio,
                            num_fossilized=episode_outcome.num_fossilized,
                            num_contributing_fossilized=episode_outcome.num_contributing_fossilized,
                            episode_reward=episode_outcome.episode_reward,
                            stability_score=episode_outcome.stability_score,
                            reward_mode=episode_outcome.reward_mode,
                            # TELE-610: Episode diagnostics
                            episode_length=epoch,  # Current epoch = episode length
                            outcome_type=outcome_type,
                            germinate_count=env_state.action_counts["GERMINATE"],
                            prune_count=env_state.action_counts["PRUNE"],
                            fossilize_count=env_state.action_counts["FOSSILIZE"],
                        ),
                    )
                )

            # Shapley contributions at episode end
            if env_state.counterfactual_helper is not None and baseline_accs[env_idx]:
                active_slot_ids = [
                    sid
                    for sid in slots
                    if model.has_active_seed_in_slot(sid)
                    and cast(SeedSlotProtocol, model.seed_slots[sid]).alpha > 0
                ]
                if active_slot_ids:
                    shapley_epoch = batch_idx + 1
                    has_exact_results = (
                        env_state.counterfactual_helper.has_precomputed_matrix_for(
                            active_slot_ids,
                            epoch=shapley_epoch,
                        )
                    )
                    if has_exact_results:
                        continue

                    cached_baselines = baseline_accs[env_idx]

                    def eval_fn(alpha_settings: dict[str, float]) -> tuple[float, float]:
                        if all(a >= 0.99 for a in alpha_settings.values()):
                            return env_state.val_loss, env_state.val_acc
                        disabled = [s for s, a in alpha_settings.items() if a < 0.01]
                        if (
                            len(disabled) == 1
                            and disabled[0] in cached_baselines
                        ):
                            return (
                                env_state.val_loss * 1.1,
                                cached_baselines[disabled[0]],
                            )
                        if disabled:
                            return env_state.val_loss * 1.2, sum(
                                cached_baselines[s]
                                for s in disabled
                            ) / len(disabled)
                        return env_state.val_loss, env_state.val_acc

                    try:
                        env_state.counterfactual_helper.compute_contributions(
                            slot_ids=active_slot_ids,
                            evaluate_fn=eval_fn,
                            epoch=shapley_epoch,
                        )
                    except (KeyError, ZeroDivisionError, ValueError) as e:
                        # HIGH-01 fix: Narrow to expected failures in Shapley computation
                        _logger.warning(f"Shapley failed for env {env_idx}: {e}")

    return ActionExecutionResult(
        truncated_bootstrap_targets=truncated_bootstrap_targets,
        post_action_signals=all_post_action_signals,
        post_action_slot_reports=all_post_action_slot_reports,
        post_action_masks=all_post_action_masks,
    )
