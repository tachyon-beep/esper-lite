from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
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
    DEFAULT_GAMMA,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    EPISODE_SUCCESS_THRESHOLD,
    EpisodeOutcome,
    EpisodeOutcomePayload,
    HEAD_NAMES,
    HINDSIGHT_CREDIT_WEIGHT,
    HeadTelemetry,
    LifecycleOp,
    MAX_HINDSIGHT_CREDIT,
    MIN_PRUNE_AGE,
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
    TEMPO_TO_EPOCHS,
    TelemetryEvent,
    TelemetryEventType,
    TempoAction,
)
from esper.simic.rewards import (
    compute_reward,
    compute_loss_reward,
    compute_scaffold_hindsight_credit,
    ContributionRewardConfig,
    RewardFamily,
    RewardMode,
    SeedInfo,
    STAGE_POTENTIALS,
)
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks

from .helpers import compute_rent_and_shock_inputs
from .parallel_env_state import ParallelEnvState

if TYPE_CHECKING:
    from esper.leyline.reports import SeedStateReport
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
    from esper.simic.telemetry.emitters import VectorizedEmitter
    from esper.simic.control import RewardNormalizer
    from esper.simic.agent import PPOAgent
    from esper.leyline.signals import TrainingSignals
    from esper.runtime.tasks import TaskSpec
    from esper.nissa import BlueprintAnalytics


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
) -> tuple[str, bool, Any, Any, bool, LifecycleOp, str, Any, float]:
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


def execute_actions(
    *,
    context: ActionExecutionContext,
    env_states: list[ParallelEnvState],
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
    governor_panic_envs: list[int],
    env_rollback_occurred: list[bool],
    reward_summary_accum: list[dict[str, float]],
    env_final_accs: list[float],
    env_total_rewards: list[float],
    episode_history: list[dict[str, Any]],
    episode_outcomes: list[EpisodeOutcome],
    step_obs_stats: Any | None,
    epoch: int,
    episodes_completed: int,
    batch_idx: int,
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

        action_dict: dict[str, int] | None = None
        if ops_telemetry_enabled:
            action_dict = {
                "slot": slot_action,
                "blueprint": blueprint_action,
                "style": style_action,
                "tempo": tempo_action,
                "alpha_target": alpha_target_action,
                "alpha_speed": alpha_speed_action,
                "alpha_curve": alpha_curve_action,
                "op": op_action,
            }
        (
            target_slot,
            slot_is_enabled,
            slot_state,
            seed_state,
            action_valid_for_reward,
            action_for_reward,
            blend_algorithm_id,
            alpha_algorithm,
            alpha_target,
        ) = _parse_sampled_action(
            env_idx,
            op_action,
            slot_action,
            style_action,
            alpha_target_action,
            slots,
            slot_config,
            model,
            context.resolve_target_slot,
        )

        # Use op name for action counting
        env_state.action_counts[action_for_reward.name] = (
            env_state.action_counts.get(action_for_reward.name, 0) + 1
        )

        action_success = False

        # Governor rollback
        if env_idx in governor_panic_envs:
            # Stream safety: rollback mutates model tensors; ensure it runs on the
            # per-env CUDA stream to avoid default-stream leakage and races.
            rollback_ctx = (
                torch.cuda.stream(env_state.stream)
                if env_state.stream
                else nullcontext()
            )
            with rollback_ctx:
                env_state.governor.execute_rollback(env_id=env_idx)
            env_rollback_occurred[env_idx] = True

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

        # Compute reward
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
        seed_info = SeedInfo.from_seed_state(seed_state, seed_params_for_slot)

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
            acc_at_germination = (
                env_state.acc_at_germination[target_slot]
                if target_slot in env_state.acc_at_germination
                else None
            )
            seed_id = seed_state.seed_id if seed_state is not None else None
            force_reward_components = (
                env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW
            )
            if (
                emit_reward_components_event
                or collect_reward_summary
                or force_reward_components
            ):
                reward, reward_components = cast(
                    tuple[float, Any],
                    compute_reward(
                        action=action_for_reward,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        seed_info=seed_info,
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=model.total_params,
                        host_params=host_params,
                        acc_at_germination=acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        committed_val_acc=env_state.committed_val_acc,
                        fossilized_seed_params=fossilized_seed_params,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                        config=env_reward_configs[env_idx],
                        return_components=True,
                        effective_seed_params=effective_seed_params,
                        alpha_delta_sq_sum=alpha_delta_sq_sum,
                        stable_val_acc=stable_val_acc,
                        escrow_credit_prev=escrow_credit_prev,
                        slot_id=target_slot,
                        seed_id=seed_id,
                    ),
                )
                if target_slot in baseline_accs[env_idx]:
                    reward_components.host_baseline_acc = baseline_accs[env_idx][
                        target_slot
                    ]
                if force_reward_components and reward_components is not None:
                    env_state.escrow_credit[target_slot] = (
                        reward_components.escrow_credit_next
                    )
            else:
                reward = cast(
                    float,
                    compute_reward(
                        action=action_for_reward,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        seed_info=seed_info,
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=model.total_params,
                        host_params=host_params,
                        acc_at_germination=acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        committed_val_acc=env_state.committed_val_acc,
                        fossilized_seed_params=fossilized_seed_params,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                        config=env_reward_configs[env_idx],
                        effective_seed_params=effective_seed_params,
                        alpha_delta_sq_sum=alpha_delta_sq_sum,
                        stable_val_acc=stable_val_acc,
                        escrow_credit_prev=escrow_credit_prev,
                        slot_id=target_slot,
                        seed_id=seed_id,
                    ),
                )
        else:
            reward = compute_loss_reward(
                action=action_for_reward,
                loss_delta=signals.metrics.loss_delta,
                val_loss=env_state.val_loss,
                seed_info=seed_info,
                epoch=epoch,
                max_epochs=max_epochs,
                total_params=model.total_params,
                host_params=host_params,
                config=loss_reward_config,
            )

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
                if slot_seed_state.stage != SeedStage.FOSSILIZED:
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

        # Normalize reward for PPO stability (P1-6 fix)
        normalized_reward = reward_normalizer.update_and_normalize(reward)
        # B11-CR-03 fix: Store RAW rewards for telemetry interpretability
        # PPO buffer uses normalized_reward (for training stability)
        # Telemetry uses raw reward (for cross-run comparability)
        env_state.episode_rewards.append(reward)

        if collect_reward_summary and reward_components is not None:
            summary = reward_summary_accum[env_idx]
            summary["total_reward"] += reward
            if reward_components.bounded_attribution is not None:
                summary["bounded_attribution"] += reward_components.bounded_attribution
            summary["compute_rent"] += reward_components.compute_rent
            summary["alpha_shock"] += reward_components.alpha_shock
            summary["hindsight_credit"] += hindsight_credit_applied
            summary["count"] += 1

        # Execute action
        # Stream safety: lifecycle ops can create/move CUDA tensors (germination
        # validation probes, module moves, etc). Run them on env_state.stream.
        lifecycle_ctx = (
            torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
        )
        with lifecycle_ctx:
            if slot_is_enabled:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[target_slot])
                if op_action == OP_GERMINATE and model.seed_slots[target_slot].state is None:
                    env_state.acc_at_germination[target_slot] = env_state.val_acc
                    env_state.escrow_credit[target_slot] = 0.0
                    blueprint_id = BLUEPRINT_IDS[blueprint_action]
                    assert blueprint_id is not None, (
                        "NULL blueprint should not reach germination"
                    )
                    model.germinate_seed(
                        blueprint_id,
                        f"ep{episodes_completed + env_idx}_env{env_idx}_seed_{env_state.seeds_created}",
                        slot=target_slot,
                        blend_algorithm_id=blend_algorithm_id,
                        blend_tempo_epochs=TEMPO_TO_EPOCHS[TempoAction(tempo_action)],
                        alpha_algorithm=alpha_algorithm,
                        alpha_target=alpha_target,
                    )
                    env_state.init_obs_v3_slot_tracking(target_slot)
                    env_state.seeds_created += 1
                    env_state.germinate_count += 1  # TELE-610
                    env_state.seed_optimizers.pop(target_slot, None)
                    action_success = True
                elif op_action == OP_FOSSILIZE:
                    action_success = context.fossilize_active_seed(model, target_slot)
                    if action_success:
                        env_state.seeds_fossilized += 1
                        env_state.fossilize_count += 1  # TELE-610
                        if seed_info is not None and (
                            seed_info.total_improvement
                            >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
                        ):
                            env_state.contributing_fossilized += 1
                        env_state.acc_at_germination.pop(target_slot, None)

                        # Compute temporally-discounted hindsight credit for scaffolds
                        beneficiary_improvement = (
                            seed_info.total_improvement if seed_info else 0.0
                        )
                        if beneficiary_improvement > 0:
                            # Use outer loop epoch variable (not per-env counter)
                            current_epoch = epoch
                            total_credit = 0.0
                            scaffold_count = 0
                            total_delay = 0

                            # Find all scaffolds that boosted this beneficiary
                            for (
                                scaffold_slot,
                                boosts,
                            ) in env_state.scaffold_boost_ledger.items():
                                for (
                                    boost_given,
                                    beneficiary_slot,
                                    epoch_of_boost,
                                ) in boosts:
                                    if (
                                        beneficiary_slot == target_slot
                                        and boost_given > 0
                                    ):
                                        # Temporal discount: credit decays with distance
                                        delay = current_epoch - epoch_of_boost
                                        discount = DEFAULT_GAMMA**delay

                                        # Compute discounted hindsight credit
                                        raw_credit = compute_scaffold_hindsight_credit(
                                            boost_given=boost_given,
                                            beneficiary_improvement=beneficiary_improvement,
                                            credit_weight=HINDSIGHT_CREDIT_WEIGHT,
                                        )
                                        total_credit += raw_credit * discount
                                        scaffold_count += 1
                                        total_delay += delay

                            # Cap total credit to prevent runaway values
                            total_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)

                            env_state.pending_hindsight_credit += total_credit

                            # Track scaffold metrics for telemetry (per-environment)
                            if collect_reward_summary:
                                summary = reward_summary_accum[env_idx]
                                summary["scaffold_count"] += scaffold_count
                                summary["scaffold_delay_total"] += total_delay

                            # Clear this beneficiary from all ledgers (it's now fossilized)
                            for scaffold_slot in list(
                                env_state.scaffold_boost_ledger.keys()
                            ):
                                env_state.scaffold_boost_ledger[scaffold_slot] = [
                                    (b, ben, e)
                                    for (b, ben, e) in env_state.scaffold_boost_ledger[
                                        scaffold_slot
                                    ]
                                    if ben != target_slot
                                ]
                                if not env_state.scaffold_boost_ledger[scaffold_slot]:
                                    del env_state.scaffold_boost_ledger[scaffold_slot]

                        # B8-DRL-02 FIX: Clean up seed optimizer after fossilization
                        # (was missing - memory leak for fossilized seed optimizers)
                        env_state.seed_optimizers.pop(target_slot, None)

                        # BUG FIX: Trigger governor snapshot after fossilization
                        # Without this, a rollback between fossilization and the next
                        # periodic snapshot (every 5 epochs) produces an incoherent state:
                        # - Fossilized seed weights not in snapshot (excluded when TRAINING)
                        # - Fossilized seeds can't be pruned during rollback
                        # - Result: host reverts but fossilized seed keeps stale weights
                        # The snapshot must be taken OUTSIDE the CUDA stream context,
                        # so we set a flag here and take the snapshot later.
                        env_state.needs_governor_snapshot = True
                elif (
                    op_action == OP_PRUNE
                    and model.has_active_seed_in_slot(target_slot)
                    and seed_state is not None
                    and seed_state.stage
                    in (
                        SeedStage.GERMINATED,
                        SeedStage.TRAINING,
                        SeedStage.BLENDING,
                        SeedStage.HOLDING,
                    )
                    # BUG-020 fix: enforce MIN_PRUNE_AGE at execution gate
                    and seed_info is not None
                    and seed_info.seed_age_epochs >= MIN_PRUNE_AGE
                ):
                    speed_steps = ALPHA_SPEED_TO_STEPS[
                        AlphaSpeedAction(alpha_speed_action)
                    ]
                    curve_action_obj = AlphaCurveAction(alpha_curve_action)
                    curve = curve_action_obj.to_curve()

                    # Schedule prune: force alpha to 0 with requested speed/curve.
                    action_success = slot_obj.schedule_prune(
                        steps=speed_steps,
                        curve=curve,
                        initiator="policy",
                    )
                    if action_success:
                        env_state.prune_count += 1  # TELE-610
                elif op_action == OP_SET_ALPHA_TARGET and seed_state is not None:
                    speed_steps = ALPHA_SPEED_TO_STEPS[
                        AlphaSpeedAction(alpha_speed_action)
                    ]
                    curve_action_obj = AlphaCurveAction(alpha_curve_action)
                    curve = curve_action_obj.to_curve()
                    action_success = slot_obj.set_alpha_target(
                        alpha_target=alpha_target,
                        steps=speed_steps,
                        curve=curve,
                        alpha_algorithm=alpha_algorithm,
                        initiator="policy",
                    )
                elif op_action == OP_ADVANCE and seed_state is not None:
                    if seed_state.stage in (
                        SeedStage.GERMINATED,
                        SeedStage.TRAINING,
                        SeedStage.BLENDING,
                    ):
                        gate_result = slot_obj.advance_stage()
                        action_success = gate_result.passed
                elif op_action == OP_WAIT:
                    # WAIT is always a valid no-op for enabled slots.
                    action_success = True
            elif op_action == OP_WAIT:
                action_success = True

        if action_success:
            env_state.successful_action_counts[action_for_reward.name] = (
                env_state.successful_action_counts.get(action_for_reward.name, 0) + 1
            )

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
                    # Fail-fast if gradient_health contains NaN/inf
                    # This would poison observation features and crash get_action()
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
            assert action_dict is not None
            masked_flags = {
                head: bool(masked_np[head_idx, env_idx])
                for head_idx, head in enumerate(HEAD_NAMES)
            }

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

            # Build HeadTelemetry for this env (typed dataclass, not raw dict)
            head_telem: HeadTelemetry | None = None
            if head_confidences_cpu is not None:
                head_telem = HeadTelemetry(
                    op_confidence=float(head_confidences_cpu[0, env_idx]),
                    slot_confidence=float(head_confidences_cpu[1, env_idx]),
                    blueprint_confidence=float(head_confidences_cpu[2, env_idx]),
                    style_confidence=float(head_confidences_cpu[3, env_idx]),
                    tempo_confidence=float(head_confidences_cpu[4, env_idx]),
                    alpha_target_confidence=float(head_confidences_cpu[5, env_idx]),
                    alpha_speed_confidence=float(head_confidences_cpu[6, env_idx]),
                    curve_confidence=float(head_confidences_cpu[7, env_idx]),
                    # Entropy (0.0 if not available)
                    op_entropy=float(head_entropies_cpu[0, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    slot_entropy=float(head_entropies_cpu[1, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    blueprint_entropy=float(head_entropies_cpu[2, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    style_entropy=float(head_entropies_cpu[3, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    tempo_entropy=float(head_entropies_cpu[4, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    alpha_target_entropy=float(head_entropies_cpu[5, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    alpha_speed_entropy=float(head_entropies_cpu[6, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                    curve_entropy=float(head_entropies_cpu[7, env_idx])
                    if head_entropies_cpu is not None
                    else 0.0,
                )

            emitters[env_idx].on_last_action(
                epoch,
                action_dict,
                target_slot,
                masked_flags,
                action_success,
                active_algo,
                total_reward=reward,
                value_estimate=value,
                host_accuracy=env_state.val_acc,
                slot_states=decision_slot_states,
                action_confidence=action_confidence,
                alternatives=alternatives,
                decision_entropy=decision_entropy,
                reward_components=reward_components,  # Pass directly (may be None for LOSS family)
                head_telemetry=head_telem,
                # TELE-OBS: Only pass for env 0 to avoid redundant data (batch-level stat)
                observation_stats=step_obs_stats if env_idx == 0 else None,
            )

        # Store transition directly into rollout buffer.
        done = epoch == max_epochs
        truncated = done
        effective_op_action = int(action_for_reward)

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
            env_final_accs[env_idx] = env_state.val_acc
            env_total_rewards[env_idx] = sum(env_state.episode_rewards)

            # Track episode completion for A/B testing
            episode_history.append(
                {
                    "env_id": env_idx,
                    "episode_reward": env_total_rewards[env_idx],
                    "final_accuracy": env_final_accs[env_idx],
                }
            )

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
                param_ratio=(model.total_params - host_params_baseline)
                / max(1, host_params_baseline),
                num_fossilized=env_state.seeds_fossilized,
                num_contributing_fossilized=env_state.contributing_fossilized,
                episode_reward=env_total_rewards[env_idx],
                stability_score=stability,
                reward_mode=env_reward_configs[env_idx].reward_mode.value,
            )
            episode_outcomes.append(episode_outcome)

            # Emit EPISODE_OUTCOME telemetry for Pareto analysis
            # B11-CR-04 fix: Skip emission for rollback episodes (will emit corrected outcome later)
            if env_state.telemetry_cb and not env_rollback_occurred[env_idx]:
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
                                cached_baselines.get(s, env_state.val_acc)
                                for s in disabled
                            ) / len(disabled)
                        return env_state.val_loss, env_state.val_acc

                    try:
                        env_state.counterfactual_helper.compute_contributions(
                            slot_ids=active_slot_ids,
                            evaluate_fn=eval_fn,
                            epoch=batch_idx + 1,
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
