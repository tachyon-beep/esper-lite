from __future__ import annotations

import logging
import os
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast

import numpy as np
import torch
import torch.nn as nn

from esper.leyline import (
    AlphaAlgorithm,
    FactoredAction,
    HEAD_NAMES,
    LifecycleOp,
    SeedSlotProtocol,
    SeedStage,
    TelemetryEvent,
    TelemetryEventType,
    TopologyManifestPayload,
)
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
    fixed_schedule_action_for_epoch,
    static_final_source_action_for_epoch,
)
from esper.leyline.slot_id import validate_slot_ids
from esper.simic.telemetry import (
    AnomalyDetector,
    GradientEMATracker,
    compute_observation_stats,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    training_profiler,
)
from esper.simic.telemetry.emitters import check_performance_degradation
from esper.simic.rewards import ContributionRewardInputs, LossRewardInputs
from esper.tamiyo.policy.action_masks import (
    MaskedCategorical,
    build_slot_states,
    compute_action_masks,
)
from esper.tamiyo.policy.features import batch_obs_to_features
from esper.utils.data import augment_cifar10_batch

from .action_execution import ActionExecutionContext, ResolveTargetSlot, execute_actions
from .batch_ops import batch_signals_to_features, process_train_batch
from .helpers import policy_amp_context
from .normalizer_checkpoint import obs_normalizer_metadata
from .counterfactual_eval import process_fused_val_batch
from .parallel_env_state import ParallelEnvState
from .env_factory import (
    EnvFactoryContext,
    configure_slot_telemetry,
    create_env_state,
    make_env_seed,
)
from .ppo_coordinator import PPOCoordinator, PPOCoordinatorConfig
from .static_final_replay import (
    TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY,
    capture_source_final_manifest,
    capture_static_final_replay_manifest,
    materialize_static_final_topology,
)
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    BatchSummary,
    EnvStepRecord,
    EpisodeRecord,
    RewardSummaryAccumulator,
)

_logger = logging.getLogger(__name__)


_PROOF_CONTROLLED_LIFECYCLE_POLICIES = (
    "apply_declared_lifecycle_schedule",
    "freeze_replayed_final_topology",
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
)


def _is_proof_controlled_lifecycle_policy(lifecycle_policy: str | None) -> bool:
    return lifecycle_policy in _PROOF_CONTROLLED_LIFECYCLE_POLICIES


def log_frag(device: str | torch.device, tag: str) -> None:
    """Read-only CUDA fragmentation probe (no sync, no empty_cache).

    frag_gap = reserved - allocated = memory the allocator holds but can't hand out.
    All three reads (memory_stats / memory_allocated / memory_reserved) are host-side
    allocator accounting; none issues a CUDA synchronize, so this is hot-path-safe.
    """
    if not torch.cuda.is_available():
        return
    dev = torch.device(device)                 # authorized device-normalization (CLAUDE.md #6)
    if dev.type != "cuda":
        return
    stats = torch.cuda.memory_stats(dev)
    allocated = torch.cuda.memory_allocated(dev) / 1024**2
    reserved = torch.cuda.memory_reserved(dev) / 1024**2
    _logger.info(
        "frag[%s] dev=%s alloc=%.0fMB reserved=%.0fMB active=%.0fMB frag_gap=%.0fMB nalloc_retries=%d",
        tag, dev, allocated, reserved,
        stats["active_bytes.all.current"] / 1024**2,
        reserved - allocated, stats["num_alloc_retries"],
    )


def _pair_slot_key(
    active_slot_list: list[str],
    pair_indices: tuple[int, int],
) -> tuple[str, str]:
    """Convert pair config indices into stable slot IDs at encode time."""
    idx_i, idx_j = pair_indices
    return active_slot_list[idx_i], active_slot_list[idx_j]


def _pair_index_accs_for_active_slots(
    pair_accs_by_slot: dict[tuple[str, str], float],
    active_slots: list[str],
) -> dict[tuple[int, int], float]:
    """Convert slot-keyed pair accuracies to emitter index keys."""
    slot_to_index = {slot_id: idx for idx, slot_id in enumerate(active_slots)}
    return {
        (slot_to_index[slot_a], slot_to_index[slot_b]): pair_acc
        for (slot_a, slot_b), pair_acc in pair_accs_by_slot.items()
    }


def _pair_interaction_index(
    pair_acc: float,
    solo_on_a: float,
    solo_on_b: float,
    all_off_acc: float,
) -> float:
    """Second-order interaction index I_ij = f({i,j}) - f({i}) - f({j}) + f(empty).

    All four terms MUST share single-coalition-ON semantics:
      - ``pair_acc``    = f({i,j})  (only slots i, j enabled),
      - ``solo_on_a/b`` = f({i}) / f({j})  (only that one slot enabled),
      - ``all_off_acc`` = f(empty)  (all seeds disabled).

    Passing leave-one-out accuracies (f(N\\{i}), "everyone but i") for the solo
    terms corrupts the index: leave-one-out and solo-ON agree ONLY in the purely
    additive (no-interaction) regime -- exactly the regime the index exists to
    detect deviation from -- so the mixed quantity has no clean game-theoretic
    meaning and its sign/magnitude do not track true synergy when seeds overlap.

    Returns 0.0 for additive (non-interacting) seeds, positive for synergy
    (the coalition beats the sum of its parts) and negative for antagonism.
    """
    return pair_acc - solo_on_a - solo_on_b + all_off_acc


def compute_forced_head_flags(
    masks_batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute per-head FORCED flags for decision telemetry.

    A head is FORCED only when a single valid candidate remains, i.e. the agent
    had no real choice (e.g. WAIT-only on the op head). A head that still has two
    or more valid candidates is NOT forced, even if some candidates were
    restricted by the mask. This is the contract for AnalyticsSnapshotPayload's
    per-head ``*_masked`` flags: "forced" means no agency, not "any candidate
    masked".
    """
    return {
        key: masks_batch[key].sum(dim=-1) <= 1  # [num_envs] bool tensor
        for key in HEAD_NAMES
    }


def _masked_op_probs_for_telemetry(
    *,
    op_logits: torch.Tensor,
    op_mask: torch.Tensor,
    probability_floor: dict[str, float] | None,
) -> torch.Tensor:
    """Return op probabilities using the same mask/floor contract as action sampling."""
    op_min_prob = None
    if probability_floor is not None and "op" in probability_floor:
        op_min_prob = probability_floor["op"]
    return MaskedCategorical(op_logits, op_mask, min_prob=op_min_prob).probs


def _reset_hidden_for_terminal_envs(
    batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    terminal_envs: list[int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Zero recurrent state for envs whose episode terminated mid-batch."""
    if batched_lstm_hidden is None or not terminal_envs:
        return batched_lstm_hidden

    h_batch, c_batch = batched_lstm_hidden
    h_reset = h_batch.clone()
    c_reset = c_batch.clone()
    for env_idx in terminal_envs:
        h_reset[:, env_idx, :].zero_()
        c_reset[:, env_idx, :].zero_()
    return h_reset, c_reset


def apply_proof_baseline_action_controls(
    *,
    masks_batch: dict[str, torch.Tensor],
    lifecycle_policy: str | None,
    schedule_id: str | None,
    schedule_hash: str | None = None,
    schedule_version: int | None = None,
    schedule_action_count: int | None = None,
    epoch: int,
    static_final_replay_validated: bool = False,
) -> None:
    """Apply proof-baseline lifecycle controls before action sampling."""
    if lifecycle_policy is None:
        if (
            schedule_id is not None
            or schedule_hash is not None
            or schedule_version is not None
            or schedule_action_count is not None
        ):
            raise ValueError(
                "proof_baseline schedule provenance requires a proof baseline lifecycle policy"
            )
        return
    if lifecycle_policy == "paired_lockstep_reward_comparison":
        if (
            schedule_id is not None
            or schedule_hash is not None
            or schedule_version is not None
            or schedule_action_count is not None
        ):
            raise ValueError(
                "paired_lockstep_reward_comparison must not carry schedule provenance"
            )
        return
    if lifecycle_policy == STATIC_FINAL_SOURCE_LIFECYCLE_POLICY:
        if (
            schedule_id != STATIC_FINAL_SOURCE_TOPOLOGY_V1
            or schedule_hash != STATIC_FINAL_SOURCE_TOPOLOGY_HASH
            or schedule_version != STATIC_FINAL_SOURCE_TOPOLOGY_VERSION
            or schedule_action_count != STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT
        ):
            raise ValueError(
                f"{STATIC_FINAL_SOURCE_LIFECYCLE_POLICY} requires "
                f"proof_baseline_schedule_id={STATIC_FINAL_SOURCE_TOPOLOGY_V1!r}, "
                "the matching hash, version, and action count"
            )
        action = static_final_source_action_for_epoch(epoch)
        _force_scheduled_action_masks(
            masks_batch=masks_batch,
            action=action,
            epoch=epoch,
        )
        return

    if lifecycle_policy == "freeze_replayed_final_topology":
        if (
            schedule_id is not None
            or schedule_hash is not None
            or schedule_version is not None
            or schedule_action_count is not None
        ):
            raise ValueError(
                "freeze_replayed_final_topology must not carry schedule provenance"
            )
        if not static_final_replay_validated:
            # Masking the op head to WAIT without prior replay would produce a
            # fake control frozen at INITIAL topology. Refuse loudly rather
            # than silently invalidating the proof.
            raise RuntimeError(
                "Proof baseline lifecycle policy "
                f"{lifecycle_policy!r} is not supported by this runner without "
                "validated static-final replay evidence: masking the op head to "
                "WAIT would produce a fake control that invalidates the "
                "morphogenesis proof. Materialize a source-final topology and "
                "emit topology manifest evidence before freezing the policy."
            )
        wait_idx = LifecycleOp.WAIT.value
        controlled_op_mask = torch.zeros_like(masks_batch["op"])
        controlled_op_mask[:, wait_idx] = True
        masks_batch["op"] = controlled_op_mask
        return

    unsupported_policies: tuple[str, ...] = ()
    if lifecycle_policy in unsupported_policies:
        raise RuntimeError(
            "Proof baseline lifecycle policy "
            f"{lifecycle_policy!r} is not supported by this runner: it has no "
            "real control implementation and masking the op head to WAIT would "
            "produce a fake control that invalidates the morphogenesis proof. "
            "Mark the cohort current_runner_supported=False or implement the "
            "real control."
        )

    if lifecycle_policy == "apply_declared_lifecycle_schedule":
        if (
            schedule_id != FIXED_SCHEDULE_GERMINATE_R0C0_V1
            or schedule_hash != FIXED_SCHEDULE_GERMINATE_R0C0_HASH
            or schedule_version != FIXED_SCHEDULE_GERMINATE_R0C0_VERSION
            or schedule_action_count != FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
        ):
            raise ValueError(
                "apply_declared_lifecycle_schedule requires "
                f"proof_baseline_schedule_id={FIXED_SCHEDULE_GERMINATE_R0C0_V1!r}, "
                "the matching hash, version, and action count"
            )
        action = fixed_schedule_action_for_epoch(epoch)
        _force_scheduled_action_masks(
            masks_batch=masks_batch,
            action=action,
            epoch=epoch,
        )
        return

    wait_only_policies = (
        "force_wait_only",
        "freeze_initial_topology",
    )
    if (
        schedule_id is not None
        or schedule_hash is not None
        or schedule_version is not None
        or schedule_action_count is not None
    ):
        raise ValueError(
            f"proof_baseline schedule provenance is not valid for lifecycle policy {lifecycle_policy!r}"
        )
    if lifecycle_policy not in wait_only_policies:
        raise ValueError(f"Unknown proof baseline lifecycle policy: {lifecycle_policy}")

    wait_idx = LifecycleOp.WAIT.value
    controlled_op_mask = torch.zeros_like(masks_batch["op"])
    controlled_op_mask[:, wait_idx] = True
    masks_batch["op"] = controlled_op_mask


def _force_scheduled_action_masks(
    *,
    masks_batch: dict[str, torch.Tensor],
    action: FactoredAction,
    epoch: int,
) -> None:
    """Force one declared fixed-schedule action without bypassing masks."""
    original_masks = {head: mask.clone() for head, mask in masks_batch.items()}
    try:
        _force_head_choice(
            masks_batch=masks_batch,
            head="op",
            index=action.op.value,
            epoch=epoch,
        )
        if action.op == LifecycleOp.WAIT:
            return

        _force_head_choice(
            masks_batch=masks_batch,
            head="slot",
            index=action.slot_idx,
            epoch=epoch,
        )
        _force_slot_by_op_choice(
            masks_batch=masks_batch,
            op_idx=action.op.value,
            slot_idx=action.slot_idx,
            epoch=epoch,
        )

        if action.op == LifecycleOp.GERMINATE:
            _force_head_choice(
                masks_batch=masks_batch,
                head="blueprint",
                index=action.blueprint.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="style",
                index=action.style.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="tempo",
                index=action.tempo.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_target",
                index=action.alpha_target.value,
                epoch=epoch,
            )
            return

        if action.op == LifecycleOp.SET_ALPHA_TARGET:
            _force_head_choice(
                masks_batch=masks_batch,
                head="style",
                index=action.style.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_target",
                index=action.alpha_target.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_speed",
                index=action.alpha_speed.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_curve",
                index=action.alpha_curve.value,
                epoch=epoch,
            )
            return

        if action.op == LifecycleOp.PRUNE:
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_speed",
                index=action.alpha_speed.value,
                epoch=epoch,
            )
            _force_head_choice(
                masks_batch=masks_batch,
                head="alpha_curve",
                index=action.alpha_curve.value,
                epoch=epoch,
            )
    except RuntimeError:
        masks_batch.clear()
        masks_batch.update(original_masks)
        raise


def _force_head_choice(
    *,
    masks_batch: dict[str, torch.Tensor],
    head: str,
    index: int,
    epoch: int,
) -> None:
    mask = masks_batch[head]
    if index < 0 or index >= mask.shape[1]:
        raise RuntimeError(
            f"Fixed-schedule action index {index} is out of range for head {head!r}"
        )
    if not bool(mask[:, index].all()):
        raise RuntimeError(
            "Fixed-schedule action is invalid under current action masks: "
            f"epoch={epoch} head={head!r} index={index}"
        )
    controlled_mask = torch.zeros_like(mask)
    controlled_mask[:, index] = True
    masks_batch[head] = controlled_mask


def _force_slot_by_op_choice(
    *,
    masks_batch: dict[str, torch.Tensor],
    op_idx: int,
    slot_idx: int,
    epoch: int,
) -> None:
    slot_by_op = masks_batch["slot_by_op"]
    if op_idx < 0 or op_idx >= slot_by_op.shape[1]:
        raise RuntimeError(
            f"Fixed-schedule op index {op_idx} is out of range for slot_by_op"
        )
    if slot_idx < 0 or slot_idx >= slot_by_op.shape[2]:
        raise RuntimeError(
            f"Fixed-schedule slot index {slot_idx} is out of range for slot_by_op"
        )
    if not bool(slot_by_op[:, op_idx, slot_idx].all()):
        raise RuntimeError(
            "Fixed-schedule action is invalid under current op-conditioned slot masks: "
            f"epoch={epoch} op_index={op_idx} slot_index={slot_idx}"
        )
    controlled_slot_by_op = torch.zeros_like(slot_by_op)
    controlled_slot_by_op[:, op_idx, slot_idx] = True
    masks_batch["slot_by_op"] = controlled_slot_by_op


def _check_vitals_before_snapshot(
    env_state: Any,
    *,
    epoch: int,
    val_loss: float,
) -> bool:
    """Check governor state before blessing the current weights as rollback-safe."""
    is_panic = env_state.governor.check_vital_signs(val_loss)
    if is_panic:
        return True

    if epoch % 5 == 0 or env_state.needs_governor_snapshot:
        env_state.governor.snapshot()
        env_state.needs_governor_snapshot = False

    return False


@dataclass(slots=True)
class FusedValResult:
    """Output of the fused validation + counterfactual pass.

    Owned by one epoch; consumed by _build_action_inputs and _run_action_transaction.
    """

    val_corrects: list[int]
    val_totals: list[int]
    baseline_accs: list[dict[str, Any]]
    solo_on_accs: list[dict[str, float]]
    all_disabled_accs: dict[int, float]
    pair_accs: dict[int, dict[tuple[str, str], float]]
    shapley_results: dict[int, dict[tuple[bool, ...], tuple[float, float]]]


@dataclass(slots=True)
class ActionInputBundle:
    """Batch-level inputs computed before execute_actions() for one epoch step.

    GPU tensors (masks_batch, head_log_probs, states_batch_normalized) are
    structure-of-arrays; they must NOT move to per-env records (I8: avoids
    N per-env GPU syncs).

    batched_lstm_hidden_post_action carries the policy's post-get_action hidden
    state back to the caller (the rollout get_action both consumes and replaces
    the batched LSTM hidden).
    """

    all_signals: list[Any]
    all_slot_reports: list[Any]
    states_batch_normalized: torch.Tensor
    blueprint_indices_batch: torch.Tensor
    pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]]
    head_log_probs: dict[str, torch.Tensor]
    masks_batch: dict[str, torch.Tensor]
    actions_np: np.ndarray
    values: list[float]
    head_confidences_cpu: np.ndarray | None
    head_entropies_cpu: np.ndarray | None
    op_probs_cpu: np.ndarray | None
    masked_np: np.ndarray | None
    step_obs_stats: Any | None
    governor_panic_envs: list[int]
    batched_lstm_hidden_post_action: tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class VectorizedPPOTrainer:
    agent: Any
    task_spec: Any
    slots: list[str]
    slot_config: Any
    n_envs: int
    max_epochs: int
    ppo_updates_per_batch: int
    total_batches: int
    total_env_episodes: int
    start_episode: int
    start_batch: int
    save_path: str | None
    seed: int
    env_device_map: list[str]
    shared_train_iter: Any
    shared_test_iter: Any
    num_train_batches: int
    num_test_batches: int
    env_reward_configs: list[Any]
    reward_family_enum: Any
    reward_config: Any
    loss_reward_config: Any
    reward_normalizer: Any
    obs_normalizer: Any
    initial_obs_normalizer_mean: torch.Tensor
    telemetry_config: Any
    telemetry_lifecycle_only: bool
    ops_telemetry_enabled: bool
    use_telemetry: bool
    gradient_telemetry_stride: int
    max_grad_norm: float | None
    plateau_threshold: float
    improvement_threshold: float
    anomaly_detector: AnomalyDetector
    hub: Any
    analytics: Any
    emitters: list[Any]
    batch_emitter: Any
    shutdown_event: Any
    group_id: str
    torch_profiler: bool
    torch_profiler_dir: str
    torch_profiler_wait: int
    torch_profiler_warmup: int
    torch_profiler_active: int
    torch_profiler_repeat: int
    torch_profiler_record_shapes: bool
    torch_profiler_profile_memory: bool
    torch_profiler_with_stack: bool
    torch_profiler_summary: bool
    gpu_preload_augment: bool
    amp_enabled: bool
    resolved_amp_dtype: torch.dtype | None
    proof_baseline_pair_id: str | None
    proof_baseline_lifecycle_policy: str | None
    proof_baseline_schedule_id: str | None
    proof_baseline_schedule_hash: str | None
    proof_baseline_schedule_version: int | None
    proof_baseline_schedule_action_count: int | None
    static_final_source_manifest: TopologyManifestPayload | None
    static_final_source_run_dir: str | None
    static_final_source_group_id: str | None
    static_final_source_episode_idx: int | None
    static_final_source_event_id: str | None
    telemetry_run_dir: str | None
    env_factory: EnvFactoryContext
    compiled_loss_and_correct: Callable[..., tuple[torch.Tensor, torch.Tensor, int]]
    run_ppo_updates: Callable[..., dict[str, Any]]
    aggregate_ppo_metrics: Callable[..., dict[str, Any]]
    handle_telemetry_escalation: Callable[..., None]
    emit_anomaly_diagnostics: Callable[..., None]
    fossilize_active_seed: Callable[[Any, str], bool]
    resolve_target_slot: ResolveTargetSlot
    host_params_baseline: int
    disable_advance: bool
    effective_max_seeds: int
    device: str
    logger: logging.Logger
    action_execution_context: ActionExecutionContext = field(init=False)
    ppo_coordinator: PPOCoordinator = field(init=False)

    def __post_init__(self) -> None:
        # Ada sm_89 TF32: ~2x FP32 matmul/conv on tensor cores at ~1e-3 rel error.
        # set_float32_matmul_precision("high") is the current API; we do NOT also set the
        # deprecated torch.backends.cuda.matmul.allow_tf32 (a redundant second path).
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        self.action_execution_context = ActionExecutionContext(
            slots=self.slots,
            ordered_slots=validate_slot_ids(list(self.slots)),
            slot_config=self.slot_config,
            task_spec=self.task_spec,
            env_reward_configs=self.env_reward_configs,
            reward_family_enum=self.reward_family_enum,
            reward_config=self.reward_config,
            loss_reward_config=self.loss_reward_config,
            reward_normalizer=self.reward_normalizer,
            telemetry_config=self.telemetry_config,
            ops_telemetry_enabled=self.ops_telemetry_enabled,
            disable_advance=self.disable_advance,
            effective_max_seeds=self.effective_max_seeds,
            max_epochs=self.max_epochs,
            num_train_batches=self.num_train_batches,
            device=self.device,
            analytics=self.analytics,
            emitters=self.emitters,
            agent=self.agent,
            fossilize_active_seed=self.fossilize_active_seed,
            resolve_target_slot=self.resolve_target_slot,
            host_params_baseline=self.host_params_baseline,
        )

        # Create PPO coordinator for update phase
        ppo_config = PPOCoordinatorConfig(
            ppo_updates_per_batch=self.ppo_updates_per_batch,
            max_epochs=self.max_epochs,
            total_env_episodes=self.total_env_episodes,
            amp_enabled=self.amp_enabled,
            resolved_amp_dtype=self.resolved_amp_dtype,
        )
        self.ppo_coordinator = PPOCoordinator(
            agent=self.agent,
            config=ppo_config,
            reward_normalizer=self.reward_normalizer,
            anomaly_detector=self.anomaly_detector,
            env_reward_configs=self.env_reward_configs,
            reward_family_enum=self.reward_family_enum,
            hub=self.hub,
            telemetry_config=self.telemetry_config,
            group_id=self.group_id,
            run_ppo_updates_fn=self.run_ppo_updates,
            handle_telemetry_escalation_fn=self.handle_telemetry_escalation,
            emit_anomaly_diagnostics_fn=self.emit_anomaly_diagnostics,
            logger=self.logger,
        )

    def _materialize_static_final_replay(
        self,
        *,
        env_states: list[Any],
        emitters: list[Any],
    ) -> None:
        if self.static_final_source_manifest is None:
            raise RuntimeError(
                "freeze_replayed_final_topology requires static_final_source_manifest"
            )
        if self.static_final_source_run_dir is None:
            raise RuntimeError(
                "freeze_replayed_final_topology requires static_final_source_run_dir"
            )
        if self.static_final_source_group_id is None:
            raise RuntimeError(
                "freeze_replayed_final_topology requires static_final_source_group_id"
            )
        if self.static_final_source_episode_idx is None:
            raise RuntimeError(
                "freeze_replayed_final_topology requires static_final_source_episode_idx"
            )
        if self.static_final_source_event_id is None:
            raise RuntimeError(
                "freeze_replayed_final_topology requires static_final_source_event_id"
            )

        for env_idx, env_state in enumerate(env_states):
            materialize_static_final_topology(
                model=env_state.model,
                source_manifest_json=(
                    self.static_final_source_manifest.topology_manifest_json
                ),
            )
            env_state.governor.snapshot()
            replay_episode_idx = emitters[env_idx].episode_idx
            if replay_episode_idx is None:
                raise RuntimeError(
                    "static-final replay capture requires emitter episode_idx"
                )
            replay_payload = capture_static_final_replay_manifest(
                model=env_state.model,
                task=self.task_spec.name,
                proof_baseline_pair_id=(
                    self.static_final_source_manifest.proof_baseline_pair_id
                ),
                source_manifest=self.static_final_source_manifest,
                source_run_dir=self.static_final_source_run_dir,
                source_group_id=self.static_final_source_group_id,
                source_episode_idx=self.static_final_source_episode_idx,
                source_event_id=self.static_final_source_event_id,
                replay_weight_policy=TOPOLOGY_ONLY_REPLAY_WEIGHT_POLICY,
                replay_env_id=env_idx,
                replay_episode_idx=replay_episode_idx,
            )
            emitters[env_idx].emit(
                TelemetryEvent(
                    event_type=TelemetryEventType.TOPOLOGY_MANIFEST_RECORDED,
                    epoch=0,
                    data=replay_payload,
                    message="Static-final replay manifest",
                    severity="info",
                )
            )

    def _emit_source_final_manifests(
        self,
        *,
        env_states: list[Any],
        emitters: list[Any],
        epoch: int,
    ) -> list[dict[str, Any]]:
        if self.proof_baseline_pair_id is None:
            raise RuntimeError(
                "static-final source capture requires proof_baseline_pair_id"
            )
        if self.telemetry_run_dir is None:
            raise RuntimeError(
                "static-final source capture requires telemetry_dir so source_run_dir "
                "can be linked by the proof packet"
            )

        records: list[dict[str, Any]] = []
        for env_idx, env_state in enumerate(env_states):
            payload = capture_source_final_manifest(
                model=env_state.model,
                task=self.task_spec.name,
                proof_baseline_pair_id=self.proof_baseline_pair_id,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.TOPOLOGY_MANIFEST_RECORDED,
                epoch=epoch,
                data=payload,
                message="Source-final topology manifest",
                severity="info",
            )
            emitters[env_idx].emit(event)
            episode_idx = emitters[env_idx].episode_idx
            if episode_idx is None:
                raise RuntimeError(
                    "static-final source capture requires emitter episode_idx"
                )
            records.append(
                {
                    "event_id": event.event_id,
                    "run_dir": self.telemetry_run_dir,
                    "group_id": emitters[env_idx].group_id,
                    "episode_idx": episode_idx,
                    "payload": asdict(payload),
                }
            )
        return records

    def _make_batch_context(
        self,
        envs_this_batch: int,
        reward_summary_accum: list[RewardSummaryAccumulator],
    ) -> list[EnvStepRecord]:
        """Pre-allocate per-env step records for one PPO batch.

        Called once per batch. The returned list is reused across all epochs
        within the batch — objects are mutated in place, not reallocated (I16).

        RESUME SEAM (HEALTH_REPORT open item): ContributionRewardInputs and
        LossRewardInputs objects are allocated here; they hold only CPU scalars,
        no GPU tensors.
        """
        return [
            EnvStepRecord(
                env_idx=i,
                action_spec=ActionSpec(),
                action_outcome=ActionOutcome(),
                mask_flags=ActionMaskFlags(),
                reward_summary=reward_summary_accum[i],
                contribution_reward_inputs=ContributionRewardInputs(
                    action=LifecycleOp.WAIT,
                    seed_contribution=None,
                    val_acc=0.0,
                    seed_info=None,
                    epoch=0,
                    max_epochs=self.max_epochs,
                    total_params=0,
                    host_params=1,
                    acc_at_germination=None,
                    acc_delta=0.0,
                    config=self.env_reward_configs[i],
                ),
                loss_reward_inputs=LossRewardInputs(
                    action=LifecycleOp.WAIT,
                    loss_delta=0.0,
                    val_loss=0.0,
                    seed_info=None,
                    epoch=0,
                    max_epochs=self.max_epochs,
                    total_params=0,
                    host_params=1,
                    config=self.loss_reward_config,
                ),
            )
            for i in range(envs_this_batch)
        ]

    def _build_fused_val_configs(
        self,
        *,
        env_states: list[ParallelEnvState],
        slots: list[str],
        epoch: int,
        envs_this_batch: int,
    ) -> tuple[
        list[list[dict[str, Any]]],  # env_configs (ablation schedule)
        list[dict[str, Any]],  # baseline_accs (empty dicts, filled by pass)
        list[dict[str, float]],  # solo_on_accs
        dict[int, float],  # all_disabled_accs
        dict[int, dict[tuple[str, str], float]],  # pair_accs
        dict[int, dict[tuple[bool, ...], tuple[float, float]]],  # shapley_results
        list[int],  # val_totals
        list[int],  # val_batch_counts
        list[torch.Tensor],  # env_cfg_correct_accums
    ]:
        """Build the per-env ablation config structures for the fused val pass.

        I12: val_totals drop_last=False semantics enforced by shared_test_iter
        configuration, not here. This method builds the ablation schedule only.

        The returned empty accumulators (baseline_accs, solo_on_accs,
        all_disabled_accs, pair_accs, shapley_results, val_totals,
        val_batch_counts, env_cfg_correct_accums) are filled by the iteration
        and result-dispatch phases that still run inline in run().
        """
        max_epochs = self.max_epochs
        # 1. Determine configurations per environment
        env_configs: list[list[dict[str, Any]]] = []
        for i, env_state in enumerate(env_states):
            model = env_state.model
            # CRITICAL: Exclude FOSSILIZED seeds from ablation in most modes.
            # Fossilized seeds are permanently integrated - disabling them
            # measures damage to the host, not the seed's contribution.
            # The host was trained WITH the fossilized seed's output;
            # suddenly removing it causes catastrophic accuracy drops.
            #
            # EXCEPTION: In BASIC_PLUS mode (drip_fraction > 0), we need
            # contribution data for fossilized seeds to compute drip payouts.
            active_slot_list = []
            for sid in slots:
                if not model.has_active_seed_in_slot(sid):
                    continue
                slot = cast(SeedSlotProtocol, model.seed_slots[sid])
                if slot.state is None or slot.alpha <= 0:
                    continue
                if slot.state.stage == SeedStage.FOSSILIZED:
                    # Only skip fossilized seeds when drip is disabled
                    # In BASIC_PLUS mode, we need contribution data for drip payouts
                    if self.reward_config.drip_fraction == 0.0:
                        continue
                active_slot_list.append(sid)

            # Config 0: Main (current alphas)
            configs = [{"_kind": "main"}]

            if active_slot_list:
                # Configs 1..N: Solo ablation (one slot off)
                for slot_id in active_slot_list:
                    solo_config: dict[str, Any] = {
                        "_kind": "solo",
                        "_slot": slot_id,
                        slot_id: 0.0,
                    }
                    configs.append(solo_config)

                # Solo-on configs: only this slot active (others forced off)
                for slot_id in active_slot_list:
                    solo_on_config: dict[str, Any] = {
                        sid: 0.0
                        for sid in active_slot_list
                        if sid != slot_id
                    }
                    solo_on_config["_kind"] = "solo_on"
                    solo_on_config["_slot"] = slot_id
                    configs.append(solo_on_config)

                n_active = len(active_slot_list)
                # Config N+1: All disabled (for 2-4 seeds)
                if 2 <= n_active <= 4:
                    all_off: dict[str, Any] = {
                        sid: 0.0 for sid in active_slot_list
                    }
                    all_off["_kind"] = "all_off"
                    configs.append(all_off)

                # Pair configs (for 3-4 seeds)
                if 3 <= n_active <= 4:
                    for idx_i in range(n_active):
                        for idx_j in range(idx_i + 1, n_active):
                            pair_config: dict[str, Any] = {
                                sid: 0.0
                                for k, sid in enumerate(active_slot_list)
                                if k != idx_i and k != idx_j
                            }
                            pair_config["_kind"] = "pair"
                            pair_config["_pair"] = _pair_slot_key(
                                active_slot_list,
                                (idx_i, idx_j),
                            )
                            configs.append(pair_config)

                # Committed accuracy: only fossilized seeds enabled.
                # This is the ground-truth metric for "permanent" value.
                committed_cfg: dict[str, Any] = {"_kind": "committed"}
                has_nonfossilized = False
                for slot_id in active_slot_list:
                    slot_obj = cast(
                        SeedSlotProtocol, model.seed_slots[slot_id]
                    )
                    seed_state = slot_obj.state
                    if seed_state is None:
                        continue
                    if seed_state.stage != SeedStage.FOSSILIZED:
                        committed_cfg[slot_id] = 0.0
                        has_nonfossilized = True
                if has_nonfossilized:
                    configs.append(committed_cfg)

            # Inject exact Shapley configurations at episode end
            # This uses the Fused Validation kernel to compute analytics in parallel
            # with the main validation pass, mitigating the O(N!) overhead.
            if (
                active_slot_list
                and env_state.counterfactual_helper
                and epoch == max_epochs
            ):
                required_configs = (
                    env_state.counterfactual_helper.get_required_configs(
                        active_slot_list
                    )
                )
                for config_tuple in required_configs:
                    # Convert tuple[bool] to alpha dict
                    shapley_cfg: dict[str, Any] = {
                        sid: 1.0 if enabled else 0.0
                        for sid, enabled in zip(
                            active_slot_list, config_tuple
                        )
                    }
                    shapley_cfg["_kind"] = "shapley"
                    shapley_cfg["_tuple"] = config_tuple
                    configs.append(shapley_cfg)

            env_configs.append(configs)

        # baseline_accs[env_idx][slot_id] = accuracy with that slot's seed disabled
        baseline_accs: list[dict[str, Any]] = [
            {} for _ in range(envs_this_batch)
        ]
        # solo_on_accs[env_idx][slot_id] = accuracy with ONLY that slot enabled
        solo_on_accs: list[dict[str, float]] = [
            {} for _ in range(envs_this_batch)
        ]
        all_disabled_accs: dict[int, float] = {}
        pair_accs: dict[int, dict[tuple[str, str], float]] = {}
        shapley_results: dict[
            int, dict[tuple[bool, ...], tuple[float, float]]
        ] = {}
        val_totals = [0] * envs_this_batch
        val_batch_counts = [0] * envs_this_batch

        # Accumulators for fused counts: env_cfg_correct_accums[env_idx] = [K] tensor
        env_cfg_correct_accums: list[torch.Tensor] = []
        for i, configs in enumerate(env_configs):
            env_cfg_correct_accums.append(
                torch.zeros(len(configs), device=env_states[i].env_device)
            )

        return (
            env_configs,
            baseline_accs,
            solo_on_accs,
            all_disabled_accs,
            pair_accs,
            shapley_results,
            val_totals,
            val_batch_counts,
            env_cfg_correct_accums,
        )
    def _run_fused_val_pass(
        self,
        *,
        env_states: list[ParallelEnvState],
        slots: list[str],
        epoch: int,
        envs_this_batch: int,
        val_criterion: nn.CrossEntropyLoss,
        batch_idx: int,
    ) -> tuple[FusedValResult, float]:
        """Run the fused validation + counterfactual ablation pass for one epoch.

        Single pass over the test data stacking every ablation config into one
        fused forward (config 0 == main). Returns the epoch-scoped validation
        outputs (FusedValResult) plus the wall time spent waiting on the test
        iterator (folded into the epoch's dataloader-wait throughput counter by
        the caller).

        I3: baseline_accs is rebuilt fresh each epoch here; the caller re-wires
        the returned reference into the execute_actions kwarg so
        build_fresh_contribution_targets reads the current-epoch freshness.

        I8: single env_state.stream.synchronize() after all fused launches; the
        GPU->CPU transfers (.cpu()/.item()/.tolist()) follow the sync.

        I12: val_loss_accum accumulates ONLY config idx 0 (loss_per_config[0]),
        the main config; ablation losses are intentionally worse and excluded.

        RESUME SEAM (HEALTH_REPORT open item): shared_test_iter is consumed here;
        iterator cursor state is not serialized for mid-run exact resume.
        """
        dataloader_wait_ms = 0.0
        # ===== Validation + Counterfactual (FUSED): Single pass over test data =====
        # Instead of iterating test data multiple times or performing sequential
        # forward passes, we stack all configurations into a single fused pass.

        # CRITICAL: Reset scaffolding metrics at START of counterfactual phase.
        # These metrics accumulate per-epoch interaction/topology data.
        # Feature extraction expects per-epoch values, not cross-epoch accumulation.
        for env_state in env_states:
            for slot_id in slots:
                if env_state.model.has_active_seed_in_slot(slot_id):
                    slot = cast(
                        SeedSlotProtocol,
                        env_state.model.seed_slots[slot_id],
                    )
                    seed_state = slot.state
                    if seed_state and seed_state.metrics:
                        seed_state.metrics.interaction_sum = 0.0
                        seed_state.metrics.boost_received = 0.0
                        seed_state.metrics.upstream_alpha_sum = 0.0
                        seed_state.metrics.downstream_alpha_sum = 0.0

        # 1. Determine configurations per environment + allocate
        # per-env ablation accumulators (extracted to a named phase).
        (
            env_configs,
            baseline_accs,
            solo_on_accs,
            all_disabled_accs,
            pair_accs,
            shapley_results,
            val_totals,
            val_batch_counts,
            env_cfg_correct_accums,
        ) = self._build_fused_val_configs(
            env_states=env_states,
            slots=slots,
            epoch=epoch,
            envs_this_batch=envs_this_batch,
        )

        # Iterate validation batches using shared iterator
        validation_start = time.perf_counter()
        test_iter = iter(self.shared_test_iter)
        for batch_step in range(self.num_test_batches):
            try:
                fetch_start = time.perf_counter()
                env_batches = next(test_iter)
                dataloader_wait_ms += (
                    time.perf_counter() - fetch_start
                ) * 1000.0
            except StopIteration:
                break

            for i, env_state in enumerate(env_states):
                if i >= len(env_batches):
                    continue
                if env_state.stream:
                    # CRITICAL: DataLoader .to(device, non_blocking=True) runs on the DEFAULT stream.
                    # We must sync env_state.stream with default stream before using the data,
                    # otherwise we may access partially-transferred data (race condition).
                    # BUG FIX: Use default_stream(), NOT current_stream() - the transfer happens
                    # on the default stream regardless of what stream is "current" in this context.
                    loader_stream = torch.cuda.default_stream(
                        torch.device(env_state.env_device)
                    )
                    env_state.stream.wait_stream(loader_stream)
                inputs, targets = env_batches[i]

                # SharedGPUBatchIterator now returns clones (not views) to fix race conditions.
                # We still need record_stream to prevent premature deallocation when the
                # tensor is used asynchronously by env_state.stream.
                if env_state.stream and inputs.is_cuda:
                    inputs.record_stream(env_state.stream)
                    targets.record_stream(env_state.stream)

                batch_size = inputs.size(0)
                configs = env_configs[i]
                num_configs = len(configs)

                # Build alpha_overrides tensors for the fused pass
                # Shape is topology-aware: [K*B, 1, 1, 1] for CNN, [K*B, 1, 1] for transformer
                #
                # IMPORTANT: Only pass alpha_override when at least one config
                # actually overrides that slot's alpha. Passing a no-op override
                # (e.g., alpha==0.0) forces SeedSlot.forward down the blending path
                # and bypasses the TRAINING-stage STE shortcut, changing semantics
                # and creating unnecessary alpha_schedule requirements.
                alpha_overrides: dict[str, torch.Tensor] = {}
                for slot_id in env_state.model._active_slots:
                    needs_override = any(slot_id in cfg for cfg in configs)
                    if not needs_override:
                        continue
                    slot = cast(
                        SeedSlotProtocol,
                        env_state.model.seed_slots[slot_id],
                    )
                    # Access concrete SeedSlot for alpha_schedule assignment
                    from esper.kasmina.slot import SeedSlot

                    assert isinstance(slot, SeedSlot), (
                        "Expected SeedSlot for alpha_schedule manipulation"
                    )
                    slot_concrete: SeedSlot = slot

                    # Enforce Phase 3 contract: alpha_schedule only valid for GATE.
                    if slot_concrete.alpha_schedule is not None and (
                        slot.state is None
                        or slot.state.alpha_algorithm != AlphaAlgorithm.GATE
                    ):
                        slot_concrete.alpha_schedule = None

                    # P4-FIX: Ensure alpha_schedule exists for GATE algorithm during fused pass.
                    # This can happen if a seed is in HOLD mode and its schedule was cleared.
                    if (
                        slot.state
                        and slot.state.alpha_algorithm
                        == AlphaAlgorithm.GATE
                        and slot_concrete.alpha_schedule is None
                    ):
                        from esper.kasmina.blending import BlendCatalog

                        topology = self.task_spec.topology
                        # Use default tempo steps since it's already in HOLD
                        slot_concrete.alpha_schedule = BlendCatalog.create(
                            "gated",
                            channels=slot_concrete.channels,
                            topology=topology,
                            total_steps=5,
                        ).to(slot_concrete.device)

                    current_alpha = slot.alpha
                    # Topology-aware shape for alpha_overrides
                    if self.task_spec.topology == "cnn":
                        alpha_shape = (num_configs * batch_size, 1, 1, 1)
                    else:  # transformer
                        alpha_shape = (num_configs * batch_size, 1, 1)
                    override_vec = torch.full(
                        alpha_shape,
                        current_alpha,
                        device=env_state.env_device,
                        dtype=inputs.dtype,
                    )

                    for cfg_idx, cfg in enumerate(configs):
                        if slot_id in cfg:
                            start, end = (
                                cfg_idx * batch_size,
                                (cfg_idx + 1) * batch_size,
                            )
                            alpha_value = cfg[slot_id]
                            assert isinstance(alpha_value, (int, float))
                            override_vec[start:end].fill_(alpha_value)
                    alpha_overrides[slot_id] = override_vec

                # Run FUSED validation pass with per-sample loss criterion
                loss_per_config, correct_per_config, total_per_config = (
                    process_fused_val_batch(
                        env_state,
                        inputs,
                        targets,
                        val_criterion,
                        alpha_overrides,
                        num_configs,
                        task_spec=self.task_spec,
                        loss_and_correct_fn=self.compiled_loss_and_correct,
                    )
                )

                stream_ctx = (
                    torch.cuda.stream(env_state.stream)
                    if env_state.stream
                    else nullcontext()
                )
                with stream_ctx:
                    env_cfg_correct_accums[i].add_(correct_per_config)
                    # Accumulate ONLY main config loss (idx 0) for Governor telemetry.
                    # Ablation losses would contaminate the signal since they're
                    # intentionally worse - measuring seed contribution not model health.
                    if env_state.val_loss_accum is not None:
                        env_state.val_loss_accum.add_(loss_per_config[0])
                val_totals[i] += total_per_config
                val_batch_counts[i] += 1

        # Single sync point at end
        for env_state in env_states:
            if env_state.stream:
                env_state.stream.synchronize()
        validation_elapsed_seconds = time.perf_counter() - validation_start

        # PERF: Batch GPU→CPU transfer before iterating
        # Moving tensors to CPU after sync is ~free (data already computed).
        # But .tolist() on GPU tensor would force per-tensor sync without this.
        env_cfg_correct_accums_cpu = [
            accum.cpu() for accum in env_cfg_correct_accums
        ]

        # Sync val_loss to env_state (for Sanctum TUI display)
        # NOTE: Loss is sum of batch means, so divide by batch count (not sample count).
        for i, env_state in enumerate(env_states):
            if (
                env_state.val_loss_accum is not None
                and val_batch_counts[i] > 0
            ):
                env_state.val_loss = (
                    env_state.val_loss_accum.item() / val_batch_counts[i]
                )
            else:
                env_state.val_loss = 0.0

        # Process results for each config
        val_corrects = [0] * envs_this_batch

        for i, env_state in enumerate(env_states):
            correct_counts = env_cfg_correct_accums_cpu[i].tolist()
            configs = env_configs[i]
            total = val_totals[i]

            if total == 0:
                continue

            for cfg_idx, cfg in enumerate(configs):
                acc = 100.0 * correct_counts[cfg_idx] / total
                kind = cfg["_kind"]

                if kind == "main":
                    val_corrects[i] = int(correct_counts[cfg_idx])
                    env_state.val_acc = acc
                    env_state.committed_val_acc = acc
                elif kind == "solo":
                    slot_id = cfg["_slot"]
                    baseline_accs[i][slot_id] = acc
                    # Sync to metrics
                    if env_state.model.has_active_seed_in_slot(slot_id):
                        slot_for_state = cast(
                            SeedSlotProtocol,
                            env_state.model.seed_slots[slot_id],
                        )
                        seed_state = slot_for_state.state
                        if seed_state and seed_state.metrics:
                            new_contribution = env_state.val_acc - acc
                            # Compute contribution velocity (EMA of delta)
                            prev = seed_state.metrics._prev_contribution
                            if prev is not None:
                                delta = new_contribution - prev
                                # EMA with decay 0.7 (responsive to recent changes)
                                seed_state.metrics.contribution_velocity = (
                                    0.7
                                    * seed_state.metrics.contribution_velocity
                                    + 0.3 * delta
                                )
                            seed_state.metrics._prev_contribution = (
                                new_contribution
                            )
                            seed_state.metrics.counterfactual_contribution = new_contribution
                            # Obs V3: Reset counterfactual staleness tracker on fresh measurement
                            env_state.epochs_since_counterfactual[
                                slot_id
                            ] = 0
                elif kind == "solo_on":
                    slot_id = cfg["_slot"]
                    solo_on_accs[i][slot_id] = acc
                elif kind == "all_off":
                    all_disabled_accs[i] = acc
                elif kind == "pair":
                    if i not in pair_accs:
                        pair_accs[i] = {}
                    pair_key = cfg["_pair"]
                    assert isinstance(pair_key, tuple)
                    pair_accs[i][pair_key] = acc
                elif kind == "shapley":
                    if i not in shapley_results:
                        shapley_results[i] = {}
                    # Validation loss approximated as 0.0 since we only track acc here
                    shapley_tuple = cfg["_tuple"]
                    assert isinstance(shapley_tuple, tuple)
                    shapley_results[i][shapley_tuple] = (0.0, acc)
                elif kind == "committed":
                    env_state.committed_val_acc = acc

            env_state.committed_acc_history.append(
                env_state.committed_val_acc
            )

            # Consolidate matrix reporting
            # CRITICAL: Sort active_slots for position-based topology computation.
            # Dict.keys() order is NOT guaranteed to match slot positions (r0c0, r0c1, r0c2...).
            # Lexicographic sort on slot IDs ensures correct upstream/downstream alpha sums.
            active_slots = sorted(baseline_accs[i].keys())
            if active_slots:
                pair_accs_for_emitter = (
                    _pair_index_accs_for_active_slots(
                        pair_accs[i],
                        active_slots,
                    )
                    if i in pair_accs
                    else {}
                )
                self.emitters[i].on_counterfactual_matrix(
                    active_slots=active_slots,
                    baseline_accs=baseline_accs[i],
                    val_acc=env_state.val_acc,
                    all_disabled_acc=all_disabled_accs.get(
                        i
                    ),  # None triggers emitter fallback
                    pair_accs=pair_accs_for_emitter,
                    solo_accs=solo_on_accs[i],
                )

            # Compute interaction terms and populate scaffolding metrics
            if len(active_slots) >= 2 and i in pair_accs:
                # Use solo ablation fallback for single-seed: min(baseline_accs) = host-only acc
                # Explicit None check: 0.0 is a valid baseline accuracy (model predicts nothing)
                all_off_acc = all_disabled_accs.get(i)
                if all_off_acc is None:
                    all_off_acc = min(baseline_accs[i].values())
                for (slot_a, slot_b), pair_acc in pair_accs[i].items():
                    # I_ij requires single-seed-ON accuracies f({i}):
                    # solo_on_accs holds "only this slot enabled" (f({i})),
                    # NOT baseline_accs' leave-one-out (f(N\{i})), which would
                    # corrupt the index when seeds interact. solo_on accs exist
                    # for every active slot (kind="solo_on"), so indexing is safe.
                    solo_on_a = solo_on_accs[i][slot_a]
                    solo_on_b = solo_on_accs[i][slot_b]
                    interaction = _pair_interaction_index(
                        pair_acc, solo_on_a, solo_on_b, all_off_acc
                    )

                    # Track positive synergy in scaffold boost ledger for hindsight credit
                    if interaction > 0:
                        # Seed A boosted Seed B (symmetric relationship)
                        env_state.scaffold_boost_ledger[slot_a].append(
                            (interaction, slot_b, epoch)
                        )
                        # Seed B boosted Seed A
                        env_state.scaffold_boost_ledger[slot_b].append(
                            (interaction, slot_a, epoch)
                        )

                    # Update metrics for both seeds
                    if env_state.model.has_active_seed_in_slot(slot_a):
                        slot_obj_a = cast(
                            SeedSlotProtocol,
                            env_state.model.seed_slots[slot_a],
                        )
                        seed_a = slot_obj_a.state
                        if seed_a and seed_a.metrics:
                            seed_a.metrics.interaction_sum += interaction
                            seed_a.metrics.boost_received = max(
                                seed_a.metrics.boost_received, interaction
                            )

                    if env_state.model.has_active_seed_in_slot(slot_b):
                        slot_obj_b = cast(
                            SeedSlotProtocol,
                            env_state.model.seed_slots[slot_b],
                        )
                        seed_b = slot_obj_b.state
                        if seed_b and seed_b.metrics:
                            seed_b.metrics.interaction_sum += interaction
                            seed_b.metrics.boost_received = max(
                                seed_b.metrics.boost_received, interaction
                            )

            # Compute topology features (upstream/downstream alpha sums)
            # active_slots is now sorted by position (lexicographic), ensuring correct topology
            for slot_idx, slot_id in enumerate(active_slots):
                if not env_state.model.has_active_seed_in_slot(slot_id):
                    continue
                slot_obj = cast(
                    SeedSlotProtocol,
                    env_state.model.seed_slots[slot_id],
                )
                seed_state = slot_obj.state
                if seed_state is None or seed_state.metrics is None:
                    continue

                upstream_sum = 0.0
                downstream_sum = 0.0
                for other_idx, other_id in enumerate(active_slots):
                    if other_id == slot_id:
                        continue
                    if not env_state.model.has_active_seed_in_slot(
                        other_id
                    ):
                        continue
                    other_slot_obj = cast(
                        SeedSlotProtocol,
                        env_state.model.seed_slots[other_id],
                    )
                    other_state = other_slot_obj.state
                    if other_state is None:
                        continue

                    other_alpha = (
                        other_state.metrics.current_alpha
                        if other_state.metrics
                        else 0.0
                    )
                    if other_idx < slot_idx:
                        upstream_sum += other_alpha
                    else:
                        downstream_sum += other_alpha

                seed_state.metrics.upstream_alpha_sum = upstream_sum
                seed_state.metrics.downstream_alpha_sum = downstream_sum

            # Feed Shapley results to helper
            if i in shapley_results and env_state.counterfactual_helper:
                try:
                    env_state.counterfactual_helper.compute_contributions_from_results(
                        slot_ids=active_slots,
                        results=shapley_results[i],
                        compute_time_seconds=validation_elapsed_seconds,
                        epoch=batch_idx + 1,
                    )
                except (KeyError, ZeroDivisionError, ValueError) as e:
                    # HIGH-01 fix: Narrow to expected failures in Shapley computation
                    self.logger.warning(
                        f"Shapley computation failed for env {i}: {e}"
                    )

        return (
            FusedValResult(
                val_corrects=val_corrects,
                val_totals=val_totals,
                baseline_accs=baseline_accs,
                solo_on_accs=solo_on_accs,
                all_disabled_accs=all_disabled_accs,
                pair_accs=pair_accs,
                shapley_results=shapley_results,
            ),
            dataloader_wait_ms,
        )

    def _run_train_pass(
        self,
        *,
        env_states: list[ParallelEnvState],
        step_records: list[EnvStepRecord],
        ordered_slots: list[str],
        epoch: int,
        criterion: nn.CrossEntropyLoss,
        last_train_corrects: list[int],
        last_train_totals: list[int],
        train_totals: list[int],
        train_batch_counts: list[int],
    ) -> tuple[list[dict[str, dict[Any, Any]] | None], float]:
        """Run one training epoch across all envs via CUDA streams.

        I8: wait_stream(default) before per-env loop; record_stream on augmented
        tensors inside stream context; synchronize() ONCE after all env launches.
        Returns (env_grad_stats, dataloader_wait_ms): the per-epoch gradient
        statistics (consumed later by the action-input phase) and the wall time
        spent waiting on the training data iterator (folded into the epoch's
        dataloader-wait throughput counter by the caller).
        Updates last_train_corrects, last_train_totals (slice assignment) and
        train_totals, train_batch_counts (in place) for the caller.

        RESUME SEAM (HEALTH_REPORT open item): shared_train_iter is consumed here;
        iterator cursor state is not serialized for mid-run exact resume.
        """
        dataloader_wait_ms = 0.0
        # Track gradient stats per env for telemetry sync
        env_grad_stats: list[dict[str, dict[Any, Any]] | None] = [None] * len(
            env_states
        )

        for env_state in env_states:
            env_state.zero_accumulators()

        # Ensure models are in training mode before training phase.
        # CRITICAL: process_val_batch/process_fused_val_batch call model.eval(), and without
        # this explicit model.train() call, all epochs after the first validation would run
        # with eval-mode semantics (frozen BatchNorm stats, disabled Dropout).
        for env_state in env_states:
            env_state.model.train()

        # ===== TRAINING: Iterate batches first, launch all envs via CUDA streams =====
        # SharedBatchIterator: single DataLoader, batches pre-split and moved to devices
        # SharedGPUBatchIterator: GPU-resident data, one DataLoader per device

        # Issue one wait_stream per env BEFORE the loop starts (not per-batch).
        # This syncs the accumulator zeroing on default stream before we write.
        # record_stream marks tensors as used by this stream, preventing deallocation.
        for i, env_state in enumerate(env_states):
            if env_state.stream:
                # Accumulators guaranteed non-None after init_accumulators()
                assert env_state.train_loss_accum is not None
                assert env_state.train_correct_accum is not None
                env_state.train_loss_accum.record_stream(env_state.stream)
                env_state.train_correct_accum.record_stream(env_state.stream)
                env_state.stream.wait_stream(
                    torch.cuda.default_stream(
                        torch.device(env_state.env_device)
                    )
                )

        # Iterate training batches using shared iterator (SharedBatchIterator or SharedGPUBatchIterator)
        # Both provide list of (inputs, targets) per environment, already on correct devices
        train_iter = iter(self.shared_train_iter)
        for batch_step in range(self.num_train_batches):
            try:
                fetch_start = time.perf_counter()
                env_batches = next(
                    train_iter
                )  # List of (inputs, targets), already on devices
                dataloader_wait_ms += (
                    time.perf_counter() - fetch_start
                ) * 1000.0
            except StopIteration:
                break

            # Launch all environments in their respective CUDA streams (async)
            # Data already moved to correct device by the shared iterator
            for i, env_state in enumerate(env_states):
                if i >= len(env_batches):
                    continue
                # CRITICAL: DataLoader .to(device, non_blocking=True) runs on the DEFAULT stream.
                # We must sync env_state.stream with default stream before using the data,
                # otherwise we may access partially-transferred data (race condition).
                # BUG FIX: Use default_stream(), NOT current_stream() - the transfer happens
                # on the default stream regardless of what stream is "current" in this context.
                if env_state.stream:
                    # Wait for default stream where async .to() transfers are scheduled
                    loader_stream = torch.cuda.default_stream(
                        torch.device(env_state.env_device)
                    )
                    env_state.stream.wait_stream(loader_stream)
                inputs, targets = env_batches[i]
                if self.gpu_preload_augment:
                    assert env_state.augment_generator is not None
                    if env_state.stream:
                        with torch.cuda.stream(env_state.stream):
                            inputs = augment_cifar10_batch(
                                inputs,
                                generator=env_state.augment_generator,
                                buffers=env_state.augment_buffers,
                            )
                            # CRITICAL: record_stream() MUST be inside the stream context.
                            # This marks the tensor as used by this stream, preventing the
                            # allocator from reusing memory while augmentation is in flight.
                            # The epoch-end sync (line ~500) ensures kernels complete before
                            # CPU reads results.
                            inputs.record_stream(env_state.stream)
                    else:
                        inputs = augment_cifar10_batch(
                            inputs,
                            generator=env_state.augment_generator,
                            buffers=env_state.augment_buffers,
                        )

                # BUG-031: Defensive validation for NLL loss assertion failures
                # If targets contain values outside [0, n_classes), the NLL loss kernel
                # will fail with "Assertion t>=0 && t < n_classes failed".
                # Enable with ESPER_DEBUG_TARGETS=1 to catch the issue with diagnostics.
                if "ESPER_DEBUG_TARGETS" in os.environ:
                    if targets.is_cuda:
                        torch.cuda.synchronize(targets.device)
                    target_min = targets.min().item()
                    target_max = targets.max().item()
                    if (
                        target_min < 0
                        or target_max >= self.task_spec.num_classes
                    ):
                        raise RuntimeError(
                            f"BUG-031: Invalid target values detected before loss computation. "
                            f"targets.min()={target_min}, targets.max()={target_max}, "
                            f"targets.device={targets.device}, env_idx={i}, batch_step={batch_step}, "
                            f"inputs.device={inputs.device}, inputs.shape={inputs.shape}, "
                            f"gpu_preload={self.gpu_preload_augment}"
                        )

                collect_gradients = self.use_telemetry and (
                    batch_step % self.gradient_telemetry_stride == 0
                )
                loss_tensor, correct_tensor, total, grad_stats = (
                    process_train_batch(
                        env_state,
                        inputs,
                        targets,
                        criterion,
                        use_telemetry=collect_gradients,
                        slots=self.slots,
                        max_grad_norm=self.max_grad_norm,
                        task_spec=self.task_spec,
                        resolved_amp_dtype=self.resolved_amp_dtype,
                        loss_and_correct_fn=self.compiled_loss_and_correct,
                    )
                )
                if grad_stats is not None:
                    env_grad_stats[i] = (
                        grad_stats  # Keep last batch's grad stats
                    )
                stream_ctx = (
                    torch.cuda.stream(env_state.stream)
                    if env_state.stream
                    else nullcontext()
                )
                with stream_ctx:
                    env_state.train_loss_accum.add_(loss_tensor)  # type: ignore[union-attr]
                    env_state.train_correct_accum.add_(correct_tensor)  # type: ignore[union-attr]
                train_totals[i] += total
                train_batch_counts[i] += 1

        # Sync all streams ONCE at epoch end
        for env_state in env_states:
            if env_state.stream:
                env_state.stream.synchronize()

        # NOW safe to call .item() - all GPU work done
        # Accumulators guaranteed non-None after init_accumulators()
        train_losses = [
            env_state.train_loss_accum.item()
            if env_state.train_loss_accum is not None
            else 0.0
            for env_state in env_states
        ]
        train_corrects = [
            env_state.train_correct_accum.item()
            if env_state.train_correct_accum is not None
            else 0.0
            for env_state in env_states
        ]
        last_train_corrects[:] = [int(value) for value in train_corrects]
        last_train_totals[:] = [total for total in train_totals]

        # Sync train metrics to env_state for telemetry (Sanctum TUI display)
        # NOTE: Loss is sum of batch means, so divide by batch count (not sample count).
        # Accuracy is sum of correct samples, so divide by sample count.
        for i, env_state in enumerate(env_states):
            env_state.train_loss = train_losses[i] / max(
                1, train_batch_counts[i]
            )
            env_state.train_acc = (
                100.0 * train_corrects[i] / max(1, train_totals[i])
            )

        return env_grad_stats, dataloader_wait_ms

    def _build_action_inputs(
        self,
        *,
        env_states: list[ParallelEnvState],
        env_grad_stats: list[dict[str, dict[Any, Any]] | None],
        raw_states_for_normalizer_update: list[torch.Tensor],
        ordered_slots: list[str],
        epoch: int,
        static_final_replay_validated: bool,
        rollout_autocast: Callable[[], AbstractContextManager[None]],
        batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> ActionInputBundle:
        """Collect signals, build masks, run get_action for one epoch step.

        I6 (BF16 symmetry): get_action runs under rollout_autocast() -- the SAME
        factory as bootstrap and the PPO update. The factory is passed explicitly
        (not captured by closure) so I6 is structurally verifiable.
        I7: raw_states_for_normalizer_update accumulated here; obs_normalizer
        stays frozen (normalize-only) during rollout collection.
        I10: apply_proof_baseline_action_controls runs on masks_batch BEFORE
        get_action.

        RESUME SEAM (HEALTH_REPORT open item): shared_train_iter/shared_test_iter
        iterator cursor is not serialized; the exact-resume fix belongs here.
        """
        agent = self.agent
        task_spec = self.task_spec
        slots = self.slots
        slot_config = self.slot_config
        max_epochs = self.max_epochs
        num_train_batches = self.num_train_batches
        obs_normalizer = self.obs_normalizer
        initial_obs_normalizer_mean = self.initial_obs_normalizer_mean
        ops_telemetry_enabled = self.ops_telemetry_enabled
        use_telemetry = self.use_telemetry
        emitters = self.emitters
        device = self.device
        effective_max_seeds = self.effective_max_seeds
        disable_advance = self.disable_advance
        last_obs_stats = None

        # ===== Compute epoch metrics and get BATCHED actions =====
        # NOTE: Telemetry sync (gradients/counterfactual) happens after record_accuracy()
        # so telemetry reflects the current epoch's metrics.

        # Collect signals, slot reports and action masks from all environments
        all_signals = []
        all_slot_reports = []
        all_masks = []

        governor_panic_envs = []  # Track which envs need rollback

        for env_idx, env_state in enumerate(env_states):
            model = env_state.model

            train_loss = env_state.train_loss
            train_acc = env_state.train_acc
            val_loss = env_state.val_loss
            val_acc = env_state.val_acc
            # Track maximum accuracy for sparse reward
            env_state.host_max_acc = max(
                env_state.host_max_acc, env_state.val_acc
            )

            # Governor watchdog: check vital signs before blessing a snapshot.
            is_panic = _check_vitals_before_snapshot(
                env_state,
                epoch=epoch,
                val_loss=val_loss,
            )
            if is_panic:
                governor_panic_envs.append(env_idx)

            # Gather active seeds across ALL enabled slots (multi-seed support)
            active_seeds = []
            for slot_id in slots:
                if model.has_active_seed_in_slot(slot_id):
                    slot_obj = cast(
                        SeedSlotProtocol, model.seed_slots[slot_id]
                    )
                    seed_state = slot_obj.state
                    if seed_state is not None:
                        active_seeds.append(seed_state)

            # Record accuracy for all active seeds (per-slot stage counters + deltas)
            for slot_id in slots:
                slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                slot_state = slot_obj.state
                if slot_state is None:
                    continue
                slot_state.metrics.record_accuracy(val_acc)

            # Sync gradient telemetry after record_accuracy so telemetry reflects this epoch's metrics.
            grad_stats_for_env = env_grad_stats[env_idx]
            synced_slot_ids: set[str] = set()
            if use_telemetry and grad_stats_for_env is not None:
                for slot_id, async_stats in grad_stats_for_env.items():
                    if not model.has_active_seed_in_slot(slot_id):
                        continue
                    slot_obj_for_grad = cast(
                        SeedSlotProtocol, model.seed_slots[slot_id]
                    )
                    seed_state = slot_obj_for_grad.state
                    if seed_state is None or seed_state.metrics is None:
                        continue

                    dual_stats = materialize_dual_grad_stats(async_stats)
                    current_ratio = dual_stats.normalized_ratio

                    prev_ema = env_state.gradient_ratio_ema.get(slot_id)
                    if prev_ema is None:
                        ema = current_ratio
                    else:
                        ema = 0.9 * prev_ema + 0.1 * current_ratio
                    env_state.gradient_ratio_ema[slot_id] = ema

                    # Sync ratio to SeedMetrics for G2 gate evaluation
                    seed_state.metrics.seed_gradient_norm_ratio = ema

                    # Materialize health stats for gradient telemetry
                    health_stats = materialize_grad_stats(
                        async_stats["_health_stats"]
                    )

                    # Sync telemetry using real gradient health from collect_seed_gradients_async
                    seed_state.sync_telemetry(
                        gradient_norm=health_stats["gradient_norm"],
                        gradient_health=health_stats["gradient_health"],
                        has_vanishing=health_stats["has_vanishing"],
                        has_exploding=health_stats["has_exploding"],
                        epoch=epoch,
                        max_epochs=max_epochs,
                    )
                    synced_slot_ids.add(slot_id)

            # Fallback: sync telemetry for active seeds that didn't get gradient stats
            # This ensures accuracy_delta is always populated from metrics.improvement_since_stage_start
            # Gradient parameters are omitted - sync_telemetry leaves gradient fields
            # UNMEASURED (None). KTS-001: an unmeasured seed must NOT look
            # healthy; permissive G2 denies it until real gradient stats arrive.
            for slot_id in slots:
                if slot_id in synced_slot_ids:
                    continue
                if not model.has_active_seed_in_slot(slot_id):
                    continue
                slot_obj_fallback = cast(
                    SeedSlotProtocol, model.seed_slots[slot_id]
                )
                seed_state_fallback = slot_obj_fallback.state
                if seed_state_fallback is None:
                    continue
                # Only sync accuracy/stage telemetry - no gradient data available
                seed_state_fallback.sync_telemetry(
                    epoch=epoch, max_epochs=max_epochs
                )

            slot_reports = model.get_slot_reports()

            # Consolidate environment-level telemetry emission
            emitters[env_idx].on_epoch_completed(
                epoch,
                env_state,
                slot_reports,
                observation_stats=last_obs_stats if env_idx == 0 else None,
            )

            # Update signal tracker
            # Phase 4: embargo/cooldown stages keep state while seed is removed.
            # Availability for germination is therefore "no state", not merely "no active seed".
            available_slots = sum(
                1
                for slot_id in slots
                if model.seed_slots[slot_id].state is None
            )
            signals = env_state.signal_tracker.update(
                epoch=epoch,
                global_step=epoch * num_train_batches,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                active_seeds=active_seeds,
                available_slots=available_slots,
            )

            all_signals.append(signals)
            all_slot_reports.append(slot_reports)
            # Cache total_seeds for this env (used in action masking)
            env_total_seeds = model.total_seeds() if model else 0

            # Compute action mask based on current state (physical constraints only)
            # Build slot states for ALL enabled slots (multi-slot masking)
            slot_states = build_slot_states(slot_reports, ordered_slots)
            mask = compute_action_masks(
                slot_states=slot_states,
                enabled_slots=ordered_slots,
                total_seeds=env_total_seeds,
                max_seeds=effective_max_seeds,
                slot_config=slot_config,
                device=torch.device(device),
                topology=task_spec.topology,
                disable_advance=disable_advance,
            )
            all_masks.append(mask)

        # OPTIMIZATION: Batched tensor-driven feature extraction (Obs V3)
        # Returns tuple: (obs [batch, obs_dim], blueprint_indices [batch, num_slots])
        states_batch, blueprint_indices_batch = batch_obs_to_features(
            batch_signals=all_signals,
            batch_slot_reports=all_slot_reports,
            batch_env_states=env_states,
            slot_config=slot_config,
            device=torch.device(device),
            max_epochs=max_epochs,
        )
        # NOTE: blueprint_indices_batch is passed to get_action() for op-conditioned value (Phase 4)

        # Stack dict masks into batched dict: {key: [n_envs, head_dim]}
        # Use static HEAD_NAMES for torch.compile compatibility
        masks_batch = {
            key: torch.stack([m[key] for m in all_masks]).to(device)
            for key in HEAD_NAMES
        }
        masks_batch["slot_by_op"] = torch.stack(
            [m["slot_by_op"] for m in all_masks]
        ).to(device)
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy=self.proof_baseline_lifecycle_policy,
            schedule_id=self.proof_baseline_schedule_id,
            schedule_hash=self.proof_baseline_schedule_hash,
            schedule_version=self.proof_baseline_schedule_version,
            schedule_action_count=(
                self.proof_baseline_schedule_action_count
            ),
            epoch=epoch,
            static_final_replay_validated=static_final_replay_validated,
        )

        # Accumulate raw states for deferred normalizer update
        raw_states_for_normalizer_update.append(states_batch.detach())

        # Normalize using FROZEN statistics during rollout collection.
        states_batch_normalized = obs_normalizer.normalize(states_batch)

        # TELE-OBS: Compute observation stats once per step (for Sanctum ObservationStats panel)
        # Only computed when ops telemetry is enabled to avoid overhead
        step_obs_stats = None
        if ops_telemetry_enabled:
            step_obs_stats = compute_observation_stats(
                states_batch,
                normalized_obs_tensor=states_batch_normalized,
                clip=10.0,
                normalizer_mean=obs_normalizer.mean,
                normalizer_var=obs_normalizer.var,
                initial_normalizer_mean=initial_obs_normalizer_mean,
            )
            last_obs_stats = step_obs_stats

        # Get BATCHED actions from policy network with action masking (single forward pass!)
        pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]] = []

        if batched_lstm_hidden is not None:
            h_batch, c_batch = batched_lstm_hidden
            for env_idx in range(len(env_states)):
                env_h = h_batch[:, env_idx : env_idx + 1, :].clone()
                env_c = c_batch[:, env_idx : env_idx + 1, :].clone()
                pre_step_hiddens.append((env_h, env_c))
        else:
            batched_lstm_hidden = agent.policy.initial_hidden(
                len(env_states)
            )
            if batched_lstm_hidden is not None:
                init_h, init_c = batched_lstm_hidden
                for env_idx in range(len(env_states)):
                    env_h = init_h[:, env_idx : env_idx + 1, :].clone()
                    env_c = init_c[:, env_idx : env_idx + 1, :].clone()
                    pre_step_hiddens.append((env_h, env_c))

        # get_action returns ActionResult dataclass.
        # P1-BF16: wrap in the SAME BF16 autocast as the PPO update so the
        # stored old_log_probs share the BF16 backbone (unbiased ratio).
        with rollout_autocast():
            action_result = agent.policy.get_action(
                states_batch_normalized,
                blueprint_indices=blueprint_indices_batch,
                masks=masks_batch,
                hidden=batched_lstm_hidden,
                deterministic=False,
                probability_floor=agent.probability_floor,
            )
        actions_dict = action_result.action
        head_log_probs = action_result.log_prob
        values_tensor = action_result.value

        # OPTIMIZATION: Update batched hidden state directly (eliminates per-env slice/cat)
        batched_lstm_hidden = action_result.hidden

        # Convert to list of dicts for per-env processing
        # PERF NOTE: Consolidate action head transfers into a single D2H copy.
        # This matters for larger env counts (16+), where per-head transfers and
        # per-env Python dict construction become a scaling bottleneck.
        actions_stacked = torch.stack(
            [actions_dict[name] for name in HEAD_NAMES]
        )
        actions_np = actions_stacked.cpu().numpy()  # [num_heads, num_envs]
        values = (
            values_tensor.cpu().tolist()
        )  # .tolist() on CPU tensor is free

        # Batch compute mask stats for telemetry
        masked_np: np.ndarray | None = None  # [num_heads, num_envs]
        if ops_telemetry_enabled:
            # FORCED, not "any-masked": a head is forced only when a
            # single valid candidate remains (no real choice). Heads
            # that still have >=2 valid candidates retain agency even
            # though some candidates were restricted.
            masked_batch = compute_forced_head_flags(masks_batch)
            masked_stacked = torch.stack(
                [masked_batch[name] for name in HEAD_NAMES]
            )
            masked_np = masked_stacked.cpu().numpy()
        else:
            masked_np = None

        # PERF: Pre-compute op_probs for telemetry ONCE before env loop.
        # Previous code called .cpu() per-env inside the loop, causing N GPU syncs
        # per step instead of 1. This was the root cause of 90% throughput drop.
        op_probs_cpu: np.ndarray | None = None
        if ops_telemetry_enabled and action_result.op_logits is not None:
            # Batch masked distribution over all envs, single GPU->CPU transfer.
            op_probs_all = _masked_op_probs_for_telemetry(
                op_logits=action_result.op_logits,
                op_mask=masks_batch["op"],
                probability_floor=agent.probability_floor,
            )
            op_probs_cpu = op_probs_all.cpu().numpy()

        # PERF: Pre-compute per-head confidences AND entropy for telemetry.
        # Uses batched GPU->CPU transfer: stack all heads, single transfer.
        #
        # Confidence = exp(log_prob) = P(chosen_action | valid_mask)
        # This properly handles masking via MaskedCategorical.
        #
        # Head names in order matching HeadTelemetry field positions.
        # We stack log_probs in this order, then index [0..7] to get each head's value.
        _HEAD_NAMES_FOR_TELEM = (
            "op",
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
        )
        head_confidences_cpu: np.ndarray | None = None  # [8, num_envs]
        head_entropies_cpu: np.ndarray | None = None

        if ops_telemetry_enabled and head_log_probs:
            # Stack all head log probs: [8, num_envs]
            stacked_log_probs = torch.stack(
                [head_log_probs[h] for h in _HEAD_NAMES_FOR_TELEM]
            )
            # Single exp + detach + transfer
            head_confidences_cpu = (
                torch.exp(stacked_log_probs).detach().cpu().numpy()
            )
            if action_result.head_entropies is None:
                raise RuntimeError(
                    "Policy action result is missing per-head rollout entropy"
                )
            stacked_head_entropies = torch.stack(
                [action_result.head_entropies[h] for h in _HEAD_NAMES_FOR_TELEM]
            )
            head_entropies_cpu = (
                stacked_head_entropies.detach().cpu().numpy()
            )


        return ActionInputBundle(
            all_signals=all_signals,
            all_slot_reports=all_slot_reports,
            states_batch_normalized=states_batch_normalized,
            blueprint_indices_batch=blueprint_indices_batch,
            pre_step_hiddens=pre_step_hiddens,
            head_log_probs=head_log_probs,
            masks_batch=masks_batch,
            actions_np=actions_np,
            values=values,
            head_confidences_cpu=head_confidences_cpu,
            head_entropies_cpu=head_entropies_cpu,
            op_probs_cpu=op_probs_cpu,
            masked_np=masked_np,
            step_obs_stats=step_obs_stats,
            governor_panic_envs=governor_panic_envs,
            batched_lstm_hidden_post_action=batched_lstm_hidden,
        )

    def run(self) -> list[dict[str, Any]]:
        agent = self.agent
        task_spec = self.task_spec
        slots = self.slots
        slot_config = self.slot_config
        n_envs = self.n_envs
        max_epochs = self.max_epochs
        total_batches = self.total_batches
        total_env_episodes = self.total_env_episodes
        start_episode = self.start_episode
        start_batch = self.start_batch
        save_path = self.save_path
        seed = self.seed
        shared_train_iter = self.shared_train_iter
        shared_test_iter = self.shared_test_iter
        num_train_batches = self.num_train_batches
        num_test_batches = self.num_test_batches
        reward_normalizer = self.reward_normalizer
        obs_normalizer = self.obs_normalizer
        initial_obs_normalizer_mean = self.initial_obs_normalizer_mean
        telemetry_config = self.telemetry_config
        telemetry_lifecycle_only = self.telemetry_lifecycle_only
        ops_telemetry_enabled = self.ops_telemetry_enabled
        use_telemetry = self.use_telemetry
        gradient_telemetry_stride = self.gradient_telemetry_stride
        max_grad_norm = self.max_grad_norm
        plateau_threshold = self.plateau_threshold
        improvement_threshold = self.improvement_threshold
        hub = self.hub
        analytics = self.analytics
        emitters = self.emitters
        batch_emitter = self.batch_emitter
        shutdown_event = self.shutdown_event
        group_id = self.group_id
        torch_profiler = self.torch_profiler
        torch_profiler_dir = self.torch_profiler_dir
        torch_profiler_wait = self.torch_profiler_wait
        torch_profiler_warmup = self.torch_profiler_warmup
        torch_profiler_active = self.torch_profiler_active
        torch_profiler_repeat = self.torch_profiler_repeat
        torch_profiler_record_shapes = self.torch_profiler_record_shapes
        torch_profiler_profile_memory = self.torch_profiler_profile_memory
        torch_profiler_with_stack = self.torch_profiler_with_stack
        torch_profiler_summary = self.torch_profiler_summary
        gpu_preload_augment = self.gpu_preload_augment
        resolved_amp_dtype = self.resolved_amp_dtype
        static_final_replay_validated = False

        def rollout_autocast() -> AbstractContextManager[None]:
            # P1-BF16 (CRITICAL-1): rollout get_action must run under the SAME policy-AMP
            # context as the PPO update (_run_ppo_updates), so old_log_probs and the update's
            # log_probs share one precision decision -> unbiased importance ratio. Both legs
            # call the single shared factory, so the symmetry is structural, not coincidental.
            return policy_amp_context(self.amp_enabled, resolved_amp_dtype)

        env_factory = self.env_factory
        compiled_loss_and_correct = self.compiled_loss_and_correct
        action_execution_context = self.action_execution_context
        effective_max_seeds = self.effective_max_seeds
        disable_advance = self.disable_advance
        device = self.device
        logger = self.logger

        profiler_cm = training_profiler(
            output_dir=torch_profiler_dir,
            enabled=torch_profiler,
            wait=torch_profiler_wait,
            warmup=torch_profiler_warmup,
            active=torch_profiler_active,
            repeat=torch_profiler_repeat,
            record_shapes=torch_profiler_record_shapes,
            profile_memory=torch_profiler_profile_memory,
            with_stack=torch_profiler_with_stack,
        )
        prof = profiler_cm.__enter__()
        prof_steps = 0

        try:
            history: list[dict[str, Any]] = []
            episode_history: list[
                EpisodeRecord
            ] = []  # Per-episode tracking for A/B testing
            episode_outcomes: list[Any] = []  # Pareto analysis outcomes
            best_avg_acc = 0.0
            recent_accuracies = []
            recent_rewards = []
            consecutive_finiteness_failures = (
                0  # Track PPO updates with all epochs skipped
            )
            prev_rolling_avg_acc: float | None = None

            episodes_completed = start_episode
            batch_idx = start_batch
            # Gradient EMA tracker for drift detection (P4-9)
            # Persists across batches to track slow degradation
            grad_ema_tracker = GradientEMATracker() if use_telemetry else None

            while batch_idx < total_batches:
                topology_manifest_records: list[dict[str, Any]] = []
                log_frag(self.device, f"batch{batch_idx}.start")
                # One PPO update per full batch of environments.
                envs_this_batch = n_envs
                # Monotonic epoch id for all per-batch snapshot events (commit barrier, PPO, analytics).
                # We use "episodes completed after this batch" so resumed runs stay monotonic.
                batch_epoch_id = episodes_completed + envs_this_batch

                # Create fresh environments for this batch
                # DataLoaders are shared via SharedBatchIterator (not per-env)
                env_states = [
                    create_env_state(
                        i,
                        make_env_seed(
                            root_seed=seed,
                            batch_idx=batch_idx,
                            env_idx=i,
                            envs_per_batch=envs_this_batch,
                        ),
                        env_factory,
                    )
                    for i in range(envs_this_batch)
                ]
                criterion = nn.CrossEntropyLoss()
                # Per-sample loss for fused validation - enables separating main config
                # from ablations for Governor telemetry (fixes ablation signal contamination)
                val_criterion = nn.CrossEntropyLoss(reduction="none")

                # Initialize episode for vectorized training
                for env_idx in range(envs_this_batch):
                    env_states[env_idx].reset_episode_state(slots)
                    agent.buffer.start_episode(env_id=env_idx)
                    # Set episode context for telemetry (used by seed lifecycle events via emit_with_env_context)
                    episode_ctx = env_states[env_idx].episode_context
                    if episode_ctx is not None:
                        episode_ctx.episode_idx = episodes_completed + env_idx
                    # Also set on VectorizedEmitter (used by on_epoch_completed, on_last_action)
                    emitters[env_idx].set_episode_idx(episodes_completed + env_idx)

                if self.proof_baseline_lifecycle_policy == "freeze_replayed_final_topology":
                    self._materialize_static_final_replay(
                        env_states=env_states,
                        emitters=emitters,
                    )
                    static_final_replay_validated = True

                # Initialize batched LSTM hidden state for all environments
                # (Batched hidden management avoids per-step cat/slice overhead)
                batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

                throughput_step_time_ms_sum = 0.0
                throughput_dataloader_wait_ms_sum = 0.0
                last_train_corrects = [0] * envs_this_batch
                last_train_totals = [0] * envs_this_batch
                reward_summary_accum = [
                    RewardSummaryAccumulator() for _ in range(envs_this_batch)
                ]

                # Accumulate raw (unnormalized) states for the pre-update normalizer refresh.
                # We freeze normalizer stats during rollout to keep normalization consistent,
                # then update stats before PPO updates in _run_ppo_updates.
                raw_states_for_normalizer_update: list[torch.Tensor] = []

                # Pre-allocate per-env step records (one per env, reused across epochs).
                # Each record owns its ActionSpec/ActionOutcome/ActionMaskFlags and
                # contribution/loss reward-input objects by composition; they are
                # mutated in place each step (I16). reward_summary_accum is held by
                # reference and also passed as a batch-level kwarg (D3). Per-env
                # rollback state lives on record.rollback_occurred (I4).
                step_records = self._make_batch_context(
                    envs_this_batch, reward_summary_accum
                )

                # Pre-compute ordered slots once per batch (not per-epoch)
                # validate_slot_ids parses/sorts slot IDs - expensive to repeat 25x per episode
                ordered_slots = validate_slot_ids(list(slots))

                # Run epochs with INVERTED CONTROL FLOW
                for epoch in range(1, max_epochs + 1):
                    epoch_start = time.perf_counter()
                    dataloader_wait_ms_epoch = 0.0
                    if telemetry_config is not None:
                        telemetry_config.tick_escalation()
                    for env_state in env_states:
                        configure_slot_telemetry(
                            env_state,
                            ops_telemetry_enabled=ops_telemetry_enabled,
                            telemetry_lifecycle_only=telemetry_lifecycle_only,
                            inner_epoch=epoch,
                            global_epoch=batch_epoch_id,
                        )
                    # Reset per-epoch training accumulators (fresh per epoch;
                    # mutated in place by _run_train_pass).
                    train_totals = [0] * envs_this_batch
                    train_batch_counts = [0] * envs_this_batch

                    # ===== TRAINING PASS (I8: CUDA stream fences) =====
                    env_grad_stats, train_dataloader_wait_ms = self._run_train_pass(
                        env_states=env_states,
                        step_records=step_records,
                        ordered_slots=ordered_slots,
                        epoch=epoch,
                        criterion=criterion,
                        last_train_corrects=last_train_corrects,
                        last_train_totals=last_train_totals,
                        train_totals=train_totals,
                        train_batch_counts=train_batch_counts,
                    )
                    dataloader_wait_ms_epoch += train_dataloader_wait_ms

                    # ===== Validation + Counterfactual (FUSED): Single pass over test data =====
                    # Extracted to _run_fused_val_pass(): one fused forward over
                    # the test data stacking every ablation config (config 0 == main).
                    (
                        fused_result,
                        val_dataloader_wait_ms,
                    ) = self._run_fused_val_pass(
                        env_states=env_states,
                        slots=slots,
                        epoch=epoch,
                        envs_this_batch=envs_this_batch,
                        val_criterion=val_criterion,
                        batch_idx=batch_idx,
                    )
                    dataloader_wait_ms_epoch += val_dataloader_wait_ms

                    # CRITICAL: baseline_accs is rebuilt fresh each epoch by
                    # _run_fused_val_pass. Re-wire the per-epoch fresh dict into the
                    # locals consumed below (and the execute_actions kwarg, per D4).
                    # I3: baseline_accs[env_idx] is read by build_fresh_contribution_targets
                    #     inside execute_actions; a stale reference from a prior epoch
                    #     would corrupt counterfactual freshness.
                    baseline_accs = fused_result.baseline_accs
                    all_disabled_accs = fused_result.all_disabled_accs
                    val_corrects = fused_result.val_corrects
                    val_totals = fused_result.val_totals

                    # ===== Compute epoch metrics and get BATCHED actions =====
                    # Extracted to _build_action_inputs(): collects per-env signals/slot
                    # reports, builds batched masks (I10 proof controls), accumulates raw
                    # states for the deferred normalizer refresh (I7), and runs the rollout
                    # get_action under the shared rollout_autocast factory (I6 BF16 symmetry).
                    _action_inputs = self._build_action_inputs(
                        env_states=env_states,
                        env_grad_stats=env_grad_stats,
                        raw_states_for_normalizer_update=raw_states_for_normalizer_update,
                        ordered_slots=ordered_slots,
                        epoch=epoch,
                        static_final_replay_validated=static_final_replay_validated,
                        rollout_autocast=rollout_autocast,
                        batched_lstm_hidden=batched_lstm_hidden,
                    )
                    all_signals = _action_inputs.all_signals
                    all_slot_reports = _action_inputs.all_slot_reports
                    states_batch_normalized = _action_inputs.states_batch_normalized
                    blueprint_indices_batch = _action_inputs.blueprint_indices_batch
                    pre_step_hiddens = _action_inputs.pre_step_hiddens
                    head_log_probs = _action_inputs.head_log_probs
                    masks_batch = _action_inputs.masks_batch
                    actions_np = _action_inputs.actions_np
                    values = _action_inputs.values
                    head_confidences_cpu = _action_inputs.head_confidences_cpu
                    head_entropies_cpu = _action_inputs.head_entropies_cpu
                    op_probs_cpu = _action_inputs.op_probs_cpu
                    masked_np = _action_inputs.masked_np
                    step_obs_stats = _action_inputs.step_obs_stats
                    governor_panic_envs = _action_inputs.governor_panic_envs
                    batched_lstm_hidden = _action_inputs.batched_lstm_hidden_post_action

                    action_result_bundle = execute_actions(
                        context=action_execution_context,
                        env_states=env_states,
                        step_records=step_records,
                        actions_np=actions_np,
                        values=values,
                        all_signals=all_signals,
                        all_slot_reports=all_slot_reports,
                        states_batch_normalized=states_batch_normalized,
                        blueprint_indices_batch=blueprint_indices_batch,
                        pre_step_hiddens=pre_step_hiddens,
                        head_log_probs=head_log_probs,
                        masks_batch=masks_batch,
                        head_confidences_cpu=head_confidences_cpu,
                        head_entropies_cpu=head_entropies_cpu,
                        op_probs_cpu=op_probs_cpu,
                        masked_np=masked_np,
                        baseline_accs=baseline_accs,
                        all_disabled_accs=all_disabled_accs,
                        governor_panic_envs=governor_panic_envs,
                        reward_summary_accum=reward_summary_accum,
                        episode_history=episode_history,
                        episode_outcomes=episode_outcomes,
                        step_obs_stats=step_obs_stats,
                        epoch=epoch,
                        episodes_completed=episodes_completed,
                        batch_idx=batch_idx,
                        proof_controlled_step=_is_proof_controlled_lifecycle_policy(
                            self.proof_baseline_lifecycle_policy
                        ),
                    )

                    truncated_bootstrap_targets = (
                        action_result_bundle.truncated_bootstrap_targets
                    )
                    all_post_action_signals = action_result_bundle.post_action_signals
                    all_post_action_slot_reports = (
                        action_result_bundle.post_action_slot_reports
                    )
                    all_post_action_masks = action_result_bundle.post_action_masks
                    rollback_terminal_envs = [
                        env_idx
                        for env_idx, rec in enumerate(step_records)
                        if rec.rollback_occurred
                    ]
                    batched_lstm_hidden = _reset_hidden_for_terminal_envs(
                        batched_lstm_hidden,
                        terminal_envs=rollback_terminal_envs,
                    )

                    # PHASE 2: Compute all bootstrap values in single batched forward pass
                    bootstrap_values: list[float] = []
                    if all_post_action_signals:
                        # Unpack Obs V3 tuple (obs, blueprint_indices)
                        post_action_features_batch, post_action_bp_indices = (
                            batch_signals_to_features(
                                batch_signals=all_post_action_signals,
                                batch_slot_reports=all_post_action_slot_reports,
                                slot_config=slot_config,
                                env_states=env_states,
                                device=torch.device(device),
                                max_epochs=max_epochs,
                            )
                        )
                        post_action_features_normalized = obs_normalizer.normalize(
                            post_action_features_batch
                        )
                        post_masks_batch = {
                            k: torch.stack([m[k] for m in all_post_action_masks]).to(
                                device
                            )
                            for k in HEAD_NAMES
                        }
                        post_masks_batch["slot_by_op"] = torch.stack(
                            [m["slot_by_op"] for m in all_post_action_masks]
                        ).to(device)

                        # P1-BF16: bootstrap value under the same BF16 autocast as the
                        # update, so Q(s,op) used in GAE is precision-consistent.
                        with rollout_autocast(), torch.inference_mode():
                            bootstrap_result = agent.policy.get_action(
                                post_action_features_normalized,
                                blueprint_indices=post_action_bp_indices,
                                masks=post_masks_batch,
                                hidden=batched_lstm_hidden,
                                deterministic=True,
                                probability_floor=agent.probability_floor,
                            )
                        # PERF: Move to CPU before .tolist() to avoid per-value GPU sync
                        bootstrap_values = bootstrap_result.value.cpu().tolist()

                    if truncated_bootstrap_targets:
                        if not bootstrap_values:
                            raise RuntimeError(
                                "Missing bootstrap values for truncated transitions."
                            )
                        for (env_id, step_idx), bootstrap_val in zip(
                            truncated_bootstrap_targets, bootstrap_values, strict=True
                        ):
                            agent.buffer.bootstrap_values[env_id, step_idx] = (
                                bootstrap_val
                            )

                    throughput_step_time_ms_sum += (
                        time.perf_counter() - epoch_start
                    ) * 1000.0
                    throughput_dataloader_wait_ms_sum += dataloader_wait_ms_epoch
                    # Check for graceful shutdown at end of each epoch (not just batch end)
                    # This gives user faster response (~seconds) instead of waiting for full batch
                    if shutdown_event is not None and shutdown_event.is_set():
                        print(
                            f"\n[Shutdown requested] Stopping at epoch {epoch}/{max_epochs} "
                            f"(batch {batch_idx + 1}, {episodes_completed}/{total_env_episodes} episodes)"
                        )
                        break  # Exit epoch loop; batch-level break below will handle cleanup
                    if prof is not None:
                        prof.step()
                        prof_steps += 1

                # PPO Update (delegated to PPOCoordinator)
                ppo_coordinator = self.ppo_coordinator

                # Handle rollbacks: inject death penalty and recompute metrics
                ppo_coordinator.handle_rollbacks(
                    env_states=env_states,
                    env_rollback_occurred=[r.rollback_occurred for r in step_records],
                    env_total_rewards=[r.env_total_reward for r in step_records],
                    episode_history=episode_history,
                    episode_outcomes=episode_outcomes,
                )

                # Execute PPO updates
                metrics, update_skipped, ppo_update_time_ms = ppo_coordinator.run_update(
                    raw_states_for_normalizer_update=raw_states_for_normalizer_update,
                    obs_normalizer=obs_normalizer,
                    envs_this_batch=envs_this_batch,
                    throughput_step_time_ms_sum=throughput_step_time_ms_sum,
                    throughput_dataloader_wait_ms_sum=throughput_dataloader_wait_ms_sum,
                )

                ppo_grad_norm = None
                if metrics:
                    # Clear after the normalizer update in _run_ppo_updates.
                    raw_states_for_normalizer_update = []

                    ppo_update_time_ms = metrics["ppo_update_time_ms"]

                    if update_skipped:
                        # Distinguish benign "no optimizer step" cases (for example
                        # epoch-0 KL early-stop) from true finiteness-gate failures.
                        consecutive_finiteness_failures, should_continue = (
                            ppo_coordinator.check_finiteness_gate(
                                metrics, consecutive_finiteness_failures
                            )
                        )
                        if not should_continue:
                            # Skip anomaly detection for this batch - metrics are NaN
                            continue
                    else:
                        ppo_grad_norm = metrics["ppo_grad_norm"]

                        # Check gradient drift
                        drift_metrics = ppo_coordinator.check_gradient_drift(
                            grad_ema_tracker, ppo_grad_norm
                        )

                        # Run anomaly detection (includes LSTM health, per-head entropy)
                        ppo_coordinator.run_anomaly_detection(
                            metrics=metrics,
                            drift_metrics=drift_metrics,
                            batched_lstm_hidden=batched_lstm_hidden,
                            batch_epoch_id=batch_epoch_id,
                            batch_idx=batch_idx,
                        )

                # If the epoch loop exited early (e.g. graceful shutdown), ensure the batch
                # summary reflects the partial episode outcomes instead of the default zeros.
                if epoch < max_epochs:
                    for env_idx, env_state in enumerate(env_states):
                        step_records[env_idx].env_final_acc = env_state.val_acc
                        step_records[env_idx].env_total_reward = sum(
                            env_state.episode_rewards
                        )

                # Track results and aggregate batch-level metrics
                avg_acc = sum(r.env_final_acc for r in step_records) / len(step_records)
                avg_reward = sum(r.env_total_reward for r in step_records) / len(
                    step_records
                )

                recent_accuracies.append(avg_acc)
                recent_rewards.append(avg_reward)
                if len(recent_accuracies) > 10:
                    recent_accuracies.pop(0)
                    recent_rewards.pop(0)

                rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

                if hub:
                    if not update_skipped:
                        # Assert non-None: values assigned in same `if not update_skipped` block above
                        assert (
                            ppo_grad_norm is not None and ppo_update_time_ms is not None
                        )
                        batch_emitter.on_ppo_update(
                            metrics=metrics,
                            episodes_completed=batch_epoch_id,
                            batch_idx=batch_idx,
                            epoch=epoch,
                            agent=agent,
                            ppo_grad_norm=ppo_grad_norm,
                            ppo_update_time_ms=ppo_update_time_ms,
                            avg_acc=avg_acc,
                            avg_reward=avg_reward,
                            rolling_avg_acc=rolling_avg_acc,
                        )

                    # Aggregate per-environment metrics for the BATCH_EPOCH_COMPLETED event
                    batch_train_losses = [es.train_loss for es in env_states]
                    batch_train_corrects = last_train_corrects
                    batch_train_totals = last_train_totals

                    batch_val_losses = [es.val_loss for es in env_states]
                    batch_val_corrects = val_corrects
                    batch_val_totals = val_totals

                    batch_emitter.on_batch_completed(
                        batch_idx=batch_idx,
                        episodes_completed=batch_epoch_id,
                        rolling_avg_acc=rolling_avg_acc,
                        avg_acc=avg_acc,
                        metrics=metrics,
                        env_states=env_states,
                        update_skipped=update_skipped,
                        plateau_threshold=plateau_threshold,
                        improvement_threshold=improvement_threshold,
                        prev_rolling_avg_acc=prev_rolling_avg_acc,
                        total_episodes=total_env_episodes,
                        start_episode=start_episode,
                        n_episodes=total_env_episodes,
                        env_final_accs=[r.env_final_acc for r in step_records],
                        avg_reward=avg_reward,
                        train_losses=batch_train_losses,
                        train_corrects=batch_train_corrects,
                        train_totals=batch_train_totals,
                        val_losses=batch_val_losses,
                        val_corrects=batch_val_corrects,
                        val_totals=batch_val_totals,
                        num_train_batches=num_train_batches,
                        num_test_batches=num_test_batches,
                        analytics=analytics,
                        epoch=epoch,
                    )
                    prev_rolling_avg_acc = rolling_avg_acc

                    # B7-DRL-02: Check for performance degradation (was previously unwired)
                    # Detects catastrophic forgetting, reward hacking, and training decay
                    training_progress = batch_epoch_id / total_env_episodes
                    check_performance_degradation(
                        hub,
                        current_acc=avg_acc,
                        rolling_avg_acc=rolling_avg_acc,
                        env_id=0,  # Aggregate metric across all envs
                        training_progress=training_progress,
                        group_id=group_id,
                    )

                    # P2-FRAGMETRIC: per-device CUDA caching-allocator snapshot (host-side,
                    # no sync) into Karn raw_events. Emitted while env_states is still alive
                    # (before the P2-DEL teardown); validates P0-ALLOC + the structural
                    # fragmentation cure (reserved-vs-allocated, retries, OOMs).
                    for frag_device in sorted({es.env_device for es in env_states}):
                        if torch.device(frag_device).type != "cuda":
                            continue
                        alloc_stats = torch.cuda.memory_stats(frag_device)
                        frag_allocated = alloc_stats["allocated_bytes.all.current"]
                        frag_reserved = alloc_stats["reserved_bytes.all.current"]
                        batch_emitter.on_allocator_stats(
                            batch_idx=batch_idx,
                            device=str(frag_device),
                            allocated_bytes=frag_allocated,
                            reserved_bytes=frag_reserved,
                            fragmentation_bytes=frag_reserved - frag_allocated,
                            num_alloc_retries=alloc_stats["num_alloc_retries"],
                            num_ooms=alloc_stats["num_ooms"],
                        )

                if (
                    self.proof_baseline_lifecycle_policy
                    == STATIC_FINAL_SOURCE_LIFECYCLE_POLICY
                ):
                    topology_manifest_records.extend(
                        self._emit_source_final_manifests(
                            env_states=env_states,
                            emitters=emitters,
                            epoch=epoch,
                        )
                    )

                # P2-DEL: the batch's telemetry consumers (on_batch_completed, the P2-FRAGMETRIC
                # emit) have read env_states. Sync each persistent stream so no async kernel still
                # references the model/optimizer tensors we are about to drop, then release the
                # refs so the caching allocator can reuse their segments next batch. At LOOP-BODY
                # indent (NOT inside `if hub:`) so the fence + release run even with telemetry off.
                # Never empty_cache() here (governor.py NOTE: it fights expandable_segments).
                for env_state in env_states:
                    if env_state.stream is not None:
                        env_state.stream.synchronize()
                # Drop the list AND the loop variable: a dangling `env_state` would pin the
                # last env's model/optimizer for another batch, defeating the segment release.
                del env_states, env_state
                del step_records  # Release ContributionRewardInputs/LossRewardInputs (Python objects, not GPU tensors)

                batch_summary = BatchSummary(
                    batch=batch_idx + 1,
                    episodes=batch_epoch_id,
                    avg_accuracy=avg_acc,
                    rolling_avg_accuracy=rolling_avg_acc,
                    metrics=metrics,
                    reward_summary=reward_summary_accum,
                    episode_history=episode_history,
                    topology_manifests=topology_manifest_records,
                )
                history.append(batch_summary.to_dict())

                if rolling_avg_acc > best_avg_acc:
                    best_avg_acc = rolling_avg_acc

                episodes_completed = batch_epoch_id
                log_frag(self.device, f"batch{batch_idx}.end")
                batch_idx += 1

                # Check for graceful shutdown request (e.g., user quit TUI)
                # Per-epoch check already printed progress; just break here
                if shutdown_event is not None and shutdown_event.is_set():
                    break

        finally:
            # Ensure profiler context is always closed, even on exceptions
            profiler_cm.__exit__(None, None, None)
            if torch_profiler_summary and prof is not None:
                print("\n=== torch.profiler: CUDA time (top 30) ===")
                print(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
                )
                print("\n=== torch.profiler: CPU time (top 30) ===")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
            if torch_profiler:
                min_steps_for_trace = (
                    torch_profiler_wait + torch_profiler_warmup + torch_profiler_active
                )
                if prof_steps < min_steps_for_trace:
                    print(
                        f"\n[torch.profiler] No trace captured (ran {prof_steps} steps; "
                        f"need >= {min_steps_for_trace} for wait={torch_profiler_wait} "
                        f"warmup={torch_profiler_warmup} active={torch_profiler_active}). "
                        "Run longer or reduce --torch-profiler-wait/--torch-profiler-warmup."
                    )

        if save_path:
            # B5-PT-02 FIX: Save normalizer state for correct training resume.
            # Observation stats are only valid for the exact Obs V3 contract they fit.
            checkpoint_metadata = {
                "checkpoint_kind": "last",
                **obs_normalizer_metadata(obs_normalizer, slot_config=slot_config),
                # Reward normalizer (RewardNormalizer)
                "reward_normalizer_mean": reward_normalizer.mean,
                "reward_normalizer_m2": reward_normalizer.m2,
                "reward_normalizer_count": reward_normalizer.count,
                # Resume counters
                "batches_completed": batch_idx,
                "n_envs": n_envs,
            }
            agent.save(save_path, metadata=checkpoint_metadata)

        return history
