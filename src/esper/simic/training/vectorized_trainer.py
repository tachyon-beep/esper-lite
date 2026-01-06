from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import numpy as np
import torch
import torch.nn as nn

from esper.leyline import (
    AlphaAlgorithm,
    HEAD_NAMES,
    LifecycleOp,
    SeedSlotProtocol,
    SeedStage,
)
from esper.leyline.slot_id import validate_slot_ids
from esper.simic.telemetry import (
    AnomalyDetector,
    GradientEMATracker,
    compute_lstm_health,
    compute_observation_stats,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    training_profiler,
)
from esper.simic.telemetry.emitters import check_performance_degradation
from esper.simic.rewards import ContributionRewardInputs, LossRewardInputs
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks
from esper.tamiyo.policy.features import batch_obs_to_features
from esper.utils.data import augment_cifar10_batch

from .action_execution import ActionExecutionContext, ResolveTargetSlot, execute_actions
from .batch_ops import batch_signals_to_features, process_train_batch
from .counterfactual_eval import process_fused_val_batch
from .env_factory import EnvFactoryContext, configure_slot_telemetry, create_env_state
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    BatchSummary,
    EpisodeRecord,
    RewardSummaryAccumulator,
)


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

    def __post_init__(self) -> None:
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

    def run(self) -> list[dict[str, Any]]:
        agent = self.agent
        task_spec = self.task_spec
        slots = self.slots
        slot_config = self.slot_config
        n_envs = self.n_envs
        max_epochs = self.max_epochs
        ppo_updates_per_batch = self.ppo_updates_per_batch
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
        env_reward_configs = self.env_reward_configs
        loss_reward_config = self.loss_reward_config
        reward_family_enum = self.reward_family_enum
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
        anomaly_detector = self.anomaly_detector
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
        amp_enabled = self.amp_enabled
        resolved_amp_dtype = self.resolved_amp_dtype
        env_factory = self.env_factory
        compiled_loss_and_correct = self.compiled_loss_and_correct
        run_ppo_updates = self.run_ppo_updates
        handle_telemetry_escalation = self.handle_telemetry_escalation
        emit_anomaly_diagnostics = self.emit_anomaly_diagnostics
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
            episode_history: list[EpisodeRecord] = []  # Per-episode tracking for A/B testing
            episode_outcomes: list[Any] = []  # Pareto analysis outcomes
            best_avg_acc = 0.0
            best_state = None
            recent_accuracies = []
            recent_rewards = []
            consecutive_finiteness_failures = 0  # Track PPO updates with all epochs skipped
            prev_rolling_avg_acc: float | None = None

            episodes_completed = start_episode
            batch_idx = start_batch
            # Gradient EMA tracker for drift detection (P4-9)
            # Persists across batches to track slow degradation
            grad_ema_tracker = GradientEMATracker() if use_telemetry else None

            while batch_idx < total_batches:
                # One PPO update per full batch of environments.
                envs_this_batch = n_envs
                # Monotonic epoch id for all per-batch snapshot events (commit barrier, PPO, analytics).
                # We use "episodes completed after this batch" so resumed runs stay monotonic.
                batch_epoch_id = episodes_completed + envs_this_batch

                # Create fresh environments for this batch
                # DataLoaders are shared via SharedBatchIterator (not per-env)
                base_seed = seed + batch_idx * 10000
                env_states = [
                    create_env_state(i, base_seed, env_factory)
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

                # Initialize batched LSTM hidden state for all environments
                # (Batched hidden management avoids per-step cat/slice overhead)
                batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

                # Per-env accumulators
                env_final_accs = [0.0] * envs_this_batch
                env_total_rewards = [0.0] * envs_this_batch

                throughput_step_time_ms_sum = 0.0
                throughput_dataloader_wait_ms_sum = 0.0
                last_train_corrects = [0] * envs_this_batch
                last_train_totals = [0] * envs_this_batch
                reward_summary_accum = [
                    RewardSummaryAccumulator() for _ in range(envs_this_batch)
                ]

                action_specs = [ActionSpec() for _ in range(envs_this_batch)]
                action_outcomes = [ActionOutcome() for _ in range(envs_this_batch)]
                action_mask_flags = [ActionMaskFlags() for _ in range(envs_this_batch)]
                contribution_reward_inputs = [
                    ContributionRewardInputs(
                        action=LifecycleOp.WAIT,
                        seed_contribution=None,
                        val_acc=0.0,
                        seed_info=None,
                        epoch=0,
                        max_epochs=max_epochs,
                        total_params=0,
                        host_params=1,
                        acc_at_germination=None,
                        acc_delta=0.0,
                        config=env_reward_configs[env_idx],
                    )
                    for env_idx in range(envs_this_batch)
                ]
                loss_reward_inputs = [
                    LossRewardInputs(
                        action=LifecycleOp.WAIT,
                        loss_delta=0.0,
                        val_loss=0.0,
                        seed_info=None,
                        epoch=0,
                        max_epochs=max_epochs,
                        total_params=0,
                        host_params=1,
                        config=loss_reward_config,
                    )
                    for _ in range(envs_this_batch)
                ]

                # Accumulate raw (unnormalized) states for the pre-update normalizer refresh.
                # We freeze normalizer stats during rollout to keep normalization consistent,
                # then update stats before PPO updates in _run_ppo_updates.
                raw_states_for_normalizer_update = []

                # Track per-environment rollback (more sample-efficient than batch-level).
                # Only envs that experienced rollback have stale transitions.
                env_rollback_occurred = [False] * envs_this_batch

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
                    # Track gradient stats per env for telemetry sync
                    env_grad_stats: list[dict[str, dict[Any, Any]] | None] = [
                        None
                    ] * envs_this_batch

                    # Reset per-epoch metrics by zeroing pre-allocated accumulators (faster than reallocating)
                    train_totals = [0] * envs_this_batch
                    train_batch_counts = [
                        0
                    ] * envs_this_batch  # Track batch count for correct loss averaging
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
                            env_state.train_loss_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                            env_state.train_correct_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                            env_state.stream.wait_stream(
                                torch.cuda.default_stream(
                                    torch.device(env_state.env_device)
                                )
                            )

                    # Iterate training batches using shared iterator (SharedBatchIterator or SharedGPUBatchIterator)
                    # Both provide list of (inputs, targets) per environment, already on correct devices
                    train_iter = iter(shared_train_iter)
                    for batch_step in range(num_train_batches):
                        try:
                            fetch_start = time.perf_counter()
                            env_batches = next(
                                train_iter
                            )  # List of (inputs, targets), already on devices
                            dataloader_wait_ms_epoch += (
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
                            if gpu_preload_augment:
                                assert env_state.augment_generator is not None
                                if env_state.stream:
                                    with torch.cuda.stream(env_state.stream):
                                        inputs = augment_cifar10_batch(
                                            inputs,
                                            generator=env_state.augment_generator,
                                        )
                                else:
                                    inputs = augment_cifar10_batch(
                                        inputs,
                                        generator=env_state.augment_generator,
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
                                if target_min < 0 or target_max >= task_spec.num_classes:
                                    raise RuntimeError(
                                        f"BUG-031: Invalid target values detected before loss computation. "
                                        f"targets.min()={target_min}, targets.max()={target_max}, "
                                        f"targets.device={targets.device}, env_idx={i}, batch_step={batch_step}, "
                                        f"inputs.device={inputs.device}, inputs.shape={inputs.shape}, "
                                        f"gpu_preload={gpu_preload_augment}"
                                    )

                            collect_gradients = use_telemetry and (
                                batch_step % gradient_telemetry_stride == 0
                            )
                            loss_tensor, correct_tensor, total, grad_stats = (
                                process_train_batch(
                                    env_state,
                                    inputs,
                                    targets,
                                    criterion,
                                    use_telemetry=collect_gradients,
                                    slots=slots,
                                    max_grad_norm=max_grad_norm,
                                    task_spec=task_spec,
                                    resolved_amp_dtype=resolved_amp_dtype,
                                    loss_and_correct_fn=compiled_loss_and_correct,
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
                    last_train_corrects = [int(value) for value in train_corrects]
                    last_train_totals = [total for total in train_totals]

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

                    # 1. Determine configurations per environment
                    env_configs: list[list[dict[str, Any]]] = []
                    for i, env_state in enumerate(env_states):
                        model = env_state.model
                        active_slot_list = [
                            sid
                            for sid in slots
                            if model.has_active_seed_in_slot(sid)
                            and cast(SeedSlotProtocol, model.seed_slots[sid]).state
                            and cast(SeedSlotProtocol, model.seed_slots[sid]).alpha
                            > 0
                        ]

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
                                        pair_config["_pair"] = (idx_i, idx_j)
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
                    pair_accs: dict[int, dict[tuple[int, int], float]] = {}
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

                    # Iterate validation batches using shared iterator
                    test_iter = iter(shared_test_iter)
                    for batch_step in range(num_test_batches):
                        try:
                            fetch_start = time.perf_counter()
                            env_batches = next(test_iter)
                            dataloader_wait_ms_epoch += (
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
                                    SeedSlotProtocol, env_state.model.seed_slots[slot_id]
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
                                    and slot.state.alpha_algorithm == AlphaAlgorithm.GATE
                                    and slot_concrete.alpha_schedule is None
                                ):
                                    from esper.kasmina.blending import BlendCatalog

                                    topology = task_spec.topology
                                    # Use default tempo steps since it's already in HOLD
                                    slot_concrete.alpha_schedule = BlendCatalog.create(
                                        "gated",
                                        channels=slot_concrete.channels,
                                        topology=topology,
                                        total_steps=5,
                                    ).to(slot_concrete.device)

                                current_alpha = slot.alpha
                                # Topology-aware shape for alpha_overrides
                                if task_spec.topology == "cnn":
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
                                    task_spec=task_spec,
                                    loss_and_correct_fn=compiled_loss_and_correct,
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

                    # PERF: Batch GPU→CPU transfer before iterating
                    # Moving tensors to CPU after sync is ~free (data already computed).
                    # But .tolist() on GPU tensor would force per-tensor sync without this.
                    env_cfg_correct_accums_cpu = [
                        accum.cpu() for accum in env_cfg_correct_accums
                    ]

                    # Sync val_loss to env_state (for Sanctum TUI display)
                    # NOTE: Loss is sum of batch means, so divide by batch count (not sample count).
                    for i, env_state in enumerate(env_states):
                        if env_state.val_loss_accum is not None and val_batch_counts[i] > 0:
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
                                        seed_state.metrics.counterfactual_contribution = (
                                            new_contribution
                                        )
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
                            emitters[i].on_counterfactual_matrix(
                                active_slots=active_slots,
                                baseline_accs=baseline_accs[i],
                                val_acc=env_state.val_acc,
                                all_disabled_acc=all_disabled_accs.get(
                                    i
                                ),  # None triggers emitter fallback
                                pair_accs=pair_accs.get(i, {}),
                                solo_accs=solo_on_accs[i],
                            )

                        # Compute interaction terms and populate scaffolding metrics
                        if len(active_slots) >= 2 and i in pair_accs:
                            # Use solo ablation fallback for single-seed: min(baseline_accs) = host-only acc
                            # Explicit None check: 0.0 is a valid baseline accuracy (model predicts nothing)
                            all_off_acc = all_disabled_accs.get(i)
                            if all_off_acc is None:
                                all_off_acc = min(baseline_accs[i].values())
                            for (idx_a, idx_b), pair_acc in pair_accs[i].items():
                                # Map indices to slot IDs
                                slot_a = active_slots[idx_a]
                                slot_b = active_slots[idx_b]
                                # Solo accuracies MUST exist - active_slots derived from baseline_accs keys
                                solo_a = baseline_accs[i][slot_a]
                                solo_b = baseline_accs[i][slot_b]
                                # I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)
                                interaction = (
                                    pair_acc - solo_a - solo_b + all_off_acc
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
                                    epoch=batch_idx + 1,
                                )
                            except (KeyError, ZeroDivisionError, ValueError) as e:
                                # HIGH-01 fix: Narrow to expected failures in Shapley computation
                                logger.warning(
                                    f"Shapley computation failed for env {i}: {e}"
                                )

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

                        # Governor watchdog: snapshot when loss is stable (every 5 epochs)
                        # Also snapshot immediately after fossilization to prevent incoherent rollback
                        # (see BUG FIX comment in OP_FOSSILIZE handling above)
                        if epoch % 5 == 0 or env_state.needs_governor_snapshot:
                            env_state.governor.snapshot()
                            env_state.needs_governor_snapshot = False

                        # Governor watchdog: check vital signs after validation
                        is_panic = env_state.governor.check_vital_signs(val_loss)
                        if is_panic:
                            governor_panic_envs.append(env_idx)

                        # Gather active seeds across ALL enabled slots (multi-seed support)
                        active_seeds = []
                        for slot_id in slots:
                            if model.has_active_seed_in_slot(slot_id):
                                slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
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
                        # Gradient parameters are omitted - sync_telemetry leaves gradient fields at defaults
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
                        emitters[env_idx].on_epoch_completed(epoch, env_state, slot_reports)

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

                    # Get BATCHED actions from policy network with action masking (single forward pass!)
                    pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]] = []

                    if batched_lstm_hidden is not None:
                        h_batch, c_batch = batched_lstm_hidden
                        for env_idx in range(len(env_states)):
                            env_h = h_batch[:, env_idx : env_idx + 1, :].clone()
                            env_c = c_batch[:, env_idx : env_idx + 1, :].clone()
                            pre_step_hiddens.append((env_h, env_c))
                    else:
                        batched_lstm_hidden = agent.policy.initial_hidden(len(env_states))
                        if batched_lstm_hidden is not None:
                            init_h, init_c = batched_lstm_hidden
                            for env_idx in range(len(env_states)):
                                env_h = init_h[:, env_idx : env_idx + 1, :].clone()
                                env_c = init_c[:, env_idx : env_idx + 1, :].clone()
                                pre_step_hiddens.append((env_h, env_c))

                    # get_action returns ActionResult dataclass
                    action_result = agent.policy.get_action(
                        states_batch_normalized,
                        blueprint_indices=blueprint_indices_batch,
                        masks=masks_batch,
                        hidden=batched_lstm_hidden,
                        deterministic=False,
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
                    values = values_tensor.cpu().tolist()  # .tolist() on CPU tensor is free

                    # Batch compute mask stats for telemetry
                    masked_np: np.ndarray | None = None  # [num_heads, num_envs]
                    if ops_telemetry_enabled:
                        masked_batch = {
                            key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool tensor
                            for key in HEAD_NAMES
                        }
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
                        # Batch softmax over all envs, single GPU->CPU transfer
                        op_probs_all = torch.softmax(action_result.op_logits, dim=-1)
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

                    # NOTE: Entropy is not available during action sampling (only during PPO evaluation).
                    # All entropy fields will be 0.0 until we add entropy computation to get_action().
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

                    action_result_bundle = execute_actions(
                        context=action_execution_context,
                        env_states=env_states,
                        actions_np=actions_np,
                        values=values,
                        all_signals=all_signals,
                        all_slot_reports=all_slot_reports,
                        states_batch_normalized=states_batch_normalized,
                        blueprint_indices_batch=blueprint_indices_batch,
                        pre_step_hiddens=pre_step_hiddens,
                        head_log_probs=head_log_probs,
                        masks_batch=masks_batch,
                        action_specs=action_specs,
                        action_outcomes=action_outcomes,
                        mask_flags=action_mask_flags,
                        contribution_reward_inputs=contribution_reward_inputs,
                        loss_reward_inputs=loss_reward_inputs,
                        head_confidences_cpu=head_confidences_cpu,
                        head_entropies_cpu=head_entropies_cpu,
                        op_probs_cpu=op_probs_cpu,
                        masked_np=masked_np,
                        baseline_accs=baseline_accs,
                        governor_panic_envs=governor_panic_envs,
                        env_rollback_occurred=env_rollback_occurred,
                        reward_summary_accum=reward_summary_accum,
                        env_final_accs=env_final_accs,
                        env_total_rewards=env_total_rewards,
                        episode_history=episode_history,
                        episode_outcomes=episode_outcomes,
                        step_obs_stats=step_obs_stats,
                        epoch=epoch,
                        episodes_completed=episodes_completed,
                        batch_idx=batch_idx,
                    )

                    truncated_bootstrap_targets = (
                        action_result_bundle.truncated_bootstrap_targets
                    )
                    all_post_action_signals = action_result_bundle.post_action_signals
                    all_post_action_slot_reports = (
                        action_result_bundle.post_action_slot_reports
                    )
                    all_post_action_masks = action_result_bundle.post_action_masks

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
                            k: torch.stack([m[k] for m in all_post_action_masks]).to(device)
                            for k in HEAD_NAMES
                        }
                        post_masks_batch["slot_by_op"] = torch.stack(
                            [m["slot_by_op"] for m in all_post_action_masks]
                        ).to(device)

                        with torch.inference_mode():
                            bootstrap_result = agent.policy.get_action(
                                post_action_features_normalized,
                                blueprint_indices=post_action_bp_indices,
                                masks=post_masks_batch,
                                hidden=batched_lstm_hidden,
                                deterministic=True,
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
                            agent.buffer.bootstrap_values[env_id, step_idx] = bootstrap_val

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

                # PPO Update
                metrics: dict[str, Any] = {}
                ppo_grad_norm, ppo_update_time_ms = None, None
                rollback_env_indices = [
                    i for i, occurred in enumerate(env_rollback_occurred) if occurred
                ]
                if rollback_env_indices:
                    for env_idx in rollback_env_indices:
                        # B1-DRL-01 fix: Inject death penalty so PPO learns to avoid
                        # catastrophic actions. Previously get_punishment_reward() was dead code.
                        # P1-NORM fix: Normalize penalty to match other rewards' scale.
                        # Use normalize_only to avoid polluting running stats with rare outliers.
                        penalty = env_states[env_idx].governor.get_punishment_reward()
                        normalized_penalty = reward_normalizer.normalize_only(penalty)
                        agent.buffer.mark_terminal_with_penalty(env_idx, normalized_penalty)
                        # B11-CR-03 fix: OVERWRITE last reward with RAW penalty (for telemetry interpretability).
                        # Buffer gets normalized_penalty (for PPO training stability).
                        # Telemetry gets raw penalty (for cross-run comparability).
                        if env_states[env_idx].episode_rewards:
                            env_states[env_idx].episode_rewards[-1] = penalty

                    # B11-CR-02 fix: Recompute metrics after penalty injection
                    # Metrics were computed in the epoch loop (lines 3173-3214) BEFORE penalty was applied.
                    # This caused EpisodeOutcome, episode_history, and stability to reflect PRE-PENALTY
                    # rewards, making rollback episodes appear ~2x more rewarding and ~1.6x more stable.
                    if rollback_env_indices:
                        for env_idx in rollback_env_indices:
                            env_state = env_states[env_idx]

                            # 1. Recompute total reward from post-penalty episode_rewards
                            env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                            # 2. Update episode_history entry for this env
                            for entry in reversed(episode_history):
                                if entry.env_id == env_idx:
                                    entry.episode_reward = env_total_rewards[env_idx]
                                    break

                            # 3. Recompute stability from post-penalty variance
                            recent_ep_rewards = (
                                env_state.episode_rewards[-20:]
                                if len(env_state.episode_rewards) >= 20
                                else env_state.episode_rewards
                            )
                            if len(recent_ep_rewards) > 1:
                                reward_var = float(np.var(recent_ep_rewards))
                                stability = 1.0 / (1.0 + reward_var)
                            else:
                                stability = 1.0

                            # 4. Find and replace EpisodeOutcome for this env
                            for idx, outcome in enumerate(episode_outcomes):
                                if outcome.env_id == env_idx:
                                    corrected_outcome = dataclasses.replace(
                                        outcome,
                                        episode_reward=env_total_rewards[env_idx],
                                        stability_score=stability,
                                    )
                                    episode_outcomes[idx] = corrected_outcome
                                    break

                if len(agent.buffer) == 0:
                    update_skipped = True
                else:
                    update_start = time.perf_counter()
                    metrics = run_ppo_updates(
                        agent=agent,
                        ppo_updates_per_batch=ppo_updates_per_batch,
                        raw_states_for_normalizer_update=raw_states_for_normalizer_update,
                        obs_normalizer=obs_normalizer,
                        use_amp=amp_enabled,
                        amp_dtype=resolved_amp_dtype,
                    )
                    ppo_update_time_ms = (time.perf_counter() - update_start) * 1000.0
                    update_skipped = False

                    if metrics:
                        metrics["ppo_update_time_ms"] = ppo_update_time_ms
                        metrics["ppo_grad_norm"] = metrics["pre_clip_grad_norm"]
                        metrics["rollout_length"] = max_epochs
                        metrics["rollout_episodes"] = envs_this_batch
                        metrics["rollout_total_steps"] = len(agent.buffer)
                        metrics["reward_mode"] = env_reward_configs[0].reward_mode.value
                        metrics["reward_family"] = reward_family_enum.value
                        metrics["entropy_coef"] = agent.entropy_coef

                        metrics["throughput_step_time_ms_sum"] = (
                            throughput_step_time_ms_sum
                        )
                        metrics["throughput_dataloader_wait_ms_sum"] = (
                            throughput_dataloader_wait_ms_sum
                        )

                        # Clear after the normalizer update in _run_ppo_updates.
                        raw_states_for_normalizer_update = []

                    # PPO update time tracking
                    ppo_update_time_ms = metrics["ppo_update_time_ms"]
                    ppo_grad_norm = metrics["ppo_grad_norm"]

                    # Gradient drift (P4-9)
                    drift_metrics = None
                    if grad_ema_tracker is not None and ppo_grad_norm is not None:
                        # Compute gradient health (0-1)
                        if ppo_grad_norm < 1e-7:
                            grad_health = 0.3  # Vanishing gradients
                        elif ppo_grad_norm > 100.0:
                            grad_health = 0.3  # Exploding gradients
                        else:
                            grad_health = 1.0  # Healthy range
                        drift_metrics = grad_ema_tracker.update(ppo_grad_norm, grad_health)

                    # FINITENESS GATE CONTRACT: Check if PPO update actually occurred
                    if not metrics["ppo_update_performed"]:
                        # All epochs skipped due to non-finite values
                        skip_count = metrics["finiteness_gate_skip_count"]
                        consecutive_finiteness_failures += 1
                        logger.warning(
                            f"PPO update skipped (all {skip_count} epochs hit finiteness gate). "
                            f"Consecutive failures: {consecutive_finiteness_failures}/3"
                        )

                        # Escalate after 3 consecutive failures (DRL best practice)
                        if consecutive_finiteness_failures >= 3:
                            raise RuntimeError(
                                f"PPO training failed: {consecutive_finiteness_failures} consecutive updates "
                                "skipped due to non-finite values. Check policy/value network outputs for NaN. "
                                f"Last failure: {metrics['finiteness_gate_failures']}"
                            )
                        # Skip anomaly detection for this batch - metrics are NaN
                        continue

                    # Reset counter on successful update
                    consecutive_finiteness_failures = 0

                    metric_values = [
                        v for v in metrics.values() if isinstance(v, (int, float))
                    ]
                    anomaly_report = anomaly_detector.check_all(
                        # MANDATORY metrics after PPO update - fail loudly if missing
                        ratio_max=metrics["ratio_max"],
                        ratio_min=metrics["ratio_min"],
                        explained_variance=metrics.get(
                            "explained_variance", 0.0
                        ),  # Optional: computed once
                        has_nan=any(math.isnan(v) for v in metric_values),
                        has_inf=any(math.isinf(v) for v in metric_values),
                        current_episode=batch_epoch_id,
                        total_episodes=total_env_episodes,
                    )

                    # B7-DRL-01: Check gradient drift and merge into anomaly report
                    if drift_metrics is not None:
                        drift_report = anomaly_detector.check_gradient_drift(
                            norm_drift=drift_metrics["norm_drift"],
                            health_drift=drift_metrics["health_drift"],
                        )
                        if drift_report.has_anomaly:
                            anomaly_report.has_anomaly = True
                            anomaly_report.anomaly_types.extend(drift_report.anomaly_types)
                            anomaly_report.details.update(drift_report.details)

                    # B7-DRL-04: Check LSTM hidden state health after PPO update
                    # LSTM hidden states can become corrupted during BPTT - monitor for
                    # explosion/saturation (RMS > threshold), vanishing (RMS < 1e-6), or NaN/Inf.
                    lstm_health = compute_lstm_health(batched_lstm_hidden)
                    if lstm_health is not None:
                        lstm_report = anomaly_detector.check_lstm_health(
                            h_rms=lstm_health.h_rms,
                            c_rms=lstm_health.c_rms,
                            h_env_rms_max=lstm_health.h_env_rms_max,
                            c_env_rms_max=lstm_health.c_env_rms_max,
                            has_nan=lstm_health.has_nan,
                            has_inf=lstm_health.has_inf,
                        )
                        if lstm_report.has_anomaly:
                            anomaly_report.has_anomaly = True
                            anomaly_report.anomaly_types.extend(lstm_report.anomaly_types)
                            anomaly_report.details.update(lstm_report.details)
                        # Add LSTM health to metrics for telemetry display in Sanctum
                        metrics.update(lstm_health.to_dict())

                    handle_telemetry_escalation(anomaly_report, telemetry_config)
                    emit_anomaly_diagnostics(
                        hub,
                        anomaly_report,
                        agent,
                        batch_epoch_id,
                        batch_idx,
                        max_epochs,
                        total_env_episodes,
                        False,
                        group_id=group_id,
                    )

                # If the epoch loop exited early (e.g. graceful shutdown), ensure the batch
                # summary reflects the partial episode outcomes instead of the default zeros.
                if epoch < max_epochs:
                    for env_idx, env_state in enumerate(env_states):
                        env_final_accs[env_idx] = env_state.val_acc
                        env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                # Track results and aggregate batch-level metrics
                avg_acc = sum(env_final_accs) / len(env_final_accs)
                avg_reward = sum(env_total_rewards) / len(env_total_rewards)

                recent_accuracies.append(avg_acc)
                recent_rewards.append(avg_reward)
                if len(recent_accuracies) > 10:
                    recent_accuracies.pop(0)
                    recent_rewards.pop(0)

                rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

                if hub:
                    if not update_skipped:
                        # Assert non-None: values assigned in same `if not update_skipped` block above
                        assert ppo_grad_norm is not None and ppo_update_time_ms is not None
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
                        env_final_accs=env_final_accs,
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
                    )

                batch_summary = BatchSummary(
                    batch=batch_idx + 1,
                    episodes=batch_epoch_id,
                    avg_accuracy=avg_acc,
                    rolling_avg_accuracy=rolling_avg_acc,
                    metrics=metrics,
                    reward_summary=reward_summary_accum,
                    episode_history=episode_history,
                )
                history.append(batch_summary.to_dict())

                if rolling_avg_acc > best_avg_acc:
                    best_avg_acc = rolling_avg_acc
                    best_state = {
                        k: v.cpu().clone() for k, v in agent.policy.state_dict().items()
                    }

                episodes_completed = batch_epoch_id
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
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
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

        if best_state:
            agent.policy.load_state_dict(best_state)

        if save_path:
            # B5-PT-02 FIX: Save normalizer state for correct training resume.
            # Resume expects these keys in metadata.
            checkpoint_metadata = {
                # Observation normalizer (RunningMeanStd)
                "obs_normalizer_mean": obs_normalizer.mean.tolist(),
                "obs_normalizer_var": obs_normalizer.var.tolist(),
                "obs_normalizer_count": obs_normalizer.count.item(),
                "obs_normalizer_momentum": obs_normalizer.momentum,
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
