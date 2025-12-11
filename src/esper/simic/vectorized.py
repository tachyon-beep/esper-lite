"""Vectorized multi-GPU PPO training for Tamiyo.

This module implements high-performance vectorized PPO training using:
- Multiple parallel environments
- CUDA streams for async GPU execution
- Inverted control flow (batch-first iteration)
- SharedBatchIterator: Single DataLoader serving all environments

Key Architecture:
Instead of N independent DataLoaders (N × workers = massive IPC overhead),
we use ONE SharedBatchIterator with combined batch size, then split batches
across environments. This reduces worker processes from N×M to just M.

Performance comparison (4 envs, 4 workers each):
- Old (independent): 16 worker processes, 16× IPC overhead
- New (shared): 4 worker processes, 1× IPC overhead

Usage:
    from esper.simic.vectorized import train_ppo_vectorized

    agent, history = train_ppo_vectorized(
        n_episodes=100,
        n_envs=4,
        devices=["cuda:0", "cuda:1"],
    )
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from esper.runtime import get_task_spec
from esper.utils.data import SharedBatchIterator
from esper.leyline import SeedStage, SeedTelemetry, TelemetryEvent, TelemetryEventType
from esper.leyline.actions import get_blueprint_from_action, is_germinate_action
from esper.simic.features import compute_action_mask
from esper.simic.gradient_collector import (
    collect_dual_gradients_async,
    materialize_dual_grad_stats,
)
from esper.simic.normalization import RunningMeanStd, RewardNormalizer
from esper.simic.ppo import PPOAgent, signals_to_features
from esper.simic.rewards import compute_contribution_reward, SeedInfo
from esper.kasmina.slot import MIN_FOSSILIZE_CONTRIBUTION
from esper.nissa import get_hub, BlueprintAnalytics
from esper.tolaria import TolariaGovernor


# =============================================================================
# Parallel Environment State
# =============================================================================

@dataclass
class ParallelEnvState:
    """State for a single parallel environment with CUDA stream for async execution.

    DataLoaders are now SHARED via SharedBatchIterator - batches are pre-split
    and data is pre-moved to each env's device with non_blocking=True.
    """
    model: nn.Module
    host_optimizer: torch.optim.Optimizer
    seed_optimizer: torch.optim.Optimizer | None
    signal_tracker: any  # SignalTracker from tamiyo
    governor: TolariaGovernor  # Fail-safe watchdog for catastrophic failure detection
    env_device: str = "cuda:0"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
    seeds_created: int = 0
    seeds_fossilized: int = 0  # Total seeds fossilized this episode
    contributing_fossilized: int = 0  # Seeds with total_improvement >= MIN_FOSSILIZE_CONTRIBUTION
    episode_rewards: list = field(default_factory=list)
    action_counts: dict = field(default_factory=dict)
    successful_action_counts: dict = field(default_factory=dict)
    action_enum: type | None = None
    # Metrics for current batch step
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    params_added_baseline: int = 0
    # Ransomware-resistant reward: track accuracy at germination for progress calculation
    acc_at_germination: float | None = None
    # Pre-allocated accumulators to avoid per-epoch tensor allocation churn
    train_loss_accum: torch.Tensor | None = None
    train_correct_accum: torch.Tensor | None = None
    val_loss_accum: torch.Tensor | None = None
    val_correct_accum: torch.Tensor | None = None
    cf_correct_accum: torch.Tensor | None = None  # Counterfactual accumulator
    # LSTM hidden state for recurrent policy
    # Shape: (h, c) where each is [num_layers, 1, hidden_dim] for this single env
    # (Batched to [num_layers, num_envs, hidden_dim] during forward pass)
    # None = fresh episode (initialized on first action selection)
    lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    # EMA tracking for seed gradient ratio (for G2 gate)
    # Smooths per-step ratio noise with momentum=0.9
    gradient_ratio_ema: float = 0.0
    gradient_ratio_ema_initialized: bool = False

    def __post_init__(self) -> None:
        if not self.action_counts and self.action_enum is not None:
            base_counts = {a.name: 0 for a in self.action_enum}
            self.action_counts = base_counts.copy()
            self.successful_action_counts = base_counts.copy()

    def init_accumulators(self) -> None:
        """Initialize pre-allocated accumulators on the environment's device."""
        self.train_loss_accum = torch.zeros(1, device=self.env_device)
        self.train_correct_accum = torch.zeros(1, device=self.env_device)
        self.val_loss_accum = torch.zeros(1, device=self.env_device)
        self.val_correct_accum = torch.zeros(1, device=self.env_device)
        self.cf_correct_accum = torch.zeros(1, device=self.env_device)

    def zero_accumulators(self) -> None:
        """Zero accumulators at the start of each epoch (faster than reallocating)."""
        self.train_loss_accum.zero_()
        self.train_correct_accum.zero_()
        self.val_loss_accum.zero_()
        self.val_correct_accum.zero_()
        self.cf_correct_accum.zero_()


def _advance_active_seed(model) -> bool:
    """Advance lifecycle for the active seed, emitting telemetry via SeedSlot.

    Returns:
        True if the seed successfully fossilized, False otherwise.
    """
    if not model.has_active_seed:
        return False

    seed_state = model.seed_state
    current_stage = seed_state.stage

    # Tamiyo only finalizes; mechanical blending/advancement handled by Kasmina.
    # NOTE: Leyline VALID_TRANSITIONS only allow PROBATIONARY → FOSSILIZED.
    if current_stage == SeedStage.PROBATIONARY:
        gate_result = model.seed_slot.advance_stage(SeedStage.FOSSILIZED)
        if gate_result.passed:
            model.seed_slot.set_alpha(1.0)
            return True
        # Gate check failure is normal; reward shaping will penalize
        return False
    return False


# =============================================================================
# Vectorized PPO Training
# =============================================================================

def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = 4,
    max_epochs: int = 25,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    task: str = "cifar10",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.05,  # Unified default
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.01,  # Unified minimum
    adaptive_entropy_floor: bool = False,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    ppo_updates_per_batch: int = 1,
    save_path: str = None,
    resume_path: str = None,
    seed: int = 42,
    num_workers: int | None = None,
    gpu_preload: bool = False,
    recurrent: bool = False,
    lstm_hidden_dim: int = 128,
    chunk_length: int = 25,  # Must match max_epochs default (25)
    telemetry_config: "TelemetryConfig | None" = None,
    plateau_threshold: float = 0.5,
    improvement_threshold: float = 2.0,
) -> tuple[PPOAgent, list[dict]]:
    """Train PPO with vectorized environments using INVERTED CONTROL FLOW.

    Key architecture: Instead of iterating environments then dataloaders,
    we iterate dataloader batches FIRST, then run all environments in parallel
    using CUDA streams. This ensures both GPUs are working simultaneously.

    Args:
        n_episodes: Total episodes to train
        n_envs: Number of parallel environments
        max_epochs: Max epochs per episode (RL timesteps per episode)
        device: Device for policy network
        devices: List of devices for environments (e.g., ["cuda:0", "cuda:1"])
        use_telemetry: Whether to use telemetry features
        lr: Learning rate
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        gamma: Discount factor
        ppo_updates_per_batch: Number of PPO updates per batch of episodes.
            Higher values improve sample efficiency but risk policy divergence.
            With KL early stopping enabled, values of 2-4 are often safe.
            Default: 1 (standard PPO behavior)
        save_path: Optional path to save model
        resume_path: Optional path to resume from checkpoint
        seed: Random seed for reproducibility
        plateau_threshold: Rolling average delta threshold below which training is considered
            plateaued (emits PLATEAU_DETECTED event). Compares current vs previous batch's
            rolling average. Scale-dependent: adjust for accuracy scales (e.g., 0-1 vs 0-100).
        improvement_threshold: Rolling average delta threshold above which training shows
            significant improvement/degradation (emits IMPROVEMENT_DETECTED/DEGRADATION_DETECTED).
            Events align with displayed rolling_avg_accuracy trend.

    Returns:
        Tuple of (trained_agent, training_history)
    """
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker

    if devices is None:
        devices = [device]

    task_spec = get_task_spec(task)
    ActionEnum = task_spec.action_enum

    print("=" * 60)
    print("PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)")
    print("=" * 60)
    print(f"Task: {task_spec.name} (topology={task_spec.topology}, type={task_spec.task_type})")
    print(f"Episodes: {n_episodes} (across {n_envs} parallel envs)")
    print(f"Max epochs per episode: {max_epochs}")
    print(f"Policy device: {device}")
    print(f"Env devices: {devices} ({n_envs // len(devices)} envs per device)")
    print(f"Random seed: {seed}")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    if entropy_anneal_episodes > 0:
        print(f"Entropy annealing: {entropy_coef_start or entropy_coef} -> {entropy_coef_end or entropy_coef} over {entropy_anneal_episodes} episodes")
    else:
        print(f"Entropy coef: {entropy_coef} (fixed)")
    print(f"Learning rate: {lr}")
    print(f"Telemetry features: {'ENABLED' if use_telemetry else 'DISABLED'}")
    if gpu_preload:
        print(f"GPU preload: ENABLED (8x faster data loading)")
    print()

    # Map environments to devices in round-robin (needed for SharedBatchIterator)
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    # Create SharedBatchIterator - single DataLoader serving all environments
    # This eliminates the N×M worker overhead from N independent DataLoaders
    if gpu_preload:
        # GPU-resident data loading: 8x faster than CPU DataLoader workers
        # Data is loaded once to GPU and reused across all environments
        # NOTE: GPU preload doesn't use SharedBatchIterator (data already on GPU)
        from esper.utils.data import load_cifar10_gpu
        print(f"Preloading {task_spec.name} to GPU (one-time cost)...")

        def create_env_dataloaders(env_idx: int, base_seed: int):
            """Create GPU-resident DataLoaders."""
            gen = torch.Generator(device='cpu')  # Generator for shuffle
            gen.manual_seed(base_seed + env_idx * 7919)
            env_device = devices[env_idx % len(devices)]
            return load_cifar10_gpu(
                batch_size=512,
                generator=gen,
                device=env_device,
            )

        # Create DataLoaders once and reuse across all batches
        env_dataloaders = [create_env_dataloaders(i, seed) for i in range(n_envs)]
        num_train_batches = len(env_dataloaders[0][0])
        num_test_batches = len(env_dataloaders[0][1])
        shared_train_iter = None  # Not using SharedBatchIterator for GPU preload
        shared_test_iter = None
    else:
        # SharedBatchIterator: Single DataLoader with combined batch size
        # Splits batches across environments and moves to correct devices
        print(f"Loading {task_spec.name} (SharedBatchIterator, 1 DataLoader for {n_envs} envs)...")

        # Get raw datasets from task spec
        train_dataset, test_dataset = task_spec.get_datasets()

        # Determine batch size and workers
        batch_size_per_env = task_spec.dataloader_defaults.get("batch_size", 128)
        if task_spec.name == "cifar10":
            batch_size_per_env = 512  # High-throughput setting for CIFAR
        effective_workers = num_workers if num_workers is not None else 4

        # Create generator for reproducible shuffling
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Create shared iterators for train and test
        shared_train_iter = SharedBatchIterator(
            dataset=train_dataset,
            batch_size_per_env=batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=effective_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            generator=gen,
        )

        # Test iterator: num_workers=0 because validation is infrequent (once per epoch)
        # No point spawning persistent workers for ~2% of total iteration time
        shared_test_iter = SharedBatchIterator(
            dataset=test_dataset,
            batch_size_per_env=batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=0,  # Validation is fast enough without workers
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        num_train_batches = len(shared_train_iter)
        num_test_batches = len(shared_test_iter)
        env_dataloaders = None  # Not using per-env dataloaders

    def loss_and_correct(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module):
        """Compute loss and correct counts for classification or LM."""
        if task_spec.task_type == "lm":
            vocab = outputs.size(-1)
            loss = criterion(outputs.view(-1, vocab), targets.view(-1))
            predicted = outputs.argmax(dim=-1)
            correct = predicted.eq(targets).sum()
            total = targets.numel()
        else:
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum()
            total = targets.size(0)
        return loss, correct, total

    # State dimension: 35 base features + 10 telemetry features if enabled
    BASE_FEATURE_DIM = 35
    state_dim = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
    # Use EMA momentum for stable normalization during long training runs
    # (prevents distribution shift that can break PPO ratio calculations)
    obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)

    # Reward normalizer for critic stability (prevents value loss explosion)
    # Essential after ransomware fix where reward magnitudes changed significantly
    reward_normalizer = RewardNormalizer(clip=10.0)

    # Convert episode-based annealing to step-based
    # Each batch of n_envs episodes = 1 PPO update step
    entropy_anneal_steps = entropy_anneal_episodes // n_envs if entropy_anneal_episodes > 0 else 0

    # Create or resume PPO agent
    start_episode = 0
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        agent = PPOAgent.load(resume_path, device=device)

        # Restore observation normalizer state
        metadata = checkpoint.get('metadata', {})
        if 'obs_normalizer_mean' in metadata:
            # Create tensors directly on target device to avoid CPU->GPU transfer
            obs_normalizer.mean = torch.tensor(metadata['obs_normalizer_mean'], device=device)
            obs_normalizer.var = torch.tensor(metadata['obs_normalizer_var'], device=device)
            obs_normalizer._device = device
            # Restore count for correct Welford/EMA continuation
            if 'obs_normalizer_count' in metadata:
                obs_normalizer.count = torch.tensor(metadata['obs_normalizer_count'], device=device)
            # Restore momentum (critical for EMA mode - affects normalization dynamics)
            if 'obs_normalizer_momentum' in metadata:
                obs_normalizer.momentum = metadata['obs_normalizer_momentum']
            print(f"  Restored observation normalizer state (momentum={obs_normalizer.momentum})")

        # Calculate starting episode from checkpoint
        if 'n_episodes' in metadata:
            start_episode = metadata['n_episodes']
            print(f"  Resuming from episode {start_episode}")
    else:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=len(ActionEnum),
            lr=lr,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_end=entropy_coef_end,
            entropy_coef_min=entropy_coef_min,
            adaptive_entropy_floor=adaptive_entropy_floor,
            entropy_anneal_steps=entropy_anneal_steps,
            gamma=gamma,
            device=device,
            recurrent=recurrent,
            lstm_hidden_dim=lstm_hidden_dim,
            chunk_length=chunk_length,
        )

    # ==========================================================================
    # Blueprint Analytics + Nissa Hub Wiring
    # ==========================================================================
    analytics = BlueprintAnalytics()
    hub = get_hub()
    hub.add_backend(analytics)

    def make_telemetry_callback(env_idx: int):
        """Create callback that injects env_id before emitting to hub."""

        def callback(event: TelemetryEvent):
            event.data["env_id"] = env_idx
            hub.emit(event)

        return callback

    def create_env_state(env_idx: int, base_seed: int) -> ParallelEnvState:
        """Create environment state with CUDA stream.

        DataLoaders are now shared via SharedBatchIterator, not per-env.
        """
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(task=task_spec, device=env_device)

        # Wire telemetry callback with env_id injection
        model.seed_slot.on_telemetry = make_telemetry_callback(env_idx)
        model.seed_slot.fast_mode = False  # Enable telemetry
        # Incubator mode gradient isolation: detach host input into the seed path so
        # host gradients remain identical to the host-only model while the seed
        # trickle-learns via STE in TRAINING. The host optimizer still steps
        # every batch; isolation only affects gradients through the seed branch.
        model.seed_slot.isolate_gradients = True

        # Set host_params baseline for scoreboard via Nissa analytics
        host_params = sum(p.numel() for p in model.host.parameters() if p.requires_grad)
        analytics.set_host_params(env_idx, host_params)
        # Snapshot current cumulative params so rent uses per-episode delta.
        params_added_baseline = analytics._get_scoreboard(env_idx).params_added

        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
        )

        # Create CUDA stream for this environment
        stream = torch.cuda.Stream(device=env_device) if 'cuda' in env_device else None

        # Create Governor for fail-safe watchdog
        # Conservative settings to avoid false positives during seed blending:
        # - sensitivity=6.0: 6-sigma is very rare for Gaussian
        # - history_window=20: longer window smooths blending transients
        # - min_panics=3: require 3 consecutive anomalies before rollback
        governor = TolariaGovernor(
            model=model,
            sensitivity=6.0,
            absolute_threshold=12.0,
            death_penalty=10.0,
            history_window=20,
            min_panics_before_rollback=3,
        )
        governor.snapshot()  # Ensure rollback is always possible before first panic

        env_state = ParallelEnvState(
            model=model,
            host_optimizer=host_optimizer,
            seed_optimizer=None,
            signal_tracker=SignalTracker(env_id=env_idx),
            governor=governor,
            env_device=env_device,
            stream=stream,
            seeds_created=0,
            episode_rewards=[],
            action_enum=ActionEnum,
            params_added_baseline=params_added_baseline,
        )
        # Pre-allocate accumulators to avoid per-epoch allocation churn
        env_state.init_accumulators()
        return env_state

    def process_train_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                            targets: torch.Tensor, criterion: nn.Module,
                            use_telemetry: bool = False) -> tuple[torch.Tensor, torch.Tensor, int, dict | None]:
        """Process a single training batch for one environment (runs in CUDA stream).

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.

        Returns:
            Tuple of (loss_tensor, correct_tensor, total, grad_stats)
            grad_stats is None if use_telemetry=False or no active seed in TRAINING stage
        """
        model = env_state.model
        seed_state = model.seed_state
        env_dev = env_state.env_device
        grad_stats = None

        # Use CUDA stream for async execution
        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            # Move data asynchronously
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            model.train()

            # Determine which optimizer to use based on seed state
            if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
                # No seed or seed fully integrated - train host only
                optimizer = env_state.host_optimizer
            elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
                # Incubator/isolated seed training: train host + seed in lockstep.
                if seed_state.stage == SeedStage.GERMINATED:
                    gate_result = model.seed_slot.advance_stage(SeedStage.TRAINING)
                    if not gate_result.passed:
                        raise RuntimeError(f"G1 gate failed during TRAINING entry: {gate_result}")
                if env_state.seed_optimizer is None:
                    env_state.seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
                    )
                # Host optimizer remains primary; seed optimizer is stepped separately.
                optimizer = env_state.host_optimizer
            elif seed_state.stage == SeedStage.BLENDING:
                # Active blending - train both host and seed jointly
                # Note: Alpha is driven by step_epoch() once per epoch, not per batch
                # This keeps alpha progression in sync with the auto-advance counter
                optimizer = env_state.host_optimizer
            elif seed_state.stage == SeedStage.PROBATIONARY:
                # Post-blending validation - alpha locked at 1.0, joint training
                optimizer = env_state.host_optimizer
            else:
                # Unknown stage - shouldn't happen
                optimizer = env_state.host_optimizer

            optimizer.zero_grad()
            if (
                seed_state
                and seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING, SeedStage.BLENDING)
                and env_state.seed_optimizer
            ):
                env_state.seed_optimizer.zero_grad()

            outputs = model(inputs)
            loss, correct_tensor, total = loss_and_correct(outputs, targets, criterion)
            loss.backward()

            # Collect gradient stats for telemetry (after backward, before step)
            # Use async version to avoid .item() sync inside stream context
            # Collect DUAL gradients (host + seed) to compute gradient ratio for G2 gate
            if use_telemetry and seed_state and seed_state.stage in (SeedStage.TRAINING, SeedStage.BLENDING):
                grad_stats = collect_dual_gradients_async(
                    model.get_host_parameters(),
                    model.get_seed_parameters(),
                )

            optimizer.step()
            if (
                seed_state
                and seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING, SeedStage.BLENDING)
                and env_state.seed_optimizer
            ):
                env_state.seed_optimizer.step()

            # Return tensors - .item() called after stream sync
            return loss.detach(), correct_tensor, total, grad_stats

    def process_val_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                          targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a validation batch for one environment.

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.
        """
        model = env_state.model
        env_dev = env_state.env_device

        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                loss, correct_tensor, total = loss_and_correct(outputs, targets, criterion)

            # Return tensors - .item() called after stream sync
            return loss, correct_tensor, total

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []
    prev_rolling_avg_acc: float | None = None  # Track previous rolling avg for trend detection

    episodes_completed = start_episode
    total_episodes = n_episodes + start_episode  # Total target including resumed episodes

    batch_idx = 0
    while episodes_completed < total_episodes:
        # Determine how many envs to run this batch (may be fewer than n_envs for last batch)
        remaining = total_episodes - episodes_completed
        envs_this_batch = min(n_envs, remaining)

        # Create fresh environments for this batch
        # DataLoaders are shared via SharedBatchIterator (not per-env)
        base_seed = seed + batch_idx * 10000
        env_states = [
            create_env_state(i, base_seed)
            for i in range(envs_this_batch)
        ]
        criterion = nn.CrossEntropyLoss()

        # Initialize episode for recurrent policy
        if recurrent:
            for env_idx in range(envs_this_batch):
                agent.recurrent_buffer.start_episode(env_id=env_idx)
                env_states[env_idx].lstm_hidden = None  # Fresh hidden for new episode

        # Per-env accumulators
        env_final_accs = [0.0] * envs_this_batch
        env_total_rewards = [0.0] * envs_this_batch

        # Accumulate raw (unnormalized) states for deferred normalizer update.
        # We freeze normalizer stats during rollout to ensure consistent normalization
        # across all states in a batch, then update stats after PPO update.
        raw_states_for_normalizer_update = []

        # Track if any Governor rollback occurred during this batch.
        # If so, the buffer contains stale transitions from a different model state
        # and must be cleared before PPO update.
        batch_rollback_occurred = False

        # Run epochs with INVERTED CONTROL FLOW
        for epoch in range(1, max_epochs + 1):
            # Track gradient stats per env for telemetry sync
            env_grad_stats = [None] * envs_this_batch

            # Reset per-epoch metrics by zeroing pre-allocated accumulators (faster than reallocating)
            train_totals = [0] * envs_this_batch
            for env_state in env_states:
                env_state.zero_accumulators()

            # ===== TRAINING: Iterate batches first, launch all envs via CUDA streams =====
            # SharedBatchIterator: single DataLoader, batches pre-split and moved to devices
            # GPU preload fallback: per-env DataLoaders (data already on GPU)

            # Issue one wait_stream per env BEFORE the loop starts (not per-batch).
            # This syncs the accumulator zeroing on default stream before we write.
            # record_stream marks tensors as used by this stream, preventing deallocation.
            for i, env_state in enumerate(env_states):
                if env_state.stream:
                    env_state.train_loss_accum.record_stream(env_state.stream)
                    env_state.train_correct_accum.record_stream(env_state.stream)
                    env_state.stream.wait_stream(torch.cuda.default_stream(env_state.env_device))

            # Choose iteration strategy based on data loading mode
            if shared_train_iter is not None:
                # SharedBatchIterator: single iterator yields list of (inputs, targets) per env
                train_iter = iter(shared_train_iter)
                for batch_step in range(num_train_batches):
                    try:
                        env_batches = next(train_iter)  # List of (inputs, targets), already on devices
                    except StopIteration:
                        break

                    # Launch all environments in their respective CUDA streams (async)
                    # Data already moved to correct device by SharedBatchIterator
                    for i, env_state in enumerate(env_states):
                        if i >= len(env_batches):
                            continue
                        inputs, targets = env_batches[i]
                        loss_tensor, correct_tensor, total, grad_stats = process_train_batch(
                            env_state, inputs, targets, criterion, use_telemetry=use_telemetry
                        )
                        if grad_stats is not None:
                            env_grad_stats[i] = grad_stats  # Keep last batch's grad stats
                        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                        with stream_ctx:
                            env_state.train_loss_accum.add_(loss_tensor)
                            env_state.train_correct_accum.add_(correct_tensor)
                        train_totals[i] += total
            else:
                # GPU preload fallback: per-env DataLoaders (data already on GPU)
                train_iters = [iter(env_dataloaders[i][0]) for i in range(envs_this_batch)]
                for batch_step in range(num_train_batches):
                    env_batches = []
                    for i, train_iter_i in enumerate(train_iters):
                        try:
                            inputs, targets = next(train_iter_i)
                            env_batches.append((inputs, targets))
                        except StopIteration:
                            env_batches.append(None)

                    # Launch all environments in their respective CUDA streams (async)
                    for i, env_state in enumerate(env_states):
                        if env_batches[i] is None:
                            continue
                        inputs, targets = env_batches[i]
                        loss_tensor, correct_tensor, total, grad_stats = process_train_batch(
                            env_state, inputs, targets, criterion, use_telemetry=use_telemetry
                        )
                        if grad_stats is not None:
                            env_grad_stats[i] = grad_stats  # Keep last batch's grad stats
                        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                        with stream_ctx:
                            env_state.train_loss_accum.add_(loss_tensor)
                            env_state.train_correct_accum.add_(correct_tensor)
                        train_totals[i] += total

            # Sync all streams ONCE at epoch end
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item() - all GPU work done
            train_losses = [env_state.train_loss_accum.item() for env_state in env_states]
            train_corrects = [env_state.train_correct_accum.item() for env_state in env_states]

            # ===== VALIDATION + COUNTERFACTUAL (FUSED): Single pass over test data =====
            # Instead of iterating test data twice (once for main validation, once for
            # counterfactual), we fuse both into a single loop. For each batch, we run:
            # 1. Main validation (real alpha) - accumulates val_correct_accum
            # 2. Counterfactual (alpha=0) - accumulates cf_correct_accum (same batch!)
            # This eliminates DataLoader overhead and halves test iteration time.
            # Note: val/cf accumulators were already zeroed by zero_accumulators() above

            val_totals = [0] * envs_this_batch

            # Determine which envs need counterfactual BEFORE the loop
            # (seed with alpha > 0 means the seed is contributing to output)
            envs_needing_counterfactual = set()
            for i, env_state in enumerate(env_states):
                seed_state = env_state.model.seed_state
                if seed_state and seed_state.alpha > 0:
                    envs_needing_counterfactual.add(i)

            cf_totals = {i: 0 for i in envs_needing_counterfactual}
            baseline_accs = [None] * envs_this_batch

            # Issue one wait_stream per env before the loop starts (not per-batch)
            # This syncs the accumulator zeroing on default stream before we write.
            for i, env_state in enumerate(env_states):
                if env_state.stream:
                    env_state.val_loss_accum.record_stream(env_state.stream)
                    env_state.val_correct_accum.record_stream(env_state.stream)
                    env_state.stream.wait_stream(torch.cuda.default_stream(env_state.env_device))
                    if i in envs_needing_counterfactual:
                        env_state.cf_correct_accum.record_stream(env_state.stream)

            # Choose iteration strategy based on data loading mode
            if shared_test_iter is not None:
                # SharedBatchIterator: single iterator yields list of (inputs, targets) per env
                test_iter = iter(shared_test_iter)
                for batch_step in range(num_test_batches):
                    try:
                        env_batches = next(test_iter)  # List of (inputs, targets), already on devices
                    except StopIteration:
                        break

                    # Launch all environments: MAIN VALIDATION + COUNTERFACTUAL on same batch
                    for i, env_state in enumerate(env_states):
                        if i >= len(env_batches):
                            continue
                        inputs, targets = env_batches[i]

                        # MAIN VALIDATION (real alpha)
                        loss_tensor, correct_tensor, total = process_val_batch(env_state, inputs, targets, criterion)
                        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                        with stream_ctx:
                            env_state.val_loss_accum.add_(loss_tensor)
                            env_state.val_correct_accum.add_(correct_tensor)
                        val_totals[i] += total

                        # COUNTERFACTUAL (alpha=0) - SAME BATCH, no DataLoader reload!
                        # Data is already on GPU from the main validation pass.
                        if i in envs_needing_counterfactual:
                            with env_state.model.seed_slot.force_alpha(0.0):
                                _, cf_correct_tensor, cf_total = process_val_batch(
                                    env_state, inputs, targets, criterion
                                )
                            with stream_ctx:
                                env_state.cf_correct_accum.add_(cf_correct_tensor)
                            cf_totals[i] += cf_total
            else:
                # GPU preload fallback: per-env DataLoaders (data already on GPU)
                test_iters = [iter(env_dataloaders[i][1]) for i in range(envs_this_batch)]
                for batch_step in range(num_test_batches):
                    env_batches = []
                    for i, test_iter_i in enumerate(test_iters):
                        try:
                            inputs, targets = next(test_iter_i)
                            env_batches.append((inputs, targets))
                        except StopIteration:
                            env_batches.append(None)

                    # Launch all environments: MAIN VALIDATION + COUNTERFACTUAL on same batch
                    for i, env_state in enumerate(env_states):
                        if env_batches[i] is None:
                            continue
                        inputs, targets = env_batches[i]

                        # MAIN VALIDATION (real alpha)
                        loss_tensor, correct_tensor, total = process_val_batch(env_state, inputs, targets, criterion)
                        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                        with stream_ctx:
                            env_state.val_loss_accum.add_(loss_tensor)
                            env_state.val_correct_accum.add_(correct_tensor)
                        val_totals[i] += total

                        # COUNTERFACTUAL (alpha=0) - SAME BATCH, no DataLoader reload!
                        # Data is already on GPU from the main validation pass.
                        if i in envs_needing_counterfactual:
                            with env_state.model.seed_slot.force_alpha(0.0):
                                _, cf_correct_tensor, cf_total = process_val_batch(
                                    env_state, inputs, targets, criterion
                                )
                            with stream_ctx:
                                env_state.cf_correct_accum.add_(cf_correct_tensor)
                            cf_totals[i] += cf_total

            # Single sync point at end (not once per pass)
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item()
            val_losses = [env_state.val_loss_accum.item() for env_state in env_states]
            val_corrects = [env_state.val_correct_accum.item() for env_state in env_states]

            # Compute baseline accuracies for counterfactual envs
            for i in envs_needing_counterfactual:
                baseline_accs[i] = 100.0 * env_states[i].cf_correct_accum.item() / max(cf_totals[i], 1)

            # ===== Compute epoch metrics and get BATCHED actions =====
            # First, sync telemetry for envs with active seeds (must happen BEFORE feature extraction)
            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state

                if use_telemetry and seed_state and env_grad_stats[env_idx]:
                    # Materialize async dual grad stats NOW (after stream sync, safe to call .item())
                    dual_stats = materialize_dual_grad_stats(env_grad_stats[env_idx])

                    # Update gradient ratio EMA for G2 gate
                    current_ratio = dual_stats.normalized_ratio
                    if not env_state.gradient_ratio_ema_initialized:
                        env_state.gradient_ratio_ema = current_ratio
                        env_state.gradient_ratio_ema_initialized = True
                    else:
                        # EMA with momentum=0.9 to smooth per-step noise
                        env_state.gradient_ratio_ema = 0.9 * env_state.gradient_ratio_ema + 0.1 * current_ratio

                    # Sync ratio to SeedMetrics for G2 gate evaluation
                    seed_state.metrics.seed_gradient_norm_ratio = env_state.gradient_ratio_ema

                    # Sync telemetry using seed gradient stats from dual collection
                    seed_state.sync_telemetry(
                        gradient_norm=dual_stats.seed_grad_norm,
                        gradient_health=1.0,  # Simplified: dual stats don't compute health
                        has_vanishing=dual_stats.seed_grad_norm < 1e-7,
                        has_exploding=dual_stats.seed_grad_norm > 100.0,
                        epoch=epoch,
                        max_epochs=max_epochs,
                    )

                # NOTE: step_epoch() moved to after record_accuracy() to ensure
                # accuracy_at_blending_start snapshot uses current epoch's val_acc

            # Collect features and action masks from all environments
            all_features = []
            all_masks = []
            all_signals = []
            governor_panic_envs = []  # Track which envs need rollback

            # Number of germinate actions = total actions - 3 (WAIT, FOSSILIZE, CULL)
            num_germinate_actions = len(ActionEnum) - 3

            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state

                train_loss = train_losses[env_idx] / num_train_batches
                train_acc = 100.0 * train_corrects[env_idx] / max(train_totals[env_idx], 1)
                val_loss = val_losses[env_idx] / num_test_batches
                val_acc = 100.0 * val_corrects[env_idx] / max(val_totals[env_idx], 1)

                # Store metrics for later
                env_state.train_loss = train_loss
                env_state.train_acc = train_acc
                env_state.val_loss = val_loss
                env_state.val_acc = val_acc

                # Governor watchdog: snapshot when loss is stable (every 5 epochs)
                if epoch % 5 == 0:
                    env_state.governor.snapshot()

                # Governor watchdog: check vital signs after validation
                is_panic = env_state.governor.check_vital_signs(val_loss)
                if is_panic:
                    governor_panic_envs.append(env_idx)

                # Record accuracy in seed metrics for reward shaping
                if seed_state and seed_state.metrics:
                    seed_state.metrics.record_accuracy(val_acc)

                    # Log counterfactual contribution if available
                    if baseline_accs[env_idx] is not None:
                        cf_contribution = val_acc - baseline_accs[env_idx]
                        if epoch == max_epochs:  # Only log at episode end to reduce noise
                            print(f"  [ENV {env_idx}] Counterfactual: {val_acc:.1f}% real, "
                                  f"{baseline_accs[env_idx]:.1f}% baseline, "
                                  f"Δ={cf_contribution:+.2f}% seed contribution")
                    # NOTE: step_epoch() moved to AFTER transition storage for state/action alignment

                # Update signal tracker
                active_seeds = [seed_state] if seed_state else []
                available_slots = 0 if model.has_active_seed else 1
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

                features = signals_to_features(signals, model, use_telemetry=use_telemetry)
                all_features.append(features)

                # Compute action mask based on current state
                has_active = 1.0 if model.has_active_seed else 0.0
                seed_stage = seed_state.stage.value if seed_state else 0
                seed_age = seed_state.metrics.epochs_total if seed_state else 0
                mask = compute_action_mask(
                    has_active_seed=has_active,
                    seed_stage=seed_stage,
                    num_germinate_actions=num_germinate_actions,
                    seed_age_epochs=seed_age,
                    epoch=signals.metrics.epoch,
                    plateau_epochs=signals.metrics.plateau_epochs,
                    host_stabilized=bool(signals.metrics.host_stabilized),
                )
                all_masks.append(mask)

            # Batch all states and masks into tensors
            states_batch = torch.tensor(all_features, dtype=torch.float32, device=device)
            masks_batch = torch.tensor(all_masks, dtype=torch.float32, device=device)

            # Accumulate raw states for deferred normalizer update
            raw_states_for_normalizer_update.append(states_batch.detach())

            # Normalize using FROZEN statistics during rollout collection.
            # IMPORTANT: We do NOT update obs_normalizer here - statistics are updated
            # AFTER the PPO update to ensure all states in a rollout batch use identical
            # normalization parameters. This prevents the "normalizer drift" bug where
            # states from different epochs within the same batch would be normalized
            # with different mean/var, causing PPO ratio calculation errors.
            states_batch_normalized = obs_normalizer.normalize(states_batch)

            # Get BATCHED actions from policy network with action masking (single forward pass!)
            if recurrent:
                # Collect and batch hidden states from all envs
                hiddens_h = []
                hiddens_c = []
                for env_state in env_states:
                    if env_state.lstm_hidden is None:
                        h, c = agent.network.get_initial_hidden(1, device)
                    else:
                        h, c = env_state.lstm_hidden
                    hiddens_h.append(h)
                    hiddens_c.append(c)

                # Batch hidden: [num_layers, n_envs, hidden]
                batch_h = torch.cat(hiddens_h, dim=1)
                batch_c = torch.cat(hiddens_c, dim=1)

                # Batched forward (no per-env loop!)
                with torch.inference_mode():
                    dist, values, (new_h, new_c) = agent.network.forward(
                        states_batch_normalized,
                        masks_batch,
                        hidden=(batch_h, batch_c),
                    )
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                # Store new hidden per-env
                # CRITICAL: .clone() is essential to avoid memory retention
                # - Slicing new_h[:, i:i+1, :] creates a VIEW into the batched tensor
                # - Views keep the entire parent tensor alive, preventing garbage collection
                # - .clone() creates an independent copy that allows the batch to be freed
                # - .detach() is technically redundant under inference_mode but defensive
                for i, env_state in enumerate(env_states):
                    env_state.lstm_hidden = (
                        new_h[:, i:i+1, :].detach().clone(),
                        new_c[:, i:i+1, :].detach().clone(),
                    )

                # Convert to lists for existing loop
                actions = actions.tolist()
                log_probs = log_probs.tolist()
                values = values.tolist()
            else:
                # Existing non-recurrent batched action selection
                actions, log_probs, values = agent.network.get_action_batch(
                    states_batch_normalized, masks_batch, deterministic=False
                )
                # Convert to Python lists to avoid per-element .item() syncs in loop below
                actions = actions.tolist()
                log_probs = log_probs.tolist()
                values = values.tolist()

            # Execute actions and store transitions for each environment
            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state
                signals = all_signals[env_idx]

                # Now Python floats/ints - no GPU sync
                action_idx = int(actions[env_idx])
                log_prob = log_probs[env_idx]
                value = values[env_idx]

                action = ActionEnum(action_idx)
                env_state.action_counts[action.name] += 1
                action_success = False

                # Governor rollback: execute if this env panicked
                if env_idx in governor_panic_envs:
                    report = env_state.governor.execute_rollback()
                    batch_rollback_occurred = True  # Mark batch as having stale transitions
                    print(f"  [ENV {env_idx}] Governor rollback: {report.reason} "
                          f"(threshold={report.loss_threshold:.4f}, panics={report.consecutive_panics})")

                # Compute reward with cost params
                # Derive cost from CURRENT architecture, not cumulative scoreboard
                # (scoreboard.params_added persists across episodes, causing stale rent)
                scoreboard = analytics._get_scoreboard(env_idx)
                host_params = scoreboard.host_params
                # Use env-local params (delta vs baseline) so rent resets with each fresh model.
                params_added_delta = max(0, scoreboard.params_added - env_state.params_added_baseline)
                total_params = params_added_delta + model.active_seed_params

                # Compute seed_contribution from counterfactual if available
                seed_contribution = None
                if baseline_accs[env_idx] is not None:
                    seed_contribution = env_state.val_acc - baseline_accs[env_idx]
                    # Store in metrics for telemetry at fossilize/cull
                    if seed_state and seed_state.metrics:
                        seed_state.metrics.counterfactual_contribution = seed_contribution

                # Determine if we need reward components (only at debug level)
                collect_reward_telemetry = (
                    telemetry_config is not None and telemetry_config.should_collect("debug")
                )

                # Unified reward computation - always use compute_contribution_reward
                # For pre-blending stages, seed_contribution is None and acc_delta is used as proxy
                reward_components = None
                if collect_reward_telemetry:
                    reward, reward_components = compute_contribution_reward(
                        action=action,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=total_params,
                        host_params=host_params,
                        acc_at_germination=env_state.acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        return_components=True,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                    )
                    if baseline_accs[env_idx] is not None:
                        reward_components.host_baseline_acc = baseline_accs[env_idx]
                else:
                    reward = compute_contribution_reward(
                        action=action,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=total_params,
                        host_params=host_params,
                        acc_at_germination=env_state.acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                    )

                # Governor punishment: inject negative reward if rollback occurred
                if env_idx in governor_panic_envs:
                    punishment = env_state.governor.get_punishment_reward()
                    reward += punishment
                    print(f"  [ENV {env_idx}] Punishment reward: {punishment:.1f} (final reward: {reward:.1f})")

                # Execute action
                if is_germinate_action(action):
                    if not model.has_active_seed:
                        # Record baseline accuracy for ransomware-resistant reward
                        env_state.acc_at_germination = env_state.val_acc
                        blueprint_id = get_blueprint_from_action(action)
                        seed_id = f"env{env_idx}_seed_{env_state.seeds_created}"
                        model.germinate_seed(blueprint_id, seed_id)
                        env_state.seeds_created += 1
                        env_state.seed_optimizer = None
                        action_success = True

                elif action == ActionEnum.FOSSILIZE:
                    # Capture total_improvement BEFORE state transition
                    seed_total_improvement = (
                        seed_state.metrics.total_improvement
                        if seed_state and seed_state.metrics else 0.0
                    )
                    action_success = _advance_active_seed(model)
                    if action_success:
                        # Track fossilization for terminal bonus calculation
                        env_state.seeds_fossilized += 1
                        # Only count as "contributing" if meets MIN_FOSSILIZE_CONTRIBUTION threshold
                        # This aligns with G5 gate and prevents terminal bonus for bad fossilizations
                        if seed_total_improvement >= MIN_FOSSILIZE_CONTRIBUTION:
                            env_state.contributing_fossilized += 1
                        # Seed is permanent now, clear germination baseline
                        env_state.acc_at_germination = None

                elif action == ActionEnum.CULL:
                    if model.has_active_seed:
                        model.cull_seed()
                        env_state.seed_optimizer = None
                        # Reset germination baseline on cull
                        env_state.acc_at_germination = None
                        action_success = True

                else:
                    # WAIT or any other no-op action always "succeeds"
                    action_success = True

                if action_success:
                    env_state.successful_action_counts[action.name] += 1

                # Emit reward telemetry if collecting (after action execution so we have action_success)
                if reward_components is not None:
                    reward_components.action_success = action_success
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.REWARD_COMPUTED,
                        seed_id=seed_state.seed_id if seed_state else None,
                        epoch=epoch,
                        data={
                            "env_id": env_idx,
                            **reward_components.to_dict(),
                        },
                        severity="debug",
                    ))

                # Normalize reward for critic stability (prevents value loss explosion)
                # Keep raw reward for episode_rewards display
                raw_reward = reward
                normalized_reward = reward_normalizer.update_and_normalize(reward)

                # Store transition with action mask (use normalized state to match what policy saw)
                # Keep tensors on policy device to avoid CPU round-trip overhead.
                # get_batches() expects tensors on CPU or will move them - since policy device
                # is consistent across all envs, keeping on GPU is more efficient.
                done = (epoch == max_epochs)
                truncated = done  # All episodes end at max_epochs (time limit truncation)
                bootstrap_value = value if truncated else 0.0  # Bootstrap from V(s_final) for truncation

                if recurrent:
                    agent.store_recurrent_transition(
                        state=states_batch_normalized[env_idx],
                        action=action_idx,
                        log_prob=log_prob,
                        value=value,
                        reward=normalized_reward,
                        done=done,
                        action_mask=masks_batch[env_idx],
                        env_id=env_idx,
                        truncated=truncated,
                        bootstrap_value=bootstrap_value,
                    )
                else:
                    agent.store_transition(
                        states_batch_normalized[env_idx],
                        action_idx,
                        log_prob,
                        value,
                        normalized_reward,  # Use normalized reward for critic training
                        done,
                        masks_batch[env_idx],
                        truncated=truncated,
                        bootstrap_value=bootstrap_value,
                    )

                # Handle episode boundaries for recurrent policy
                if done and recurrent:
                    agent.recurrent_buffer.end_episode(env_id=env_idx)
                    env_state.lstm_hidden = None  # Reset for next episode

                env_state.episode_rewards.append(raw_reward)  # Display raw for interpretability

                # Mechanical lifecycle advance (blending/shadowing dwell) AFTER RL transition
                # This ensures state/action/reward alignment - advance happens after the step is recorded
                # Must check seed_state exists since actions may have culled it
                if model.seed_state:
                    model.seed_slot.step_epoch()

                if epoch == max_epochs:
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

        # PPO Update after all episodes in batch complete
        # Truncation bootstrapping: Episodes end at max_epochs (time limit), not natural
        # termination. Each transition stores its bootstrap_value (V(s_final)) which GAE
        # uses instead of 0 for truncated episodes. This prevents systematic downward bias
        # in advantage estimates.
        #
        # Multiple updates per batch improves sample efficiency by reusing data.
        # With KL early stopping, the policy won't diverge too far from the
        # data collection distribution even with multiple updates.
        metrics = {}

        # If a Governor rollback occurred, the buffer contains transitions from
        # a different model state - training on them would cause distribution shift.
        # Clear the buffer and skip this PPO update.
        if batch_rollback_occurred:
            if recurrent:
                agent.recurrent_buffer.clear()
            else:
                agent.buffer.clear()
            print("[PPO] Buffer cleared due to Governor rollback - skipping update")
        else:
            if recurrent:
                # Single epoch for recurrent to avoid hidden state drift
                # (Multiple epochs would cause ratio bias as policy changes between epochs)
                update_metrics = agent.update_recurrent(n_epochs=1)
                metrics = update_metrics
            else:
                for update_i in range(ppo_updates_per_batch):
                    is_last_update = (update_i == ppo_updates_per_batch - 1)
                    update_metrics = agent.update(
                        last_value=0.0,
                        clear_buffer=is_last_update,
                        telemetry_config=telemetry_config,
                        current_episode=episodes_completed,
                        total_episodes=total_episodes,
                    )
                    if update_i == 0:
                        metrics = update_metrics
                    else:
                        # Aggregate metrics across updates (average)
                        for k, v in update_metrics.items():
                            if k in metrics:
                                metrics[k] = (metrics[k] + v) / 2
                            else:
                                metrics[k] = v
                    # Early exit if KL triggered early stopping
                    if update_metrics.get('early_stopped'):
                        break

            # NOW update the observation normalizer with all raw states from this batch.
            # This ensures the next batch will use updated statistics, but all states
            # within the same batch used identical normalization parameters.
            #
            # [Code Review Fix] Only update normalizer when PPO update succeeds.
            # If Governor rolled back, normalizer stats would be contaminated with
            # observations from a bad model state, causing distribution shift.
            if raw_states_for_normalizer_update:
                all_raw_states = torch.cat(raw_states_for_normalizer_update, dim=0)
                obs_normalizer.update(all_raw_states)

        # Track results
        avg_acc = sum(env_final_accs) / len(env_final_accs)
        avg_reward = sum(env_total_rewards) / len(env_total_rewards)

        recent_accuracies.append(avg_acc)
        recent_rewards.append(avg_reward)
        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

        episodes_completed += envs_this_batch
        print(f"Batch {batch_idx + 1}: Episodes {episodes_completed}/{n_episodes}")
        print(f"  Env accuracies: {[f'{a:.1f}%' for a in env_final_accs]}")
        print(f"  Avg acc: {avg_acc:.1f}% (rolling: {rolling_avg_acc:.1f}%)")
        print(f"  Avg reward: {avg_reward:.1f}")

        total_actions = {a.name: 0 for a in ActionEnum}
        successful_actions = {a.name: 0 for a in ActionEnum}
        for env_state in env_states:
            for a, c in env_state.action_counts.items():
                total_actions[a] += c
            for a, c in env_state.successful_action_counts.items():
                successful_actions[a] += c
        print(f"  Actions: {total_actions}")
        print(f"  Successful: {successful_actions}")

        if metrics:
            current_entropy_coef = agent.get_entropy_coef()
            print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                  f"Value loss: {metrics['value_loss']:.4f}, "
                  f"Entropy: {metrics['entropy']:.4f}, "
                  f"Entropy coef: {current_entropy_coef:.4f}")

            # Emit PPO telemetry
            # Note: clip_fraction, ratio_*, explained_variance not available in recurrent path
            ppo_event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data={
                    "batch": batch_idx + 1,
                    "episodes_completed": episodes_completed,
                    "train_steps": agent.train_steps,
                    # Core losses
                    "policy_loss": metrics.get("policy_loss", 0.0),
                    "value_loss": metrics.get("value_loss", 0.0),
                    "entropy": metrics.get("entropy", 0.0),
                    "entropy_coef": current_entropy_coef,
                    # PPO health (KL, clipping)
                    "approx_kl": metrics.get("approx_kl", 0.0),
                    "clip_fraction": metrics.get("clip_fraction", 0.0),
                    # Ratio statistics (early warning for policy collapse)
                    "ratio_max": metrics.get("ratio_max", 1.0),
                    "ratio_min": metrics.get("ratio_min", 1.0),
                    "ratio_std": metrics.get("ratio_std", 0.0),
                    # Value function health (negative = critic broken)
                    "explained_variance": metrics.get("explained_variance", 0.0),
                    # Early stopping info
                    "early_stop_epoch": metrics.get("early_stop_epoch"),
                    # Episode-level metrics
                    "avg_accuracy": avg_acc,
                    "avg_reward": avg_reward,
                    "rolling_avg_accuracy": rolling_avg_acc,
                },
            )
            hub.emit(ppo_event)

            # Emit training progress events based on actual rolling average trend
            # This aligns events with the displayed rolling_avg_accuracy
            if prev_rolling_avg_acc is not None:
                rolling_delta = rolling_avg_acc - prev_rolling_avg_acc

                if abs(rolling_delta) < plateau_threshold:  # True plateau - no significant change
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.PLATEAU_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
                elif rolling_delta < -improvement_threshold:  # Significant degradation
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.DEGRADATION_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
                elif rolling_delta > improvement_threshold:  # Significant improvement
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
            prev_rolling_avg_acc = rolling_avg_acc

        # Print analytics summary every 5 episodes
        if episodes_completed % 5 == 0 and len(analytics.stats) > 0:
            print()
            print(analytics.summary_table())
            for env_idx in range(n_envs):
                if env_idx in analytics.scoreboards:
                    print(analytics.scoreboard_table(env_idx))
            print()

        history.append({
            'batch': batch_idx + 1,
            'episodes': episodes_completed,
            'env_accuracies': list(env_final_accs),
            'avg_accuracy': avg_acc,
            'rolling_avg_accuracy': rolling_avg_acc,
            'avg_reward': avg_reward,
            'action_counts': total_actions,
            'entropy_coef': agent.get_entropy_coef(),
            **metrics,
        })

        if rolling_avg_acc > best_avg_acc:
            best_avg_acc = rolling_avg_acc
            # Store on CPU to save GPU memory (checkpoint is rarely loaded)
            best_state = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}

        batch_idx += 1

    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': episodes_completed,  # Total episodes trained (for resume)
            'n_envs': n_envs,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
            'use_telemetry': use_telemetry,
            'seed': seed,
            'obs_normalizer_mean': obs_normalizer.mean.tolist(),
            'obs_normalizer_var': obs_normalizer.var.tolist(),
            'obs_normalizer_count': obs_normalizer.count.item(),
            'obs_normalizer_momentum': obs_normalizer.momentum,
        })
        print(f"Model saved to {save_path}")

    # Add analytics to final history entry
    if history:
        history[-1]["blueprint_analytics"] = analytics.snapshot()

    return agent, history


__all__ = [
    "ParallelEnvState",
    "train_ppo_vectorized",
]
