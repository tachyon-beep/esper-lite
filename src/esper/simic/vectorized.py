"""Vectorized multi-GPU PPO training for Tamiyo.

This module implements high-performance vectorized PPO training using:
- Multiple parallel environments
- CUDA streams for async GPU execution
- Inverted control flow (batch-first iteration)
- Independent DataLoaders per environment to avoid GIL contention

Key Architecture:
Instead of iterating environments then dataloaders, we iterate dataloader
batches FIRST, then run all environments in parallel using CUDA streams.
This ensures both GPUs are working simultaneously.

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

from esper.leyline import SimicAction, SeedStage, SeedTelemetry
from esper.simic.buffers import RolloutBuffer
from esper.simic.networks import ActorCritic
from esper.simic.normalization import RunningMeanStd
from esper.simic.ppo import PPOAgent, signals_to_features
from esper.simic.rewards import compute_shaped_reward, SeedInfo


# =============================================================================
# Parallel Environment State
# =============================================================================

@dataclass
class ParallelEnvState:
    """State for a single parallel environment with CUDA stream for async execution.

    Each environment has its own DataLoader instances to avoid GIL contention
    when multiple environments iterate in parallel. This eliminates the bottleneck
    of a shared iterator being accessed from multiple CUDA streams.
    """
    model: nn.Module
    host_optimizer: torch.optim.Optimizer
    seed_optimizer: torch.optim.Optimizer | None
    signal_tracker: any  # SignalTracker from tamiyo
    env_device: str = "cuda:0"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
    # Independent DataLoaders per env to avoid GIL contention
    trainloader: any = None  # Environment's own trainloader
    testloader: any = None   # Environment's own testloader
    train_iter: any = None  # Persistent dataloader iterator
    test_iter: any = None   # Persistent testloader iterator
    seeds_created: int = 0
    episode_rewards: list = field(default_factory=list)
    action_counts: dict = field(default_factory=lambda: {a.name: 0 for a in SimicAction})
    # Metrics for current batch step
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0


# =============================================================================
# Vectorized PPO Training
# =============================================================================

def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = 4,
    max_epochs: int = 25,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    use_telemetry: bool = False,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.1,
    gamma: float = 0.99,
    save_path: str = None,
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
        save_path: Optional path to save model

    Returns:
        Tuple of (trained_agent, training_history)
    """
    from esper.tolaria import create_model
    from esper.utils import load_cifar10
    from esper.tamiyo import SignalTracker

    if devices is None:
        devices = [device]

    print("=" * 60)
    print("PPO Vectorized Training (INVERTED CONTROL FLOW + CUDA STREAMS)")
    print("=" * 60)
    print(f"Episodes: {n_episodes} (across {n_envs} parallel envs)")
    print(f"Max epochs per episode: {max_epochs}")
    print(f"Policy device: {device}")
    print(f"Env devices: {devices} ({n_envs // len(devices)} envs per device)")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Learning rate: {lr}")
    print()

    # Create independent DataLoaders per environment to avoid GIL contention.
    # When multiple CUDA streams try to iterate a shared DataLoader, the GIL
    # serializes access. Independent DataLoaders with unique random seeds
    # allow true parallel data loading.
    print(f"Loading CIFAR-10 ({n_envs} independent DataLoaders)...")

    def create_env_dataloaders(env_idx: int, base_seed: int):
        """Create DataLoaders with unique random seed for this environment."""
        # Use unique seed per environment for data shuffling
        gen = torch.Generator()
        gen.manual_seed(base_seed + env_idx * 7919)  # Prime number for seed spacing
        return load_cifar10(batch_size=512, generator=gen)

    # Create DataLoaders once and reuse across all batches
    # This allows persistent_workers to actually persist
    env_dataloaders = [create_env_dataloaders(i, 42) for i in range(n_envs)]
    num_train_batches = len(env_dataloaders[0][0])
    num_test_batches = len(env_dataloaders[0][1])

    # State dimension: 27 base features + 10 telemetry features if enabled
    BASE_FEATURE_DIM = 27
    state_dim = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
    obs_normalizer = RunningMeanStd((state_dim,))

    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=len(SimicAction),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        gamma=gamma,
        device=device,
    )

    # Map environments to devices in round-robin
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    def create_env_state(env_idx: int, base_seed: int, trainloader, testloader) -> ParallelEnvState:
        """Create environment state with CUDA stream, reusing pre-created DataLoaders."""
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(env_device)
        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        # Create CUDA stream for this environment
        stream = torch.cuda.Stream(device=env_device) if 'cuda' in env_device else None

        return ParallelEnvState(
            model=model,
            host_optimizer=host_optimizer,
            seed_optimizer=None,
            signal_tracker=SignalTracker(),
            env_device=env_device,
            stream=stream,
            trainloader=trainloader,
            testloader=testloader,
            train_iter=None,  # Will be set per epoch
            test_iter=None,
            seeds_created=0,
            episode_rewards=[],
            action_counts={a.name: 0 for a in SimicAction},
        )

    def process_train_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                            targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a single training batch for one environment (runs in CUDA stream).

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.
        """
        model = env_state.model
        seed_state = model.seed_state
        env_dev = env_state.env_device

        # Use CUDA stream for async execution
        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            # Move data asynchronously
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            model.train()

            # Determine which optimizer to use based on seed state
            if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
                optimizer = env_state.host_optimizer
            elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
                if seed_state.stage == SeedStage.GERMINATED:
                    seed_state.transition(SeedStage.TRAINING)
                    env_state.seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=0.01, momentum=0.9
                    )
                if env_state.seed_optimizer is None:
                    env_state.seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=0.01, momentum=0.9
                    )
                optimizer = env_state.seed_optimizer
            else:  # BLENDING
                optimizer = env_state.host_optimizer
                # Update blend alpha for this step
                if model.seed_slot and seed_state:
                    # Use epochs in current stage as step count (matches comparison.py pattern)
                    step = seed_state.metrics.epochs_in_current_stage
                    model.seed_slot.update_alpha_for_step(step)

            optimizer.zero_grad()
            if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
                env_state.seed_optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            if seed_state and seed_state.stage == SeedStage.BLENDING and env_state.seed_optimizer:
                env_state.seed_optimizer.step()

            _, predicted = outputs.max(1)
            correct_tensor = predicted.eq(targets).sum()  # Keep as tensor, no .item()

            # Return tensors - .item() called after stream sync
            return loss.detach(), correct_tensor, targets.size(0)

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
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct_tensor = predicted.eq(targets).sum()  # Keep as tensor, no .item()

            # Return tensors - .item() called after stream sync
            return loss, correct_tensor, targets.size(0)

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []

    episodes_completed = 0

    batch_idx = 0
    while episodes_completed < n_episodes:
        # Determine how many envs to run this batch (may be fewer than n_envs for last batch)
        remaining = n_episodes - episodes_completed
        envs_this_batch = min(n_envs, remaining)

        # Create fresh environments for this batch, reusing persistent DataLoaders
        base_seed = 42 + batch_idx * 10000
        env_states = [
            create_env_state(i, base_seed, env_dataloaders[i][0], env_dataloaders[i][1])
            for i in range(envs_this_batch)
        ]
        criterion = nn.CrossEntropyLoss()

        # Per-env accumulators
        env_final_accs = [0.0] * envs_this_batch
        env_total_rewards = [0.0] * envs_this_batch

        # Run epochs with INVERTED CONTROL FLOW
        for epoch in range(1, max_epochs + 1):
            # Reset per-epoch metrics - GPU tensors for accumulation, sync only at epoch end
            train_loss_accum = [torch.zeros(1, device=env_states[i].env_device) for i in range(envs_this_batch)]
            train_correct_accum = [torch.zeros(1, device=env_states[i].env_device) for i in range(envs_this_batch)]
            train_totals = [0] * envs_this_batch

            # ===== TRAINING: Iterate batches first, launch all envs via CUDA streams =====
            # Each env has its own DataLoader with independent shuffling
            # DataLoader workers (num_workers=4) provide parallel prefetching
            train_iters = [iter(env_state.trainloader) for env_state in env_states]
            for batch_step in range(num_train_batches):
                # Get batch from each env's own dataloader (independent shuffling)
                env_batches = []
                for i, train_iter in enumerate(train_iters):
                    try:
                        inputs, targets = next(train_iter)
                        env_batches.append((inputs, targets))
                    except StopIteration:
                        env_batches.append(None)

                # Launch all environments in their respective CUDA streams (async)
                # Accumulate on GPU inside stream context - no sync until epoch end
                for i, env_state in enumerate(env_states):
                    if env_batches[i] is None:
                        continue
                    inputs, targets = env_batches[i]
                    loss_tensor, correct_tensor, total = process_train_batch(env_state, inputs, targets, criterion)
                    # Accumulate inside stream context (in-place add respects stream ordering)
                    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                    with stream_ctx:
                        train_loss_accum[i].add_(loss_tensor)
                        train_correct_accum[i].add_(correct_tensor)
                    train_totals[i] += total

            # Sync all streams ONCE at epoch end
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item() - all GPU work done
            train_losses = [t.item() for t in train_loss_accum]
            train_corrects = [t.item() for t in train_correct_accum]

            # ===== VALIDATION: Same pattern - accumulate on GPU, sync once =====
            val_loss_accum = [torch.zeros(1, device=env_states[i].env_device) for i in range(envs_this_batch)]
            val_correct_accum = [torch.zeros(1, device=env_states[i].env_device) for i in range(envs_this_batch)]
            val_totals = [0] * envs_this_batch

            test_iters = [iter(env_state.testloader) for env_state in env_states]
            for batch_step in range(num_test_batches):
                env_batches = []
                for i, test_iter in enumerate(test_iters):
                    try:
                        inputs, targets = next(test_iter)
                        env_batches.append((inputs, targets))
                    except StopIteration:
                        env_batches.append(None)

                # Launch all environments in their respective CUDA streams (async)
                for i, env_state in enumerate(env_states):
                    if env_batches[i] is None:
                        continue
                    inputs, targets = env_batches[i]
                    loss_tensor, correct_tensor, total = process_val_batch(env_state, inputs, targets, criterion)
                    # Accumulate inside stream context
                    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                    with stream_ctx:
                        val_loss_accum[i].add_(loss_tensor)
                        val_correct_accum[i].add_(correct_tensor)
                    val_totals[i] += total

            # Sync all streams ONCE at epoch end
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item()
            val_losses = [t.item() for t in val_loss_accum]
            val_corrects = [t.item() for t in val_correct_accum]

            # ===== Compute epoch metrics and get BATCHED actions =====
            # Collect features from all environments
            all_features = []
            all_signals = []

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

                # Record accuracy in seed metrics for reward shaping
                if seed_state and seed_state.metrics:
                    seed_state.metrics.record_accuracy(val_acc)

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

                features = signals_to_features(signals, model, tracker=None, use_telemetry=use_telemetry)
                all_features.append(features)

            # Batch all states into a single tensor
            states_batch = torch.tensor(all_features, dtype=torch.float32, device=device)

            # Update observation normalizer and normalize
            obs_normalizer.update(states_batch.cpu())
            states_batch = obs_normalizer.normalize(states_batch)

            # Get BATCHED actions from policy network (single forward pass!)
            actions, log_probs, values = agent.network.get_action_batch(states_batch, deterministic=False)

            # Execute actions and store transitions for each environment
            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                seed_state = model.seed_state
                signals = all_signals[env_idx]

                action_idx = actions[env_idx].item()
                log_prob = log_probs[env_idx].item()
                value = values[env_idx].item()

                action = SimicAction(action_idx)
                env_state.action_counts[action.name] += 1

                # Compute shaped reward
                reward = compute_shaped_reward(
                    action=action.value,
                    acc_delta=signals.metrics.accuracy_delta,
                    val_acc=env_state.val_acc,
                    seed_info=SeedInfo.from_seed_state(seed_state),
                    epoch=epoch,
                    max_epochs=max_epochs,
                )

                # Execute action
                if SimicAction.is_germinate(action):
                    if not model.has_active_seed:
                        blueprint_id = SimicAction.get_blueprint_id(action)
                        seed_id = f"env{env_idx}_seed_{env_state.seeds_created}"
                        model.germinate_seed(blueprint_id, seed_id)
                        env_state.seeds_created += 1
                        env_state.seed_optimizer = None

                elif action == SimicAction.ADVANCE:
                    if model.has_active_seed:
                        if model.seed_state.stage == SeedStage.TRAINING:
                            model.seed_state.transition(SeedStage.BLENDING)
                            model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                        elif model.seed_state.stage == SeedStage.BLENDING:
                            model.seed_state.transition(SeedStage.FOSSILIZED)
                            model.seed_slot.set_alpha(1.0)

                elif action == SimicAction.CULL:
                    if model.has_active_seed:
                        model.cull_seed()
                        env_state.seed_optimizer = None

                # Store transition
                done = (epoch == max_epochs)
                agent.store_transition(
                    states_batch[env_idx].cpu(),
                    action_idx,
                    log_prob,
                    value,
                    reward,
                    done
                )

                env_state.episode_rewards.append(reward)

                if epoch == max_epochs:
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

        # PPO Update after all episodes in batch complete
        metrics = agent.update(last_value=0.0)

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

        total_actions = {a.name: 0 for a in SimicAction}
        for env_state in env_states:
            for a, c in env_state.action_counts.items():
                total_actions[a] += c
        print(f"  Actions: {total_actions}")

        if metrics:
            print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                  f"Value loss: {metrics['value_loss']:.4f}, "
                  f"Entropy: {metrics['entropy']:.4f}")

        history.append({
            'batch': batch_idx + 1,
            'episodes': episodes_completed,
            'env_accuracies': list(env_final_accs),
            'avg_accuracy': avg_acc,
            'rolling_avg_accuracy': rolling_avg_acc,
            'avg_reward': avg_reward,
            'action_counts': total_actions,
            **metrics,
        })

        if rolling_avg_acc > best_avg_acc:
            best_avg_acc = rolling_avg_acc
            best_state = {k: v.clone() for k, v in agent.network.state_dict().items()}

        batch_idx += 1

    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': n_episodes,
            'n_envs': n_envs,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
            'use_telemetry': use_telemetry,
            'obs_normalizer_mean': obs_normalizer.mean.tolist(),
            'obs_normalizer_var': obs_normalizer.var.tolist(),
        })
        print(f"Model saved to {save_path}")

    return agent, history


__all__ = [
    "ParallelEnvState",
    "train_ppo_vectorized",
]
