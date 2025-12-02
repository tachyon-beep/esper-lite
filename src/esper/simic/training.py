"""Training loops for PPO.

This module contains the main training functions extracted from ppo.py.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from esper.leyline.actions import get_blueprint_from_action, is_germinate_action
from esper.leyline import SeedTelemetry
from esper.runtime import get_task_spec
from esper.simic.rewards import compute_shaped_reward, SeedInfo
from esper.simic.gradient_collector import collect_seed_gradients
from esper.nissa import get_hub


# =============================================================================
# PPO helpers
# =============================================================================

def _loss_and_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, float, int]:
    """Compute loss and token/sample accuracy counts for classification or LM."""
    if task_type == "lm":
        vocab = outputs.size(-1)
        loss = criterion(outputs.view(-1, vocab), targets.view(-1))
        predicted = outputs.argmax(dim=-1)
        correct = float(predicted.eq(targets).sum().item())
        total = targets.numel()
    else:
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = float(predicted.eq(targets).sum().item())
        total = targets.size(0)
    return loss, correct, total


# =============================================================================
# PPO Episode Runner
# =============================================================================

def run_ppo_episode(
    agent,
    trainloader,
    testloader,
    max_epochs: int = 25,
    base_seed: int = 42,
    device: str = "cuda:0",
    task_spec=None,
    use_telemetry: bool = True,
    collect_rollout: bool = True,
    deterministic: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with the PPO agent."""
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
    from esper.simic.ppo import signals_to_features

    if task_spec is None:
        task_spec = get_task_spec("cifar10")
    ActionEnum = task_spec.action_enum
    task_type = task_spec.task_type

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(task=task_spec, device=device)

    # Wire Kasmina telemetry into global Nissa hub so fossilization and
    # lifecycle events propagate to configured backends (console, analytics).
    hub = get_hub()

    def telemetry_callback(event):
        # Single-env PPO uses env_id=0 for analytics compatibility.
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    model.seed_slot.on_telemetry = telemetry_callback
    model.seed_slot.fast_mode = False
    # Womb mode gradient isolation: detach host input into the seed path so host
    # gradients match the baseline model while the seed trickle-learns via STE.
    # Host optimizer still steps every batch; isolation only affects gradients
    # flowing through the seed branch.
    model.seed_slot.isolate_gradients = True

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    action_counts = {a.name: 0 for a in ActionEnum}
    episode_rewards = []

    # Track host params and added params for compute rent
    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    params_added = 0  # Accumulates when seeds are fossilized

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_state
        grad_stats = None  # Will collect gradient stats if use_telemetry and seed active

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if seed_state is None:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.GERMINATED:
            gate_result = model.seed_slot.advance_stage(SeedStage.TRAINING)
            if not gate_result.passed:
                raise RuntimeError(f"G1 gate failed during TRAINING entry: {gate_result}")
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=0.01, momentum=0.9
                )
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                host_optimizer.step()
                seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.TRAINING:
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=0.01, momentum=0.9
                )
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                host_optimizer.step()
                seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.BLENDING:
            # Alpha progression handled by model.seed_slot.step_epoch()
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                if seed_optimizer:
                    seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                host_optimizer.step()
                if seed_optimizer:
                    seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage in (SeedStage.SHADOWING, SeedStage.PROBATIONARY):
            # Seed fully blended; continue joint training without alpha ramp
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()

                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                host_optimizer.step()
                seed_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        elif seed_state.stage == SeedStage.FOSSILIZED:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                total += batch_total
                correct += correct_batch

        train_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                val_loss += loss.item()
                total += batch_total
                correct += correct_batch

        val_loss = val_loss / len(testloader) if len(testloader) > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        # Record accuracy in seed metrics for reward shaping
        if seed_state and seed_state.metrics:
            seed_state.metrics.record_accuracy(val_acc)

        # Update signal tracker
        active_seeds = [seed_state] if seed_state else []
        available_slots = 0 if model.has_active_seed else 1
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        # Sync telemetry after validation
        if use_telemetry and seed_state and grad_stats:
            seed_state.sync_telemetry(
                gradient_norm=grad_stats['gradient_norm'],
                gradient_health=grad_stats['gradient_health'],
                has_vanishing=grad_stats['has_vanishing'],
                has_exploding=grad_stats['has_exploding'],
                epoch=epoch,
                max_epochs=max_epochs,
            )

        acc_delta = signals.metrics.accuracy_delta

        # Mechanical lifecycle advance (blending/shadowing dwell)
        model.seed_slot.step_epoch()
        seed_state = model.seed_state

        # Get features and action
        # Note: tracker would provide DiagnosticTracker telemetry if available
        features = signals_to_features(signals, model, tracker=None, use_telemetry=use_telemetry)
        state = torch.tensor([features], dtype=torch.float32, device=device)

        action_idx, log_prob, value = agent.get_action(state, deterministic=deterministic)
        action = ActionEnum(action_idx)
        action_counts[action.name] += 1

        # Compute total params for rent (fossilized + active)
        total_params = params_added + model.active_seed_params
        reward = compute_shaped_reward(
            action=action,
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
        )

        # Execute action
        if is_germinate_action(action):
            if not model.has_active_seed:
                blueprint_id = get_blueprint_from_action(action)
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = None

        elif action == ActionEnum.FOSSILIZE:
            # NOTE: Only PROBATIONARY → FOSSILIZED is a valid lifecycle
            # transition per Leyline. From SHADOWING this advance_stage call
            # will fail the transition and return passed=False (no change).
            if model.has_active_seed and model.seed_state.stage in (SeedStage.PROBATIONARY, SeedStage.SHADOWING):
                gate_result = model.seed_slot.advance_stage(SeedStage.FOSSILIZED)
                if gate_result.passed:
                    params_added += model.active_seed_params
                    model.seed_slot.set_alpha(1.0)

        elif action == ActionEnum.CULL:
            if model.has_active_seed:
                model.cull_seed()
                seed_optimizer = None

        done = (epoch == max_epochs)

        if collect_rollout:
            agent.store_transition(
                state.squeeze(0).cpu(),
                action_idx,
                log_prob,
                value,
                reward,
                done
            )

        episode_rewards.append(reward)

    return val_acc, action_counts, episode_rewards


# =============================================================================
# PPO Training Loop
# =============================================================================

def train_ppo(
    n_episodes: int = 100,
    max_epochs: int = 25,
    update_every: int = 5,
    device: str = "cuda:0",
    task: str = "cifar10",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.1,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str | None = None,
    resume_path: str | None = None,
    seed: int | None = None,
):
    """Train PPO agent."""
    from esper.simic.ppo import PPOAgent
    from esper.utils import load_cifar10

    task_spec = get_task_spec(task)
    ActionEnum = task_spec.action_enum

    print("=" * 60)
    print("PPO Training for Tamiyo")
    print("=" * 60)
    print(f"Task: {task_spec.name} (topology={task_spec.topology}, type={task_spec.task_type})")
    print(f"Episodes: {n_episodes}, Max epochs: {max_epochs}")
    print(f"Device: {device}, Telemetry: {use_telemetry}")

    trainloader, testloader = task_spec.create_dataloaders()
    # State dimension: 27 base features + 10 telemetry features if enabled
    BASE_FEATURE_DIM = 27
    state_dim = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)

    # Convert episode-based annealing to step-based
    # CRITICAL: Non-vectorized training only updates every `update_every` episodes
    # So actual PPO updates = n_episodes / update_every
    # If update_every=5 and entropy_anneal_episodes=100, we get 20 PPO updates
    entropy_anneal_steps = (entropy_anneal_episodes // update_every) if entropy_anneal_episodes > 0 else 0

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=len(ActionEnum),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        entropy_coef_start=entropy_coef_start,
        entropy_coef_end=entropy_coef_end,
        entropy_coef_min=entropy_coef_min,
        entropy_anneal_steps=entropy_anneal_steps,
        gamma=gamma,
        device=device,
    )

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []

    for ep in range(1, n_episodes + 1):
        base_seed = 42 + ep * 1000

        final_acc, action_counts, rewards = run_ppo_episode(
            agent=agent,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            base_seed=base_seed,
            device=device,
            task_spec=task_spec,
            use_telemetry=use_telemetry,
            collect_rollout=True,
            deterministic=False,
        )

        total_reward = sum(rewards)
        recent_accuracies.append(final_acc)
        recent_rewards.append(total_reward)

        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        if ep % update_every == 0 or ep == n_episodes:
            metrics = agent.update(last_value=0.0)

            avg_acc = sum(recent_accuracies) / len(recent_accuracies)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            print(f"Episode {ep:3d}/{n_episodes}: acc={final_acc:.1f}% (avg={avg_acc:.1f}%), "
                  f"reward={total_reward:.1f}")

            history.append({
                'episode': ep,
                'accuracy': final_acc,
                'avg_accuracy': avg_acc,
                'total_reward': total_reward,
                'action_counts': dict(action_counts),
                **metrics,
            })

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_state = {k: v.clone() for k, v in agent.network.state_dict().items()}

    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': n_episodes,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
        })
        print(f"Model saved to {save_path}")

    return agent, history


# =============================================================================
# Heuristic Training
# =============================================================================

def run_heuristic_episode(
    policy,
    trainloader,
    testloader,
    max_epochs: int = 75,
    max_batches: int | None = None,
    base_seed: int = 42,
    device: str = "cuda:0",
    task_spec=None,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with heuristic policy.

    Args:
        policy: HeuristicTamiyo instance
        trainloader: Training data loader
        testloader: Test data loader
        max_epochs: Maximum epochs per episode
        max_batches: Limit batches per epoch (None = all)
        base_seed: Random seed
        device: Device to use
        task_spec: Task specification

    Returns:
        (final_accuracy, action_counts, episode_rewards)
    """
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker

    if task_spec is None:
        task_spec = get_task_spec("cifar10")
    ActionEnum = task_spec.action_enum
    task_type = task_spec.task_type

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(task=task_spec, device=device)

    # Wire telemetry
    hub = get_hub()
    def telemetry_callback(event):
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    model.seed_slot.on_telemetry = telemetry_callback
    model.seed_slot.fast_mode = False
    model.seed_slot.isolate_gradients = True

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    action_counts = {a.name: 0 for a in ActionEnum}
    episode_rewards = []

    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    params_added = 0

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_state

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for inputs, targets in trainloader:
            if max_batches and batch_count >= max_batches:
                break
            batch_count += 1

            inputs, targets = inputs.to(device), targets.to(device)
            host_optimizer.zero_grad()
            if seed_optimizer:
                seed_optimizer.zero_grad()

            outputs = model(inputs)
            loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
            loss.backward()

            host_optimizer.step()
            if seed_optimizer:
                seed_optimizer.step()

            running_loss += loss.item()
            total += batch_total
            correct += correct_batch

        train_loss = running_loss / max(1, batch_count)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                if max_batches and batch_count >= max_batches:
                    break
                batch_count += 1

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
                val_loss += loss.item()
                total += batch_total
                correct += correct_batch

        val_loss = val_loss / max(1, batch_count)
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        # Record accuracy in seed metrics
        if seed_state and seed_state.metrics:
            seed_state.metrics.record_accuracy(val_acc)

        # Update signal tracker
        active_seeds = [seed_state] if seed_state else []
        available_slots = 0 if model.has_active_seed else 1
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        acc_delta = signals.metrics.accuracy_delta

        # Mechanical lifecycle advance
        model.seed_slot.step_epoch()
        seed_state = model.seed_state

        # Get heuristic decision
        decision = policy.decide(signals, active_seeds)
        action = decision.action
        action_counts[action.name] += 1

        # Compute reward (for comparison with PPO)
        total_params = params_added + model.active_seed_params
        reward = compute_shaped_reward(
            action=action,
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
        )
        episode_rewards.append(reward)

        # Execute action
        if is_germinate_action(action):
            if not model.has_active_seed:
                blueprint_id = get_blueprint_from_action(action)
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=0.01, momentum=0.9
                )

        elif action.name == "FOSSILIZE":
            if model.has_active_seed and model.seed_state.stage == SeedStage.PROBATIONARY:
                gate_result = model.seed_slot.advance_stage(SeedStage.FOSSILIZED)
                if gate_result.passed:
                    params_added += model.active_seed_params
                    model.seed_slot.set_alpha(1.0)

        elif action.name == "CULL":
            if model.has_active_seed:
                model.cull_seed()
                seed_optimizer = None

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs:
            seed_info = f"seed={seed_state.stage.name}" if seed_state else "no seed"
            print(f"  Epoch {epoch:3d}/{max_epochs}: acc={val_acc:.1f}%, {seed_info}, action={action.name}")

    return val_acc, action_counts, episode_rewards


def train_heuristic(
    n_episodes: int = 1,
    max_epochs: int = 75,
    max_batches: int | None = 50,
    device: str = "cuda:0",
    task: str = "cifar10",
    seed: int = 42,
):
    """Train with heuristic policy.

    Args:
        n_episodes: Number of episodes to run
        max_epochs: Maximum epochs per episode
        max_batches: Limit batches per epoch (None = all, 50 = fast mode)
        device: Device to use
        task: Task preset (cifar10 or tinystories)
        seed: Random seed
    """
    from esper.tamiyo import HeuristicTamiyo

    task_spec = get_task_spec(task)

    print("=" * 60)
    print("Heuristic Training (h-esper)")
    print("=" * 60)
    print(f"Task: {task_spec.name} (topology={task_spec.topology})")
    print(f"Episodes: {n_episodes}, Max epochs: {max_epochs}")
    print(f"Batches per epoch: {max_batches or 'all'}")
    print(f"Device: {device}")
    print("=" * 60)

    trainloader, testloader = task_spec.create_dataloaders()

    policy = HeuristicTamiyo(topology=task_spec.topology)
    history = []

    for ep in range(1, n_episodes + 1):
        print(f"\nEpisode {ep}/{n_episodes}")
        print("-" * 40)

        policy.reset()
        base_seed = seed + ep * 1000

        final_acc, action_counts, rewards = run_heuristic_episode(
            policy=policy,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            max_batches=max_batches,
            base_seed=base_seed,
            device=device,
            task_spec=task_spec,
        )

        total_reward = sum(rewards)
        print(f"\nEpisode {ep} complete: acc={final_acc:.1f}%, reward={total_reward:.1f}")
        print(f"Actions: {dict(action_counts)}")

        history.append({
            'episode': ep,
            'accuracy': final_acc,
            'total_reward': total_reward,
            'action_counts': dict(action_counts),
        })

    return history


__all__ = [
    "run_ppo_episode",
    "train_ppo",
    "run_heuristic_episode",
    "train_heuristic",
]
