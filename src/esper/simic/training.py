"""Training loops for PPO and IQL.

This module contains the main training functions extracted from ppo.py and iql.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from esper.leyline import SimicAction, SeedTelemetry
from esper.simic.buffers import Transition, ReplayBuffer
from esper.simic.features import obs_to_base_features
from esper.simic.rewards import compute_shaped_reward, SeedInfo
from esper.simic.gradient_collector import collect_seed_gradients


# =============================================================================
# IQL Data Loading
# =============================================================================

def extract_transitions(
    pack: dict,
    reward_scale: float = 0.1,
    use_reward_shaping: bool = False,
    gamma: float = 0.99,
) -> list[Transition]:
    """Extract (s, a, r, s', done) transitions from episodes.

    Args:
        pack: Data pack dictionary
        reward_scale: Scale factor for rewards
        use_reward_shaping: Whether to add potential-based reward shaping
        gamma: Discount factor for reward shaping

    Returns:
        List of Transition objects (27-dim base features only)
    """
    from esper.simic.rewards import compute_seed_potential

    transitions = []

    for ep in pack['episodes']:
        decisions = ep['decisions']

        for i, decision in enumerate(decisions):
            state = obs_to_base_features(decision['observation'])
            # Legacy telemetry removed - only 27-dim base features supported

            # NOTE: Data packs must use current action space (7 actions with blueprint-specific
            # germinate variants). Old packs with 4-action space (generic GERMINATE) will KeyError.
            # See data/archive_old_action_space/README.md
            action = SimicAction[decision['action']['action']].value
            raw_reward = decision['outcome'].get('reward', 0) * reward_scale

            is_last = (i == len(decisions) - 1)
            if is_last:
                next_state = state
                done = True
                next_obs = None
            else:
                next_decision = decisions[i + 1]
                next_obs = next_decision['observation']
                next_state = obs_to_base_features(next_obs)
                done = False

            if use_reward_shaping:
                current_obs = decision['observation']
                phi_s = compute_seed_potential(current_obs)
                phi_s_prime = 0.0 if done else compute_seed_potential(next_obs)
                shaping_bonus = gamma * phi_s_prime - phi_s
                reward = raw_reward + shaping_bonus * reward_scale
            else:
                reward = raw_reward

            transitions.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))

    return transitions


# =============================================================================
# IQL Evaluation
# =============================================================================

def evaluate_iql_policy(
    iql,
    buffer: ReplayBuffer,
    n_samples: int = 1000,
) -> dict:
    """Evaluate IQL policy against behavior policy."""
    import torch

    idx = torch.randint(0, buffer.size, (n_samples,), device=buffer.device)
    states = buffer.states[idx]
    behavior_actions = buffer.actions[idx]

    with torch.no_grad():
        q_values = iql.q_network(states)
        iql_actions = q_values.argmax(dim=1)

    agreement = (iql_actions == behavior_actions).float().mean().item()
    behavior_dist = torch.bincount(behavior_actions, minlength=len(SimicAction)).float() / n_samples
    iql_dist = torch.bincount(iql_actions, minlength=len(SimicAction)).float() / n_samples

    q_behavior = q_values.gather(1, behavior_actions.unsqueeze(1)).squeeze(1).mean().item()
    q_iql = q_values.gather(1, iql_actions.unsqueeze(1)).squeeze(1).mean().item()

    return {
        "agreement": agreement,
        "q_behavior": q_behavior,
        "q_iql": q_iql,
        "q_improvement": q_iql - q_behavior,
        "behavior_dist": behavior_dist.tolist(),
        "iql_dist": iql_dist.tolist(),
    }


# =============================================================================
# IQL Training
# =============================================================================

def train_iql(
    pack_path: str,
    epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.7,
    beta: float = 3.0,
    lr: float = 3e-4,
    cql_alpha: float = 0.0,
    use_reward_shaping: bool = False,
    device: str = "cuda:0",
    save_path: Optional[str] = None,
):
    """Train IQL on offline data.

    Note: IQL only supports 27-dim base features. Offline data packs do not
    contain SeedTelemetry, so telemetry-enabled IQL models cannot be trained
    from offline data. Use PPO for telemetry-enabled policies.
    """
    import copy
    from esper.simic.iql import IQL

    print("=" * 60)
    print("Tamiyo Phase 3: Implicit Q-Learning")
    print("=" * 60)

    print(f"Loading {pack_path}...")
    with open(pack_path) as f:
        pack = json.load(f)
    print(f"Episodes: {pack['metadata']['num_episodes']}")

    print("Extracting transitions...")
    transitions = extract_transitions(
        pack,
        use_reward_shaping=use_reward_shaping,
        gamma=gamma,
    )
    print(f"Transitions: {len(transitions)}")

    buffer = ReplayBuffer(transitions, device=device)
    print(f"State dim: {buffer.state_dim}")

    rewards = buffer.rewards
    print(f"Reward range: [{rewards.min():.2f}, {rewards.max():.2f}], mean: {rewards.mean():.2f}")

    iql = IQL(
        state_dim=buffer.state_dim,
        action_dim=len(SimicAction),
        gamma=gamma,
        tau=tau,
        beta=beta,
        lr=lr,
        cql_alpha=cql_alpha,
        device=device,
    )

    algo_name = "IQL+CQL" if cql_alpha > 0 else "IQL"
    print(f"\nTraining {algo_name} for {epochs} epochs")

    best_q_improvement = -float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_cql_loss = 0.0

        for _ in range(steps_per_epoch):
            losses = iql.train_step(buffer, batch_size)
            epoch_v_loss += losses["v_loss"]
            epoch_q_loss += losses["q_loss"]
            epoch_cql_loss += losses["cql_loss"]

        epoch_v_loss /= steps_per_epoch
        epoch_q_loss /= steps_per_epoch
        epoch_cql_loss /= steps_per_epoch

        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = evaluate_iql_policy(iql, buffer)
            print(f"Epoch {epoch:3d}: v_loss={epoch_v_loss:.4f}, q_loss={epoch_q_loss:.4f} | "
                  f"agreement={metrics['agreement']*100:.1f}%, Q_improve={metrics['q_improvement']:.3f}")

            if metrics['q_improvement'] > best_q_improvement:
                best_q_improvement = metrics['q_improvement']
                best_state_dict = {
                    'q_network': copy.deepcopy(iql.q_network.state_dict()),
                    'v_network': copy.deepcopy(iql.v_network.state_dict()),
                }

    if best_state_dict:
        iql.q_network.load_state_dict(best_state_dict['q_network'])
        iql.v_network.load_state_dict(best_state_dict['v_network'])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_iql_policy(iql, buffer, n_samples=min(5000, buffer.size))
        torch.save({
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
            'state_dim': buffer.state_dim,
            'action_dim': len(SimicAction),
            'gamma': gamma,
            'tau': tau,
            'beta': beta,
            'metrics': metrics,
        }, save_path)
        print(f"Model saved to {save_path}")

    return iql


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
    use_telemetry: bool = True,
    collect_rollout: bool = True,
    deterministic: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with the PPO agent."""
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
    from esper.simic.ppo import signals_to_features

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    action_counts = {a.name: 0 for a in SimicAction}
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
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.GERMINATED:
            seed_state.transition(SeedStage.TRAINING)
            seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.TRAINING:
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.BLENDING:
            step = 0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                if seed_optimizer:
                    seed_optimizer.zero_grad()
                # Update blend alpha for this step
                if model.seed_slot:
                    model.seed_slot.update_alpha_for_step(step)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Collect gradient stats for telemetry (once per epoch, last batch)
                if use_telemetry:
                    grad_stats = collect_seed_gradients(model.get_seed_parameters())

                host_optimizer.step()
                if seed_optimizer:
                    seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                step += 1

        elif seed_state.stage == SeedStage.FOSSILIZED:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(testloader)
        val_acc = 100.0 * correct / total

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

        # Get features and action
        # Note: tracker would provide DiagnosticTracker telemetry if available
        features = signals_to_features(signals, model, tracker=None, use_telemetry=use_telemetry)
        state = torch.tensor([features], dtype=torch.float32, device=device)

        action_idx, log_prob, value = agent.get_action(state, deterministic=deterministic)
        action = SimicAction(action_idx)
        action_counts[action.name] += 1

        # Compute total params for rent (fossilized + active)
        total_params = params_added + model.active_seed_params
        reward = compute_shaped_reward(
            action=action.value,
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
        )

        # Execute action
        if SimicAction.is_germinate(action):
            if not model.has_active_seed:
                blueprint_id = SimicAction.get_blueprint_id(action)
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = None

        elif action == SimicAction.ADVANCE:
            if model.has_active_seed:
                if model.seed_state.stage == SeedStage.TRAINING:
                    model.seed_state.transition(SeedStage.BLENDING)
                    model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                elif model.seed_state.stage == SeedStage.BLENDING:
                    # Track params before fossilization
                    params_added += model.active_seed_params
                    model.seed_state.transition(SeedStage.FOSSILIZED)
                    model.seed_slot.set_alpha(1.0)

        elif action == SimicAction.CULL:
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
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str | None = None,
):
    """Train PPO agent."""
    from esper.simic.ppo import PPOAgent
    from esper.utils import load_cifar10

    print("=" * 60)
    print("PPO Training for Tamiyo")
    print("=" * 60)
    print(f"Episodes: {n_episodes}, Max epochs: {max_epochs}")
    print(f"Device: {device}, Telemetry: {use_telemetry}")

    trainloader, testloader = load_cifar10(batch_size=128)
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
        action_dim=len(SimicAction),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        entropy_coef_start=entropy_coef_start,
        entropy_coef_end=entropy_coef_end,
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


__all__ = [
    "extract_transitions",
    "evaluate_iql_policy",
    "train_iql",
    "run_ppo_episode",
    "train_ppo",
]
