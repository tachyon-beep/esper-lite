"""Training loops for PPO.

This module contains the main training functions extracted from ppo.py.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from esper.leyline import SeedTelemetry
from esper.leyline.factored_actions import FactoredAction, LifecycleOp
from esper.runtime import get_task_spec
from esper.simic.rewards import compute_contribution_reward, SeedInfo
from esper.simic.gradient_collector import (
    collect_seed_gradients_async,
    materialize_grad_stats,
)
from esper.simic.action_masks import build_slot_states, compute_action_masks
from esper.nissa import get_hub
from esper.utils.loss import compute_task_loss_with_metrics


# =============================================================================
# Compiled Training Step
# =============================================================================

# Flag to enable/disable torch.compile (set to False if compilation causes issues)
USE_COMPILED_TRAIN_STEP = True


def _train_step_impl(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner training step - forward pass and loss computation.

    This is the compilable core that can be optimized with torch.compile.
    Control flow (optimizer steps, seed handling) stays OUTSIDE this function.

    Args:
        model: The model to train
        inputs: Input batch tensor
        targets: Target batch tensor
        criterion: Loss function (CrossEntropyLoss)

    Returns:
        Tuple of (loss tensor, output logits)
    """
    outputs = model(inputs)
    # Reshape for criterion if needed (handles both LM and classification)
    if outputs.dim() == 3:  # LM: (batch, seq, vocab)
        vocab = outputs.size(-1)
        loss = criterion(outputs.view(-1, vocab), targets.view(-1))
    else:  # Classification: (batch, classes)
        loss = criterion(outputs, targets)
    return loss, outputs


# Compile the training step for reduced overhead with CUDA graphs
# mode="reduce-overhead" uses CUDA graphs for repeated calls with same shapes
try:
    _compiled_train_step = torch.compile(_train_step_impl, mode="reduce-overhead")
except Exception:
    # Fallback if compilation fails (e.g., older PyTorch version)
    _compiled_train_step = _train_step_impl
    USE_COMPILED_TRAIN_STEP = False


def compiled_train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training step with optional torch.compile optimization.

    Uses compiled version if available and enabled, otherwise falls back
    to regular implementation.
    """
    if USE_COMPILED_TRAIN_STEP:
        return _compiled_train_step(model, inputs, targets, criterion)
    return _train_step_impl(model, inputs, targets, criterion)


# =============================================================================
# PPO helpers
# =============================================================================


def _train_one_epoch(
    model: nn.Module,
    trainloader: "torch.utils.data.DataLoader",
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str,
    collect_gradients: bool = False,
) -> tuple[float, float, int, dict | None]:
    """Unified training loop for all seed stages.

    This function extracts the repeated inline loop pattern. Callers use
    returned values to compute metrics:
        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

    Args:
        model: The model to train
        trainloader: Training data loader
        criterion: Loss function
        host_optimizer: Optimizer for host parameters
        seed_optimizer: Optimizer for seed parameters (optional)
        device: Device to train on
        task_type: "classification" or "lm"
        collect_gradients: If True, collect gradient stats for telemetry

    Returns:
        Tuple of (running_loss, correct_count, total_count, grad_stats)
        - running_loss: Sum of loss values across batches (float)
        - correct_count: Sum of correct predictions (float/int)
        - total_count: Total samples processed (int)
        - grad_stats: Gradient statistics dict if collect_gradients=True, else None

    Note:
        Uses tensor accumulation internally with a single .item() sync at epoch end
        to avoid CUDA synchronization overhead in the hot path.
    """
    model.train()

    # Pre-allocate accumulators on device to avoid .item() sync per batch
    # This is the key optimization: accumulate as tensors, sync once at epoch end
    running_loss = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device, dtype=torch.long)
    total = 0
    grad_stats = None

    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)

        loss, outputs = compiled_train_step(model, inputs, targets, criterion)
        # Compute metrics from outputs (compiled_train_step already computed loss)
        if task_type == "classification":
            _, predicted = outputs.max(1)
            correct_batch = predicted.eq(targets).sum()
            batch_total = targets.size(0)
        else:  # LM task
            correct_batch = torch.tensor(0, device=outputs.device)
            batch_total = targets.numel()
        loss.backward()

        # Collect gradient stats as tensors (async-safe, no .item() sync)
        # Overwrites each batch; final value materialized after loop
        if collect_gradients:
            grad_stats = collect_seed_gradients_async(model.get_seed_parameters())

        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()

        # Accumulate on device - no .item() sync in hot path
        running_loss.add_(loss.detach())
        running_correct.add_(correct_batch)
        total += batch_total

    # Single sync at epoch end (forces all CUDA ops to complete)
    epoch_loss = running_loss.item()
    epoch_correct = running_correct.item()

    # Now safe to materialize gradient tensors (after implicit sync above)
    if grad_stats is not None and not grad_stats.get('_empty', False):
        grad_stats = materialize_grad_stats(grad_stats)

    return epoch_loss, epoch_correct, total, grad_stats


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
    slots: list[str] | None = None,
    max_seeds: int = 0,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with the PPO agent."""
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
    from esper.simic.ppo import signals_to_features

    if task_spec is None:
        task_spec = get_task_spec("cifar10")
    task_type = task_spec.task_type

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(task=task_spec, device=device, slots=slots)

    # Determine target slot from slots parameter
    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")
    target_slot = slots[0]

    # Wire Kasmina telemetry into global Nissa hub so fossilization and
    # lifecycle events propagate to configured backends (console, analytics).
    hub = get_hub()

    def telemetry_callback(event):
        # Single-env PPO uses env_id=0 for analytics compatibility.
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    slot = model.seed_slots[target_slot]
    slot.on_telemetry = telemetry_callback
    slot.fast_mode = False
    # Incubator mode gradient isolation: detach host input into the seed path so host
    # gradients match the baseline model while the seed trickle-learns via STE.
    # Host optimizer still steps every batch; isolation only affects gradients
    # flowing through the seed branch.
    slot.isolate_gradients = True

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    seed_created_epoch = 0  # Track when current seed was created for age computation
    # Track ops by LifecycleOp enum value
    action_counts = {op.name: 0 for op in LifecycleOp}
    episode_rewards = []

    # Track host params and added params for compute rent
    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    params_added = 0  # Accumulates when seeds are fossilized

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_slots[target_slot].state if model.has_active_seed else None

        # Handle GERMINATED→TRAINING transition
        if seed_state and seed_state.stage == SeedStage.GERMINATED:
            gate_result = model.seed_slots[target_slot].advance_stage(SeedStage.TRAINING)
            if not gate_result.passed:
                raise RuntimeError(f"G1 gate failed during TRAINING entry: {gate_result}")

        # Initialize seed optimizer if needed (active seed, not fossilized)
        needs_seed_optimizer = (
            seed_state is not None
            and seed_state.stage not in (SeedStage.DORMANT, SeedStage.FOSSILIZED)
        )
        if needs_seed_optimizer and seed_optimizer is None:
            seed_optimizer = torch.optim.SGD(
                model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
            )

        # Determine if we should collect gradients (active training seed)
        should_collect_gradients = (
            use_telemetry
            and seed_state is not None
            and seed_state.stage in (
                SeedStage.TRAINING, SeedStage.BLENDING,
                SeedStage.PROBATIONARY
            )
        )

        # Single unified training call
        running_loss, correct, total, grad_stats = _train_one_epoch(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            host_optimizer=host_optimizer,
            seed_optimizer=seed_optimizer if needs_seed_optimizer else None,
            device=device,
            task_type=task_type,
            collect_gradients=should_collect_gradients,
        )

        train_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Validate - use tensor accumulation for deferred sync
        model.eval()
        val_loss_accum = torch.zeros(1, device=device)
        val_correct_accum = torch.zeros(1, device=device, dtype=torch.long)
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
                val_loss_accum.add_(loss)
                val_correct_accum.add_(correct_batch)
                total += batch_total

        # Single sync at end of validation
        val_loss = val_loss_accum.item() / len(testloader) if len(testloader) > 0 else 0.0
        val_acc = 100.0 * val_correct_accum.item() / total if total > 0 else 0.0

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

        # Get features and action BEFORE step_epoch() to maintain state/action alignment
        # The RL transition (s, a, r, s') must use consistent state for observation and reward
        # Note: tracker would provide DiagnosticTracker telemetry if available
        features = signals_to_features(
            signals,
            model,
            use_telemetry=use_telemetry,
            slots=slots,
            total_seeds=model.count_active_seeds() if model else 0,
            max_seeds=max_seeds,
        )
        state = torch.tensor([features], dtype=torch.float32, device=device)

        # Compute action mask for valid actions (physical constraints only)
        slot_states = build_slot_states(model, [target_slot])
        masks = compute_action_masks(
            slot_states=slot_states,
            target_slot=target_slot,
            total_seeds=model.count_active_seeds() if model else 0,
            max_seeds=max_seeds,
            device=torch.device(device),
        )

        # Get factored action from agent
        action_dict, log_prob, value = agent.network.get_action_batch(
            state,
            {k: v.unsqueeze(0) for k, v in masks.items()},  # Add batch dim
            deterministic=deterministic,
        )
        # Extract single action from batch
        factored_action = FactoredAction.from_indices(
            slot_idx=action_dict["slot"][0].item(),
            blueprint_idx=action_dict["blueprint"][0].item(),
            blend_idx=action_dict["blend"][0].item(),
            op_idx=action_dict["op"][0].item(),
        )
        log_prob = log_prob[0].item()
        value = value[0].item()
        action_counts[factored_action.op.name] += 1

        # Compute total params for rent (fossilized + active)
        total_params = params_added + model.active_seed_params
        reward = compute_contribution_reward(
            action=factored_action.op,  # Pass the LifecycleOp enum
            seed_contribution=None,  # No counterfactual in non-vectorized path
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_delta=acc_delta,  # Used as proxy signal
        )

        # Execute action using FactoredAction properties
        if factored_action.is_germinate:
            if not model.has_active_seed:
                blueprint_id = factored_action.blueprint_id
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id, slot=target_slot)
                seeds_created += 1
                seed_created_epoch = epoch  # Track when seed was created for age computation
                seed_optimizer = None

        elif factored_action.is_fossilize:
            # NOTE: Only PROBATIONARY → FOSSILIZED is a valid lifecycle transition.
            if model.has_active_seed and model.seed_slots[target_slot].state.stage == SeedStage.PROBATIONARY:
                slot = model.seed_slots[target_slot]
                gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
                if gate_result.passed:
                    params_added += model.active_seed_params
                    slot.set_alpha(1.0)

        elif factored_action.is_cull:
            if model.has_active_seed:
                model.cull_seed(slot=target_slot)
                seed_optimizer = None

        done = (epoch == max_epochs)
        truncated = done  # All episodes end at max_epochs (time limit truncation)
        bootstrap_value = value if truncated else 0.0  # Bootstrap from V(s_final) for truncation

        if collect_rollout:
            # Build action dict for factored storage
            action_dict_for_storage = {
                "slot": action_dict["slot"][0].item(),
                "blueprint": action_dict["blueprint"][0].item(),
                "blend": action_dict["blend"][0].item(),
                "op": action_dict["op"][0].item(),
            }
            agent.store_factored_transition(
                state=state.squeeze(0).cpu(),
                action=action_dict_for_storage,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
                action_masks={k: v.cpu() for k, v in masks.items()},
                truncated=truncated,
                bootstrap_value=bootstrap_value,
            )

        episode_rewards.append(reward)

        # Mechanical lifecycle advance (blending/shadowing dwell) AFTER RL transition
        # This ensures state/action/reward alignment - advance happens after the step is recorded
        # Must check seed_state exists since actions may have culled it
        if model.has_active_seed:
            model.seed_slots[target_slot].step_epoch()

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
    entropy_coef: float = 0.05,  # Unified default
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.01,  # Unified minimum
    adaptive_entropy_floor: bool = False,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str | None = None,
    seed: int | None = None,
    telemetry_config: "TelemetryConfig | None" = None,
    slots: list[str] | None = None,
    max_seeds: int | None = None,
    max_seeds_per_slot: int | None = None,
):
    """Train PPO agent."""
    from esper.simic.ppo import PPOAgent
    from esper.utils import load_cifar10
    from esper.simic.features import MULTISLOT_FEATURE_SIZE

    task_spec = get_task_spec(task)

    print("=" * 60)
    print("PPO Training for Tamiyo")
    print("=" * 60)
    print(f"Task: {task_spec.name} (topology={task_spec.topology}, type={task_spec.task_type})")
    print(f"Episodes: {n_episodes}, Max epochs: {max_epochs}")
    print(f"Device: {device}, Telemetry: {use_telemetry}")

    trainloader, testloader = task_spec.create_dataloaders()
    # State dimension uses MULTISLOT_FEATURE_SIZE for factored actions
    state_dim = MULTISLOT_FEATURE_SIZE

    # Convert episode-based annealing to step-based
    # CRITICAL: Non-vectorized training only updates every `update_every` episodes
    # So actual PPO updates = n_episodes / update_every
    # If update_every=5 and entropy_anneal_episodes=100, we get 20 PPO updates
    entropy_anneal_steps = (entropy_anneal_episodes // update_every) if entropy_anneal_episodes > 0 else 0

    agent = PPOAgent(
        state_dim=state_dim,
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
        factored=True,  # Use factored action space
    )

    # Compute effective seed limit
    # max_seeds=None means unlimited (use 0 to indicate no limit)
    effective_max_seeds = max_seeds if max_seeds is not None else 0

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
            slots=slots,
            max_seeds=effective_max_seeds,
        )

        total_reward = sum(rewards)
        recent_accuracies.append(final_acc)
        recent_rewards.append(total_reward)

        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        if ep % update_every == 0 or ep == n_episodes:
            metrics = agent.update_factored(last_value=0.0)

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
                # Store on CPU to save GPU memory (checkpoint is rarely loaded)
                best_state = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}

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

def _convert_flat_to_factored(action, topology: str = "cnn") -> FactoredAction:
    """Convert flat action enum to FactoredAction for heuristic path.

    Maps flat action names to factored action components.
    """
    from esper.leyline.factored_actions import (
        LifecycleOp,
        BLUEPRINT_IDS,
        BLEND_IDS,
        SLOT_IDS,
    )

    action_name = action.name

    if action_name.startswith("GERMINATE_"):
        # Extract blueprint from action name like "GERMINATE_BOTTLENECK"
        blueprint_name = action_name.replace("GERMINATE_", "").lower()
        blueprint_idx = BLUEPRINT_IDS.index(blueprint_name) if blueprint_name in BLUEPRINT_IDS else 0
        return FactoredAction.from_indices(
            slot_idx=0,  # Default to first slot
            blueprint_idx=blueprint_idx,
            blend_idx=0,  # Default blend
            op_idx=LifecycleOp.GERMINATE,
        )
    elif action_name == "FOSSILIZE":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            blend_idx=0,
            op_idx=LifecycleOp.FOSSILIZE,
        )
    elif action_name == "CULL":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            blend_idx=0,
            op_idx=LifecycleOp.CULL,
        )
    else:  # WAIT or unknown
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            blend_idx=0,
            op_idx=LifecycleOp.WAIT,
        )


def run_heuristic_episode(
    policy,
    trainloader,
    testloader,
    max_epochs: int = 75,
    max_batches: int | None = None,
    base_seed: int = 42,
    device: str = "cuda:0",
    task_spec=None,
    slots: list[str] | None = None,
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
    task_type = task_spec.task_type

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(task=task_spec, device=device, slots=slots)

    # Determine target slot from slots parameter
    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")
    target_slot = slots[0]

    # Wire telemetry
    hub = get_hub()
    def telemetry_callback(event):
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    slot = model.seed_slots[target_slot]
    slot.on_telemetry = telemetry_callback
    slot.fast_mode = False
    slot.isolate_gradients = True

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    # Track ops by LifecycleOp enum value
    action_counts = {op.name: 0 for op in LifecycleOp}
    episode_rewards = []

    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    params_added = 0

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_slots[target_slot].state if model.has_active_seed else None

        # Training phase - use tensor accumulation for deferred sync
        model.train()
        running_loss = torch.zeros(1, device=device)
        running_correct = torch.zeros(1, device=device, dtype=torch.long)
        total = 0
        batch_count = 0

        for inputs, targets in trainloader:
            if max_batches and batch_count >= max_batches:
                break
            batch_count += 1

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            host_optimizer.zero_grad(set_to_none=True)
            if seed_optimizer:
                seed_optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
            loss.backward()

            host_optimizer.step()
            if seed_optimizer:
                seed_optimizer.step()

            # Accumulate on device - no .item() sync in hot path
            running_loss.add_(loss.detach())
            running_correct.add_(correct_batch)
            total += batch_total

        # Single sync at end of training
        train_loss = running_loss.item() / max(1, batch_count)
        train_acc = 100.0 * running_correct.item() / total if total > 0 else 0.0

        # Validation - use tensor accumulation for deferred sync
        model.eval()
        val_loss_accum = torch.zeros(1, device=device)
        val_correct_accum = torch.zeros(1, device=device, dtype=torch.long)
        total = 0
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                if max_batches and batch_count >= max_batches:
                    break
                batch_count += 1

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
                val_loss_accum.add_(loss)
                val_correct_accum.add_(correct_batch)
                total += batch_total

        # Single sync at end of validation
        val_loss = val_loss_accum.item() / max(1, batch_count)
        val_acc = 100.0 * val_correct_accum.item() / total if total > 0 else 0.0

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
        if model.has_active_seed:
            model.seed_slots[target_slot].step_epoch()
        seed_state = model.seed_slots[target_slot].state if model.has_active_seed else None

        # Get heuristic decision and convert to factored action
        decision = policy.decide(signals, active_seeds)
        flat_action = decision.action
        factored_action = _convert_flat_to_factored(flat_action, task_spec.topology)
        action_counts[factored_action.op.name] += 1

        # Compute reward (for comparison with PPO)
        total_params = params_added + model.active_seed_params
        reward = compute_contribution_reward(
            action=factored_action.op,  # Pass the LifecycleOp enum
            seed_contribution=None,  # No counterfactual in heuristic path
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_delta=acc_delta,  # Used as proxy signal
        )
        episode_rewards.append(reward)

        # Execute action using FactoredAction properties
        if factored_action.is_germinate:
            if not model.has_active_seed:
                blueprint_id = factored_action.blueprint_id
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id, slot=target_slot)
                seeds_created += 1
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
                )

        elif factored_action.is_fossilize:
            if model.has_active_seed and model.seed_slots[target_slot].state.stage == SeedStage.PROBATIONARY:
                slot = model.seed_slots[target_slot]
                gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
                if gate_result.passed:
                    params_added += model.active_seed_params
                    slot.set_alpha(1.0)

        elif factored_action.is_cull:
            if model.has_active_seed:
                model.cull_seed(slot=target_slot)
                seed_optimizer = None

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs:
            seed_info = f"seed={seed_state.stage.name}" if seed_state else "no seed"
            print(f"  Epoch {epoch:3d}/{max_epochs}: acc={val_acc:.1f}%, {seed_info}, action={factored_action.op.name}")

    return val_acc, action_counts, episode_rewards


def train_heuristic(
    n_episodes: int = 1,
    max_epochs: int = 75,
    max_batches: int | None = 50,
    device: str = "cuda:0",
    task: str = "cifar10",
    seed: int = 42,
    slots: list[str] | None = None,
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
            slots=slots,
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
