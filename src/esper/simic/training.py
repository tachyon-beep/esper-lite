"""Training loops for PPO.

This module contains the main training functions extracted from ppo.py.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from esper.leyline.factored_actions import FactoredAction, LifecycleOp
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.runtime import get_task_spec
from esper.simic.rewards import compute_contribution_reward, SeedInfo
from esper.simic.gradient_collector import (
    collect_seed_gradients_async,
    materialize_grad_stats,
)
from esper.simic.action_masks import build_slot_states, compute_action_masks
from esper.simic.slots import ordered_slots
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
# Heuristic Training
# =============================================================================

def _convert_flat_to_factored(action, topology: str = "cnn") -> FactoredAction:
    """Convert flat action enum to FactoredAction for heuristic path.

    Maps flat action names to factored action components.
    """
    from esper.leyline.factored_actions import BlueprintAction

    action_name = action.name

    if action_name.startswith("GERMINATE_"):
        # Extract blueprint from action name like "GERMINATE_CONV_LIGHT"
        blueprint_name_upper = action_name.replace("GERMINATE_", "")
        # Look up BlueprintAction by name, default to NOOP for unknown blueprints
        try:
            blueprint = BlueprintAction[blueprint_name_upper]
        except KeyError:
            blueprint = BlueprintAction.NOOP
        return FactoredAction.from_indices(
            slot_idx=0,  # Default to first slot
            blueprint_idx=blueprint.value,
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

    episode_id = f"heur_{base_seed}"
    model = create_model(task=task_spec, device=device, slots=slots)

    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"slots contains duplicates: {slots}")
    enabled_slots = list(ordered_slots(slots))

    # Wire telemetry
    hub = get_hub()

    def telemetry_callback(event):
        event.data.setdefault("env_id", 0)
        hub.emit(event)

    for slot_id in enabled_slots:
        slot = model.seed_slots[slot_id]
        slot.on_telemetry = telemetry_callback
        slot.fast_mode = False
        slot.isolate_gradients = True

    # Emit TRAINING_STARTED to activate Karn (P1 fix)
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={
            "episode_id": episode_id,
            "seed": base_seed,
            "max_epochs": max_epochs,
            "task": task_spec.name,
            "mode": "heuristic",
        },
    ))

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

    for epoch in range(1, max_epochs + 1):
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

        # Gather active seeds across ALL enabled slots (multi-slot support).
        # Assert seed_id uniqueness - duplicate IDs would make target resolution ambiguous.
        active_seeds = []
        seed_ids: set[str] = set()
        for slot_id in enabled_slots:
            if not model.has_active_seed_in_slot(slot_id):
                continue
            state = model.seed_slots[slot_id].state
            if state is None:
                continue
            if state.seed_id in seed_ids:
                raise RuntimeError(f"Duplicate seed_id '{state.seed_id}' across slots in one env")
            seed_ids.add(state.seed_id)
            active_seeds.append(state)

        # Record accuracy in seed metrics (per-slot counters + deltas).
        for seed_state in active_seeds:
            if seed_state.metrics:
                seed_state.metrics.record_accuracy(val_acc)

        # Update signal tracker
        available_slots = sum(
            1 for slot_id in enabled_slots if not model.has_active_seed_in_slot(slot_id)
        )
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
        for slot_id in enabled_slots:
            model.seed_slots[slot_id].step_epoch()

        def resolve_slot_for_seed_id(seed_id: str) -> str:
            slot_matches = [
                slot_id for slot_id in enabled_slots
                if model.seed_slots[slot_id].state is not None
                and model.seed_slots[slot_id].state.seed_id == seed_id
            ]
            if len(slot_matches) != 1:
                raise RuntimeError(
                    f"target_seed_id '{seed_id}' expected in exactly 1 slot, found {slot_matches}"
                )
            return slot_matches[0]

        # Get heuristic decision and convert to factored action
        decision = policy.decide(signals, active_seeds)
        flat_action = decision.action
        factored_action = _convert_flat_to_factored(flat_action, task_spec.topology)
        action_counts[factored_action.op.name] += 1

        # Compute reward (for comparison with PPO)
        total_params = model.active_seed_params
        reward_seed_state = None
        reward_seed_params = 0
        if decision.target_seed_id:
            target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
            reward_seed_state = model.seed_slots[target_slot].state
            reward_seed_params = model.seed_slots[target_slot].active_seed_params
        reward = compute_contribution_reward(
            action=factored_action.op,  # Pass the LifecycleOp enum
            seed_contribution=None,  # No counterfactual in heuristic path
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(
                reward_seed_state,
                reward_seed_params,
            ),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_delta=acc_delta,  # Used as proxy signal
        )
        episode_rewards.append(reward)

        germinate_slot = next(
            (slot_id for slot_id in enabled_slots if not model.has_active_seed_in_slot(slot_id)),
            None,
        )

        # Execute action using FactoredAction properties
        if factored_action.is_germinate:
            if germinate_slot is not None:
                blueprint_id = factored_action.blueprint_id
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id, slot=germinate_slot)
                seeds_created += 1
                seed_optimizer = torch.optim.SGD(
                    model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
                )

        elif factored_action.is_fossilize:
            if decision.target_seed_id:
                target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
                slot_state = model.seed_slots[target_slot].state
                if slot_state is not None and slot_state.stage == SeedStage.PROBATIONARY:
                    slot = model.seed_slots[target_slot]
                    gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
                    if gate_result.passed:
                        slot.set_alpha(1.0)

        elif factored_action.is_cull:
            if decision.target_seed_id:
                target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
                model.cull_seed(slot=target_slot)
                seed_optimizer = (
                    torch.optim.SGD(model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9)
                    if model.has_active_seed else None
                )

        summary_seed_id = signals.active_seeds[0] if signals.active_seeds else None
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=epoch,
            seed_id=decision.target_seed_id,
            data={
                "env_id": 0,
                "episode_id": episode_id,
                "mode": "heuristic",
                "task": task_spec.name,
                "device": device,
                "enabled_slots": enabled_slots,
                "max_epochs": max_epochs,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "available_slots": available_slots,
                "seeds_active": len(active_seeds),
                "summary_seed_id": summary_seed_id,
                "target_seed_id": decision.target_seed_id,
                "action": decision.action.name,
                "op": factored_action.op.name,
                "reward": reward,
            },
        ))

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

    hub = get_hub()
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        data={
            "env_id": 0,
            "mode": "heuristic",
            "task": task_spec.name,
            "topology": task_spec.topology,
            "episodes": n_episodes,
            "max_epochs": max_epochs,
            "max_batches": max_batches,
            "device": device,
            "slots": slots,
        },
        message="Heuristic training run configuration",
    ))

    trainloader, testloader = task_spec.create_dataloaders()

    policy = HeuristicTamiyo(topology=task_spec.topology)
    history = []

    for ep in range(1, n_episodes + 1):
        policy.reset()
        base_seed = seed + ep * 1000
        episode_id = f"heur_{base_seed}"

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
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data={
                "env_id": 0,
                "mode": "heuristic",
                "task": task_spec.name,
                "episode_id": episode_id,
                "episode": ep,
                "episodes_total": n_episodes,
                "base_seed": base_seed,
                "final_accuracy": final_acc,
                "total_reward": total_reward,
                "action_counts": dict(action_counts),
            },
            message="Heuristic episode completed",
        ))

        history.append({
            'episode': ep,
            'accuracy': final_acc,
            'total_reward': total_reward,
            'action_counts': dict(action_counts),
        })

    return history


__all__ = [
    "run_heuristic_episode",
    "train_heuristic",
]
