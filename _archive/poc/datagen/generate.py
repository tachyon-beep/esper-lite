"""Main script for diverse data generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from esper.datagen.orchestrator import GenerationOrchestrator, GenerationPlan
from esper.datagen.architectures import create_model
from esper.datagen.policies import BehaviorPolicy
from esper.datagen.configs import RewardComponents, BehaviorPolicyConfig
from esper.datagen.health import DatasetHealthCheck
from esper.simic import (
    EpisodeCollector,
    TrainingSnapshot,
    ActionTaken,
    StepOutcome,
    SimicAction,
)
from esper.tamiyo import SignalTracker


# =============================================================================
# Data Loading
# =============================================================================

def load_cifar10(batch_size: int = 128):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# =============================================================================
# Training Utilities
# =============================================================================

def _train_epoch(model, trainloader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def _validate(model, trainloader, testloader, criterion, device):
    """Validate model and return metrics."""
    model.eval()

    # Validation
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(testloader)
    val_accuracy = 100.0 * val_correct / val_total

    # Training metrics (sample first few batches)
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 5:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

    train_loss /= min(5, len(trainloader))
    train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

    return val_loss, val_accuracy, train_loss, train_accuracy


def _snapshot_from_signals(signals, seed_state=None) -> TrainingSnapshot:
    """Create TrainingSnapshot from SignalTracker signals."""
    # Get last 5 losses/accuracies, padded with zeros
    loss_history = signals.loss_history[-5:] if signals.loss_history else []
    acc_history = signals.accuracy_history[-5:] if signals.accuracy_history else []
    while len(loss_history) < 5:
        loss_history.insert(0, 0.0)
    while len(acc_history) < 5:
        acc_history.insert(0, 0.0)

    snapshot = TrainingSnapshot(
        epoch=signals.epoch,
        global_step=signals.global_step,
        train_loss=signals.train_loss,
        val_loss=signals.val_loss,
        loss_delta=signals.loss_delta,
        train_accuracy=signals.train_accuracy,
        val_accuracy=signals.val_accuracy,
        accuracy_delta=signals.accuracy_delta,
        plateau_epochs=signals.plateau_epochs,
        best_val_accuracy=signals.best_val_accuracy,
        best_val_loss=min(signals.loss_history) if signals.loss_history else float('inf'),
        loss_history_5=tuple(loss_history),
        accuracy_history_5=tuple(acc_history),
        has_active_seed=seed_state is not None,
        available_slots=signals.available_slots,
    )

    if seed_state is not None:
        snapshot.seed_stage = seed_state.stage.value
        snapshot.seed_epochs_in_stage = seed_state.metrics.epochs_in_current_stage
        snapshot.seed_alpha = getattr(seed_state, 'alpha', 0.0)
        snapshot.seed_improvement = seed_state.metrics.improvement_since_stage_start

    return snapshot


# =============================================================================
# Episode Generation
# =============================================================================

def generate_episode(
    plan: GenerationPlan,
    trainloader,
    testloader,
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """Generate a single episode with full training.

    Integrates behavior policy decisions with real model training.

    Returns:
        Episode dict with full metadata including action probabilities
    """
    env_config = plan.get_env_config()
    policy_config = plan.get_policy_config()

    if verbose:
        print(f"  [{plan.episode_id}]", end=" ", flush=True)

    # Set random seed for reproducibility
    if plan.random_seed is not None:
        torch.manual_seed(plan.random_seed)

    # Create model using the architecture factory
    model = create_model(env_config.architecture, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Create optimizer based on config
    if env_config.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=env_config.learning_rate,
            weight_decay=env_config.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=env_config.learning_rate,
            momentum=env_config.momentum,
            weight_decay=env_config.weight_decay,
        )

    # Create behavior policy
    policy = BehaviorPolicy(policy_config)

    # Setup tracking
    tracker = SignalTracker()
    collector = EpisodeCollector()
    collector.start_episode(
        episode_id=plan.episode_id,
        max_epochs=env_config.max_epochs,
        initial_lr=env_config.learning_rate,
        model_name=env_config.architecture,
    )

    # Episode state
    prev_accuracy = 0.0
    best_accuracy = 0.0
    all_rewards = []
    action_counts = {"WAIT": 0, "GERMINATE": 0, "ADVANCE": 0, "CULL": 0}

    start_time = time.time()

    for epoch in range(1, env_config.max_epochs + 1):
        # Training step
        _train_epoch(model, trainloader, criterion, optimizer, device)

        # Validation
        val_loss, val_accuracy, train_loss, train_accuracy = _validate(
            model, trainloader, testloader, criterion, device
        )

        # Update best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

        # Update signal tracker
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            active_seeds=[],  # Simplified: no seed management in this version
            available_slots=1,
        )

        # Record observation
        snapshot = _snapshot_from_signals(signals, seed_state=None)
        collector.record_observation(snapshot)

        # Get policy decision with probabilities
        action_name, action_probs = policy.decide(signals)
        action_counts[action_name] += 1

        # Compute reward components
        accuracy_delta = val_accuracy - prev_accuracy

        reward_components = RewardComponents(
            accuracy_delta=accuracy_delta,
            loss_delta=0.0,  # Simplified
            potential_prev=prev_accuracy,
            potential_next=val_accuracy,
            intervention_cost=RewardComponents.INTERVENTION_COSTS.get(action_name, 0.0),
        )
        reward = reward_components.total(shaping_weight=0.0)  # Pure accuracy-based
        all_rewards.append(reward)

        # Record action with full probability information
        action = ActionTaken(
            action=SimicAction[action_name],
            confidence=action_probs.behavior_prob,
            reason=f"greedy={action_probs.greedy_action}, eps={action_probs.epsilon}",
        )
        collector.record_action(action)

        # Record outcome
        outcome = StepOutcome(
            accuracy_after=val_accuracy,
            accuracy_change=accuracy_delta,
            loss_after=val_loss,
            loss_change=0.0,
            reward=reward,
        )
        collector.record_outcome(outcome)

        prev_accuracy = val_accuracy

    # End episode
    episode = collector.end_episode(
        final_accuracy=val_accuracy,
        best_accuracy=best_accuracy,
        seeds_created=0,
        seeds_fossilized=0,
        seeds_culled=0,
    )

    # Convert to dict and add extended metadata
    episode_dict = episode.to_dict()
    episode_dict["schema_version"] = "2.0.0"
    episode_dict["behavior_policy"] = policy_config.to_dict()
    episode_dict["environment"] = env_config.to_dict()
    episode_dict["random_seed"] = plan.random_seed
    episode_dict["action_counts"] = action_counts
    episode_dict["total_return"] = sum(all_rewards)

    # Add action probabilities to each decision
    for i, decision in enumerate(episode_dict.get("decisions", [])):
        # Re-run policy to get probs (or store during collection)
        # For now, add placeholder - full impl would store during collection
        pass

    elapsed = time.time() - start_time
    if verbose:
        print(f"acc={val_accuracy:.1f}% ({elapsed:.1f}s)")

    return episode_dict


def generate_episode_skeleton(
    plan: GenerationPlan,
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """Generate a skeleton episode (for dry-run testing without training)."""
    env_config = plan.get_env_config()
    policy_config = plan.get_policy_config()

    if verbose:
        print(f"  [SKELETON] {plan.episode_id}")

    return {
        "episode_id": plan.episode_id,
        "schema_version": "2.0.0",
        "behavior_policy": policy_config.to_dict(),
        "environment": env_config.to_dict(),
        "random_seed": plan.random_seed,
        "decisions": [],
        "final_accuracy": 0.0,
        "best_accuracy": 0.0,
        "total_return": 0.0,
        "episode_length": 0,
        "termination_reason": "skeleton",
    }


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate diverse offline RL data")
    parser.add_argument("--output-dir", default="data/datagen_v3", help="Output directory")
    parser.add_argument("--episodes-per-combo", type=int, default=10, help="Episodes per combination")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without generating")
    parser.add_argument("--skeleton", action="store_true", help="Generate skeleton episodes (no training)")
    parser.add_argument("--health-check", action="store_true", help="Run health checks on existing data")
    parser.add_argument("--env-ids", nargs="+", help="Specific environment IDs to generate")
    parser.add_argument("--policy-ids", nargs="+", help="Specific policy IDs to generate")
    parser.add_argument("--max-epochs", type=int, default=25, help="Max epochs per episode")
    args = parser.parse_args()

    print("=" * 60)
    print("Diverse Data Generation System")
    print("=" * 60)

    # Create orchestrator
    orch = GenerationOrchestrator(
        output_dir=args.output_dir,
        episodes_per_combo=args.episodes_per_combo,
        env_ids=args.env_ids,
        policy_ids=args.policy_ids,
    )

    summary = orch.get_progress_summary()
    print(f"Total planned: {summary['total_planned']}")
    print(f"Completed: {summary['completed']}")
    print(f"Remaining: {summary['remaining']}")
    print()

    if args.dry_run:
        print("DRY RUN - showing first 20 planned episodes:")
        for plan in orch.plans[:20]:
            print(f"  {plan.episode_id}")
        if len(orch.plans) > 20:
            print(f"  ... and {len(orch.plans) - 20} more")
        return

    if args.health_check:
        print("Running health checks on existing data...")
        episodes = _load_existing_episodes(args.output_dir)
        if episodes:
            checker = DatasetHealthCheck()
            results = checker.run_all(episodes)
            checker.print_report(results)
        else:
            print("No episodes found to check")
        return

    # Load data once for all episodes
    if not args.skeleton:
        print("Loading CIFAR-10...")
        # Use batch size from first plan's env config
        batch_size = 128
        if orch.plans:
            batch_size = orch.plans[0].get_env_config().batch_size
        trainloader, testloader = load_cifar10(batch_size=batch_size)
        print(f"Data loaded (batch_size={batch_size})")
    else:
        trainloader, testloader = None, None

    # Generation loop
    print(f"\nStarting generation (batch_size={args.batch_size})...")
    start_time = time.time()
    episodes_generated = 0

    while True:
        batch = orch.get_next_batch(batch_size=args.batch_size)
        if not batch:
            break

        for plan in batch:
            try:
                if args.skeleton:
                    episode = generate_episode_skeleton(plan, device=args.device, verbose=True)
                else:
                    episode = generate_episode(
                        plan,
                        trainloader,
                        testloader,
                        device=args.device,
                        verbose=True,
                    )

                # Save episode
                episode_path = Path(args.output_dir) / f"{plan.episode_id}.json"
                with open(episode_path, "w") as f:
                    json.dump(episode, f, indent=2)

                orch.mark_complete(plan.episode_id)
                episodes_generated += 1

            except Exception as e:
                print(f"\n  ERROR generating {plan.episode_id}: {e}")
                continue

        # Progress update
        summary = orch.get_progress_summary()
        elapsed = time.time() - start_time
        eps_per_sec = episodes_generated / elapsed if elapsed > 0 else 0
        print(f"\nProgress: {summary['completed']}/{summary['total_planned']} "
              f"({summary['progress_pct']:.1f}%) - {elapsed:.0f}s elapsed ({eps_per_sec:.2f} ep/s)")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Episodes generated: {episodes_generated}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 60)

    # Run health check on generated data
    if episodes_generated > 0:
        print("\nRunning health checks on generated data...")
        episodes = _load_existing_episodes(args.output_dir)
        checker = DatasetHealthCheck()
        results = checker.run_all(episodes)
        checker.print_report(results)


def _load_existing_episodes(output_dir: str) -> list[dict]:
    """Load existing episodes from output directory."""
    episodes = []
    for path in Path(output_dir).glob("*.json"):
        if path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                episodes.append(json.load(f))
        except json.JSONDecodeError:
            continue
    return episodes


if __name__ == "__main__":
    main()
