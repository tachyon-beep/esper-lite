"""Overnight runner for Policy Tamiyo training.

This script:
1. Generates N episodes using Heuristic Tamiyo
2. Trains a PolicyNetwork on collected data
3. Evaluates with accuracy + confusion matrix
4. Runs live comparison between heuristic and learned policy

Usage:
    # Generate 50 episodes and train
    PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --episodes 50

    # Just train on existing data
    PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --train-only

    # Run live comparison
    PYTHONPATH=src .venv/bin/python src/esper/simic_overnight.py --compare
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from esper.tamiyo import (
    TrainingSignals, SignalTracker, HeuristicTamiyo,
    HeuristicPolicyConfig, TamiyoDecision, TamiyoAction,
)
from esper.kasmina import SeedStage, SeedSlot, BlueprintCatalog
from esper.simic import (
    EpisodeCollector, DatasetManager, StepOutcome,
    snapshot_from_signals, action_from_decision,
    PolicyNetwork, print_confusion_matrix, SimicAction,
)
from esper.poc_tamiyo import ConvBlock, HostCNN, MorphogeneticModel


# =============================================================================
# Model Setup (uses MorphogeneticModel from poc_tamiyo)
# =============================================================================

def create_model(device: str = "cuda") -> MorphogeneticModel:
    """Create a MorphogeneticModel with HostCNN."""
    host = HostCNN(num_classes=10)
    model = MorphogeneticModel(host, device=device)
    return model.to(device)


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
# Episode Generation
# =============================================================================

def generate_episode(
    episode_id: str,
    trainloader,
    testloader,
    max_epochs: int = 25,
    device: str = "cuda",
    verbose: bool = False,
) -> None:
    """Generate one episode using Heuristic Tamiyo with real seed execution."""

    # Setup model with seed infrastructure
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizers - host optimizer always active, seed optimizer when needed
    host_optimizer = optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)
    seed_optimizer = None

    # Setup Tamiyo
    config = HeuristicPolicyConfig(
        min_epochs_before_germinate=5,
        plateau_epochs_to_germinate=3,
        min_training_epochs=3,
        cull_after_epochs_without_improvement=5,
    )
    tamiyo = HeuristicTamiyo(config)
    tracker = SignalTracker()

    # Setup collector
    collector = EpisodeCollector()
    collector.start_episode(
        episode_id=episode_id,
        max_epochs=max_epochs,
        initial_lr=0.01,
    )

    # Episode stats
    prev_accuracy = 0.0
    seeds_created = 0
    seeds_fossilized = 0
    seeds_culled = 0

    for epoch in range(1, max_epochs + 1):
        # Determine training mode based on seed state
        seed_state = model.seed_state

        if seed_state is None:
            # No seed - normal training
            _train_epoch_normal(model, trainloader, criterion, host_optimizer, device)
        elif seed_state.stage == SeedStage.GERMINATED:
            # Just germinated - auto-advance to TRAINING
            seed_state.transition(SeedStage.TRAINING)
            seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            _train_epoch_seed_isolated(model, trainloader, criterion, seed_optimizer, device)
            # Note: baseline set below after record_accuracy to avoid overwrite
        elif seed_state.stage == SeedStage.TRAINING:
            # Seed in isolated training
            if seed_optimizer is None:
                seed_optimizer = optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            _train_epoch_seed_isolated(model, trainloader, criterion, seed_optimizer, device)
        elif seed_state.stage in (SeedStage.BLENDING, SeedStage.FOSSILIZED):
            # Blending or fossilized - joint training
            if seed_state.stage == SeedStage.BLENDING:
                # Update alpha during blending
                step = seed_state.metrics.epochs_in_current_stage
                model.seed_slot.update_alpha_for_step(step)
            _train_epoch_blended(model, trainloader, criterion, host_optimizer, seed_optimizer, device)
        else:
            # Fallback
            _train_epoch_normal(model, trainloader, criterion, host_optimizer, device)

        # Validation phase
        model.eval()
        val_loss, val_accuracy, train_loss, train_accuracy = _validate_and_get_metrics(
            model, trainloader, testloader, criterion, device
        )

        # Update seed metrics if active
        if model.has_active_seed:
            # Check if first epoch in TRAINING (to set proper baseline)
            first_training_epoch = (model.seed_state.metrics.epochs_total == 0 and
                                    model.seed_state.stage == SeedStage.TRAINING)
            model.seed_state.metrics.record_accuracy(val_accuracy)
            # Override baseline to pre-germination accuracy for first TRAINING epoch
            if first_training_epoch:
                model.seed_state.metrics.accuracy_at_stage_start = prev_accuracy

        # Build active seeds list for signal tracker
        active_seeds = []
        if model.has_active_seed:
            active_seeds = [model.seed_state]

        # Update signal tracker
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            active_seeds=active_seeds,
            available_slots=0 if model.has_active_seed else 1,
        )

        # Record observation with seed state
        snapshot = snapshot_from_signals(signals, seed_state=model.seed_state)
        collector.record_observation(snapshot)

        # Get Tamiyo's decision
        decision = tamiyo.decide(signals)
        action = action_from_decision(decision)
        collector.record_action(action)

        # Execute decision
        if decision.action == TamiyoAction.GERMINATE:
            if not model.has_active_seed:
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(decision.blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = None
                if verbose:
                    print(f"    [GERMINATE] {seed_id}")

        elif decision.action == TamiyoAction.ADVANCE_BLENDING:
            if model.has_active_seed and model.seed_state.stage == SeedStage.TRAINING:
                model.seed_state.transition(SeedStage.BLENDING)
                model.seed_state.metrics.reset_stage_baseline()
                model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                if verbose:
                    print(f"    [ADVANCE] -> BLENDING")

        elif decision.action == TamiyoAction.ADVANCE_FOSSILIZE:
            if model.has_active_seed and model.seed_state.stage == SeedStage.BLENDING:
                model.seed_state.transition(SeedStage.FOSSILIZED)
                model.seed_state.metrics.reset_stage_baseline()
                model.seed_slot.set_alpha(1.0)
                seeds_fossilized += 1
                if verbose:
                    print(f"    [ADVANCE] -> FOSSILIZED")

        elif decision.action == TamiyoAction.CULL:
            if model.has_active_seed:
                model.cull_seed()
                seed_optimizer = None
                seeds_culled += 1
                if verbose:
                    print(f"    [CULL]")

        # Record outcome
        accuracy_change = val_accuracy - prev_accuracy
        outcome = StepOutcome(
            accuracy_after=val_accuracy,
            accuracy_change=accuracy_change,
            loss_after=val_loss,
            loss_change=prev_accuracy - val_loss if epoch > 1 else 0.0,
        )
        collector.record_outcome(outcome)

        prev_accuracy = val_accuracy

        if verbose:
            stage_str = model.seed_state.stage.name if model.has_active_seed else "none"
            print(f"  Epoch {epoch:2d}: Val={val_accuracy:.2f}% | "
                  f"stage={stage_str} | action={decision.action.name}")

    # End episode
    episode = collector.end_episode(
        final_accuracy=val_accuracy,
        best_accuracy=tracker._best_accuracy,
        seeds_created=seeds_created,
        seeds_fossilized=seeds_fossilized,
        seeds_culled=seeds_culled,
    )

    return episode


def _train_epoch_normal(model, trainloader, criterion, optimizer, device):
    """Train one epoch without seed."""
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def _train_epoch_seed_isolated(model, trainloader, criterion, seed_optimizer, device):
    """Train one epoch with seed in isolation (only seed weights updated).

    Note: We don't freeze host params because that breaks gradient flow.
    Instead, we just don't step the host optimizer - gradients flow through
    but host weights stay fixed.
    """
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero ALL grads to prevent accumulation
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Only step seed optimizer - host grads computed but not applied
        seed_optimizer.step()


def _train_epoch_blended(model, trainloader, criterion, host_optimizer, seed_optimizer, device):
    """Train one epoch with blended host+seed."""
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        host_optimizer.zero_grad()
        if seed_optimizer:
            seed_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()


def _validate_and_get_metrics(model, trainloader, testloader, criterion, device):
    """Get validation and training metrics."""
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

    # Training metrics (quick sample)
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 10:  # Sample first 10 batches
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

    train_loss /= min(10, len(trainloader))
    train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

    return val_loss, val_accuracy, train_loss, train_accuracy


def generate_episodes(
    n_episodes: int,
    data_dir: str = "data/simic_episodes",
    max_epochs: int = 25,
    device: str = "cuda",
    start_id: int = 0,
) -> DatasetManager:
    """Generate multiple episodes."""

    print(f"\n{'='*60}")
    print(f"Generating {n_episodes} episodes")
    print(f"{'='*60}")

    # Load data once
    print("Loading CIFAR-10...")
    trainloader, testloader = load_cifar10()
    print(f"Device: {device}")

    dm = DatasetManager(data_dir)
    start_time = time.time()

    for i in range(n_episodes):
        episode_id = f"episode_{start_id + i:04d}"

        # Skip if already exists
        if episode_id in dm.list_episodes():
            print(f"  [{i+1:3d}/{n_episodes}] {episode_id} (exists, skipping)")
            continue

        print(f"  [{i+1:3d}/{n_episodes}] {episode_id}...", end=" ", flush=True)
        ep_start = time.time()

        episode = generate_episode(
            episode_id=episode_id,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            device=device,
            verbose=False,
        )

        dm.save_episode(episode)
        ep_time = time.time() - ep_start
        print(f"done ({ep_time:.1f}s, final_acc={episode.final_accuracy:.1f}%)")

    total_time = time.time() - start_time
    print(f"\nGeneration complete in {total_time:.1f}s")
    print(f"Dataset: {dm.summary()}")

    return dm


# =============================================================================
# Training
# =============================================================================

def train_policy(
    data_dir: str = "data/simic_episodes",
    epochs: int = 100,
    save_path: str = "data/policy_tamiyo.pt",
) -> PolicyNetwork:
    """Train policy on collected episodes."""

    print(f"\n{'='*60}")
    print("Training Policy Tamiyo")
    print(f"{'='*60}")

    dm = DatasetManager(data_dir)
    episodes = dm.load_all()

    if not episodes:
        print("ERROR: No episodes found!")
        return None

    print(f"Loaded {len(episodes)} episodes")
    print(f"Total decisions: {sum(len(e.decisions) for e in episodes)}")

    # Train
    policy = PolicyNetwork()
    result = policy.train_on_episodes(episodes, epochs=epochs)

    print(f"\nTraining complete:")
    print(f"  Final val accuracy: {result['final_val_accuracy']:.1f}%")

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    policy.save(save_path)
    print(f"  Saved to: {save_path}")

    return policy


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(
    data_dir: str = "data/simic_episodes",
    policy_path: str = "data/policy_tamiyo.pt",
) -> dict:
    """Evaluate policy with confusion matrix."""

    print(f"\n{'='*60}")
    print("Evaluating Policy Tamiyo")
    print(f"{'='*60}")

    dm = DatasetManager(data_dir)
    episodes = dm.load_all()

    if not episodes:
        print("ERROR: No episodes found!")
        return None

    policy = PolicyNetwork()
    policy.load(policy_path)

    result = policy.evaluate(episodes)

    print(f"\nOverall accuracy: {result['accuracy']:.1f}%")
    print(f"Total samples: {result['total_samples']}")
    print_confusion_matrix(result)

    return result


# =============================================================================
# Live Comparison
# =============================================================================

def live_comparison(
    policy_path: str = "data/policy_tamiyo.pt",
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cuda",
) -> dict:
    """Run live comparison between heuristic and learned policy."""

    print(f"\n{'='*60}")
    print("Live Comparison: Heuristic vs Learned Policy")
    print(f"{'='*60}")

    # Load policy
    policy = PolicyNetwork()
    policy.load(policy_path)

    # Load data
    print("Loading CIFAR-10...")
    trainloader, testloader = load_cifar10()

    results = []

    for ep_idx in range(n_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{n_episodes} ---")

        # Run with same random seed for fair comparison
        torch.manual_seed(42 + ep_idx)

        # Setup shared components
        config = HeuristicPolicyConfig(
            min_epochs_before_germinate=5,
            plateau_epochs_to_germinate=3,
        )

        # Track decisions from both
        heuristic_decisions = []
        policy_decisions = []
        agreements = 0
        total = 0

        # Run training
        model = create_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.get_host_parameters(), lr=0.01, momentum=0.9)

        tamiyo = HeuristicTamiyo(config)
        tracker = SignalTracker()

        for epoch in range(1, max_epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss /= len(trainloader)
            train_accuracy = 100.0 * train_correct / train_total

            # Validation
            model.eval()
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

            # Get signals
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                active_seeds=[],
                available_slots=1,
            )

            # Heuristic decision
            h_decision = tamiyo.decide(signals)
            h_action = action_from_decision(h_decision).action

            # Policy decision
            snapshot = snapshot_from_signals(signals, seed_state=None)
            p_action = policy.predict(snapshot)

            heuristic_decisions.append(h_action)
            policy_decisions.append(p_action)

            if h_action == p_action:
                agreements += 1
            total += 1

            match = "✓" if h_action == p_action else "✗"
            print(f"  Epoch {epoch:2d}: Val={val_accuracy:.2f}% | "
                  f"H={h_action.name:10} P={p_action.name:10} {match}")

        agreement_rate = agreements / total * 100
        print(f"  Agreement: {agreements}/{total} ({agreement_rate:.1f}%)")

        results.append({
            "agreement_rate": agreement_rate,
            "heuristic_decisions": [d.name for d in heuristic_decisions],
            "policy_decisions": [d.name for d in policy_decisions],
        })

    # Summary
    avg_agreement = sum(r["agreement_rate"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: Average agreement rate: {avg_agreement:.1f}%")
    print(f"{'='*60}")

    return {
        "episodes": results,
        "avg_agreement": avg_agreement,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Policy Tamiyo overnight runner")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to generate")
    parser.add_argument("--max-epochs", type=int, default=25,
                        help="Max epochs per episode")
    parser.add_argument("--train-epochs", type=int, default=100,
                        help="Training epochs for policy")
    parser.add_argument("--data-dir", type=str, default="data/simic_episodes",
                        help="Directory for episode data")
    parser.add_argument("--policy-path", type=str, default="data/policy_tamiyo.pt",
                        help="Path to save/load policy")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train on existing data")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate episodes (for parallel runs)")
    parser.add_argument("--compare", action="store_true",
                        help="Only run live comparison")
    parser.add_argument("--compare-episodes", type=int, default=5,
                        help="Episodes for live comparison")
    parser.add_argument("--start-id", type=int, default=0,
                        help="Starting episode ID (for parallel runs)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.compare:
        # Just run comparison
        live_comparison(
            policy_path=args.policy_path,
            n_episodes=args.compare_episodes,
            max_epochs=args.max_epochs,
            device=device,
        )
    elif args.train_only:
        # Just train on existing data
        policy = train_policy(
            data_dir=args.data_dir,
            epochs=args.train_epochs,
            save_path=args.policy_path,
        )
        if policy:
            evaluate_policy(
                data_dir=args.data_dir,
                policy_path=args.policy_path,
            )
    elif args.generate_only:
        # Just generate episodes (for parallel runs)
        generate_episodes(
            n_episodes=args.episodes,
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
            device=device,
            start_id=args.start_id,
        )
    else:
        # Full pipeline: generate, train, evaluate, compare
        generate_episodes(
            n_episodes=args.episodes,
            data_dir=args.data_dir,
            max_epochs=args.max_epochs,
            device=device,
            start_id=args.start_id,
        )

        policy = train_policy(
            data_dir=args.data_dir,
            epochs=args.train_epochs,
            save_path=args.policy_path,
        )

        if policy:
            evaluate_policy(
                data_dir=args.data_dir,
                policy_path=args.policy_path,
            )

            live_comparison(
                policy_path=args.policy_path,
                n_episodes=args.compare_episodes,
                max_epochs=args.max_epochs,
                device=device,
            )


if __name__ == "__main__":
    main()
