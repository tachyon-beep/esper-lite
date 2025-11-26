"""Test script for Simic episode collection.

Runs a single training session and collects data to verify
the collection pipeline works end-to-end.

Usage:
    PYTHONPATH=src python src/esper/simic_test_collection.py
"""

from __future__ import annotations

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
from esper.simic import (
    EpisodeCollector, StepOutcome,
    snapshot_from_signals, action_from_decision,
)


# =============================================================================
# Simple Model (no seeds, just baseline for testing collection)
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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
# Training Loop with Collection
# =============================================================================

def train_with_collection(
    max_epochs: int = 15,  # Short for testing
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Train and collect episode data."""

    print("Loading CIFAR-10...")
    trainloader, testloader = load_cifar10()
    print(f"Device: {device}")

    # Setup model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup Tamiyo (just for decision making, no actual seeds)
    config = HeuristicPolicyConfig(
        min_epochs_before_germinate=5,
        plateau_epochs_to_germinate=3,
    )
    tamiyo = HeuristicTamiyo(config)
    tracker = SignalTracker()

    # Setup Simic collector
    collector = EpisodeCollector()
    collector.start_episode(
        episode_id="test_collection_001",
        max_epochs=max_epochs,
        initial_lr=0.001,
    )

    print(f"\n{'='*60}")
    print("Training with Episode Collection")
    print(f"{'='*60}")
    print(f"Max epochs: {max_epochs}")

    prev_accuracy = 0.0

    for epoch in range(1, max_epochs + 1):
        # Training phase
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

        # Validation phase
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

        # Update signal tracker
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            active_seeds=[],  # No seeds in this test
            available_slots=1,
        )

        # === SIMIC COLLECTION ===
        # 1. Record observation (before decision)
        snapshot = snapshot_from_signals(signals, seed_state=None)
        collector.record_observation(snapshot)

        # 2. Get Tamiyo's decision
        decision = tamiyo.decide(signals)

        # 3. Record action
        action = action_from_decision(decision)
        collector.record_action(action)

        # 4. Record outcome (based on this epoch's results)
        accuracy_change = val_accuracy - prev_accuracy
        outcome = StepOutcome(
            accuracy_after=val_accuracy,
            accuracy_change=accuracy_change,
            loss_after=val_loss,
            loss_change=prev_accuracy - val_loss if epoch > 1 else 0.0,
        )
        collector.record_outcome(outcome)

        prev_accuracy = val_accuracy

        # Print progress
        action_str = decision.action.name
        print(f"  Epoch {epoch:2d}: Val={val_accuracy:.2f}% | "
              f"plateau={signals.plateau_epochs} | action={action_str} | "
              f"reward={outcome.reward:.2f}")

    # End episode
    episode = collector.end_episode(
        final_accuracy=val_accuracy,
        best_accuracy=tracker._best_accuracy,
        seeds_created=0,
        seeds_fossilized=0,
        seeds_culled=0,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Collection Summary")
    print(f"{'='*60}")
    print(f"Episode ID: {episode.episode_id}")
    print(f"Decision points collected: {len(episode.decisions)}")
    print(f"Final accuracy: {episode.final_accuracy:.2f}%")
    print(f"Best accuracy: {episode.best_accuracy:.2f}%")
    print(f"Total reward: {episode.total_reward():.2f}")

    # Show sample data
    print(f"\nSample decision point (epoch 5):")
    if len(episode.decisions) >= 5:
        dp = episode.decisions[4]
        print(f"  Observation vector length: {len(dp.observation.to_vector())}")
        print(f"  Action: {dp.action.action.name}")
        print(f"  Reward: {dp.outcome.reward:.2f}")

    # Test to_training_data
    training_data = episode.to_training_data()
    print(f"\nTraining data tuples: {len(training_data)}")
    if training_data:
        obs, act, rew = training_data[0]
        print(f"  First tuple: obs_len={len(obs)}, act_len={len(act)}, reward={rew:.2f}")

    print("\n✓ Data collection test PASSED")


if __name__ == "__main__":
    train_with_collection()
