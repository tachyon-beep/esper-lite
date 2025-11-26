#!/usr/bin/env python3
"""Esper-Lite POC with Tiny Tamiyo: Reactive Seed Injection

This demonstrates Tamiyo-driven training where seed injection is
reactive to training dynamics rather than following a fixed schedule.

Run with:
    python src/esper/poc_tamiyo.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from esper.kasmina import (
    SeedSlot,
    SeedStage,
    BlueprintCatalog,
    AlphaSchedule,
    GradientIsolationMonitor,
)
from esper.tamiyo import (
    SignalTracker,
    HeuristicTamiyo,
    HeuristicPolicyConfig,
    TamiyoAction,
    TamiyoDecision,
    FieldReportCollector,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TamiyoExperimentConfig:
    """Configuration for Tamiyo-driven experiment."""

    # Dataset
    batch_size: int = 128
    num_workers: int = 2

    # Training limits
    max_epochs: int = 40
    max_seeds: int = 3  # Maximum seeds to try

    # Tamiyo policy config
    plateau_epochs_to_germinate: int = 3
    min_epochs_before_germinate: int = 5
    min_training_epochs: int = 3
    blending_epochs: int = 5
    cull_after_epochs_without_improvement: int = 5

    # Optimization
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# =============================================================================
# Models
# =============================================================================

class ConvBlock(nn.Module):
    """Standard conv-bn-relu block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class HostCNN(nn.Module):
    """Host CNN with injection point."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Injection point info
        self.injection_channels = 64  # After block2

    def forward_to_injection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return x

    def forward_from_injection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_to_injection(x)
        return self.forward_from_injection(x)


class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slot."""

    def __init__(self, host: HostCNN, device: str = "cpu"):
        super().__init__()
        self.host = host
        self._device = device

        # Single seed slot at injection point
        self.seed_slot = SeedSlot(
            slot_id="injection_point",
            channels=host.injection_channels,
            device=device,
        )

        # Isolation monitor
        self.isolation_monitor = GradientIsolationMonitor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.host.forward_to_injection(x)
        features = self.seed_slot.forward(features)
        return self.host.forward_from_injection(features)

    def germinate_seed(self, blueprint_id: str, seed_id: str) -> None:
        """Germinate a new seed."""
        state = self.seed_slot.germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )
        print(f"    [Kasmina] Germinated seed '{seed_id}' with blueprint '{blueprint_id}'")

    def cull_seed(self) -> None:
        """Cull the current seed."""
        if self.seed_slot.state:
            print(f"    [Kasmina] Culling seed '{self.seed_slot.state.seed_id}'")
        self.seed_slot.cull()

    def get_seed_parameters(self):
        return self.seed_slot.get_parameters()

    def get_host_parameters(self):
        return self.host.parameters()

    @property
    def has_active_seed(self) -> bool:
        return self.seed_slot.is_active

    @property
    def seed_state(self):
        return self.seed_slot.state


# =============================================================================
# Data Loading
# =============================================================================

def get_cifar10_loaders(config: TamiyoExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders."""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    testloader = DataLoader(
        testset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )

    return trainloader, testloader


# =============================================================================
# Training Utilities
# =============================================================================

def evaluate(model: nn.Module, testloader: DataLoader, device: str) -> tuple[float, float]:
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total, total_loss / total


def train_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


# =============================================================================
# Tamiyo-Driven Training Loop
# =============================================================================

class TamiyoTrainer:
    """Training loop driven by Tamiyo decisions."""

    def __init__(
        self,
        model: MorphogeneticModel,
        trainloader: DataLoader,
        testloader: DataLoader,
        config: TamiyoExperimentConfig,
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.config = config
        self.device = config.device

        # Create Tamiyo
        policy_config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=config.plateau_epochs_to_germinate,
            min_epochs_before_germinate=config.min_epochs_before_germinate,
            min_training_epochs=config.min_training_epochs,
            blending_epochs=config.blending_epochs,
            cull_after_epochs_without_improvement=config.cull_after_epochs_without_improvement,
        )
        self.tamiyo = HeuristicTamiyo(policy_config)
        self.signal_tracker = SignalTracker()
        self.field_reports = FieldReportCollector()

        # Optimizers
        self.host_optimizer = torch.optim.SGD(
            model.get_host_parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.seed_optimizer: torch.optim.Optimizer | None = None

        # State
        self.epoch = 0
        self.global_step = 0
        self.seeds_created = 0
        self.history: list[dict] = []

    def run(self) -> dict:
        """Run Tamiyo-driven training."""

        print("=" * 60)
        print("Tamiyo-Driven Training")
        print("=" * 60)
        print(f"Max epochs: {self.config.max_epochs}")
        print(f"Max seeds: {self.config.max_seeds}")
        print(f"Available blueprints: {BlueprintCatalog.list_blueprints()}")
        print()

        for epoch in range(1, self.config.max_epochs + 1):
            self.epoch = epoch
            self._run_epoch()

            # Check termination conditions
            if self.seeds_created >= self.config.max_seeds:
                # Check if current seed is done
                if not self.model.has_active_seed:
                    print(f"\n  Reached max seeds ({self.config.max_seeds}), stopping")
                    break
                elif self.model.seed_state.stage == SeedStage.FOSSILIZED:
                    print(f"\n  Final seed fossilized, stopping")
                    break

        return self._summarize()

    def _run_epoch(self) -> None:
        """Run a single epoch with Tamiyo decision."""

        # Determine what to train based on seed state
        if self.model.has_active_seed:
            state = self.model.seed_state
            if state.stage == SeedStage.TRAINING:
                self._train_isolated_seed()
            elif state.stage in (SeedStage.BLENDING, SeedStage.FOSSILIZED):
                self._train_blended()
            else:
                self._train_host_only()
        else:
            self._train_host_only()

        # Evaluate
        val_acc, val_loss = evaluate(self.model, self.testloader, self.device)
        train_loss, train_acc = self.history[-1]["train_loss"], self.history[-1]["train_acc"]

        # Update seed metrics if active
        if self.model.has_active_seed:
            self.model.seed_state.metrics.record_accuracy(val_acc)
            self.model.seed_state.increment_epoch()

        # Get training signals
        active_seeds = [self.model.seed_state] if self.model.has_active_seed else []
        signals = self.signal_tracker.update(
            epoch=self.epoch,
            global_step=self.global_step,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=0 if self.model.has_active_seed else 1,
        )

        # Update history
        self.history[-1].update({
            "val_acc": val_acc,
            "val_loss": val_loss,
            "plateau_epochs": signals.plateau_epochs,
        })

        # Print status
        stage_str = self.model.seed_state.stage.name if self.model.has_active_seed else "NO_SEED"
        alpha_str = f"α={self.model.seed_slot.alpha:.2f}" if self.model.has_active_seed else ""
        print(f"  Epoch {self.epoch:2d} [{stage_str:12s}] {alpha_str:8s} "
              f"Val={val_acc:.2f}% (plateau={signals.plateau_epochs})")

        # Ask Tamiyo for decision
        decision = self.tamiyo.decide(signals)
        self._execute_decision(decision, signals)

    def _train_host_only(self) -> None:
        """Train only the host model."""
        train_loss, train_acc = train_epoch(
            self.model, self.trainloader, self.host_optimizer, self.device
        )
        self.history.append({
            "epoch": self.epoch,
            "stage": "host_only",
            "train_loss": train_loss,
            "train_acc": train_acc,
        })

    def _train_isolated_seed(self) -> None:
        """Train seed in isolation (host frozen)."""
        # Ensure seed optimizer exists
        if self.seed_optimizer is None:
            self.seed_optimizer = torch.optim.SGD(
                self.model.get_seed_parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        # Freeze host
        for p in self.model.get_host_parameters():
            p.requires_grad = False
            if p.grad is not None:
                p.grad = None

        # Enable isolation
        self.model.seed_slot.isolate_gradients = True
        self.model.seed_slot.set_alpha(1.0)

        # Train
        train_loss, train_acc = train_epoch(
            self.model, self.trainloader, self.seed_optimizer, self.device
        )

        self.history.append({
            "epoch": self.epoch,
            "stage": "isolated_training",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "alpha": 1.0,
        })

    def _train_blended(self) -> None:
        """Train both host and seed together."""
        # Unfreeze host
        for p in self.model.get_host_parameters():
            p.requires_grad = True

        # Disable isolation
        self.model.seed_slot.isolate_gradients = False

        # Update alpha if blending
        if self.model.seed_state.stage == SeedStage.BLENDING:
            step = self.model.seed_state.epochs_in_stage + 1
            alpha = self.model.seed_slot.update_alpha_for_step(step)
        else:
            alpha = 1.0
            self.model.seed_slot.set_alpha(alpha)

        # Combined optimizer
        combined_params = list(self.model.get_host_parameters()) + list(self.model.get_seed_parameters())
        combined_optimizer = torch.optim.SGD(
            combined_params,
            lr=self.config.learning_rate * 0.5,  # Reduced LR for blending
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        train_loss, train_acc = train_epoch(
            self.model, self.trainloader, combined_optimizer, self.device
        )

        self.history.append({
            "epoch": self.epoch,
            "stage": self.model.seed_state.stage.name.lower(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "alpha": alpha,
        })

    def _execute_decision(self, decision: TamiyoDecision, signals) -> None:
        """Execute a Tamiyo decision."""

        if decision.action == TamiyoAction.WAIT:
            return

        print(f"    [Tamiyo] {decision}")

        if decision.action == TamiyoAction.GERMINATE:
            if self.seeds_created < self.config.max_seeds:
                seed_id = f"seed_{self.seeds_created}"
                self.model.germinate_seed(decision.blueprint_id, seed_id)
                self.seeds_created += 1
                self.seed_optimizer = None  # Reset optimizer
                self.field_reports.start_tracking(seed_id, decision.blueprint_id, signals)

        elif decision.action == TamiyoAction.ADVANCE_TRAINING:
            if self.model.has_active_seed:
                self.model.seed_state.transition(SeedStage.TRAINING)
                print(f"    [Kasmina] Seed entering isolated training")

        elif decision.action == TamiyoAction.ADVANCE_BLENDING:
            if self.model.has_active_seed:
                self.model.seed_state.transition(SeedStage.BLENDING)
                self.model.seed_slot.start_blending(
                    total_steps=self.config.blending_epochs,
                    temperature=1.0,
                )
                print(f"    [Kasmina] Seed entering blending phase")

        elif decision.action == TamiyoAction.ADVANCE_FOSSILIZE:
            if self.model.has_active_seed and self.model.seed_state.stage != SeedStage.FOSSILIZED:
                seed_id = self.model.seed_state.seed_id
                self.model.seed_state.transition(SeedStage.FOSSILIZED)
                self.model.seed_slot.set_alpha(1.0)
                self.field_reports.complete_tracking(seed_id, signals, SeedStage.FOSSILIZED)
                print(f"    [Kasmina] Seed fossilized permanently!")

        elif decision.action == TamiyoAction.CULL:
            if self.model.has_active_seed:
                seed_id = self.model.seed_state.seed_id
                self.field_reports.complete_tracking(seed_id, signals, SeedStage.CULLED)
                self.model.cull_seed()
                self.seed_optimizer = None

    def _summarize(self) -> dict:
        """Generate summary of training."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        final_acc = self.history[-1]["val_acc"]
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Total Epochs: {self.epoch}")
        print(f"  Seeds Created: {self.seeds_created}")

        # Field reports
        print(f"\n  Field Reports:")
        for report in self.field_reports.reports:
            status = "FOSSILIZED" if report.success else "CULLED"
            print(f"    - {report.seed_id} ({report.blueprint_id}): {status}, "
                  f"improvement={report.improvement:+.2f}%")

        # Decisions summary
        decisions = self.tamiyo.decisions
        action_counts = {}
        for d in decisions:
            action_counts[d.action.name] = action_counts.get(d.action.name, 0) + 1
        print(f"\n  Tamiyo Decisions: {len(decisions)} total")
        for action, count in sorted(action_counts.items()):
            print(f"    - {action}: {count}")

        return {
            "final_accuracy": final_acc,
            "total_epochs": self.epoch,
            "seeds_created": self.seeds_created,
            "history": self.history,
            "field_reports": [r.to_dict() for r in self.field_reports.reports],
        }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Tamiyo-driven experiment."""
    config = TamiyoExperimentConfig()

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    print("Loading CIFAR-10...")
    trainloader, testloader = get_cifar10_loaders(config)

    print(f"Device: {config.device}")
    print(f"Train batches: {len(trainloader)}")
    print()

    # Create model
    host = HostCNN().to(config.device)
    model = MorphogeneticModel(host, device=config.device).to(config.device)

    # Run training
    trainer = TamiyoTrainer(model, trainloader, testloader, config)

    start_time = time.time()
    results = trainer.run()
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
