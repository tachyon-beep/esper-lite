#!/usr/bin/env python3
"""Esper-Lite Proof of Concept: Morphogenetic Seed Injection

This single file demonstrates the core premise of Esper:
1. Train a seed module in gradient isolation (host model unchanged)
2. Blend the seed into the host using alpha ramping
3. Compare performance against baseline and from-scratch training

Run with:
    python src/esper/poc.py

Requirements:
    pip install torch torchvision
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Try to import torchvision, provide helpful error if missing
try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print("Please install torchvision: pip install torchvision")
    raise


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the proof-of-concept experiment."""

    # Dataset
    batch_size: int = 128
    num_workers: int = 2

    # Training phases
    baseline_epochs: int = 10      # Train host alone
    isolation_epochs: int = 5      # Train seed in isolation
    blending_epochs: int = 5       # Blend seed into host
    post_blend_epochs: int = 5     # Fine-tune after blending
    from_scratch_epochs: int = 25  # Train combined architecture from scratch

    # Alpha blending
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_temperature: float = 1.0  # Controls sigmoid steepness

    # Gradient isolation
    isolation_threshold: float = 1e-6  # Max allowed gradient overlap

    # Optimization
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
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
    """Simple CNN for CIFAR-10 classification.

    This is the "host" model that we'll inject seeds into.
    Architecture: 3 conv blocks -> global avg pool -> classifier
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Injection point marker - this is where seeds attach
        self._seed_injection_point = "after_block2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))   # 32x32 -> 16x16
        x = self.pool(self.block2(x))   # 16x16 -> 8x8
        # Seed injection would happen here (between block2 and block3)
        x = self.pool(self.block3(x))   # 8x8 -> 4x4
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def forward_to_injection_point(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the injection point."""
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return x

    def forward_from_injection_point(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from injection point to output."""
        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


class SeedModule(nn.Module):
    """A seed module that can be injected into the host.

    This represents an architectural enhancement - an extra processing
    block that we want to add to the host model during training.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        # A residual-style enhancement block
        self.enhance = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return x + self.enhance(x)


# =============================================================================
# Seed Lifecycle
# =============================================================================

class SeedStage(Enum):
    """Lifecycle stages for a seed."""
    DORMANT = auto()       # Not yet active
    TRAINING = auto()      # Training in isolation
    BLENDING = auto()      # Being blended into host
    FOSSILIZED = auto()    # Permanently part of the model
    CULLED = auto()        # Removed/failed


@dataclass
class SeedState:
    """Tracks the state of a seed through its lifecycle."""

    stage: SeedStage = SeedStage.DORMANT
    alpha: float = 0.0
    epochs_in_stage: int = 0
    isolation_violations: int = 0
    metrics: dict = field(default_factory=dict)

    def transition(self, new_stage: SeedStage) -> None:
        """Transition to a new stage."""
        print(f"  Seed: {self.stage.name} -> {new_stage.name}")
        self.stage = new_stage
        self.epochs_in_stage = 0


# =============================================================================
# Alpha Blending
# =============================================================================

class AlphaSchedule:
    """Sigmoid-based alpha schedule for smooth blending."""

    def __init__(
        self,
        total_steps: int,
        start: float = 0.0,
        end: float = 1.0,
        temperature: float = 1.0,
    ):
        self.total_steps = max(1, total_steps)
        self.start = start
        self.end = end
        self.temperature = max(temperature, 1e-6)

    def __call__(self, step: int) -> float:
        """Get alpha value at given step."""
        if step <= 0:
            return self.start
        if step >= self.total_steps:
            return self.end

        # Sigmoid schedule centered at midpoint
        midpoint = self.total_steps / 2
        scaled = (step - midpoint) / self.temperature
        sigmoid = 0.5 * (1.0 + math.tanh(scaled * 0.5))

        return self.start + (self.end - self.start) * sigmoid


def blend_with_isolation(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend host and seed features with gradient isolation.

    The key insight: host_features.detach() prevents gradients from
    flowing back through the host path during seed training.

    When alpha=0: output = host (seed not used)
    When alpha=1: output = seed (host detached, no gradients to host)
    When 0<alpha<1: output = blend (only seed receives gradients from seed path)
    """
    alpha = max(0.0, min(1.0, alpha))
    return alpha * seed_features + (1.0 - alpha) * host_features.detach()


# =============================================================================
# Gradient Isolation Monitor
# =============================================================================

class GradientIsolationMonitor:
    """Monitors gradient flow to verify isolation between host and seed.

    During seed training, we want to verify that gradients don't leak
    back to the host model. This monitor tracks gradient magnitudes
    and computes the overlap between host and seed gradients.
    """

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.host_grad_norm: float = 0.0
        self.seed_grad_norm: float = 0.0
        self.violations: int = 0
        self._host_params: list[nn.Parameter] = []
        self._seed_params: list[nn.Parameter] = []

    def register(self, host: nn.Module, seed: nn.Module) -> None:
        """Register host and seed modules for monitoring."""
        self._host_params = [p for p in host.parameters() if p.requires_grad]
        self._seed_params = [p for p in seed.parameters() if p.requires_grad]

    def check_isolation(self) -> tuple[bool, dict]:
        """Check if gradient isolation is maintained.

        Returns (is_isolated, stats_dict)
        """
        host_norm = 0.0
        seed_norm = 0.0

        for p in self._host_params:
            if p.grad is not None:
                host_norm += p.grad.norm().item() ** 2

        for p in self._seed_params:
            if p.grad is not None:
                seed_norm += p.grad.norm().item() ** 2

        host_norm = host_norm ** 0.5
        seed_norm = seed_norm ** 0.5

        self.host_grad_norm = host_norm
        self.seed_grad_norm = seed_norm

        # During seed training with alpha < 1, host should have near-zero gradients
        is_isolated = host_norm < self.threshold

        if not is_isolated:
            self.violations += 1

        return is_isolated, {
            "host_grad_norm": host_norm,
            "seed_grad_norm": seed_norm,
            "isolated": is_isolated,
            "violations": self.violations,
        }

    def reset_stats(self) -> None:
        """Reset per-step statistics."""
        self.host_grad_norm = 0.0
        self.seed_grad_norm = 0.0


# =============================================================================
# Combined Model with Seed Injection
# =============================================================================

class MorphogeneticModel(nn.Module):
    """Host model with injectable seed support.

    This wraps a host model and allows a seed to be attached, trained
    in isolation, and gradually blended in.
    """

    def __init__(self, host: HostCNN, seed: SeedModule | None = None):
        super().__init__()
        self.host = host
        self.seed = seed
        self.alpha = 0.0
        self.isolate_seed = False  # When True, detach seed input from host
        self.state = SeedState()
        self.isolation_monitor = GradientIsolationMonitor()

        if seed is not None:
            self.isolation_monitor.register(host, seed)

    def attach_seed(self, seed: SeedModule) -> None:
        """Attach a seed module."""
        self.seed = seed
        self.state = SeedState()
        self.isolation_monitor.register(self.host, seed)
        print(f"  Seed attached, stage: {self.state.stage.name}")

    def set_alpha(self, alpha: float) -> None:
        """Set the blending alpha."""
        self.alpha = max(0.0, min(1.0, alpha))
        self.state.alpha = self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through host up to injection point
        host_features = self.host.forward_to_injection_point(x)

        if self.seed is None or self.alpha == 0.0:
            # No seed or seed not active - pure host path
            return self.host.forward_from_injection_point(host_features)

        # Seed is active - compute seed features and blend
        # During isolation training: detach to prevent gradients to host
        # During blending/fossilized: allow gradients to flow to both
        seed_input = host_features.detach() if self.isolate_seed else host_features
        seed_features = self.seed(seed_input)

        # Blend with gradient isolation
        blended = blend_with_isolation(host_features, seed_features, self.alpha)

        return self.host.forward_from_injection_point(blended)

    def get_host_parameters(self):
        """Get parameters belonging to the host."""
        return self.host.parameters()

    def get_seed_parameters(self):
        """Get parameters belonging to the seed."""
        if self.seed is None:
            return iter([])
        return self.seed.parameters()


# =============================================================================
# Training Utilities
# =============================================================================

def get_cifar10_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders."""

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


def evaluate(model: nn.Module, testloader: DataLoader, device: str) -> tuple[float, float]:
    """Evaluate model accuracy and loss."""
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

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def train_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    check_isolation: bool = False,
    isolation_monitor: GradientIsolationMonitor | None = None,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    isolation_checks = []

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Check gradient isolation if requested
        if check_isolation and isolation_monitor is not None:
            is_isolated, stats = isolation_monitor.check_isolation()
            isolation_checks.append(stats)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
        "isolation_checks": isolation_checks,
    }


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs the morphogenetic training experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.results: dict = {}

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    def run(self) -> dict:
        """Run the complete experiment."""
        print("=" * 60)
        print("Esper-Lite Proof of Concept")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Config: {self.config}")
        print()

        # Load data
        print("Loading CIFAR-10...")
        trainloader, testloader = get_cifar10_loaders(self.config)
        print(f"  Train batches: {len(trainloader)}")
        print(f"  Test batches: {len(testloader)}")
        print()

        # Run experiments
        self.results["baseline"] = self._run_baseline(trainloader, testloader)
        self.results["morphogenetic"] = self._run_morphogenetic(trainloader, testloader)
        self.results["from_scratch"] = self._run_from_scratch(trainloader, testloader)

        # Summary
        self._print_summary()

        return self.results

    def _run_baseline(self, trainloader: DataLoader, testloader: DataLoader) -> dict:
        """Run baseline training (host model only, no seed)."""
        print("-" * 60)
        print("PHASE 1: Baseline Training (Host Only)")
        print("-" * 60)

        model = HostCNN().to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        history = []
        for epoch in range(self.config.baseline_epochs):
            train_stats = train_epoch(model, trainloader, optimizer, self.device)
            test_acc, test_loss = evaluate(model, testloader, self.device)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
                "test_loss": test_loss,
            })

            print(f"  Epoch {epoch+1:2d}: "
                  f"Train Loss={train_stats['loss']:.4f}, "
                  f"Train Acc={train_stats['accuracy']:.2f}%, "
                  f"Test Acc={test_acc:.2f}%")

        final_acc = history[-1]["test_acc"]
        print(f"  Baseline Final Accuracy: {final_acc:.2f}%")
        print()

        return {
            "final_accuracy": final_acc,
            "history": history,
        }

    def _run_morphogenetic(self, trainloader: DataLoader, testloader: DataLoader) -> dict:
        """Run morphogenetic training with seed injection."""
        print("-" * 60)
        print("PHASE 2: Morphogenetic Training (Seed Injection)")
        print("-" * 60)

        # Create host and wrap in morphogenetic model
        host = HostCNN().to(self.device)
        model = MorphogeneticModel(host).to(self.device)

        # Host-only optimizer for initial training
        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        history = []
        epoch_counter = 0

        # Stage 1: Train host alone (same as baseline start)
        print("\n  Stage 1: Initial Host Training")
        model.state.transition(SeedStage.DORMANT)

        for epoch in range(self.config.baseline_epochs):
            train_stats = train_epoch(model, trainloader, host_optimizer, self.device)
            test_acc, test_loss = evaluate(model, testloader, self.device)
            epoch_counter += 1

            history.append({
                "epoch": epoch_counter,
                "stage": "host_only",
                "alpha": 0.0,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
            })

            print(f"    Epoch {epoch_counter:2d}: Test Acc={test_acc:.2f}%")

        pre_seed_accuracy = history[-1]["test_acc"]
        print(f"  Pre-seed accuracy: {pre_seed_accuracy:.2f}%")

        # Stage 2: Attach seed and train in isolation
        print("\n  Stage 2: Seed Training (Gradient Isolation)")
        seed = SeedModule(channels=64).to(self.device)
        model.attach_seed(seed)
        model.state.transition(SeedStage.TRAINING)
        model.set_alpha(1.0)  # Full seed path for training
        model.isolate_seed = True  # Enable gradient isolation

        # Seed-only optimizer
        seed_optimizer = torch.optim.SGD(
            model.get_seed_parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        # Freeze host parameters explicitly and clear any residual gradients
        for p in model.get_host_parameters():
            p.requires_grad = False
            if p.grad is not None:
                p.grad = None

        isolation_violations = 0
        for epoch in range(self.config.isolation_epochs):
            train_stats = train_epoch(
                model, trainloader, seed_optimizer, self.device,
                check_isolation=True,
                isolation_monitor=model.isolation_monitor,
            )
            test_acc, test_loss = evaluate(model, testloader, self.device)
            epoch_counter += 1

            # Check isolation
            violations_this_epoch = sum(
                1 for check in train_stats["isolation_checks"]
                if not check["isolated"]
            )
            isolation_violations += violations_this_epoch

            history.append({
                "epoch": epoch_counter,
                "stage": "seed_isolation",
                "alpha": 1.0,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
                "isolation_violations": violations_this_epoch,
            })

            print(f"    Epoch {epoch_counter:2d}: Test Acc={test_acc:.2f}%, "
                  f"Isolation Violations={violations_this_epoch}")

        # Unfreeze host for blending phase
        for p in model.get_host_parameters():
            p.requires_grad = True

        post_isolation_accuracy = history[-1]["test_acc"]
        print(f"  Post-isolation accuracy: {post_isolation_accuracy:.2f}%")
        print(f"  Total isolation violations: {isolation_violations}")

        # Stage 3: Blend seed into host
        print("\n  Stage 3: Alpha Blending")
        model.state.transition(SeedStage.BLENDING)
        model.isolate_seed = False  # Disable isolation, allow gradients to both

        # Combined optimizer for both host and seed
        combined_optimizer = torch.optim.SGD(
            list(model.get_host_parameters()) + list(model.get_seed_parameters()),
            lr=self.config.learning_rate * 0.1,  # Reduced LR for stability
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        alpha_schedule = AlphaSchedule(
            total_steps=self.config.blending_epochs,
            start=0.0,  # Start from no seed contribution
            end=1.0,    # End with full seed contribution
            temperature=self.config.alpha_temperature,
        )

        for epoch in range(self.config.blending_epochs):
            alpha = alpha_schedule(epoch + 1)
            model.set_alpha(alpha)

            train_stats = train_epoch(model, trainloader, combined_optimizer, self.device)
            test_acc, test_loss = evaluate(model, testloader, self.device)
            epoch_counter += 1

            history.append({
                "epoch": epoch_counter,
                "stage": "blending",
                "alpha": alpha,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
            })

            print(f"    Epoch {epoch_counter:2d}: Alpha={alpha:.3f}, Test Acc={test_acc:.2f}%")

        # Stage 4: Fossilize and fine-tune
        print("\n  Stage 4: Post-Blend Fine-tuning")
        model.state.transition(SeedStage.FOSSILIZED)
        model.set_alpha(1.0)  # Seed fully integrated

        for epoch in range(self.config.post_blend_epochs):
            train_stats = train_epoch(model, trainloader, combined_optimizer, self.device)
            test_acc, test_loss = evaluate(model, testloader, self.device)
            epoch_counter += 1

            history.append({
                "epoch": epoch_counter,
                "stage": "fossilized",
                "alpha": 1.0,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
            })

            print(f"    Epoch {epoch_counter:2d}: Test Acc={test_acc:.2f}%")

        final_accuracy = history[-1]["test_acc"]
        print(f"\n  Morphogenetic Final Accuracy: {final_accuracy:.2f}%")
        print()

        return {
            "final_accuracy": final_accuracy,
            "pre_seed_accuracy": pre_seed_accuracy,
            "post_isolation_accuracy": post_isolation_accuracy,
            "isolation_violations": isolation_violations,
            "history": history,
        }

    def _run_from_scratch(self, trainloader: DataLoader, testloader: DataLoader) -> dict:
        """Train the combined architecture from scratch (upper bound)."""
        print("-" * 60)
        print("PHASE 3: From-Scratch Training (Combined Architecture)")
        print("-" * 60)

        # Build model with seed baked in from the start
        host = HostCNN().to(self.device)
        seed = SeedModule(channels=64).to(self.device)
        model = MorphogeneticModel(host, seed).to(self.device)
        model.set_alpha(1.0)  # Seed always active

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        history = []
        for epoch in range(self.config.from_scratch_epochs):
            train_stats = train_epoch(model, trainloader, optimizer, self.device)
            test_acc, test_loss = evaluate(model, testloader, self.device)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "test_acc": test_acc,
            })

            print(f"  Epoch {epoch+1:2d}: "
                  f"Train Acc={train_stats['accuracy']:.2f}%, "
                  f"Test Acc={test_acc:.2f}%")

        final_accuracy = history[-1]["test_acc"]
        print(f"  From-Scratch Final Accuracy: {final_accuracy:.2f}%")
        print()

        return {
            "final_accuracy": final_accuracy,
            "history": history,
        }

    def _print_summary(self) -> None:
        """Print experiment summary."""
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        baseline = self.results["baseline"]["final_accuracy"]
        morpho = self.results["morphogenetic"]["final_accuracy"]
        scratch = self.results["from_scratch"]["final_accuracy"]
        violations = self.results["morphogenetic"]["isolation_violations"]

        print(f"  Baseline (host only):      {baseline:.2f}%")
        print(f"  Morphogenetic (seed+host): {morpho:.2f}%")
        print(f"  From-scratch (combined):   {scratch:.2f}%")
        print()
        print(f"  Gradient isolation violations: {violations}")
        print()

        # Analysis
        morpho_vs_baseline = morpho - baseline
        morpho_vs_scratch = morpho - scratch

        print("  Analysis:")
        print(f"    Morphogenetic vs Baseline: {morpho_vs_baseline:+.2f}%")
        print(f"    Morphogenetic vs Scratch:  {morpho_vs_scratch:+.2f}%")
        print()

        if violations == 0:
            print("  [PASS] Gradient isolation maintained throughout seed training")
        else:
            print(f"  [WARN] {violations} gradient isolation violations detected")

        if morpho_vs_baseline > 0:
            print("  [PASS] Seed injection improved performance over baseline")
        else:
            print("  [FAIL] Seed injection did not improve over baseline")

        if abs(morpho_vs_scratch) < 2.0:
            print("  [PASS] Morphogenetic training competitive with from-scratch")
        elif morpho_vs_scratch < -2.0:
            print("  [WARN] Morphogenetic underperforms from-scratch by >2%")
        else:
            print("  [GOOD] Morphogenetic outperforms from-scratch")

        print()
        print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the proof-of-concept experiment."""
    config = ExperimentConfig()
    runner = ExperimentRunner(config)

    start_time = time.time()
    results = runner.run()
    elapsed = time.time() - start_time

    print(f"Total time: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    main()
