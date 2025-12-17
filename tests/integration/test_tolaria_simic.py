"""Integration tests for Tolaria-Simic interaction.

Tests the core integration where:
- Tolaria's trainer runs training epochs
- TolariaGovernor monitors training health
- Training metrics flow to RL reward computation
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria.trainer import (
    train_epoch_normal,
    validate_and_get_metrics,
)
from esper.tolaria.governor import TolariaGovernor
from esper.kasmina import MorphogeneticModel, CNNHost


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Simple model for training tests."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def train_loader():
    """Small training data loader."""
    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def test_loader():
    """Small test data loader."""
    X = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def morphogenetic_model():
    """MorphogeneticModel for governor tests."""
    host = CNNHost(num_classes=10)
    return MorphogeneticModel(host, device="cpu", slots=["r0c1"])


# =============================================================================
# Training Loop Tests
# =============================================================================


class TestTrainingLoop:
    """Tests for Tolaria training loop producing Simic-relevant metrics."""

    def test_train_epoch_updates_parameters(self, simple_model, train_loader):
        """train_epoch_normal should update model parameters.

        Verifies gradient flow through training - essential for Simic RL.
        """
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Get initial param snapshot
        initial_params = [p.clone() for p in simple_model.parameters()]

        train_epoch_normal(
            model=simple_model,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )

        # Parameters should have changed
        params_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_params, simple_model.parameters())
        )
        assert params_changed, "Training should update parameters"

    def test_validation_returns_accuracy(self, simple_model, train_loader, test_loader):
        """validate_and_get_metrics should return accuracy percentage.

        This accuracy is what Simic uses for contribution reward.
        """
        criterion = nn.CrossEntropyLoss()

        val_loss, val_acc, train_loss, train_acc, _, _ = validate_and_get_metrics(
            model=simple_model,
            trainloader=train_loader,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
        )

        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 100, "Accuracy should be percentage"
        assert isinstance(val_loss, float)
        assert val_loss > 0


# =============================================================================
# Governor Integration Tests
# =============================================================================


class TestGovernorIntegration:
    """Tests for TolariaGovernor monitoring training for Simic."""

    def test_governor_detects_nan_loss(self, morphogenetic_model):
        """Governor should detect NaN loss immediately.

        NaN/Inf are immediate triggers (no history needed) - critical for
        catching training explosions early.
        """
        governor = TolariaGovernor(
            model=morphogenetic_model,
            sensitivity=2.0,
            absolute_threshold=10.0,
        )

        # NaN should trigger immediately
        is_bad = governor.check_vital_signs(current_loss=float('nan'))
        assert is_bad, "Governor should detect NaN loss immediately"

    def test_governor_detects_inf_loss(self, morphogenetic_model):
        """Governor should detect Inf loss immediately."""
        governor = TolariaGovernor(
            model=morphogenetic_model,
        )

        is_bad = governor.check_vital_signs(current_loss=float('inf'))
        assert is_bad, "Governor should detect Inf loss immediately"

    def test_governor_provides_punishment_reward(self, morphogenetic_model):
        """Governor should provide punishment reward for RL buffer."""
        governor = TolariaGovernor(
            model=morphogenetic_model,
            death_penalty=-2.0,
        )

        punishment = governor.get_punishment_reward()
        assert punishment == 2.0, "Punishment should be positive (negated penalty)"
