"""Integration tests for Tolaria-Kasmina interaction.

Tests the core integration where:
- Tolaria trainer works with MorphogeneticModel
- Validation correctly measures seed contribution (attribution)
- force_alpha context manager works for counterfactual evaluation
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria.trainer import (
    train_epoch_normal,
    train_epoch_blended,
    validate_with_attribution,
    AttributionResult,
)
from esper.kasmina import MorphogeneticModel, CNNHost
from esper.leyline import SeedStage


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def morphogenetic_model():
    """MorphogeneticModel with seed slot."""
    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    return model


@pytest.fixture
def model_with_active_seed(morphogenetic_model):
    """Model with germinated seed ready for training."""
    morphogenetic_model.germinate_seed("conv_light", "test_seed", slot="r0c1")
    slot = morphogenetic_model.seed_slots["r0c1"]
    slot.state.stage = SeedStage.TRAINING
    return morphogenetic_model


@pytest.fixture
def cifar_loaders():
    """CIFAR-like data loaders."""
    X_train = torch.randn(64, 3, 32, 32)
    y_train = torch.randint(0, 10, (64,))
    X_test = torch.randn(32, 3, 32, 32)
    y_test = torch.randint(0, 10, (32,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
    return train_loader, test_loader


# =============================================================================
# Training with MorphogeneticModel Tests
# =============================================================================


class TestTrainingWithMorphogeneticModel:
    """Tests for Tolaria training MorphogeneticModel."""

    def test_train_epoch_with_dormant_seed(self, morphogenetic_model, cifar_loaders):
        """Training should work when seed is dormant (host-only mode)."""
        train_loader, _ = cifar_loaders
        optimizer = torch.optim.SGD(morphogenetic_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Should not raise
        train_epoch_normal(
            model=morphogenetic_model,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )

    def test_train_epoch_with_active_seed(self, model_with_active_seed, cifar_loaders):
        """Training should work with active seed in TRAINING stage."""
        train_loader, _ = cifar_loaders
        optimizer = torch.optim.SGD(model_with_active_seed.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Should not raise
        train_epoch_normal(
            model=model_with_active_seed,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )


# =============================================================================
# Attribution Validation Tests
# =============================================================================


class TestAttributionValidation:
    """Tests for validate_with_attribution measuring seed contribution."""

    def test_attribution_returns_result(self, model_with_active_seed, cifar_loaders):
        """Attribution validation should return AttributionResult."""
        _, test_loader = cifar_loaders
        criterion = nn.CrossEntropyLoss()

        # Advance to BLENDING so alpha > 0
        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot.state.alpha = 0.5

        result = validate_with_attribution(
            model=model_with_active_seed,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
            slot="r0c1",
        )

        assert isinstance(result, AttributionResult)
        assert hasattr(result, "real_accuracy")
        assert hasattr(result, "baseline_accuracy")
        assert hasattr(result, "seed_contribution")

    def test_attribution_contribution_equals_difference(
        self, model_with_active_seed, cifar_loaders
    ):
        """seed_contribution should equal real_accuracy - baseline_accuracy."""
        _, test_loader = cifar_loaders
        criterion = nn.CrossEntropyLoss()

        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot.state.alpha = 0.5

        result = validate_with_attribution(
            model=model_with_active_seed,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
            slot="r0c1",
        )

        expected = result.real_accuracy - result.baseline_accuracy
        assert abs(result.seed_contribution - expected) < 0.001

    def test_force_alpha_context_restores_alpha(
        self, model_with_active_seed, cifar_loaders
    ):
        """force_alpha context manager should restore original alpha."""
        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        original_alpha = 0.7
        slot.state.alpha = original_alpha

        # Use force_alpha directly
        with slot.force_alpha(0.0):
            assert slot.alpha == 0.0, "Alpha should be forced to 0"

        assert slot.alpha == original_alpha, "Alpha should be restored"
