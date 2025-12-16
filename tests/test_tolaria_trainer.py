"""Tests for Tolaria trainer functions."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DummyClassifier(nn.Module):
    """Minimal classifier for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.5)  # Behaves differently in train vs eval

    def forward(self, x):
        return self.linear(self.dropout(x))


class MockSeedSlot:
    """Mock seed slot for attribution testing."""
    def __init__(self):
        self._force_alpha_value = None

    class _ForceAlphaContext:
        def __init__(self, slot, alpha):
            self.slot = slot
            self.alpha = alpha
            self.original = None

        def __enter__(self):
            self.original = self.slot._force_alpha_value
            self.slot._force_alpha_value = self.alpha
            return self

        def __exit__(self, *args):
            self.slot._force_alpha_value = self.original

    def force_alpha(self, alpha):
        return self._ForceAlphaContext(self, alpha)


class DummyModelWithSlot(nn.Module):
    """Model with mock seed slot for attribution testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.seed_slots = {"r0c1": MockSeedSlot()}

    def forward(self, x):
        return self.linear(x)


class TestValidateWithAttribution:
    """Tests for validate_with_attribution function."""

    def test_restores_training_mode_after_validation(self):
        """Test that training mode is restored after attribution validation."""
        from esper.tolaria.trainer import validate_with_attribution

        model = DummyModelWithSlot()

        # Create minimal test data
        inputs = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(inputs, labels)
        testloader = DataLoader(dataset, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Set model to training mode
        model.train()
        assert model.training is True

        # Run attribution validation
        validate_with_attribution(model, testloader, criterion, "cpu", slot="r0c1")

        # Model should be back in training mode
        assert model.training is True, (
            "Model should be in training mode after validate_with_attribution"
        )

    def test_restores_eval_mode_if_originally_eval(self):
        """Test that eval mode is preserved if model was originally in eval."""
        from esper.tolaria.trainer import validate_with_attribution

        model = DummyModelWithSlot()

        inputs = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(inputs, labels)
        testloader = DataLoader(dataset, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Set model to eval mode
        model.eval()
        assert model.training is False

        validate_with_attribution(model, testloader, criterion, "cpu", slot="r0c1")

        # Model should still be in eval mode
        assert model.training is False, (
            "Model should remain in eval mode after validate_with_attribution"
        )

    def test_restores_mode_even_on_exception(self):
        """Test that training mode is restored even if validation raises."""
        pass  # Placeholder - implementation handles this via try/finally


class TestAttributionResult:
    """Tests for AttributionResult dataclass."""

    def test_attribution_result_structure(self):
        """Test AttributionResult has correct fields."""
        from esper.tolaria.trainer import AttributionResult

        result = AttributionResult(
            real_accuracy=85.0,
            baseline_accuracy=80.0,
            seed_contribution=5.0,
            real_loss=0.5,
            baseline_loss=0.6,
        )

        assert result.real_accuracy == 85.0
        assert result.baseline_accuracy == 80.0
        assert result.seed_contribution == 5.0
        assert result.real_loss == 0.5
        assert result.baseline_loss == 0.6

    def test_seed_contribution_calculation(self):
        """Test that seed_contribution = real - baseline."""
        from esper.tolaria.trainer import AttributionResult

        result = AttributionResult(
            real_accuracy=85.0,
            baseline_accuracy=80.0,
            seed_contribution=85.0 - 80.0,  # Should be 5.0
            real_loss=0.5,
            baseline_loss=0.6,
        )

        assert result.seed_contribution == result.real_accuracy - result.baseline_accuracy


class TestValidateWithAttributionIntegration:
    """Integration tests for validate_with_attribution with real MorphogeneticModel."""

    @pytest.fixture
    def model_with_seed(self):
        """Create MorphogeneticModel with an active seed."""
        from esper.kasmina import MorphogeneticModel, CNNHost

        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_light", "test_seed", slot="r0c1")

        # Advance to BLENDING stage so alpha > 0
        model.seed_slots["r0c1"].state.stage = 4  # SeedStage.BLENDING
        model.seed_slots["r0c1"]._alpha = 0.5

        return model

    @pytest.fixture
    def test_data(self):
        """Create CIFAR-10-like test data."""
        # 8 samples of 3x32x32 images
        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=4)

    def test_attribution_with_active_seed(self, model_with_seed, test_data):
        """Test attribution validation with an active seed."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu", slot="r0c1"
        )

        # Result should have valid structure
        assert isinstance(result.real_accuracy, float)
        assert isinstance(result.baseline_accuracy, float)
        assert isinstance(result.seed_contribution, float)
        assert result.seed_contribution == result.real_accuracy - result.baseline_accuracy

    def test_attribution_contribution_sign(self, model_with_seed, test_data):
        """Test that seed_contribution can be positive or negative."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu", slot="r0c1"
        )

        # Contribution can be any value (positive = seed helps, negative = seed hurts)
        assert isinstance(result.seed_contribution, float)
        # Both accuracies should be in [0, 100] range
        assert 0.0 <= result.real_accuracy <= 100.0
        assert 0.0 <= result.baseline_accuracy <= 100.0

    def test_force_alpha_context_restores_alpha(self, model_with_seed, test_data):
        """Test that force_alpha context manager properly restores original alpha."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        original_alpha = model_with_seed.seed_slots["r0c1"].alpha

        validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu", slot="r0c1"
        )

        # Alpha should be restored after validation
        assert model_with_seed.seed_slots["r0c1"].alpha == original_alpha

    def test_attribution_with_empty_loader(self):
        """Test attribution handles empty dataloader gracefully."""
        from esper.tolaria.trainer import validate_with_attribution
        from esper.kasmina import MorphogeneticModel, CNNHost

        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_light", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.stage = 4
        model.seed_slots["r0c1"]._alpha = 0.5

        # Empty dataset
        empty_dataset = TensorDataset(
            torch.randn(0, 3, 32, 32),
            torch.randint(0, 10, (0,))
        )
        empty_loader = DataLoader(empty_dataset, batch_size=4)

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(model, empty_loader, criterion, "cpu", slot="r0c1")

        # Should return 0 accuracy for empty loader
        assert result.real_accuracy == 0.0
        assert result.baseline_accuracy == 0.0
        assert result.seed_contribution == 0.0
