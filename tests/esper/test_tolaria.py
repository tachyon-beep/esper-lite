"""Tests for Tolaria subsystem."""

import itertools

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria import (
    create_model,
    train_epoch_blended,
    train_epoch_normal,
    train_epoch_incubator_mode,
    validate_and_get_metrics,
)


class TestEnvironment:
    """Tests for tolaria.environment module."""

    def test_create_model_cpu(self):
        """Test model creation on CPU."""
        model = create_model(device="cpu")
        assert not next(model.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_cuda(self):
        """Test model creation on CUDA."""
        model = create_model(device="cuda")
        assert next(model.parameters()).is_cuda

    def test_create_model_invalid_cuda_raises(self):
        """Test error handling for invalid CUDA device."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")

        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            create_model(device="cuda")


class TestTrainer:
    """Tests for tolaria.trainer module."""

    @pytest.fixture
    def model_and_loader(self):
        """Create model and minimal data loader for testing."""
        model = create_model(device="cpu")
        x = torch.randn(64, 3, 32, 32)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        mini_train = list(itertools.islice(loader, 2))
        mini_test = list(itertools.islice(loader, 2))
        return model, mini_train, mini_test

    def test_train_epoch_normal_runs(self, model_and_loader):
        """Smoke test for train_epoch_normal."""
        model, mini_train, _ = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        train_epoch_normal(model, mini_train, criterion, optimizer, "cpu")

    def test_train_epoch_incubator_mode_runs(self, model_and_loader):
        """Smoke test for train_epoch_incubator_mode (STE training)."""
        model, mini_train, _ = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()
        # For smoke test, use any optimizer (even for all params)
        # In real usage, host_optimizer has host params, seed_optimizer has seed params
        host_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        seed_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        train_epoch_incubator_mode(model, mini_train, criterion, host_optimizer, seed_optimizer, "cpu", slot="mid")

    def test_train_epoch_blended_runs(self, model_and_loader):
        """Smoke test for train_epoch_blended."""
        model, mini_train, _ = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()
        host_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # seed_optimizer can be None or another optimizer
        seed_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        train_epoch_blended(model, mini_train, criterion, host_optimizer, seed_optimizer, "cpu")

    def test_validate_and_get_metrics_returns_tuple(self, model_and_loader):
        """Test validate_and_get_metrics returns expected structure."""
        model, mini_train, mini_test = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()

        result = validate_and_get_metrics(
            model, mini_train, mini_test, criterion, "cpu"
        )

        assert len(result) == 6
        val_loss, val_acc, train_loss, train_acc, per_class, perplexity = result
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 100
        assert per_class is None  # compute_per_class=False by default
        assert perplexity is None

    def test_validate_and_get_metrics_per_class(self, model_and_loader):
        """Test per-class accuracy computation."""
        model, mini_train, mini_test = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()

        result = validate_and_get_metrics(
            model, mini_train, mini_test, criterion, "cpu",
            compute_per_class=True
        )

        _, _, _, _, per_class, perplexity = result
        assert per_class is not None
        assert len(per_class) == 10  # CIFAR-10 has 10 classes
        assert perplexity is None
