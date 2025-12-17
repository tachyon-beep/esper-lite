"""Tests for extracted training loop helper."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestTrainOneEpoch:
    """Tests for _train_one_epoch helper function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_returns_correct_tuple_types(self, simple_model, simple_dataloader):
        """Should return (float, float, int, None) tuple without gradient collection.

        Note: running_loss and correct are floats because _train_one_epoch
        accumulates tensors internally and calls .item() at the end for the
        caller's convenience. The optimization is that .item() is called once
        per epoch, not once per batch.
        """
        from esper.simic.training.helpers import _train_one_epoch

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        result = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="classification",
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        running_loss, correct, total, grad_stats = result
        assert isinstance(running_loss, float)
        assert isinstance(correct, (float, int))  # Tensor.item() returns int for long tensors
        assert isinstance(total, int)
        assert grad_stats is None  # Not collected by default

    def test_accumulates_correctly(self, simple_model, simple_dataloader):
        """Should accumulate loss, correct, and total across batches."""
        from esper.simic.training.helpers import _train_one_epoch

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        running_loss, correct, total, grad_stats = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="classification",
        )

        # Should process all 32 samples
        assert total == 32
        # Loss should be positive
        assert running_loss > 0
        # Correct should be between 0 and total
        assert 0 <= correct <= total
        # Gradients not collected
        assert grad_stats is None

    def test_with_seed_optimizer(self, simple_model, simple_dataloader):
        """Should work with both host and seed optimizers.

        Note: This creates a seed optimizer for a module not in the forward pass,
        testing the code path without verifying actual gradient updates.
        """
        from esper.simic.training.helpers import _train_one_epoch

        # Create a second "seed" module (not used in forward pass)
        seed_module = nn.Linear(10, 10)
        criterion = nn.CrossEntropyLoss()
        host_optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        seed_optimizer = torch.optim.SGD(seed_module.parameters(), lr=0.01)

        result = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=host_optimizer,
            seed_optimizer=seed_optimizer,
            device="cpu",
            task_type="classification",
        )

        assert len(result) == 4
        running_loss, correct, total, grad_stats = result
        assert total == 32
        assert grad_stats is None

    def test_lm_task_type(self):
        """Should handle language modeling task type."""
        from esper.simic.training.helpers import _train_one_epoch

        # Simple LM-like model: input (batch, seq, features) -> output (batch, seq, vocab)
        model = nn.Linear(16, 100)  # 16 features -> 100 vocab

        # Create LM-style data: (batch, seq) for both input features and targets
        X = torch.randn(8, 4, 16)  # batch=8, seq=4, features=16
        y = torch.randint(0, 100, (8, 4))  # batch=8, seq=4, vocab targets

        # Wrap model to handle 3D input
        class LMWrapper(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, x):
                # (batch, seq, features) -> (batch, seq, vocab)
                return self.linear(x)

        wrapped_model = LMWrapper(model)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=0.01)

        running_loss, correct, total, grad_stats = _train_one_epoch(
            model=wrapped_model,
            trainloader=dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="lm",
        )

        # Should process all tokens: 8 samples * 4 seq_len = 32 tokens
        assert total == 32
        assert running_loss > 0
        assert 0 <= correct <= total
        assert grad_stats is None

    def test_empty_dataloader(self):
        """Should handle empty dataloader without errors."""
        from esper.simic.training.helpers import _train_one_epoch

        model = nn.Linear(10, 2)
        # Create empty dataloader
        empty_dataloader = DataLoader(TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long)), batch_size=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        running_loss, correct, total, grad_stats = _train_one_epoch(
            model=model,
            trainloader=empty_dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="classification",
        )

        # Should return zeros for empty dataloader
        assert running_loss == 0.0
        assert correct == 0.0
        assert total == 0
        assert grad_stats is None
