"""Tests for extracted validation loop helper."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestRunValidationPass:
    """Tests for _run_validation_pass helper."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        return nn.Linear(10, 2)

    @pytest.fixture
    def testloader(self):
        """Create a simple test dataloader."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_returns_loss_and_accuracy(self, simple_model, testloader):
        """Should return (average_loss, accuracy) tuple."""
        from esper.tolaria.trainer import _run_validation_pass

        simple_model.eval()
        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy = _run_validation_pass(
            model=simple_model,
            testloader=testloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert avg_loss > 0
        assert 0 <= accuracy <= 100

    def test_accuracy_bounds(self, simple_model, testloader):
        """Accuracy should be between 0 and 100."""
        from esper.tolaria.trainer import _run_validation_pass

        simple_model.eval()
        criterion = nn.CrossEntropyLoss()

        _, accuracy = _run_validation_pass(
            model=simple_model,
            testloader=testloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        assert 0 <= accuracy <= 100

    def test_loss_accumulation(self, simple_model, testloader):
        """Loss should be averaged over batches."""
        from esper.tolaria.trainer import _run_validation_pass

        simple_model.eval()
        criterion = nn.CrossEntropyLoss()

        avg_loss, _ = _run_validation_pass(
            model=simple_model,
            testloader=testloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        # Loss should be positive and reasonable (not accumulated sum)
        assert avg_loss > 0
        # Random 2-class classification: loss ~ log(2) ≈ 0.69
        # Allow some margin for randomness
        assert avg_loss < 10  # Sanity check

    def test_lm_task_type(self):
        """Should handle language modeling task type."""
        from esper.tolaria.trainer import _run_validation_pass

        # Create LM-style model and data
        vocab_size = 100
        seq_len = 16
        batch_size = 4

        model = nn.Linear(32, vocab_size)  # Simple projection to vocab
        model.eval()

        # LM data: input is embeddings, target is token ids
        X = torch.randn(batch_size * 2, seq_len, 32)  # (batch, seq, hidden)
        y = torch.randint(0, vocab_size, (batch_size * 2, seq_len))

        # Wrap model to reshape input
        class LMWrapper(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj

            def forward(self, x):
                # x: (batch, seq, hidden) -> (batch, seq, vocab)
                return self.proj(x)

        lm_model = LMWrapper(model)
        lm_model.eval()

        testloader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy = _run_validation_pass(
            model=lm_model,
            testloader=testloader,
            criterion=criterion,
            device="cpu",
            task_type="lm",
        )

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert avg_loss > 0
        assert 0 <= accuracy <= 100

    def test_empty_dataloader_handling(self, simple_model):
        """Should handle empty dataloader gracefully."""
        from esper.tolaria.trainer import _run_validation_pass

        simple_model.eval()
        criterion = nn.CrossEntropyLoss()

        # Empty dataset
        X = torch.randn(0, 10)
        y = torch.randint(0, 2, (0,))
        empty_loader = DataLoader(TensorDataset(X, y), batch_size=8)

        avg_loss, accuracy = _run_validation_pass(
            model=simple_model,
            testloader=empty_loader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        # Should return 0 for both without crashing
        assert avg_loss == 0.0
        assert accuracy == 0.0
