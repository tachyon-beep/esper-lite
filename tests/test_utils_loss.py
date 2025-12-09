"""Tests for consolidated loss computation utilities.

These utilities replace the duplicate _loss_and_correct and _compute_loss
functions scattered across simic/training.py, tolaria/trainer.py, and
scripts/evaluate.py.
"""

import pytest
import torch
import torch.nn as nn


class TestComputeTaskLoss:
    """Tests for compute_task_loss function."""

    def test_classification_loss(self):
        """Should compute CrossEntropyLoss for classification."""
        from esper.utils.loss import compute_task_loss

        outputs = torch.randn(8, 10)  # batch=8, classes=10
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "classification")

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_lm_loss(self):
        """Should reshape and compute loss for language modeling."""
        from esper.utils.loss import compute_task_loss

        outputs = torch.randn(4, 16, 1000)  # batch=4, seq=16, vocab=1000
        targets = torch.randint(0, 1000, (4, 16))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "lm")

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_lm_loss_reshapes_correctly(self):
        """Should flatten batch and sequence dims for LM loss."""
        from esper.utils.loss import compute_task_loss

        batch, seq, vocab = 2, 4, 100
        outputs = torch.randn(batch, seq, vocab)
        targets = torch.randint(0, vocab, (batch, seq))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "lm")

        # Verify by computing manually
        expected = criterion(outputs.view(-1, vocab), targets.view(-1))
        assert torch.allclose(loss, expected)

    def test_classification_preserves_gradient(self):
        """Loss should be differentiable for backprop."""
        from esper.utils.loss import compute_task_loss

        outputs = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "classification")
        loss.backward()

        assert outputs.grad is not None
        assert outputs.grad.shape == outputs.shape


class TestComputeTaskLossWithMetrics:
    """Tests for compute_task_loss_with_metrics function."""

    def test_returns_correct_types(self):
        """Should return (Tensor, Tensor, int) tuple.

        Note: correct is now a Tensor (not float) to enable deferred .item()
        calls and avoid CUDA sync overhead in hot paths.
        """
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(correct, torch.Tensor)
        assert isinstance(total, int)

    def test_correct_count_accuracy(self):
        """Should count correct predictions accurately."""
        from esper.utils.loss import compute_task_loss_with_metrics

        # Create outputs where we know the predictions
        outputs = torch.tensor([
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
        ])
        targets = torch.tensor([0, 1, 1, 0])  # 2 correct, 2 wrong
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert total == 4
        assert correct == 2.0

    def test_lm_correct_count(self):
        """Should count token-level correct predictions for LM."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(2, 4, 100)  # batch=2, seq=4, vocab=100
        targets = torch.randint(0, 100, (2, 4))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "lm"
        )

        assert total == 8  # 2 * 4 tokens
        assert 0 <= correct <= total

    def test_all_correct_classification(self):
        """Should return correct=total when all predictions are right."""
        from esper.utils.loss import compute_task_loss_with_metrics

        # Create outputs that strongly predict the correct class
        outputs = torch.tensor([
            [10.0, -10.0],  # Predicts 0
            [-10.0, 10.0],  # Predicts 1
            [10.0, -10.0],  # Predicts 0
        ])
        targets = torch.tensor([0, 1, 0])  # All match
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert correct == 3.0
        assert total == 3

    def test_all_wrong_classification(self):
        """Should return correct=0 when all predictions are wrong."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.tensor([
            [10.0, -10.0],  # Predicts 0
            [-10.0, 10.0],  # Predicts 1
        ])
        targets = torch.tensor([1, 0])  # All wrong
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert correct == 0.0
        assert total == 2

    def test_loss_matches_compute_task_loss(self):
        """Loss from with_metrics should match compute_task_loss."""
        from esper.utils.loss import compute_task_loss, compute_task_loss_with_metrics

        outputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss_only = compute_task_loss(outputs, targets, criterion, "classification")
        loss_with_metrics, _, _ = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert torch.allclose(loss_only, loss_with_metrics)

    def test_preserves_gradient(self):
        """Loss should be differentiable for backprop."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss, _, _ = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )
        loss.backward()

        assert outputs.grad is not None


class TestAsyncSafeMetrics:
    """Tests for async-safe (tensor return) behavior to avoid CUDA sync overhead.

    These tests verify that compute_task_loss_with_metrics returns a tensor
    for the correct count instead of a float, allowing deferred .item() calls.
    """

    def test_correct_is_tensor(self):
        """correct should be a Tensor, not float, to avoid .item() sync per batch."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert isinstance(correct, torch.Tensor), f"Expected Tensor, got {type(correct)}"
        assert correct.ndim == 0, "correct should be a scalar tensor"

    def test_correct_tensor_on_same_device_as_outputs(self):
        """correct tensor should stay on the same device as outputs."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(8, 10)  # CPU
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert correct.device == outputs.device

    def test_correct_tensor_value_matches_float(self):
        """correct.item() should give the same value as the old float return."""
        from esper.utils.loss import compute_task_loss_with_metrics

        # Deterministic outputs for predictable results
        outputs = torch.tensor([
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
        ])
        targets = torch.tensor([0, 1, 1, 0])  # 2 correct, 2 wrong
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert correct.item() == 2.0

    def test_correct_tensor_accumulation_pattern(self):
        """Verify tensor accumulation works for deferred sync pattern."""
        from esper.utils.loss import compute_task_loss_with_metrics

        criterion = nn.CrossEntropyLoss()

        # Simulate accumulation loop (the hot path optimization)
        running_correct = torch.zeros(1, dtype=torch.long)
        running_total = 0

        for _ in range(3):
            outputs = torch.randn(8, 10)
            targets = torch.randint(0, 10, (8,))

            loss, correct, total = compute_task_loss_with_metrics(
                outputs, targets, criterion, "classification"
            )

            # This is the optimized pattern: tensor += tensor (no .item())
            running_correct.add_(correct)
            running_total += total

        # Single sync at end
        assert running_total == 24
        assert 0 <= running_correct.item() <= 24

    def test_lm_correct_is_tensor(self):
        """LM task should also return tensor for correct count."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(2, 4, 100)  # batch=2, seq=4, vocab=100
        targets = torch.randint(0, 100, (2, 4))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "lm"
        )

        assert isinstance(correct, torch.Tensor), f"Expected Tensor, got {type(correct)}"
        assert correct.ndim == 0


class TestBackwardsCompatibility:
    """Tests ensuring the new functions match the old implementations."""

    def test_matches_simic_loss_and_correct(self):
        """Should produce identical results to simic._loss_and_correct."""
        from esper.utils.loss import compute_task_loss_with_metrics

        # Test classification
        outputs = torch.randn(16, 10)
        targets = torch.randint(0, 10, (16,))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        # Manual reimplementation of old logic
        old_loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        old_correct = float(predicted.eq(targets).sum().item())
        old_total = targets.size(0)

        assert torch.allclose(loss, old_loss)
        assert correct == old_correct
        assert total == old_total

    def test_matches_tolaria_compute_loss(self):
        """Should produce identical results to tolaria._compute_loss."""
        from esper.utils.loss import compute_task_loss

        # Test LM
        outputs = torch.randn(4, 8, 500)
        targets = torch.randint(0, 500, (4, 8))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "lm")

        # Manual reimplementation of old logic
        vocab = outputs.size(-1)
        old_loss = criterion(outputs.view(-1, vocab), targets.view(-1))

        assert torch.allclose(loss, old_loss)
