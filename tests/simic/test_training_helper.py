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


class TestCompiledTrainStepCache:
    """Tests for _get_compiled_train_step caching behavior (B8-PT-03)."""

    def test_compile_retry_after_transient_failure(self, monkeypatch):
        """Verify compilation retries after transient failure instead of caching fallback.

        This tests the fix for B8-PT-03: if torch.compile fails once (e.g., GPU memory
        pressure), subsequent calls should retry compilation rather than being stuck
        with the uncompiled fallback for the process lifetime.
        """
        import esper.simic.training.helpers as helpers

        # Reset the module-level cache
        helpers._compiled_train_step_cache = None

        call_count = 0
        original_impl = helpers._train_step_impl

        def mock_compile(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated transient GPU memory pressure")
            # Second call succeeds - return the original impl as "compiled"
            return original_impl

        monkeypatch.setattr("torch.compile", mock_compile)

        # First call - fails, returns uncompiled fallback
        fn1 = helpers._get_compiled_train_step(use_compile=True)
        assert fn1 is original_impl, "Should return uncompiled impl on failure"
        assert helpers._compiled_train_step_cache is None, "Should NOT cache failures"

        # Second call - should retry and succeed
        fn2 = helpers._get_compiled_train_step(use_compile=True)
        assert helpers._compiled_train_step_cache is not None, "Should cache success"
        assert call_count == 2, "Should have retried compilation"

        # Third call - should use cached version, not recompile
        fn3 = helpers._get_compiled_train_step(use_compile=True)
        assert call_count == 2, "Should use cache, not recompile"
        assert fn3 is fn2, "Should return same cached callable"

    def test_use_compile_false_bypasses_compilation(self, monkeypatch):
        """When use_compile=False, should return uncompiled impl without trying."""
        import esper.simic.training.helpers as helpers

        helpers._compiled_train_step_cache = None
        compile_called = False

        def mock_compile(*args, **kwargs):
            nonlocal compile_called
            compile_called = True
            return args[0]

        monkeypatch.setattr("torch.compile", mock_compile)

        fn = helpers._get_compiled_train_step(use_compile=False)
        assert fn is helpers._train_step_impl
        assert not compile_called, "Should not attempt compilation when use_compile=False"

    def test_successful_compile_is_cached(self, monkeypatch):
        """Successful compilation should be cached and reused."""
        import esper.simic.training.helpers as helpers

        helpers._compiled_train_step_cache = None
        call_count = 0

        def mock_compile(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fn  # Return the function as "compiled"

        monkeypatch.setattr("torch.compile", mock_compile)

        # Multiple calls should only compile once
        helpers._get_compiled_train_step(use_compile=True)
        helpers._get_compiled_train_step(use_compile=True)
        helpers._get_compiled_train_step(use_compile=True)

        assert call_count == 1, "Should compile only once, then use cache"
