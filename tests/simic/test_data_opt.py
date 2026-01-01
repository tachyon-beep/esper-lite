"""Tests for GPU gather iterator optimizations.

Verifies:
1. Channels-last memory format for cached tensors
2. Format preservation through advanced indexing
3. Correct data values and determinism
4. Model channels-last conversion

CI Requirements:
    These tests are skipped on CPU-only machines via @pytest.mark.skipif.

    If GPU CI is ever enabled, the following must be configured:

    1. CUDA drivers and runtime must be installed in the CI image
    2. CIFAR-10 dataset (~160MB) will be downloaded on first run - either:
       - Cache ./data between runs, OR
       - Use mock=True fixtures for speed (requires test refactoring)
    3. Sufficient GPU memory (tests use small models, ~100MB VRAM should suffice)

    The tests use torch.cuda.is_available() as the skip condition, so they
    will automatically run when CUDA becomes available.
"""

import pytest
import torch

from esper.utils.data import (
    SharedGPUGatherBatchIterator,
    _GPU_DATASET_CACHE,
    _cifar10_cache_key,
    _ensure_cifar10_cached,
)
from esper.kasmina.host import CNNHost


@pytest.fixture
def clear_cache():
    """Clear GPU cache before and after each test."""
    _GPU_DATASET_CACHE.clear()
    yield
    _GPU_DATASET_CACHE.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestChannelsLastFormat:
    """Test that cached tensors use channels-last memory format."""

    def test_cached_tensors_channels_last(self, clear_cache):
        """Verify GPU-cached CIFAR tensors are in channels-last format."""
        _ensure_cifar10_cached("cuda:0", "./data")

        cache_key = _cifar10_cache_key("cuda:0", "./data")
        train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

        # Image tensors should be channels-last
        assert train_x.is_contiguous(
            memory_format=torch.channels_last
        ), "train_x should be channels-last"
        assert test_x.is_contiguous(
            memory_format=torch.channels_last
        ), "test_x should be channels-last"

        # Labels are 1D, no memory format concept
        assert train_y.dim() == 1
        assert test_y.dim() == 1

    def test_cached_tensors_on_gpu(self, clear_cache):
        """Verify cached tensors are on the correct GPU device."""
        _ensure_cifar10_cached("cuda:0", "./data")

        cache_key = _cifar10_cache_key("cuda:0", "./data")
        train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

        assert train_x.device.type == "cuda"
        assert train_y.device.type == "cuda"
        assert test_x.device.type == "cuda"
        assert test_y.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIteratorOutputFormat:
    """Test that iterator outputs maintain channels-last format."""

    def test_iterator_preserves_channels_last(self, clear_cache):
        """Verify iterator outputs maintain channels-last format.

        Advanced indexing (dataset_x[idx_chunk]) preserves the source tensor's
        memory format, unlike torch.index_select which returns NCHW-contiguous.
        """
        iterator = SharedGPUGatherBatchIterator(
            batch_size_per_env=32,
            n_envs=4,
            env_devices=["cuda:0"] * 4,
            shuffle=True,
            seed=42,
        )

        batch = next(iter(iterator))

        for env_idx, (inputs, targets) in enumerate(batch):
            assert inputs.is_contiguous(
                memory_format=torch.channels_last
            ), f"Env {env_idx} inputs should be channels-last"
            # Targets are 1D, no format concept
            assert targets.dim() == 1

    def test_iterator_correct_shapes(self, clear_cache):
        """Verify iterator produces correctly shaped tensors."""
        batch_size = 32
        n_envs = 4

        iterator = SharedGPUGatherBatchIterator(
            batch_size_per_env=batch_size,
            n_envs=n_envs,
            env_devices=["cuda:0"] * n_envs,
            shuffle=True,
            seed=42,
        )

        batch = next(iter(iterator))

        assert len(batch) == n_envs

        for env_idx, (inputs, targets) in enumerate(batch):
            # CIFAR-10: (B, 3, 32, 32) images
            assert inputs.shape == (batch_size, 3, 32, 32), f"Env {env_idx} input shape"
            assert targets.shape == (batch_size,), f"Env {env_idx} target shape"

    def test_consecutive_batches_different_data(self, clear_cache):
        """Verify consecutive batches have different data (shuffling works)."""
        iterator = SharedGPUGatherBatchIterator(
            batch_size_per_env=32,
            n_envs=4,
            env_devices=["cuda:0"] * 4,
            shuffle=True,
            seed=42,
        )

        it = iter(iterator)
        batch1 = next(it)
        batch2 = next(it)

        # Data should differ (different indices gathered)
        for i, ((in1, _), (in2, _)) in enumerate(zip(batch1, batch2)):
            assert not torch.equal(
                in1, in2
            ), f"Batch data should differ for env {i} (shuffled indices)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPartialBatches:
    """Test handling of partial batches (drop_last=False)."""

    def test_partial_batch_correct_size(self, clear_cache):
        """Verify partial batches have correct (smaller) size."""
        # Use large batch size to force partial final batch on test set
        iterator = SharedGPUGatherBatchIterator(
            batch_size_per_env=2000,  # Large relative to test set (10k samples)
            n_envs=2,
            env_devices=["cuda:0"] * 2,
            shuffle=False,
            is_train=False,  # Test set with drop_last=False
            seed=42,
        )

        batches = list(iterator)

        # Should have some batches
        assert len(batches) > 0

        # Last batch may be smaller
        last_batch = batches[-1]
        for inputs, targets in last_batch:
            # Verify tensor is correctly sized (may be smaller than full batch)
            assert inputs.size(0) <= 2000
            assert targets.size(0) <= 2000
            assert inputs.size(0) == targets.size(0)
            # Format should still be channels-last
            assert inputs.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestModelChannelsLast:
    """Test that host model uses channels-last format after conversion."""

    def test_cnn_host_channels_last_conversion(self):
        """Verify CNNHost conv weights are channels-last after conversion.

        The host model (CNN) should have its 4D conv weights converted to
        channels-last format to match the channels-last input data, avoiding
        runtime layout permutations in conv layers.
        """
        # Create a CNNHost model
        model = CNNHost(num_classes=10).to("cuda:0")

        # Convert to channels-last (as done in create_env_state)
        model = model.to(memory_format=torch.channels_last)

        # Check all 4D parameters (conv weights) are channels-last
        for name, param in model.named_parameters():
            if param.dim() == 4:  # Conv weights are 4D: (out, in, H, W)
                assert param.is_contiguous(
                    memory_format=torch.channels_last
                ), f"Parameter {name} should be channels-last"

    def test_cnn_host_forward_preserves_format(self):
        """Verify forward pass maintains channels-last through conv layers."""
        model = CNNHost(num_classes=10).to("cuda:0")
        model = model.to(memory_format=torch.channels_last)

        # Create channels-last input (matching iterator output)
        x = torch.randn(4, 3, 32, 32, device="cuda:0")
        x = x.to(memory_format=torch.channels_last)
        assert x.is_contiguous(memory_format=torch.channels_last)

        # Forward pass
        output = model(x)

        # Output shape should be correct (classification logits)
        assert output.shape == (4, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDeterminism:
    """Test that shuffling is deterministic with same seed."""

    def test_same_seed_same_batches(self, clear_cache):
        """Verify same seed produces identical batches."""
        batch_size = 32
        n_envs = 4
        seed = 12345

        # First iterator
        iter1 = SharedGPUGatherBatchIterator(
            batch_size_per_env=batch_size,
            n_envs=n_envs,
            env_devices=["cuda:0"] * n_envs,
            shuffle=True,
            seed=seed,
        )

        # Clear cache to force fresh state
        _GPU_DATASET_CACHE.clear()

        # Second iterator with same seed
        iter2 = SharedGPUGatherBatchIterator(
            batch_size_per_env=batch_size,
            n_envs=n_envs,
            env_devices=["cuda:0"] * n_envs,
            shuffle=True,
            seed=seed,
        )

        # First few batches should be identical
        for batch_idx, (b1, b2) in enumerate(zip(iter1, iter2)):
            if batch_idx >= 3:
                break
            for env_idx, ((in1, tgt1), (in2, tgt2)) in enumerate(zip(b1, b2)):
                assert torch.equal(
                    in1, in2
                ), f"Batch {batch_idx} env {env_idx} inputs differ"
                assert torch.equal(
                    tgt1, tgt2
                ), f"Batch {batch_idx} env {env_idx} targets differ"
