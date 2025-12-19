"""Tests for esper.utils package."""

import torch
from torch.utils.data import TensorDataset

from esper.utils import load_cifar10
from esper.utils.data import SharedBatchIterator


class TestData:
    """Tests for utils.data module."""

    def test_load_cifar10_returns_loaders(self):
        """Test CIFAR-10 loading returns DataLoaders."""
        trainloader, testloader = load_cifar10(batch_size=32, mock=True)
        assert len(trainloader) > 0
        assert len(testloader) > 0

    def test_load_cifar10_batch_size(self):
        """Test CIFAR-10 respects batch size."""
        trainloader, _ = load_cifar10(batch_size=64, mock=True)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape[0] == 64
        assert labels.shape[0] == 64

    def test_load_cifar10_data_shape(self):
        """Test CIFAR-10 data has correct shape."""
        trainloader, _ = load_cifar10(batch_size=32, mock=True)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape == (32, 3, 32, 32)  # CIFAR-10 is 32x32 RGB

    def test_shared_batch_iterator_retains_env_splits_on_partial_batch(self):
        """SharedBatchIterator should keep one batch per env on partial batches."""
        dataset = TensorDataset(
            torch.arange(10).float().unsqueeze(1),
            torch.arange(10),
        )
        iterator = SharedBatchIterator(
            dataset=dataset,
            batch_size_per_env=2,
            n_envs=3,
            env_devices=["cpu"] * 3,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        batches = list(iter(iterator))
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert [batch[0].shape[0] for batch in batches[0]] == [2, 2, 2]
        assert [batch[0].shape[0] for batch in batches[1]] == [2, 1, 1]
