"""Tests for esper.utils package."""

from esper.utils import load_cifar10


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
