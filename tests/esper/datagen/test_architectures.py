"""Tests for architecture factory."""

import pytest
import torch
from esper.datagen.architectures import create_model, SUPPORTED_ARCHITECTURES


class TestArchitectureFactory:
    def test_supported_architectures(self):
        expected = [
            "HostCNN", "HostCNN-Wide", "HostCNN-Deep",
            "ResNet-18", "ResNet-34"
        ]
        assert set(SUPPORTED_ARCHITECTURES) == set(expected)

    def test_create_hostcnn(self):
        model = create_model("HostCNN", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_create_hostcnn_wide(self):
        model = create_model("HostCNN-Wide", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)
        # Wide should have more parameters
        hostcnn = create_model("HostCNN", num_classes=10)
        assert sum(p.numel() for p in model.parameters()) > sum(p.numel() for p in hostcnn.parameters())

    def test_create_hostcnn_deep(self):
        model = create_model("HostCNN-Deep", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_create_resnet18(self):
        model = create_model("ResNet-18", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_create_resnet34(self):
        model = create_model("ResNet-34", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("InvalidArch", num_classes=10)
