"""MorphogeneticModel tests for multi-slot architecture."""

import pytest
import torch

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel


class TestMorphogeneticModelMultiSlot:
    """Test MorphogeneticModel without legacy single-slot mode."""

    def test_requires_slots_parameter(self):
        """MorphogeneticModel must require explicit slots parameter."""
        host = CNNHost(num_classes=10)

        # Without slots parameter should raise
        with pytest.raises(TypeError):
            MorphogeneticModel(host, device="cpu")

    def test_creates_specified_slots(self):
        """MorphogeneticModel should create only the requested slots."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1", "r0c2"])

        assert "r0c1" in model.seed_slots
        assert "r0c2" in model.seed_slots
        assert "r0c0" not in model.seed_slots
        assert len(model.seed_slots) == 2

    def test_no_seed_slot_property(self):
        """MorphogeneticModel should NOT have seed_slot property."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

        # Accessing seed_slot should raise AttributeError
        with pytest.raises(AttributeError):
            _ = model.seed_slot

    def test_no_seed_state_property(self):
        """MorphogeneticModel should NOT have seed_state property."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

        # Accessing seed_state should raise AttributeError
        with pytest.raises(AttributeError):
            _ = model.seed_state

    def test_germinate_requires_slot(self):
        """germinate_seed must require explicit slot parameter."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

        # Without slot should raise TypeError
        with pytest.raises(TypeError):
            model.germinate_seed("noop", "seed_1")

    def test_cull_requires_slot(self):
        """cull_seed must require explicit slot parameter."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Without slot should raise TypeError
        with pytest.raises(TypeError):
            model.cull_seed()

    def test_forward_with_cnn_host(self):
        """Forward pass should work with CNNHost multi-slot."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        x = torch.randn(2, 3, 32, 32)

        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_with_transformer_host(self):
        """Forward pass should work with TransformerHost multi-slot."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        x = torch.randint(0, 100, (2, 16))

        out = model(x)
        assert out.shape == (2, 16, 100)

    def test_no_legacy_single_slot_attribute(self):
        """MorphogeneticModel should not have _legacy_single_slot attribute."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

        assert not hasattr(model, "_legacy_single_slot")
