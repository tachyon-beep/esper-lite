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
        """prune_seed must require explicit slot parameter."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Without slot should raise TypeError
        with pytest.raises(TypeError):
            model.prune_seed()

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


class TestMorphogeneticModelDynamicSlots:
    """Test MorphogeneticModel with variable numbers of injection points."""

    def test_cnn_host_5_blocks(self):
        """MorphogeneticModel should work with 5-block CNNHost."""
        host = CNNHost(num_classes=10, n_blocks=5)
        # 5-block host has 5 injection points: r0c0, r0c1, r0c2, r0c3, r0c4
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1", "r0c3"])

        # Verify slots were created
        assert len(model.seed_slots) == 2
        assert "r0c1" in model.seed_slots
        assert "r0c3" in model.seed_slots

        # Verify forward pass works
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cnn_host_5_blocks_all_slots(self):
        """MorphogeneticModel should support all 5 slots in 5-block CNNHost."""
        host = CNNHost(num_classes=10, n_blocks=5)
        all_slots = ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]
        model = MorphogeneticModel(host, device="cpu", slots=all_slots)

        # Verify all slots were created
        assert len(model.seed_slots) == 5
        for slot_id in all_slots:
            assert slot_id in model.seed_slots

        # Verify forward pass works with all slots
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_transformer_host_4_segments(self):
        """MorphogeneticModel should work with 4-segment TransformerHost."""
        host = TransformerHost(
            vocab_size=100, n_embd=64, n_head=2, n_layer=8,
            block_size=32, num_segments=4
        )
        # 4-segment host has 4 injection points: r0c0, r0c1, r0c2, r0c3
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c2"])

        # Verify slots were created
        assert len(model.seed_slots) == 2
        assert "r0c0" in model.seed_slots
        assert "r0c2" in model.seed_slots

        # Verify forward pass works
        x = torch.randint(0, 100, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 100)

    def test_unknown_slot_raises_error(self):
        """Requesting a slot not in host's injection specs should raise ValueError."""
        host = CNNHost(num_classes=10, n_blocks=3)
        # 3-block host has r0c0, r0c1, r0c2 only

        with pytest.raises(ValueError, match="Unknown slot: r0c5"):
            MorphogeneticModel(host, device="cpu", slots=["r0c5"])

    def test_slot_order_matches_host_specs(self):
        """MorphogeneticModel._slot_order should match host injection_specs order."""
        host = CNNHost(num_classes=10, n_blocks=4)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1", "r0c3"])

        # Verify slot order matches host specs
        expected_order = [spec.slot_id for spec in host.injection_specs()]
        assert model._slot_order == expected_order
        assert expected_order == ["r0c0", "r0c1", "r0c2", "r0c3"]

        # Verify active slots are filtered correctly
        assert model._active_slots == ["r0c1", "r0c3"]
