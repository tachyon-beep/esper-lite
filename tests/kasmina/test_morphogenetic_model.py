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

        assert "_legacy_single_slot" not in dir(model)


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


class TestFusedForwardAlphaShapeValidation:
    """Test fused_forward() alpha_override shape validation (B2-DRL-03)."""

    def test_cnn_correct_shape_accepted(self):
        """CNN topology should accept 4D alpha_override [B, 1, 1, 1]."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Correct CNN shape: [B, 1, 1, 1]
        x = torch.randn(4, 3, 32, 32)
        alpha = torch.full((4, 1, 1, 1), 0.5)

        # Should not raise
        out = model.fused_forward(x, {"r0c1": alpha})
        assert out.shape == (4, 10)

    def test_cnn_wrong_shape_rejected(self):
        """CNN topology should reject 3D alpha_override [B, 1, 1]."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Wrong shape: [B, 1, 1] (transformer shape for CNN topology)
        x = torch.randn(4, 3, 32, 32)
        alpha_wrong = torch.full((4, 1, 1), 0.5)

        with pytest.raises(ValueError, match="expected.*cnn topology"):
            model.fused_forward(x, {"r0c1": alpha_wrong})

    def test_transformer_correct_shape_accepted(self):
        """Transformer topology should accept 3D alpha_override [B, 1, 1]."""
        from esper.tamiyo.policy.features import TaskConfig

        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        task_config = TaskConfig(
            task_type="classification",
            topology="transformer",
            baseline_loss=2.3,
            target_loss=0.5,
            typical_loss_delta_std=0.1,
            max_epochs=100,
        )
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=task_config)
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Correct transformer shape: [B, 1, 1]
        x = torch.randint(0, 100, (4, 16))
        alpha = torch.full((4, 1, 1), 0.5)

        # Should not raise
        out = model.fused_forward(x, {"r0c1": alpha})
        assert out.shape == (4, 16, 100)

    def test_transformer_wrong_shape_rejected(self):
        """Transformer topology should reject 4D alpha_override [B, 1, 1, 1]."""
        from esper.tamiyo.policy.features import TaskConfig

        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        task_config = TaskConfig(
            task_type="classification",
            topology="transformer",
            baseline_loss=2.3,
            target_loss=0.5,
            typical_loss_delta_std=0.1,
            max_epochs=100,
        )
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=task_config)
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Wrong shape: [B, 1, 1, 1] (CNN shape for transformer topology)
        x = torch.randint(0, 100, (4, 16))
        alpha_wrong = torch.full((4, 1, 1, 1), 0.5)

        with pytest.raises(ValueError, match="expected.*transformer topology"):
            model.fused_forward(x, {"r0c1": alpha_wrong})

    def test_batch_size_mismatch_rejected(self):
        """Alpha with wrong batch size should be rejected."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Mismatched batch: input is 4, alpha is 2
        x = torch.randn(4, 3, 32, 32)
        alpha_wrong_batch = torch.full((2, 1, 1, 1), 0.5)

        with pytest.raises(ValueError, match="expected"):
            model.fused_forward(x, {"r0c1": alpha_wrong_batch})

    def test_empty_alpha_overrides_allowed(self):
        """Empty alpha_overrides dict should be allowed (no validation needed)."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        x = torch.randn(4, 3, 32, 32)

        # Empty dict - slots use their default alpha
        out = model.fused_forward(x, {})
        assert out.shape == (4, 10)

    def test_unknown_alpha_override_key_rejected(self):
        """Unknown keys in alpha_overrides should raise ValueError (typo protection)."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        x = torch.randn(4, 3, 32, 32)
        alpha = torch.full((4, 1, 1, 1), 0.5)

        # Typo: "r0cl" (letter L) instead of "r0c1" (number 1)
        with pytest.raises(ValueError, match="Unknown alpha_overrides keys"):
            model.fused_forward(x, {"r0cl": alpha})

    def test_transformer_without_task_config_uses_host_topology(self):
        """TransformerHost should use transformer topology even without task_config."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        # No task_config provided - topology comes from host
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        model.germinate_seed("noop", "seed_1", slot="r0c1")

        # Transformer shape: [B, 1, 1]
        x = torch.randint(0, 100, (4, 16))
        alpha = torch.full((4, 1, 1), 0.5)

        # Should work - host provides topology, not task_config
        out = model.fused_forward(x, {"r0c1": alpha})
        assert out.shape == (4, 16, 100)


class TestHostTopologyProperty:
    """Test that hosts expose topology property correctly."""

    def test_cnn_host_topology(self):
        """CNNHost should return 'cnn' topology."""
        host = CNNHost(num_classes=10)
        assert host.topology == "cnn"

    def test_transformer_host_topology(self):
        """TransformerHost should return 'transformer' topology."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        assert host.topology == "transformer"
