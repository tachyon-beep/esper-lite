"""Integration tests for BlendAction wiring.

Tests that BlendAction flows from action selection through to execution,
verifying the complete wiring from factored_actions → vectorized → host → slot → blending.
"""
import pytest
import torch

from esper.kasmina.blending import BlendCatalog, GatedBlend
from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
from esper.leyline.alpha import AlphaCurve
from esper.leyline.factored_actions import BlendAction, FactoredAction, LifecycleOp, BlueprintAction, TempoAction
from esper.tamiyo.policy.features import TaskConfig


class TestBlendAlgorithmCreation:
    """Test BlendCatalog creates correct algorithm types."""

    def test_linear_blend_creation(self):
        """LinearBlend created with total_steps."""
        blend = BlendCatalog.create("linear", total_steps=10)
        assert blend.algorithm_id == "linear"
        assert blend.get_alpha(5) == 0.5

    def test_sigmoid_blend_creation(self):
        """SigmoidBlend created with total_steps."""
        blend = BlendCatalog.create("sigmoid", total_steps=10)
        assert blend.algorithm_id == "sigmoid"
        # Sigmoid at midpoint should be close to 0.5
        assert 0.4 < blend.get_alpha(5) < 0.6

    def test_gated_blend_cnn_creation(self):
        """GatedBlend created with channels and topology=cnn."""
        blend = BlendCatalog.create("gated", channels=64, topology="cnn")
        assert isinstance(blend, GatedBlend)
        assert blend.topology == "cnn"

    def test_gated_blend_transformer_creation(self):
        """GatedBlend created with channels and topology=transformer."""
        blend = BlendCatalog.create("gated", channels=128, topology="transformer")
        assert isinstance(blend, GatedBlend)
        assert blend.topology == "transformer"


class TestGatedBlendForward:
    """Test GatedBlend works with both CNN and transformer tensors."""

    def test_gated_blend_cnn_forward(self):
        """GatedBlend produces valid alpha for CNN tensors."""
        blend = GatedBlend(channels=64, topology="cnn")
        x = torch.randn(2, 64, 8, 8)  # (B, C, H, W)
        alpha = blend.get_alpha_for_blend(x)
        assert alpha.shape == (2, 1, 1, 1)
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_gated_blend_transformer_forward(self):
        """GatedBlend produces valid alpha for transformer tensors."""
        blend = GatedBlend(channels=128, topology="transformer")
        x = torch.randn(2, 16, 128)  # (B, T, C)
        alpha = blend.get_alpha_for_blend(x)
        assert alpha.shape == (2, 1, 1)
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_linear_blend_unified_interface(self):
        """LinearBlend implements get_alpha_for_blend returning scalar tensor."""
        blend = BlendCatalog.create("linear", total_steps=10)
        blend.step(5)  # Set current step
        x = torch.randn(2, 64, 8, 8)
        alpha = blend.get_alpha_for_blend(x)
        assert alpha.shape == ()  # 0-dim tensor
        assert alpha.item() == pytest.approx(0.5)

    def test_sigmoid_blend_unified_interface(self):
        """SigmoidBlend implements get_alpha_for_blend returning scalar tensor."""
        blend = BlendCatalog.create("sigmoid", total_steps=10)
        blend.step(5)  # Midpoint
        x = torch.randn(2, 64, 8, 8)
        alpha = blend.get_alpha_for_blend(x)
        assert alpha.shape == ()  # 0-dim tensor
        assert 0.4 < alpha.item() < 0.6  # Sigmoid at midpoint ≈ 0.5


class TestBlendActionIntegration:
    """Test BlendAction flows through complete germination pipeline."""

    @pytest.mark.parametrize("algorithm", ["linear", "sigmoid"])
    def test_cnn_blend_algorithms(self, algorithm: str):
        """Each blend algorithm configures alpha control correctly for CNN."""
        config = TaskConfig.for_cifar10()
        host = CNNHost(num_classes=10, n_blocks=3)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=config)

        model.germinate_seed(
            "norm", f"test_seed_{algorithm}", slot="r0c1",
            blend_algorithm_id=algorithm,
        )

        # Verify blend algorithm stored
        slot = model.seed_slots["r0c1"]
        assert slot._blend_algorithm_id == algorithm

        # Trigger blending
        slot.start_blending(total_steps=10)

        # Phase 2: linear/sigmoid are curves for AlphaController (no alpha_schedule module)
        assert slot.alpha_schedule is None
        assert slot.state.alpha_controller.alpha_steps_total == 10
        expected_curve = AlphaCurve.LINEAR if algorithm == "linear" else AlphaCurve.SIGMOID
        assert slot.state.alpha_controller.alpha_curve == expected_curve

    @pytest.mark.parametrize("algorithm", ["linear", "sigmoid"])
    def test_transformer_blend_algorithms(self, algorithm: str):
        """Each blend algorithm configures alpha control for transformer topology."""
        config = TaskConfig.for_tinystories()
        host = TransformerHost(vocab_size=1000, n_layer=3, n_embd=64, n_head=4, num_segments=3)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=config)

        model.germinate_seed(
            "lora", f"test_seed_{algorithm}", slot="r0c1",
            blend_algorithm_id=algorithm,
        )

        slot = model.seed_slots["r0c1"]
        slot.start_blending(total_steps=10)

        assert slot.alpha_schedule is None
        assert slot.state.alpha_controller.alpha_steps_total == 10
        expected_curve = AlphaCurve.LINEAR if algorithm == "linear" else AlphaCurve.SIGMOID
        assert slot.state.alpha_controller.alpha_curve == expected_curve

    def test_gated_blend_cnn_integration(self):
        """GatedBlend creates learnable gate network for CNN."""
        config = TaskConfig.for_cifar10()
        host = CNNHost(num_classes=10, n_blocks=3)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=config)

        model.germinate_seed(
            "norm", "test_gated", slot="r0c1",
            blend_algorithm_id="gated",
        )

        slot = model.seed_slots["r0c1"]
        slot.start_blending(total_steps=10)

        assert isinstance(slot.alpha_schedule, GatedBlend)
        assert slot.alpha_schedule.topology == "cnn"
        # GatedBlend should have learnable parameters
        assert sum(p.numel() for p in slot.alpha_schedule.parameters()) > 0

    def test_gated_blend_transformer_integration(self):
        """GatedBlend creates learnable gate network for transformer."""
        config = TaskConfig.for_tinystories()
        host = TransformerHost(vocab_size=1000, n_layer=3, n_embd=64, n_head=4, num_segments=3)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=config)

        model.germinate_seed(
            "lora", "test_gated", slot="r0c1",
            blend_algorithm_id="gated",
        )

        slot = model.seed_slots["r0c1"]
        slot.start_blending(total_steps=10)

        assert isinstance(slot.alpha_schedule, GatedBlend)
        assert slot.alpha_schedule.topology == "transformer"


class TestFactoredActionBlendExtraction:
    """Test FactoredAction correctly extracts blend_algorithm_id."""

    def test_linear_blend_action(self):
        """BlendAction.LINEAR maps to 'linear'."""
        action = FactoredAction(
            slot_idx=1,  # Was SlotAction.MID
            blueprint=BlueprintAction.NORM,
            blend=BlendAction.LINEAR,
            tempo=TempoAction.STANDARD,
            op=LifecycleOp.GERMINATE,
        )
        assert action.blend_algorithm_id == "linear"

    def test_sigmoid_blend_action(self):
        """BlendAction.SIGMOID maps to 'sigmoid'."""
        action = FactoredAction(
            slot_idx=1,  # Was SlotAction.MID
            blueprint=BlueprintAction.NORM,
            blend=BlendAction.SIGMOID,
            tempo=TempoAction.STANDARD,
            op=LifecycleOp.GERMINATE,
        )
        assert action.blend_algorithm_id == "sigmoid"

    def test_gated_blend_action(self):
        """BlendAction.GATED maps to 'gated'."""
        action = FactoredAction(
            slot_idx=1,  # Was SlotAction.MID
            blueprint=BlueprintAction.NORM,
            blend=BlendAction.GATED,
            tempo=TempoAction.STANDARD,
            op=LifecycleOp.GERMINATE,
        )
        assert action.blend_algorithm_id == "gated"


class TestTransformerNoopBlueprint:
    """Test that noop blueprint is registered for transformer."""

    def test_noop_blueprint_exists(self):
        """Transformer topology should have noop blueprint."""
        from esper.kasmina.blueprints import BlueprintRegistry

        specs = BlueprintRegistry.list_for_topology("transformer")
        names = [s.name for s in specs]
        assert "noop" in names

    def test_noop_blueprint_is_identity(self):
        """Noop blueprint should be nn.Identity."""
        from esper.kasmina.blueprints import BlueprintRegistry
        import torch.nn as nn

        noop = BlueprintRegistry.create("transformer", "noop", dim=64)
        assert isinstance(noop, nn.Identity)

        # Identity should preserve input exactly
        x = torch.randn(2, 16, 64)
        assert torch.equal(noop(x), x)
