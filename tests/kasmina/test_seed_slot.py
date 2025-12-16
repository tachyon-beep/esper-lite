"""Tests for SeedSlot behavior."""

import pytest
import torch
import torch.nn as nn
from esper.leyline import SeedStage
from esper.kasmina.slot import SeedMetrics, SeedState


def test_seed_slot_forward_no_seed_identity():
    """SeedSlot forward returns input unchanged when no seed."""
    from esper.kasmina.slot import SeedSlot

    slot = SeedSlot(slot_id="test", channels=64)

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    assert torch.allclose(out, x)


def test_seed_slot_forward_dormant_identity():
    """SeedSlot forward returns identity for DORMANT stage."""
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import SeedStage

    slot = SeedSlot(slot_id="test", channels=64)
    slot.germinate("norm", "test-seed")

    slot.state.stage = SeedStage.DORMANT

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    assert torch.allclose(out, x)


def test_seed_slot_forward_with_seed():
    """SeedSlot forward applies seed transformation."""
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import SeedStage

    slot = SeedSlot(slot_id="test", channels=64)
    slot.germinate("norm", "test-seed")

    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    x = torch.randn(2, 64, 8, 8)
    out = slot.forward(x)

    assert not torch.allclose(out, x)


def test_germinate_cnn_shape_validation_host_agnostic():
    """CNN seeds validate shape without touching host BatchNorm."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.control import TaskConfig

    config = TaskConfig.for_cifar10()

    slot = SeedSlot(
        slot_id="block2_post",
        channels=64,
        task_config=config,
    )

    state = slot.germinate("norm", "cnn-seed")
    assert state is not None
    assert slot.seed is not None

    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        y = slot.seed(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


def test_germinate_transformer_shape_validation_host_agnostic():
    """Transformer seeds validate shape without host-specific helpers."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.control import TaskConfig

    config = TaskConfig.for_tinystories()
    dim = 64

    slot = SeedSlot(
        slot_id="layer_0_post_block",
        channels=dim,
        fast_mode=True,
        task_config=config,
    )

    state = slot.germinate("lora", "transformer-seed")
    assert state is not None
    assert slot.seed is not None

    x = torch.randn(2, 4, dim)
    with torch.no_grad():
        y = slot.seed(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


def test_germinate_cnn_shape_mismatch_raises_assertion():
    """CNN blueprints that change feature shape must fail germinate."""
    from esper.kasmina.slot import SeedSlot
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.simic.control import TaskConfig

    @BlueprintRegistry.register(
        name="__test_bad_cnn_shape__",
        topology="cnn",
        param_estimate=1,
        description="test-only: deliberately changes spatial shape",
    )
    def _bad_cnn_blueprint(dim: int) -> nn.Module:
        class BadCNNSeed(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.channels = channels

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Halve spatial resolution to force a shape mismatch.
                return x[:, :, ::2, ::2]

        return BadCNNSeed(dim)

    try:
        config = TaskConfig.for_cifar10()
        slot = SeedSlot(
            slot_id="block2_post",
            channels=64,
            task_config=config,
        )

        with pytest.raises(AssertionError, match="changed shape"):
            slot.germinate("__test_bad_cnn_shape__", "cnn-bad-seed")
    finally:
        BlueprintRegistry.unregister("cnn", "__test_bad_cnn_shape__")


def test_germinate_transformer_shape_mismatch_raises_assertion():
    """Transformer blueprints that change embedding dim must fail germinate."""
    from esper.kasmina.slot import SeedSlot
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.simic.control import TaskConfig

    @BlueprintRegistry.register(
        name="__test_bad_transformer_shape__",
        topology="transformer",
        param_estimate=1,
        description="test-only: deliberately changes embedding dimension",
    )
    def _bad_transformer_blueprint(dim: int) -> nn.Module:
        class BadTransformerSeed(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.proj = nn.Linear(d, d // 2, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Change last dimension to violate shape invariants.
                b, t, d = x.shape
                return self.proj(x.view(b * t, d)).view(b, t, -1)

        return BadTransformerSeed(dim)

    try:
        config = TaskConfig.for_tinystories()
        dim = 64
        slot = SeedSlot(
            slot_id="layer_0_post_block",
            channels=dim,
            fast_mode=True,
            task_config=config,
        )

        with pytest.raises(AssertionError, match="changed shape"):
            slot.germinate("__test_bad_transformer_shape__", "transformer-bad-seed")
    finally:
        BlueprintRegistry.unregister("transformer", "__test_bad_transformer_shape__")


class TestG5RequiresCounterfactual:
    """G5 gate must require counterfactual - no fallback to total_improvement."""

    def test_g5_fails_without_counterfactual(self):
        """G5 should fail if counterfactual_contribution is None."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
        )
        # Set high total_improvement but NO counterfactual
        state.metrics.initial_val_accuracy = 50.0
        state.metrics.current_val_accuracy = 60.0  # 10% total improvement
        state.metrics.counterfactual_contribution = None  # No counterfactual!

        result = gates._check_g5(state)

        assert not result.passed, "G5 should fail without counterfactual"
        assert "counterfactual_not_available" in result.checks_failed

    def test_g5_passes_with_positive_counterfactual(self):
        """G5 should pass with positive counterfactual contribution."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = 2.5  # Positive contribution

        result = gates._check_g5(state)

        assert result.passed, f"G5 should pass: {result.checks_failed}"
        assert "sufficient_contribution" in str(result.checks_passed)

    def test_g5_fails_with_negative_counterfactual(self):
        """G5 should fail with negative counterfactual contribution."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = -1.0  # Negative!

        result = gates._check_g5(state)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_fails_with_zero_counterfactual(self):
        """G5 should fail with zero counterfactual contribution (no value added)."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = 0.0  # Zero = no value added

        result = gates._check_g5(state)

        assert not result.passed, "Zero contribution should not pass G5"
        assert any("insufficient_contribution" in c for c in result.checks_failed)


def test_seed_state_report_includes_telemetry_fields():
    """SeedStateReport should carry causal/gradient/param metrics from SeedMetrics."""
    metrics = SeedMetrics()
    metrics.counterfactual_contribution = 3.2
    metrics.seed_gradient_norm_ratio = 0.12
    metrics.seed_param_count = 123
    metrics.host_param_count = 456
    metrics.current_alpha = 0.5
    metrics.alpha_ramp_step = 7

    state = SeedState(
        seed_id="seed-1",
        blueprint_id="bp-1",
        slot_id="r0c1",
        stage=SeedStage.TRAINING,
        metrics=metrics,
    )

    report = state.to_report()
    assert report.metrics.counterfactual_contribution == 3.2
    assert report.metrics.seed_gradient_norm_ratio == 0.12
    assert report.metrics.seed_param_count == 123
    assert report.metrics.host_param_count == 456
    assert report.metrics.current_alpha == 0.5
    assert report.metrics.alpha_ramp_step == 7


class TestShapeProbeCacheDeviceTransfer:
    """Shape probe cache should be cleared on device transfer."""

    def test_cache_only_cleared_on_device_change(self):
        """Cache should only be cleared when device actually changes."""
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Warm the cache
        probe1 = slot._get_shape_probe("cnn")
        probe2 = slot._get_shape_probe("transformer")
        assert len(slot._shape_probe_cache) == 2, "Cache should have 2 entries"

        # Transfer to SAME device - cache should NOT be cleared (optimization)
        slot.to("cpu")
        assert len(slot._shape_probe_cache) == 2, "Cache should persist on same-device transfer"

        # Verify cached probes are still the same objects
        assert slot._get_shape_probe("cnn") is probe1
        assert slot._get_shape_probe("transformer") is probe2

    def test_to_returns_self(self):
        """The .to() method should return self for chaining."""
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        result = slot.to("cpu")

        assert result is slot, ".to() should return self"

    def test_to_updates_device(self):
        """The .to() method should update self.device."""
        import torch
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot.to("cpu")  # No-op but should still work

        assert slot.device == torch.device("cpu")


def test_morphogenetic_model_to_device_consistency():
    """Verify device transfer doesn't cause inconsistencies."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])

    # Germinate a seed
    model.germinate_seed("norm", "test-seed", slot="r0c1")
    assert model.seed_slots["r0c1"].seed is not None

    # Transfer to CPU (no-op but exercises the code path)
    model = model.to("cpu")

    # Verify consistency
    assert str(model._device) == "cpu"
    assert model.seed_slots["r0c1"].device == torch.device("cpu")

    # Verify seed is on correct device
    seed_param = next(model.seed_slots["r0c1"].seed.parameters())
    assert seed_param.device == torch.device("cpu")


def test_gradient_health_monitor_batch_sync():
    """Verify compute_gradient_health works correctly with batched computation."""
    from esper.kasmina.isolation import GradientHealthMonitor

    monitor = GradientHealthMonitor()

    # Create simple modules
    host = torch.nn.Linear(10, 10)
    seed = torch.nn.Linear(10, 10)

    monitor.register(host, seed)

    # Simulate gradients
    for p in host.parameters():
        p.grad = torch.randn_like(p) * 0.01
    for p in seed.parameters():
        p.grad = torch.randn_like(p)

    metrics = monitor.compute_gradient_health()

    # Should compute gradient norms for health monitoring
    assert metrics["host_grad_norm"] > 0
    assert metrics["seed_grad_norm"] > 0
    assert "seed_gradient_ratio" in metrics


def test_shape_probe_cache_device_comparison():
    """Shape probe cache should use direct device comparison."""
    from esper.kasmina.slot import SeedSlot

    slot = SeedSlot("test", channels=64, device="cpu")

    # Get probe - should create and cache
    probe1 = slot._get_shape_probe("cnn")
    assert probe1.device == torch.device("cpu")

    # Get again - should return cached
    probe2 = slot._get_shape_probe("cnn")
    assert probe1 is probe2  # Same object

    # Different topology - should create new
    probe3 = slot._get_shape_probe("transformer")
    assert probe3.device == torch.device("cpu")
    assert probe1 is not probe3
