"""Tests for SeedSlot behavior."""

import pytest
import torch
import torch.nn as nn


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
    from esper.simic.features import TaskConfig

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
    from esper.simic.features import TaskConfig

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
    from esper.simic.features import TaskConfig

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
    from esper.simic.features import TaskConfig

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
