import torch


def test_linear_blend_schedule():
    """Linear blend should ramp alpha from 0 to 1 over steps."""
    from esper.kasmina.blending import LinearBlend

    blend = LinearBlend(total_steps=10)

    assert blend.get_alpha(0) == 0.0
    assert blend.get_alpha(5) == 0.5
    assert blend.get_alpha(10) == 1.0
    assert blend.get_alpha(15) == 1.0  # Clamp at 1.0


def test_sigmoid_blend_schedule():
    """Sigmoid blend should have smooth S-curve."""
    from esper.kasmina.blending import SigmoidBlend

    blend = SigmoidBlend(total_steps=10)

    # Sigmoid properties: starts slow, ends slow, fast in middle
    alpha_0 = blend.get_alpha(0)
    alpha_5 = blend.get_alpha(5)
    alpha_10 = blend.get_alpha(10)

    assert alpha_0 < 0.1, "Sigmoid should start near 0"
    assert 0.4 < alpha_5 < 0.6, "Sigmoid midpoint should be ~0.5"
    assert alpha_10 > 0.9, "Sigmoid should end near 1"


def test_gated_blend_schedule():
    """Gated blend should use learned gate."""
    from esper.kasmina.blending import GatedBlend

    for channels in (1, 4, 64):
        blend = GatedBlend(channels=channels)

        # Should have learnable parameters
        assert sum(p.numel() for p in blend.parameters()) > 0

        # Should produce valid alpha tensor via unified interface
        x = torch.randn(2, channels, 8, 8)
        alpha = blend.get_alpha_for_blend(x)
        assert alpha.shape == (2, 1, 1, 1)  # (batch, 1, 1, 1) for broadcasting
        assert (alpha >= 0).all() and (alpha <= 1).all()


def test_blend_registry():
    """BlendCatalog should list and create blends."""
    from esper.kasmina.blending import BlendCatalog

    available = BlendCatalog.list_algorithms()
    assert "linear" in available
    assert "sigmoid" in available
    assert "gated" in available

    blend = BlendCatalog.create("linear", total_steps=10)
    assert blend.get_alpha(5) == 0.5
