import torch


def test_alpha_schedule_protocol_contract():
    """All blend algorithms must satisfy AlphaScheduleProtocol.

    The protocol requires:
    - algorithm_id: str
    - total_steps: int
    - _current_step: int

    This contract is used by SeedSlot serialization (slot.py lines 2485-2489).
    """
    from esper.kasmina.blending import GatedBlend

    # Test all blend algorithm types
    blend_instances = [
        GatedBlend(channels=64, total_steps=10),
    ]

    for blend in blend_instances:
        # Verify all required attributes exist
        assert hasattr(blend, "algorithm_id"), \
            f"{type(blend).__name__} missing algorithm_id"
        assert hasattr(blend, "total_steps"), \
            f"{type(blend).__name__} missing total_steps"
        assert hasattr(blend, "_current_step"), \
            f"{type(blend).__name__} missing _current_step"

        # Verify attribute types
        assert isinstance(blend.algorithm_id, str), \
            f"{type(blend).__name__}.algorithm_id must be str"
        assert isinstance(blend.total_steps, int), \
            f"{type(blend).__name__}.total_steps must be int"
        assert isinstance(blend._current_step, int), \
            f"{type(blend).__name__}._current_step must be int"

        # Verify attributes are accessible (no AttributeError)
        _ = blend.algorithm_id
        _ = blend.total_steps
        _ = blend._current_step


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
    assert "gated" in available

    blend = BlendCatalog.create("gated", channels=64, total_steps=10)
    assert hasattr(blend, "algorithm_id")


def test_alpha_schedule_protocol_includes_get_alpha_for_blend():
    """AlphaScheduleProtocol must include get_alpha_for_blend method.

    Regression test: Protocol previously only defined attributes, forcing
    slot.py to use type: ignore[attr-defined] when calling the method.
    The Protocol should be "behavior-complete" - if runtime code calls it,
    it belongs in the Protocol.
    """
    from esper.kasmina.blending import AlphaScheduleProtocol, GatedBlend

    # Verify Protocol defines the method
    assert hasattr(AlphaScheduleProtocol, "get_alpha_for_blend"), (
        "AlphaScheduleProtocol must define get_alpha_for_blend method. "
        "Without this, callers need type: ignore to call the method."
    )

    # Verify GatedBlend satisfies the full protocol including the method
    blend = GatedBlend(channels=64, total_steps=10)
    assert hasattr(blend, "get_alpha_for_blend"), (
        "GatedBlend must implement get_alpha_for_blend"
    )
    assert callable(blend.get_alpha_for_blend), (
        "get_alpha_for_blend must be callable"
    )


def test_gated_blend_rejects_invalid_total_steps():
    """GatedBlend should fail fast on invalid total_steps.

    Regression test: total_steps was previously silently coerced via
    max(1, total_steps), masking configuration bugs. Invalid lifecycle
    parameters should raise immediately, not produce "instant completion"
    semantics that are painful to debug in long RL runs.
    """
    import pytest
    from esper.kasmina.blending import GatedBlend

    # Zero should raise
    with pytest.raises(ValueError, match="total_steps > 0"):
        GatedBlend(channels=64, total_steps=0)

    # Negative should raise
    with pytest.raises(ValueError, match="total_steps > 0"):
        GatedBlend(channels=64, total_steps=-1)

    # Valid value should work
    blend = GatedBlend(channels=64, total_steps=1)
    assert blend.total_steps == 1

    blend = GatedBlend(channels=64, total_steps=100)
    assert blend.total_steps == 100


def test_gated_blend_cnn_topology_rejects_transformer_input():
    """CNN topology should reject 3D transformer input."""
    import pytest
    from esper.kasmina.blending import GatedBlend

    gate = GatedBlend(channels=64, topology="cnn", total_steps=10)
    x_transformer = torch.randn(2, 16, 64)  # (B, T, C) - wrong for CNN

    with pytest.raises(AssertionError, match="CNN topology expects 4D input"):
        gate.get_alpha_for_blend(x_transformer)


def test_gated_blend_transformer_topology_rejects_cnn_input():
    """Transformer topology should reject 4D CNN input."""
    import pytest
    from esper.kasmina.blending import GatedBlend

    gate = GatedBlend(channels=64, topology="transformer", total_steps=10)
    x_cnn = torch.randn(2, 64, 8, 8)  # (B, C, H, W) - wrong for transformer

    with pytest.raises(AssertionError, match="Transformer topology expects 3D input"):
        gate.get_alpha_for_blend(x_cnn)
