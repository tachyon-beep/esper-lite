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
