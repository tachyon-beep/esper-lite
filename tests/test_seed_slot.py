"""Tests for SeedSlot behavior."""

import torch


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
