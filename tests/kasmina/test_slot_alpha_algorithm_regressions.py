"""Regression tests for alpha algorithm edge cases."""

from __future__ import annotations

import torch

from esper.kasmina.slot import SeedSlot
from esper.leyline.alpha import AlphaAlgorithm
from esper.leyline.stages import SeedStage


def test_gate_alpha_schedule_survives_blending_complete_and_checkpoint_roundtrip() -> None:
    slot = SeedSlot(slot_id="r0c0", channels=16, device="cpu")
    slot.germinate(
        blueprint_id="norm",
        seed_id="test-seed",
        blend_algorithm_id="gated",
        alpha_algorithm=AlphaAlgorithm.GATE,
    )
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=3)

    assert slot.state.alpha_algorithm == AlphaAlgorithm.GATE
    assert slot.alpha_schedule is not None

    old_stage = slot.state.stage
    ok = slot.state.transition(SeedStage.HOLDING)
    assert ok
    slot._on_enter_stage(SeedStage.HOLDING, old_stage)

    assert slot.state.stage == SeedStage.HOLDING
    assert slot.alpha_schedule is not None

    x = torch.randn(2, 16, 8, 8)
    y = slot(x)
    assert y.shape == x.shape

    # Checkpoint roundtrip: schedule must be reconstructed in non-BLENDING stages too.
    state_dict = slot.state_dict()

    restored = SeedSlot(slot_id="r0c0", channels=16, device="cpu")
    restored.germinate(
        blueprint_id="norm",
        seed_id="test-seed",
        blend_algorithm_id="gated",
        alpha_algorithm=AlphaAlgorithm.GATE,
    )
    restored.load_state_dict(state_dict)
    assert restored.state is not None
    assert restored.state.stage == SeedStage.HOLDING
    assert restored.state.alpha_algorithm == AlphaAlgorithm.GATE
    assert restored.alpha_schedule is not None
    restored_out = restored(x)
    assert restored_out.shape == x.shape


def test_force_alpha_host_only_preserves_gate_schedule() -> None:
    slot = SeedSlot(slot_id="r0c0", channels=16, device="cpu")
    slot.germinate(
        blueprint_id="norm",
        seed_id="test-seed",
        blend_algorithm_id="gated",
        alpha_algorithm=AlphaAlgorithm.GATE,
    )
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=3)

    assert slot.state.alpha_algorithm == AlphaAlgorithm.GATE
    assert slot.alpha_schedule is not None

    slot.set_alpha(0.7)
    x = torch.randn(2, 16, 8, 8)

    with slot.force_alpha(0.0):
        assert slot.alpha_schedule is not None
        out = slot(x)
        assert torch.allclose(out, x, atol=0.0, rtol=0.0)

    assert slot.alpha_schedule is not None


def test_germinate_multiply_applies_identity_init_for_conv_light() -> None:
    slot = SeedSlot(slot_id="r0c0", channels=16, device="cpu")
    slot.germinate(
        blueprint_id="conv_light",
        seed_id="test-seed",
        alpha_algorithm=AlphaAlgorithm.MULTIPLY,
    )

    assert slot.seed is not None
    x = torch.randn(2, 16, 8, 8)
    seed_out = slot.seed(x)
    assert torch.allclose(seed_out, x, atol=0.0, rtol=0.0)
