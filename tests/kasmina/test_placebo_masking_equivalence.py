"""Masking-equivalence check for the PIN-E null player (drl Rec3, plan §2.6 item 6).

The offline φ assembly measures a HOLDING placebo through the fused val pass's
alpha_override tensors, while the live committed-Shapley term will measure
FOSSILIZED seeds through their live scalar-alpha forward. tau transfers between
the two regimes only if the blend paths agree numerically for a null player.

These tests pin the ENABLED-config leg of that equivalence:

  (1) HOLDING forward via alpha_override=1.0 (broadcast [B,1,1,1] lerp weight —
      the fused-pass path) == HOLDING forward via the cached 0-dim scalar alpha
      tensor (the live path), bit-identically.
  (2) The same slot forced to FOSSILIZED produces the identical live forward —
      so a fossil's contribution is the exact quantity the HOLDING measurement saw.

The DISABLED-config leg (alpha_override=0 == host-only == never germinated, at
every active stage incl. FOSSILIZED) is already pinned by
test_committed_shapley_kernel_masking.py; together the two files close Rec3.

CPU-only. These verify EXISTING behavior mandated as a check by the plan; a
failure here is a real finding (tau would not transfer to the fossil regime),
not a test bug.
"""

import torch

from esper.kasmina.slot import SeedSlot
from esper.leyline import AlphaAlgorithm, SeedStage


def _holding_placebo_slot(channels: int = 16) -> SeedSlot:
    """Standalone slot with a placebo seed walked to HOLDING at full amplitude."""
    slot = SeedSlot(slot_id="r0c0", channels=channels)
    slot.germinate("placebo", seed_id="s", alpha_algorithm=AlphaAlgorithm.ADD)
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.state.transition(SeedStage.HOLDING)
    slot.set_alpha(1.0)
    return slot


def test_holding_override_path_equals_live_scalar_path() -> None:
    """alpha_override=1.0 (fused-pass broadcast weight) == live cached-scalar forward.

    This is the exact seam between the offline measurement (override tensor) and
    the live forward (0-dim cached alpha tensor): both must dispatch torch.lerp
    to bit-identical results at the same alpha.
    """
    torch.manual_seed(0)
    b, c, h, w = 4, 16, 8, 8
    slot = _holding_placebo_slot(channels=c)
    host = torch.randn(b, c, h, w)

    with torch.inference_mode():
        via_override = slot(host, alpha_override=torch.ones(b, 1, 1, 1))
        via_live_scalar = slot(host)

    assert torch.isfinite(via_override).all()
    assert torch.equal(via_override, via_live_scalar), (
        "HOLDING forward must be bit-identical between the alpha_override tensor "
        "path and the cached scalar-alpha path at alpha=1.0"
    )
    # Non-vacuous: the placebo is near-inert but NOT identity — it genuinely
    # perturbs the host stream, so the equality above compares real contributions.
    assert not torch.equal(via_live_scalar, host), (
        "placebo at alpha=1.0 must move the output (near-inert, not identity)"
    )


def test_fossilized_gate_override_one_equals_live_forward() -> None:
    """FOSSILIZED GATE via alpha_override=1.0 == the live natural forward.

    Criterion 6(c) extension (esper-lite-fbeead4efc): admitting GATE fossilized
    slots to the coalition family means their member-present configs travel the
    tensor-override blend path. That measurement is honest only if the override
    path reproduces the deployed forward bit-identically — the same
    "numerically inert" standard already verified for non-GATE fossils. The
    trained gate module is shared by both paths; only the amplitude's tensor
    form (broadcast override vs cached scalar) differs.
    """
    torch.manual_seed(2)
    b, c, h, w = 4, 16, 8, 8
    slot = SeedSlot(slot_id="r0c0", channels=c)
    slot.germinate(
        "placebo",
        seed_id="s",
        blend_algorithm_id="gated",
        alpha_algorithm=AlphaAlgorithm.GATE,
    )
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.start_blending(total_steps=5)
    slot.state.transition(SeedStage.HOLDING)
    slot.set_alpha(1.0)
    # Match existing kasmina test idiom: force the terminal stage directly;
    # weights, gate module, and alpha are untouched by the stage flip.
    slot.state.stage = SeedStage.FOSSILIZED
    host = torch.randn(b, c, h, w)

    with torch.inference_mode():
        via_override = slot(host, alpha_override=torch.ones(b, 1, 1, 1))
        live = slot(host)

    assert torch.isfinite(via_override).all()
    assert torch.equal(via_override, live), (
        "FOSSILIZED GATE forward must be bit-identical between the fused-pass "
        "alpha_override tensor path and the live cached-scalar path at alpha=1.0 "
        "— otherwise coalition member-present configs measure a different "
        "quantity than the deployed contribution"
    )


def test_fossilized_live_forward_equals_holding_measurement() -> None:
    """The same seed FOSSILIZED yields the identical live forward it had at HOLDING.

    Offline φ measures the placebo while HOLDING (via override tensors); the live
    term will see fossils via their scalar-alpha forward. For tau to transfer, the
    fossil forward must be the exact quantity the HOLDING measurement captured.
    """
    torch.manual_seed(1)
    b, c, h, w = 4, 16, 8, 8
    slot = _holding_placebo_slot(channels=c)
    host = torch.randn(b, c, h, w)

    with torch.inference_mode():
        holding_via_override = slot(host, alpha_override=torch.ones(b, 1, 1, 1))
        holding_live = slot(host)

    # Force the terminal stage directly (matches existing kasmina test idiom);
    # weights and alpha are untouched by the stage flip.
    slot.state.stage = SeedStage.FOSSILIZED

    with torch.inference_mode():
        fossil_live = slot(host)

    assert torch.equal(fossil_live, holding_live), (
        "FOSSILIZED live forward must be bit-identical to the HOLDING live forward "
        "for unchanged weights and alpha"
    )
    assert torch.equal(fossil_live, holding_via_override), (
        "FOSSILIZED live forward must be bit-identical to the HOLDING "
        "alpha_override=1.0 measurement — otherwise tau does not transfer from "
        "the placebo (HOLDING) regime to the live fossil regime"
    )
