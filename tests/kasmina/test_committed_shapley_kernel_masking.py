"""Kernel-verification tests for the Committed-Shapley coalition evaluation (WI-9a).

These pin the two masking properties the coalition value v(S) relies on:

  (1) SeedSlot.forward(host_features, alpha_override=0) returns the host features
      BIT-IDENTICALLY for a slot with an active (germinated) seed — at every active
      stage (TRAINING/BLENDING/HOLDING/FOSSILIZED) and for BOTH blend branches
      (ADD = torch.lerp, MULTIPLY = multiplicative valve).

  (2) At the fused / multi-slot level, masking a slot to alpha=0.0 is BIT-IDENTICAL
      to that slot's seed never having germinated (physically detached), while the
      OTHER slot is held in >= 2 distinct coalition contexts (active at its natural
      alpha AND masked to 0.0). This is the exact property that makes v(S), v({s})
      and v(empty) well defined: alpha_override=0 removes a slot's contribution
      EXACTLY, for arbitrary coalitions.

pytorch F6 nuances honored (see plan WI-9):
  - Exact equality is valid ONLY for FINITE seed features (0*inf = nan). Finite
    synthetic inputs are used and finiteness is asserted on the masked output.
  - The masked path is the BLEND branch, NOT the STE branch. Passing an
    alpha_override tensor forces the blend branch — STE fires only when
    alpha_override is None. We never assert the seed forward is skipped: it runs and
    its output is zeroed. The meaningfulness assertions (the seed genuinely changes
    the output at its natural alpha) make the equality non-vacuous rather than a
    trivial pass-through.

CPU-only, no CUDA requirement.
"""

import pytest
import torch

from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.kasmina.slot import SeedSlot
from esper.leyline import AlphaAlgorithm, SeedStage

# The four stages for which is_active_stage() is True — a seed in any of these runs
# its forward and participates in the blend (leyline/stages.py:84-91).
ACTIVE_STAGES = (
    SeedStage.TRAINING,
    SeedStage.BLENDING,
    SeedStage.HOLDING,
    SeedStage.FOSSILIZED,
)

NATURAL_ALPHA = 0.7


def _advance_to(slot: SeedSlot, stage: SeedStage) -> None:
    """Walk a freshly germinated (GERMINATED) slot to the requested active stage."""
    slot.state.transition(SeedStage.TRAINING)
    if stage in (SeedStage.BLENDING, SeedStage.HOLDING, SeedStage.FOSSILIZED):
        slot.state.transition(SeedStage.BLENDING)
    if stage in (SeedStage.HOLDING, SeedStage.FOSSILIZED):
        slot.state.transition(SeedStage.HOLDING)
    if stage == SeedStage.FOSSILIZED:
        # Match existing kasmina tests: force the terminal stage directly.
        slot.state.stage = SeedStage.FOSSILIZED


def _make_nonidentity_norm(seed: torch.nn.Module) -> None:
    """Turn a freshly germinated 'norm' seed into a genuinely non-identity function.

    The NormSeed forward is ``x + tanh(scale) * (norm(x) - x)`` with ``scale``
    zero-initialised (identity at birth). Setting ``scale = 1.0`` gives a finite,
    deterministic transform whose output differs from the host stream — required so
    the masking equality is meaningful, not a vacuous identity pass-through.
    """
    seed.scale.data.fill_(1.0)


def _active_norm_slot(
    *,
    channels: int,
    stage: SeedStage,
    algo: AlphaAlgorithm,
) -> SeedSlot:
    """Build a standalone SeedSlot with an active, non-identity 'norm' seed."""
    slot = SeedSlot(slot_id="r0c0", channels=channels)
    slot.germinate("norm", seed_id="s", alpha_algorithm=algo)
    _advance_to(slot, stage)
    _make_nonidentity_norm(slot.seed)
    slot.set_alpha(NATURAL_ALPHA)
    return slot


# ---------------------------------------------------------------------------
# (1) Slot-level exact zeroing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ACTIVE_STAGES)
def test_slot_alpha_override_zero_returns_host_bit_identical_add(stage: SeedStage) -> None:
    """alpha_override=0 (ADD/lerp branch) => output == host features bit-identically."""
    torch.manual_seed(0)
    b, c, h, w = 4, 16, 8, 8
    slot = _active_norm_slot(channels=c, stage=stage, algo=AlphaAlgorithm.ADD)
    host = torch.randn(b, c, h, w)

    masked = slot(host, alpha_override=torch.zeros(b, 1, 1, 1))
    natural = slot(host, alpha_override=torch.full((b, 1, 1, 1), NATURAL_ALPHA))

    # Exact zeroing: masked output is the host stream, unchanged, to the last bit.
    assert torch.isfinite(masked).all(), "masked output must be finite for exact equality"
    assert torch.equal(masked, host), (
        f"alpha_override=0 must return host features bit-identically (stage={stage.name})"
    )
    # Meaningfulness: at its natural alpha the seed genuinely moves the output, so the
    # exact-zeroing result above is not a trivial identity-seed pass-through.
    assert not torch.equal(natural, host), (
        f"seed must be non-identity at natural alpha (stage={stage.name})"
    )


@pytest.mark.parametrize("stage", (SeedStage.TRAINING, SeedStage.BLENDING, SeedStage.FOSSILIZED))
def test_slot_alpha_override_zero_returns_host_bit_identical_multiply(stage: SeedStage) -> None:
    """alpha_override=0 (MULTIPLY valve branch) => output == host features bit-identically.

    MULTIPLY is ``h * (1 + a*tanh(m))``; at a=0 the multiplier collapses to 1 exactly.
    """
    torch.manual_seed(1)
    b, c, h, w = 4, 16, 8, 8
    slot = _active_norm_slot(channels=c, stage=stage, algo=AlphaAlgorithm.MULTIPLY)
    host = torch.randn(b, c, h, w)

    masked = slot(host, alpha_override=torch.zeros(b, 1, 1, 1))
    natural = slot(host, alpha_override=torch.full((b, 1, 1, 1), NATURAL_ALPHA))

    assert torch.isfinite(masked).all()
    assert torch.equal(masked, host), (
        f"MULTIPLY alpha_override=0 must return host features bit-identically (stage={stage.name})"
    )
    assert not torch.equal(natural, host), (
        f"MULTIPLY seed must be non-identity at natural alpha (stage={stage.name})"
    )


def test_slot_zeroing_is_blend_branch_not_ste() -> None:
    """A TRAINING slot with alpha==0 STILL zeroes exactly when an override tensor is passed.

    When alpha_override is None the TRAINING/alpha==0 path takes the STE branch; passing
    a 0-tensor override forces the BLEND branch instead and must still yield host exactly.
    This documents that the coalition evaluator (which always passes an override tensor)
    never touches the STE path.
    """
    torch.manual_seed(2)
    b, c, h, w = 4, 16, 8, 8
    slot = SeedSlot(slot_id="r0c0", channels=c)
    slot.germinate("norm", seed_id="s", alpha_algorithm=AlphaAlgorithm.ADD)
    slot.state.transition(SeedStage.TRAINING)
    _make_nonidentity_norm(slot.seed)
    slot.set_alpha(0.0)  # natural alpha is 0 => STE regime when override is None
    assert slot.state.stage == SeedStage.TRAINING and slot.alpha == 0.0

    host = torch.randn(b, c, h, w)
    masked = slot(host, alpha_override=torch.zeros(b, 1, 1, 1))
    assert torch.isfinite(masked).all()
    assert torch.equal(masked, host)


# ---------------------------------------------------------------------------
# (2) Fused / multi-slot: masked (alpha=0) == physically detached (never germinated)
# ---------------------------------------------------------------------------


def _av(batch: int, value: float) -> torch.Tensor:
    """CNN alpha_override vector of shape [B, 1, 1, 1] filled with ``value``."""
    return torch.full((batch, 1, 1, 1), float(value))


def _germinate_active(model: MorphogeneticModel, slot_id: str, seed_id: str, alpha: float) -> None:
    model.germinate_seed("norm", seed_id, slot=slot_id)
    slot = model.seed_slots[slot_id]
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    _make_nonidentity_norm(slot.seed)
    slot.set_alpha(alpha)


def _assert_masked_equals_detached(target: str, other: str) -> None:
    """Pin: masking ``target`` to 0.0 == ``target`` never germinated, across two
    coalition contexts for ``other`` (natural alpha AND masked).

    Fidelity to "never germinated" (advisor note): the reference (detached) legs are
    run while ``target`` is still DORMANT — literally never germinated — on the same
    model/host with the SAME ``other`` seed weights. ``target`` is germinated only
    afterwards for the masked legs, so nothing about ``other`` or the host changes
    between the two.
    """
    torch.manual_seed(7)
    host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
    model = MorphogeneticModel(host, slots=["r0c0", "r0c1"])
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    b = x.shape[0]
    other_alpha = NATURAL_ALPHA
    target_alpha = 0.5

    # v(empty): pure host, all slots dormant.
    with torch.inference_mode():
        host_only = model.fused_forward(x, {})

    # ``other`` germinated & active; ``target`` NEVER germinated (dormant).
    _germinate_active(model, other, seed_id="other", alpha=other_alpha)
    with torch.inference_mode():
        detached_other_nat = model.fused_forward(x, {other: _av(b, other_alpha)})
        detached_other_mask = model.fused_forward(x, {other: torch.zeros(b, 1, 1, 1)})

    # Now germinate ``target`` and mask it to 0.0 — the seed runs but is zeroed.
    _germinate_active(model, target, seed_id="target", alpha=target_alpha)
    with torch.inference_mode():
        masked_other_nat = model.fused_forward(
            x, {target: torch.zeros(b, 1, 1, 1), other: _av(b, other_alpha)}
        )
        masked_other_mask = model.fused_forward(
            x, {target: torch.zeros(b, 1, 1, 1), other: torch.zeros(b, 1, 1, 1)}
        )
        full = model.fused_forward(
            x, {target: _av(b, target_alpha), other: _av(b, other_alpha)}
        )

    for t in (masked_other_nat, masked_other_mask, detached_other_nat, detached_other_mask):
        assert torch.isfinite(t).all()

    # Meaningfulness: ``target`` at its natural alpha genuinely changes the logits, so
    # the masked==detached equalities below are not a vacuous identity pass-through.
    assert not torch.equal(full, masked_other_nat), (
        f"target slot {target} must contribute at natural alpha (vs masked)"
    )

    # Core property: masking ``target`` to 0.0 == ``target`` physically detached, in
    # BOTH coalition contexts for ``other``.
    assert torch.equal(masked_other_nat, detached_other_nat), (
        f"masked({target}) must equal detached({target}) with {other} at natural alpha"
    )
    assert torch.equal(masked_other_mask, detached_other_mask), (
        f"masked({target}) must equal detached({target}) with {other} masked"
    )

    # v(empty) identity: both slots masked to 0.0 == pure host-only forward.
    assert torch.equal(masked_other_mask, host_only), (
        "both slots masked to 0.0 must reproduce the host-only forward (v(empty))"
    )


def test_fused_masking_equals_detachment_r0c0_target() -> None:
    """Mask r0c0 (target); hold r0c1 (other) at natural alpha and masked."""
    _assert_masked_equals_detached(target="r0c0", other="r0c1")


def test_fused_masking_equals_detachment_r0c1_target() -> None:
    """Symmetric: mask r0c1 (target); hold r0c0 (other) at natural alpha and masked."""
    _assert_masked_equals_detached(target="r0c1", other="r0c0")
