"""Tests for the PIN-E placebo blueprint (near-inert small-real seed).

The placebo is the noise-floor instrument for the Committed-Shapley enablement
gate (esper-lite-94869250f1): a residual seed whose delta is tiny-random at
birth and — run with seed_lr=0 — stays tiny for an entire run, so its measured
counterfactual contribution is pure estimator noise. It must NOT be
deterministic-zero (the reviewer-rejected degenerate case: bit-identical
leave-out logits) and must NOT renormalize its delta back to O(1) (the
GroupNorm wrinkle).
"""

import pytest
import torch
import torch.nn as nn


def _make_placebo(dim: int = 64) -> nn.Module:
    from esper.kasmina.blueprints import BlueprintRegistry

    return BlueprintRegistry.create("cnn", "placebo", dim=dim)


def test_placebo_registered_for_cnn():
    from esper.kasmina.blueprints import BlueprintRegistry

    names = {spec.name for spec in BlueprintRegistry.list_for_topology("cnn")}
    assert "placebo" in names


def test_placebo_rejects_unexpected_kwargs():
    from esper.kasmina.blueprints import BlueprintRegistry

    with pytest.raises(ValueError, match="placebo"):
        BlueprintRegistry.create("cnn", "placebo", dim=64, reduction=4)


def test_placebo_is_residual_and_near_inert_but_not_identity():
    """y = x + delta(x) with tiny delta: near-inert, NOT bit-identical."""
    torch.manual_seed(0)
    seed = _make_placebo(64)
    x = torch.randn(2, 64, 8, 8)
    y = seed(x)

    delta = y - x
    rel = delta.norm() / x.norm()
    # Non-degenerate: the delta must be a real perturbation...
    assert rel > 1e-5, "placebo is deterministic-zero (degenerate: vacuous noise floor)"
    # ...but near-inert: orders of magnitude below a real seed's contribution.
    assert rel < 1e-2, f"placebo delta too large ({rel:.2e}): contaminates the noise floor"


def test_placebo_param_count_matches_estimate():
    """Single 3x3 depthwise conv, bias=False: dim*9 params, all trainable."""
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = _make_placebo(64)
    actual = sum(p.numel() for p in seed.parameters())
    assert actual == 64 * 9
    assert all(p.requires_grad for p in seed.parameters())

    spec = next(
        s for s in BlueprintRegistry.list_for_topology("cnn") if s.name == "placebo"
    )
    assert spec.param_estimate == 64 * 9


def test_placebo_has_no_normalizer_or_activation():
    """The delta path must end at the conv: GroupNorm would renormalize the tiny
    delta back to O(1); an activation would sign-bias the measured mean."""
    seed = _make_placebo(64)
    banned = (
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.LayerNorm,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Tanh,
    )
    for module in seed.modules():
        assert not isinstance(module, banned), (
            f"placebo contains {type(module).__name__}: breaks near-inertness"
        )
    convs = [m for m in seed.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 1
    assert convs[0].bias is None


def test_placebo_gradients_clear_vanishing_threshold_with_margin():
    """G2's gradient-health check counts per-tensor grad norms < 1e-7 as
    vanishing; the placebo has ONE grad tensor so health is 1.0-or-0.5 (binary).
    Prove the margin on a synthetic batch."""
    torch.manual_seed(0)
    seed = _make_placebo(64)
    x = torch.randn(4, 64, 8, 8, requires_grad=True)
    y = seed(x)
    y.sum().backward()

    (weight,) = [p for p in seed.parameters()]
    grad_norm = weight.grad.norm().item()
    assert grad_norm > 1e-4, (
        f"placebo grad norm {grad_norm:.2e} has no margin over the 1e-7 vanishing "
        "threshold; G2 would be one bad batch away from blocking"
    )


def test_placebo_delta_stays_fixed_without_optimizer_step():
    """With seed_lr=0 the weights never move; delta is a fixed function.
    Backward alone (no step) must not change the forward output."""
    torch.manual_seed(0)
    seed = _make_placebo(64)
    x = torch.randn(2, 64, 8, 8)
    y_before = seed(x).detach().clone()

    out = seed(torch.randn(2, 64, 8, 8))
    out.sum().backward()

    y_after = seed(x).detach()
    assert torch.equal(y_before, y_after)
