"""Stage 2 (EV-stab): conditional construction of the head-only cf_value_head.

The cf value head exists ONLY when ``hra_value_decomposition=True`` (true bypass —
no dead weights on the default-OFF leg). Construction must not perturb the existing
``state_value_head``/``q_head`` initialization, so the OFF leg stays byte-identical to
today and the ON leg preserves the existing value head's weights (cf head registered
AFTER q_head, so its RNG draws come last among the value-like heads).
"""

from __future__ import annotations

import torch

from esper.leyline import OBS_V3_NON_BLUEPRINT_DIM
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


def _net(hra: bool) -> FactoredRecurrentActorCritic:
    return FactoredRecurrentActorCritic(
        state_dim=OBS_V3_NON_BLUEPRINT_DIM, hra_value_decomposition=hra
    )


def test_cf_value_head_absent_by_default() -> None:
    net = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)
    assert net.cf_value_head is None


def test_cf_value_head_built_when_enabled_mirrors_state_value_head() -> None:
    net = _net(hra=True)
    assert net.cf_value_head is not None
    # Mirrors state_value_head: maps lstm_hidden_dim -> scalar.
    x = torch.zeros(2, 3, net.lstm_hidden_dim)
    out = net.cf_value_head(x)
    assert out.shape == (2, 3, 1)


def test_off_leg_is_byte_identical_to_default_ctor() -> None:
    # The required true-bypass property: adding the flag (default False) must leave the
    # OFF leg byte-identical to today. With the same seed, the no-arg ctor and the
    # explicit-OFF ctor must produce identical full state_dicts — i.e. the OFF branch
    # constructs nothing extra and does not touch the init RNG stream.
    # (NOTE: the ON leg legitimately differs — building cf_value_head consumes RNG
    # before _init_weights, reshuffling all heads' init. That is fine: the plan uses
    # fresh-init paired A/B and only the OFF leg must match today.)
    torch.manual_seed(2024)
    net_default = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)
    torch.manual_seed(2024)
    net_off = _net(hra=False)

    sd_default = net_default.state_dict()
    sd_off = net_off.state_dict()
    assert sd_default.keys() == sd_off.keys()
    for key in sd_default:
        assert torch.equal(sd_default[key], sd_off[key]), f"OFF leg diverged from default at {key}"

    # And the OFF leg carries no cf parameters at all (no dead weights).
    assert not any("cf_value_head" in name for name, _ in net_off.named_parameters())


def test_compute_cf_value_is_head_only_and_correct_shape() -> None:
    net = _net(hra=True)
    lstm_out = torch.randn(2, 3, net.lstm_hidden_dim, requires_grad=True)

    v_cf = net._compute_cf_value(lstm_out)
    assert v_cf.shape == (2, 3)

    # Head-only: gradient must NOT flow into the LSTM trunk (lstm_out), but the cf
    # head's own params must receive gradient.
    v_cf.sum().backward()
    assert lstm_out.grad is None, "cf value leaked gradient into the LSTM trunk"
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in net.cf_value_head.parameters()
    ), "cf_value_head received no gradient"


def test_get_action_threads_cf_value() -> None:
    batch = 2
    state = torch.randn(batch, 126)
    bp_idx = torch.randint(0, 13, (batch, 3))

    net_off = FactoredRecurrentActorCritic(state_dim=126)
    res_off = net_off.get_action(state, bp_idx, hidden=None, deterministic=True)
    assert res_off.cf_value is None  # OFF leg: no cf value

    net_on = FactoredRecurrentActorCritic(state_dim=126, hra_value_decomposition=True)
    res_on = net_on.get_action(state, bp_idx, hidden=None, deterministic=True)
    assert res_on.cf_value is not None  # ON leg: cf value present
    assert res_on.cf_value.shape == (batch,)
    # Same per-batch value shape as the V(s) baseline.
    assert res_on.cf_value.shape == res_on.values.shape


def test_off_leg_construction_is_deterministic() -> None:
    torch.manual_seed(7)
    a = _net(hra=False)
    torch.manual_seed(7)
    b = _net(hra=False)
    for (_, pa), (_, pb) in zip(
        a.state_value_head.named_parameters(), b.state_value_head.named_parameters(), strict=True
    ):
        assert torch.equal(pa, pb)
