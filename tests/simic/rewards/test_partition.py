"""Tests for the subtractive reward-stream partition (EV-stabilization Stage 0/2).

`split_reward_streams` is the single-source helper that both Stage 0 (per-stream EV
diagnostics) and Stage 2 (the HRA cf value head) use to split the finalized step
reward into:
- R_cf   = the counterfactual stream = ``components.bounded_attribution`` (or 0.0)
- R_main = ``reward_raw - R_cf``      = everything else, by subtraction

The defining property is EXHAUSTIVE-BY-CONSTRUCTION: ``r_main + r_cf == reward_raw``
for any inputs, so no reward term can be dropped or double-counted.
"""

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards.partition import split_reward_streams


def test_split_extracts_bounded_attribution_as_cf_and_rest_as_main() -> None:
    components = RewardComponentsTelemetry(bounded_attribution=2.5)
    reward_raw = 4.0

    r_main, r_cf = split_reward_streams(reward_raw, components)

    assert r_cf == 2.5
    assert r_main == 1.5


def test_split_none_attribution_yields_zero_cf() -> None:
    # SHAPED steps with no contribution signal: bounded_attribution is None,
    # meaning "no attribution this step" -> the whole reward is R_main.
    components = RewardComponentsTelemetry(bounded_attribution=None)
    reward_raw = 3.0

    r_main, r_cf = split_reward_streams(reward_raw, components)

    assert r_cf == 0.0
    assert r_main == 3.0


def test_split_is_exhaustive_main_plus_cf_equals_raw() -> None:
    # The keystone invariant across a range of reward / attribution values,
    # including negative counterfactual contributions and a clip-scale total.
    cases = [
        (4.0, 2.5),
        (3.0, None),
        (-1.0, -0.75),
        (10.0, 9.9),
        (0.0, 0.0),
        (-5.5, 2.25),
    ]
    for reward_raw, attribution in cases:
        components = RewardComponentsTelemetry(bounded_attribution=attribution)
        r_main, r_cf = split_reward_streams(reward_raw, components)
        assert r_main + r_cf == reward_raw, f"not exhaustive for {(reward_raw, attribution)}"


def test_split_handles_negative_attribution() -> None:
    components = RewardComponentsTelemetry(bounded_attribution=-0.75)
    reward_raw = -1.0

    r_main, r_cf = split_reward_streams(reward_raw, components)

    assert r_cf == -0.75
    assert r_main == -0.25
