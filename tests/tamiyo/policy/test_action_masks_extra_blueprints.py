"""extra_blueprints union in compute_action_masks (PIN-E declared-schedule availability).

A declared proof-baseline schedule may germinate blueprints (e.g. PLACEBO) that
are deliberately absent from the per-topology availability sets. The mask
builder accepts an explicit extra_blueprints union so the schedule's forced
germination is legal WITHOUT polluting normal runs' action space and WITHOUT
weakening the force-guard (which validates forced heads against the base mask).
"""

import torch

from esper.leyline.factored_actions import BlueprintAction
from esper.tamiyo.policy.action_masks import compute_action_masks

_EMPTY_SLOTS = {"r0c0": None, "r0c1": None, "r0c2": None}
_ALL = ["r0c0", "r0c1", "r0c2"]


def test_placebo_masked_off_by_default():
    masks = compute_action_masks(_EMPTY_SLOTS, enabled_slots=_ALL)
    assert not masks["blueprint"][BlueprintAction.PLACEBO]


def test_extra_blueprints_unions_placebo_into_the_mask():
    masks = compute_action_masks(
        _EMPTY_SLOTS,
        enabled_slots=_ALL,
        extra_blueprints=frozenset({BlueprintAction.PLACEBO}),
    )
    assert masks["blueprint"][BlueprintAction.PLACEBO]


def test_extra_blueprints_changes_nothing_else():
    base = compute_action_masks(_EMPTY_SLOTS, enabled_slots=_ALL)
    with_extra = compute_action_masks(
        _EMPTY_SLOTS,
        enabled_slots=_ALL,
        extra_blueprints=frozenset({BlueprintAction.PLACEBO}),
    )
    diff = base["blueprint"] ^ with_extra["blueprint"]
    assert diff.sum().item() == 1
    assert diff[BlueprintAction.PLACEBO]
    # NOOP stays force-disabled even if someone tries to union it back in.
    noop_extra = compute_action_masks(
        _EMPTY_SLOTS,
        enabled_slots=_ALL,
        extra_blueprints=frozenset({BlueprintAction.NOOP}),
    )
    assert not noop_extra["blueprint"][BlueprintAction.NOOP]
    for key in ("op", "slot", "style", "tempo"):
        assert torch.equal(base[key], with_extra[key])
