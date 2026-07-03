"""PLACEBO blueprint action contract (PIN-E harness, esper-lite-94869250f1).

PLACEBO must be a first-class BlueprintAction member (the observation encoder
fail-louds on any blueprint_id outside the enum-derived BLUEPRINT_ID_TO_INDEX)
but must NEVER appear in the per-topology availability sets — only a declared
schedule may surface it via the mask's extra_blueprints union.
"""

from esper.leyline import NUM_BLUEPRINTS
from esper.leyline.factored_actions import (
    BLUEPRINT_ID_TO_INDEX,
    BLUEPRINT_IDS,
    CNN_BLUEPRINTS,
    TRANSFORMER_BLUEPRINTS,
    BlueprintAction,
)


def test_placebo_is_an_enum_member_with_stable_index():
    assert BlueprintAction.PLACEBO.value == 13
    assert BlueprintAction.PLACEBO.to_blueprint_id() == "placebo"
    assert BLUEPRINT_IDS[13] == "placebo"
    assert BLUEPRINT_ID_TO_INDEX["placebo"] == 13
    assert NUM_BLUEPRINTS == 14


def test_placebo_absent_from_every_topology_availability_set():
    assert BlueprintAction.PLACEBO not in CNN_BLUEPRINTS
    assert BlueprintAction.PLACEBO not in TRANSFORMER_BLUEPRINTS
