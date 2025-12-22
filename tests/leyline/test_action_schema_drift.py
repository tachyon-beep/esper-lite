"""Phase 4 guard: action/schema drift detection.

This test is meant to fail loudly if the LifecycleOp schema changes (PRUNE rename,
new ops, reorders) but dependent subsystems are not updated atomically.
"""

from __future__ import annotations

from esper.leyline import HEAD_NAMES
from esper.leyline.factored_actions import (
    ACTION_HEAD_NAMES,
    ACTION_HEAD_SPECS,
    AlphaAlgorithmAction,
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlendAction,
    BlueprintAction,
    LifecycleOp,
    NUM_ALPHA_ALGORITHMS,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLENDS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_TEMPO,
    OP_NAMES,
    OP_PRUNE,
    TempoAction,
    get_action_head_sizes,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.rewards.rewards import INTERVENTION_COSTS
from esper.simic.training import vectorized as vec
from esper.tamiyo.policy.action_masks import compute_action_masks


def test_lifecycle_op_tables_and_dependents_stay_in_sync() -> None:
    # Leyline lookup tables must mirror the enum exactly.
    assert NUM_OPS == len(LifecycleOp)
    assert OP_NAMES == tuple(op.name for op in LifecycleOp)

    # Rewards must define costs for every op (no silent missing branches).
    assert set(INTERVENTION_COSTS.keys()) == set(LifecycleOp)

    # Vectorized runtime must use the same lookup tables/constants.
    assert vec.OP_NAMES is OP_NAMES
    assert vec.OP_PRUNE == OP_PRUNE

    # Action masking must produce one mask per head with correct op dimensionality.
    slot_config = SlotConfig.default()
    enabled_slots = list(slot_config.slot_ids)
    slot_states = {slot_id: None for slot_id in enabled_slots}
    masks = compute_action_masks(
        slot_states=slot_states,
        enabled_slots=enabled_slots,
        total_seeds=0,
        max_seeds=0,
        slot_config=slot_config,
        topology="cnn",
    )
    assert set(masks.keys()) == set(HEAD_NAMES)
    assert masks["op"].shape == (NUM_OPS,)
    assert masks["op"][LifecycleOp.WAIT].item() is True


def test_action_head_schema_contract() -> None:
    slot_config = SlotConfig.default()
    head_sizes = get_action_head_sizes(slot_config)

    assert HEAD_NAMES == ACTION_HEAD_NAMES
    assert head_sizes["slot"] == slot_config.num_slots
    assert head_sizes["blueprint"] == NUM_BLUEPRINTS
    assert head_sizes["blend"] == NUM_BLENDS
    assert head_sizes["tempo"] == NUM_TEMPO
    assert head_sizes["alpha_target"] == NUM_ALPHA_TARGETS
    assert head_sizes["alpha_speed"] == NUM_ALPHA_SPEEDS
    assert head_sizes["alpha_curve"] == NUM_ALPHA_CURVES
    assert head_sizes["alpha_algorithm"] == NUM_ALPHA_ALGORITHMS
    assert head_sizes["op"] == NUM_OPS

    specs_by_name = {spec.name: spec for spec in ACTION_HEAD_SPECS}
    assert specs_by_name["slot"].names(slot_config) == slot_config.slot_ids
    assert specs_by_name["blueprint"].names(slot_config) == tuple(bp.name for bp in BlueprintAction)
    assert specs_by_name["blend"].names(slot_config) == tuple(bl.name for bl in BlendAction)
    assert specs_by_name["tempo"].names(slot_config) == tuple(tp.name for tp in TempoAction)
    assert specs_by_name["alpha_target"].names(slot_config) == tuple(at.name for at in AlphaTargetAction)
    assert specs_by_name["alpha_speed"].names(slot_config) == tuple(sp.name for sp in AlphaSpeedAction)
    assert specs_by_name["alpha_curve"].names(slot_config) == tuple(cv.name for cv in AlphaCurveAction)
    assert specs_by_name["alpha_algorithm"].names(slot_config) == tuple(alg.name for alg in AlphaAlgorithmAction)
    assert specs_by_name["op"].names(slot_config) == tuple(op.name for op in LifecycleOp)
