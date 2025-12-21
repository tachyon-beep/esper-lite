"""Phase 4 guard: action/schema drift detection.

This test is meant to fail loudly if the LifecycleOp schema changes (PRUNE rename,
new ops, reorders) but dependent subsystems are not updated atomically.
"""

from __future__ import annotations

from esper.leyline import HEAD_NAMES
from esper.leyline.factored_actions import LifecycleOp, NUM_OPS, OP_NAMES, OP_PRUNE
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
