"""Declared-schedule registry (PIN-E WI-3, esper-lite-94869250f1).

One registry for every declared proof-baseline schedule: the legacy R0C0
germinate schedule, the static-final source topology, and the 3-slot placebo
hold schedule. Each entry carries an import-time hash drift guard; dispatch is
by schedule_id (no per-schedule hardcoded functions).
"""

import pytest

from esper.leyline.factored_actions import BlueprintAction, LifecycleOp
from esper.leyline.proof_baselines import (
    DECLARED_SCHEDULES,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH,
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
    declared_schedule_action_for_epoch,
    declared_schedule_germinate_blueprints,
)


def test_registry_contains_all_declared_schedules():
    assert set(DECLARED_SCHEDULES) == {
        FIXED_SCHEDULE_GERMINATE_R0C0_V1,
        STATIC_FINAL_SOURCE_TOPOLOGY_V1,
        FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
    }


def test_registry_entries_carry_matching_hash_and_action_count():
    for schedule_id, schedule in DECLARED_SCHEDULES.items():
        assert schedule.schedule_id == schedule_id
        assert schedule.hash == schedule.expected_hash
        assert schedule.action_count == len(schedule.steps)
    assert (
        DECLARED_SCHEDULES[FIXED_SCHEDULE_GERMINATE_R0C0_V1].hash
        == FIXED_SCHEDULE_GERMINATE_R0C0_HASH
    )
    assert (
        DECLARED_SCHEDULES[STATIC_FINAL_SOURCE_TOPOLOGY_V1].hash
        == STATIC_FINAL_SOURCE_TOPOLOGY_HASH
    )
    assert (
        DECLARED_SCHEDULES[FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1].hash
        == FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH
    )


def test_dispatch_matches_legacy_schedules():
    germ = declared_schedule_action_for_epoch(FIXED_SCHEDULE_GERMINATE_R0C0_V1, 1)
    assert germ.op == LifecycleOp.GERMINATE
    assert germ.slot_idx == 0
    assert declared_schedule_action_for_epoch(
        FIXED_SCHEDULE_GERMINATE_R0C0_V1, 2
    ).op == LifecycleOp.WAIT

    foss = declared_schedule_action_for_epoch(STATIC_FINAL_SOURCE_TOPOLOGY_V1, 18)
    assert foss.op == LifecycleOp.FOSSILIZE


def test_dispatch_rejects_unknown_schedule_and_bad_epoch():
    with pytest.raises(ValueError, match="Unknown declared schedule"):
        declared_schedule_action_for_epoch("no-such-schedule", 1)
    with pytest.raises(ValueError, match="epoch"):
        declared_schedule_action_for_epoch(FIXED_SCHEDULE_GERMINATE_R0C0_V1, 0)


def test_placebo_schedule_shape():
    """3 staggered placebo lifecycles: germinate 1/2/3, ADVANCE to TRAINING
    4/5/6, ADVANCE to BLENDING after the 10-epoch G2 dwell 14/15/16, ADVANCE to
    HOLDING 17/18/19, WAIT forever after — and NO fossilize step ever
    (fossilized slots are ablation-invisible at shapley_synergy_scale=0)."""
    sched = DECLARED_SCHEDULES[FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1]

    for epoch, slot in ((1, 0), (2, 1), (3, 2)):
        action = declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        )
        assert action.op == LifecycleOp.GERMINATE
        assert action.slot_idx == slot
        assert action.blueprint == BlueprintAction.PLACEBO

    for epoch, slot in (
        (4, 0), (5, 1), (6, 2),        # -> TRAINING
        (14, 0), (15, 1), (16, 2),     # -> BLENDING (>=10 epochs TRAINING dwell)
        (17, 0), (18, 1), (19, 2),     # -> HOLDING
    ):
        action = declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        )
        assert action.op == LifecycleOp.ADVANCE, f"epoch {epoch}"
        assert action.slot_idx == slot, f"epoch {epoch}"

    for epoch in (7, 13, 20, 150):
        assert declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        ).op == LifecycleOp.WAIT

    assert all(step.action.op != LifecycleOp.FOSSILIZE for step in sched.steps)
    # 10-epoch TRAINING dwell per slot (G2 permissive requirement).
    germinate_to_training = {0: 4, 1: 5, 2: 6}
    training_to_blending = {0: 14, 1: 15, 2: 16}
    for slot in (0, 1, 2):
        assert training_to_blending[slot] - germinate_to_training[slot] == 10


def test_germinate_blueprints_helper():
    assert declared_schedule_germinate_blueprints(
        FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1
    ) == frozenset({BlueprintAction.PLACEBO})
    assert declared_schedule_germinate_blueprints(
        FIXED_SCHEDULE_GERMINATE_R0C0_V1
    ) == frozenset({BlueprintAction.NORM})
    with pytest.raises(ValueError, match="Unknown declared schedule"):
        declared_schedule_germinate_blueprints("no-such-schedule")
