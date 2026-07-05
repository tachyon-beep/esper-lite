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
    """3 SERIAL placebo lifecycles (the D3 rule masks GERMINATE while any seed
    is GERMINATED/TRAINING, so each placebo reaches HOLDING before the next
    germinates; BLENDING ramps alpha over 5 STANDARD-tempo epochs):
    s0 germ@1/train@2/blend@12/hold@17; s1 18/19/29/34; s2 35/36/46/51.
    WAIT forever after — and NO fossilize step ever (fossilized slots are
    ablation-invisible at shapley_synergy_scale=0)."""
    sched = DECLARED_SCHEDULES[FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1]

    for epoch, slot in ((1, 0), (18, 1), (35, 2)):
        action = declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        )
        assert action.op == LifecycleOp.GERMINATE
        assert action.slot_idx == slot
        assert action.blueprint == BlueprintAction.PLACEBO

    for epoch, slot in (
        (2, 0), (19, 1), (36, 2),      # -> TRAINING
        (12, 0), (29, 1), (46, 2),     # -> BLENDING (10-epoch TRAINING dwell)
        (17, 0), (34, 1), (51, 2),     # -> HOLDING (5-epoch alpha ramp)
    ):
        action = declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        )
        assert action.op == LifecycleOp.ADVANCE, f"epoch {epoch}"
        assert action.slot_idx == slot, f"epoch {epoch}"

    for epoch in (5, 11, 28, 45, 52, 150):
        assert declared_schedule_action_for_epoch(
            FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1, epoch
        ).op == LifecycleOp.WAIT

    assert all(step.action.op != LifecycleOp.FOSSILIZE for step in sched.steps)
    # 10-epoch TRAINING dwell + 5-epoch blend ramp per slot.
    training_entry = {0: 2, 1: 19, 2: 36}
    blending_entry = {0: 12, 1: 29, 2: 46}
    holding_entry = {0: 17, 1: 34, 2: 51}
    for slot in (0, 1, 2):
        assert blending_entry[slot] - training_entry[slot] == 10
        assert holding_entry[slot] - blending_entry[slot] == 5


def test_germinate_blueprints_helper():
    assert declared_schedule_germinate_blueprints(
        FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1
    ) == frozenset({BlueprintAction.PLACEBO})
    assert declared_schedule_germinate_blueprints(
        FIXED_SCHEDULE_GERMINATE_R0C0_V1
    ) == frozenset({BlueprintAction.NORM})
    with pytest.raises(ValueError, match="Unknown declared schedule"):
        declared_schedule_germinate_blueprints("no-such-schedule")
