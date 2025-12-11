# tests/simic/test_action_masks.py
import torch


def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not ADVANCE/CULL."""
    from esper.simic.action_masks import compute_action_masks

    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    masks = compute_action_masks(slot_states)

    # All slots should be valid targets
    assert masks["slot"].all()

    # WAIT and GERMINATE should be valid, ADVANCE and CULL should not
    # op order: WAIT=0, GERMINATE=1, ADVANCE=2, CULL=3
    assert masks["op"][0] == True   # WAIT
    assert masks["op"][1] == True   # GERMINATE
    assert masks["op"][2] == False  # ADVANCE (no seed to advance)
    assert masks["op"][3] == False  # CULL (no seed to cull)


def test_compute_action_masks_active_slot():
    """Active slot should allow ADVANCE/CULL, not GERMINATE."""
    from esper.simic.action_masks import compute_action_masks
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    slot_states = {
        "early": None,
        "mid": SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.0,
            total_improvement=2.5,
            epochs_in_stage=5
        ),
        "late": None,
    }

    masks = compute_action_masks(slot_states, target_slot="mid")

    # GERMINATE should be invalid for occupied slot
    assert masks["op"][1] == False  # GERMINATE

    # ADVANCE and CULL should be valid
    assert masks["op"][2] == True   # ADVANCE
    assert masks["op"][3] == True   # CULL


def test_compute_action_masks_wait_always_valid():
    """WAIT should be valid in all situations."""
    from esper.simic.action_masks import compute_action_masks
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    # Empty slots
    masks_empty = compute_action_masks({"early": None, "mid": None, "late": None})
    assert masks_empty["op"][0] == True  # WAIT

    # Active slot
    slot_states = {
        "mid": SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.0,
            total_improvement=2.5,
            epochs_in_stage=5
        ),
    }
    masks_active = compute_action_masks(slot_states, target_slot="mid")
    assert masks_active["op"][0] == True  # WAIT


def test_compute_action_masks_blueprint_blend_always_valid():
    """Blueprint and blend masks should always be all-valid (filtered by op)."""
    from esper.simic.action_masks import compute_action_masks

    slot_states = {"early": None, "mid": None, "late": None}
    masks = compute_action_masks(slot_states)

    # All blueprints and blends should be valid
    assert masks["blueprint"].all()
    assert masks["blend"].all()


def test_compute_batch_masks():
    """Should compute masks for a batch of observations."""
    from esper.simic.action_masks import compute_batch_masks
    from esper.simic.rewards import SeedInfo
    from esper.leyline import SeedStage

    batch_slot_states = [
        # Env 0: empty slots
        {"early": None, "mid": None, "late": None},
        # Env 1: mid slot active
        {
            "early": None,
            "mid": SeedInfo(
                stage=SeedStage.TRAINING.value,
                improvement_since_stage_start=1.0,
                total_improvement=2.5,
                epochs_in_stage=5
            ),
            "late": None,
        },
    ]

    masks = compute_batch_masks(batch_slot_states, target_slots=[None, "mid"])

    # Check shapes
    assert masks["slot"].shape == (2, 3)
    assert masks["blueprint"].shape == (2, 5)
    assert masks["blend"].shape == (2, 3)
    assert masks["op"].shape == (2, 4)

    # WAIT always valid for both
    assert masks["op"][0, 0] == True
    assert masks["op"][1, 0] == True

    # Env 0: can GERMINATE, not ADVANCE/CULL
    assert masks["op"][0, 1] == True   # GERMINATE
    assert masks["op"][0, 2] == False  # ADVANCE
    assert masks["op"][0, 3] == False  # CULL

    # Env 1: cannot GERMINATE, can ADVANCE/CULL
    assert masks["op"][1, 1] == False  # GERMINATE
    assert masks["op"][1, 2] == True   # ADVANCE
    assert masks["op"][1, 3] == True   # CULL
