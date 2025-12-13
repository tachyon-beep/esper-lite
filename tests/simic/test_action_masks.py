# tests/simic/test_action_masks.py
"""Tests for action masking with multi-slot support.

The mask system only blocks PHYSICALLY IMPOSSIBLE actions:
- SLOT: only enabled slots (from --slots arg) are selectable
- GERMINATE: blocked if ALL enabled slots occupied OR at seed limit
- FOSSILIZE: blocked if NO enabled slot has a PROBATIONARY seed
- CULL: blocked if NO enabled slot has a cullable seed with age >= MIN_CULL_AGE
- WAIT: always valid
- BLUEPRINT: NOOP always blocked (0 trainable parameters)
"""
import torch
import pytest

from esper.simic.action_masks import (
    MaskSeedInfo,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
)
from esper.leyline import SeedStage, MIN_CULL_AGE
from esper.leyline.factored_actions import LifecycleOp, NUM_OPS, SlotAction


def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not CULL/FOSSILIZE."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    masks = compute_action_masks(slot_states, enabled_slots=["early", "mid", "late"])

    # All enabled slots should be valid targets
    assert masks["slot"][SlotAction.EARLY] == True
    assert masks["slot"][SlotAction.MID] == True
    assert masks["slot"][SlotAction.LATE] == True

    # WAIT and GERMINATE should be valid
    assert masks["op"][LifecycleOp.WAIT] == True
    assert masks["op"][LifecycleOp.GERMINATE] == True

    # No seed means no CULL/FOSSILIZE
    assert masks["op"][LifecycleOp.CULL] == False
    assert masks["op"][LifecycleOp.FOSSILIZE] == False


def test_compute_action_masks_single_slot_enabled():
    """Only enabled slots should be selectable."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    # Only mid is enabled
    masks = compute_action_masks(slot_states, enabled_slots=["mid"])

    # Only mid should be selectable
    assert masks["slot"][SlotAction.EARLY] == False
    assert masks["slot"][SlotAction.MID] == True
    assert masks["slot"][SlotAction.LATE] == False


def test_compute_action_masks_active_slot_training_stage():
    """Active slot in TRAINING should allow CULL, not FOSSILIZE."""
    slot_states = {
        "early": None,
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # >= MIN_CULL_AGE
        ),
        "late": None,
    }

    masks = compute_action_masks(slot_states, enabled_slots=["early", "mid", "late"])

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT] == True

    # GERMINATE still valid (empty slots exist)
    assert masks["op"][LifecycleOp.GERMINATE] == True

    # CULL valid (mid has seed age >= MIN_CULL_AGE)
    assert masks["op"][LifecycleOp.CULL] == True

    # FOSSILIZE not valid (no PROBATIONARY seed)
    assert masks["op"][LifecycleOp.FOSSILIZE] == False


def test_compute_action_masks_probationary_stage():
    """PROBATIONARY stage should allow FOSSILIZE."""
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["mid"])

    # FOSSILIZE valid from PROBATIONARY
    assert masks["op"][LifecycleOp.FOSSILIZE] == True

    # CULL valid (seed exists and age >= 1)
    assert masks["op"][LifecycleOp.CULL] == True


def test_compute_action_masks_fossilized_stage():
    """FOSSILIZED stage should not allow GERMINATE, FOSSILIZE, or CULL."""
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            seed_age_epochs=20,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["mid"])

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT] == True

    # No GERMINATE (slot occupied)
    assert masks["op"][LifecycleOp.GERMINATE] == False

    # No FOSSILIZE (already fossilized)
    assert masks["op"][LifecycleOp.FOSSILIZE] == False

    # No CULL - FOSSILIZED is terminal success, cannot be removed
    assert masks["op"][LifecycleOp.CULL] == False


def test_compute_action_masks_wait_always_valid():
    """WAIT should be valid in all situations."""
    # Empty slots
    masks_empty = compute_action_masks(
        {"early": None, "mid": None, "late": None},
        enabled_slots=["early", "mid", "late"]
    )
    assert masks_empty["op"][LifecycleOp.WAIT] == True

    # Active slot
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
    }
    masks_active = compute_action_masks(slot_states, enabled_slots=["mid"])
    assert masks_active["op"][LifecycleOp.WAIT] == True


def test_compute_action_masks_blueprint_blend_masks():
    """Blueprint mask excludes NOOP (0 params); blend mask is all-valid."""
    from esper.leyline.factored_actions import BlueprintAction

    slot_states = {"early": None, "mid": None, "late": None}
    masks = compute_action_masks(slot_states, enabled_slots=["mid"])

    # NOOP is masked (0 trainable parameters), others are valid
    assert masks["blueprint"][BlueprintAction.NOOP] == False
    assert masks["blueprint"][BlueprintAction.CONV_ENHANCE] == True
    assert masks["blueprint"][BlueprintAction.ATTENTION] == True
    assert masks["blueprint"][BlueprintAction.NORM] == True
    assert masks["blueprint"][BlueprintAction.DEPTHWISE] == True

    # All blends should be valid
    assert masks["blend"].all()


def test_compute_action_masks_min_cull_age():
    """CULL should be blocked if seed_age < MIN_CULL_AGE."""
    # Seed age 0 (just germinated)
    slot_states_age0 = {
        "mid": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=0,
        ),
    }
    masks_age0 = compute_action_masks(slot_states_age0, enabled_slots=["mid"])
    assert masks_age0["op"][LifecycleOp.CULL] == False

    # Seed age 1 (minimum for cull)
    slot_states_age1 = {
        "mid": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=1,
        ),
    }
    masks_age1 = compute_action_masks(slot_states_age1, enabled_slots=["mid"])
    assert masks_age1["op"][LifecycleOp.CULL] == True


def test_compute_batch_masks():
    """Should compute masks for a batch of observations."""
    batch_slot_states = [
        # Env 0: empty slots
        {"early": None, "mid": None, "late": None},
        # Env 1: mid slot active in TRAINING
        {
            "early": None,
            "mid": MaskSeedInfo(
                stage=SeedStage.TRAINING.value,
                seed_age_epochs=5,
            ),
            "late": None,
        },
    ]

    masks = compute_batch_masks(batch_slot_states, enabled_slots=["early", "mid", "late"])

    # Check shapes (NUM_OPS=4 now)
    assert masks["slot"].shape == (2, 3)
    assert masks["blueprint"].shape == (2, 5)
    assert masks["blend"].shape == (2, 3)
    assert masks["op"].shape == (2, NUM_OPS)

    # WAIT always valid for both
    assert masks["op"][0, LifecycleOp.WAIT] == True
    assert masks["op"][1, LifecycleOp.WAIT] == True

    # Env 0: can GERMINATE, not CULL/FOSSILIZE
    assert masks["op"][0, LifecycleOp.GERMINATE] == True
    assert masks["op"][0, LifecycleOp.CULL] == False
    assert masks["op"][0, LifecycleOp.FOSSILIZE] == False

    # Env 1: can GERMINATE (empty slots), CULL; not FOSSILIZE
    assert masks["op"][1, LifecycleOp.GERMINATE] == True
    assert masks["op"][1, LifecycleOp.CULL] == True
    assert masks["op"][1, LifecycleOp.FOSSILIZE] == False


def test_compute_action_masks_at_seed_limit():
    """GERMINATE should be masked when at hard seed limit."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    # At limit: 10 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["mid"], total_seeds=10, max_seeds=10
    )

    # GERMINATE should be masked
    assert masks["op"][LifecycleOp.GERMINATE] == False

    # Other ops should be unaffected
    assert masks["op"][LifecycleOp.WAIT] == True
    assert masks["op"][LifecycleOp.CULL] == False  # no seed


def test_compute_action_masks_over_seed_limit():
    """GERMINATE should be masked when over hard seed limit."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    # Over limit: 15 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["mid"], total_seeds=15, max_seeds=10
    )

    # GERMINATE should be masked
    assert masks["op"][LifecycleOp.GERMINATE] == False
    assert masks["op"][LifecycleOp.WAIT] == True


def test_compute_action_masks_under_seed_limit():
    """GERMINATE should be allowed when under hard seed limit."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    # Under limit: 5 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["mid"], total_seeds=5, max_seeds=10
    )

    # GERMINATE should be allowed (empty slot + under limit)
    assert masks["op"][LifecycleOp.GERMINATE] == True
    assert masks["op"][LifecycleOp.WAIT] == True
    assert masks["op"][LifecycleOp.CULL] == False  # no seed


def test_compute_action_masks_seed_limit_with_active_seed():
    """Seed limit should only affect GERMINATE, not WAIT/CULL."""
    # Active slot with seed
    slot_states = {
        "early": None,
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
        "late": None,
    }

    # At limit with active slot
    masks = compute_action_masks(
        slot_states, enabled_slots=["early", "mid", "late"], total_seeds=10, max_seeds=10
    )

    # GERMINATE masked due to limit (even though empty slots exist)
    assert masks["op"][LifecycleOp.GERMINATE] == False

    # CULL should still be valid
    assert masks["op"][LifecycleOp.CULL] == True
    assert masks["op"][LifecycleOp.WAIT] == True


def test_compute_batch_masks_with_seed_limits():
    """Batch masks should respect per-env seed limits."""
    batch_slot_states = [
        # Env 0: empty slots, under limit
        {"early": None, "mid": None, "late": None},
        # Env 1: empty slots, at limit
        {"early": None, "mid": None, "late": None},
        # Env 2: empty slots, over limit
        {"early": None, "mid": None, "late": None},
    ]

    masks = compute_batch_masks(
        batch_slot_states,
        enabled_slots=["mid"],
        total_seeds_list=[5, 10, 15],
        max_seeds=10,
    )

    # Env 0: under limit, can GERMINATE
    assert masks["op"][0, LifecycleOp.GERMINATE] == True

    # Env 1: at limit, cannot GERMINATE
    assert masks["op"][1, LifecycleOp.GERMINATE] == False

    # Env 2: over limit, cannot GERMINATE
    assert masks["op"][2, LifecycleOp.GERMINATE] == False

    # WAIT should be valid for all
    assert masks["op"][0, LifecycleOp.WAIT] == True
    assert masks["op"][1, LifecycleOp.WAIT] == True
    assert masks["op"][2, LifecycleOp.WAIT] == True


def test_compute_batch_masks_seed_limit_default():
    """When total_seeds not provided, should default to 0 (under limit)."""
    batch_slot_states = [
        {"early": None, "mid": None, "late": None},
    ]

    # No total_seeds_list provided - should default to 0
    masks = compute_batch_masks(batch_slot_states, enabled_slots=["mid"], max_seeds=10)

    # Should allow GERMINATE (0 < 10)
    assert masks["op"][0, LifecycleOp.GERMINATE] == True


def test_mask_seed_info_dataclass():
    """MaskSeedInfo should be a frozen dataclass with correct fields."""
    info = MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5)

    assert info.stage == SeedStage.TRAINING.value
    assert info.seed_age_epochs == 5

    # Should be frozen (immutable)
    with pytest.raises(AttributeError):
        info.stage = SeedStage.BLENDING.value


def test_min_cull_age_constant():
    """MIN_CULL_AGE should be 1 (need one epoch for counterfactual)."""
    assert MIN_CULL_AGE == 1


# =============================================================================
# Tests for multi-slot semantics (optimistic masking)
# =============================================================================


def test_compute_action_masks_fossilize_any_slot():
    """FOSSILIZE should be valid if ANY enabled slot is PROBATIONARY."""
    slot_states = {
        "early": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,  # Not fossilizable
            seed_age_epochs=10,
        ),
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,  # Fossilizable
            seed_age_epochs=10,
        ),
    }

    # Both slots enabled - FOSSILIZE valid because mid is PROBATIONARY
    masks = compute_action_masks(slot_states, enabled_slots=["early", "mid"])
    assert masks["op"][LifecycleOp.FOSSILIZE] == True


def test_compute_action_masks_cull_any_slot():
    """CULL should be valid if ANY enabled slot has a cullable seed."""
    slot_states = {
        "early": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=0,  # Too young to cull
        ),
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # Old enough to cull
        ),
    }

    # Both slots enabled - CULL valid because mid has age >= MIN_CULL_AGE
    masks = compute_action_masks(slot_states, enabled_slots=["early", "mid"])
    assert masks["op"][LifecycleOp.CULL] == True


def test_compute_action_masks_germinate_any_empty_slot():
    """GERMINATE should be valid if ANY enabled slot is empty."""
    slot_states = {
        "early": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
        "mid": None,  # Empty
    }

    # Both slots enabled - GERMINATE valid because mid is empty
    masks = compute_action_masks(slot_states, enabled_slots=["early", "mid"])
    assert masks["op"][LifecycleOp.GERMINATE] == True


def test_compute_action_masks_only_enabled_slots_checked():
    """Only enabled slots should affect op validity."""
    slot_states = {
        "early": None,  # Empty but not enabled
        "mid": MaskSeedInfo(
            stage=SeedStage.FOSSILIZED.value,  # Not germinate-able
            seed_age_epochs=20,
        ),
    }

    # Only mid enabled - GERMINATE invalid (mid is occupied)
    masks = compute_action_masks(slot_states, enabled_slots=["mid"])
    assert masks["op"][LifecycleOp.GERMINATE] == False

    # Both enabled - GERMINATE valid (early is empty)
    masks_both = compute_action_masks(slot_states, enabled_slots=["early", "mid"])
    assert masks_both["op"][LifecycleOp.GERMINATE] == True


def test_compute_action_masks_enabled_slots_required():
    """enabled_slots is required - raises TypeError if not provided."""
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    # enabled_slots is required, not optional
    with pytest.raises(TypeError):
        compute_action_masks(slot_states)  # Missing enabled_slots


def test_compute_batch_masks_enabled_slots_required():
    """enabled_slots is required for batch masks."""
    batch_slot_states = [{"mid": None}]

    with pytest.raises(TypeError):
        compute_batch_masks(batch_slot_states)  # Missing enabled_slots


def test_slot_id_to_index():
    """slot_id_to_index should correctly map slot names to indices."""
    assert slot_id_to_index("early") == SlotAction.EARLY.value
    assert slot_id_to_index("mid") == SlotAction.MID.value
    assert slot_id_to_index("late") == SlotAction.LATE.value

    with pytest.raises(KeyError):
        slot_id_to_index("invalid")


# =============================================================================
# Tests for build_slot_states() helper function
# =============================================================================


class TestBuildSlotStates:
    """Tests for build_slot_states() helper function."""

    def test_empty_model_returns_none_states(self):
        """Empty slots return None for each slot."""
        from esper.simic.action_masks import build_slot_states
        from esper.leyline import SeedStage

        # Mock model with empty slots
        class MockMetrics:
            epochs_total = 0

        class MockState:
            stage = SeedStage.DORMANT
            metrics = MockMetrics()

        class MockSlot:
            state = MockState()

        class MockModel:
            seed_slots = {"mid": MockSlot()}

        result = build_slot_states(MockModel(), ["mid"])
        assert result == {"mid": None}

    def test_active_seed_returns_mask_seed_info(self):
        """Active seed returns MaskSeedInfo with correct stage and age."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedStage

        class MockMetrics:
            epochs_total = 5

        class MockState:
            stage = SeedStage.TRAINING
            metrics = MockMetrics()

        class MockSlot:
            state = MockState()

        class MockModel:
            seed_slots = {"mid": MockSlot()}

        result = build_slot_states(MockModel(), ["mid"])

        assert "mid" in result
        assert isinstance(result["mid"], MaskSeedInfo)
        assert result["mid"].stage == SeedStage.TRAINING.value
        assert result["mid"].seed_age_epochs == 5

    def test_multiple_slots(self):
        """Multiple slots are all processed."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedStage

        class MockMetricsActive:
            epochs_total = 3

        class MockMetricsDormant:
            epochs_total = 0

        class MockStateActive:
            stage = SeedStage.BLENDING
            metrics = MockMetricsActive()

        class MockStateDormant:
            stage = SeedStage.DORMANT
            metrics = MockMetricsDormant()

        class MockActiveSlot:
            state = MockStateActive()

        class MockEmptySlot:
            state = MockStateDormant()

        class MockModel:
            seed_slots = {
                "early": MockEmptySlot(),
                "mid": MockActiveSlot(),
                "late": MockEmptySlot(),
            }

        result = build_slot_states(MockModel(), ["early", "mid", "late"])

        assert result["early"] is None
        assert isinstance(result["mid"], MaskSeedInfo)
        assert result["late"] is None
