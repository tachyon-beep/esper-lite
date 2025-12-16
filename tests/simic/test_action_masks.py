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
import pytest

from esper.simic.action_masks import (
    MaskSeedInfo,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
)
from esper.leyline import SeedStage, MIN_CULL_AGE
from esper.leyline.factored_actions import LifecycleOp, NUM_OPS


def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not CULL/FOSSILIZE."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # All enabled slots should be valid targets
    assert masks["slot"][0]  # r0c0 (formerly EARLY)
    assert masks["slot"][1]  # r0c1 (formerly MID)
    assert masks["slot"][2]  # r0c2 (formerly LATE)

    # WAIT and GERMINATE should be valid
    assert masks["op"][LifecycleOp.WAIT]
    assert masks["op"][LifecycleOp.GERMINATE]

    # No seed means no CULL/FOSSILIZE
    assert not masks["op"][LifecycleOp.CULL]
    assert not masks["op"][LifecycleOp.FOSSILIZE]


def test_compute_action_masks_single_slot_enabled():
    """Only enabled slots should be selectable."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    # Only r0c1 (mid) is enabled
    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # Only r0c1 should be selectable
    assert not masks["slot"][0]  # r0c0 not enabled
    assert masks["slot"][1]  # r0c1 enabled
    assert not masks["slot"][2]  # r0c2 not enabled


def test_compute_action_masks_active_slot_training_stage():
    """Active slot in TRAINING should allow CULL, not FOSSILIZE."""
    slot_states = {
        "r0c0": None,
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # >= MIN_CULL_AGE
        ),
        "r0c2": None,
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT]

    # GERMINATE still valid (empty slots exist)
    assert masks["op"][LifecycleOp.GERMINATE]

    # CULL valid (mid has seed age >= MIN_CULL_AGE)
    assert masks["op"][LifecycleOp.CULL]

    # FOSSILIZE not valid (no PROBATIONARY seed)
    assert not masks["op"][LifecycleOp.FOSSILIZE]


def test_compute_action_masks_probationary_stage():
    """PROBATIONARY stage should allow FOSSILIZE."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # FOSSILIZE valid from PROBATIONARY
    assert masks["op"][LifecycleOp.FOSSILIZE]

    # CULL valid (seed exists and age >= 1)
    assert masks["op"][LifecycleOp.CULL]


def test_compute_action_masks_fossilized_stage():
    """FOSSILIZED stage should not allow GERMINATE, FOSSILIZE, or CULL."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            seed_age_epochs=20,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT]

    # No GERMINATE (slot occupied)
    assert not masks["op"][LifecycleOp.GERMINATE]

    # No FOSSILIZE (already fossilized)
    assert not masks["op"][LifecycleOp.FOSSILIZE]

    # No CULL - FOSSILIZED is terminal success, cannot be removed
    assert not masks["op"][LifecycleOp.CULL]


def test_compute_action_masks_wait_always_valid():
    """WAIT should be valid in all situations."""
    # Empty slots
    masks_empty = compute_action_masks(
        {"r0c0": None, "r0c1": None, "r0c2": None},
        enabled_slots=["r0c0", "r0c1", "r0c2"]
    )
    assert masks_empty["op"][LifecycleOp.WAIT]

    # Active slot
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
    }
    masks_active = compute_action_masks(slot_states, enabled_slots=["r0c1"])
    assert masks_active["op"][LifecycleOp.WAIT]


def test_compute_action_masks_blueprint_blend_masks():
    """Blueprint mask excludes NOOP (0 params); blend mask is all-valid."""
    from esper.leyline.factored_actions import BlueprintAction

    slot_states = {"r0c0": None, "r0c1": None, "r0c2": None}
    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # NOOP is masked (0 trainable parameters), others are valid
    assert not masks["blueprint"][BlueprintAction.NOOP]
    assert masks["blueprint"][BlueprintAction.CONV_LIGHT]
    assert masks["blueprint"][BlueprintAction.ATTENTION]
    assert masks["blueprint"][BlueprintAction.NORM]
    assert masks["blueprint"][BlueprintAction.DEPTHWISE]

    # All blends should be valid
    assert masks["blend"].all()


def test_compute_action_masks_min_cull_age():
    """CULL should be blocked if seed_age < MIN_CULL_AGE."""
    # Seed age 0 (just germinated)
    slot_states_age0 = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=0,
        ),
    }
    masks_age0 = compute_action_masks(slot_states_age0, enabled_slots=["r0c1"])
    assert not masks_age0["op"][LifecycleOp.CULL]

    # Seed age 1 (minimum for cull)
    slot_states_age1 = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=1,
        ),
    }
    masks_age1 = compute_action_masks(slot_states_age1, enabled_slots=["r0c1"])
    assert masks_age1["op"][LifecycleOp.CULL]


def test_compute_batch_masks():
    """Should compute masks for a batch of observations."""
    batch_slot_states = [
        # Env 0: empty slots
        {"r0c0": None, "r0c1": None, "r0c2": None},
        # Env 1: r0c1 slot active in TRAINING
        {
            "r0c0": None,
            "r0c1": MaskSeedInfo(
                stage=SeedStage.TRAINING.value,
                seed_age_epochs=5,
            ),
            "r0c2": None,
        },
    ]

    masks = compute_batch_masks(batch_slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # Check shapes (NUM_OPS=4 now)
    assert masks["slot"].shape == (2, 3)
    assert masks["blueprint"].shape == (2, 5)
    assert masks["blend"].shape == (2, 3)
    assert masks["op"].shape == (2, NUM_OPS)

    # WAIT always valid for both
    assert masks["op"][0, LifecycleOp.WAIT]
    assert masks["op"][1, LifecycleOp.WAIT]

    # Env 0: can GERMINATE, not CULL/FOSSILIZE
    assert masks["op"][0, LifecycleOp.GERMINATE]
    assert not masks["op"][0, LifecycleOp.CULL]
    assert not masks["op"][0, LifecycleOp.FOSSILIZE]

    # Env 1: can GERMINATE (empty slots), CULL; not FOSSILIZE
    assert masks["op"][1, LifecycleOp.GERMINATE]
    assert masks["op"][1, LifecycleOp.CULL]
    assert not masks["op"][1, LifecycleOp.FOSSILIZE]


def test_compute_action_masks_at_seed_limit():
    """GERMINATE should be masked when at hard seed limit."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    # At limit: 10 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["r0c1"], total_seeds=10, max_seeds=10
    )

    # GERMINATE should be masked
    assert not masks["op"][LifecycleOp.GERMINATE]

    # Other ops should be unaffected
    assert masks["op"][LifecycleOp.WAIT]
    assert not masks["op"][LifecycleOp.CULL]  # no seed


def test_compute_action_masks_over_seed_limit():
    """GERMINATE should be masked when over hard seed limit."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    # Over limit: 15 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["r0c1"], total_seeds=15, max_seeds=10
    )

    # GERMINATE should be masked
    assert not masks["op"][LifecycleOp.GERMINATE]
    assert masks["op"][LifecycleOp.WAIT]


def test_compute_action_masks_under_seed_limit():
    """GERMINATE should be allowed when under hard seed limit."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    # Under limit: 5 seeds consumed, max is 10
    masks = compute_action_masks(
        slot_states, enabled_slots=["r0c1"], total_seeds=5, max_seeds=10
    )

    # GERMINATE should be allowed (empty slot + under limit)
    assert masks["op"][LifecycleOp.GERMINATE]
    assert masks["op"][LifecycleOp.WAIT]
    assert not masks["op"][LifecycleOp.CULL]  # no seed


def test_compute_action_masks_seed_limit_with_active_seed():
    """Seed limit should only affect GERMINATE, not WAIT/CULL."""
    # Active slot with seed
    slot_states = {
        "r0c0": None,
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
        "r0c2": None,
    }

    # At limit with active slot
    masks = compute_action_masks(
        slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"], total_seeds=10, max_seeds=10
    )

    # GERMINATE masked due to limit (even though empty slots exist)
    assert not masks["op"][LifecycleOp.GERMINATE]

    # CULL should still be valid
    assert masks["op"][LifecycleOp.CULL]
    assert masks["op"][LifecycleOp.WAIT]


def test_compute_batch_masks_with_seed_limits():
    """Batch masks should respect per-env seed limits."""
    batch_slot_states = [
        # Env 0: empty slots, under limit
        {"r0c0": None, "r0c1": None, "r0c2": None},
        # Env 1: empty slots, at limit
        {"r0c0": None, "r0c1": None, "r0c2": None},
        # Env 2: empty slots, over limit
        {"r0c0": None, "r0c1": None, "r0c2": None},
    ]

    masks = compute_batch_masks(
        batch_slot_states,
        enabled_slots=["r0c1"],
        total_seeds_list=[5, 10, 15],
        max_seeds=10,
    )

    # Env 0: under limit, can GERMINATE
    assert masks["op"][0, LifecycleOp.GERMINATE]

    # Env 1: at limit, cannot GERMINATE
    assert not masks["op"][1, LifecycleOp.GERMINATE]

    # Env 2: over limit, cannot GERMINATE
    assert not masks["op"][2, LifecycleOp.GERMINATE]

    # WAIT should be valid for all
    assert masks["op"][0, LifecycleOp.WAIT]
    assert masks["op"][1, LifecycleOp.WAIT]
    assert masks["op"][2, LifecycleOp.WAIT]


def test_compute_batch_masks_seed_limit_default():
    """When total_seeds not provided, should default to 0 (under limit)."""
    batch_slot_states = [
        {"r0c0": None, "r0c1": None, "r0c2": None},
    ]

    # No total_seeds_list provided - should default to 0
    masks = compute_batch_masks(batch_slot_states, enabled_slots=["r0c1"], max_seeds=10)

    # Should allow GERMINATE (0 < 10)
    assert masks["op"][0, LifecycleOp.GERMINATE]


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
        "r0c0": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,  # Not fossilizable
            seed_age_epochs=10,
        ),
        "r0c1": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,  # Fossilizable
            seed_age_epochs=10,
        ),
    }

    # Both slots enabled - FOSSILIZE valid because r0c1 is PROBATIONARY
    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks["op"][LifecycleOp.FOSSILIZE]


def test_compute_action_masks_cull_any_slot():
    """CULL should be valid if ANY enabled slot has a cullable seed."""
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=0,  # Too young to cull
        ),
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # Old enough to cull
        ),
    }

    # Both slots enabled - CULL valid because r0c1 has age >= MIN_CULL_AGE
    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks["op"][LifecycleOp.CULL]


def test_compute_action_masks_germinate_any_empty_slot():
    """GERMINATE should be valid if ANY enabled slot is empty."""
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,
        ),
        "r0c1": None,  # Empty
    }

    # Both slots enabled - GERMINATE valid because r0c1 is empty
    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks["op"][LifecycleOp.GERMINATE]


def test_compute_action_masks_only_enabled_slots_checked():
    """Only enabled slots should affect op validity."""
    slot_states = {
        "r0c0": None,  # Empty but not enabled
        "r0c1": MaskSeedInfo(
            stage=SeedStage.FOSSILIZED.value,  # Not germinate-able
            seed_age_epochs=20,
        ),
    }

    # Only r0c1 enabled - GERMINATE invalid (r0c1 is occupied)
    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])
    assert not masks["op"][LifecycleOp.GERMINATE]

    # Both enabled - GERMINATE valid (r0c0 is empty)
    masks_both = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks_both["op"][LifecycleOp.GERMINATE]


def test_compute_action_masks_enabled_slots_required():
    """enabled_slots is required - raises TypeError if not provided."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    # enabled_slots is required, not optional
    with pytest.raises(TypeError):
        compute_action_masks(slot_states)  # Missing enabled_slots


def test_compute_batch_masks_enabled_slots_required():
    """enabled_slots is required for batch masks."""
    batch_slot_states = [{"r0c1": None}]

    with pytest.raises(TypeError):
        compute_batch_masks(batch_slot_states)  # Missing enabled_slots


def test_slot_id_to_index_canonical():
    """slot_id_to_index should accept canonical slot IDs and reject legacy names."""
    # Canonical IDs should work
    assert slot_id_to_index("r0c0") == 0
    assert slot_id_to_index("r0c1") == 1
    assert slot_id_to_index("r0c2") == 2

    # Legacy names should raise ValueError with helpful message
    with pytest.raises(ValueError, match="no longer supported"):
        slot_id_to_index("early")

    with pytest.raises(ValueError, match="no longer supported"):
        slot_id_to_index("mid")

    with pytest.raises(ValueError, match="no longer supported"):
        slot_id_to_index("late")


# =============================================================================
# Tests for SlotConfig integration
# =============================================================================


def test_compute_action_masks_with_slot_config_default():
    """SlotConfig.default() should work identically to NUM_SLOTS."""
    from esper.leyline.slot_config import SlotConfig

    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    masks = compute_action_masks(
        slot_states,
        enabled_slots=["r0c0", "r0c1", "r0c2"],
        slot_config=SlotConfig.default(),
    )

    # Should have 3 slots
    assert masks["slot"].shape[0] == 3
    assert masks["slot"][0]  # r0c0
    assert masks["slot"][1]  # r0c1
    assert masks["slot"][2]  # r0c2


def test_compute_action_masks_with_slot_config_two_slots():
    """SlotConfig with 2 slots should create mask with 2 slots."""
    from esper.leyline.slot_config import SlotConfig

    slot_states = {
        "r0c0": None,
        "r0c1": None,
    }

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    masks = compute_action_masks(
        slot_states,
        enabled_slots=["r0c0", "r0c1"],
        slot_config=slot_config,
    )

    # Should have 2 slots
    assert masks["slot"].shape[0] == 2
    assert masks["slot"][0]  # r0c0
    assert masks["slot"][1]  # r0c1


def test_compute_action_masks_with_slot_config_five_slots():
    """SlotConfig with 5 slots should create mask with 5 slots."""
    from esper.leyline.slot_config import SlotConfig

    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
        "r0c3": None,
        "r0c4": None,
    }

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r0c3", "r0c4"))
    masks = compute_action_masks(
        slot_states,
        enabled_slots=["r0c1", "r0c3"],
        slot_config=slot_config,
    )

    # Should have 5 slots
    assert masks["slot"].shape[0] == 5
    # Only enabled slots should be masked
    assert not masks["slot"][0]  # r0c0 not enabled
    assert masks["slot"][1]  # r0c1 enabled
    assert not masks["slot"][2]  # r0c2 not enabled
    assert masks["slot"][3]  # r0c3 enabled
    assert not masks["slot"][4]  # r0c4 not enabled


def test_compute_action_masks_slot_config_none_defaults():
    """When slot_config is None, should default to SlotConfig.default()."""
    slot_states = {
        "r0c0": None,
        "r0c1": None,
        "r0c2": None,
    }

    # Pass None explicitly (or omit parameter)
    masks = compute_action_masks(
        slot_states,
        enabled_slots=["r0c0", "r0c1", "r0c2"],
        slot_config=None,
    )

    # Should use default 3 slots
    assert masks["slot"].shape[0] == 3


def test_compute_batch_masks_with_slot_config():
    """Batch masks should respect slot_config."""
    from esper.leyline.slot_config import SlotConfig

    batch_slot_states = [
        {"r0c0": None, "r0c1": None},
        {"r0c0": None, "r0c1": None},
    ]

    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    masks = compute_batch_masks(
        batch_slot_states,
        enabled_slots=["r0c0", "r0c1"],
        slot_config=slot_config,
    )

    # Should have 2 slots per batch element
    assert masks["slot"].shape == (2, 2)


# =============================================================================
# Tests for build_slot_states() helper function
# =============================================================================


class TestBuildSlotStates:
    """Tests for build_slot_states() helper function."""

    def test_empty_model_returns_none_states(self):
        """Empty slots return None for each slot."""
        from esper.simic.action_masks import build_slot_states
        from esper.leyline import SeedMetrics, SeedStage, SeedStateReport

        slot_reports = {
            "r0c1": SeedStateReport(stage=SeedStage.DORMANT, metrics=SeedMetrics(epochs_total=0)),
        }

        result = build_slot_states(slot_reports, ["r0c1"])
        assert result == {"r0c1": None}

    def test_active_seed_returns_mask_seed_info(self):
        """Active seed returns MaskSeedInfo with correct stage and age."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedMetrics, SeedStage, SeedStateReport

        slot_reports = {
            "r0c1": SeedStateReport(stage=SeedStage.TRAINING, metrics=SeedMetrics(epochs_total=5)),
        }

        result = build_slot_states(slot_reports, ["r0c1"])

        assert "r0c1" in result
        assert isinstance(result["r0c1"], MaskSeedInfo)
        assert result["r0c1"].stage == SeedStage.TRAINING.value
        assert result["r0c1"].seed_age_epochs == 5

    def test_multiple_slots(self):
        """Multiple slots are all processed."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedMetrics, SeedStage, SeedStateReport

        slot_reports = {
            "r0c1": SeedStateReport(stage=SeedStage.BLENDING, metrics=SeedMetrics(epochs_total=3)),
        }

        result = build_slot_states(slot_reports, ["r0c0", "r0c1", "r0c2"])

        assert result["r0c0"] is None
        assert isinstance(result["r0c1"], MaskSeedInfo)
        assert result["r0c2"] is None


# =============================================================================
# Edge Case Tests for Action Masking (Phase 2.2)
# =============================================================================


class TestActionMaskEdgeCases:
    """Edge case tests for action masking boundary conditions."""

    def test_all_slots_disabled_only_wait_valid(self):
        """When no slots are enabled, only WAIT should be valid."""
        from esper.leyline.slot_config import SlotConfig

        slot_states = {
            "r0c0": None,
            "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5),
            "r0c2": None,
        }

        masks = compute_action_masks(
            slot_states,
            enabled_slots=[],  # No slots enabled
            slot_config=SlotConfig.default(),
        )

        # No slots selectable
        assert not masks["slot"].any(), "No slots should be selectable when none enabled"

        # Only WAIT should be valid
        assert masks["op"][LifecycleOp.WAIT], "WAIT must always be valid"
        assert not masks["op"][LifecycleOp.GERMINATE], "GERMINATE requires enabled empty slot"
        assert not masks["op"][LifecycleOp.CULL], "CULL requires enabled slot with seed"
        assert not masks["op"][LifecycleOp.FOSSILIZE], "FOSSILIZE requires enabled PROBATIONARY"

    def test_large_config_9_slots(self):
        """9-slot (3x3) grid should mask correctly."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=3, cols=3)

        # Mix of states
        slot_states = {
            "r0c0": None,
            "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5),
            "r0c2": None,
            "r1c0": MaskSeedInfo(stage=SeedStage.PROBATIONARY.value, seed_age_epochs=10),
            "r1c1": None,
            "r1c2": MaskSeedInfo(stage=SeedStage.BLENDING.value, seed_age_epochs=3),
            "r2c0": None,
            "r2c1": None,
            "r2c2": MaskSeedInfo(stage=SeedStage.FOSSILIZED.value, seed_age_epochs=20),
        }

        # Enable all slots
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(slot_config.slot_ids),
            slot_config=slot_config,
        )

        # Verify dimensions
        assert masks["slot"].shape[0] == 9
        assert masks["op"].shape[0] == NUM_OPS

        # All enabled slots should be selectable
        assert masks["slot"].all(), "All 9 slots should be selectable when all enabled"

        # Ops check
        assert masks["op"][LifecycleOp.WAIT]  # Always valid
        assert masks["op"][LifecycleOp.GERMINATE]  # Empty slots exist
        assert masks["op"][LifecycleOp.CULL]  # Seeds with age >= 1 exist
        assert masks["op"][LifecycleOp.FOSSILIZE]  # PROBATIONARY seed exists

    def test_large_config_25_slots(self):
        """25-slot (5x5) grid should mask correctly."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=5, cols=5)

        # All slots empty
        slot_states = {slot_id: None for slot_id in slot_config.slot_ids}

        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(slot_config.slot_ids),
            slot_config=slot_config,
        )

        # Verify dimensions
        assert masks["slot"].shape[0] == 25

        # GERMINATE should be valid (all empty)
        assert masks["op"][LifecycleOp.GERMINATE]
        assert masks["op"][LifecycleOp.WAIT]

        # No seeds means no CULL/FOSSILIZE
        assert not masks["op"][LifecycleOp.CULL]
        assert not masks["op"][LifecycleOp.FOSSILIZE]

    def test_seed_age_exactly_min_cull_age(self):
        """Seed age exactly at MIN_CULL_AGE boundary should allow CULL."""
        slot_states = {
            "r0c1": MaskSeedInfo(
                stage=SeedStage.GERMINATED.value,
                seed_age_epochs=MIN_CULL_AGE,  # Exactly at boundary
            ),
        }

        masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

        assert masks["op"][LifecycleOp.CULL], f"CULL should be valid at age {MIN_CULL_AGE}"

    def test_seed_age_one_below_min_cull_age(self):
        """Seed age one below MIN_CULL_AGE should block CULL."""
        slot_states = {
            "r0c1": MaskSeedInfo(
                stage=SeedStage.GERMINATED.value,
                seed_age_epochs=MIN_CULL_AGE - 1,  # Just below boundary
            ),
        }

        masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

        assert not masks["op"][LifecycleOp.CULL], f"CULL should be blocked at age {MIN_CULL_AGE - 1}"

    def test_single_slot_all_ops_isolated(self):
        """Single slot should correctly isolate all operations."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig(slot_ids=("r0c0",))

        # Test with PROBATIONARY seed (allows FOSSILIZE and CULL)
        slot_states = {
            "r0c0": MaskSeedInfo(stage=SeedStage.PROBATIONARY.value, seed_age_epochs=5),
        }

        masks = compute_action_masks(
            slot_states,
            enabled_slots=["r0c0"],
            slot_config=slot_config,
        )

        # Single slot should be selectable
        assert masks["slot"].shape[0] == 1
        assert masks["slot"][0]

        # WAIT always valid
        assert masks["op"][LifecycleOp.WAIT]

        # GERMINATE blocked (slot occupied)
        assert not masks["op"][LifecycleOp.GERMINATE]

        # FOSSILIZE valid (PROBATIONARY)
        assert masks["op"][LifecycleOp.FOSSILIZE]

        # CULL valid (age >= MIN_CULL_AGE)
        assert masks["op"][LifecycleOp.CULL]

    def test_seed_limit_exactly_zero_unlimited(self):
        """max_seeds=0 should mean unlimited (no blocking)."""
        slot_states = {
            "r0c0": None,
            "r0c1": None,
            "r0c2": None,
        }

        # total_seeds=1000, max_seeds=0 (unlimited)
        masks = compute_action_masks(
            slot_states,
            enabled_slots=["r0c1"],
            total_seeds=1000,
            max_seeds=0,
        )

        # GERMINATE should still be valid (0 = unlimited)
        assert masks["op"][LifecycleOp.GERMINATE]

    def test_partial_enabled_slots_mask_correctly(self):
        """When only some slots enabled, mask should only enable those."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=2, cols=3)  # 6 slots

        slot_states = {slot_id: None for slot_id in slot_config.slot_ids}

        # Enable only r0c0 and r1c2 (first and last)
        masks = compute_action_masks(
            slot_states,
            enabled_slots=["r0c0", "r1c2"],
            slot_config=slot_config,
        )

        # Check individual slot masks
        expected = [True, False, False, False, False, True]  # r0c0 and r1c2
        for i, exp in enumerate(expected):
            assert masks["slot"][i].item() == exp, f"Slot {i} ({slot_config.slot_ids[i]}) mask mismatch"

    def test_all_stages_fossilize_conditions(self):
        """Test FOSSILIZE masking for all seed stages."""
        # Only PROBATIONARY should allow FOSSILIZE
        fossilizable_stages = {SeedStage.PROBATIONARY}
        all_stages = [
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.PROBATIONARY,
            SeedStage.FOSSILIZED,
        ]

        for stage in all_stages:
            slot_states = {
                "r0c1": MaskSeedInfo(stage=stage.value, seed_age_epochs=10),
            }

            masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

            expected = stage in fossilizable_stages
            actual = masks["op"][LifecycleOp.FOSSILIZE].item()
            assert actual == expected, (
                f"FOSSILIZE mask for {stage.name}: expected {expected}, got {actual}"
            )

    def test_all_stages_cull_conditions(self):
        """Test CULL masking for all seed stages (with sufficient age)."""
        # Stages that can transition to CULLED
        cullable_stages = {
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.PROBATIONARY,
        }
        all_stages = [
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.PROBATIONARY,
            SeedStage.FOSSILIZED,
        ]

        for stage in all_stages:
            slot_states = {
                "r0c1": MaskSeedInfo(stage=stage.value, seed_age_epochs=10),  # Age sufficient
            }

            masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

            expected = stage in cullable_stages
            actual = masks["op"][LifecycleOp.CULL].item()
            assert actual == expected, (
                f"CULL mask for {stage.name}: expected {expected}, got {actual}"
            )
