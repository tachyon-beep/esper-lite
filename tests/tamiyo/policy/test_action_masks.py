# tests/tamiyo/policy/test_action_masks.py
"""Tests for action masking with multi-slot support.

The mask system only blocks PHYSICALLY IMPOSSIBLE actions:
- SLOT: only enabled slots (from --slots arg) are selectable
- GERMINATE: blocked if ALL enabled slots occupied OR at seed limit
- FOSSILIZE: blocked if NO enabled slot has a HOLDING seed
- PRUNE: blocked if NO enabled slot has a prunable seed with age >= MIN_PRUNE_AGE
         while alpha_mode is HOLD (governor override can bypass)
- WAIT: always valid
- BLUEPRINT: NOOP always blocked (0 trainable parameters)
"""
import pytest

import torch

from esper.tamiyo.policy.action_masks import (
    MaskSeedInfo,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
    build_slot_states,
    MaskedCategorical,
    InvalidStateMachineError,
)
from esper.leyline import (
    AlphaMode,
    AlphaTargetAction,
    GerminationStyle,
    LifecycleOp,
    MIN_PRUNE_AGE,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    SeedStage,
)


def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not PRUNE/FOSSILIZE."""
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

    # No seed means no PRUNE/FOSSILIZE
    assert not masks["op"][LifecycleOp.PRUNE]
    assert not masks["op"][LifecycleOp.FOSSILIZE]
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET]


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
    """Active slot in TRAINING should allow PRUNE, not FOSSILIZE."""
    slot_states = {
        "r0c0": None,
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # >= MIN_PRUNE_AGE
        ),
        "r0c2": None,
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT]

    # GERMINATE still valid (empty slots exist)
    assert masks["op"][LifecycleOp.GERMINATE]

    # PRUNE valid (mid has seed age >= MIN_PRUNE_AGE)
    assert masks["op"][LifecycleOp.PRUNE]

    # FOSSILIZE not valid (no HOLDING seed)
    assert not masks["op"][LifecycleOp.FOSSILIZE]
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET]


def test_compute_action_masks_holding_stage():
    """HOLDING stage should allow FOSSILIZE."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.HOLDING.value,
            seed_age_epochs=10,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # FOSSILIZE valid from HOLDING
    assert masks["op"][LifecycleOp.FOSSILIZE]

    # PRUNE valid (seed exists and age >= 1)
    assert masks["op"][LifecycleOp.PRUNE]
    assert masks["op"][LifecycleOp.SET_ALPHA_TARGET]


def test_compute_action_masks_style_open_on_germinate():
    """Germination style head should be open when GERMINATE is possible."""
    slot_states = {"r0c1": None}

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert masks["op"][LifecycleOp.GERMINATE]
    assert masks["style"].all()


def test_compute_action_masks_alpha_target_open_on_germinate():
    """Alpha target head should be open when GERMINATE is possible."""
    slot_states = {"r0c1": None}

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert masks["op"][LifecycleOp.GERMINATE]
    assert masks["alpha_target"].all()


def test_compute_action_masks_style_defaults_when_no_germinate():
    """Style head should default when GERMINATE is impossible."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.BLENDING.value,
            seed_age_epochs=5,
            alpha_mode=AlphaMode.UP.value,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert not masks["op"][LifecycleOp.GERMINATE]
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET]
    assert masks["style"][GerminationStyle.SIGMOID_ADD]
    assert masks["style"].sum().item() == 1


def test_compute_action_masks_alpha_target_requires_hold_or_germinate():
    """Alpha target changes should be HOLD-only when no GERMINATE is possible."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.BLENDING.value,
            seed_age_epochs=5,
            alpha_mode=AlphaMode.UP.value,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert not masks["op"][LifecycleOp.GERMINATE]
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET]
    assert masks["alpha_target"][AlphaTargetAction.FULL]
    assert masks["alpha_target"].sum().item() == 1


def test_compute_action_masks_style_open_on_hold_retarget():
    """Style head should be open when HOLD retargeting is allowed."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.BLENDING.value,
            seed_age_epochs=5,
            alpha_mode=AlphaMode.HOLD.value,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert masks["op"][LifecycleOp.SET_ALPHA_TARGET]
    assert masks["style"].all()


def test_compute_action_masks_alpha_target_open_on_hold_retarget():
    """Alpha target head should be open when HOLD retargeting is allowed."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.BLENDING.value,
            seed_age_epochs=5,
            alpha_mode=AlphaMode.HOLD.value,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    assert masks["op"][LifecycleOp.SET_ALPHA_TARGET]
    assert masks["alpha_target"].all()


def test_compute_action_masks_fossilized_stage():
    """FOSSILIZED stage should not allow GERMINATE, FOSSILIZE, or PRUNE."""
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

    # No PRUNE - FOSSILIZED is terminal success, cannot be removed
    assert not masks["op"][LifecycleOp.PRUNE]
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET]


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


def test_compute_action_masks_blueprint_style_masks():
    """Blueprint mask excludes NOOP (0 params); style mask is all-valid when GERMINATE possible."""
    from esper.leyline import BlueprintAction

    slot_states = {"r0c0": None, "r0c1": None, "r0c2": None}
    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

    # NOOP is masked (0 trainable parameters), others are valid
    assert not masks["blueprint"][BlueprintAction.NOOP]
    assert masks["blueprint"][BlueprintAction.CONV_LIGHT]
    assert masks["blueprint"][BlueprintAction.ATTENTION]
    assert masks["blueprint"][BlueprintAction.NORM]
    assert masks["blueprint"][BlueprintAction.DEPTHWISE]

    # All styles should be valid when GERMINATE is possible
    assert masks["style"].all()


def test_compute_action_masks_min_prune_age():
    """PRUNE should be blocked if seed_age < MIN_PRUNE_AGE."""
    # Seed age 0 (just germinated)
    slot_states_age0 = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=0,
        ),
    }
    masks_age0 = compute_action_masks(slot_states_age0, enabled_slots=["r0c1"])
    assert not masks_age0["op"][LifecycleOp.PRUNE]

    # Seed age 1 (minimum for cull)
    slot_states_age1 = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.GERMINATED.value,
            seed_age_epochs=1,
        ),
    }
    masks_age1 = compute_action_masks(slot_states_age1, enabled_slots=["r0c1"])
    assert masks_age1["op"][LifecycleOp.PRUNE]


def test_compute_action_masks_prune_requires_hold():
    """PRUNE should be blocked if alpha_mode is not HOLD."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=MIN_PRUNE_AGE,
            alpha_mode=AlphaMode.UP.value,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])
    assert not masks["op"][LifecycleOp.PRUNE]


def test_compute_action_masks_governor_override_allows_prune():
    """Governor override should allow PRUNE even if alpha_mode is not HOLD."""
    slot_states = {
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=MIN_PRUNE_AGE,
            alpha_mode=AlphaMode.DOWN.value,
        ),
    }

    masks = compute_action_masks(
        slot_states,
        enabled_slots=["r0c1"],
        allow_governor_override=True,
    )
    assert masks["op"][LifecycleOp.PRUNE]


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

    # Check shapes (NUM_OPS now includes ADVANCE)
    assert masks["slot"].shape == (2, 3)
    assert masks["blueprint"].shape == (2, NUM_BLUEPRINTS)
    assert masks["style"].shape == (2, NUM_STYLES)
    assert masks["tempo"].shape == (2, 3)
    assert masks["alpha_target"].shape == (2, NUM_ALPHA_TARGETS)
    assert masks["alpha_speed"].shape == (2, NUM_ALPHA_SPEEDS)
    assert masks["alpha_curve"].shape == (2, NUM_ALPHA_CURVES)
    assert masks["op"].shape == (2, NUM_OPS)

    # WAIT always valid for both
    assert masks["op"][0, LifecycleOp.WAIT]
    assert masks["op"][1, LifecycleOp.WAIT]

    # Env 0: can GERMINATE, not PRUNE/FOSSILIZE
    assert masks["op"][0, LifecycleOp.GERMINATE]
    assert not masks["op"][0, LifecycleOp.PRUNE]
    assert not masks["op"][0, LifecycleOp.FOSSILIZE]
    assert not masks["op"][0, LifecycleOp.ADVANCE]

    # Env 1: can GERMINATE (empty slots), PRUNE; not FOSSILIZE
    assert masks["op"][1, LifecycleOp.GERMINATE]
    assert masks["op"][1, LifecycleOp.PRUNE]
    assert not masks["op"][1, LifecycleOp.FOSSILIZE]
    assert masks["op"][1, LifecycleOp.ADVANCE]


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
    assert not masks["op"][LifecycleOp.PRUNE]  # no seed


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
    assert not masks["op"][LifecycleOp.PRUNE]  # no seed


def test_compute_action_masks_seed_limit_with_active_seed():
    """Seed limit should only affect GERMINATE, not WAIT/PRUNE."""
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

    # PRUNE should still be valid
    assert masks["op"][LifecycleOp.PRUNE]
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


def test_min_prune_age_constant():
    """MIN_PRUNE_AGE should be 1 (need one epoch for counterfactual)."""
    assert MIN_PRUNE_AGE == 1


# =============================================================================
# Tests for multi-slot semantics (optimistic masking)
# =============================================================================


def test_compute_action_masks_fossilize_any_slot():
    """FOSSILIZE should be valid if ANY enabled slot is HOLDING."""
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,  # Not fossilizable
            seed_age_epochs=10,
        ),
        "r0c1": MaskSeedInfo(
            stage=SeedStage.HOLDING.value,  # Fossilizable
            seed_age_epochs=10,
        ),
    }

    # Both slots enabled - FOSSILIZE valid because r0c1 is HOLDING
    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks["op"][LifecycleOp.FOSSILIZE]


def test_compute_action_masks_prune_any_slot():
    """PRUNE should be valid if ANY enabled slot has a prunable seed."""
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=0,  # Too young to prune
        ),
        "r0c1": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # Old enough to prune
        ),
    }

    # Both slots enabled - PRUNE valid because r0c1 has age >= MIN_PRUNE_AGE
    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1"])
    assert masks["op"][LifecycleOp.PRUNE]


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
            stage=SeedStage.HOLDING.value,
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
        slot_reports = {}

        result = build_slot_states(slot_reports, ["r0c1"])
        assert result == {"r0c1": None}

    def test_dormant_report_is_treated_as_occupied(self):
        """A present report (even if DORMANT) must not be treated as empty."""
        from esper.leyline import SeedMetrics, SeedStage, SeedStateReport

        slot_reports = {
            "r0c1": SeedStateReport(stage=SeedStage.DORMANT, metrics=SeedMetrics(epochs_total=0)),
        }

        result = build_slot_states(slot_reports, ["r0c1"])

        assert isinstance(result["r0c1"], MaskSeedInfo)
        assert result["r0c1"].stage == SeedStage.DORMANT.value

    def test_active_seed_returns_mask_seed_info(self):
        """Active seed returns MaskSeedInfo with correct stage and age."""
        # build_slot_states imported at module level, MaskSeedInfo
        from esper.leyline import AlphaMode, SeedMetrics, SeedStage, SeedStateReport

        slot_reports = {
            "r0c1": SeedStateReport(
                stage=SeedStage.TRAINING,
                metrics=SeedMetrics(epochs_total=5),
                alpha_mode=AlphaMode.DOWN.value,
            ),
        }

        result = build_slot_states(slot_reports, ["r0c1"])

        assert "r0c1" in result
        assert isinstance(result["r0c1"], MaskSeedInfo)
        assert result["r0c1"].stage == SeedStage.TRAINING.value
        assert result["r0c1"].seed_age_epochs == 5
        assert result["r0c1"].alpha_mode == AlphaMode.DOWN.value

    def test_multiple_slots(self):
        """Multiple slots are all processed."""
        # build_slot_states imported at module level, MaskSeedInfo
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
        assert not masks["op"][LifecycleOp.PRUNE], "PRUNE requires enabled slot with seed"
        assert not masks["op"][LifecycleOp.FOSSILIZE], "FOSSILIZE requires enabled HOLDING"

    def test_large_config_9_slots(self):
        """9-slot (3x3) grid should mask correctly."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=3, cols=3)

        # Mix of states
        slot_states = {
            "r0c0": None,
            "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5),
            "r0c2": None,
            "r1c0": MaskSeedInfo(stage=SeedStage.HOLDING.value, seed_age_epochs=10),
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
        assert masks["op"][LifecycleOp.PRUNE]  # Seeds with age >= 1 exist
        assert masks["op"][LifecycleOp.FOSSILIZE]  # HOLDING seed exists

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

        # No seeds means no PRUNE/FOSSILIZE
        assert not masks["op"][LifecycleOp.PRUNE]
        assert not masks["op"][LifecycleOp.FOSSILIZE]

    def test_seed_age_exactly_min_prune_age(self):
        """Seed age exactly at MIN_PRUNE_AGE boundary should allow PRUNE."""
        slot_states = {
            "r0c1": MaskSeedInfo(
                stage=SeedStage.GERMINATED.value,
                seed_age_epochs=MIN_PRUNE_AGE,  # Exactly at boundary
            ),
        }

        masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

        assert masks["op"][LifecycleOp.PRUNE], f"PRUNE should be valid at age {MIN_PRUNE_AGE}"

    def test_seed_age_one_below_min_prune_age(self):
        """Seed age one below MIN_PRUNE_AGE should block PRUNE."""
        slot_states = {
            "r0c1": MaskSeedInfo(
                stage=SeedStage.GERMINATED.value,
                seed_age_epochs=MIN_PRUNE_AGE - 1,  # Just below boundary
            ),
        }

        masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

        assert not masks["op"][LifecycleOp.PRUNE], f"PRUNE should be blocked at age {MIN_PRUNE_AGE - 1}"

    def test_single_slot_all_ops_isolated(self):
        """Single slot should correctly isolate all operations."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig(slot_ids=("r0c0",))

        # Test with HOLDING seed (allows FOSSILIZE and PRUNE)
        slot_states = {
            "r0c0": MaskSeedInfo(stage=SeedStage.HOLDING.value, seed_age_epochs=5),
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

        # FOSSILIZE valid (HOLDING)
        assert masks["op"][LifecycleOp.FOSSILIZE]

        # PRUNE valid (age >= MIN_PRUNE_AGE)
        assert masks["op"][LifecycleOp.PRUNE]

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
        # Only HOLDING should allow FOSSILIZE
        fossilizable_stages = {SeedStage.HOLDING}
        all_stages = [
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.HOLDING,
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

    def test_all_stages_prune_conditions(self):
        """Test PRUNE masking for all seed stages (with sufficient age)."""
        # Stages that can transition to PRUNED
        prunable_stages = {
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.HOLDING,
        }
        all_stages = [
            SeedStage.GERMINATED,
            SeedStage.TRAINING,
            SeedStage.BLENDING,
            SeedStage.HOLDING,
            SeedStage.FOSSILIZED,
        ]

        for stage in all_stages:
            slot_states = {
                "r0c1": MaskSeedInfo(stage=stage.value, seed_age_epochs=10),  # Age sufficient
            }

            masks = compute_action_masks(slot_states, enabled_slots=["r0c1"])

            expected = stage in prunable_stages
            actual = masks["op"][LifecycleOp.PRUNE].item()
            assert actual == expected, (
                f"PRUNE mask for {stage.name}: expected {expected}, got {actual}"
            )


# =============================================================================
# MaskedCategorical Tests
# =============================================================================


class TestMaskedCategorical:
    """Tests for MaskedCategorical distribution with action masking."""

    def test_basic_sampling_with_valid_logits(self):
        """MaskedCategorical should sample correctly with valid logits."""
        logits = torch.tensor([[0.0, 1.0, 2.0]])
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask)
        sample = dist.sample()

        assert sample.shape == (1,)
        assert 0 <= sample.item() < 3

    def test_masked_actions_have_zero_probability(self):
        """Masked actions should have ~0 probability."""
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, True]])

        dist = MaskedCategorical(logits, mask)
        probs = dist.probs

        # Masked action should have near-zero probability
        assert probs[0, 1].item() < 1e-4

    def test_raises_on_nan_logits(self):
        """MaskedCategorical should raise ValueError on NaN logits."""
        logits = torch.tensor([[0.0, float('nan'), 1.0]])
        mask = torch.tensor([[True, True, True]])

        with pytest.raises(ValueError, match="inf/nan"):
            MaskedCategorical(logits, mask)

    def test_raises_on_inf_logits(self):
        """MaskedCategorical should raise ValueError on inf logits."""
        logits = torch.tensor([[0.0, float('inf'), 1.0]])
        mask = torch.tensor([[True, True, True]])

        with pytest.raises(ValueError, match="inf/nan"):
            MaskedCategorical(logits, mask)

    def test_raises_on_neg_inf_logits(self):
        """MaskedCategorical should raise ValueError on -inf logits."""
        logits = torch.tensor([[float('-inf'), 0.0, 1.0]])
        mask = torch.tensor([[True, True, True]])

        with pytest.raises(ValueError, match="inf/nan"):
            MaskedCategorical(logits, mask)

    def test_error_message_includes_stats(self):
        """Error message should include helpful logit statistics."""
        logits = torch.tensor([[0.0, float('nan'), float('inf')]])
        mask = torch.tensor([[True, True, True]])

        with pytest.raises(ValueError) as exc_info:
            MaskedCategorical(logits, mask)

        error_msg = str(exc_info.value)
        assert "network instability" in error_msg
        assert "nan_count=" in error_msg
        assert "inf_count=" in error_msg

    def test_raises_on_all_false_mask(self):
        """MaskedCategorical should raise InvalidStateMachineError on all-false mask."""
        logits = torch.tensor([[0.0, 1.0, 2.0]])
        mask = torch.tensor([[False, False, False]])

        with pytest.raises(InvalidStateMachineError):
            MaskedCategorical(logits, mask)

    def test_entropy_normalized_to_zero_one(self):
        """Entropy should be normalized to [0, 1] range."""
        logits = torch.tensor([[0.0, 0.0, 0.0]])  # Uniform distribution
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask)
        entropy = dist.entropy()

        # Uniform over 3 actions = max entropy
        assert 0.99 <= entropy.item() <= 1.01

    def test_entropy_zero_for_single_valid_action(self):
        """Entropy should be 0 when only one action is valid."""
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, False]])

        dist = MaskedCategorical(logits, mask)
        entropy = dist.entropy()

        # Single valid action = no uncertainty = zero entropy
        assert entropy.item() == 0.0

    def test_log_prob_correct(self):
        """Log probability should be computed correctly."""
        logits = torch.tensor([[0.0, 1.0, 2.0]])
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask)
        log_prob = dist.log_prob(torch.tensor([2]))

        # Action 2 has highest logit, should have highest log_prob
        assert log_prob.item() > -1.0

    def test_fp16_numerical_stability(self):
        """MaskedCategorical should be numerically stable in FP16."""
        logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float16)
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask)

        # Should not produce NaN/inf
        assert not torch.isnan(dist.probs).any()
        assert not torch.isinf(dist.probs).any()
        assert not torch.isnan(dist.entropy()).any()

    def test_bf16_numerical_stability(self):
        """MaskedCategorical should be numerically stable in BF16."""
        logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.bfloat16)
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask)

        # Should not produce NaN/inf
        assert not torch.isnan(dist.probs).any()
        assert not torch.isinf(dist.probs).any()

    def test_batch_processing(self):
        """MaskedCategorical should handle batch inputs correctly."""
        logits = torch.tensor([
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
        ])
        mask = torch.tensor([
            [True, True, True],
            [True, True, True],
        ])

        dist = MaskedCategorical(logits, mask)

        assert dist.probs.shape == (2, 3)
        assert dist.entropy().shape == (2,)

        samples = dist.sample()
        assert samples.shape == (2,)


class TestMaskedCategoricalValidation:
    """Tests for validation toggle behavior."""

    def test_validation_enabled_by_default(self):
        """Validation should be enabled by default for safety."""
        from esper.tamiyo.policy.action_masks import MaskedCategorical
        assert MaskedCategorical.validate is True

    def test_validation_can_be_disabled(self):
        """Validation can be disabled for production performance."""
        from esper.tamiyo.policy.action_masks import MaskedCategorical
        original = MaskedCategorical.validate
        try:
            MaskedCategorical.validate = False
            # This would raise InvalidStateMachineError with validation enabled
            logits = torch.zeros(1, 5)
            mask = torch.zeros(1, 5, dtype=torch.bool)  # All invalid!
            # Should NOT raise when validation is disabled
            dist = MaskedCategorical(logits, mask)
            assert dist is not None
        finally:
            MaskedCategorical.validate = original

    def test_validation_catches_invalid_mask_when_enabled(self):
        """Validation raises error for invalid masks when enabled."""
        from esper.tamiyo.policy.action_masks import (
            MaskedCategorical,
            InvalidStateMachineError,
        )
        MaskedCategorical.validate = True
        logits = torch.zeros(1, 5)
        mask = torch.zeros(1, 5, dtype=torch.bool)  # All invalid!
        with pytest.raises(InvalidStateMachineError):
            MaskedCategorical(logits, mask)


# === Lifecycle Gating Validation Tests (Phase 7) ===


def test_embargo_blocks_germination():
    """Verify EMBARGOED slots block GERMINATE operation.

    Critical test from Phase 7: After a slot is pruned, it enters EMBARGOED stage
    for DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE epochs to prevent thrashing (immediate
    regermination). This test verifies that:
    1. EMBARGOED slots cannot be germinated
    2. GERMINATE is only enabled if at least one slot is DORMANT (not EMBARGOED)
    """
    from esper.leyline import DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE

    # Scenario: All 3 slots are embargoed (recently pruned)
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=2,  # 2 epochs into embargo period
        ),
        "r0c1": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=1,
        ),
        "r0c2": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=0,  # Just pruned
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # GERMINATE should be blocked (no dormant slots available)
    assert not masks["op"][LifecycleOp.GERMINATE], (
        "GERMINATE should be blocked when all slots are embargoed. "
        f"Embargo period is {DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE} epochs."
    )

    # WAIT should still be available
    assert masks["op"][LifecycleOp.WAIT], "WAIT should always be available"


def test_embargoed_vs_dormant_germination():
    """Verify GERMINATE is enabled only for DORMANT slots, not EMBARGOED.

    This test distinguishes between:
    - DORMANT: Empty, available for germination
    - EMBARGOED: Empty but in cooldown period, NOT available

    Both are "empty" (no seed), but only DORMANT allows germination.
    """
    # Scenario: Mix of DORMANT and EMBARGOED slots
    slot_states = {
        "r0c0": None,  # Truly empty (DORMANT)
        "r0c1": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=1,
        ),
        "r0c2": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=3,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0", "r0c1", "r0c2"])

    # GERMINATE should be enabled (r0c0 is dormant/None)
    assert masks["op"][LifecycleOp.GERMINATE], (
        "GERMINATE should be enabled when at least one slot is DORMANT (None). "
        "r0c0 is None (dormant), so germination is possible."
    )

    # Slot mask should allow all slots (policy decides which to target)
    assert masks["slot"][0], "r0c0 (DORMANT) should be selectable"
    assert masks["slot"][1], "r0c1 should be selectable"
    assert masks["slot"][2], "r0c2 should be selectable"


def test_pruned_stage_blocks_operations():
    """Verify PRUNED stage blocks most operations.

    PRUNED is a transient state between execution and EMBARGOED. Seeds in PRUNED
    stage should not be prunable/fossilizable/alpha-settable.
    """
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.PRUNED.value,
            seed_age_epochs=5,
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0"])

    # PRUNE should be blocked (seed is already pruned)
    assert not masks["op"][LifecycleOp.PRUNE], (
        "PRUNE should be blocked for PRUNED stage"
    )

    # FOSSILIZE should be blocked (can only fossilize from HOLDING)
    assert not masks["op"][LifecycleOp.FOSSILIZE], (
        "FOSSILIZE should be blocked for PRUNED stage"
    )

    # SET_ALPHA_TARGET should be blocked (no seed to set alpha for)
    assert not masks["op"][LifecycleOp.SET_ALPHA_TARGET], (
        "SET_ALPHA_TARGET should be blocked for PRUNED stage"
    )

    # WAIT should still be available
    assert masks["op"][LifecycleOp.WAIT], "WAIT should always be available"


def test_embargo_duration_correctness():
    """Verify embargo period matches DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE.

    This test documents the embargo period and ensures it's applied correctly.
    The embargo mechanism is enforced by Kasmina (slot management), not by
    action masks - masks just reflect the current state.
    """
    from esper.leyline import DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE

    # Document expected embargo duration
    assert DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE == 5, (
        "Embargo period should be 5 epochs. If this changed, "
        "update related tests and documentation."
    )

    # Verify slot in embargo blocks germination
    slot_states = {
        "r0c0": MaskSeedInfo(
            stage=SeedStage.EMBARGOED.value,
            seed_age_epochs=2,  # 2/5 epochs through embargo
        ),
    }

    masks = compute_action_masks(slot_states, enabled_slots=["r0c0"])

    # GERMINATE blocked (only slot is embargoed)
    assert not masks["op"][LifecycleOp.GERMINATE], (
        "GERMINATE should be blocked during embargo period"
    )
