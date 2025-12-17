"""Property-based tests for action masking.

Tests invariants for action masks that must hold for ALL valid inputs:
- WAIT always valid
- Mask dimensions match slot configuration
- GERMINATE requires empty enabled slot
- FOSSILIZE requires PROBATIONARY seed in enabled slot
- CULL requires cullable stage + minimum age
"""

from hypothesis import given, assume, settings
from hypothesis import strategies as st

from esper.leyline import SeedStage, MIN_CULL_AGE
from esper.leyline.slot_config import SlotConfig
from esper.leyline.factored_actions import LifecycleOp, NUM_OPS, NUM_BLUEPRINTS, NUM_BLENDS
from esper.tamiyo.policy.action_masks import (
    compute_action_masks,
    compute_batch_masks,
    MaskSeedInfo,
)
from tests.strategies import (
    slot_configs,
    slot_states_for_config,
    enabled_slots_for_config,
)


class TestWaitAlwaysValid:
    """Property: WAIT is always a valid action regardless of state."""

    @given(
        config=slot_configs(),
        data=st.data(),
    )
    def test_wait_always_enabled_single_env(self, config: SlotConfig, data):
        """Property: op_mask[WAIT] == True for all single-env states."""
        slot_states = data.draw(slot_states_for_config(config))
        enabled = data.draw(enabled_slots_for_config(config))

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.WAIT].item() is True, (
            f"WAIT should always be valid but got {masks['op']}"
        )

    @given(
        config=slot_configs(),
        n_envs=st.integers(min_value=1, max_value=8),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_wait_always_enabled_batch(self, config: SlotConfig, n_envs: int, data):
        """Property: op_mask[:, WAIT] == True for all batched states."""
        batch_states = [data.draw(slot_states_for_config(config)) for _ in range(n_envs)]
        enabled = data.draw(enabled_slots_for_config(config))

        masks = compute_batch_masks(
            batch_slot_states=batch_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        # All envs should have WAIT enabled
        wait_mask = masks["op"][:, LifecycleOp.WAIT]
        assert wait_mask.all(), (
            f"WAIT should be valid for all envs but got {wait_mask}"
        )


class TestMaskDimensions:
    """Property: Mask dimensions match slot configuration."""

    @given(config=slot_configs(max_slots=25))  # Up to 5x5
    def test_slot_mask_dimension(self, config: SlotConfig):
        """Property: slot mask has num_slots dimensions."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["slot"].shape == (config.num_slots,), (
            f"Slot mask shape {masks['slot'].shape} != ({config.num_slots},)"
        )

    @given(config=slot_configs(max_slots=10))
    def test_op_mask_dimension(self, config: SlotConfig):
        """Property: op mask has NUM_OPS dimensions."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"].shape == (NUM_OPS,), (
            f"Op mask shape {masks['op'].shape} != ({NUM_OPS},)"
        )

    @given(config=slot_configs(max_slots=10))
    def test_blueprint_mask_dimension(self, config: SlotConfig):
        """Property: blueprint mask has NUM_BLUEPRINTS dimensions."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["blueprint"].shape == (NUM_BLUEPRINTS,), (
            f"Blueprint mask shape {masks['blueprint'].shape} != ({NUM_BLUEPRINTS},)"
        )

    @given(config=slot_configs(max_slots=10))
    def test_blend_mask_dimension(self, config: SlotConfig):
        """Property: blend mask has NUM_BLENDS dimensions."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["blend"].shape == (NUM_BLENDS,), (
            f"Blend mask shape {masks['blend'].shape} != ({NUM_BLENDS},)"
        )

    @given(
        config=slot_configs(max_slots=10),
        n_envs=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=50)
    def test_batch_mask_dimensions(self, config: SlotConfig, n_envs: int):
        """Property: batched masks have correct (n_envs, dim) shape."""
        batch_states = [{slot_id: None for slot_id in config.slot_ids} for _ in range(n_envs)]
        enabled = list(config.slot_ids)

        masks = compute_batch_masks(
            batch_slot_states=batch_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["slot"].shape == (n_envs, config.num_slots)
        assert masks["op"].shape == (n_envs, NUM_OPS)
        assert masks["blueprint"].shape == (n_envs, NUM_BLUEPRINTS)
        assert masks["blend"].shape == (n_envs, NUM_BLENDS)


class TestGerminateRequiresEmptySlot:
    """Property: GERMINATE valid iff there exists an empty enabled slot (and under seed limit)."""

    @given(config=slot_configs())
    def test_germinate_enabled_when_empty_slot_exists(self, config: SlotConfig):
        """Property: GERMINATE enabled when at least one enabled slot is empty."""
        # All slots empty
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.GERMINATE].item() is True, (
            "GERMINATE should be enabled when empty slots exist"
        )

    @given(config=slot_configs())
    def test_germinate_disabled_when_all_slots_full(self, config: SlotConfig):
        """Property: GERMINATE disabled when all enabled slots are occupied."""
        # All slots occupied with TRAINING seeds
        slot_states = {
            slot_id: MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5)
            for slot_id in config.slot_ids
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.GERMINATE].item() is False, (
            "GERMINATE should be disabled when all enabled slots are full"
        )

    @given(config=slot_configs())
    def test_germinate_disabled_at_seed_limit(self, config: SlotConfig):
        """Property: GERMINATE disabled when at max_seeds limit."""
        # One empty slot but at seed limit
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            total_seeds=3,
            max_seeds=3,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.GERMINATE].item() is False, (
            "GERMINATE should be disabled when at max_seeds limit"
        )

    @given(
        config=slot_configs(),
        data=st.data(),
    )
    def test_germinate_respects_enabled_slots_only(self, config: SlotConfig, data):
        """Property: GERMINATE only considers enabled slots, not all slots."""
        assume(config.num_slots >= 2)

        # First slot empty, rest full
        slot_states = {
            slot_id: (
                None if i == 0
                else MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5)
            )
            for i, slot_id in enumerate(config.slot_ids)
        }

        # Enable only non-empty slots
        enabled = list(config.slot_ids[1:])

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        # GERMINATE should be disabled because no ENABLED slots are empty
        assert masks["op"][LifecycleOp.GERMINATE].item() is False, (
            "GERMINATE should be disabled when no ENABLED slots are empty"
        )


class TestFossilizeRequiresProbationary:
    """Property: FOSSILIZE valid iff there exists a PROBATIONARY seed in enabled slot."""

    @given(config=slot_configs())
    def test_fossilize_enabled_with_probationary_seed(self, config: SlotConfig):
        """Property: FOSSILIZE enabled when PROBATIONARY seed exists in enabled slot."""
        slot_states = {
            slot_id: (
                MaskSeedInfo(stage=SeedStage.PROBATIONARY.value, seed_age_epochs=10)
                if i == 0 else None
            )
            for i, slot_id in enumerate(config.slot_ids)
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.FOSSILIZE].item() is True, (
            "FOSSILIZE should be enabled when PROBATIONARY seed exists"
        )

    @given(config=slot_configs())
    def test_fossilize_disabled_without_probationary(self, config: SlotConfig):
        """Property: FOSSILIZE disabled when no PROBATIONARY seed in enabled slots."""
        # Seeds in TRAINING stage (not PROBATIONARY)
        slot_states = {
            slot_id: MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5)
            for slot_id in config.slot_ids
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.FOSSILIZE].item() is False, (
            "FOSSILIZE should be disabled without PROBATIONARY seed"
        )

    @given(config=slot_configs())
    def test_fossilize_disabled_with_empty_slots(self, config: SlotConfig):
        """Property: FOSSILIZE disabled when all enabled slots are empty."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.FOSSILIZE].item() is False, (
            "FOSSILIZE should be disabled when all slots are empty"
        )


class TestCullRequiresCullableStageAndAge:
    """Property: CULL valid iff seed in cullable stage AND age >= MIN_CULL_AGE."""

    # Stages that can transition to CULLED (based on VALID_TRANSITIONS)
    CULLABLE_STAGES = [
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.PROBATIONARY,
    ]

    @given(
        config=slot_configs(),
        stage=st.sampled_from(CULLABLE_STAGES),
    )
    def test_cull_enabled_with_cullable_stage_and_sufficient_age(
        self, config: SlotConfig, stage: SeedStage
    ):
        """Property: CULL enabled when seed in cullable stage with age >= MIN_CULL_AGE."""
        slot_states = {
            slot_id: (
                MaskSeedInfo(stage=stage.value, seed_age_epochs=MIN_CULL_AGE)
                if i == 0 else None
            )
            for i, slot_id in enumerate(config.slot_ids)
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.CULL].item() is True, (
            f"CULL should be enabled for stage {stage.name} with age {MIN_CULL_AGE}"
        )

    @given(
        config=slot_configs(),
        stage=st.sampled_from(CULLABLE_STAGES),
    )
    def test_cull_disabled_with_insufficient_age(
        self, config: SlotConfig, stage: SeedStage
    ):
        """Property: CULL disabled when seed age < MIN_CULL_AGE."""
        slot_states = {
            slot_id: (
                MaskSeedInfo(stage=stage.value, seed_age_epochs=0)  # Below MIN_CULL_AGE
                if i == 0 else None
            )
            for i, slot_id in enumerate(config.slot_ids)
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.CULL].item() is False, (
            f"CULL should be disabled for stage {stage.name} with age 0 < MIN_CULL_AGE"
        )

    @given(config=slot_configs())
    def test_cull_disabled_with_non_cullable_stage(self, config: SlotConfig):
        """Property: CULL disabled when seed in non-cullable stage."""
        # FOSSILIZED is terminal and cannot be culled
        slot_states = {
            slot_id: (
                MaskSeedInfo(stage=SeedStage.FOSSILIZED.value, seed_age_epochs=100)
                if i == 0 else None
            )
            for i, slot_id in enumerate(config.slot_ids)
        }
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.CULL].item() is False, (
            "CULL should be disabled for FOSSILIZED seed"
        )

    @given(config=slot_configs())
    def test_cull_disabled_with_empty_slots(self, config: SlotConfig):
        """Property: CULL disabled when all enabled slots are empty."""
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["op"][LifecycleOp.CULL].item() is False, (
            "CULL should be disabled when all slots are empty"
        )


class TestSlotMaskRespectEnabledSlots:
    """Property: Slot mask only enables slots in the enabled_slots list."""

    @given(
        config=slot_configs(),
        data=st.data(),
    )
    def test_only_enabled_slots_are_selectable(self, config: SlotConfig, data):
        """Property: slot_mask[i] == True iff slot_ids[i] in enabled_slots."""
        enabled = data.draw(enabled_slots_for_config(config))
        slot_states = {slot_id: None for slot_id in config.slot_ids}

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        slot_mask = masks["slot"]
        for i, slot_id in enumerate(config.slot_ids):
            expected = slot_id in enabled
            actual = slot_mask[i].item()
            assert actual == expected, (
                f"Slot {slot_id} (index {i}): expected mask={expected}, got {actual}. "
                f"Enabled slots: {enabled}"
            )

    @given(config=slot_configs())
    def test_all_slots_enabled_means_all_selectable(self, config: SlotConfig):
        """Property: enabling all slots makes all slots selectable."""
        enabled = list(config.slot_ids)
        slot_states = {slot_id: None for slot_id in config.slot_ids}

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["slot"].all(), (
            f"All slots should be selectable when all enabled: {masks['slot']}"
        )

    def test_no_enabled_slots_means_none_selectable(self):
        """Property: enabling no slots makes none selectable (edge case)."""
        config = SlotConfig.default()
        slot_states = {slot_id: None for slot_id in config.slot_ids}

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=[],  # No slots enabled
            slot_config=config,
        )

        assert not masks["slot"].any(), (
            f"No slots should be selectable when none enabled: {masks['slot']}"
        )


class TestBlueprintMask:
    """Property: Blueprint mask disables NOOP (zero-parameter blueprints)."""

    @given(config=slot_configs())
    def test_noop_blueprint_always_disabled(self, config: SlotConfig):
        """Property: BlueprintAction.NOOP is always masked out."""
        from esper.leyline.factored_actions import BlueprintAction

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        assert masks["blueprint"][BlueprintAction.NOOP].item() is False, (
            "NOOP blueprint should always be masked out"
        )

    @given(config=slot_configs())
    def test_non_noop_blueprints_enabled(self, config: SlotConfig):
        """Property: Non-NOOP blueprints are enabled."""
        from esper.leyline.factored_actions import BlueprintAction

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        enabled = list(config.slot_ids)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=enabled,
            slot_config=config,
        )

        for bp in BlueprintAction:
            if bp != BlueprintAction.NOOP:
                assert masks["blueprint"][bp].item() is True, (
                    f"Blueprint {bp.name} should be enabled"
                )
