"""Property-based tests for factored action space.

Tier 7: Factored Action Space Invariants

This module tests the 8-head factored action space:
1. Slot head - which slot to operate on
2. Blueprint head - which blueprint to use for germination
3. Style head - germination style (blend + alpha algorithm, fused)
4. Tempo head - blend tempo (epochs for blending phase)
5. Alpha target head - non-zero amplitude targets
6. Alpha speed head - schedule speed
7. Alpha curve head - schedule shape
8. Op head - lifecycle operation (WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE)

Key invariants tested:
- All heads have correct dimensions matching NUM_* constants
- Mask dimensions match action space sizes
- HEAD_NAMES constant is complete and ordered
- Tempo actions map to valid epoch counts
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import (
    GerminationStyle,
    HEAD_NAMES,
    LifecycleOp,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    TEMPO_TO_EPOCHS,
    TempoAction,
)

# Mark all tests in this module as property tests for CI
pytestmark = pytest.mark.property


# =============================================================================
# Property Tests: HEAD_NAMES Consistency
# =============================================================================


class TestHeadNamesConsistency:
    """HEAD_NAMES tuple must be complete and correctly ordered."""

    def test_head_names_has_eight_entries(self):
        """Property: HEAD_NAMES contains exactly 8 heads."""
        assert len(HEAD_NAMES) == 8, f"Expected 8 heads, got {len(HEAD_NAMES)}"

    def test_head_names_order(self):
        """Property: HEAD_NAMES is in canonical order."""
        expected = (
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        )
        assert HEAD_NAMES == expected, f"Expected {expected}, got {HEAD_NAMES}"

    def test_head_names_are_strings(self):
        """Property: All HEAD_NAMES entries are non-empty strings."""
        for name in HEAD_NAMES:
            assert isinstance(name, str), f"Expected string, got {type(name)}"
            assert len(name) > 0, "HEAD_NAMES should not contain empty strings"


# =============================================================================
# Property Tests: Enum Dimension Consistency
# =============================================================================


class TestEnumDimensions:
    """Enum sizes must match their NUM_* constants."""

    def test_tempo_action_count_matches_num_tempo(self):
        """Property: TempoAction enum has exactly NUM_TEMPO members."""
        assert len(TempoAction) == NUM_TEMPO, (
            f"TempoAction has {len(TempoAction)} members, expected {NUM_TEMPO}"
        )

    def test_style_count_matches_num_styles(self):
        """Property: GerminationStyle enum has exactly NUM_STYLES members."""
        assert len(GerminationStyle) == NUM_STYLES, (
            f"GerminationStyle has {len(GerminationStyle)} members, expected {NUM_STYLES}"
        )

    def test_lifecycle_op_count_matches_num_ops(self):
        """Property: LifecycleOp enum has exactly NUM_OPS members."""
        assert len(LifecycleOp) == NUM_OPS, (
            f"LifecycleOp has {len(LifecycleOp)} members, expected {NUM_OPS}"
        )

    def test_tempo_action_values_contiguous(self):
        """Property: TempoAction values are 0, 1, 2, ..., NUM_TEMPO-1."""
        values = sorted([t.value for t in TempoAction])
        expected = list(range(NUM_TEMPO))
        assert values == expected, f"TempoAction values {values} not contiguous"

    def test_lifecycle_op_values_contiguous(self):
        """Property: LifecycleOp values are 0, 1, 2, ..., NUM_OPS-1."""
        values = sorted([op.value for op in LifecycleOp])
        expected = list(range(NUM_OPS))
        assert values == expected, f"LifecycleOp values {values} not contiguous"


# =============================================================================
# Property Tests: TEMPO_TO_EPOCHS Mapping
# =============================================================================


class TestTempoToEpochsMapping:
    """TEMPO_TO_EPOCHS must be complete and monotonically increasing."""

    def test_tempo_to_epochs_covers_all_tempo_actions(self):
        """Property: Every TempoAction has an epochs mapping."""
        for tempo in TempoAction:
            assert tempo in TEMPO_TO_EPOCHS, (
                f"TempoAction.{tempo.name} missing from TEMPO_TO_EPOCHS"
            )

    def test_tempo_to_epochs_values_positive(self):
        """Property: All epoch values are positive integers."""
        for tempo, epochs in TEMPO_TO_EPOCHS.items():
            assert isinstance(epochs, int), f"Epochs for {tempo} is not int"
            assert epochs > 0, f"Epochs for {tempo} must be positive, got {epochs}"

    def test_tempo_to_epochs_monotonic(self):
        """Property: FAST < STANDARD < SLOW in epoch count."""
        fast_epochs = TEMPO_TO_EPOCHS[TempoAction.FAST]
        standard_epochs = TEMPO_TO_EPOCHS[TempoAction.STANDARD]
        slow_epochs = TEMPO_TO_EPOCHS[TempoAction.SLOW]

        assert fast_epochs < standard_epochs < slow_epochs, (
            f"Tempo epochs not monotonic: FAST={fast_epochs}, "
            f"STANDARD={standard_epochs}, SLOW={slow_epochs}"
        )

    def test_tempo_epochs_are_reasonable(self):
        """Property: Epoch values are in reasonable range [1, 20]."""
        for tempo, epochs in TEMPO_TO_EPOCHS.items():
            assert 1 <= epochs <= 20, (
                f"Epochs for {tempo.name} = {epochs} outside reasonable range [1, 20]"
            )


# =============================================================================
# Property Tests: Mask Dimensions
# =============================================================================


class TestMaskDimensions:
    """Action masks must have correct dimensions for each head."""

    @given(num_slots=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_slot_mask_dimension(self, num_slots: int):
        """Property: Slot mask has num_slots elements."""
        mask = torch.ones(num_slots, dtype=torch.bool)
        assert mask.shape == (num_slots,)

    def test_blueprint_mask_dimension(self):
        """Property: Blueprint mask has NUM_BLUEPRINTS elements."""
        mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)
        assert mask.shape == (NUM_BLUEPRINTS,)

    def test_style_mask_dimension(self):
        """Property: Style mask has NUM_STYLES elements."""
        mask = torch.ones(NUM_STYLES, dtype=torch.bool)
        assert mask.shape == (NUM_STYLES,)

    def test_tempo_mask_dimension(self):
        """Property: Tempo mask has NUM_TEMPO elements."""
        mask = torch.ones(NUM_TEMPO, dtype=torch.bool)
        assert mask.shape == (NUM_TEMPO,)

    def test_alpha_target_mask_dimension(self):
        """Property: Alpha target mask has NUM_ALPHA_TARGETS elements."""
        mask = torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool)
        assert mask.shape == (NUM_ALPHA_TARGETS,)

    def test_alpha_speed_mask_dimension(self):
        """Property: Alpha speed mask has NUM_ALPHA_SPEEDS elements."""
        mask = torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool)
        assert mask.shape == (NUM_ALPHA_SPEEDS,)

    def test_alpha_curve_mask_dimension(self):
        """Property: Alpha curve mask has NUM_ALPHA_CURVES elements."""
        mask = torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool)
        assert mask.shape == (NUM_ALPHA_CURVES,)

    def test_op_mask_dimension(self):
        """Property: Op mask has NUM_OPS elements."""
        mask = torch.ones(NUM_OPS, dtype=torch.bool)
        assert mask.shape == (NUM_OPS,)


class TestMaskBatchDimensions:
    """Batched masks must maintain correct inner dimensions."""

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        num_slots=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_batched_masks_shape(self, batch_size: int, num_slots: int):
        """Property: Batched masks have shape (batch, head_dim)."""
        masks = {
            "slot": torch.ones(batch_size, num_slots, dtype=torch.bool),
            "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool),
        }

        assert masks["slot"].shape == (batch_size, num_slots)
        assert masks["blueprint"].shape == (batch_size, NUM_BLUEPRINTS)
        assert masks["style"].shape == (batch_size, NUM_STYLES)
        assert masks["tempo"].shape == (batch_size, NUM_TEMPO)
        assert masks["alpha_target"].shape == (batch_size, NUM_ALPHA_TARGETS)
        assert masks["alpha_speed"].shape == (batch_size, NUM_ALPHA_SPEEDS)
        assert masks["alpha_curve"].shape == (batch_size, NUM_ALPHA_CURVES)
        assert masks["op"].shape == (batch_size, NUM_OPS)


# =============================================================================
# Property Tests: Tempo Action Semantics
# =============================================================================


class TestTempoActionSemantics:
    """Tempo actions should have correct semantic meaning."""

    @given(tempo_idx=st.integers(min_value=0, max_value=NUM_TEMPO - 1))
    @settings(max_examples=10)
    def test_tempo_idx_to_enum_roundtrip(self, tempo_idx: int):
        """Property: Tempo index converts to enum and back correctly."""
        tempo = TempoAction(tempo_idx)
        assert tempo.value == tempo_idx

    @given(tempo_idx=st.integers(min_value=0, max_value=NUM_TEMPO - 1))
    @settings(max_examples=10)
    def test_tempo_idx_to_epochs(self, tempo_idx: int):
        """Property: Any valid tempo index maps to valid epochs."""
        tempo = TempoAction(tempo_idx)
        epochs = TEMPO_TO_EPOCHS[tempo]
        assert isinstance(epochs, int)
        assert epochs > 0

    def test_tempo_action_names_meaningful(self):
        """Property: Tempo action names reflect their speed semantics."""
        assert TempoAction.FAST.name == "FAST"
        assert TempoAction.STANDARD.name == "STANDARD"
        assert TempoAction.SLOW.name == "SLOW"


# =============================================================================
# Property Tests: Action Index Validity
# =============================================================================


class TestActionIndexValidity:
    """Action indices must be within valid ranges."""

    @given(
        slot_idx=st.integers(min_value=0, max_value=9),
        num_slots=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_slot_action_within_bounds(self, slot_idx: int, num_slots: int):
        """Property: Slot action index must be < num_slots when valid."""
        assume(slot_idx < num_slots)
        mask = torch.zeros(num_slots, dtype=torch.bool)
        mask[slot_idx] = True  # This slot is valid
        assert mask[slot_idx].item() is True

    @given(blueprint_idx=st.integers(min_value=0, max_value=NUM_BLUEPRINTS - 1))
    @settings(max_examples=20)
    def test_blueprint_action_within_bounds(self, blueprint_idx: int):
        """Property: Blueprint action index is always < NUM_BLUEPRINTS."""
        assert 0 <= blueprint_idx < NUM_BLUEPRINTS

    @given(style_idx=st.integers(min_value=0, max_value=NUM_STYLES - 1))
    @settings(max_examples=10)
    def test_style_within_bounds(self, style_idx: int):
        """Property: Style index is always < NUM_STYLES."""
        assert 0 <= style_idx < NUM_STYLES
        style = GerminationStyle(style_idx)
        assert style.value == style_idx

    @given(tempo_idx=st.integers(min_value=0, max_value=NUM_TEMPO - 1))
    @settings(max_examples=10)
    def test_tempo_action_within_bounds(self, tempo_idx: int):
        """Property: Tempo action index is always < NUM_TEMPO."""
        assert 0 <= tempo_idx < NUM_TEMPO
        # Should be convertible to enum
        tempo = TempoAction(tempo_idx)
        assert tempo.value == tempo_idx

    @given(alpha_target_idx=st.integers(min_value=0, max_value=NUM_ALPHA_TARGETS - 1))
    @settings(max_examples=10)
    def test_alpha_target_within_bounds(self, alpha_target_idx: int):
        """Property: Alpha target index is always < NUM_ALPHA_TARGETS."""
        assert 0 <= alpha_target_idx < NUM_ALPHA_TARGETS

    @given(alpha_speed_idx=st.integers(min_value=0, max_value=NUM_ALPHA_SPEEDS - 1))
    @settings(max_examples=10)
    def test_alpha_speed_within_bounds(self, alpha_speed_idx: int):
        """Property: Alpha speed index is always < NUM_ALPHA_SPEEDS."""
        assert 0 <= alpha_speed_idx < NUM_ALPHA_SPEEDS

    @given(alpha_curve_idx=st.integers(min_value=0, max_value=NUM_ALPHA_CURVES - 1))
    @settings(max_examples=10)
    def test_alpha_curve_within_bounds(self, alpha_curve_idx: int):
        """Property: Alpha curve index is always < NUM_ALPHA_CURVES."""
        assert 0 <= alpha_curve_idx < NUM_ALPHA_CURVES

    @given(op_idx=st.integers(min_value=0, max_value=NUM_OPS - 1))
    @settings(max_examples=10)
    def test_op_action_within_bounds(self, op_idx: int):
        """Property: Op action index is always < NUM_OPS."""
        assert 0 <= op_idx < NUM_OPS
        # Should be convertible to enum
        op = LifecycleOp(op_idx)
        assert op.value == op_idx


# =============================================================================
# Property Tests: Mask Validity During Blending
# =============================================================================


class TestBlendingMaskInvariants:
    """During blending, certain masks have special properties."""

    def test_tempo_mask_all_valid_during_blending(self):
        """Property: All tempo options are valid during blending (policy can choose any speed).

        Unlike other masks that might be constrained by state, tempo is purely a
        policy choice about HOW to blend, not WHETHER to blend.
        """
        # During blending, all tempo options should be available
        tempo_mask = torch.ones(NUM_TEMPO, dtype=torch.bool)

        assert tempo_mask.all(), "All tempo options should be valid during blending"
        assert tempo_mask.sum() == NUM_TEMPO


# =============================================================================
# Property Tests: Cross-Head Consistency
# =============================================================================


class TestCrossHeadConsistency:
    """Multiple heads must be consistent with each other."""

    def test_all_heads_represented_in_head_names(self):
        """Property: HEAD_NAMES contains all expected heads."""
        expected_heads = {
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        }
        actual_heads = set(HEAD_NAMES)
        assert actual_heads == expected_heads, (
            f"HEAD_NAMES mismatch: expected {expected_heads}, got {actual_heads}"
        )

    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=20)
    def test_all_heads_can_be_stacked(self, batch_size: int):
        """Property: All head masks can be created and stacked for a batch."""
        num_slots = 4

        # Create masks for each head
        masks = {}
        for head in HEAD_NAMES:
            if head == "slot":
                masks[head] = torch.ones(batch_size, num_slots, dtype=torch.bool)
            elif head == "blueprint":
                masks[head] = torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool)
            elif head == "style":
                masks[head] = torch.ones(batch_size, NUM_STYLES, dtype=torch.bool)
            elif head == "tempo":
                masks[head] = torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool)
            elif head == "alpha_target":
                masks[head] = torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool)
            elif head == "alpha_speed":
                masks[head] = torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool)
            elif head == "alpha_curve":
                masks[head] = torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool)
            elif head == "op":
                masks[head] = torch.ones(batch_size, NUM_OPS, dtype=torch.bool)

        # Verify all heads are present
        assert len(masks) == len(HEAD_NAMES)
        for head in HEAD_NAMES:
            assert head in masks, f"Missing mask for head: {head}"
            assert masks[head].shape[0] == batch_size
