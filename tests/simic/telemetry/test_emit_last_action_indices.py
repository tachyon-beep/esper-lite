"""Test emit_last_action with raw indices instead of FactoredAction."""

import pytest

from esper.leyline.factored_actions import (
    FactoredAction,
    OP_NAMES,
    BLUEPRINT_IDS,
    NUM_OPS,
    ALPHA_TARGET_VALUES,
    ALPHA_SPEED_NAMES,
    ALPHA_CURVE_NAMES,
    STYLE_NAMES,
    STYLE_ALPHA_ALGORITHMS,
)
from esper.leyline.alpha import AlphaAlgorithm
from esper.simic.telemetry.emitters import emit_last_action


class TestEmitLastActionWithIndices:
    """Verify emit_last_action works with raw indices."""

    def test_emit_with_indices_matches_factored_action(self):
        """Emitting with indices produces same data as with FactoredAction."""
        slot_idx, blueprint_idx, style_idx, tempo_idx = 0, 1, 3, 1
        alpha_target_idx, alpha_speed_idx, alpha_curve_idx = 2, 1, 0
        op_idx = 1  # GERMINATE with CONV_LIGHT, GATED_GATE, STANDARD

        # Create FactoredAction for comparison
        fa = FactoredAction.from_indices(
            slot_idx,
            blueprint_idx,
            style_idx,
            tempo_idx,
            alpha_target_idx,
            alpha_speed_idx,
            alpha_curve_idx,
            op_idx,
        )

        masked = {
            "slot": False,
            "blueprint": True,
            "style": False,
            "tempo": False,
            "alpha_target": False,
            "alpha_speed": False,
            "alpha_curve": False,
            "op": False,
        }

        # Call with indices
        result = emit_last_action(
            env_id=0,
            epoch=5,
            slot_idx=slot_idx,
            blueprint_idx=blueprint_idx,
            style_idx=style_idx,
            tempo_idx=tempo_idx,
            alpha_target_idx=alpha_target_idx,
            alpha_speed_idx=alpha_speed_idx,
            alpha_curve_idx=alpha_curve_idx,
            op_idx=op_idx,
            slot_id="r0c0",
            masked=masked,
            success=True,
        )

        # Verify data matches what FactoredAction would produce
        assert result["op"] == fa.op.name
        assert result["blueprint_id"] == fa.blueprint_id
        assert result["style"] == STYLE_NAMES[style_idx]
        assert result["blend_id"] == fa.blend_algorithm_id
        assert result["alpha_target"] == ALPHA_TARGET_VALUES[alpha_target_idx]
        assert result["alpha_speed"] == ALPHA_SPEED_NAMES[alpha_speed_idx]
        assert result["alpha_curve"] == ALPHA_CURVE_NAMES[alpha_curve_idx]
        assert result["alpha_algorithm"] == fa.alpha_algorithm_value.name
        assert result["alpha_algorithm_selected"] == fa.alpha_algorithm_value.name

    def test_emit_prefers_active_alpha_algorithm_when_provided(self):
        style_idx = 0
        selected = STYLE_ALPHA_ALGORITHMS[style_idx]
        active = AlphaAlgorithm.GATE if selected != AlphaAlgorithm.GATE else AlphaAlgorithm.ADD
        result = emit_last_action(
            env_id=0,
            epoch=1,
            slot_idx=0,
            blueprint_idx=0,
            style_idx=style_idx,
            tempo_idx=0,
            alpha_target_idx=0,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=0,
            slot_id="r0c0",
            masked={
                "slot": False,
                "blueprint": False,
                "style": False,
                "tempo": False,
                "alpha_target": False,
                "alpha_speed": False,
                "alpha_curve": False,
                "op": False,
            },
            success=True,
            active_alpha_algorithm=active.name,
        )

        assert result["alpha_algorithm"] == active.name
        assert result["alpha_algorithm_selected"] == selected.name

    def test_emit_with_all_ops(self):
        """All operation types produce correct op names."""
        for op_idx in range(NUM_OPS):
            result = emit_last_action(
                env_id=0,
                epoch=1,
                slot_idx=0,
                blueprint_idx=0,
                style_idx=0,
                tempo_idx=0,
                alpha_target_idx=0,
                alpha_speed_idx=0,
                alpha_curve_idx=0,
                op_idx=op_idx,
                slot_id="r0c0",
                masked={
                    "slot": False,
                    "blueprint": False,
                    "style": False,
                    "tempo": False,
                    "alpha_target": False,
                    "alpha_speed": False,
                    "alpha_curve": False,
                    "op": False,
                },
                success=True,
            )
            assert result["op"] == OP_NAMES[op_idx]
