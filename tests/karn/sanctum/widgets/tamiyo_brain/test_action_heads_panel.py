"""Tests for ActionHeadsPanel widget."""
from __future__ import annotations

import math


def test_heads_config_matches_leyline_head_max_entropies():
    """HEAD_CONFIG keys must match Leyline's HEAD_MAX_ENTROPIES keys.

    This is enforced by module-level assertion in heads_grid.py, but we test
    it explicitly to document the contract and provide clear failure messages.
    """
    from esper.leyline import HEAD_MAX_ENTROPIES
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import (
        HEAD_CONFIG,
        _get_head_key,
    )

    config_keys = {_get_head_key(ent_field) for _, ent_field, *_ in HEAD_CONFIG}
    leyline_keys = set(HEAD_MAX_ENTROPIES.keys())

    assert config_keys == leyline_keys, (
        f"HEAD_CONFIG and Leyline's HEAD_MAX_ENTROPIES must have matching keys. "
        f"Config has: {config_keys}, Leyline has: {leyline_keys}"
    )


def test_head_max_entropies_computed_from_enum_sizes():
    """HEAD_MAX_ENTROPIES should be ln(N) for each action space enum."""
    from esper.leyline import (
        HEAD_MAX_ENTROPIES,
        LifecycleOp,
        BlueprintAction,
        GerminationStyle,
        TempoAction,
        AlphaTargetAction,
        AlphaSpeedAction,
        AlphaCurveAction,
        DEFAULT_NUM_SLOTS,
    )

    # Verify computed values match expected
    assert abs(HEAD_MAX_ENTROPIES["op"] - math.log(len(LifecycleOp))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["slot"] - math.log(DEFAULT_NUM_SLOTS)) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["blueprint"] - math.log(len(BlueprintAction))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["style"] - math.log(len(GerminationStyle))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["tempo"] - math.log(len(TempoAction))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["alpha_target"] - math.log(len(AlphaTargetAction))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["alpha_speed"] - math.log(len(AlphaSpeedAction))) < 0.001
    assert abs(HEAD_MAX_ENTROPIES["alpha_curve"] - math.log(len(AlphaCurveAction))) < 0.001


def test_wait_op_only_op_head_relevant():
    """When op=WAIT, only the 'op' head should be causally relevant."""
    from esper.leyline import is_head_relevant, HEAD_RELEVANCE_BY_OP

    # WAIT only affects op selection
    assert is_head_relevant("WAIT", "op") is True
    assert is_head_relevant("WAIT", "slot") is False
    assert is_head_relevant("WAIT", "blueprint") is False
    assert is_head_relevant("WAIT", "style") is False
    assert is_head_relevant("WAIT", "tempo") is False
    assert is_head_relevant("WAIT", "alpha_target") is False
    assert is_head_relevant("WAIT", "alpha_speed") is False
    assert is_head_relevant("WAIT", "alpha_curve") is False

    # Verify HEAD_RELEVANCE_BY_OP matches
    assert HEAD_RELEVANCE_BY_OP["WAIT"] == frozenset({"op"})


def test_germinate_op_all_germinate_heads_relevant():
    """When op=GERMINATE, slot/blueprint/style/tempo/alpha_target should be relevant."""
    from esper.leyline import is_head_relevant, HEAD_RELEVANCE_BY_OP

    assert is_head_relevant("GERMINATE", "op") is True
    assert is_head_relevant("GERMINATE", "slot") is True
    assert is_head_relevant("GERMINATE", "blueprint") is True
    assert is_head_relevant("GERMINATE", "style") is True
    assert is_head_relevant("GERMINATE", "tempo") is True
    assert is_head_relevant("GERMINATE", "alpha_target") is True
    # Alpha schedule params not relevant for GERMINATE
    assert is_head_relevant("GERMINATE", "alpha_speed") is False
    assert is_head_relevant("GERMINATE", "alpha_curve") is False

    assert HEAD_RELEVANCE_BY_OP["GERMINATE"] == frozenset({
        "op", "slot", "blueprint", "style", "tempo", "alpha_target"
    })


def test_set_alpha_op_alpha_heads_relevant():
    """When op=SET_ALPHA_TARGET, alpha-related heads should be relevant."""
    from esper.leyline import is_head_relevant, HEAD_RELEVANCE_BY_OP

    assert is_head_relevant("SET_ALPHA_TARGET", "op") is True
    assert is_head_relevant("SET_ALPHA_TARGET", "slot") is True
    assert is_head_relevant("SET_ALPHA_TARGET", "style") is True
    assert is_head_relevant("SET_ALPHA_TARGET", "alpha_target") is True
    assert is_head_relevant("SET_ALPHA_TARGET", "alpha_speed") is True
    assert is_head_relevant("SET_ALPHA_TARGET", "alpha_curve") is True
    # Not relevant for SET_ALPHA
    assert is_head_relevant("SET_ALPHA_TARGET", "blueprint") is False
    assert is_head_relevant("SET_ALPHA_TARGET", "tempo") is False

    assert HEAD_RELEVANCE_BY_OP["SET_ALPHA_TARGET"] == frozenset({
        "op", "slot", "style", "alpha_target", "alpha_speed", "alpha_curve"
    })


def test_prune_op_has_alpha_schedule_heads():
    """When op=PRUNE, slot and alpha_speed/alpha_curve should be relevant."""
    from esper.leyline import is_head_relevant

    assert is_head_relevant("PRUNE", "op") is True
    assert is_head_relevant("PRUNE", "slot") is True
    assert is_head_relevant("PRUNE", "alpha_speed") is True
    assert is_head_relevant("PRUNE", "alpha_curve") is True
    # Not relevant for PRUNE
    assert is_head_relevant("PRUNE", "blueprint") is False
    assert is_head_relevant("PRUNE", "style") is False
    assert is_head_relevant("PRUNE", "tempo") is False
    assert is_head_relevant("PRUNE", "alpha_target") is False


def test_head_state_health_classification():
    """ActionHeadsPanel._head_state should classify based on entropy + gradient health."""
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import ActionHeadsPanel

    panel = ActionHeadsPanel()

    # Healthy: moderate entropy + normal gradients
    state, style = panel._head_state("slot", entropy=0.5, grad_norm=0.5)
    assert state == "●"
    assert style == "green"

    # Dead: low entropy + vanishing gradients
    state, style = panel._head_state("slot", entropy=0.05, grad_norm=0.005)
    assert state == "○"
    assert style == "red"

    # Exploding gradients
    state, style = panel._head_state("slot", entropy=0.5, grad_norm=10.0)
    assert state == "▲"
    assert style == "red bold"

    # Deterministic: low entropy but normal gradients (concerning - multiple slots available)
    state, style = panel._head_state("slot", entropy=0.05, grad_norm=0.5, n_slots=3)
    assert state == "◇"
    assert style == "yellow"

    # Deterministic but expected: slot head with single slot configured
    state, style = panel._head_state("slot", entropy=0.05, grad_norm=0.5, n_slots=1)
    assert state == "◇"
    assert style == "dim"  # Expected determinism, not concerning

    # Deterministic: other head (not slot) remains yellow even if n_slots=1
    state, style = panel._head_state("op", entropy=0.05, grad_norm=0.5, n_slots=1)
    assert state == "◇"
    assert style == "yellow"

    # Confused: very high entropy with normal gradients
    state, style = panel._head_state("slot", entropy=1.8, grad_norm=0.5)
    assert state == "◐"
    assert style == "yellow"


def test_heads_panel_shows_gradient_flow_footer():
    """ActionHeadsPanel should show gradient flow metrics as footer row."""
    from esper.karn.sanctum.schema import (
        GradientQualityMetrics,
        SanctumSnapshot,
        TamiyoState,
    )
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import ActionHeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        gradient_cv=0.42,
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    # Set existing dead/exploding layers (already tracked)
    snapshot.tamiyo.dead_layers = 0
    snapshot.tamiyo.exploding_layers = 0

    panel = ActionHeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should show gradient CV with value and status
    assert "CV:0.42" in content_str
    assert "stable" in content_str
    # Should show layer health
    assert "Dead:" in content_str
    assert "Exploding:" in content_str
    # Should show NaN/Inf counts (always visible, dim when zero)
    assert "NaN:" in content_str
    assert "Inf:" in content_str


def test_heads_panel_shows_nan_inf_indicator_rows():
    """ActionHeadsPanel should show NaN and Inf indicator rows below State row."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import ActionHeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    # Latch NaN on op head, Inf on slot head
    snapshot.tamiyo.head_nan_latch["op"] = True
    snapshot.tamiyo.head_inf_latch["slot"] = True

    panel = ActionHeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should have NaN and Inf row labels
    assert "NaN" in content_str
    assert "Inf" in content_str
    # Should show filled circle (●) for latched heads
    assert "●" in content_str


def test_heads_panel_shows_all_clear_when_no_nan_inf():
    """ActionHeadsPanel should show all empty indicators when no NaN/Inf latched."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import ActionHeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    # All latches are pre-populated with False (default)

    panel = ActionHeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should have NaN and Inf row labels
    assert "NaN" in content_str
    assert "Inf" in content_str
    # Should show only empty circles (no filled circles)
    assert "○" in content_str
    # Count filled circles in NaN/Inf rows only (not State row)
    # The NaN and Inf rows should have zero filled circles
    lines = content_str.split("\n")
    nan_line = next((line for line in lines if line.strip().startswith("NaN")), "")
    inf_line = next((line for line in lines if line.strip().startswith("Inf")), "")
    assert "●" not in nan_line
    assert "●" not in inf_line
