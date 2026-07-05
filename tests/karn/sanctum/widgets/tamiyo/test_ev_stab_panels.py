"""EV-stab critic surfaces: ReturnVariancePanel + per-stream EV lines.

Placement contract (PDR-0028): the value-free Stage-0 shares render with gate
markers (diagnosis semantics), NOT green/red health colors; the contaminated
cov_rcf/r_main_cov diagnostics never appear on any panel.
"""

from esper.karn.sanctum.schema import SanctumSnapshot
from esper.karn.sanctum.widgets.tamiyo.critic_calibration_panel import (
    CriticCalibrationPanel,
)
from esper.karn.sanctum.widgets.tamiyo.return_variance_panel import (
    ReturnVariancePanel,
)


def _snapshot_with_ppo() -> SanctumSnapshot:
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = True
    return snapshot


class TestReturnVariancePanel:
    def test_off_leg_shows_hint_not_dashes(self) -> None:
        panel = ReturnVariancePanel()
        panel.update_snapshot(_snapshot_with_ppo())
        output = panel.render().plain
        assert "return_variance_telemetry" in output  # tells the operator WHY

    def test_rvt_leg_renders_shares_with_gate_marker(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.rvt_leg_active = True
        snapshot.tamiyo.return_var_cf_share = 0.53
        snapshot.tamiyo.return_var_main_share = 0.46
        snapshot.tamiyo.return_var_residual_share = 0.01
        snapshot.tamiyo.return_var_cf_share_history.extend([0.50, 0.53])

        panel = ReturnVariancePanel()
        panel.update_snapshot(snapshot)
        output = panel.render().plain

        assert "0.53" in output
        assert "▲" in output  # gate tripped marker (0.53 > 0.40)
        assert "0.46" in output
        assert "0.40" in output  # threshold shown, so the number is interpretable

    def test_untripped_gate_shows_threshold_without_marker(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.rvt_leg_active = True
        snapshot.tamiyo.return_var_cf_share = 0.22
        snapshot.tamiyo.return_var_main_share = 0.75
        snapshot.tamiyo.return_var_residual_share = 0.0

        panel = ReturnVariancePanel()
        panel.update_snapshot(snapshot)
        output = panel.render().plain

        assert "0.22" in output
        assert "▲" not in output

    def test_residual_breach_marked(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.rvt_leg_active = True
        snapshot.tamiyo.return_var_cf_share = 0.30
        snapshot.tamiyo.return_var_main_share = 0.55
        snapshot.tamiyo.return_var_residual_share = 0.12  # > 0.05 alarm

        panel = ReturnVariancePanel()
        panel.update_snapshot(snapshot)
        output = panel.render().plain
        assert "✗" in output

    def test_residual_clean_marked_ok(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.rvt_leg_active = True
        snapshot.tamiyo.return_var_cf_share = 0.30
        snapshot.tamiyo.return_var_main_share = 0.68
        snapshot.tamiyo.return_var_residual_share = 0.01

        panel = ReturnVariancePanel()
        panel.update_snapshot(snapshot)
        output = panel.render().plain
        assert "●" in output


class TestCriticCalibrationPerStreamEV:
    def test_hra_leg_renders_per_stream_ev(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.hra_leg_active = True
        snapshot.tamiyo.ev_main = 0.42
        snapshot.tamiyo.ev_cf = 0.11
        snapshot.tamiyo.cf_value_loss = 0.021

        panel = CriticCalibrationPanel()
        panel.update_snapshot(snapshot)
        output = panel.render().plain

        assert "EV main" in output
        assert "0.42" in output
        assert "EV cf" in output
        assert "0.11" in output
        assert "0.021" in output

    def test_off_leg_renders_no_per_stream_lines(self) -> None:
        panel = CriticCalibrationPanel()
        panel.update_snapshot(_snapshot_with_ppo())
        output = panel.render().plain

        assert "EV main" not in output
        assert "EV cf" not in output


class TestActionHeadsAdvantageNormFooter:
    """Per-head advantage-norm observability rides the ACTION HEADS footer."""

    def _panel(self, snapshot: SanctumSnapshot):
        from esper.karn.sanctum.widgets.tamiyo.action_heads_panel import (
            ActionHeadsPanel,
        )

        panel = ActionHeadsPanel()
        panel.update_snapshot(snapshot)
        return panel

    def test_min_sparse_std_shown_even_with_ablation_off(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.min_sparse_head_advantage_std = 0.07

        output = self._panel(snapshot).render().plain
        assert "AdvNorm" in output
        assert "0.07" in output
        assert "per-head:off" in output
        assert "fellback" not in output  # structurally 0 when PHN off: no dead chip

    def test_fellback_count_shown_when_ablation_on(self) -> None:
        snapshot = _snapshot_with_ppo()
        snapshot.tamiyo.advantage_per_head_normalized = True
        snapshot.tamiyo.advantage_norm_fellback_count = 2
        snapshot.tamiyo.min_sparse_head_advantage_std = 0.04

        output = self._panel(snapshot).render().plain
        assert "per-head:on" in output
        assert "fellback:2" in output


def test_value_diagnostics_shows_value_target_scale():
    """b4c1: the EV-denominator (value_target_scale) is rendered on the critic panel."""
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.karn.sanctum.widgets.tamiyo.value_diagnostics_panel import (
        ValueDiagnosticsPanel,
    )

    snap = SanctumSnapshot()
    snap.tamiyo.ppo_data_received = True
    snap.tamiyo.value_target_scale = 3.14
    panel = ValueDiagnosticsPanel()
    panel.update_snapshot(snap)
    text = panel.render().plain
    assert "V-tgt σ" in text
    assert "3.14" in text


def test_critic_calibration_ev_artifact_badge():
    """b4c1: ev_low_return_variance flags EV as a floored-denominator artifact."""
    from esper.karn.sanctum.schema import SanctumSnapshot
    from esper.karn.sanctum.widgets.tamiyo.critic_calibration_panel import (
        CriticCalibrationPanel,
    )

    snap = SanctumSnapshot()
    snap.tamiyo.ppo_data_received = True
    snap.tamiyo.ev_low_return_variance = True
    panel = CriticCalibrationPanel()
    panel.update_snapshot(snap)
    assert "lowRV" in panel.render().plain

    snap.tamiyo.ev_low_return_variance = False
    panel.update_snapshot(snap)
    assert "lowRV" not in panel.render().plain
