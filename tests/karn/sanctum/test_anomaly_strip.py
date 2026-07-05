"""Tests for AnomalyStrip widget."""
from esper.karn.sanctum.widgets.anomaly_strip import AnomalyStrip
from esper.karn.sanctum.schema import (
    SanctumSnapshot, EnvState, SeedState, TamiyoState, SystemVitals
)


def test_anomaly_strip_no_anomalies():
    """When everything is OK, show green 'ALL CLEAR'."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    # Check that the widget reports no anomalies
    assert strip.has_anomalies is False


def test_anomaly_strip_stalled_envs():
    """Stalled envs should be counted and displayed."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="stalled")
    snapshot.envs[1] = EnvState(env_id=1, status="stalled")
    snapshot.envs[2] = EnvState(env_id=2, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.stalled_count == 2


def test_anomaly_strip_gradient_issues():
    """Seeds with gradient issues should be counted."""
    snapshot = SanctumSnapshot()
    env = EnvState(env_id=0, status="healthy")
    env.seeds["r0c0"] = SeedState(slot_id="r0c0", has_exploding=True)
    env.seeds["r0c1"] = SeedState(slot_id="r0c1", has_vanishing=True)
    env.seeds["r1c0"] = SeedState(slot_id="r1c0")  # OK
    snapshot.envs[0] = env
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.gradient_issues == 2


def test_anomaly_strip_ppo_health():
    """PPO issues (entropy collapse, high KL) should be flagged."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState(entropy_collapsed=True)

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.ppo_issues is True


def test_anomaly_strip_memory_pressure():
    """Memory pressure should be flagged."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    vitals = SystemVitals(ram_used_gb=14.5, ram_total_gb=16.0)  # 90.6%
    snapshot.vitals = vitals
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.memory_alarm is True


# =============================================================================
# EV-stab Stage-0 rules (residual breach = anomaly; gate trip = info chip)
# =============================================================================


def test_residual_share_breach_is_an_anomaly():
    """|residual share| above the leyline alarm threshold = decomposition broken."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.return_var_residual_share = 0.09  # > 0.05 alarm

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.residual_breach is True
    text = strip.render().plain
    assert "decomp" in text


def test_residual_share_within_epsilon_is_clean():
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.return_var_residual_share = 0.01

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is False
    assert strip.residual_breach is False


def test_residual_share_none_is_clean():
    """Flags-off legs (None) must not alarm."""
    strip = AnomalyStrip()
    strip.update_snapshot(SanctumSnapshot())

    assert strip.has_anomalies is False
    assert strip.residual_breach is False


def test_gate_trip_is_informational_not_anomaly():
    """CF share over the Stage-0 gate shows as info chip WITHOUT tripping the strip."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.rvt_leg_active = True
    snapshot.tamiyo.return_var_cf_share = 0.53

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is False  # a diagnosis, not a failure
    text = strip.render().plain
    assert "S0 gate" in text
    assert "0.53" in text


def test_gate_untripped_shows_no_chip():
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.rvt_leg_active = True
    snapshot.tamiyo.return_var_cf_share = 0.22

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert "S0 gate" not in strip.render().plain


def test_gate_chip_coexists_with_anomalies():
    """Info chip still renders when real anomalies are present."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="stalled")
    snapshot.tamiyo.return_var_cf_share = 0.61

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    text = strip.render().plain
    assert "stalled" in text
    assert "S0 gate" in text


# =============================================================================
# Severity split: critical paints error-red, warnings get a muted tint
# =============================================================================


def test_stalled_is_a_warning_not_a_critical():
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="stalled")

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_warning is True
    assert strip.has_critical is False
    assert "WARNINGS:" in strip.render().plain


def test_degraded_is_a_critical():
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="degraded")

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_critical is True
    assert "ANOMALIES:" in strip.render().plain


# =============================================================================
# Off-tab critical conditions the strip must now cover (glossary "critical")
# =============================================================================


def test_ev_non_positive_is_critical_when_data_present():
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = True
    snapshot.tamiyo.explained_variance = -0.1
    # Give it a healthy value range so ONLY EV trips.
    snapshot.tamiyo.value_min, snapshot.tamiyo.value_max = -8.0, 14.0
    snapshot.tamiyo.value_std, snapshot.tamiyo.value_mean = 6.0, 3.0

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.ev_critical is True
    assert strip.has_critical is True
    assert "EV≤0" in strip.render().plain


def test_ev_artifact_is_suppressed():
    """A floored EV denominator is an artifact, not a critic collapse."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = True
    snapshot.tamiyo.explained_variance = -0.5
    snapshot.tamiyo.ev_low_return_variance = True
    snapshot.tamiyo.value_min, snapshot.tamiyo.value_max = -8.0, 14.0
    snapshot.tamiyo.value_std, snapshot.tamiyo.value_mean = 6.0, 3.0

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.ev_critical is False


def test_value_collapse_is_critical_when_data_present():
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = True
    # All-zero value stats = collapsed value function.
    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.value_critical is True
    assert strip.has_critical is True
    assert "value collapse" in strip.render().plain


def test_ratio_explosion_is_critical_when_data_present():
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = True
    snapshot.tamiyo.joint_ratio_max = 6.0
    # Healthy value range so only the ratio trips.
    snapshot.tamiyo.value_min, snapshot.tamiyo.value_max = -8.0, 14.0
    snapshot.tamiyo.value_std, snapshot.tamiyo.value_mean = 6.0, 3.0

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.ratio_explosion is True
    assert strip.has_critical is True
    assert "RatioJnt" in strip.render().plain


def test_metric_criticals_gated_on_ppo_data():
    """A fresh snapshot (no PPO data yet) must NOT paint the strip red.

    EV defaults to 0.0 (<=0) and value stats default to all-zero (a collapse),
    so ungated these would false-alarm on every run's first seconds — the exact
    alarm-fatigue failure the strip is meant to kill.
    """
    snapshot = SanctumSnapshot()
    snapshot.tamiyo.ppo_data_received = False
    snapshot.tamiyo.explained_variance = -0.5  # would trip if ungated
    snapshot.tamiyo.joint_ratio_max = 9.0  # would trip if ungated

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.ev_critical is False
    assert strip.value_critical is False
    assert strip.ratio_explosion is False
    assert strip.has_anomalies is False


# =============================================================================
# Governor: the loudest safety event, independent of the policy
# =============================================================================


def test_governor_rollback_is_critical_and_leads():
    snapshot = SanctumSnapshot()
    e0 = EnvState(env_id=0, status="healthy")
    e0.rolled_back = True
    e0.rollback_reason = "nan"
    e1 = EnvState(env_id=1, status="healthy")
    e1.rolled_back = True
    e1.rollback_reason = "divergence"
    snapshot.envs[0], snapshot.envs[1] = e0, e1

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.governor_rollback_count == 2
    assert strip.has_critical is True
    text = strip.render().plain
    assert "GOV ROLLBACK ×2" in text
    assert "nan" in text and "divergence" in text
    # It leads: nothing renders before the rollback segment.
    assert text.index("GOV ROLLBACK") < text.index("ANOMALIES:") + len("ANOMALIES: ") + 1


def test_governor_rollback_not_gated_on_ppo_data():
    """A rollback surfaces even before any PPO update — the governor is
    independent of the policy."""
    snapshot = SanctumSnapshot()
    e0 = EnvState(env_id=0, status="healthy")
    e0.rolled_back = True
    e0.rollback_reason = "lobotomy"
    snapshot.envs[0] = e0
    assert snapshot.tamiyo.ppo_data_received is False

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.governor_rollback_count == 1
    assert strip.has_critical is True


def test_rollback_unattributed_is_a_warning():
    snapshot = SanctumSnapshot()
    snapshot.governor.rollback_unattributed_count = 3

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.rollback_unattributed is True
    assert strip.has_warning is True
    assert strip.has_critical is False


# =============================================================================
# Cross-leg aggregation: a critical unique to a non-primary leg must fire
# =============================================================================


def _leg_with_ev_collapse() -> SanctumSnapshot:
    s = SanctumSnapshot()
    s.tamiyo.ppo_data_received = True
    s.tamiyo.explained_variance = -0.2
    s.tamiyo.value_min, s.tamiyo.value_max = -8.0, 14.0
    s.tamiyo.value_std, s.tamiyo.value_mean = 6.0, 3.0
    return s


def _healthy_leg() -> SanctumSnapshot:
    s = SanctumSnapshot()
    s.tamiyo.ppo_data_received = True
    s.tamiyo.explained_variance = 0.5
    s.tamiyo.value_min, s.tamiyo.value_max = -8.0, 14.0
    s.tamiyo.value_std, s.tamiyo.value_mean = 6.0, 3.0
    s.envs[0] = EnvState(env_id=0, status="healthy")
    return s


def test_critical_in_non_primary_leg_fires_the_strip():
    """Primary leg A is clean; leg B has EV<=0 — the strip must still fire."""
    strip = AnomalyStrip()
    strip.update_snapshots(
        {"A": _healthy_leg(), "B": _leg_with_ev_collapse()},
        primary_group_id="A",
    )

    assert strip.has_critical is True
    text = strip.render().plain
    assert "EV≤0 (B)" in text  # attributed to the offending leg


def test_single_leg_via_update_snapshots_has_no_leg_tag():
    strip = AnomalyStrip()
    strip.update_snapshots({"default": _leg_with_ev_collapse()}, primary_group_id="default")

    text = strip.render().plain
    assert "EV≤0" in text
    assert "(default)" not in text  # one leg → no attribution noise


def test_governor_rollback_attributed_to_leg():
    healthy = _healthy_leg()
    rolled = SanctumSnapshot()
    e = EnvState(env_id=0, status="healthy")
    e.rolled_back = True
    e.rollback_reason = "nan"
    rolled.envs[0] = e

    strip = AnomalyStrip()
    strip.update_snapshots({"A": healthy, "B": rolled}, primary_group_id="A")

    assert strip.governor_rollback_count == 1
    text = strip.render().plain
    assert "GOV ROLLBACK ×1 [nan] (B)" in text


def test_counts_sum_across_legs():
    a, b = SanctumSnapshot(), SanctumSnapshot()
    a.envs[0] = EnvState(env_id=0, status="stalled")
    b.envs[0] = EnvState(env_id=0, status="stalled")
    b.envs[1] = EnvState(env_id=1, status="stalled")

    strip = AnomalyStrip()
    strip.update_snapshots({"A": a, "B": b}, primary_group_id="A")

    assert strip.stalled_count == 3  # 1 + 2
    assert strip.has_warning is True
    assert strip.has_critical is False
