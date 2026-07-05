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
