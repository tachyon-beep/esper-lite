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
