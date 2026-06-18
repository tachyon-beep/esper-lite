"""Tests for RewardHealthPanel widget."""

from esper.karn.sanctum.widgets.reward_health import RewardHealthPanel, RewardHealthData


def test_reward_health_data_from_components():
    """RewardHealthData computed from telemetry."""
    data = RewardHealthData(
        pbrs_fraction=0.25,
        anti_gaming_trigger_rate=0.03,
        ev_explained=0.65,
        value_nrmse=0.4,
        hypervolume=42.5,
    )
    assert data.pbrs_fraction == 0.25
    assert data.is_pbrs_healthy  # 0.1-0.4 range
    assert data.is_gaming_healthy  # <0.05
    assert data.is_ev_healthy  # value_nrmse < 1.0


def test_is_ev_healthy_uses_robust_signal():
    """is_ev_healthy keys on the robust value_nrmse, NOT the artifact-prone EV.

    An artefactual-low-EV update (ev_low_return_variance=True, EV deeply negative)
    with a healthy value_nrmse must report is_ev_healthy == True: the EV crater is a
    denominator artifact, the value head is actually fitting (low NRMSE).
    """
    artifact = RewardHealthData(
        ev_explained=-12.0,  # crater EV — would be "unhealthy" under the old >0.5 rule
        value_nrmse=0.3,  # but the robust fit signal is healthy
        ev_low_return_variance=True,
    )
    assert artifact.is_ev_healthy

    # Conversely, a genuinely bad fit (high NRMSE) is unhealthy regardless of EV sign.
    genuine_bad = RewardHealthData(
        ev_explained=0.9,  # EV looks fine
        value_nrmse=1.5,  # but residual RMSE exceeds return std — bad fit
        ev_low_return_variance=False,
    )
    assert not genuine_bad.is_ev_healthy


def test_reward_health_pbrs_boundary_low():
    """PBRS below 10% is unhealthy (shaping too weak)."""
    data = RewardHealthData(pbrs_fraction=0.05)
    assert not data.is_pbrs_healthy


def test_reward_health_pbrs_boundary_high():
    """PBRS above 40% is unhealthy (shaping dominates)."""
    data = RewardHealthData(pbrs_fraction=0.45)
    assert not data.is_pbrs_healthy


def test_reward_health_pbrs_at_boundaries():
    """PBRS exactly at 10% and 40% is healthy."""
    low_boundary = RewardHealthData(pbrs_fraction=0.10)
    high_boundary = RewardHealthData(pbrs_fraction=0.40)
    assert low_boundary.is_pbrs_healthy
    assert high_boundary.is_pbrs_healthy


def test_reward_health_warnings():
    """Warnings triggered for unhealthy metrics."""
    unhealthy = RewardHealthData(
        pbrs_fraction=0.7,  # >0.4 = too much shaping
        anti_gaming_trigger_rate=0.15,  # >0.05 = policy exploiting
        value_nrmse=2.0,  # >=1.0 = residual RMSE exceeds return std (poor value fit)
        hypervolume=10.0,
    )
    assert not unhealthy.is_pbrs_healthy
    assert not unhealthy.is_gaming_healthy
    assert not unhealthy.is_ev_healthy


def test_reward_health_default_values():
    """Default values should be unhealthy (zero state / no data yet)."""
    data = RewardHealthData()
    assert data.pbrs_fraction == 0.0
    assert data.anti_gaming_trigger_rate == 0.0
    assert data.ev_explained == 0.0
    assert data.hypervolume == 0.0
    # Zero PBRS is unhealthy (below 10%)
    assert not data.is_pbrs_healthy
    # Zero gaming rate is healthy (<5%)
    assert data.is_gaming_healthy
    # No value_nrmse data yet (inf sentinel) is unhealthy
    assert data.value_nrmse == float("inf")
    assert not data.is_ev_healthy


def test_reward_health_panel_initialization():
    """RewardHealthPanel can be initialized without data."""
    panel = RewardHealthPanel()
    assert panel._data is not None
    assert panel._data.pbrs_fraction == 0.0


def test_reward_health_panel_with_data():
    """RewardHealthPanel can be initialized with data."""
    data = RewardHealthData(
        pbrs_fraction=0.25,
        anti_gaming_trigger_rate=0.02,
        ev_explained=0.7,
        hypervolume=50.0,
    )
    panel = RewardHealthPanel(data=data)
    assert panel._data.pbrs_fraction == 0.25
    assert panel._data.hypervolume == 50.0


def test_reward_health_panel_update_data():
    """RewardHealthPanel.update_data updates internal state."""
    panel = RewardHealthPanel()
    assert panel._data.pbrs_fraction == 0.0

    new_data = RewardHealthData(
        pbrs_fraction=0.30,
        anti_gaming_trigger_rate=0.01,
        ev_explained=0.8,
        hypervolume=100.0,
    )
    panel.update_data(new_data)
    assert panel._data.pbrs_fraction == 0.30
    assert panel._data.hypervolume == 100.0
