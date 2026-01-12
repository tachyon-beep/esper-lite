"""Tests for HistoricalEnvDetail state toggle."""

from esper.karn.sanctum.schema import BestRunRecord, SeedState
from esper.karn.sanctum.widgets.historical_env_detail import HistoricalEnvDetail


def test_historical_env_detail_has_state_toggle_binding():
    """HistoricalEnvDetail should have 's' key binding for state toggle."""
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
    )
    modal = HistoricalEnvDetail(record)

    # Check for 's' binding
    binding_keys = [b.key for b in modal.BINDINGS]
    assert "s" in binding_keys


def test_historical_env_detail_starts_in_peak_state():
    """HistoricalEnvDetail should start showing peak state."""
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
    )
    modal = HistoricalEnvDetail(record)

    assert modal._view_state == "peak"


def test_historical_env_detail_get_current_seeds():
    """_get_current_seeds should return correct seeds based on state."""
    peak_seed = SeedState(slot_id="r0c0", stage="FOSSILIZED", blueprint_id="conv_heavy")
    end_seed = SeedState(slot_id="r0c0", stage="PRUNED", blueprint_id="conv_heavy")

    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
        seeds={"r0c0": peak_seed},
        end_seeds={"r0c0": end_seed},
    )
    modal = HistoricalEnvDetail(record)

    # Peak state
    assert modal._view_state == "peak"
    seeds = modal._get_current_seeds()
    assert seeds["r0c0"].stage == "FOSSILIZED"

    # Toggle to end state
    modal._view_state = "end"
    seeds = modal._get_current_seeds()
    assert seeds["r0c0"].stage == "PRUNED"


def test_historical_env_detail_get_current_graveyard():
    """_get_current_graveyard returns graveyard data (same for peak and end).

    Note: BestRunRecord.blueprint_* contains peak graveyard data.
    End-state graveyard was not added to BestRunRecord, so both views
    return the same data.
    """
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
        # Peak graveyard data (stored in blueprint_* fields)
        blueprint_spawns={"conv_light": 3},
        blueprint_fossilized={"conv_light": 2},
        blueprint_prunes={"conv_light": 1},
    )
    modal = HistoricalEnvDetail(record)

    # Peak state - returns blueprint_* data
    assert modal._view_state == "peak"
    spawns, fossilized, prunes = modal._get_current_graveyard()
    assert spawns == {"conv_light": 3}
    assert fossilized == {"conv_light": 2}
    assert prunes == {"conv_light": 1}

    # End state - also returns blueprint_* (no separate end graveyard data)
    modal._view_state = "end"
    spawns, fossilized, prunes = modal._get_current_graveyard()
    assert spawns == {"conv_light": 3}
    assert fossilized == {"conv_light": 2}
    assert prunes == {"conv_light": 1}


def test_historical_env_detail_toggle_state():
    """action_toggle_state should toggle between peak and end states."""
    from unittest.mock import patch

    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
    )
    modal = HistoricalEnvDetail(record)

    # Starts in peak
    assert modal._view_state == "peak"

    # Mock _update_display since widget isn't mounted
    with patch.object(modal, "_update_display"):
        # Toggle to end - calling the actual action method
        modal.action_toggle_state()
        assert modal._view_state == "end"

        # Toggle back to peak
        modal.action_toggle_state()
        assert modal._view_state == "peak"


def test_historical_env_detail_header_shows_state():
    """Header should show PEAK STATE or END STATE based on view state."""
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
    )
    modal = HistoricalEnvDetail(record)

    # Default is peak state
    header = modal._render_header()
    assert "PEAK STATE" in header.plain

    # Toggle to end state
    modal._view_state = "end"
    header = modal._render_header()
    assert "END STATE" in header.plain


def test_historical_env_detail_has_lifecycle_panel():
    """HistoricalEnvDetail should show lifecycle panel."""
    from esper.karn.sanctum.schema import SeedLifecycleEvent

    events = [
        SeedLifecycleEvent(
            epoch=5, action="GERMINATE(conv_heavy)", from_stage="DORMANT",
            to_stage="GERMINATED", blueprint_id="conv_heavy", slot_id="r0c0",
            alpha=None, accuracy_delta=None,
        ),
    ]
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
        best_lifecycle_events=events,
        end_lifecycle_events=events,
    )
    modal = HistoricalEnvDetail(record)

    # Should have lifecycle panel in compose
    # This is a structural test - full integration would need async test
    assert hasattr(modal, "_get_current_lifecycle_events")

    peak_events = modal._get_current_lifecycle_events()
    assert len(peak_events) == 1
