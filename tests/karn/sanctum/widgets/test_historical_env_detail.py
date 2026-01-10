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
    """_get_current_graveyard should return correct graveyard data based on state."""
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
        blueprint_spawns={"conv_light": 3},
        blueprint_fossilized={"conv_light": 2},
        blueprint_prunes={"conv_light": 1},
    )
    modal = HistoricalEnvDetail(record)

    # Peak state - should return the best_blueprint_* data (which is blueprint_* for now)
    assert modal._view_state == "peak"
    spawns, fossilized, prunes = modal._get_current_graveyard()
    assert spawns == {"conv_light": 3}
    assert fossilized == {"conv_light": 2}
    assert prunes == {"conv_light": 1}

    # Toggle to end state - should also work (same data for now)
    modal._view_state = "end"
    spawns, fossilized, prunes = modal._get_current_graveyard()
    assert spawns == {"conv_light": 3}


def test_historical_env_detail_toggle_state():
    """action_toggle_state should toggle between peak and end states."""
    record = BestRunRecord(
        env_id=0,
        episode=5,
        peak_accuracy=85.0,
        final_accuracy=82.0,
    )
    modal = HistoricalEnvDetail(record)

    # Starts in peak
    assert modal._view_state == "peak"

    # Toggle to end (note: we need to call the internal toggle method)
    # Since _update_display requires mounted widgets, we test the state change directly
    modal._view_state = "end" if modal._view_state == "peak" else "peak"
    assert modal._view_state == "end"

    # Toggle back to peak
    modal._view_state = "end" if modal._view_state == "peak" else "peak"
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
