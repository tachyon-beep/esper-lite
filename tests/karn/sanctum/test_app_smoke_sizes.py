"""Smoke tests at the documented terminal sizes (120x40 min, 140x50 rec).

Guards the review's Critical finding: fixed-width panels used to clip whole
panels invisible below ~150 columns. With fr+min-width sizing every panel
must mount and render at both documented sizes, on every tab.
"""
import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.schema import SanctumSnapshot
from esper.karn.sanctum.widgets.reward_health import RewardHealthData


def _populated_snapshot() -> SanctumSnapshot:
    snapshot = SanctumSnapshot(connected=True)
    snapshot.tamiyo.ppo_data_received = True
    snapshot.tamiyo.hra_leg_active = True
    snapshot.tamiyo.rvt_leg_active = True
    snapshot.tamiyo.ev_main = 0.42
    snapshot.tamiyo.ev_cf = 0.11
    snapshot.tamiyo.cf_value_loss = 0.021
    snapshot.tamiyo.return_var_cf_share = 0.53
    snapshot.tamiyo.return_var_main_share = 0.46
    snapshot.tamiyo.return_var_residual_share = 0.01
    return snapshot


def _mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.get_all_snapshots.return_value = {"default": _populated_snapshot()}
    backend.compute_reward_health_by_group.return_value = {
        "default": RewardHealthData()
    }
    return backend


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (140, 50)])
async def test_all_tabs_render_at_documented_sizes(size: tuple[int, int]):
    app = SanctumApp(backend=_mock_backend())
    async with app.run_test(size=size) as pilot:
        app._poll_and_refresh()
        await pilot.pause()

        # Every panel of the overview must be mounted and displayed
        for selector in (
            "#env-overview",
            "#scoreboard",
            "#metrics-reward-health",
            "#tamiyo-container",
        ):
            widget = app.query_one(selector)
            assert widget.display, f"{selector} not displayed at {size}"

        # Cycle through the other tabs; rendering must not raise at either size
        await pilot.press("right_square_bracket")
        await pilot.pause()
        critic = app.query_one("#critic-screen")
        assert critic.query_one("#critic-tab-return-variance").display

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert app.query_one("#experiment-panel").display
