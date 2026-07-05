"""Tab-based main screen: Overview / Critic / Experiment.

The Overview tab keeps the mission-control layout (env overview + scoreboard +
TamiyoBrain) and finally mounts the RewardHealthPanel; the Critic tab is the
EV-stab surface; the Experiment tab is the A/B leg comparison.
"""
import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.schema import SanctumSnapshot
from esper.karn.sanctum.widgets.reward_health import RewardHealthData, RewardHealthPanel


def _mock_backend(groups: dict[str, SanctumSnapshot] | None = None) -> MagicMock:
    backend = MagicMock()
    snapshots = groups or {"default": SanctumSnapshot()}
    backend.get_all_snapshots.return_value = snapshots
    backend.compute_reward_health_by_group.return_value = {
        group_id: RewardHealthData() for group_id in snapshots
    }
    return backend


@pytest.mark.asyncio
async def test_app_has_three_tabs():
    from textual.widgets import TabbedContent

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test():
        tabs = app.query_one("#main-tabs", TabbedContent)
        pane_ids = {pane.id for pane in tabs.query("TabPane")}
        assert {"tab-overview", "tab-critic", "tab-experiment"} <= pane_ids
        assert tabs.active == "tab-overview"


@pytest.mark.asyncio
async def test_overview_mounts_reward_health_panel():
    """The orphaned RewardHealthPanel is finally on the main screen."""
    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        panel = app.query_one("#metrics-reward-health", RewardHealthPanel)
        assert panel is not None


@pytest.mark.asyncio
async def test_critic_tab_hosts_ev_stab_panels():
    from esper.karn.sanctum.widgets.critic_screen import CriticScreen
    from esper.karn.sanctum.widgets.tamiyo.return_variance_panel import (
        ReturnVariancePanel,
    )

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        critic = app.query_one("#critic-screen", CriticScreen)
        assert critic.query_one(ReturnVariancePanel) is not None


@pytest.mark.asyncio
async def test_critic_panels_receive_snapshot_updates():
    from esper.karn.sanctum.widgets.critic_screen import CriticScreen

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        critic = app.query_one("#critic-screen", CriticScreen)
        assert critic.snapshot is not None  # routed through _apply_view


@pytest.mark.asyncio
async def test_experiment_panel_receives_groups():
    from esper.karn.sanctum.widgets.experiment_panel import ExperimentPanel

    leg_a, leg_b = SanctumSnapshot(), SanctumSnapshot()
    app = SanctumApp(backend=_mock_backend({"A": leg_a, "B": leg_b}))
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        panel = app.query_one("#experiment-panel", ExperimentPanel)
        assert set(panel.group_ids) == {"A", "B"}


@pytest.mark.asyncio
async def test_bracket_keys_cycle_tabs():
    from textual.widgets import TabbedContent

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert tabs.active == "tab-overview"
        await pilot.press("right_square_bracket")
        assert tabs.active == "tab-critic"
        await pilot.press("right_square_bracket")
        assert tabs.active == "tab-experiment"
        await pilot.press("right_square_bracket")  # wraps
        assert tabs.active == "tab-overview"
        await pilot.press("left_square_bracket")  # wraps back
        assert tabs.active == "tab-experiment"
