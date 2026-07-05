"""Tab-based main screen: Overview / Policy / Critic / Experiment.

The Overview tab keeps the mission-control layout (env overview + scoreboard +
slimmed TamiyoBrain digest) and mounts the RewardHealthPanel; the Policy tab
hosts the deep actor panels; the Critic tab is the EV-stab surface; the
Experiment tab is the A/B leg comparison.
"""
import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.schema import EnvState, SanctumSnapshot
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
async def test_app_has_five_tabs():
    from textual.widgets import TabbedContent

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test():
        tabs = app.query_one("#main-tabs", TabbedContent)
        pane_ids = {pane.id for pane in tabs.query("TabPane")}
        assert {
            "tab-overview", "tab-policy", "tab-critic", "tab-experiment", "tab-governor",
        } <= pane_ids
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
        assert tabs.active == "tab-policy"
        await pilot.press("right_square_bracket")
        assert tabs.active == "tab-critic"
        await pilot.press("right_square_bracket")
        assert tabs.active == "tab-experiment"
        await pilot.press("right_square_bracket")
        assert tabs.active == "tab-governor"
        await pilot.press("right_square_bracket")  # wraps
        assert tabs.active == "tab-overview"
        await pilot.press("left_square_bracket")  # wraps back
        assert tabs.active == "tab-governor"


@pytest.mark.asyncio
async def test_policy_tab_hosts_actor_panels():
    from esper.karn.sanctum.widgets.policy_screen import PolicyScreen
    from esper.karn.sanctum.widgets.tamiyo.action_heads_panel import ActionHeadsPanel

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        policy = app.query_one("#policy-screen", PolicyScreen)
        assert policy.query_one(ActionHeadsPanel) is not None
        assert policy.snapshot is not None  # routed through _apply_view


@pytest.mark.asyncio
async def test_overview_tamiyo_is_a_digest():
    """The deep actor/critic panels no longer live inside TamiyoBrain."""
    from esper.karn.sanctum.widgets.tamiyo import TamiyoBrain
    from esper.karn.sanctum.widgets.tamiyo.action_heads_panel import ActionHeadsPanel
    from esper.karn.sanctum.widgets.tamiyo.critic_calibration_panel import (
        CriticCalibrationPanel,
    )
    from esper.karn.sanctum.widgets.tamiyo.health_status_panel import HealthStatusPanel

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        brain = app.query_one(TamiyoBrain)
        assert brain.query_one("#health-panel", HealthStatusPanel) is not None
        assert not brain.query(ActionHeadsPanel)  # moved to Policy tab
        assert not brain.query(CriticCalibrationPanel)  # lives on Critic tab only


@pytest.mark.asyncio
async def test_governor_tab_hosts_panels_and_routes_snapshot():
    from esper.karn.sanctum.widgets.governor_screen import GovernorScreen
    from esper.karn.sanctum.widgets.governor_ledger_panel import GovernorLedgerPanel
    from esper.karn.sanctum.widgets.prune_attribution_panel import PruneAttributionPanel

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        governor = app.query_one("#governor-screen", GovernorScreen)
        assert governor.query_one(GovernorLedgerPanel) is not None
        assert governor.query_one(PruneAttributionPanel) is not None  # prune attribution promoted
        assert governor.snapshot is not None  # routed through _apply_view


@pytest.mark.asyncio
async def test_governor_rollback_badges_governor_tab():
    leg = SanctumSnapshot()
    e = EnvState(env_id=0, status="healthy")
    e.rolled_back = True
    e.rollback_reason = "governor_nan"
    leg.envs[0] = e

    app = SanctumApp(backend=_mock_backend({"default": leg}))
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        assert app._tab_severity_map["tab-governor"] == "critical"
        assert app._tab_severity_map["tab-overview"] == ""  # rollback no longer badges Overview


@pytest.mark.asyncio
async def test_tab_badges_reflect_anomaly_severity():
    """A critic-level critical badges the Critic tab; an env warning badges Overview."""
    from textual.widgets import TabbedContent

    leg = SanctumSnapshot()
    leg.tamiyo.ppo_data_received = True
    leg.tamiyo.explained_variance = -0.2  # critic critical → tab-critic
    leg.tamiyo.value_min, leg.tamiyo.value_max = -8.0, 14.0
    leg.tamiyo.value_std, leg.tamiyo.value_mean = 6.0, 3.0
    leg.envs[0] = EnvState(env_id=0, status="stalled")  # env warning → tab-overview

    app = SanctumApp(backend=_mock_backend({"default": leg}))
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()

        assert app._tab_severity_map["tab-critic"] == "critical"
        assert app._tab_severity_map["tab-overview"] == "warning"
        assert app._tab_severity_map["tab-experiment"] == ""

        tabs = app.query_one("#main-tabs", TabbedContent)
        assert "●" in str(tabs.get_tab("tab-critic").label)
        assert "●" in str(tabs.get_tab("tab-overview").label)
        assert "●" not in str(tabs.get_tab("tab-experiment").label)


@pytest.mark.asyncio
async def test_tab_badges_clear_when_healthy():
    leg = SanctumSnapshot()
    leg.tamiyo.ppo_data_received = True
    leg.tamiyo.explained_variance = 0.5
    leg.tamiyo.value_min, leg.tamiyo.value_max = -8.0, 14.0
    leg.tamiyo.value_std, leg.tamiyo.value_mean = 6.0, 3.0
    leg.envs[0] = EnvState(env_id=0, status="healthy")

    app = SanctumApp(backend=_mock_backend({"default": leg}))
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        assert all(sev == "" for sev in app._tab_severity_map.values())


@pytest.mark.asyncio
async def test_scroll_keys_are_tab_aware():
    """j/k/g/G scroll the active tab's content; on Overview they drive the table."""
    from textual.widgets import TabbedContent

    app = SanctumApp(backend=_mock_backend())
    async with app.run_test() as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)

        tabs.active = "tab-overview"
        await pilot.pause()
        assert app._active_scroll_target() is None  # env-table cursor path

        tabs.active = "tab-critic"
        await pilot.pause()
        assert app._active_scroll_target().id == "critic-screen"

        tabs.active = "tab-policy"
        await pilot.pause()
        assert app._active_scroll_target().id == "policy-action-heads"


@pytest.mark.asyncio
async def test_preview_backend_drives_the_app():
    """The staged-data preview harness satisfies the backend contract."""
    from esper.karn.sanctum.preview import PreviewBackend

    app = SanctumApp(backend=PreviewBackend(), num_envs=6)
    async with app.run_test() as pilot:
        app._poll_and_refresh()
        await pilot.pause()
        # Both staged legs arrive; drift advances between polls
        assert set(app.view.snapshots_by_group) == {"A", "B"}
        first_epoch = app.view.primary.current_epoch
        app._poll_and_refresh()
        await pilot.pause()
        assert app.view.primary.current_epoch > first_epoch
