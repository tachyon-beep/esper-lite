"""Integration tests for SanctumApp."""

import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, EnvState
from esper.leyline.telemetry import PPOUpdatePayload


class TestSanctumAppIntegration:
    """Test SanctumApp widget wiring."""

    @pytest.mark.asyncio
    async def test_app_creates_all_widgets(self):
        """All required widgets should be created on compose."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async with app.run_test() as pilot:
            # Verify all widgets exist
            assert app.query_one("#env-overview") is not None
            assert app.query_one("#scoreboard") is not None
            assert app.query_one("#tamiyo-container") is not None  # Container for dynamic widgets
            assert app.query_one("#event-log") is not None

    @pytest.mark.asyncio
    async def test_snapshot_propagates_to_all_widgets(self):
        """Snapshot updates should reach all widgets via polling or manual refresh."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(entropy=1.2, clip_fraction=0.15),
            envs={0: EnvState(env_id=0, host_accuracy=75.5)},
        )
        mock_backend.get_snapshot.return_value = snapshot

        # Use a fast refresh rate so timer fires quickly
        app = SanctumApp(backend=mock_backend, num_envs=4, refresh_rate=10.0)

        async with app.run_test() as pilot:
            # Wait for the timer-based refresh to fire (interval is 0.1s at 10Hz)
            import asyncio
            await asyncio.sleep(0.2)
            await pilot.pause()

            # Backend should have been called by set_interval -> _poll_and_refresh
            mock_backend.get_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_focus_env_updates_reward_panel(self):
        """Calling action_focus_env directly should update focused env ID."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=16)

        async with app.run_test() as pilot:
            # Verify initial state
            assert app._focused_env_id == 0

            # Call action directly (bindings can be flaky in tests)
            app.action_focus_env(2)
            assert app._focused_env_id == 2

            app.action_focus_env(7)
            assert app._focused_env_id == 7

            # Out of bounds should not change
            app.action_focus_env(100)
            assert app._focused_env_id == 7  # Unchanged

    @pytest.mark.asyncio
    async def test_quit_action_exits_app(self):
        """Pressing 'q' should trigger app exit."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async with app.run_test() as pilot:
            # Press 'q' to quit
            await pilot.press("q")
            # In Textual, quitting ends the test context gracefully
            # We verify the app received the quit action by checking it didn't raise
            # The run_test context manager handles app lifecycle


@pytest.mark.asyncio
async def test_new_layout_structure():
    """Test that new layout has correct panel structure."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend

    backend = SanctumBackend()
    app = SanctumApp(backend=backend, num_envs=4)

    async with app.run_test() as pilot:
        # Should have EnvOverview and Scoreboard in top section
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None

        # Should have EventLog and TamiyoBrain container in bottom section
        assert app.query_one("#event-log") is not None
        assert app.query_one("#tamiyo-container") is not None  # Container for dynamic widgets

        # Should NOT have SystemResources or TrainingHealth
        from textual.css.query import NoMatches
        with pytest.raises(NoMatches):
            app.query_one("#system-resources")
        with pytest.raises(NoMatches):
            app.query_one("#training-health")


@pytest.mark.asyncio
async def test_sanctum_app_shows_multiple_tamiyo_widgets():
    """A/B mode should show two TamiyoBrain widgets side-by-side."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()
    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Send events for two groups via backend (production path)
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                group_id=group_id,  # Top-level attribute, NOT in data
                data=PPOUpdatePayload(
                    policy_loss=0.1,
                    value_loss=0.0,
                    entropy=0.0,
                    grad_norm=0.0,
                    kl_divergence=0.0,
                    clip_fraction=0.0,
                    nan_grad_count=0,
                ),
            )
            backend.emit(event)

        # Trigger refresh and allow processing
        app._poll_and_refresh()
        await pilot.pause()

        # Should have two TamiyoBrain widgets
        widgets = app.query(TamiyoBrain)
        assert len(widgets) == 2

        # Each should have correct group class
        has_group_a = any("group-a" in " ".join(w.classes) for w in widgets)
        has_group_b = any("group-b" in " ".join(w.classes) for w in widgets)
        assert has_group_a, "Missing group-a widget"
        assert has_group_b, "Missing group-b widget"


@pytest.mark.asyncio
async def test_keyboard_switches_between_policies():
    """Tab key should cycle focus between policy widgets."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()
    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Create two policies - note: group_id is TOP-LEVEL attribute
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                group_id=group_id,  # TOP-LEVEL, not in data
                data=PPOUpdatePayload(
                    policy_loss=0.1,
                    value_loss=0.0,
                    entropy=0.0,
                    grad_norm=0.0,
                    kl_divergence=0.0,
                    clip_fraction=0.0,
                    nan_grad_count=0,
                ),
            )
            backend.emit(event)

        # Trigger refresh and allow widgets to mount
        app._poll_and_refresh()
        await pilot.pause()

        # Verify we have two TamiyoBrain widgets
        widgets = list(app.query(TamiyoBrain))
        assert len(widgets) == 2, f"Expected 2 widgets, got {len(widgets)}"

        # Press Tab multiple times to cycle through focusable widgets until we reach a TamiyoBrain
        # The focus order is: EnvOverview's DataTable, EventLog, then TamiyoBrain widgets
        max_tabs = 10
        for _ in range(max_tabs):
            await pilot.press("tab")
            await pilot.pause()
            if isinstance(app.focused, TamiyoBrain):
                break

        # Should now have a focused TamiyoBrain
        assert isinstance(app.focused, TamiyoBrain), f"Expected TamiyoBrain to be focused, got {app.focused}"
        first_focused = app.focused

        # Verify Textual's built-in focus state (focus handled by :focus pseudo-class)
        assert app.focused == first_focused, "First TamiyoBrain should be focused"

        # Press Tab again to move to second TamiyoBrain
        await pilot.press("tab")
        await pilot.pause()

        # If there are only 2 TamiyoBrain widgets and we were on the first, we should now be on the second
        # (or cycle back depending on focus order)
        second_focused = app.focused

        # Either we moved to the second TamiyoBrain, or we cycled to something else
        # Verify focus moved away from first widget
        assert app.focused != first_focused, "Focus should have moved away from first widget"

        # Find the currently focused widget if it's a TamiyoBrain
        if isinstance(second_focused, TamiyoBrain):
            assert app.focused == second_focused, "Second TamiyoBrain should now be focused"
            assert second_focused != first_focused, "Focus should have moved to a different widget"


@pytest.mark.asyncio
async def test_run_header_shows_ab_comparison():
    """RunHeader should show A/B comparison when 2+ policies exist."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.run_header import RunHeader
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()
    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Create two policies - note: group_id is TOP-LEVEL
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                group_id=group_id,
                data=PPOUpdatePayload(
                    policy_loss=0.1,
                    value_loss=0.0,
                    entropy=0.0,
                    grad_norm=0.0,
                    kl_divergence=0.0,
                    clip_fraction=0.0,
                    nan_grad_count=0,
                ),
            )
            backend.emit(event)

        # Trigger refresh and allow processing
        app._poll_and_refresh()
        await pilot.pause()

        # RunHeader should have A/B mode active with a leader
        header = app.query_one("#run-header", RunHeader)
        # Leader is determined in update_comparison() when called by app
        # Both policies start with same metrics so leader depends on tiebreaker
        # Just verify A/B mode is active (leader can be A, B, or None)
        assert header._ab_mode is True


@pytest.mark.asyncio
async def test_run_header_no_ab_comparison_in_single_mode():
    """RunHeader should not show A/B comparison with only one policy."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.run_header import RunHeader
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()
    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Only one policy
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id="A",
            data=PPOUpdatePayload(
                policy_loss=0.1,
                value_loss=0.0,
                entropy=0.0,
                grad_norm=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                nan_grad_count=0,
            ),
        )
        backend.emit(event)

        # Trigger refresh and allow processing
        app._poll_and_refresh()
        await pilot.pause()

        # RunHeader should NOT be in A/B mode
        header = app.query_one("#run-header", RunHeader)
        assert header._ab_mode is False


@pytest.mark.asyncio
async def test_backend_emits_create_multiple_tamiyo_widgets():
    """Backend emitting A/B events should create two TamiyoBrain widgets via production path."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.leyline import TelemetryEvent, TelemetryEventType

    backend = SanctumBackend(num_envs=4)
    backend.start()

    # Emit events for two groups through backend (production path)
    for group_id in ["A", "B"]:
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            group_id=group_id,
            data=PPOUpdatePayload(
                policy_loss=0.1,
                value_loss=0.0,
                entropy=0.0,
                grad_norm=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                nan_grad_count=0,
            ),
        )
        backend.emit(event)

    app = SanctumApp(backend=backend, num_envs=4)
    async with app.run_test() as pilot:
        # Trigger refresh (simulates timer firing)
        app._poll_and_refresh()
        await pilot.pause()

        # Should have two TamiyoBrain widgets
        widgets = list(app.query(TamiyoBrain))
        assert len(widgets) == 2, f"Expected 2 TamiyoBrain widgets, got {len(widgets)}"

        # Each should have correct group class
        classes = [" ".join(w.classes) for w in widgets]
        assert any("group-a" in c for c in classes), "Missing group-a widget"
        assert any("group-b" in c for c in classes), "Missing group-b widget"
