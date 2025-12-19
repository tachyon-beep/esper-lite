"""Tests for EventFeed widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import FeedEvent


class TestEventBadge:
    """Tests for event badge rendering."""

    def test_event_badge_gate(self) -> None:
        """GATE events get cyan badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("GATE")
        assert "GATE" in badge
        assert "cyan" in badge

    def test_event_badge_stage(self) -> None:
        """STAGE events get blue badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("STAGE")
        assert "STAGE" in badge
        assert "blue" in badge

    def test_event_badge_ppo(self) -> None:
        """PPO events get magenta badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("PPO")
        assert "PPO" in badge
        assert "magenta" in badge

    def test_event_badge_germ(self) -> None:
        """GERM events get green badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("GERM")
        assert "GERM" in badge
        assert "green" in badge

    def test_event_badge_cull(self) -> None:
        """CULL events get red badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("CULL")
        assert "CULL" in badge
        assert "red" in badge

    def test_event_badge_unknown(self) -> None:
        """Unknown events get white badge."""
        from esper.karn.overwatch.widgets.event_feed import event_badge

        badge = event_badge("UNKNOWN")
        assert "UNKNOWN" in badge
        assert "white" in badge


class TestEventFeed:
    """Tests for EventFeed widget."""

    def test_event_feed_imports(self) -> None:
        """EventFeed can be imported."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        assert EventFeed is not None

    def test_event_feed_renders_events(self) -> None:
        """EventFeed renders list of events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Env 0: Gate G1 passed"),
            FeedEvent("12:00:02", "GERM", 1, "Env 1: Seed germinated in r0c1"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "12:00:01" in content
        assert "Gate G1 passed" in content
        assert "12:00:02" in content
        assert "Seed germinated" in content

    def test_event_feed_shows_badges(self) -> None:
        """EventFeed shows event type badges."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "GATE" in content
        assert "PPO" in content

    def test_event_feed_shows_env_id(self) -> None:
        """EventFeed shows env ID when present."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 3, "Gate passed"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        assert "E3" in content or "Env 3" in content or "[3]" in content

    def test_event_feed_empty_state(self) -> None:
        """EventFeed shows empty state when no events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        feed = EventFeed()

        content = feed.render_events()
        assert "No events" in content or "Waiting" in content

    def test_event_feed_compact_mode(self) -> None:
        """EventFeed has compact mode (fewer visible lines)."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        feed = EventFeed()
        assert feed.expanded is False  # Starts compact

        feed.toggle_expanded()
        assert feed.expanded is True

        feed.toggle_expanded()
        assert feed.expanded is False

    def test_event_feed_filters_by_type(self) -> None:
        """EventFeed can filter events by type."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
            FeedEvent("12:00:03", "GATE", 1, "Gate failed"),
        ]
        feed = EventFeed()
        feed.update_events(events)
        feed.set_filter("GATE")

        content = feed.render_events()
        assert "Gate passed" in content
        assert "Gate failed" in content
        assert "Policy updated" not in content

    def test_event_feed_clear_filter(self) -> None:
        """EventFeed can clear filter to show all events."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "Gate passed"),
            FeedEvent("12:00:02", "PPO", None, "Policy updated"),
        ]
        feed = EventFeed()
        feed.update_events(events)
        feed.set_filter("GATE")
        feed.clear_filter()

        content = feed.render_events()
        assert "Gate passed" in content
        assert "Policy updated" in content

    def test_event_feed_newest_first(self) -> None:
        """EventFeed shows newest events at bottom (log style)."""
        from esper.karn.overwatch.widgets.event_feed import EventFeed

        events = [
            FeedEvent("12:00:01", "GATE", 0, "First event"),
            FeedEvent("12:00:02", "GATE", 0, "Second event"),
            FeedEvent("12:00:03", "GATE", 0, "Third event"),
        ]
        feed = EventFeed()
        feed.update_events(events)

        content = feed.render_events()
        # Newest at bottom means Third appears after First
        first_pos = content.find("First event")
        third_pos = content.find("Third event")
        assert first_pos < third_pos
