"""Tests for ReplayStatusBar widget."""

from __future__ import annotations


class TestReplayStatusBar:
    """Tests for ReplayStatusBar widget."""

    def test_replay_status_imports(self) -> None:
        """ReplayStatusBar can be imported."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        assert ReplayStatusBar is not None

    def test_replay_status_shows_mode(self) -> None:
        """ReplayStatusBar shows replay mode indicator."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "REPLAY" in content
        assert "▶" in content

    def test_replay_status_shows_paused(self) -> None:
        """ReplayStatusBar shows paused indicator."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=False, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "⏸" in content or "PAUSED" in content

    def test_replay_status_shows_speed(self) -> None:
        """ReplayStatusBar shows playback speed."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=2.0, current=5, total=10)

        content = bar.render_bar()
        assert "2" in content or "2x" in content

    def test_replay_status_shows_progress_bar(self) -> None:
        """ReplayStatusBar shows visual progress bar."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "█" in content or "▓" in content or "░" in content

    def test_replay_status_shows_frame_count(self) -> None:
        """ReplayStatusBar shows current/total frame count."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(playing=True, speed=1.0, current=5, total=10)

        content = bar.render_bar()
        assert "5" in content or "6" in content  # 0-indexed or 1-indexed
        assert "10" in content

    def test_replay_status_shows_timestamp(self) -> None:
        """ReplayStatusBar shows snapshot timestamp."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.update_status(
            playing=True,
            speed=1.0,
            current=5,
            total=10,
            timestamp="12:00:05",
        )

        content = bar.render_bar()
        assert "12:00:05" in content

    def test_replay_status_hidden_when_not_replay(self) -> None:
        """ReplayStatusBar can be hidden for live mode."""
        from esper.karn.overwatch.widgets.replay_status import ReplayStatusBar

        bar = ReplayStatusBar()
        bar.set_visible(False)

        assert bar.is_visible is False
