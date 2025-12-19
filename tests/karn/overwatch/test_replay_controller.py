"""Tests for ReplayController state machine."""

from __future__ import annotations

from pathlib import Path


class TestReplayController:
    """Tests for ReplayController."""

    def test_replay_controller_imports(self) -> None:
        """ReplayController can be imported."""
        from esper.karn.overwatch.replay_controller import ReplayController

        assert ReplayController is not None

    def test_replay_controller_load_file(self, tmp_path: Path) -> None:
        """ReplayController loads snapshots from file."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        # Create test file
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.total_frames == 5
        assert controller.current_index == 0

    def test_replay_controller_current_snapshot(self, tmp_path: Path) -> None:
        """ReplayController returns current snapshot."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
                episode=42,
            )
            writer.write(snap)

        controller = ReplayController(path)
        current = controller.current_snapshot
        assert current is not None
        assert current.episode == 42

    def test_replay_controller_step_forward(self, tmp_path: Path) -> None:
        """ReplayController steps forward through snapshots."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.current_snapshot.episode == 0

        controller.step_forward()
        assert controller.current_snapshot.episode == 1

        controller.step_forward()
        assert controller.current_snapshot.episode == 2

    def test_replay_controller_step_backward(self, tmp_path: Path) -> None:
        """ReplayController steps backward through snapshots."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        controller.step_forward()
        controller.step_forward()
        assert controller.current_index == 2

        controller.step_backward()
        assert controller.current_index == 1

        controller.step_backward()
        assert controller.current_index == 0

    def test_replay_controller_bounds_checking(self, tmp_path: Path) -> None:
        """ReplayController respects bounds."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)

        # Can't go below 0
        controller.step_backward()
        assert controller.current_index == 0

        # Can't go above max
        controller.step_forward()
        controller.step_forward()
        controller.step_forward()  # Try to exceed
        assert controller.current_index == 2

    def test_replay_controller_play_pause(self, tmp_path: Path) -> None:
        """ReplayController toggles play/pause state."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)
        assert controller.playing is False

        controller.toggle_play()
        assert controller.playing is True

        controller.toggle_play()
        assert controller.playing is False

    def test_replay_controller_speed(self, tmp_path: Path) -> None:
        """ReplayController adjusts playback speed."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)
        assert controller.speed == 1.0

        controller.increase_speed()
        assert controller.speed == 2.0

        controller.increase_speed()
        assert controller.speed == 4.0

        controller.decrease_speed()
        assert controller.speed == 2.0

        controller.decrease_speed()
        assert controller.speed == 1.0

        controller.decrease_speed()
        assert controller.speed == 0.5

    def test_replay_controller_speed_bounds(self, tmp_path: Path) -> None:
        """ReplayController speed has min/max bounds."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        controller = ReplayController(path)

        # Min speed is 0.25
        for _ in range(10):
            controller.decrease_speed()
        assert controller.speed == 0.25

        # Max speed is 8.0
        for _ in range(10):
            controller.increase_speed()
        assert controller.speed == 8.0

    def test_replay_controller_progress(self, tmp_path: Path) -> None:
        """ReplayController reports progress percentage."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        controller = ReplayController(path)
        assert controller.progress == 0.0

        controller.step_forward()
        controller.step_forward()
        assert controller.progress == 0.5  # 2 of 4 steps (index 2 of 0-4)

        controller.step_forward()
        controller.step_forward()
        assert controller.progress == 1.0

    def test_replay_controller_seek(self, tmp_path: Path) -> None:
        """ReplayController can seek to specific index."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(10):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:{i:02d}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        controller = ReplayController(path)
        controller.seek(5)
        assert controller.current_index == 5
        assert controller.current_snapshot.episode == 5

    def test_replay_controller_status_text(self, tmp_path: Path) -> None:
        """ReplayController generates status text."""
        from esper.karn.overwatch.replay_controller import ReplayController
        from esper.karn.overwatch import SnapshotWriter, TuiSnapshot, ConnectionStatus, TamiyoState

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        controller = ReplayController(path)
        status = controller.status_text

        assert "REPLAY" in status or "Paused" in status
        assert "1x" in status or "1.0" in status
        assert "1/5" in status or "0%" in status
