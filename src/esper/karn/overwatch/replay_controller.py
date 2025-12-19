"""Replay Controller for Overwatch TUI.

State machine for controlling replay playback:
- Play/pause toggle
- Step forward/backward
- Speed adjustment (0.25x to 8x)
- Progress tracking and seeking

Usage:
    controller = ReplayController(Path("training.jsonl"))
    controller.toggle_play()
    controller.step_forward()
    snapshot = controller.current_snapshot
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


# Available speed multipliers
SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class ReplayController:
    """State machine for replay playback control.

    Manages:
    - Loading snapshots from JSONL file
    - Current position (frame index)
    - Play/pause state
    - Playback speed
    - Navigation (step, seek)

    The controller does NOT handle timing - it just provides
    state and navigation. The app's timer calls step_forward()
    based on speed when playing.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize replay controller.

        Args:
            path: Path to JSONL replay file
        """
        self._path = Path(path)
        self._snapshots: list[TuiSnapshot] = []
        self._current_index: int = 0
        self._playing: bool = False
        self._speed_index: int = 2  # 1.0x

        self._load_snapshots()

    def _load_snapshots(self) -> None:
        """Load all snapshots from file into memory."""
        from esper.karn.overwatch.replay import SnapshotReader

        reader = SnapshotReader(self._path)
        self._snapshots = list(reader)

    @property
    def total_frames(self) -> int:
        """Total number of frames in replay."""
        return len(self._snapshots)

    @property
    def current_index(self) -> int:
        """Current frame index (0-based)."""
        return self._current_index

    @property
    def current_snapshot(self) -> TuiSnapshot | None:
        """Current snapshot at playback position."""
        if not self._snapshots:
            return None
        return self._snapshots[self._current_index]

    @property
    def playing(self) -> bool:
        """Whether replay is currently playing."""
        return self._playing

    @property
    def speed(self) -> float:
        """Current playback speed multiplier."""
        return SPEED_LEVELS[self._speed_index]

    @property
    def progress(self) -> float:
        """Progress through replay (0.0 to 1.0)."""
        if self.total_frames <= 1:
            return 0.0
        return self._current_index / (self.total_frames - 1)

    @property
    def status_text(self) -> str:
        """Human-readable status string.

        Returns:
            Status like "[▶ REPLAY 2x] 3/10 30%"
        """
        icon = "▶" if self._playing else "⏸"
        speed_str = f"{self.speed}x" if self.speed != 1.0 else "1x"
        frame_str = f"{self._current_index + 1}/{self.total_frames}"
        pct_str = f"{int(self.progress * 100)}%"

        return f"[{icon} REPLAY {speed_str}] {frame_str} {pct_str}"

    def toggle_play(self) -> None:
        """Toggle play/pause state."""
        self._playing = not self._playing

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False

    def play(self) -> None:
        """Start playback."""
        self._playing = True

    def step_forward(self) -> bool:
        """Advance to next frame.

        Returns:
            True if advanced, False if at end
        """
        if self._current_index < self.total_frames - 1:
            self._current_index += 1
            return True
        return False

    def step_backward(self) -> bool:
        """Go back to previous frame.

        Returns:
            True if moved back, False if at start
        """
        if self._current_index > 0:
            self._current_index -= 1
            return True
        return False

    def seek(self, index: int) -> None:
        """Seek to specific frame index.

        Args:
            index: Target frame index (clamped to valid range)
        """
        self._current_index = max(0, min(index, self.total_frames - 1))

    def increase_speed(self) -> None:
        """Increase playback speed to next level."""
        if self._speed_index < len(SPEED_LEVELS) - 1:
            self._speed_index += 1

    def decrease_speed(self) -> None:
        """Decrease playback speed to previous level."""
        if self._speed_index > 0:
            self._speed_index -= 1

    def tick_interval_ms(self) -> float:
        """Get timer interval for current speed.

        Returns:
            Milliseconds between frames at current speed.
            Base rate is 1000ms (1 frame per second).
        """
        return 1000.0 / self.speed
