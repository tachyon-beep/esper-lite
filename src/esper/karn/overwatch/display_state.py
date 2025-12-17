"""Display State Management.

Handles UI state that isn't directly derived from TuiSnapshot,
such as sort order stability (hysteresis) and selection state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HysteresisConfig:
    """Configuration for hysteresis sorting.

    Hysteresis prevents visual jitter when envs have similar anomaly scores.
    An env must exceed threshold positions to actually move in the display.

    Example with threshold_up=3, threshold_down=5:
    - Env at position 5 needs natural position <= 2 to move up (5 - 3 = 2)
    - Env at position 2 needs natural position >= 7 to move down (2 + 5 = 7)
    """

    threshold_up: int = 3  # Positions needed to move up
    threshold_down: int = 5  # Positions needed to move down


@dataclass
class HysteresisSorter:
    """Sorts env_ids with hysteresis to prevent visual jitter.

    Maintains the previous sort order and only allows envs to move
    when the position delta exceeds the configured thresholds.

    Usage:
        sorter = HysteresisSorter()

        # Each frame:
        scores = {env_id: anomaly_score for env in envs}
        display_order = sorter.sort(scores)
    """

    config: HysteresisConfig = field(default_factory=HysteresisConfig)
    _previous_order: list[int] = field(default_factory=list)

    def sort(self, scores: dict[int, float]) -> list[int]:
        """Sort env_ids by anomaly score with hysteresis.

        Args:
            scores: Mapping of env_id to anomaly_score (higher = more anomalous)

        Returns:
            List of env_ids in display order (highest anomaly first)
        """
        if not scores:
            self._previous_order = []
            return []

        env_ids = set(scores.keys())

        # Natural order: sorted by score descending
        natural_order = sorted(scores.keys(), key=lambda e: scores[e], reverse=True)
        natural_positions = {env_id: idx for idx, env_id in enumerate(natural_order)}

        # If no previous order, use natural order
        if not self._previous_order:
            self._previous_order = natural_order.copy()
            return natural_order.copy()

        # Remove envs that no longer exist
        current_order = [e for e in self._previous_order if e in env_ids]

        # Add new envs at their natural position
        new_envs = env_ids - set(current_order)
        for new_env in sorted(new_envs, key=lambda e: natural_positions[e]):
            # Insert at natural position (clamped to valid range)
            insert_pos = min(natural_positions[new_env], len(current_order))
            current_order.insert(insert_pos, new_env)

        # Current positions
        current_positions = {env_id: idx for idx, env_id in enumerate(current_order)}

        # Check each env for movement
        result = current_order.copy()
        moved = set()

        for env_id in env_ids:
            if env_id in moved:
                continue

            current_pos = current_positions[env_id]
            natural_pos = natural_positions[env_id]
            delta = current_pos - natural_pos  # Positive = should move up

            should_move = False

            if delta > 0 and delta >= self.config.threshold_up:
                # Needs to move up (lower index)
                should_move = True
            elif delta < 0 and abs(delta) >= self.config.threshold_down:
                # Needs to move down (higher index)
                should_move = True

            if should_move:
                # Remove from current position
                result.remove(env_id)
                # Insert at natural position
                insert_pos = min(natural_pos, len(result))
                result.insert(insert_pos, env_id)
                moved.add(env_id)

        self._previous_order = result.copy()
        return result

    def reset(self) -> None:
        """Clear sort history (next sort will use natural order)."""
        self._previous_order = []


@dataclass
class DisplayState:
    """Complete display state for the Overwatch TUI.

    Tracks UI state that persists across snapshot updates:
    - Sort order with hysteresis
    - Selected env/slot
    - Expanded envs
    - Panel visibility
    """

    sorter: HysteresisSorter = field(default_factory=HysteresisSorter)
    selected_env_id: int | None = None
    expanded_env_ids: set[int] = field(default_factory=set)

    def get_sorted_env_ids(self, scores: dict[int, float]) -> list[int]:
        """Get env_ids in display order."""
        return self.sorter.sort(scores)

    def select_env(self, env_id: int) -> None:
        """Select an environment."""
        self.selected_env_id = env_id

    def toggle_expand(self, env_id: int) -> bool:
        """Toggle env expansion. Returns new expansion state."""
        if env_id in self.expanded_env_ids:
            self.expanded_env_ids.discard(env_id)
            return False
        else:
            self.expanded_env_ids.add(env_id)
            return True

    def is_expanded(self, env_id: int) -> bool:
        """Check if env is expanded."""
        return env_id in self.expanded_env_ids
