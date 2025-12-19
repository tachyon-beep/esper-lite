"""Tests for hysteresis sorting logic."""

from __future__ import annotations

import pytest


class TestHysteresisConfig:
    """Tests for HysteresisConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config uses 3 up, 5 down."""
        from esper.karn.overwatch.display_state import HysteresisConfig

        config = HysteresisConfig()
        assert config.threshold_up == 3
        assert config.threshold_down == 5

    def test_custom_config(self) -> None:
        """Config can be customized."""
        from esper.karn.overwatch.display_state import HysteresisConfig

        config = HysteresisConfig(threshold_up=2, threshold_down=4)
        assert config.threshold_up == 2
        assert config.threshold_down == 4


class TestHysteresisSorter:
    """Tests for HysteresisSorter class."""

    def test_initial_sort_by_anomaly_score(self) -> None:
        """First sort orders by anomaly score (highest first)."""
        from esper.karn.overwatch.display_state import HysteresisSorter

        sorter = HysteresisSorter()

        # env_id -> anomaly_score
        scores = {0: 0.1, 1: 0.8, 2: 0.5, 3: 0.3}
        result = sorter.sort(scores)

        # Highest anomaly first
        assert result == [1, 2, 3, 0]

    def test_stable_sort_within_threshold(self) -> None:
        """Small score changes don't change order (hysteresis)."""
        from esper.karn.overwatch.display_state import HysteresisSorter

        sorter = HysteresisSorter()

        # Initial order
        scores1 = {0: 0.1, 1: 0.8, 2: 0.5, 3: 0.3}
        result1 = sorter.sort(scores1)
        assert result1 == [1, 2, 3, 0]

        # Small change: env 3 increases slightly but not enough to pass env 2
        scores2 = {0: 0.1, 1: 0.8, 2: 0.5, 3: 0.48}
        result2 = sorter.sort(scores2)
        # Order should stay stable
        assert result2 == [1, 2, 3, 0]

    def test_reorder_when_exceeds_threshold_up(self) -> None:
        """Env moves up when it exceeds threshold_up positions in natural order."""
        from esper.karn.overwatch.display_state import HysteresisSorter, HysteresisConfig

        config = HysteresisConfig(threshold_up=2, threshold_down=3)
        sorter = HysteresisSorter(config)

        # Initial: 4 envs with distinct scores
        scores1 = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1, 2, 3]  # env 0 first (highest)

        # env 3 jumps to highest score - exceeds threshold_up (was pos 3, now pos 0)
        # Delta = 3 positions, threshold_up = 2, so it should move
        scores2 = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.95}
        result2 = sorter.sort(scores2)
        assert result2[0] == 3  # env 3 should be first now

    def test_reorder_when_exceeds_threshold_down(self) -> None:
        """Env moves down when it exceeds threshold_down positions in natural order."""
        from esper.karn.overwatch.display_state import HysteresisSorter, HysteresisConfig

        config = HysteresisConfig(threshold_up=2, threshold_down=3)
        sorter = HysteresisSorter(config)

        # Initial
        scores1 = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1, 2, 3]

        # env 0 drops to lowest - exceeds threshold_down (was pos 0, now pos 3)
        # Delta = 3 positions, threshold_down = 3, so it should move
        scores2 = {0: 0.1, 1: 0.7, 2: 0.5, 3: 0.3}
        result2 = sorter.sort(scores2)
        assert result2[-1] == 0  # env 0 should be last now

    def test_handles_new_env(self) -> None:
        """New envs are inserted at their natural position."""
        from esper.karn.overwatch.display_state import HysteresisSorter

        sorter = HysteresisSorter()

        # Initial
        scores1 = {0: 0.8, 1: 0.5}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1]

        # New env appears
        scores2 = {0: 0.8, 1: 0.5, 2: 0.9}
        result2 = sorter.sort(scores2)
        assert result2[0] == 2  # New env at top

    def test_handles_removed_env(self) -> None:
        """Removed envs are dropped from order."""
        from esper.karn.overwatch.display_state import HysteresisSorter

        sorter = HysteresisSorter()

        # Initial
        scores1 = {0: 0.8, 1: 0.5, 2: 0.3}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1, 2]

        # Env 1 disappears
        scores2 = {0: 0.8, 2: 0.3}
        result2 = sorter.sort(scores2)
        assert result2 == [0, 2]
        assert 1 not in result2

    def test_reset_clears_history(self) -> None:
        """Reset clears previous order history."""
        from esper.karn.overwatch.display_state import HysteresisSorter

        sorter = HysteresisSorter()

        # Build up history
        scores1 = {0: 0.8, 1: 0.5}
        sorter.sort(scores1)

        # Reset
        sorter.reset()

        # Next sort should be fresh (no hysteresis effect)
        scores2 = {0: 0.3, 1: 0.9}
        result2 = sorter.sort(scores2)
        assert result2 == [1, 0]  # Pure score order

    def test_simultaneous_movements(self) -> None:
        """Multiple envs moving at once should be handled correctly."""
        from esper.karn.overwatch.display_state import HysteresisSorter, HysteresisConfig

        config = HysteresisConfig(threshold_up=2, threshold_down=2)
        sorter = HysteresisSorter(config)

        # Initial: [0, 1, 2, 3] with scores [0.9, 0.7, 0.5, 0.3]
        scores1 = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1, 2, 3]

        # Flip scores: multiple envs should move
        scores2 = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9}
        result2 = sorter.sort(scores2)
        # Natural order would be [3, 2, 1, 0]
        # With threshold=2: env 3 moves up (delta=3), env 0 moves down (delta=3)
        # env 1 and env 2 stay put (delta=1 each, below threshold)
        # Result: env 3 to front, env 0 to back, env 1 and 2 stay in middle
        assert result2 == [3, 1, 2, 0]

    def test_multi_move_positions_are_correct(self) -> None:
        """Verify multi-move insertion positions are calculated correctly.

        This tests the specific concern that natural_pos insertion after
        removals might cause incorrect positioning. The algorithm processes
        moves sorted by natural_pos (ascending), ensuring earlier insertions
        are not disturbed by later ones.
        """
        from esper.karn.overwatch.display_state import HysteresisSorter, HysteresisConfig

        # Use threshold=0 to force ALL items to move to natural positions
        config = HysteresisConfig(threshold_up=0, threshold_down=0)
        sorter = HysteresisSorter(config)

        # Initial: [0, 1, 2, 3, 4] with descending scores
        scores1 = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1}
        result1 = sorter.sort(scores1)
        assert result1 == [0, 1, 2, 3, 4]

        # Reverse all scores - all items must move to opposite positions
        scores2 = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}
        result2 = sorter.sort(scores2)

        # With threshold=0, all items move to natural positions
        # Natural order: [4, 3, 2, 1, 0]
        assert result2 == [4, 3, 2, 1, 0], (
            f"All items should move to natural positions with threshold=0. "
            f"Got {result2}, expected [4, 3, 2, 1, 0]"
        )
