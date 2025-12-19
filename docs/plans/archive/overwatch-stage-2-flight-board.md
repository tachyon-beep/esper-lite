# Overwatch Stage 2: Flight Board (Seeds/Hosts Telemetry)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render the main Flight Board with environment rows, slot chips, and keyboard navigation. This is the primary UI surface for monitoring training environments.

**Architecture:** Three-tier widget hierarchy: `FlightBoard` (scrollable container) → `EnvRow` (single environment) → `SlotChip` (single slot). Hysteresis logic prevents sort jitter at anomaly score boundaries. Navigation uses vim-style j/k keys with Enter to expand.

**Tech Stack:** Python 3.11, Textual widgets (Container, Static), reactive properties, CSS styling.

**Prerequisites:**
- Stage 0 complete (schema.py, replay.py)
- Stage 1 complete (app.py, styles.tcss, HelpOverlay)
- Branch: `feat/overwatch-textual-ui`

---

## Task 1: Implement Hysteresis Sort Logic

**Files:**
- Create: `src/esper/karn/overwatch/display_state.py`
- Create: `tests/karn/overwatch/test_hysteresis.py`

**Step 1: Write failing tests for hysteresis sorter**

```python
# tests/karn/overwatch/test_hysteresis.py
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
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_hysteresis.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement hysteresis sorter**

```python
# src/esper/karn/overwatch/display_state.py
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
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_hysteresis.py -v`

Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/display_state.py tests/karn/overwatch/test_hysteresis.py
git commit -m "feat(overwatch): add hysteresis sort logic for flight board"
```

---

## Task 2: Create SlotChip Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/slot_chip.py`
- Modify: `tests/karn/overwatch/test_widgets.py`

**Step 1: Write failing tests for SlotChip**

```python
# tests/karn/overwatch/test_widgets.py (append to file)


class TestSlotChip:
    """Tests for SlotChip widget."""

    def test_slot_chip_imports(self) -> None:
        """SlotChip can be imported."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip

        assert SlotChip is not None

    def test_slot_chip_renders_slot_id(self) -> None:
        """SlotChip displays slot ID in [r0c1] format."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.5,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "[r0c1]" in rendered

    def test_slot_chip_renders_stage(self) -> None:
        """SlotChip displays stage name."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "BLENDING" in rendered

    def test_slot_chip_renders_alpha_bar(self) -> None:
        """SlotChip displays alpha progress bar."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.5,  # 50%
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        # Should have some filled and some empty
        assert "█" in rendered or "▓" in rendered
        assert "░" in rendered or "▒" in rendered

    def test_slot_chip_renders_gate_status(self) -> None:
        """SlotChip displays gate status when present."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
            gate_last="G2",
            gate_passed=True,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "G2" in rendered
        assert "✓" in rendered or "✔" in rendered

    def test_slot_chip_gate_failed_indicator(self) -> None:
        """SlotChip shows failure indicator for failed gate."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c0",
            stage="PROBATIONARY",
            blueprint_id="bad_seed",
            alpha=0.3,
            gate_last="G1",
            gate_passed=False,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "G1" in rendered
        assert "✗" in rendered or "✘" in rendered or "×" in rendered
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestSlotChip -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement SlotChip**

```python
# src/esper/karn/overwatch/widgets/slot_chip.py
"""Slot Chip Widget.

Renders a single slot's state as a compact chip showing:
- Slot ID: [r0c1]
- Stage name: TRAINING, BLENDING, etc.
- Alpha progress bar: ████░░ 0.7α
- Gate status: G2✓ or G1✗
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from esper.karn.overwatch.schema import SlotChipState


# Stage colors (CSS class names)
STAGE_COLORS = {
    "DORMANT": "stage-dormant",
    "GERMINATED": "stage-germinated",
    "TRAINING": "stage-training",
    "BLENDING": "stage-blending",
    "PROBATIONARY": "stage-probationary",
    "FOSSILIZED": "stage-fossilized",
    "CULLED": "stage-culled",
    "EMBARGOED": "stage-embargoed",
    "RESETTING": "stage-resetting",
}

# Compact stage names for display
STAGE_SHORT = {
    "DORMANT": "DORM",
    "GERMINATED": "GERM",
    "TRAINING": "TRAIN",
    "BLENDING": "BLEND",
    "PROBATIONARY": "PROB",
    "FOSSILIZED": "FOSSIL",
    "CULLED": "CULL",
    "EMBARGOED": "EMBAR",
    "RESETTING": "RESET",
}


class SlotChip(Static):
    """Widget displaying a single slot's state.

    Compact format:
        [r0c1] BLEND ████░░ 0.7α G2✓

    Expanded format (when env is expanded):
        [r0c1] BLENDING
        Blueprint: conv_light
        Alpha: ████████░░ 0.78
        Gate: G2 ✓ (epoch 15/20)
    """

    DEFAULT_CSS = """
    SlotChip {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    SlotChip.stage-dormant {
        color: #666666;
    }

    SlotChip.stage-germinated {
        color: #98c379;
    }

    SlotChip.stage-training {
        color: #61afef;
    }

    SlotChip.stage-blending {
        color: #c678dd;
    }

    SlotChip.stage-probationary {
        color: #e5c07b;
    }

    SlotChip.stage-fossilized {
        color: #56b6c2;
    }

    SlotChip.stage-culled {
        color: #e06c75;
    }

    SlotChip.stage-embargoed {
        color: #be5046;
    }

    SlotChip.stage-resetting {
        color: #d19a66;
    }
    """

    def __init__(
        self,
        state: SlotChipState,
        expanded: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the slot chip.

        Args:
            state: SlotChipState data
            expanded: Show expanded view with more details
            **kwargs: Additional args for Static
        """
        super().__init__(**kwargs)
        self._state = state
        self._expanded = expanded

        # Apply stage color class
        stage_class = STAGE_COLORS.get(state.stage, "")
        if stage_class:
            self.add_class(stage_class)

    def render_chip(self) -> str:
        """Render the chip content."""
        if self._expanded:
            return self._render_expanded()
        return self._render_compact()

    def _render_compact(self) -> str:
        """Render compact single-line format."""
        s = self._state

        # Slot ID
        slot_id = f"[{s.slot_id}]"

        # Stage (shortened)
        stage = STAGE_SHORT.get(s.stage, s.stage[:5])

        # Alpha bar (6 chars wide)
        alpha_bar = self._render_alpha_bar(s.alpha, width=6)
        alpha_val = f"{s.alpha:.1f}α" if s.alpha < 1.0 else "1.0α"

        # Gate status
        gate = ""
        if s.gate_last:
            gate_icon = "✓" if s.gate_passed else "✗"
            gate = f" {s.gate_last}{gate_icon}"

        return f"{slot_id} {stage} {alpha_bar} {alpha_val}{gate}"

    def _render_expanded(self) -> str:
        """Render expanded multi-line format."""
        s = self._state
        lines = [
            f"[{s.slot_id}] {s.stage}",
            f"  Blueprint: {s.blueprint_id}",
            f"  Alpha: {self._render_alpha_bar(s.alpha, width=10)} {s.alpha:.2f}",
        ]

        if s.gate_last:
            gate_icon = "✓" if s.gate_passed else "✗"
            lines.append(f"  Gate: {s.gate_last} {gate_icon} (epoch {s.epochs_in_stage}/{s.epochs_total})")

        return "\n".join(lines)

    def _render_alpha_bar(self, alpha: float, width: int = 6) -> str:
        """Render alpha as a progress bar.

        Args:
            alpha: Value 0.0-1.0
            width: Total bar width in characters

        Returns:
            Progress bar string like "████░░"
        """
        filled = int(alpha * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    def compose(self) -> ComposeResult:
        """Compose is not used - we render directly."""
        yield from []

    def render(self) -> str:
        """Render the widget content."""
        return self.render_chip()

    def update_state(self, state: SlotChipState) -> None:
        """Update with new state."""
        self._state = state
        self.refresh()
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestSlotChip -v`

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/slot_chip.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): add SlotChip widget"
```

---

## Task 3: Create EnvRow Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/env_row.py`
- Modify: `tests/karn/overwatch/test_widgets.py`

**Step 1: Write failing tests for EnvRow**

```python
# tests/karn/overwatch/test_widgets.py (append to file)


class TestEnvRow:
    """Tests for EnvRow widget."""

    def test_env_row_imports(self) -> None:
        """EnvRow can be imported."""
        from esper.karn.overwatch.widgets.env_row import EnvRow

        assert EnvRow is not None

    def test_env_row_renders_env_id(self) -> None:
        """EnvRow displays environment ID."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "Env 3" in rendered or "env:3" in rendered.lower()

    def test_env_row_renders_device_id(self) -> None:
        """EnvRow displays device/GPU ID."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=2,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "gpu:2" in rendered.lower() or "GPU 2" in rendered

    def test_env_row_renders_status_ok(self) -> None:
        """EnvRow displays OK status with appropriate styling."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "OK" in rendered

    def test_env_row_renders_status_warn(self) -> None:
        """EnvRow displays WARN status."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=1,
            device_id=0,
            status="WARN",
            anomaly_score=0.65,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "WARN" in rendered

    def test_env_row_renders_status_crit(self) -> None:
        """EnvRow displays CRIT status."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=2,
            device_id=1,
            status="CRIT",
            anomaly_score=0.85,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "CRIT" in rendered

    def test_env_row_renders_throughput(self) -> None:
        """EnvRow displays throughput."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "98" in rendered or "fps" in rendered.lower()

    def test_env_row_renders_slots_inline(self) -> None:
        """EnvRow renders slot chips inline."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5),
            },
        )
        row = EnvRow(env)

        content = row.render_slots_inline()
        assert "[r0c1]" in content
        assert "TRAIN" in content

    def test_env_row_focus_indicator(self) -> None:
        """EnvRow shows focus indicator when selected."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(env_id=0, device_id=0, status="OK")
        row = EnvRow(env, selected=True)

        # Should have some visual indicator
        header = row.render_header()
        # Either [!] prefix or special character
        assert "[" in header or "▶" in header or "●" in header
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestEnvRow -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement EnvRow**

```python
# src/esper/karn/overwatch/widgets/env_row.py
"""Environment Row Widget.

Renders a single training environment in the Flight Board:
- Status indicator and env ID
- Device (GPU) assignment
- Throughput metrics
- Slot chips (inline or expanded)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.schema import EnvSummary, SlotChipState
from esper.karn.overwatch.widgets.slot_chip import SlotChip, STAGE_SHORT


# Status styling
STATUS_ICONS = {
    "OK": "[ ]",
    "INFO": "[i]",
    "WARN": "[!]",
    "CRIT": "[‼]",
}

STATUS_COLORS = {
    "OK": "green",
    "INFO": "blue",
    "WARN": "yellow",
    "CRIT": "red",
}


class EnvRow(Container):
    """Widget displaying a single environment row.

    Compact format (single line):
        [ ] Env 0  gpu:0  OK     98 fps  [r0c1] TRAIN ████░░ 0.5α

    Expanded format (multiple lines):
        [▶] Env 0  gpu:0  OK     98 fps
            [r0c1] TRAINING
            Blueprint: conv_light
            Alpha: ████████░░ 0.78
            Gate: G2 ✓
    """

    DEFAULT_CSS = """
    EnvRow {
        width: 100%;
        height: auto;
        padding: 0;
    }

    EnvRow.selected {
        background: $primary-darken-1;
    }

    EnvRow.status-ok .env-status {
        color: $success;
    }

    EnvRow.status-warn .env-status {
        color: $warning;
    }

    EnvRow.status-crit .env-status {
        color: $error;
    }

    EnvRow .env-header {
        width: 100%;
        height: 1;
    }

    EnvRow .env-slots-expanded {
        padding-left: 4;
    }
    """

    def __init__(
        self,
        env: EnvSummary,
        selected: bool = False,
        expanded: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the environment row.

        Args:
            env: EnvSummary data
            selected: Is this row currently selected
            expanded: Show expanded slot details
            **kwargs: Additional args for Container
        """
        super().__init__(**kwargs)
        self._env = env
        self._selected = selected
        self._expanded = expanded

        # Apply status class
        status_lower = env.status.lower()
        self.add_class(f"status-{status_lower}")

        if selected:
            self.add_class("selected")

    def render_header(self) -> str:
        """Render the header line."""
        e = self._env

        # Selection/status indicator
        if self._selected:
            indicator = "[▶]"
        else:
            indicator = STATUS_ICONS.get(e.status, "[ ]")

        # Format: [▶] Env 0  gpu:0  OK     98 fps
        status_styled = f"[{STATUS_COLORS.get(e.status, 'white')}]{e.status}[/]"

        throughput = f"{e.throughput_fps:.0f} fps" if e.throughput_fps > 0 else ""

        header = f"{indicator} Env {e.env_id}  gpu:{e.device_id}  {e.status:<4}  {throughput}"

        # Add inline slots if not expanded
        if not self._expanded and e.slots:
            slots_inline = self.render_slots_inline()
            header = f"{header}  {slots_inline}"

        return header

    def render_slots_inline(self) -> str:
        """Render slots inline (compact, single line)."""
        if not self._env.slots:
            return ""

        parts = []
        # Sort slots by slot_id for consistent display
        for slot_id in sorted(self._env.slots.keys()):
            slot = self._env.slots[slot_id]
            chip = SlotChip(slot)
            parts.append(chip.render_chip())

        return "  ".join(parts)

    def render_slots_expanded(self) -> list[str]:
        """Render slots expanded (multi-line)."""
        if not self._env.slots:
            return ["    (no slots)"]

        lines = []
        for slot_id in sorted(self._env.slots.keys()):
            slot = self._env.slots[slot_id]
            chip = SlotChip(slot, expanded=True)
            # Indent expanded slot content
            for line in chip.render_chip().split("\n"):
                lines.append(f"    {line}")

        return lines

    def compose(self) -> ComposeResult:
        """Compose the row layout."""
        yield Static(self.render_header(), classes="env-header env-status")

        if self._expanded:
            for line in self.render_slots_expanded():
                yield Static(line, classes="env-slots-expanded")

    def update_env(self, env: EnvSummary) -> None:
        """Update with new environment data."""
        self._env = env
        self.refresh()

    def set_selected(self, selected: bool) -> None:
        """Update selection state."""
        self._selected = selected
        if selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")
        self.refresh()

    def set_expanded(self, expanded: bool) -> None:
        """Update expansion state."""
        self._expanded = expanded
        # Need to recompose to add/remove slot lines
        self.refresh(recompose=True)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestEnvRow -v`

Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/env_row.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): add EnvRow widget"
```

---

## Task 4: Create FlightBoard Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/flight_board.py`
- Create: `tests/karn/overwatch/test_flight_board.py`

**Step 1: Write failing tests for FlightBoard**

```python
# tests/karn/overwatch/test_flight_board.py
"""Tests for FlightBoard widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    EnvSummary,
    SlotChipState,
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def sample_snapshot() -> TuiSnapshot:
    """Create a sample snapshot with multiple envs."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(),
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                throughput_fps=98.5,
                anomaly_score=0.1,
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="WARN",
                throughput_fps=45.0,
                anomaly_score=0.65,
                anomaly_reasons=["Low throughput"],
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="CRIT",
                throughput_fps=10.0,
                anomaly_score=0.85,
                anomaly_reasons=["Throughput critical", "Memory pressure"],
            ),
            EnvSummary(
                env_id=3,
                device_id=1,
                status="OK",
                throughput_fps=100.0,
                anomaly_score=0.05,
            ),
        ],
    )


class TestFlightBoard:
    """Tests for FlightBoard widget."""

    def test_flight_board_imports(self) -> None:
        """FlightBoard can be imported."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        assert FlightBoard is not None

    def test_flight_board_sorts_by_anomaly(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard sorts envs by anomaly score (highest first)."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        order = board.get_display_order()
        # Highest anomaly first: env 2 (0.85), env 1 (0.65), env 0 (0.1), env 3 (0.05)
        assert order[0] == 2
        assert order[1] == 1
        assert order[-1] == 3

    def test_flight_board_initial_selection(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard selects first env initially."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # First in display order (highest anomaly)
        assert board.selected_env_id == 2

    def test_flight_board_navigate_down(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigates down with j/down."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Start at env 2 (first in order)
        assert board.selected_env_id == 2

        board.navigate_down()
        assert board.selected_env_id == 1  # Second in order

        board.navigate_down()
        assert board.selected_env_id == 0  # Third

    def test_flight_board_navigate_up(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigates up with k/up."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Go down first
        board.navigate_down()
        board.navigate_down()
        assert board.selected_env_id == 0

        # Navigate back up
        board.navigate_up()
        assert board.selected_env_id == 1

    def test_flight_board_navigate_wraps(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigation doesn't wrap by default."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # At top, navigate up should stay at top
        board.navigate_up()
        assert board.selected_env_id == 2  # Still at first

    def test_flight_board_expand_collapse(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard expands and collapses envs."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Initially not expanded
        assert not board.is_expanded(2)

        # Expand
        board.toggle_expand()
        assert board.is_expanded(2)

        # Collapse
        board.toggle_expand()
        assert not board.is_expanded(2)

    def test_flight_board_empty_state(self) -> None:
        """FlightBoard handles empty snapshot."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()

        # Should handle no data gracefully
        order = board.get_display_order()
        assert order == []
        assert board.selected_env_id is None
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_flight_board.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement FlightBoard**

```python
# src/esper/karn/overwatch/widgets/flight_board.py
"""Flight Board Widget.

The main display surface showing all training environments.
Supports navigation, selection, and expansion of env rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static

from esper.karn.overwatch.display_state import DisplayState, HysteresisSorter
from esper.karn.overwatch.widgets.env_row import EnvRow

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary, TuiSnapshot


class FlightBoard(VerticalScroll):
    """Scrollable container displaying all training environments.

    Features:
    - Sorts envs by anomaly score with hysteresis
    - j/k or arrow navigation between envs
    - Enter to expand env details, Esc to collapse
    - Visual focus indicator on selected row

    Messages:
    - EnvSelected: Fired when selection changes
    - EnvExpanded: Fired when env is expanded/collapsed
    """

    DEFAULT_CSS = """
    FlightBoard {
        width: 100%;
        height: 100%;
        background: $surface;
    }

    FlightBoard > EnvRow {
        width: 100%;
    }

    FlightBoard:focus {
        border: tall $primary;
    }

    FlightBoard .empty-state {
        width: 100%;
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("j", "navigate_down", "Down", show=False),
        Binding("k", "navigate_up", "Up", show=False),
        Binding("down", "navigate_down", "Down", show=False),
        Binding("up", "navigate_up", "Up", show=False),
        Binding("enter", "toggle_expand", "Expand", show=False),
    ]

    class EnvSelected(Message):
        """Fired when env selection changes."""

        def __init__(self, env_id: int | None) -> None:
            super().__init__()
            self.env_id = env_id

    class EnvExpanded(Message):
        """Fired when env expansion state changes."""

        def __init__(self, env_id: int, expanded: bool) -> None:
            super().__init__()
            self.env_id = env_id
            self.expanded = expanded

    def __init__(self, **kwargs) -> None:
        """Initialize the flight board."""
        super().__init__(**kwargs)
        self._display_state = DisplayState()
        self._snapshot: TuiSnapshot | None = None
        self._envs_by_id: dict[int, EnvSummary] = {}
        self._display_order: list[int] = []

    @property
    def selected_env_id(self) -> int | None:
        """Currently selected environment ID."""
        return self._display_state.selected_env_id

    def get_display_order(self) -> list[int]:
        """Get current display order of env IDs."""
        return self._display_order.copy()

    def is_expanded(self, env_id: int) -> bool:
        """Check if env is expanded."""
        return self._display_state.is_expanded(env_id)

    def update_snapshot(self, snapshot: TuiSnapshot) -> None:
        """Update with new snapshot data.

        Args:
            snapshot: New TuiSnapshot to display
        """
        self._snapshot = snapshot

        # Build lookup
        self._envs_by_id = {e.env_id: e for e in snapshot.flight_board}

        # Get sorted order with hysteresis
        scores = {e.env_id: e.anomaly_score for e in snapshot.flight_board}
        self._display_order = self._display_state.get_sorted_env_ids(scores)

        # Auto-select first if nothing selected
        if self._display_state.selected_env_id is None and self._display_order:
            self._display_state.select_env(self._display_order[0])

        # Validate selection still exists
        if self._display_state.selected_env_id not in self._envs_by_id:
            if self._display_order:
                self._display_state.select_env(self._display_order[0])
            else:
                self._display_state.selected_env_id = None

        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Compose the flight board content."""
        if not self._display_order:
            yield Static("No environments", classes="empty-state")
            return

        for env_id in self._display_order:
            env = self._envs_by_id.get(env_id)
            if env is None:
                continue

            is_selected = env_id == self._display_state.selected_env_id
            is_expanded = self._display_state.is_expanded(env_id)

            yield EnvRow(
                env,
                selected=is_selected,
                expanded=is_expanded,
                id=f"env-{env_id}",
            )

    def navigate_down(self) -> None:
        """Move selection down one row."""
        if not self._display_order:
            return

        current = self._display_state.selected_env_id
        if current is None:
            # Select first
            self._select(self._display_order[0])
            return

        try:
            idx = self._display_order.index(current)
            if idx < len(self._display_order) - 1:
                self._select(self._display_order[idx + 1])
        except ValueError:
            # Current not in list, select first
            self._select(self._display_order[0])

    def navigate_up(self) -> None:
        """Move selection up one row."""
        if not self._display_order:
            return

        current = self._display_state.selected_env_id
        if current is None:
            # Select last
            self._select(self._display_order[-1])
            return

        try:
            idx = self._display_order.index(current)
            if idx > 0:
                self._select(self._display_order[idx - 1])
        except ValueError:
            # Current not in list, select first
            self._select(self._display_order[0])

    def toggle_expand(self) -> None:
        """Toggle expansion of selected env."""
        env_id = self._display_state.selected_env_id
        if env_id is None:
            return

        expanded = self._display_state.toggle_expand(env_id)
        self.post_message(self.EnvExpanded(env_id, expanded))
        self.refresh(recompose=True)

    def _select(self, env_id: int) -> None:
        """Select an env and update UI."""
        old_id = self._display_state.selected_env_id
        self._display_state.select_env(env_id)

        # Update old row
        if old_id is not None:
            old_row = self.query_one(f"#env-{old_id}", EnvRow)
            if old_row:
                old_row.set_selected(False)

        # Update new row
        new_row = self.query_one(f"#env-{env_id}", EnvRow)
        if new_row:
            new_row.set_selected(True)
            self.scroll_to_widget(new_row)

        self.post_message(self.EnvSelected(env_id))

    # Action handlers for bindings
    def action_navigate_down(self) -> None:
        """Action: navigate down."""
        self.navigate_down()

    def action_navigate_up(self) -> None:
        """Action: navigate up."""
        self.navigate_up()

    def action_toggle_expand(self) -> None:
        """Action: toggle expand."""
        self.toggle_expand()
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_flight_board.py -v`

Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/flight_board.py tests/karn/overwatch/test_flight_board.py
git commit -m "feat(overwatch): add FlightBoard widget with navigation"
```

---

## Task 5: Update Widgets Package Exports

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/__init__.py`

**Step 1: Write failing test for imports**

```python
# tests/karn/overwatch/test_widgets.py (append to file)


class TestWidgetExports:
    """Tests for widget package exports."""

    def test_all_widgets_importable(self) -> None:
        """All widgets are importable from package."""
        from esper.karn.overwatch.widgets import (
            HelpOverlay,
            SlotChip,
            EnvRow,
            FlightBoard,
        )

        assert HelpOverlay is not None
        assert SlotChip is not None
        assert EnvRow is not None
        assert FlightBoard is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestWidgetExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update widgets __init__.py**

```python
# src/esper/karn/overwatch/widgets/__init__.py
"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.slot_chip import SlotChip
from esper.karn.overwatch.widgets.env_row import EnvRow
from esper.karn.overwatch.widgets.flight_board import FlightBoard

__all__ = [
    "HelpOverlay",
    "SlotChip",
    "EnvRow",
    "FlightBoard",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestWidgetExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/__init__.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): export all widgets from package"
```

---

## Task 6: Update Styles with Stage Colors

**Files:**
- Modify: `src/esper/karn/overwatch/styles.tcss`

**Step 1: Add stage and status color definitions**

```css
/* src/esper/karn/overwatch/styles.tcss */

/* Overwatch TUI Styles
 *
 * Layout regions:
 * - Header (2 rows): Run identity, connection, health indicators
 * - TamiyoStrip (2 rows): PPO vitals, action summary
 * - MainArea (flexible): FlightBoard + DetailPanel side-by-side
 * - EventFeed (4 rows collapsed, 8 expanded)
 */

/* === Root Layout === */

Screen {
    layout: grid;
    grid-size: 1;
    grid-rows: 2 2 1fr 4;
}

/* === Region Placeholders === */

#header {
    background: $surface;
    border-bottom: solid $primary-darken-2;
    padding: 0 1;
}

#tamiyo-strip {
    background: $surface;
    color: $text;
    border-bottom: solid $secondary;
    padding: 0 1;
}

#main-area {
    layout: grid;
    grid-size: 2 1;
    grid-columns: 2fr 1fr;
}

#flight-board {
    background: $surface;
    border-right: solid $primary-darken-3;
    padding: 0 1;
}

#detail-panel {
    background: $surface;
    padding: 0 1;
}

#event-feed {
    background: $surface;
    border-top: solid $primary-darken-2;
    padding: 0 1;
}

/* === Status Colors === */

.status-ok {
    color: $success;
}

.status-info {
    color: $primary;
}

.status-warn {
    color: $warning;
}

.status-crit {
    color: $error;
}

/* === Stage Colors (Seed Lifecycle) === */

.stage-dormant {
    color: #666666;
}

.stage-germinated {
    color: #98c379;  /* Light green - new growth */
}

.stage-training {
    color: #61afef;  /* Blue - active learning */
}

.stage-blending {
    color: #c678dd;  /* Purple/magenta - integration */
}

.stage-probationary {
    color: #e5c07b;  /* Yellow/amber - under observation */
}

.stage-fossilized {
    color: #56b6c2;  /* Cyan - permanent/stable */
}

.stage-culled {
    color: #e06c75;  /* Red - removed */
}

.stage-embargoed {
    color: #be5046;  /* Dark red - blocked */
}

.stage-resetting {
    color: #d19a66;  /* Orange - transitioning */
}

/* === Tamiyo Theme (Magenta) === */

.tamiyo {
    color: #c678dd;
}

.tamiyo-muted {
    color: #8b5a9e;
}

/* === Focus Indicators === */

.focused {
    background: $primary-darken-1;
}

.selected {
    background: $primary-darken-2;
}

/* === Hidden State === */

.hidden {
    display: none;
}

/* === FlightBoard Specific === */

FlightBoard {
    scrollbar-gutter: stable;
}

FlightBoard:focus-within {
    border: tall $accent;
}

EnvRow {
    height: auto;
    min-height: 1;
}

EnvRow.selected {
    background: $primary-darken-2;
}

EnvRow:hover {
    background: $surface-lighten-1;
}

/* === Progress Bars === */

.alpha-bar-filled {
    color: $success;
}

.alpha-bar-empty {
    color: $surface-lighten-2;
}

/* === Anomaly Score Indicator === */

.anomaly-low {
    color: $success;
}

.anomaly-medium {
    color: $warning;
}

.anomaly-high {
    color: $error;
}
```

**Step 2: Verify syntax is valid**

Run: `PYTHONPATH=src uv run python -c "from textual.css.parse import parse; parse(open('src/esper/karn/overwatch/styles.tcss').read()); print('CSS OK')"`

Expected: `CSS OK` (or import the app to check)

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/styles.tcss
git commit -m "feat(overwatch): add stage and status colors to stylesheet"
```

---

## Task 7: Wire FlightBoard into OverwatchApp

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`

**Step 1: Update app.py to use real FlightBoard**

Replace the placeholder `#flight-board` Static with the real FlightBoard widget:

```python
# src/esper/karn/overwatch/app.py
"""Overwatch Textual Application.

Main application class for the Overwatch TUI monitoring interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, Static

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.flight_board import FlightBoard

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class OverwatchApp(App):
    """Overwatch TUI for monitoring Esper training runs.

    Provides real-time visibility into:
    - Training environments (Flight Board)
    - Seed lifecycle and health
    - Tamiyo agent decisions
    - System resources

    Usage:
        app = OverwatchApp()
        app.run()

        # Or with replay file:
        app = OverwatchApp(replay_path="training.jsonl")
        app.run()
    """

    TITLE = "Esper Overwatch"
    SUB_TITLE = "Training Monitor"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "toggle_help", "Help", show=True),
        Binding("escape", "dismiss", "Dismiss", show=False),
        # Navigation bindings are handled by FlightBoard
    ]

    def __init__(
        self,
        replay_path: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Overwatch app.

        Args:
            replay_path: Optional path to JSONL replay file
            **kwargs: Additional args passed to App
        """
        super().__init__(**kwargs)
        self._replay_path = Path(replay_path) if replay_path else None
        self._snapshot: TuiSnapshot | None = None
        self._help_visible = False

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        # Header region (run identity, connection status)
        yield Static(
            self._render_header_content(),
            id="header",
        )

        # Tamiyo Strip (PPO vitals, action summary)
        yield Static(
            self._render_tamiyo_content(),
            id="tamiyo-strip",
        )

        # Main area with flight board and detail panel
        with Container(id="main-area"):
            # Real FlightBoard widget
            yield FlightBoard(id="flight-board")

            yield Static(
                self._render_detail_panel_content(),
                id="detail-panel",
            )

        # Event feed
        yield Static(
            self._render_event_feed_content(),
            id="event-feed",
        )

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

    def _render_header_content(self) -> str:
        """Render header placeholder content."""
        if self._snapshot:
            ts = self._snapshot.captured_at
            run_id = self._snapshot.run_id or "unknown"
            task = self._snapshot.task_name or "unknown"
            ep = self._snapshot.episode
            return f"[HEADER] Run: {run_id} | Task: {task} | Ep: {ep} | Snapshot: {ts}"
        return "[HEADER] Waiting for data..."

    def _render_tamiyo_content(self) -> str:
        """Render Tamiyo Strip placeholder content."""
        if self._snapshot and self._snapshot.tamiyo:
            kl = self._snapshot.tamiyo.kl_divergence
            ent = self._snapshot.tamiyo.entropy
            ev = self._snapshot.tamiyo.explained_variance
            return f"[TAMIYO] KL: {kl:.3f} | Entropy: {ent:.2f} | EV: {ev:.2f}"
        return "[TAMIYO STRIP] Waiting for policy data..."

    def _render_detail_panel_content(self) -> str:
        """Render Detail Panel placeholder content."""
        return "[DETAIL PANEL] Select an environment (j/k to navigate)"

    def _render_event_feed_content(self) -> str:
        """Render Event Feed placeholder content."""
        if self._snapshot and self._snapshot.event_feed:
            n = len(self._snapshot.event_feed)
            return f"[EVENT FEED] {n} events"
        return "[EVENT FEED] No events"

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Load initial snapshot if replay file provided
        if self._replay_path:
            self._load_first_snapshot()

        # Set focus to flight board for navigation
        self.query_one(FlightBoard).focus()

    def _load_first_snapshot(self) -> None:
        """Load the first snapshot from replay file."""
        from esper.karn.overwatch.replay import SnapshotReader

        if not self._replay_path or not self._replay_path.exists():
            self.notify(f"Replay file not found: {self._replay_path}", severity="error")
            return

        reader = SnapshotReader(self._replay_path)
        for snapshot in reader:
            self._snapshot = snapshot
            break  # Take first snapshot only

        if self._snapshot:
            self.notify(f"Loaded snapshot from {self._snapshot.captured_at}")
            # Update flight board with snapshot
            self._update_flight_board()
            # Refresh placeholders
            self._refresh_placeholders()
        else:
            self.notify("No snapshots found in replay file", severity="warning")

    def _update_flight_board(self) -> None:
        """Update the flight board with current snapshot."""
        if self._snapshot:
            flight_board = self.query_one(FlightBoard)
            flight_board.update_snapshot(self._snapshot)

    def _refresh_placeholders(self) -> None:
        """Refresh placeholder widgets with current snapshot."""
        self.query_one("#header", Static).update(self._render_header_content())
        self.query_one("#tamiyo-strip", Static).update(self._render_tamiyo_content())
        self.query_one("#detail-panel", Static).update(self._render_detail_panel_content())
        self.query_one("#event-feed", Static).update(self._render_event_feed_content())

    def action_toggle_help(self) -> None:
        """Toggle the help overlay visibility."""
        help_overlay = self.query_one("#help-overlay")
        help_overlay.toggle_class("hidden")
        self._help_visible = not self._help_visible

    def action_dismiss(self) -> None:
        """Dismiss overlays or collapse expanded elements."""
        if self._help_visible:
            self.action_toggle_help()

    def on_flight_board_env_selected(self, message: FlightBoard.EnvSelected) -> None:
        """Handle env selection in flight board."""
        self._update_detail_panel(message.env_id)

    def on_flight_board_env_expanded(self, message: FlightBoard.EnvExpanded) -> None:
        """Handle env expansion in flight board."""
        pass  # Could update detail panel

    def _update_detail_panel(self, env_id: int | None) -> None:
        """Update detail panel with selected env info."""
        if env_id is None or self._snapshot is None:
            self.query_one("#detail-panel", Static).update(
                "[DETAIL PANEL] Select an environment"
            )
            return

        # Find the env
        env = None
        for e in self._snapshot.flight_board:
            if e.env_id == env_id:
                env = e
                break

        if env is None:
            return

        # Build detail content
        lines = [f"[DETAIL] Env {env.env_id}"]
        lines.append(f"Status: {env.status}")
        lines.append(f"Anomaly: {env.anomaly_score:.2f}")

        if env.anomaly_reasons:
            lines.append("Reasons:")
            for reason in env.anomaly_reasons:
                lines.append(f"  • {reason}")

        self.query_one("#detail-panel", Static).update("\n".join(lines))
```

**Step 2: Run app test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/app.py
git commit -m "feat(overwatch): wire FlightBoard into OverwatchApp"
```

---

## Task 8: Integration Tests for Flight Board Navigation

**Files:**
- Modify: `tests/karn/overwatch/test_integration.py`

**Step 1: Add integration tests for navigation**

```python
# tests/karn/overwatch/test_integration.py (append to file)


class TestFlightBoardNavigation:
    """Integration tests for flight board navigation."""

    @pytest.fixture
    def multi_env_replay(self, tmp_path: Path) -> Path:
        """Create replay with multiple envs for navigation testing."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "multi_env.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
                flight_board=[
                    EnvSummary(
                        env_id=0, device_id=0, status="OK",
                        anomaly_score=0.1,
                        slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
                    ),
                    EnvSummary(
                        env_id=1, device_id=0, status="WARN",
                        anomaly_score=0.65,
                    ),
                    EnvSummary(
                        env_id=2, device_id=1, status="CRIT",
                        anomaly_score=0.85,
                        anomaly_reasons=["High gradient ratio"],
                    ),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_navigation_j_k(self, multi_env_replay: Path) -> None:
        """j/k keys navigate between envs."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Should start at highest anomaly (env 2)
            assert board.selected_env_id == 2

            # Navigate down
            await pilot.press("j")
            assert board.selected_env_id == 1

            # Navigate down again
            await pilot.press("j")
            assert board.selected_env_id == 0

            # Navigate back up
            await pilot.press("k")
            assert board.selected_env_id == 1

    @pytest.mark.asyncio
    async def test_navigation_arrows(self, multi_env_replay: Path) -> None:
        """Arrow keys navigate between envs."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Navigate with arrows
            await pilot.press("down")
            assert board.selected_env_id == 1

            await pilot.press("up")
            assert board.selected_env_id == 2

    @pytest.mark.asyncio
    async def test_expand_collapse(self, multi_env_replay: Path) -> None:
        """Enter expands env, Esc collapses."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Initially not expanded
            assert not board.is_expanded(2)

            # Expand with Enter
            await pilot.press("enter")
            assert board.is_expanded(2)

            # Collapse with Enter again
            await pilot.press("enter")
            assert not board.is_expanded(2)

    @pytest.mark.asyncio
    async def test_detail_panel_updates_on_selection(self, multi_env_replay: Path) -> None:
        """Detail panel updates when env is selected."""
        from esper.karn.overwatch import OverwatchApp
        from textual.widgets import Static

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            # Initial selection should show env 2 details
            detail = app.query_one("#detail-panel", Static)
            assert "Env 2" in str(detail.renderable)

            # Navigate to env 1
            await pilot.press("j")
            # Panel should update
            detail = app.query_one("#detail-panel", Static)
            assert "Env 1" in str(detail.renderable)
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/karn/overwatch/test_integration.py
git commit -m "test(overwatch): add flight board navigation integration tests"
```

---

## Task 9: Manual Verification

**Step 1: Launch with healthy fixture**

Run: `PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay tests/karn/overwatch/fixtures/healthy_run.jsonl`

Expected:
- TUI launches
- Flight Board shows envs sorted by anomaly (highest first)
- `j`/`k` navigation works
- Selection highlight visible
- `Enter` expands env to show slot details
- `Esc` collapses
- Slot chips show `[r0c1]` format, stage, alpha bar

**Step 2: Launch with anomaly fixture**

Run: `PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay tests/karn/overwatch/fixtures/anomaly_detected.jsonl`

Expected:
- WARN env shows yellow status
- CRIT env shows red status
- CRIT env appears first (highest anomaly)
- Detail panel shows anomaly reasons when env selected

**Step 3: Test navigation edge cases**

- Navigate to bottom, press `j` (should stay at bottom)
- Navigate to top, press `k` (should stay at top)
- Expand multiple envs, navigate between them

---

## Task 10: Run Full Test Suite and Final Commit

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (~50+ tests)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/`

Expected: No errors

**Step 3: Run type checking**

Run: `uv run mypy src/esper/karn/overwatch/ --ignore-missing-imports`

Expected: No errors (or only minor issues from Textual)

**Step 4: Final summary commit**

```bash
git add -A
git status
```

If any uncommitted changes:
```bash
git commit -m "chore(overwatch): Stage 2 complete - Flight Board with navigation"
```

---

## Verification Checklist

- [ ] Flight Board renders all envs from snapshot
- [ ] Envs sorted by anomaly score (highest first)
- [ ] Hysteresis prevents sort jitter (test with similar scores)
- [ ] Slot chips show `[r0c1]` position, stage name, alpha bar
- [ ] Gate status shows `G2✓` or `G1✗`
- [ ] `j`/`k` and arrow navigation works
- [ ] `Enter` expands env to show slot details
- [ ] `Esc` collapses expanded env
- [ ] Selection highlight visible (background color change)
- [ ] Detail panel updates on selection
- [ ] All tests pass (~50+)
- [ ] Linting passes
- [ ] Can be merged to main independently

---

## Files Created/Modified

```
src/esper/karn/overwatch/
├── __init__.py          # (Stage 1)
├── schema.py            # (Stage 0)
├── replay.py            # (Stage 0)
├── app.py               # Modified: wire FlightBoard
├── styles.tcss          # Modified: stage colors
├── display_state.py     # NEW: hysteresis sort logic
└── widgets/
    ├── __init__.py      # Modified: export all widgets
    ├── help.py          # (Stage 1)
    ├── slot_chip.py     # NEW: slot chip rendering
    ├── env_row.py       # NEW: environment row
    └── flight_board.py  # NEW: main flight board

tests/karn/overwatch/
├── test_schema.py       # (Stage 0)
├── test_replay.py       # (Stage 0)
├── test_widgets.py      # Modified: add slot/env tests
├── test_app.py          # (Stage 1)
├── test_cli.py          # (Stage 1)
├── test_integration.py  # Modified: add navigation tests
├── test_hysteresis.py   # NEW: hysteresis tests
└── test_flight_board.py # NEW: flight board tests
```

---

## Next Stage

After Stage 2 is merged, proceed to **Stage 3: Header + Tamiyo Strip** which will:
- Replace Header placeholder with real widget
- Add Tamiyo Strip with PPO vitals
- Implement dual health indicators
- Add trend arrows (↑↓) for metrics
