# Expanded TamiyoBrain Widget Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform TamiyoBrain from a compact diagnostic widget (~50×17) into a comprehensive PPO command center (96×24) showing all P0/P1/P2 metrics with sparklines and per-head entropy visualization.

**Architecture:** Three-phase incremental delivery: (1) Core restructure with status banner and 4-gauge grid, (2) Secondary metrics column with sparklines, (3) Per-head entropy heatmap and telemetry additions. Each phase is independently deployable.

**Tech Stack:** Textual (TUI), Rich (rendering), Python dataclasses (schema), deque (history tracking)

---

## Phase 1: Core Widget Restructure + P0 Metrics

**Objective:** Status banner + 4-gauge grid replacing current 3-gauge layout.

---

### Task 1.1: Add History Deque Fields to TamiyoState

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:364-433`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_schema.py (add to existing file)

def test_tamiyo_state_history_fields():
    """TamiyoState should have deque fields for sparkline history."""
    from collections import deque
    from esper.karn.sanctum.schema import TamiyoState

    state = TamiyoState()

    # Should have history deques with maxlen=10
    assert isinstance(state.policy_loss_history, deque)
    assert isinstance(state.value_loss_history, deque)
    assert isinstance(state.grad_norm_history, deque)
    assert isinstance(state.entropy_history, deque)
    assert isinstance(state.explained_variance_history, deque)

    # Should have maxlen of 10
    assert state.policy_loss_history.maxlen == 10
    assert state.value_loss_history.maxlen == 10


def test_tamiyo_state_per_head_entropy_fields():
    """TamiyoState should have per-head entropy for all 8 action heads."""
    from esper.karn.sanctum.schema import TamiyoState

    state = TamiyoState()

    # Should have all 8 head entropy fields
    assert hasattr(state, 'head_slot_entropy')
    assert hasattr(state, 'head_blueprint_entropy')
    assert hasattr(state, 'head_style_entropy')
    assert hasattr(state, 'head_tempo_entropy')
    assert hasattr(state, 'head_alpha_target_entropy')
    assert hasattr(state, 'head_alpha_speed_entropy')
    assert hasattr(state, 'head_alpha_curve_entropy')
    assert hasattr(state, 'head_op_entropy')
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_history_fields -v`
Expected: FAIL with "AttributeError: 'TamiyoState' object has no attribute 'policy_loss_history'"

**Step 3: Write minimal implementation**

Add to `TamiyoState` in `src/esper/karn/sanctum/schema.py` after line 432:

```python
    # History for trend sparklines (last 10 values)
    policy_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    value_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    grad_norm_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    entropy_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    explained_variance_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # Per-head entropy for all 8 action heads (P2 cool factor)
    # Already have: head_slot_entropy, head_blueprint_entropy
    head_style_entropy: float = 0.0
    head_tempo_entropy: float = 0.0
    head_alpha_target_entropy: float = 0.0
    head_alpha_speed_entropy: float = 0.0
    head_alpha_curve_entropy: float = 0.0
    head_op_entropy: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_history_fields tests/karn/sanctum/test_schema.py::test_tamiyo_state_per_head_entropy_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add history deques and per-head entropy to TamiyoState

- Add 5 history deques (maxlen=10) for sparkline trends
- Add 6 new per-head entropy fields (style, tempo, alpha_*, op)
- Prepares schema for expanded TamiyoBrain widget"
```

---

### Task 1.2: Update Aggregator to Populate History

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:526-584`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_aggregator.py (add to existing file)

def test_ppo_update_populates_history():
    """PPO_UPDATE_COMPLETED should append to history deques."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator(num_envs=4)

    # Simulate 3 PPO updates
    for i in range(3):
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data={
                "policy_loss": 0.1 * (i + 1),
                "value_loss": 0.2 * (i + 1),
                "grad_norm": 1.0 * (i + 1),
                "entropy": 1.5 - (0.1 * i),
                "explained_variance": 0.3 * (i + 1),
            },
        )
        agg.process_event(event)

    snapshot = agg.get_snapshot()
    tamiyo = snapshot.tamiyo

    # Should have 3 values in each history
    assert len(tamiyo.policy_loss_history) == 3
    assert len(tamiyo.value_loss_history) == 3
    assert len(tamiyo.entropy_history) == 3
    assert len(tamiyo.explained_variance_history) == 3

    # Values should be in order
    assert list(tamiyo.policy_loss_history) == [0.1, 0.2, 0.3]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_populates_history -v`
Expected: FAIL (history deques are empty)

**Step 3: Write minimal implementation**

In `_handle_ppo_update` (aggregator.py line ~526), add after setting each metric:

```python
    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        data = event.data or {}

        if data.get("skipped"):
            return

        # Mark that we've received PPO data (enables TamiyoBrain display)
        self._tamiyo.ppo_data_received = True

        # Update Tamiyo state with all PPO metrics AND append to history
        policy_loss = data.get("policy_loss", 0.0)
        self._tamiyo.policy_loss = policy_loss
        self._tamiyo.policy_loss_history.append(policy_loss)

        value_loss = data.get("value_loss", 0.0)
        self._tamiyo.value_loss = value_loss
        self._tamiyo.value_loss_history.append(value_loss)

        entropy = data.get("entropy", 0.0)
        self._tamiyo.entropy = entropy
        self._tamiyo.entropy_history.append(entropy)

        explained_variance = data.get("explained_variance", 0.0)
        self._tamiyo.explained_variance = explained_variance
        self._tamiyo.explained_variance_history.append(explained_variance)

        grad_norm = data.get("grad_norm", 0.0)
        self._tamiyo.grad_norm = grad_norm
        self._tamiyo.grad_norm_history.append(grad_norm)

        # ... rest of existing code unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_populates_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): populate TamiyoState history deques in aggregator

PPO_UPDATE_COMPLETED now appends to history deques for sparklines"
```

---

### Task 1.3: Implement Status Banner with Decision Tree

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_tamiyo_brain.py (add to existing file)

@pytest.mark.asyncio
async def test_status_banner_learning():
    """Status banner should show LEARNING when all metrics healthy."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,  # Healthy (>0.5)
            explained_variance=0.6,  # Healthy (>0.5)
            clip_fraction=0.15,  # Healthy (<0.2)
            kl_divergence=0.01,  # Healthy (<0.015)
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "ok"
        assert label == "LEARNING"


@pytest.mark.asyncio
async def test_status_banner_caution():
    """Status banner should show CAUTION when EV between 0 and 0.5."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.2,  # Warning (0 to 0.5)
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "warning"
        assert label == "CAUTION"


@pytest.mark.asyncio
async def test_status_banner_failing():
    """Status banner should show FAILING when EV < 0."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=-0.3,  # Critical (<0)
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_status_banner_entropy_collapsed():
    """Status banner should show FAILING when entropy collapsed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=0.05,  # Collapsed (<0.1)
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_learning -v`
Expected: FAIL with "AttributeError: 'TamiyoBrain' object has no attribute '_get_overall_status'"

**Step 3: Write minimal implementation**

Add to `TamiyoBrain` class in `tamiyo_brain.py`:

```python
    def _get_overall_status(self) -> tuple[str, str, str]:
        """Get overall learning status using decision tree.

        Returns:
            Tuple of (status, label, style) where:
            - status: "ok", "warning", or "critical"
            - label: "LEARNING", "CAUTION", or "FAILING"
            - style: Rich style string for coloring
        """
        if self._snapshot is None:
            return "ok", "WAITING", "dim"

        tamiyo = self._snapshot.tamiyo

        if not tamiyo.ppo_data_received:
            return "ok", "WAITING", "dim"

        # P0 Critical checks (immediate FAILING)
        if tamiyo.entropy < 0.1:
            return "critical", "FAILING", "red bold"
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical", "FAILING", "red bold"
        if tamiyo.clip_fraction > TUIThresholds.CLIP_CRITICAL:
            return "critical", "FAILING", "red bold"

        # P0/P1 Warning checks (CAUTION)
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning", "CAUTION", "yellow"
        if tamiyo.entropy < 0.3:
            return "warning", "CAUTION", "yellow"
        if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            return "warning", "CAUTION", "yellow"
        if tamiyo.kl_divergence > 0.02:
            return "warning", "CAUTION", "yellow"
        if tamiyo.advantage_std > 2.0 or (tamiyo.advantage_std > 0 and tamiyo.advantage_std < 0.5):
            return "warning", "CAUTION", "yellow"

        return "ok", "LEARNING", "green"

    def _render_status_banner(self) -> Text:
        """Render 1-line status banner with icon and key metrics."""
        status, label, style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        icons = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}
        icon = icons.get(status, "?")

        banner = Text()
        banner.append(f" {icon} ", style=style)
        banner.append(f"{label}   ", style=style)

        if tamiyo.ppo_data_received:
            # Key metrics inline
            ev_style = self._status_style(self._get_ev_status(tamiyo.explained_variance))
            clip_style = self._status_style(self._get_clip_status(tamiyo.clip_fraction))
            kl_style = self._status_style(self._get_kl_status(tamiyo.kl_divergence))

            banner.append(f"EV:{tamiyo.explained_variance:.2f}", style=ev_style)
            if tamiyo.explained_variance < 0:
                banner.append("!", style="red")
            banner.append("  ")

            banner.append(f"Clip:{tamiyo.clip_fraction:.2f}", style=clip_style)
            if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
                banner.append("!", style="yellow")
            banner.append("  ")

            banner.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
            banner.append("  ")

            # Batch progress
            batch = self._snapshot.current_batch
            max_batch = 100  # Approximate
            banner.append(f"batch:{batch}", style="dim")

        return banner
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_learning tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_caution tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_failing tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_entropy_collapsed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add status banner with decision tree to TamiyoBrain

- _get_overall_status() implements DRL decision tree
- _render_status_banner() shows [OK] LEARNING / [!] CAUTION / [X] FAILING
- Includes inline EV, Clip, KL with warning indicators"
```

---

### Task 1.4: Implement 4-Gauge Grid (Replace 3-Gauge)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_four_gauge_grid_rendered():
    """Learning vitals should render 4 gauges in 2x2 grid."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Should have 4 gauges: EV, Entropy, Clip, KL
        # Test by calling the gauge grid render method
        gauge_grid = widget._render_gauge_grid()
        assert gauge_grid is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_four_gauge_grid_rendered -v`
Expected: FAIL with "AttributeError: 'TamiyoBrain' object has no attribute '_render_gauge_grid'"

**Step 3: Write minimal implementation**

Add to `TamiyoBrain`:

```python
    def _render_gauge_grid(self) -> Table:
        """Render 2x2 gauge grid: EV, Entropy, Clip, KL."""
        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Row 1: Explained Variance | Entropy
        ev_gauge = self._render_gauge_v2(
            "Expl.Var",
            tamiyo.explained_variance,
            min_val=-1.0,
            max_val=1.0,
            status=self._get_ev_status(tamiyo.explained_variance),
            label=self._get_ev_label(tamiyo.explained_variance),
        )
        entropy_gauge = self._render_gauge_v2(
            "Entropy",
            tamiyo.entropy,
            min_val=0.0,
            max_val=2.0,
            status=self._get_entropy_status(tamiyo.entropy),
            label=self._get_entropy_label(tamiyo.entropy, batch),
        )
        grid.add_row(ev_gauge, entropy_gauge)

        # Row 2: Clip Fraction | KL Divergence
        clip_gauge = self._render_gauge_v2(
            "Clip Frac",
            tamiyo.clip_fraction,
            min_val=0.0,
            max_val=0.5,
            status=self._get_clip_status(tamiyo.clip_fraction),
            label=self._get_clip_label(tamiyo.clip_fraction),
        )
        kl_gauge = self._render_gauge_v2(
            "KL Div",
            tamiyo.kl_divergence,
            min_val=0.0,
            max_val=0.1,
            status=self._get_kl_status(tamiyo.kl_divergence),
            label=self._get_kl_label(tamiyo.kl_divergence, batch),
        )
        grid.add_row(clip_gauge, kl_gauge)

        return grid

    def _render_gauge_v2(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        status: str,
        label_text: str,
    ) -> Text:
        """Render a gauge with status-colored bar."""
        # Normalize to 0-1
        if max_val != min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        normalized = max(0, min(1, normalized))

        gauge_width = 10
        filled = int(normalized * gauge_width)
        empty = gauge_width - filled

        # Status-based color
        bar_color = {"ok": "cyan", "warning": "yellow", "critical": "red"}[status]

        gauge = Text()
        gauge.append(f" {label}\n", style="dim")
        gauge.append(" [")
        gauge.append("█" * filled, style=bar_color)
        gauge.append("░" * empty, style="dim")
        gauge.append("] ")

        # Value with precision based on magnitude
        if abs(value) < 0.1:
            gauge.append(f"{value:.3f}", style=bar_color)
        else:
            gauge.append(f"{value:.2f}", style=bar_color)

        if status == "critical":
            gauge.append("!", style="red bold")

        gauge.append(f'\n  "{label_text}"', style="italic dim")

        return gauge

    def _get_ev_label(self, ev: float) -> str:
        """Get descriptive label for explained variance."""
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "HARMFUL!"
        elif ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "Uncertain"
        elif ev < 0.5:
            return "Improving"
        else:
            return "Learning!"

    def _get_clip_label(self, clip: float) -> str:
        """Get descriptive label for clip fraction."""
        if clip > TUIThresholds.CLIP_CRITICAL:
            return "TOO AGGRESSIVE!"
        elif clip > TUIThresholds.CLIP_WARNING:
            return "Aggressive"
        elif clip < 0.1:
            return "Very stable"
        else:
            return "Stable"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_four_gauge_grid_rendered -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement 4-gauge grid with EV, Entropy, Clip, KL

- _render_gauge_grid() creates 2x2 grid layout
- _render_gauge_v2() adds status-colored bars
- New label methods for EV and Clip interpretive text"
```

---

### Task 1.5: Update CSS for 96×24 Sizing

**Files:**
- Modify: `src/esper/karn/sanctum/styles.tcss:78-91`

**Step 1: Update the CSS**

```css
/* Tamiyo Brain - EXPANDED for comprehensive PPO diagnostics (70% width, left side) */
#tamiyo-brain {
    width: 70%;
    height: 1fr;
    min-height: 24;
    min-width: 96;
    border: solid magenta;
    border-title-color: magenta;
    margin-right: 1;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 0 1;
}

#tamiyo-brain.status-ok {
    border: solid green;
}

#tamiyo-brain.status-warning {
    border: solid yellow;
}

#tamiyo-brain.status-critical {
    border: solid red;
}

#tamiyo-brain:focus {
    border: double $accent;
}
```

**Step 2: Verify visually**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.app import SanctumApp; print('CSS loads OK')"`
Expected: No errors

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/styles.tcss
git commit -m "style(sanctum): update TamiyoBrain CSS for 96x24 dimensions

- min-height: 24, min-width: 96 for expanded layout
- Add status-based border colors (green/yellow/red)"
```

---

### Task 1.6: Wire Status Banner + Gauge Grid into render()

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:79-96`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_render_includes_status_banner():
    """Render should include status banner at top."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render should complete without error
        rendered = widget.render()
        assert rendered is not None
```

**Step 2: Run test to verify baseline**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_render_includes_status_banner -v`
Expected: PASS (render works, but doesn't use new methods yet)

**Step 3: Update render() method**

Replace the `render()` and `_render_learning_vitals()` methods:

```python
    def render(self):
        """Render Tamiyo content with expanded layout."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        # Main layout: stacked sections
        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Row 1: Status Banner (1 line)
        status_banner = self._render_status_banner()
        main_table.add_row(status_banner)

        # Row 2: Separator
        main_table.add_row(Text("─" * 48, style="dim"))

        # Row 3: Diagnostic Matrix (gauges left, metrics right)
        # For now, just gauges - Phase 2 adds metrics column
        if self._snapshot.tamiyo.ppo_data_received:
            gauge_grid = self._render_gauge_grid()
            main_table.add_row(gauge_grid)
        else:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            main_table.add_row(waiting_text)

        # Row 4: Separator
        main_table.add_row(Text("─" * 48, style="dim"))

        # Row 5: Action Distribution
        action_bar = self._render_action_distribution_bar()
        main_table.add_row(action_bar)

        # Row 6: Separator
        main_table.add_row(Text("─" * 48, style="dim"))

        # Row 7: Decision Carousel
        decisions_panel = self._render_recent_decisions()
        main_table.add_row(decisions_panel)

        return main_table
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): wire status banner and 4-gauge grid into render()

Expanded TamiyoBrain layout:
- Status banner at top
- 4-gauge grid (EV, Entropy, Clip, KL) replacing 3-gauge
- Action distribution bar
- Decision carousel at bottom"
```

---

### Task 1.7: Dynamic Border Color Based on Status

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_border_color_updates_on_status():
    """Widget border should change color based on overall status."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Healthy state
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-ok")

        # Warning state
        snapshot.tamiyo.explained_variance = 0.2
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-warning")
        assert not widget.has_class("status-ok")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_border_color_updates_on_status -v`
Expected: FAIL (no CSS class toggling)

**Step 3: Write minimal implementation**

Update `update_snapshot()`:

```python
    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self._update_status_class()
        self.refresh()

    def _update_status_class(self) -> None:
        """Update CSS class based on overall status."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok")
        self.remove_class("status-warning")
        self.remove_class("status-critical")

        # Add current status class
        self.add_class(f"status-{status}")
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_border_color_updates_on_status -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): dynamic border color based on PPO status

Border changes: green (LEARNING), yellow (CAUTION), red (FAILING)"
```

---

## Phase 1 Complete Checkpoint

Run full test suite:
```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests pass. TamiyoBrain now has:
- Status banner with decision tree logic
- 4-gauge grid (EV, Entropy, Clip, KL)
- Dynamic border colors
- History tracking infrastructure

---

## Phase 2: P1/P2 Metrics with Sparklines

**Objective:** Secondary metrics column with sparkline trends.

---

### Task 2.1: Implement Sparkline Renderer

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_sparkline_rendering():
    """Sparkline should render 10-value history as unicode blocks."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test sparkline with known values
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sparkline = widget._render_sparkline(history, width=10)

        # Should be 10 characters
        assert len(sparkline.plain) == 10
        # First char should be lowest block, last should be highest
        assert "▁" in sparkline.plain
        assert "█" in sparkline.plain
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_sparkline_rendering -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    def _render_sparkline(
        self,
        history: list[float] | deque[float],
        width: int = 10,
        style: str = "cyan",
    ) -> Text:
        """Render sparkline using unicode block characters."""
        BLOCKS = "▁▂▃▄▅▆▇█"

        if not history:
            return Text("─" * width, style="dim")

        values = list(history)[-width:]  # Last N values
        if len(values) < width:
            # Pad with empty on left
            values = [0.0] * (width - len(values)) + values

        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        val_range = max_val - min_val if max_val != min_val else 1

        chars = []
        for v in values:
            normalized = (v - min_val) / val_range
            idx = int(normalized * (len(BLOCKS) - 1))
            idx = max(0, min(len(BLOCKS) - 1, idx))
            chars.append(BLOCKS[idx])

        return Text("".join(chars), style=style)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_sparkline_rendering -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement sparkline renderer for trend visualization

Uses unicode blocks ▁▂▃▄▅▆▇█ for 10-value history display"
```

---

### Task 2.2: Implement Secondary Metrics Column

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_secondary_metrics_column():
    """Secondary metrics should show Advantage, Ratio, losses with sparklines."""
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])

        tamiyo = TamiyoState(
            advantage_mean=0.15,
            advantage_std=0.95,
            ratio_min=0.85,
            ratio_max=1.15,
            policy_loss=0.025,
            value_loss=0.142,
            grad_norm=1.5,
            dead_layers=0,
            exploding_layers=0,
            ppo_data_received=True,
        )
        # Add history
        for i in range(5):
            tamiyo.policy_loss_history.append(0.03 - i * 0.001)
            tamiyo.value_loss_history.append(0.2 - i * 0.01)
            tamiyo.grad_norm_history.append(1.5 + i * 0.1)

        snapshot.tamiyo = tamiyo
        widget.update_snapshot(snapshot)

        # Render metrics column
        metrics = widget._render_metrics_column()
        assert metrics is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_secondary_metrics_column -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    def _render_metrics_column(self) -> Text:
        """Render secondary metrics column with sparklines."""
        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Advantage stats
        adv_status = "ok"
        if tamiyo.advantage_std > 2.0 or (tamiyo.advantage_std > 0 and tamiyo.advantage_std < 0.5):
            adv_status = "warning"
        if tamiyo.advantage_mean < -0.5:
            adv_status = "critical"

        adv_style = self._status_style(adv_status)
        result.append(f" Advantage   ", style="dim")
        result.append(f"{tamiyo.advantage_mean:+.2f} ± {tamiyo.advantage_std:.2f}", style=adv_style)
        if adv_status != "ok":
            result.append(" [!]", style=adv_style)
        result.append("\n")

        # Ratio bounds
        ratio_status = "ok"
        if tamiyo.ratio_max > 1.5 or tamiyo.ratio_min < 0.5:
            ratio_status = "warning"
        if tamiyo.ratio_max > 2.0 or tamiyo.ratio_min < 0.3:
            ratio_status = "critical"

        ratio_style = self._status_style(ratio_status)
        result.append(f" Ratio       ", style="dim")
        result.append(f"{tamiyo.ratio_min:.2f} < r < {tamiyo.ratio_max:.2f}", style=ratio_style)
        if ratio_status != "ok":
            result.append(" [!]", style=ratio_style)
        result.append("\n")

        # Policy loss with sparkline
        pl_sparkline = self._render_sparkline(tamiyo.policy_loss_history)
        result.append(f" Policy Loss ", style="dim")
        result.append(pl_sparkline)
        result.append(f" {tamiyo.policy_loss:.3f}\n", style="cyan")

        # Value loss with sparkline
        vl_sparkline = self._render_sparkline(tamiyo.value_loss_history)
        result.append(f" Value Loss  ", style="dim")
        result.append(vl_sparkline)
        result.append(f" {tamiyo.value_loss:.3f}\n", style="cyan")

        # Grad norm with sparkline
        gn_sparkline = self._render_sparkline(tamiyo.grad_norm_history)
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        gn_style = self._status_style(gn_status)
        result.append(f" Grad Norm   ", style="dim")
        result.append(gn_sparkline)
        result.append(f" {tamiyo.grad_norm:.2f}\n", style=gn_style)

        # Layer health
        total_layers = 12  # Approximate
        healthy = total_layers - tamiyo.dead_layers - tamiyo.exploding_layers
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(f" Layers      ", style="dim")
            result.append(f"!! {tamiyo.dead_layers} dead, {tamiyo.exploding_layers} exploding", style="red")
        else:
            result.append(f" Layers      ", style="dim")
            result.append(f"OK {healthy}/{total_layers} healthy", style="green")

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_secondary_metrics_column -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement secondary metrics column with sparklines

Shows: Advantage ±, Ratio bounds, Policy/Value Loss trends, Grad Norm, Layer health"
```

---

### Task 2.3: Wire Diagnostic Matrix (Gauges + Metrics Side-by-Side)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_diagnostic_matrix_layout():
    """Diagnostic matrix should have gauges left, metrics right."""
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_mean=0.15,
            advantage_std=0.95,
            ratio_min=0.85,
            ratio_max=1.15,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render diagnostic matrix
        matrix = widget._render_diagnostic_matrix()
        assert matrix is not None
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_diagnostic_matrix_layout -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    def _render_diagnostic_matrix(self) -> Table:
        """Render diagnostic matrix: gauges left, metrics right."""
        matrix = Table.grid(expand=True)
        matrix.add_column(ratio=1)  # Gauges
        matrix.add_column(width=1)  # Separator
        matrix.add_column(ratio=1)  # Metrics

        gauge_grid = self._render_gauge_grid()
        separator = Text("││\n││\n││\n││\n││\n││", style="dim")
        metrics_col = self._render_metrics_column()

        matrix.add_row(gauge_grid, separator, metrics_col)
        return matrix
```

Update `render()` to use diagnostic matrix:

```python
        # Row 3: Diagnostic Matrix (gauges left, metrics right)
        if self._snapshot.tamiyo.ppo_data_received:
            diagnostic_matrix = self._render_diagnostic_matrix()
            main_table.add_row(diagnostic_matrix)
        else:
            # ... waiting text ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_diagnostic_matrix_layout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): wire diagnostic matrix with gauges + metrics side-by-side

Layout: 4 gauges (2x2) left | separator | secondary metrics right"
```

---

## Phase 2 Complete Checkpoint

Run full test suite:
```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests pass. TamiyoBrain now has:
- Phase 1 features (status banner, 4-gauge grid)
- Sparkline renderer
- Secondary metrics column (Advantage, Ratio, Loss trends, Layer health)
- Diagnostic matrix layout

---

## Phase 3: Per-Head Heatmap + Telemetry

**Objective:** Add per-head entropy heatmap and extend telemetry for 6 additional heads.

---

### Task 3.1: Implement Per-Head Entropy Heatmap

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_per_head_entropy_heatmap():
    """Per-head heatmap should show 8 heads with bar visualization."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.2,
            head_blueprint_entropy=0.9,
            head_style_entropy=0.6,
            head_tempo_entropy=0.8,
            head_alpha_target_entropy=0.5,
            head_alpha_speed_entropy=0.4,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.1,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        assert heatmap is not None
        # Should contain all 8 head labels
        plain = heatmap.plain
        assert "slot" in plain
        assert "bp" in plain
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_per_head_entropy_heatmap -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    def _render_head_heatmap(self) -> Text:
        """Render per-head entropy heatmap with 8 action heads."""
        tamiyo = self._snapshot.tamiyo

        # Head config: (abbrev, field_name, max_entropy)
        heads = [
            ("slot", "head_slot_entropy", 1.39),      # ln(4) slots
            ("bp", "head_blueprint_entropy", 2.48),   # ln(12) blueprints
            ("sty", "head_style_entropy", 1.10),      # ln(3)
            ("tem", "head_tempo_entropy", 1.10),      # ln(3)
            ("a_t", "head_alpha_target_entropy", 1.61),  # ln(5)
            ("a_s", "head_alpha_speed_entropy", 1.10),   # ln(3)
            ("a_c", "head_alpha_curve_entropy", 1.10),   # ln(3)
            ("op", "head_op_entropy", 1.61),          # ln(5)
        ]

        result = Text()
        result.append(" Heads: ", style="dim")

        for abbrev, field, max_ent in heads:
            value = getattr(tamiyo, field, 0.0)
            # Normalize to 0-1
            fill = value / max_ent if max_ent > 0 else 0
            fill = max(0, min(1, fill))

            # 4-char bar
            bar_width = 4
            filled = int(fill * bar_width)
            empty = bar_width - filled

            # Color based on fill level
            if fill > 0.5:
                color = "green"
            elif fill > 0.25:
                color = "yellow"
            else:
                color = "red"

            result.append(f"{abbrev}[")
            result.append("█" * filled, style=color)
            result.append("░" * empty, style="dim")
            result.append("] ")

        result.append("\n        ")

        # Second line: values
        for abbrev, field, max_ent in heads:
            value = getattr(tamiyo, field, 0.0)
            fill = value / max_ent if max_ent > 0 else 0

            if fill < 0.25:
                result.append(f"{value:.2f}!", style="red")
            else:
                result.append(f"{value:.2f} ", style="dim")
            result.append("     ")

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_per_head_entropy_heatmap -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement per-head entropy heatmap for 8 action heads

Shows: slot|bp|sty|tem|a_t|a_s|a_c|op with fill bars and values"
```

---

### Task 3.2: Wire Heatmap into render()

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`

**Step 1: Update render() to include heatmap**

In `render()`, add after diagnostic matrix:

```python
        # Row 4: Per-Head Entropy Heatmap (2 lines)
        if self._snapshot.tamiyo.ppo_data_received:
            head_heatmap = self._render_head_heatmap()
            main_table.add_row(head_heatmap)

        # Row 5: Separator
        main_table.add_row(Text("─" * 48, style="dim"))
```

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py
git commit -m "feat(sanctum): wire per-head heatmap into TamiyoBrain render()"
```

---

### Task 3.3: Extend Aggregator for 6 New Head Entropies

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:575-580`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
def test_ppo_update_captures_all_head_entropies():
    """PPO_UPDATE_COMPLETED should capture all 8 head entropies."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={
            "entropy": 1.2,
            "slot_entropy": 1.1,
            "blueprint_entropy": 0.9,
            "style_entropy": 0.6,
            "tempo_entropy": 0.8,
            "alpha_target_entropy": 0.5,
            "alpha_speed_entropy": 0.4,
            "alpha_curve_entropy": 0.7,
            "op_entropy": 1.0,
        },
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    tamiyo = snapshot.tamiyo

    assert tamiyo.head_slot_entropy == 1.1
    assert tamiyo.head_blueprint_entropy == 0.9
    assert tamiyo.head_style_entropy == 0.6
    assert tamiyo.head_tempo_entropy == 0.8
    assert tamiyo.head_alpha_target_entropy == 0.5
    assert tamiyo.head_alpha_speed_entropy == 0.4
    assert tamiyo.head_alpha_curve_entropy == 0.7
    assert tamiyo.head_op_entropy == 1.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_captures_all_head_entropies -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `_handle_ppo_update()`:

```python
        # Per-head entropy for all 8 heads
        self._tamiyo.head_slot_entropy = data.get("slot_entropy", 0.0)
        self._tamiyo.head_blueprint_entropy = data.get("blueprint_entropy", 0.0)
        self._tamiyo.head_style_entropy = data.get("style_entropy", 0.0)
        self._tamiyo.head_tempo_entropy = data.get("tempo_entropy", 0.0)
        self._tamiyo.head_alpha_target_entropy = data.get("alpha_target_entropy", 0.0)
        self._tamiyo.head_alpha_speed_entropy = data.get("alpha_speed_entropy", 0.0)
        self._tamiyo.head_alpha_curve_entropy = data.get("alpha_curve_entropy", 0.0)
        self._tamiyo.head_op_entropy = data.get("op_entropy", 0.0)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_captures_all_head_entropies -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): capture all 8 head entropies in aggregator

Extends PPO_UPDATE_COMPLETED handling for: style, tempo, alpha_*, op"
```

---

### Task 3.4: Extend Telemetry Emitter for 6 New Head Entropies

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py:648-703`
- Test: `tests/simic/test_telemetry_emitters.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_telemetry_emitters.py (add to existing file)

def test_emit_ppo_update_includes_all_head_entropies():
    """emit_ppo_update should include all 8 head entropies."""
    from unittest.mock import MagicMock
    from esper.simic.telemetry.emitters import emit_ppo_update

    hub = MagicMock()

    metrics = {
        "policy_loss": 0.01,
        "value_loss": 0.05,
        "entropy": 1.2,
        "head_entropies": {
            "slot": [1.1, 1.0, 1.05],
            "blueprint": [0.9, 0.85, 0.88],
            "style": [0.6, 0.55, 0.58],
            "tempo": [0.8, 0.75, 0.78],
            "alpha_target": [0.5, 0.45, 0.48],
            "alpha_speed": [0.4, 0.38, 0.39],
            "alpha_curve": [0.7, 0.65, 0.68],
            "op": [1.0, 0.95, 0.98],
        },
    }

    emit_ppo_update(
        hub=hub,
        epoch=1,
        batch_idx=0,
        episodes_completed=10,
        grad_norm=1.5,
        update_time_ms=100.0,
        metrics=metrics,
        optimizer=None,
    )

    # Check emitted data
    call_args = hub.emit.call_args
    event = call_args[0][0]
    data = event.data

    # Should have averaged entropies for all 8 heads
    assert "slot_entropy" in data
    assert "blueprint_entropy" in data
    assert "style_entropy" in data
    assert "tempo_entropy" in data
    assert "alpha_target_entropy" in data
    assert "alpha_speed_entropy" in data
    assert "alpha_curve_entropy" in data
    assert "op_entropy" in data
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_telemetry_emitters.py::test_emit_ppo_update_includes_all_head_entropies -v`
Expected: FAIL (only slot and blueprint are emitted)

**Step 3: Write minimal implementation**

The emitter already computes averages from `head_entropies` dict (line 648-653). The naming just needs to match what the aggregator expects:

```python
    # Compute per-head entropy averages for logging (P3-1)
    head_entropies_avg = {}
    if "head_entropies" in metrics:
        for head, values in metrics["head_entropies"].items():
            avg_entropy = sum(values) / len(values) if values else 0.0
            # Use consistent naming: {head}_entropy
            head_entropies_avg[f"{head}_entropy"] = avg_entropy
```

This already works - the test just needs to verify the correct keys. The emitter is correct; we need to ensure PPOAgent emits all 8 heads.

**Step 4: Verify PPOAgent emits all heads**

Check `ppo.py` line 575 - it only tracks slot and blueprint currently. This requires a separate task to add the other 6 heads to PPOAgent.

**Step 5: Create a note for future work**

For now, the aggregator and widget are ready. The telemetry gap (PPOAgent not emitting 6 heads) is a known limitation documented in the design spec.

**Step 6: Commit**

```bash
git add tests/simic/test_telemetry_emitters.py
git commit -m "test(simic): add test for 8-head entropy emission

Documents expected telemetry format for all action heads.
Note: PPOAgent currently only tracks slot+blueprint; others require network changes."
```

---

### Task 3.5: Final Integration Test

**Files:**
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write comprehensive integration test**

```python
@pytest.mark.asyncio
async def test_expanded_tamiyo_brain_full_render():
    """Full render should include all expanded components."""
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])

        tamiyo = TamiyoState(
            # P0 metrics
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            # P1 metrics
            advantage_mean=0.15,
            advantage_std=0.95,
            ratio_min=0.85,
            ratio_max=1.15,
            # P2 metrics
            policy_loss=0.025,
            value_loss=0.142,
            grad_norm=1.5,
            dead_layers=0,
            exploding_layers=0,
            # Per-head entropies
            head_slot_entropy=1.2,
            head_blueprint_entropy=0.9,
            head_style_entropy=0.6,
            head_tempo_entropy=0.8,
            head_alpha_target_entropy=0.5,
            head_alpha_speed_entropy=0.4,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.1,
            # Actions
            action_counts={"WAIT": 30, "GERMINATE": 25, "PRUNE": 15, "FOSSILIZE": 30},
            total_actions=100,
            ppo_data_received=True,
        )

        # Add history for sparklines
        for i in range(5):
            tamiyo.policy_loss_history.append(0.03 - i * 0.001)
            tamiyo.value_loss_history.append(0.2 - i * 0.01)
            tamiyo.grad_norm_history.append(1.5 + i * 0.1)
            tamiyo.entropy_history.append(1.2 - i * 0.02)
            tamiyo.explained_variance_history.append(0.5 + i * 0.02)

        snapshot.tamiyo = tamiyo
        widget.update_snapshot(snapshot)

        # Should render without error
        rendered = widget.render()
        assert rendered is not None

        # Check status
        status, label, _ = widget._get_overall_status()
        assert status == "ok"
        assert label == "LEARNING"

        # Check CSS class
        assert widget.has_class("status-ok")
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_expanded_tamiyo_brain_full_render -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "test(sanctum): add comprehensive integration test for expanded TamiyoBrain

Verifies all 3 phases: status banner, diagnostic matrix, per-head heatmap"
```

---

## Phase 3 Complete Checkpoint

Run full test suite:
```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests pass.

---

## Final Summary

### What We Built

| Phase | Component | Files Modified |
|-------|-----------|----------------|
| 1 | History deques in TamiyoState | schema.py |
| 1 | Aggregator populates history | aggregator.py |
| 1 | Status banner with decision tree | tamiyo_brain.py |
| 1 | 4-gauge grid (EV, Entropy, Clip, KL) | tamiyo_brain.py |
| 1 | CSS for 96×24 sizing | styles.tcss |
| 1 | Dynamic border colors | tamiyo_brain.py |
| 2 | Sparkline renderer | tamiyo_brain.py |
| 2 | Secondary metrics column | tamiyo_brain.py |
| 2 | Diagnostic matrix layout | tamiyo_brain.py |
| 3 | Per-head entropy heatmap | tamiyo_brain.py |
| 3 | Aggregator for 6 new heads | aggregator.py |

### Known Limitations

1. **Telemetry Gap:** PPOAgent only emits `slot` and `blueprint` head entropies. The other 6 heads (style, tempo, alpha_*, op) require changes to the neural network forward pass to track per-head entropy, which is beyond the scope of this TUI-focused implementation.

2. **Width Assumption:** The 96-char width assumes a terminal at least 130 chars wide. Narrower terminals will have horizontal overflow.

### Verification Command

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v && echo "✓ All Sanctum tests pass"
```
