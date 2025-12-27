# TamiyoBrain Side-by-Side Layout Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign TamiyoBrain widget from vertical stack to side-by-side layout with compact decision cards, giving learning vitals more breathing room while preserving 3 decisions for pattern detection.

**Architecture:** The widget switches between two layout modes based on terminal width: "horizontal" (≥96 chars) shows vitals on left 2/3 and 3 compact decisions on right 1/3; "stacked" (<85 chars) falls back to current vertical layout. Compact decision cards reduce from 5 lines to 2 lines while preserving critical diagnostic info.

**Tech Stack:** Python 3.12, Textual TUI, Rich tables/text, pytest async

---

## Current State

```
┌─ TAMIYO ──────────────────────────────────────────────────────────────────────┐
│ [OK] LEARNING  EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94 GradHP:OK batch:47/100│
│ ──────────────────────────────────────────────────────────────────────────────│
│ Expl.Var [████] 0.72  │  Entropy [██░░] 1.23  │  Advantage +0.12 ± 0.94       │
│ Clip Frac [█░░░] 0.08 │  KL Div [░░░░] 0.00   │  Policy Loss ▁▂▃▄ 0.032      │
│ ──────────────────────────────────────────────────────────────────────────────│
│ Heads: sl[███] bp[██░] sy[---] te[---] at[---] as[---] ac[---] op[---]        │
│ ──────────────────────────────────────────────────────────────────────────────│
│ [▓▓▓▓▓░░░░░░░░░░░░░░░] G=09 A=02 F=00 P=06 W=60                               │
│ ──────────────────────────────────────────────────────────────────────────────│
│ ┌─ DECISION 1 (12s ago) ─────────────────────────────────────────────────────┐│
│ │ SAW: r0c0: TRAINING | r0c1: BLENDING | Host: 87%                           ││
│ │ CHOSE: WAIT (92%)    Also: GERMINATE (5%) FOSSILIZE (3%)                   ││
│ │ EXPECTED: +0.12 → GOT: +0.08 ✓                                             ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ DECISION 2 ... ┐  ┌─ DECISION 3 ... ┐                                      │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Problems:**
- Gauges cramped (only ~45 chars for 4 gauges)
- Decisions take 15+ lines (5 lines × 3 decisions)
- Vertical scrolling required on smaller terminals

## Target State

```
┌─ TAMIYO ──────────────────────────────────────────────────────────────────────┐
│ [OK] LEARNING  EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94 GradHP:OK batch:47/100│
├────────────────────────────────────────────────────────┬─────────────────────┤
│ LEARNING VITALS (2/3)                                  │ DECISIONS (1/3)     │
│                                                        │                     │
│  Expl.Var            Entropy                           │ ┌─ D1 12s ────────┐ │
│  [████████░░] 0.72   [██████░░░░] 1.23                 │ │ WAIT 92%  H:87% │ │
│  "Learning!"         "Exploring"                       │ │ +0.12→+0.08 ✓   │ │
│                                                        │ └─────────────────┘ │
│  Clip Frac           KL Div            Advantage       │ ┌─ D2 28s ────────┐ │
│  [██░░░░░░░░] 0.08   [░░░░░░░░░░] 0.00 +0.12 ± 0.94   │ │ GERM 87%  H:91% │ │
│  "Very stable"       "Stable"          Ratio 0.9-1.1   │ │ +0.05→+0.11 ✓   │ │
│                                                        │ └─────────────────┘ │
│  Policy Loss ▁▂▃▄▅▆▇█ 0.032   Grad Norm ▁▂▃▄ 1.42     │ ┌─ D3 45s ────────┐ │
│  Value Loss  ▁▂▃▄▅▆▇█ 0.089   Layers OK 12/12         │ │ WAIT 95%  H:82% │ │
│ ──────────────────────────────────────────────────────│ │ +0.08→+0.06 ✓   │ │
│  Heads: sl[███] bp[██░] sy[---] te[---] at[---] ...   │ └─────────────────┘ │
│  Actions: [▓▓▓▓▓░░░░░░░░░░░] G=09 A=02 F=00 P=06 W=60 │                     │
└────────────────────────────────────────────────────────┴─────────────────────┘
```

**Improvements:**
- Gauges get full 2/3 width (~60 chars)
- 3 compact decisions in 1/3 column (~30 chars each)
- No vertical scrolling needed
- Clear cognitive separation: "system state" left, "recent behavior" right

---

## Task 1: Create Compact Decision Card Renderer

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 1.1: Write failing test for compact decision card

Add to `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
@pytest.mark.asyncio
async def test_compact_decision_card_format():
    """Compact decision card should fit in ~20 chars width with 2 lines."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        decision = DecisionSnapshot(
            decision_id="test-1",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[("GERMINATE", 0.05), ("FOSSILIZE", 0.03)],
            pinned=False,
        )

        # Render compact card
        card = widget._render_compact_decision(decision, index=0)
        card_str = str(card)

        # Should contain key info in compact format
        assert "D1" in card_str  # Decision number
        assert "WAIT" in card_str  # Action
        assert "92%" in card_str  # Confidence
        assert "H:87" in card_str or "H:88" in card_str  # Host accuracy (rounded)
        assert "0.12" in card_str  # Expected
        assert "0.08" in card_str  # Actual
        assert "✓" in card_str  # Good prediction indicator
```

### Step 1.2: Run test to verify it fails

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_compact_decision_card_format -v
```

Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_render_compact_decision'`

### Step 1.3: Implement _render_compact_decision

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py` after `_render_recent_decisions`:

```python
def _render_compact_decision(self, decision: "DecisionSnapshot", index: int) -> Text:
    """Render a compact 2-line decision card for side-by-side layout.

    Format (fits in ~22 chars width):
    ┌─ D1 12s ─────────┐
    │ WAIT 92%  H:87%  │
    │ +0.12→+0.08 ✓    │
    └──────────────────┘

    Args:
        decision: The decision snapshot to render.
        index: 0-indexed position (0=most recent).

    Returns:
        Rich Text with compact decision card.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    age = (now - decision.timestamp).total_seconds()
    age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"

    # Action colors
    action_colors = {
        "GERMINATE": "green",
        "WAIT": "dim",
        "FOSSILIZE": "blue",
        "PRUNE": "red",
        "SET_ALPHA_TARGET": "cyan",
        "ADVANCE": "cyan",
    }
    action_style = action_colors.get(decision.chosen_action, "white")

    # Pin indicator
    pin = "📌" if decision.pinned else ""

    # Build card
    card = Text()

    # Title line: D1 12s (with pin if applicable)
    card.append(f"┌─ D{index+1} {age_str} {pin}─────────┐\n", style="dim")

    # Line 1: ACTION PROB%  H:XX%
    action_abbrev = decision.chosen_action[:4].upper()  # WAIT, GERM, FOSS, PRUN, ADVA
    card.append("│ ")
    card.append(f"{action_abbrev}", style=action_style)
    card.append(f" {decision.confidence:.0%}", style="dim")
    card.append(f"  H:{decision.host_accuracy:.0f}%", style="cyan")
    card.append(" │\n")

    # Line 2: +0.12→+0.08 ✓/✗
    card.append("│ ")
    card.append(f"{decision.expected_value:+.2f}", style="dim")
    card.append("→", style="dim")
    if decision.actual_reward is not None:
        diff = abs(decision.actual_reward - decision.expected_value)
        style = "green" if diff < 0.1 else ("yellow" if diff < 0.3 else "red")
        icon = "✓" if diff < 0.1 else "✗"
        card.append(f"{decision.actual_reward:+.2f}", style=style)
        card.append(f" {icon}", style=style)
    else:
        card.append("...", style="dim italic")
    card.append("    │\n")

    # Bottom border
    card.append("└──────────────────┘", style="dim")

    return card
```

### Step 1.4: Run test to verify it passes

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_compact_decision_card_format -v
```

Expected: PASS

### Step 1.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(tamiyo): add _render_compact_decision for 2-line cards"
```

---

## Task 2: Create Decisions Column Renderer

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 2.1: Write failing test for decisions column

```python
@pytest.mark.asyncio
async def test_decisions_column_renders_three_cards():
    """Decisions column should render 3 compact decision cards vertically."""
    from esper.karn.sanctum.schema import DecisionSnapshot, TamiyoState, SanctumSnapshot
    from datetime import datetime, timezone, timedelta

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Create 3 decisions
        now = datetime.now(timezone.utc)
        decisions = [
            DecisionSnapshot(
                decision_id=f"test-{i}",
                timestamp=now - timedelta(seconds=i * 15),
                slot_states={"r0c0": "TRAINING"},
                host_accuracy=85.0 + i,
                chosen_action="WAIT" if i % 2 == 0 else "GERMINATE",
                chosen_slot=None,
                confidence=0.90 - i * 0.05,
                expected_value=0.1 * i,
                actual_reward=0.1 * i + 0.02,
                alternatives=[],
                pinned=False,
            )
            for i in range(3)
        ]

        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                recent_decisions=decisions,
                ppo_data_received=True,
            )
        )
        widget.update_snapshot(snapshot)

        # Render decisions column
        column = widget._render_decisions_column()
        column_str = str(column)

        # Should have 3 decision cards
        assert "D1" in column_str
        assert "D2" in column_str
        assert "D3" in column_str
        assert "WAIT" in column_str
        assert "GERM" in column_str
```

### Step 2.2: Run test to verify it fails

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decisions_column_renders_three_cards -v
```

Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_render_decisions_column'`

### Step 2.3: Implement _render_decisions_column

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
def _render_decisions_column(self) -> Text:
    """Render vertical stack of 3 compact decision cards for right column.

    Returns:
        Rich Text with stacked compact decision cards.
    """
    tamiyo = self._snapshot.tamiyo
    decisions = tamiyo.recent_decisions

    if not decisions:
        result = Text()
        result.append("DECISIONS\n", style="dim bold")
        result.append("No decisions yet\n", style="dim italic")
        result.append("Waiting for\n", style="dim")
        result.append("agent actions...", style="dim")
        return result

    # Store decision IDs for click handling
    self._decision_ids = [d.decision_id for d in decisions[:3]]

    result = Text()
    result.append("DECISIONS\n", style="dim bold")

    for i, decision in enumerate(decisions[:3]):
        card = self._render_compact_decision(decision, index=i)
        result.append(card)
        if i < 2:  # Add spacing between cards
            result.append("\n")

    return result
```

### Step 2.4: Run test to verify it passes

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decisions_column_renders_three_cards -v
```

Expected: PASS

### Step 2.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(tamiyo): add _render_decisions_column for side-by-side layout"
```

---

## Task 3: Create Vitals Column Renderer

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 3.1: Write failing test for vitals column

```python
@pytest.mark.asyncio
async def test_vitals_column_contains_all_components():
    """Vitals column should contain gauges, metrics, heads, and action bar."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=1.5,
                explained_variance=0.7,
                clip_fraction=0.1,
                kl_divergence=0.005,
                advantage_mean=0.15,
                advantage_std=0.8,
                policy_loss=0.03,
                value_loss=0.09,
                grad_norm=1.2,
            )
        )
        widget.update_snapshot(snapshot)

        # Render vitals column
        vitals = widget._render_vitals_column()
        vitals_str = str(vitals)

        # Should contain gauge labels
        assert "Expl.Var" in vitals_str
        assert "Entropy" in vitals_str
        assert "Clip" in vitals_str
        assert "KL" in vitals_str

        # Should contain metrics
        assert "Advantage" in vitals_str
        assert "Policy Loss" in vitals_str
        assert "Grad Norm" in vitals_str

        # Should contain heads heatmap marker
        assert "Heads:" in vitals_str

        # Should contain action bar marker
        assert "G=" in vitals_str or "W=" in vitals_str  # Action legend
```

### Step 3.2: Run test to verify it fails

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_vitals_column_contains_all_components -v
```

Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_render_vitals_column'`

### Step 3.3: Implement _render_vitals_column

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
def _render_vitals_column(self) -> Table:
    """Render left 2/3 column with all learning vitals.

    Contains (top to bottom):
    - Diagnostic matrix (gauges + metrics)
    - Separator
    - Head heatmap
    - Separator
    - Action distribution bar

    Returns:
        Rich Table with vertically stacked vitals components.
    """
    tamiyo = self._snapshot.tamiyo

    content = Table.grid(expand=True)
    content.add_column(ratio=1)

    if not tamiyo.ppo_data_received:
        waiting_text = Text(style="dim italic")
        waiting_text.append("⏳ Waiting for PPO vitals\n")
        waiting_text.append(
            f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
            style="cyan",
        )
        content.add_row(waiting_text)
        return content

    # Row 1: Diagnostic matrix (gauges left, metrics right)
    diagnostic_matrix = self._render_diagnostic_matrix()
    content.add_row(diagnostic_matrix)

    # Row 2: Separator
    content.add_row(self._render_separator())

    # Row 3: Head heatmap
    head_heatmap = self._render_head_heatmap()
    content.add_row(head_heatmap)

    # Row 4: Separator
    content.add_row(self._render_separator())

    # Row 5: Action distribution bar
    action_bar = self._render_action_distribution_bar()
    content.add_row(action_bar)

    return content
```

### Step 3.4: Run test to verify it passes

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_vitals_column_contains_all_components -v
```

Expected: PASS

### Step 3.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(tamiyo): add _render_vitals_column for side-by-side layout"
```

---

## Task 4: Add Layout Mode Detection

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 4.1: Write failing test for layout mode

```python
@pytest.mark.asyncio
async def test_layout_mode_horizontal_for_wide_terminal():
    """Wide terminals (≥96 chars) should use horizontal layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "horizontal"


@pytest.mark.asyncio
async def test_layout_mode_stacked_for_narrow_terminal():
    """Narrow terminals (<85 chars) should use stacked layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "stacked"


@pytest.mark.asyncio
async def test_layout_mode_compact_horizontal_for_medium_terminal():
    """Medium terminals (85-95 chars) should use compact-horizontal layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(90, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "compact-horizontal"
```

### Step 4.2: Run tests to verify they fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_layout_mode -v
```

Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_get_layout_mode'`

### Step 4.3: Implement _get_layout_mode

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py` after `COMPACT_THRESHOLD`:

```python
# Layout mode thresholds
HORIZONTAL_THRESHOLD = 96  # Full side-by-side
COMPACT_HORIZONTAL_THRESHOLD = 85  # Compressed side-by-side

def _get_layout_mode(self) -> str:
    """Determine layout mode based on terminal width.

    Returns:
        - "horizontal": Full side-by-side (≥96 chars)
        - "compact-horizontal": Compressed side-by-side (85-95 chars)
        - "stacked": Vertical stack fallback (<85 chars)
    """
    width = self.size.width
    if width >= self.HORIZONTAL_THRESHOLD:
        return "horizontal"
    elif width >= self.COMPACT_HORIZONTAL_THRESHOLD:
        return "compact-horizontal"
    else:
        return "stacked"
```

### Step 4.4: Run tests to verify they pass

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_layout_mode -v
```

Expected: PASS

### Step 4.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(tamiyo): add _get_layout_mode for responsive layout"
```

---

## Task 5: Update render() for Side-by-Side Layout

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 5.1: Write failing test for horizontal render

```python
@pytest.mark.asyncio
async def test_horizontal_layout_has_two_columns():
    """Horizontal layout should have vitals (left) and decisions (right)."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot, DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)):
        widget = app.query_one(TamiyoBrain)

        # Create snapshot with decisions
        decisions = [
            DecisionSnapshot(
                decision_id="test-1",
                timestamp=datetime.now(timezone.utc),
                slot_states={"r0c0": "TRAINING"},
                host_accuracy=87.0,
                chosen_action="WAIT",
                chosen_slot=None,
                confidence=0.92,
                expected_value=0.12,
                actual_reward=0.08,
                alternatives=[],
                pinned=False,
            )
        ]
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=1.5,
                recent_decisions=decisions,
            )
        )
        widget.update_snapshot(snapshot)

        # Force render
        rendered = widget.render()
        rendered_str = str(rendered)

        # Should have both vitals and decisions visible
        assert "Entropy" in rendered_str  # Vitals
        assert "D1" in rendered_str  # Compact decision
        assert "WAIT" in rendered_str  # Action in decision
```

### Step 5.2: Run test to verify current behavior

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_horizontal_layout_has_two_columns -v
```

Note: This may pass with current layout but we need to verify the side-by-side structure.

### Step 5.3: Update render() method

Replace the `render()` method in `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
def render(self):
    """Render Tamiyo content with responsive layout.

    Layout modes:
    - horizontal (≥96 chars): Side-by-side [vitals 2/3 | decisions 1/3]
    - compact-horizontal (85-95 chars): Compressed side-by-side
    - stacked (<85 chars): Vertical stack (legacy layout)
    """
    if self._snapshot is None:
        return Text("No data", style="dim")

    layout_mode = self._get_layout_mode()

    # Main layout container
    main_table = Table.grid(expand=True)

    # Row 1: Status Banner (always full width)
    main_table.add_column(ratio=1)
    status_banner = self._render_status_banner()
    main_table.add_row(status_banner)

    # Row 2: Separator
    main_table.add_row(self._render_separator())

    # Row 3: Content (layout-dependent)
    if layout_mode in ("horizontal", "compact-horizontal"):
        # Side-by-side: vitals left (2/3), decisions right (1/3)
        content_table = Table.grid(expand=True)
        content_table.add_column(ratio=2)  # Vitals (2/3)
        content_table.add_column(width=1)  # Separator
        content_table.add_column(ratio=1)  # Decisions (1/3)

        vitals_col = self._render_vitals_column()
        separator = Text("│\n" * 15, style="dim")  # Vertical separator
        decisions_col = self._render_decisions_column()

        content_table.add_row(vitals_col, separator, decisions_col)
        main_table.add_row(content_table)
    else:
        # Stacked layout (legacy)
        main_table.add_row(self._render_stacked_content())

    return main_table


def _render_stacked_content(self) -> Table:
    """Render legacy stacked layout for narrow terminals.

    Preserves original vertical stack behavior for <85 char terminals.
    """
    tamiyo = self._snapshot.tamiyo

    content = Table.grid(expand=True)
    content.add_column(ratio=1)

    # Diagnostic Matrix (gauges + metrics)
    if tamiyo.ppo_data_received:
        diagnostic_matrix = self._render_diagnostic_matrix()
        content.add_row(diagnostic_matrix)
    else:
        waiting_text = Text(style="dim italic")
        waiting_text.append("⏳ Waiting for PPO vitals\n")
        waiting_text.append(
            f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
            style="cyan",
        )
        content.add_row(waiting_text)

    # Separator
    content.add_row(self._render_separator())

    # Head heatmap
    if tamiyo.ppo_data_received:
        content.add_row(self._render_head_heatmap())
        content.add_row(self._render_separator())

    # Action bar
    content.add_row(self._render_action_distribution_bar())
    content.add_row(self._render_separator())

    # Full decision panels (legacy format)
    content.add_row(self._render_recent_decisions())

    return content
```

### Step 5.4: Run test and full suite

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_horizontal_layout_has_two_columns -v
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v
```

Expected: All tests PASS

### Step 5.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(tamiyo): implement side-by-side layout with responsive fallback"
```

---

## Task 6: Update Click Handling for New Layout

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 6.1: Write failing test for click in horizontal layout

```python
@pytest.mark.asyncio
async def test_click_decision_in_horizontal_layout():
    """Clicking on decision column should post DecisionPinToggled message."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot, DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)) as pilot:
        widget = app.query_one(TamiyoBrain)

        decisions = [
            DecisionSnapshot(
                decision_id="click-test-1",
                timestamp=datetime.now(timezone.utc),
                slot_states={},
                host_accuracy=87.0,
                chosen_action="WAIT",
                chosen_slot=None,
                confidence=0.92,
                expected_value=0.12,
                actual_reward=0.08,
                alternatives=[],
                pinned=False,
            )
        ]
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                recent_decisions=decisions,
            )
        )
        widget.update_snapshot(snapshot)
        await pilot.pause()

        # Decision IDs should be populated
        assert widget._decision_ids == ["click-test-1"]
```

### Step 6.2: Run test

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_click_decision_in_horizontal_layout -v
```

### Step 6.3: Update on_click for layout awareness

Update `on_click` in `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
def on_click(self, event) -> None:
    """Handle click to toggle decision pin.

    In horizontal layout: decisions are in right 1/3 column
    In stacked layout: decisions are at bottom after vitals
    """
    if not self._decision_ids:
        return

    layout_mode = self._get_layout_mode()

    if layout_mode in ("horizontal", "compact-horizontal"):
        # In horizontal layout, check if click is in right 1/3
        widget_width = self.size.width
        decision_column_start = int(widget_width * 2 / 3)

        if event.x < decision_column_start:
            return  # Click was in vitals column, not decisions

        # Calculate which decision card based on Y position
        # Each compact card is ~4 lines (title + 2 content + gap)
        header_height = 2  # Status banner + separator
        card_height = 4
        decision_y = event.y - header_height
        decision_index = max(0, decision_y // card_height)

    else:
        # Stacked layout: original click handling
        vitals_height = 7  # Approximate height of Learning Vitals section
        decision_height = 5  # Each full decision panel height

        y = event.y
        if y < vitals_height:
            return  # Click was in Learning Vitals

        decision_y = y - vitals_height
        decision_index = decision_y // decision_height

    if 0 <= decision_index < len(self._decision_ids):
        decision_id = self._decision_ids[decision_index]
        self.post_message(self.DecisionPinToggled(decision_id))
```

### Step 6.4: Run tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v
```

Expected: All tests PASS

### Step 6.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "fix(tamiyo): update click handling for side-by-side layout"
```

---

## Task 7: Final Verification and Polish

### Step 7.1: Run full test suite

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests PASS

### Step 7.2: Manual visual verification

```bash
# Run training to see live TUI
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 3
```

Verify:
- [ ] Wide terminal (≥96 chars): Side-by-side layout visible
- [ ] 3 compact decision cards in right column
- [ ] Vitals have more breathing room in left 2/3
- [ ] Narrow terminal (<85 chars): Falls back to stacked layout
- [ ] Click on decision cards still works for pinning
- [ ] A/B mode shows two TamiyoBrain widgets side-by-side

### Step 7.3: Final commit if any fixes needed

```bash
git add -A
git commit -m "fix(tamiyo): polish side-by-side layout based on manual testing"
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | Compact decision card renderer | `tamiyo_brain.py` | `test_compact_decision_card_format` |
| 2 | Decisions column renderer | `tamiyo_brain.py` | `test_decisions_column_renders_three_cards` |
| 3 | Vitals column renderer | `tamiyo_brain.py` | `test_vitals_column_contains_all_components` |
| 4 | Layout mode detection | `tamiyo_brain.py` | `test_layout_mode_*` (3 tests) |
| 5 | Update render() method | `tamiyo_brain.py` | `test_horizontal_layout_has_two_columns` |
| 6 | Update click handling | `tamiyo_brain.py` | `test_click_decision_in_horizontal_layout` |
| 7 | Final verification | - | Full suite + manual |

**Key Design Decisions:**
1. **Keep 3 decisions** - Pattern detection requires N, N-1, N-2 per DRL specialist
2. **Compact cards (2 lines)** - Fit in 1/3 width (~22 chars) per UX specialist
3. **Graceful fallback** - <85 chars reverts to stacked layout
4. **Preserve all critical info** - Action, probability, host%, expected→actual

**Risk Assessment: LOW** - Changes are additive (new renderers), existing functionality preserved in stacked mode.
