# Sanctum Head Decision Display Enhancement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance Sanctum TUI to better display policy head decisions using the expanded 80/20 layout, making collapse detection and decision reasoning more visible.

**Architecture:** Three changes: (1) Widen head heatmap bars from 3→5 chars for better granularity, (2) Add head choice details (blueprint, tempo) to decision cards, (3) Add last-action environment highlighting in EnvOverview table.

**Tech Stack:** Textual TUI, Rich Text formatting, Python dataclasses

---

## Task 1: Widen Head Heatmap Bars (3→5 chars)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:1787-1809`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Context:** The current 3-char bars (e.g., `[███]`) have poor granularity. With the 80% width expansion, we have room for 5-char bars which show visible difference between 60% and 80% fill levels.

**Specialist Amendment (UX Review):**
- Also update `HEAD_SEGMENT_WIDTH` constant from 10 to 12 (line 1659)

**Step 1: Write failing test for 5-char bars**

```python
# In tests/karn/sanctum/test_tamiyo_brain.py

def test_head_heatmap_uses_5_char_bars():
    """Head heatmap should use 5-char bars for improved granularity."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(head_slot_entropy=1.5)  # ~75% fill

    widget = TamiyoBrain()
    widget._snapshot = snapshot

    result = widget._render_head_heatmap()
    text = str(result)

    # 5-char bar should have 3-4 filled blocks at 75%
    # Pattern: abbrev[█████] or abbrev[████░]
    assert "████" in text or "███░" in text, f"Expected 5-char bar pattern, got: {text}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_head_heatmap_uses_5_char_bars -v`
Expected: FAIL (current bars are 3 chars)

**Step 3: Update bar_width constant and segment calculations**

In `src/esper/karn/sanctum/widgets/tamiyo_brain.py`, modify `_render_head_heatmap`:

```python
# Line ~1787: Change bar width from 3 to 5
# Before:
#             # 3-char bar (narrower for 80-char terminal compatibility)
#             bar_width = 3

# After:
            # 5-char bar (expanded for 80% width layout - better granularity)
            bar_width = 5
```

Also update the segment width comment at line ~1805:
```python
# Before:
#             # 9-char segment: "abbr[███] " = 4 + 1 + 3 + 1 + space = 9

# After:
            # 11-char segment: "abbr[█████] " = 4 + 1 + 5 + 1 + space = 11
```

And update the second line alignment at line ~1811:
```python
# Before:
#         result.append("\n        ")

# After:
        result.append("\n          ")  # 10 spaces to align under bars (adjusted for 5-char)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_head_heatmap_uses_5_char_bars -v`
Expected: PASS

**Step 5: Run full TamiyoBrain test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): widen head heatmap bars from 3 to 5 chars

Improves granularity for collapse detection. With the 80% layout width,
5-char bars show visible difference between 60% and 80% fill levels.

Per DRL specialist: entropy collapse on a single head can kill exploration
while aggregate metrics look fine - better bar resolution helps operators
spot issues earlier."
```

---

## Task 2: Add Head Choice Fields to DecisionSnapshot

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:619-640`
- Test: `tests/karn/sanctum/test_schema.py`

**Context:** Decision cards currently show action and slot but not the sub-decisions (blueprint, tempo, style). Adding these fields enables Task 3 to display them.

**Specialist Amendments (DRL Review):**
- Replace `alpha_curve` with `style` (more informative for credit assignment - style determines blend algorithm)
- Add `op_confidence` (needed for credit assignment debugging)

**Step 1: Write failing test for new fields**

```python
# In tests/karn/sanctum/test_schema.py (create if doesn't exist)

def test_decision_snapshot_has_head_choice_fields():
    """DecisionSnapshot should include blueprint and tempo head choices."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=75.0,
        chosen_action="GERMINATE",
        chosen_slot="slot_0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=None,
        alternatives=[],
        decision_id="test-1",
        decision_entropy=1.2,
        value_residual=0.1,
        # New fields (per DRL specialist review)
        chosen_blueprint="conv_light",
        chosen_tempo="STANDARD",
        chosen_style="LINEAR_ADD",  # Replaces alpha_curve - more diagnostic
        blueprint_confidence=0.87,
        tempo_confidence=0.65,
        op_confidence=0.92,  # Added per DRL review
    )

    assert decision.chosen_blueprint == "conv_light"
    assert decision.chosen_tempo == "STANDARD"
    assert decision.chosen_style == "LINEAR_ADD"
    assert decision.blueprint_confidence == 0.87
    assert decision.tempo_confidence == 0.65
    assert decision.op_confidence == 0.92
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_has_head_choice_fields -v`
Expected: FAIL (fields don't exist)

**Step 3: Add new fields to DecisionSnapshot**

In `src/esper/karn/sanctum/schema.py`, add to the `DecisionSnapshot` dataclass (after line ~649):

```python
@dataclass
class DecisionSnapshot:
    """Snapshot of a single Tamiyo decision for display.
    ...existing docstring...
    """
    timestamp: datetime
    slot_states: dict[str, str]
    host_accuracy: float
    chosen_action: str
    chosen_slot: str | None
    confidence: float
    expected_value: float
    actual_reward: float | None
    alternatives: list[tuple[str, float]]
    decision_id: str
    decision_entropy: float = 0.0
    value_residual: float = 0.0
    td_advantage: float | None = None
    is_pinned: bool = False
    # Head choice details for GERMINATE actions (per DRL/UX specialist review)
    chosen_blueprint: str | None = None  # e.g., "conv_light", "attention"
    chosen_tempo: str | None = None  # "FAST", "STANDARD", "SLOW"
    chosen_style: str | None = None  # "LINEAR_ADD", "GATED_GATE", etc. (replaces alpha_curve)
    blueprint_confidence: float = 0.0  # Probability of chosen blueprint
    tempo_confidence: float = 0.0  # Probability of chosen tempo
    op_confidence: float = 0.0  # Probability of chosen operation (added per DRL review)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_has_head_choice_fields -v`
Expected: PASS

**Step 5: Run full schema tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add head choice fields to DecisionSnapshot

Adds chosen_blueprint, chosen_tempo, chosen_style, op_confidence and their
confidence values. These enable decision cards to show sub-decision
details like 'bpnt:conv_l(87%) temp:STD'.

Per DRL specialist: understanding which heads drive decisions helps
diagnose credit assignment issues."
```

---

## Task 3: Display Head Choices in Decision Cards

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:984-1001`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Context:** Decision cards should show head choices (blueprint, tempo) when available, especially for GERMINATE actions.

**Step 1: Write failing test for head choice display**

```python
# In tests/karn/sanctum/test_tamiyo_brain.py

def test_decision_card_shows_head_choices():
    """Decision cards should display blueprint and tempo for GERMINATE."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=75.0,
        chosen_action="GERMINATE",
        chosen_slot="slot_0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=0.3,
        alternatives=[("WAIT", 0.06)],
        decision_id="test-1",
        decision_entropy=1.2,
        value_residual=0.1,
        chosen_blueprint="conv_light",
        chosen_tempo="STANDARD",
        blueprint_confidence=0.87,
        tempo_confidence=0.65,
    )

    widget = TamiyoBrain()
    card = widget._render_enriched_decision(decision, index=0, total_cards=1)
    text = str(card)

    # Should contain head choice line
    assert "conv" in text.lower() or "bpnt" in text.lower(), f"Expected blueprint in card: {text}"
    assert "std" in text.lower() or "tempo" in text.lower(), f"Expected tempo in card: {text}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_shows_head_choices -v`
Expected: FAIL (head choices not displayed)

**Step 3: Add head choice line to _render_enriched_decision**

In `src/esper/karn/sanctum/widgets/tamiyo_brain.py`, after line 5 (alternatives line, ~line 1001), add a new line 6:

```python
        # Line 6: Head choices for GERMINATE (per DRL/UX specialist review)
        # Format: "bpnt:conv_l(87%) tmp:STD"
        card.append("│", style=border_style)
        card.append(" ")
        if decision.chosen_action == "GERMINATE" and decision.chosen_blueprint:
            # Abbreviate blueprint name (first 6 chars)
            bp_abbrev = decision.chosen_blueprint[:6] if decision.chosen_blueprint else "?"
            bp_conf = f"({decision.blueprint_confidence:.0%})" if decision.blueprint_confidence else ""
            # Abbreviate tempo (FAST→FST, STANDARD→STD, SLOW→SLO)
            tempo_abbrev = {
                "FAST": "FST", "STANDARD": "STD", "SLOW": "SLO"
            }.get(decision.chosen_tempo or "", decision.chosen_tempo[:3] if decision.chosen_tempo else "")

            line6 = f"bpnt:{bp_abbrev}{bp_conf} tmp:{tempo_abbrev}"
            card.append(f"bpnt:", style="dim")
            card.append(f"{bp_abbrev}", style="cyan")
            card.append(f"{bp_conf}", style="dim")
            card.append(f" tmp:", style="dim")
            card.append(f"{tempo_abbrev}", style="magenta")
        else:
            line6 = ""
            card.append("", style="dim")
        card.append(" " * max(0, CONTENT_WIDTH - len(line6)) + " ")
        card.append("│", style=border_style)
        card.append("\n")
```

Also update the docstring at line ~859 to reflect 7 lines instead of 6:
```python
        """Render an enriched 7-line decision card (24 chars wide).
        ...
        ┌─ D1 16s ──────────────┐
        │ GERM s:1 92%          │  Action, slot, confidence
        │ H:25% ent:0.85        │  Host accuracy, decision entropy
        │ V:+0.45 A:-0.12       │  Value estimate, advantage
        │ -0.68→+0.00 ✓ HIT     │  Expected vs actual + text
        │ bpnt:conv_l(87%) STD  │  Head choices (NEW)
        │ alt: G:12% P:8%       │  Alternatives
        └───────────────────────┘
        """
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_shows_head_choices -v`
Expected: PASS

**Step 5: Run full TamiyoBrain test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): display head choices in decision cards

Shows blueprint and tempo choices for GERMINATE actions:
  bpnt:conv_l(87%) tmp:STD

Per UX specialist: surfaces sub-decisions without dashboard clutter.
Per DRL specialist: helps diagnose credit assignment issues."
```

---

## Task 4: Add Last-Action Environment Tracking

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:742-820`
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Context:** To highlight which environment was last targeted by an action, we need to track `last_action_env_id` in the snapshot.

**Specialist Amendments (UX Review):**
- Add `last_action_timestamp` for hysteresis (5-second decay prevents visual jitter)

**Step 1: Write failing test for last_action_env_id**

```python
# In tests/karn/sanctum/test_aggregator.py

def test_snapshot_tracks_last_action_env_id():
    """Snapshot should track which env received the last action."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import ActionTakenPayload

    agg = SanctumAggregator(n_envs=4, n_slots=2)

    # Simulate an action on env 2
    payload = ActionTakenPayload(
        env_id=2,
        action="GERMINATE",
        slot_id="slot_0",
        blueprint_id="conv_light",
    )
    agg.handle_action_taken(payload)

    snapshot = agg.get_snapshot()
    assert snapshot.last_action_env_id == 2
    assert snapshot.last_action_timestamp is not None  # Added per UX review
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_snapshot_tracks_last_action_env_id -v`
Expected: FAIL (field doesn't exist)

**Step 3: Add last_action_env_id to SanctumSnapshot**

In `src/esper/karn/sanctum/schema.py`, add to `SanctumSnapshot` dataclass:

```python
@dataclass
class SanctumSnapshot:
    """...existing docstring..."""
    # ... existing fields ...
    focused_env_id: int = 0

    # Last action target (for row highlighting in EnvOverview)
    last_action_env_id: int | None = None  # None if no actions yet
    last_action_timestamp: datetime | None = None  # For hysteresis (5s decay)
```

**Step 4: Track last_action_env_id in aggregator**

In `src/esper/karn/sanctum/aggregator.py`:

Add field to aggregator state (around line ~186):
```python
    _focused_env_id: int = 0
    _last_action_env_id: int | None = None  # Track last action target
    _last_action_timestamp: datetime | None = None  # For hysteresis
```

Update `handle_action_taken` method to track the env:
```python
def handle_action_taken(self, payload: ActionTakenPayload) -> None:
    """Handle ACTION_TAKEN telemetry."""
    # ... existing code ...

    # Track last action target for EnvOverview highlighting (with timestamp for hysteresis)
    self._last_action_env_id = payload.env_id
    self._last_action_timestamp = datetime.now(timezone.utc)
```

Include in snapshot (in `get_snapshot` method):
```python
    return SanctumSnapshot(
        # ... existing fields ...
        focused_env_id=self._focused_env_id,
        last_action_env_id=self._last_action_env_id,
        last_action_timestamp=self._last_action_timestamp,
        # ... remaining fields ...
    )
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_snapshot_tracks_last_action_env_id -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): track last-action environment ID

Adds last_action_env_id to SanctumSnapshot, populated from
ACTION_TAKEN events. Enables EnvOverview row highlighting."
```

---

## Task 5: Highlight Last-Action Environment Row

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_overview.py:410-416`
- Test: `tests/karn/sanctum/test_env_overview.py`

**Context:** The EnvOverview table should visually indicate which environment was last targeted by an action, using a pip indicator before the env ID.

**Specialist Amendments (UX Review):**
- Use `▶` instead of `●` for accessibility (shape implies "targeted", works on monochrome)
- Add hysteresis: only show indicator if action was within last 5 seconds

**Step 1: Write failing test for last-action highlighting**

```python
# In tests/karn/sanctum/test_env_overview.py (create if doesn't exist)

def test_env_overview_highlights_last_action_env():
    """EnvOverview should show pip indicator for last-action env."""
    from esper.karn.sanctum.widgets.env_overview import EnvOverview
    from esper.karn.sanctum.schema import SanctumSnapshot, EnvState

    snapshot = SanctumSnapshot()
    snapshot.last_action_env_id = 2
    snapshot.envs = {
        0: EnvState(env_id=0, host_accuracy=70.0, status="healthy"),
        1: EnvState(env_id=1, host_accuracy=72.0, status="healthy"),
        2: EnvState(env_id=2, host_accuracy=75.0, status="healthy"),  # Last action target
    }
    snapshot.slot_ids = ["slot_0"]

    widget = EnvOverview(num_envs=3)

    # Format env ID for the last-action env
    env = snapshot.envs[2]
    result = widget._format_env_id(env, last_action_env_id=2)

    # Should have action indicator pip (▶ per UX accessibility review)
    assert "▶" in result, f"Expected action indicator ▶ in: {result}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py::test_env_overview_highlights_last_action_env -v`
Expected: FAIL (no highlighting)

**Step 3: Modify _format_env_id to accept last_action_env_id**

In `src/esper/karn/sanctum/widgets/env_overview.py`, update `_format_env_id`:

```python
def _format_env_id(
    self,
    env: "EnvState",
    last_action_env_id: int | None = None,
    last_action_timestamp: datetime | None = None,
) -> str:
    """Format env ID with A/B test cohort pip and action target indicator.

    Args:
        env: Environment state.
        last_action_env_id: ID of env that received last action (for highlighting).
        last_action_timestamp: When the last action occurred (for hysteresis).

    Returns:
        Formatted env ID string with indicators.
    """
    # Action target indicator (cyan ▶ prefix) - per UX accessibility review
    # Only show if action was within last 5 seconds (hysteresis prevents jitter)
    action_pip = ""
    if last_action_env_id is not None and env.env_id == last_action_env_id:
        show_indicator = True
        if last_action_timestamp is not None:
            age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
            show_indicator = age < 5.0  # 5-second hysteresis
        if show_indicator:
            action_pip = "[cyan]▶[/cyan]"

    # A/B cohort pip (existing logic)
    if env.reward_mode and env.reward_mode in _AB_STYLES:
        pip, color = _AB_STYLES[env.reward_mode]
        return f"{action_pip}[{color}]{pip}[/{color}]{env.env_id}"
    return f"{action_pip}{env.env_id}"
```

**Step 4: Pass last_action_env_id through to _format_env_id**

In `_add_env_row`, update the call to pass the snapshot's last_action_env_id:

```python
def _add_env_row(self, env: "EnvState", dim: bool = False) -> None:
    """Add a single environment row."""
    # Env ID with indicators (including last-action highlighting with hysteresis)
    last_action_env_id = self._snapshot.last_action_env_id if self._snapshot else None
    last_action_timestamp = self._snapshot.last_action_timestamp if self._snapshot else None
    env_id_cell = self._format_env_id(
        env,
        last_action_env_id=last_action_env_id,
        last_action_timestamp=last_action_timestamp,
    )
    # ... rest of method unchanged ...
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py::test_env_overview_highlights_last_action_env -v`
Expected: PASS

**Step 6: Run full EnvOverview tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_overview.py tests/karn/sanctum/test_env_overview.py
git commit -m "feat(sanctum): highlight last-action environment row

Adds cyan ▶ pip before env ID for the environment that received
the most recent Tamiyo action. Includes 5-second hysteresis to
prevent visual jitter during rapid actions.

Per UX specialist:
- Use ▶ instead of ● for accessibility (shape implies 'targeted')
- Add hysteresis to prevent jitter
- Link policy to env via highlighting, not column duplication"
```

---

## Task 6: Integration Test and Final Verification

**Files:**
- Test: `tests/karn/sanctum/test_app_integration.py`

**Step 1: Run full sanctum test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
Expected: All tests PASS

**Step 2: Run visual smoke test (optional)**

If running in a terminal:
```bash
cd /home/john/esper-lite
PYTHONPATH=src uv run python -m esper.karn.sanctum.app --demo
```

Verify:
- Head heatmap bars are 5 chars wide
- Decision cards show head choices for GERMINATE
- EnvOverview shows ● pip on last-action env row

**Step 3: Final commit with summary**

```bash
git add .
git commit -m "feat(sanctum): complete head decision display enhancement

Summary of changes:
- Layout: 80/20 split (from 70/30) giving more space to left panels
- Head heatmap: 5-char bars for better collapse detection granularity
- Decision cards: Show blueprint/tempo head choices
- EnvOverview: Highlight last-action environment with ● pip

Per DRL specialist recommendations:
- Better entropy visualization catches head collapse earlier
- Head choice display helps diagnose credit assignment issues

Per UX specialist recommendations:
- Progressive disclosure: heatmaps at glance, details in cards
- Link env↔policy via highlighting, not column duplication"
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Widen head heatmap bars (3→5) | Low |
| 2 | Add head choice fields to DecisionSnapshot | Low |
| 3 | Display head choices in decision cards | Medium |
| 4 | Track last-action environment ID | Low |
| 5 | Highlight last-action env row | Low |
| 6 | Integration test | Low |

**Total estimated time:** 30-45 minutes with TDD approach

**Dependencies:**
- Task 3 depends on Task 2 (fields must exist before display)
- Task 5 depends on Task 4 (tracking must exist before highlighting)
- Tasks 1-3 and 4-5 can be done in parallel
