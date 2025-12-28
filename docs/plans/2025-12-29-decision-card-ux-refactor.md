# Decision Card UX Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the TamiyoBrain decision cards for improved readability while preserving all information.

**Architecture:** Expand card width from 45 to 65 chars, add semantic entropy labels, replace cryptic abbreviations with readable labels, use contextual notes for non-GERMINATE actions, add visual separator between "what happened" and "outcome" sections.

**Tech Stack:** Python, Rich/Textual TUI, pytest

---

## Task 1: Add Helper Methods for Entropy Labels and Outcome Badges

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:115-135` (add after constants)
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write failing tests for entropy label helper**

Add to `tests/karn/sanctum/test_tamiyo_brain.py` after the existing imports section (~line 30):

```python
# =============================================================================
# ENTROPY LABEL AND OUTCOME BADGE HELPER TESTS
# =============================================================================


def test_entropy_label_collapsed():
    """Entropy < 0.3 should return [collapsed] in red."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.1)
    assert label == "[collapsed]"
    assert style == "red"


def test_entropy_label_confident():
    """Entropy 0.3-0.7 should return [confident] in yellow."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.5)
    assert label == "[confident]"
    assert style == "yellow"


def test_entropy_label_balanced():
    """Entropy 0.7-1.2 should return [balanced] in green."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.9)
    assert label == "[balanced]"
    assert style == "green"


def test_entropy_label_exploring():
    """Entropy > 1.2 should return [exploring] in cyan."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(1.5)
    assert label == "[exploring]"
    assert style == "cyan"


def test_outcome_badge_hit():
    """Prediction error < 0.1 should return [HIT] in bright_green."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=0.55)
    assert badge == "[HIT]"
    assert style == "bright_green"


def test_outcome_badge_ok():
    """Prediction error 0.1-0.3 should return [~OK] in yellow."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=0.7)
    assert badge == "[~OK]"
    assert style == "yellow"


def test_outcome_badge_miss():
    """Prediction error >= 0.3 should return [MISS] in red."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=1.0)
    assert badge == "[MISS]"
    assert style == "red"


def test_outcome_badge_pending():
    """None reward should return [...] in dim."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=None)
    assert badge == "[...]"
    assert style == "dim"
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_entropy_label_collapsed -v`
Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_entropy_label'`

**Step 3: Implement helper methods**

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py` inside the `TamiyoBrain` class, after the `_get_separator_width` method (~line 257):

```python
    def _entropy_label(self, entropy: float) -> tuple[str, str]:
        """Return (label, style) for entropy value.

        Semantic labels help operators understand policy health at a glance:
        - collapsed: Policy stuck, needs intervention
        - confident: Exploiting known strategy
        - balanced: Healthy exploration/exploitation
        - exploring: High exploration mode
        """
        if entropy < 0.3:
            return "[collapsed]", "red"
        elif entropy < 0.7:
            return "[confident]", "yellow"
        elif entropy < 1.2:
            return "[balanced]", "green"
        else:
            return "[exploring]", "cyan"

    def _outcome_badge(self, expect: float, reward: float | None) -> tuple[str, str]:
        """Return (badge, style) for prediction accuracy.

        Badges provide quick visual feedback on value function quality:
        - [HIT]: Prediction was accurate (error < 0.1)
        - [~OK]: Prediction was acceptable (error < 0.3)
        - [MISS]: Prediction was poor (error >= 0.3)
        - [...]: Reward not yet received
        """
        if reward is None:
            return "[...]", "dim"
        diff = abs(reward - expect)
        if diff < 0.1:
            return "[HIT]", "bright_green"
        elif diff < 0.3:
            return "[~OK]", "yellow"
        else:
            return "[MISS]", "red"
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "entropy_label or outcome_badge" -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add tests/karn/sanctum/test_tamiyo_brain.py src/esper/karn/sanctum/widgets/tamiyo_brain.py
git commit -m "feat(karn): add entropy label and outcome badge helpers for decision cards"
```

---

## Task 2: Add Contextual Note Helper for Non-GERMINATE Actions

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write failing tests**

Add to `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
def test_action_context_note_wait():
    """WAIT action should have contextual note."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="WAIT",
        chosen_slot=None,
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "waiting" in note.lower()


def test_action_context_note_prune():
    """PRUNE action should mention removing underperformer."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="PRUNE",
        chosen_slot="r0c1",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "removing" in note.lower() or "prune" in note.lower()


def test_action_context_note_fossilize():
    """FOSSILIZE action should mention fusing module."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="FOSSILIZE",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "fusing" in note.lower() or "fossiliz" in note.lower()


def test_action_context_note_set_alpha():
    """SET_ALPHA_TARGET action should mention blend adjustment."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="SET_ALPHA_TARGET",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "blend" in note.lower() or "alpha" in note.lower()


def test_action_context_note_germinate_empty():
    """GERMINATE action should return empty string (uses head choices instead)."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert note == ""
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_action_context_note_wait -v`
Expected: FAIL with `AttributeError: 'TamiyoBrain' object has no attribute '_action_context_note'`

**Step 3: Implement contextual note helper**

Add to `src/esper/karn/sanctum/widgets/tamiyo_brain.py` after `_outcome_badge`:

```python
    def _action_context_note(self, decision: "DecisionSnapshot") -> str:
        """Return contextual note for non-GERMINATE actions.

        Replaces the empty head-choices line with meaningful context
        explaining WHY the action was taken.
        """
        action = decision.chosen_action
        slot = decision.chosen_slot or "?"

        if action == "GERMINATE":
            return ""  # GERMINATE uses head choices line instead
        elif action == "WAIT":
            return "(waiting for training progress)"
        elif action == "PRUNE":
            return f"(removing underperformer from {slot})"
        elif action == "FOSSILIZE":
            return f"(fusing trained module in {slot})"
        elif action == "SET_ALPHA_TARGET":
            return "(adjusting blend parameters)"
        else:
            return ""
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "action_context_note" -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tests/karn/sanctum/test_tamiyo_brain.py src/esper/karn/sanctum/widgets/tamiyo_brain.py
git commit -m "feat(karn): add contextual note helper for non-GERMINATE decision cards"
```

---

## Task 3: Update Card Width Constant

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:119` (DECISION_CARD_WIDTH constant)

**Step 1: Write failing test for new width**

Add to `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
def test_decision_card_width_is_65():
    """Decision card should be 65 chars wide for improved readability."""
    assert TamiyoBrain.DECISION_CARD_WIDTH == 65
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_width_is_65 -v`
Expected: FAIL with `assert 45 == 65`

**Step 3: Update the constant**

In `src/esper/karn/sanctum/widgets/tamiyo_brain.py`, change line ~119:

```python
    # Enriched decision card width (Task 5)
    DECISION_CARD_WIDTH = 65  # Widened from 45 for improved readability
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_width_is_65 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(karn): widen decision card from 45 to 65 chars"
```

---

## Task 4: Refactor `_render_enriched_decision` Method

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:853-1058` (the `_render_enriched_decision` method)
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write failing tests for new format**

Add to `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
def test_redesigned_decision_card_structure():
    """Redesigned card should have semantic sections with dashed separator."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=25.0,
        chosen_action="WAIT",
        chosen_slot="r0c0",
        confidence=0.98,
        expected_value=0.80,
        actual_reward=-0.10,
        alternatives=[("GERMINATE", 0.01), ("PRUNE", 0.01)],
        decision_id="test",
        decision_entropy=0.12,
        value_residual=-0.90,
        td_advantage=0.05,
    )

    card = widget._render_enriched_decision(decision, index=0)
    text = card.plain

    # Should be 65 chars wide
    lines = text.split('\n')
    for line in lines:
        if line:  # Skip empty lines
            assert len(line) == 65, f"Line length {len(line)} != 65: '{line}'"

    # Should contain dashed separator
    assert "─ ─ ─" in text, "Expected dashed separator in card"

    # Should use expanded labels instead of abbreviations
    assert "host:" in text, "Expected 'host:' instead of 'H:'"
    assert "entropy:" in text, "Expected 'entropy:' instead of 'ent:'"
    assert "expect:" in text, "Expected 'expect:' instead of 'V:'"
    assert "reward:" in text, "Expected 'reward:' instead of 'r:'"

    # Should NOT contain Greek delta
    assert "δ:" not in text, "Should not contain Greek delta"

    # Should contain entropy label
    assert "[collapsed]" in text or "[confident]" in text or "[balanced]" in text or "[exploring]" in text

    # Should contain outcome badge
    assert "[HIT]" in text or "[~OK]" in text or "[MISS]" in text or "[...]" in text


def test_redesigned_card_contextual_note_for_wait():
    """Non-GERMINATE cards should show contextual note on line 2."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="WAIT",
        chosen_slot=None,
        confidence=0.95,
        expected_value=0.0,
        actual_reward=0.0,
        alternatives=[],
        decision_id="test",
        decision_entropy=0.5,
    )

    card = widget._render_enriched_decision(decision, index=0)
    text = card.plain

    # Should contain contextual note instead of "bp:- - - -"
    assert "waiting" in text.lower(), f"Expected contextual note in card: {text}"
    assert "bp:- - - -" not in text, "Should not have placeholder bp line"


def test_redesigned_card_germinate_shows_blueprint():
    """GERMINATE cards should show blueprint info on line 2."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=0.3,
        alternatives=[("WAIT", 0.06)],
        decision_id="test",
        decision_entropy=1.2,
        value_residual=0.1,
        chosen_blueprint="conv_light",
        chosen_tempo="STANDARD",
        chosen_style="LINEAR_ADD",
        chosen_curve="LINEAR",
        blueprint_confidence=0.87,
    )

    card = widget._render_enriched_decision(decision, index=0)
    text = card.plain

    # Should contain blueprint info
    assert "blueprint:" in text or "conv" in text.lower()
    assert "▸▸" in text  # STANDARD tempo
    assert "╱" in text  # LINEAR curve


def test_redesigned_card_title_format():
    """Title should show #N ACTION on left, age on right."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    widget = TamiyoBrain()
    # Create decision 30 seconds ago
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=30),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=0.3,
        alternatives=[],
        decision_id="test",
        decision_entropy=1.0,
    )

    card = widget._render_enriched_decision(decision, index=0)
    text = card.plain
    first_line = text.split('\n')[0]

    # Should have #1 and action name
    assert "#1" in first_line, f"Expected '#1' in title: {first_line}"
    assert "GERMINATE" in first_line, f"Expected 'GERMINATE' in title: {first_line}"
    # Age should be on right side (30s or close to it)
    assert "30s" in first_line or "31s" in first_line or "29s" in first_line, f"Expected age in title: {first_line}"


def test_redesigned_card_also_line():
    """Alternatives line should use 'also:' prefix."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="WAIT",
        chosen_slot=None,
        confidence=0.90,
        expected_value=0.0,
        actual_reward=0.0,
        alternatives=[("GERMINATE", 0.06), ("PRUNE", 0.04)],
        decision_id="test",
        decision_entropy=0.5,
    )

    card = widget._render_enriched_decision(decision, index=0)
    text = card.plain

    # Should use "also:" instead of "alt:"
    assert "also:" in text, f"Expected 'also:' in card: {text}"
    assert "alt:" not in text, "Should not have 'alt:' abbreviation"
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_redesigned_decision_card_structure -v`
Expected: FAIL (old format doesn't match new expectations)

**Step 3: Implement the refactored method**

Replace the `_render_enriched_decision` method in `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (lines ~853-1058) with:

```python
    def _render_enriched_decision(
        self,
        decision: "DecisionSnapshot",
        index: int,
        total_cards: int = 3,
    ) -> Text:
        """Render a redesigned 8-line decision card (65 chars wide).

        Layout per UX specialist review:
        ┌─ #1 GERMINATE ──────────────────────────────────────── 16s ───┐
        │  slot:1  confidence:92%                                       │
        │  blueprint:conv_light(87%)  STD  sigmoid+add  ╱               │  <- or contextual note
        │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
        │  host:25%  entropy:0.85 [balanced]                      [HIT] │
        │  expect:+0.45  reward:+0.30  TD:+0.22                         │
        │  also: WAIT 6%  PRUNE 2%                                      │
        └───────────────────────────────────────────────────────────────┘

        Age-based border colors:
        - Newest (index 0): cyan border
        - Middle: dim grey border
        - Oldest (index == total-1): yellow border
        """
        from datetime import datetime, timezone

        CARD_WIDTH = self.DECISION_CARD_WIDTH  # 65
        CONTENT_WIDTH = CARD_WIDTH - 4  # "│ " + content + " │" = 61 inner chars

        # Age-based border colors
        if total_cards <= 1:
            border_style = "cyan"
        elif index == 0:
            border_style = "cyan"
        elif index == total_cards - 1:
            border_style = "yellow"
        else:
            border_style = "dim"

        # Calculate age
        now = datetime.now(timezone.utc)
        age = (now - decision.timestamp).total_seconds()
        if age < 60:
            age_str = f"{age:.0f}s"
        else:
            mins = int(age // 60)
            secs = int(age % 60)
            age_str = f"{mins}:{secs:02d}"

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

        card = Text()

        # === TITLE LINE: #N ACTION ... age ===
        # Format: ┌─ #1 GERMINATE ──────────────────────────────────────── 16s ───┐
        title_left = f"#{ index + 1} {decision.chosen_action}"
        title_right = f"{age_str} "
        # Calculate fill: total width - borders (4) - left content - right content - spacing
        fill_width = CARD_WIDTH - 4 - len(title_left) - len(title_right) - 2
        fill = "─" * max(1, fill_width)
        card.append("┌─ ", style=border_style)
        card.append(title_left, style=action_style)
        card.append(f" {fill} ", style=border_style)
        card.append(title_right, style="dim")
        card.append("───┐\n", style=border_style)

        # === LINE 1: slot:N  confidence:XX% ===
        slot_display = decision.chosen_slot[-1] if decision.chosen_slot else "-"
        line1_content = f"slot:{slot_display}  confidence:{decision.confidence:.0%}"
        line1_pad = CONTENT_WIDTH - len(line1_content)
        card.append("│  ", style=border_style)
        card.append("slot:", style="dim")
        card.append(slot_display, style="cyan")
        card.append("  confidence:", style="dim")
        card.append(f"{decision.confidence:.0%}", style="white")
        card.append(" " * max(0, line1_pad) + " │\n", style=border_style)

        # === LINE 2: Head choices (GERMINATE) OR contextual note (others) ===
        card.append("│  ", style=border_style)
        if decision.chosen_action == "GERMINATE" and decision.chosen_blueprint:
            # Blueprint info for GERMINATE
            bp_name = decision.chosen_blueprint[:12] if decision.chosen_blueprint else "?"
            bp_conf = f"({decision.blueprint_confidence:.0%})" if decision.blueprint_confidence else ""

            tempo_map = {"FAST": "▸▸▸", "STANDARD": "▸▸", "SLOW": "▸"}
            tempo_display = tempo_map.get(decision.chosen_tempo or "", "-")
            tempo_style = "magenta" if decision.chosen_tempo in tempo_map else "dim"

            style_map = {
                "LINEAR_ADD": "linear+add",
                "LINEAR_MULTIPLY": "linear*mul",
                "SIGMOID_ADD": "sigmoid+add",
                "GATED_GATE": "gated*gate",
            }
            style_display = style_map.get(decision.chosen_style or "", "-")
            style_style = "yellow" if decision.chosen_style in style_map else "dim"

            curve_glyph = ALPHA_CURVE_GLYPHS.get(decision.chosen_curve or "", "-")
            curve_style = "green" if decision.chosen_curve in ALPHA_CURVE_GLYPHS else "dim"

            card.append("blueprint:", style="dim")
            card.append(bp_name, style="cyan")
            card.append(bp_conf, style="dim")
            card.append(f"  {tempo_display}", style=tempo_style)
            card.append(f"  {style_display}", style=style_style)
            card.append(f"  {curve_glyph}", style=curve_style)

            line2_content = f"blueprint:{bp_name}{bp_conf}  {tempo_display}  {style_display}  {curve_glyph}"
        else:
            # Contextual note for non-GERMINATE
            note = self._action_context_note(decision)
            if not note:
                note = "(no additional context)"
            card.append(note, style="dim italic")
            line2_content = note

        line2_pad = CONTENT_WIDTH - len(line2_content)
        card.append(" " * max(0, line2_pad) + " │\n", style=border_style)

        # === SEPARATOR LINE ===
        separator = "─ " * ((CONTENT_WIDTH) // 2)
        card.append("│ ", style=border_style)
        card.append(separator[:CONTENT_WIDTH], style="dim")
        card.append("│\n", style=border_style)

        # === LINE 3: host:XX%  entropy:X.XX [label]  [BADGE] ===
        entropy_label, entropy_style = self._entropy_label(decision.decision_entropy)
        outcome_badge, outcome_style = self._outcome_badge(decision.expected_value, decision.actual_reward)

        host_part = f"host:{decision.host_accuracy:.0f}%"
        entropy_part = f"entropy:{decision.decision_entropy:.2f}"
        # Calculate spacing to right-align badge
        left_content = f"{host_part}  {entropy_part} {entropy_label}"
        right_content = outcome_badge
        middle_pad = CONTENT_WIDTH - len(left_content) - len(right_content)

        card.append("│  ", style=border_style)
        card.append(host_part, style="cyan")
        card.append("  ", style="dim")
        card.append(entropy_part, style="white")
        card.append(" ", style="dim")
        card.append(entropy_label, style=entropy_style)
        card.append(" " * max(1, middle_pad), style="dim")
        card.append(outcome_badge, style=outcome_style)
        card.append(" │\n", style=border_style)

        # === LINE 4: expect:+X.XX  reward:+X.XX  TD:+X.XX ===
        expect_str = f"expect:{decision.expected_value:+.2f}"
        if decision.actual_reward is not None:
            reward_str = f"reward:{decision.actual_reward:+.2f}"
            reward_style = "green" if decision.actual_reward >= 0 else "red"
        else:
            reward_str = "reward:..."
            reward_style = "dim italic"

        if decision.td_advantage is not None:
            td_str = f"TD:{decision.td_advantage:+.2f}"
        else:
            td_str = "TD:..."

        line4_content = f"{expect_str}  {reward_str}  {td_str}"
        line4_pad = CONTENT_WIDTH - len(line4_content)

        card.append("│  ", style=border_style)
        card.append(expect_str, style="cyan")
        card.append("  ", style="dim")
        card.append(reward_str, style=reward_style)
        card.append("  ", style="dim")
        card.append(td_str, style="bright_cyan")
        card.append(" " * max(0, line4_pad) + " │\n", style=border_style)

        # === LINE 5: also: ACTION X%  ACTION X% ===
        card.append("│  ", style=border_style)
        if decision.alternatives:
            card.append("also: ", style="dim")
            alt_parts = []
            for alt_action, prob in decision.alternatives[:2]:
                alt_style = action_colors.get(alt_action, "dim")
                card.append(f"{alt_action} {prob:.0%}", style=alt_style)
                card.append("  ", style="dim")
                alt_parts.append(f"{alt_action} {prob:.0%}")
            line5_content = "also: " + "  ".join(alt_parts)
        else:
            card.append("also: (none)", style="dim")
            line5_content = "also: (none)"

        line5_pad = CONTENT_WIDTH - len(line5_content)
        card.append(" " * max(0, line5_pad) + " │\n", style=border_style)

        # === BOTTOM BORDER ===
        card.append("└" + "─" * (CARD_WIDTH - 2) + "┘", style=border_style)

        return card
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "redesigned" -v`
Expected: All 5 new tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(karn): redesign decision cards with semantic sections and expanded labels"
```

---

## Task 5: Update Existing Tests to Match New Format

**Files:**
- Modify: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Run all existing decision card tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "decision" -v`
Expected: Several tests FAIL due to format changes

**Step 2: Update `test_enriched_decision_card_format`**

Update the test at ~line 1525 to expect 65-char width and new labels:

```python
@pytest.mark.asyncio
async def test_enriched_decision_card_format():
    """Enriched decision card should be 65 chars wide with 8 lines."""
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
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[("GERMINATE", 0.05), ("FOSSILIZE", 0.03)],
            pinned=False,
            value_residual=-0.04,
            decision_entropy=0.85,
        )

        card = widget._render_enriched_decision(decision, index=0)
        card_plain = card.plain
        lines = card_plain.split('\n')

        # Should have exactly 8 lines (title + 5 content + separator + bottom border)
        assert len(lines) == 8

        # All lines should be exactly 65 chars (widened for readability)
        for i, line in enumerate(lines[:8]):
            if line:  # Skip empty
                assert len(line) == 65, f"Line {i} has length {len(line)}, expected 65: '{line}'"

        # Verify border structure
        assert lines[0].startswith("┌─")
        assert lines[0].endswith("┐")
        assert lines[7].startswith("└")
        assert lines[7].endswith("┘")

        # Should contain new format labels
        card_str = card_plain
        assert "#1" in card_str  # Decision number with #
        assert "WAIT" in card_str  # Action in title
        assert "confidence:" in card_str  # Expanded label
        assert "host:" in card_str  # Expanded label
        assert "entropy:" in card_str  # Expanded label
        assert "expect:" in card_str  # Expanded label (was V:)
        assert "also:" in card_str  # Expanded label (was alt:)
```

**Step 3: Update `test_enriched_decision_card_hit_miss`**

Update to check for badge format instead of icon:

```python
@pytest.mark.asyncio
async def test_enriched_decision_card_hit_miss():
    """Enriched card should show [HIT]/[MISS] badge based on prediction accuracy."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test HIT (diff < 0.1)
        decision_hit = DecisionSnapshot(
            decision_id="test-hit",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.15,  # diff = 0.03 < 0.1
            alternatives=[],
            pinned=False,
            value_residual=0.03,
            decision_entropy=0.85,
        )
        card = widget._render_enriched_decision(decision_hit, index=0)
        assert "[HIT]" in card.plain

        # Test MISS (diff >= 0.3)
        decision_miss = DecisionSnapshot(
            decision_id="test-miss",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.50,  # diff = 0.38 >= 0.3
            alternatives=[],
            pinned=False,
            value_residual=0.38,
            decision_entropy=0.85,
        )
        card = widget._render_enriched_decision(decision_miss, index=0)
        assert "[MISS]" in card.plain
```

**Step 4: Update `test_decision_card_shows_head_choices`**

Update to use new "blueprint:" label:

```python
def test_decision_card_shows_head_choices():
    """Decision cards should display blueprint, tempo arrows, style, and curve for GERMINATE."""
    from datetime import datetime, timezone

    snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
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
        chosen_style="LINEAR_ADD",
        chosen_curve="LINEAR",
        blueprint_confidence=0.87,
        tempo_confidence=0.65,
    )

    widget = TamiyoBrain()
    widget._snapshot = snapshot

    card = widget._render_enriched_decision(decision, index=0, total_cards=1)
    text = str(card)

    # Should contain head choice info for GERMINATE
    assert "blueprint:" in text, f"Expected 'blueprint:' label in card: {text}"
    assert "conv" in text.lower(), f"Expected blueprint name in card: {text}"
    assert "▸▸" in text, f"Expected tempo arrows in card: {text}"
    assert "linear" in text.lower(), f"Expected style in card: {text}"
    assert "╱" in text, f"Expected curve glyph in card: {text}"
```

**Step 5: Run all decision card tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "decision" -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "test(karn): update decision card tests for new 65-char format"
```

---

## Task 6: Run Full Test Suite and Visual Verification

**Files:**
- None (verification only)

**Step 1: Run full tamiyo_brain test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: All tests PASS

**Step 2: Run a quick training to visually verify**

Run: `PYTHONPATH=src timeout 60 uv run python -m esper.scripts.train ppo --preset cifar10 --rounds 5 --sanctum`

Expected: Sanctum TUI launches, decision cards display with new format:
- 65 chars wide
- `#N ACTION` title format
- Dashed separator between sections
- Expanded labels (`host:`, `entropy:`, `expect:`, `reward:`, `also:`)
- Entropy labels (`[collapsed]`, `[balanced]`, etc.)
- Outcome badges (`[HIT]`, `[MISS]`, `[~OK]`)

**Step 3: Commit final verification**

```bash
git commit --allow-empty -m "test(karn): visual verification of redesigned decision cards complete"
```

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Add entropy label and outcome badge helpers | `tamiyo_brain.py`, `test_tamiyo_brain.py` |
| 2 | Add contextual note helper | `tamiyo_brain.py`, `test_tamiyo_brain.py` |
| 3 | Update card width constant (45→65) | `tamiyo_brain.py`, `test_tamiyo_brain.py` |
| 4 | Refactor `_render_enriched_decision` | `tamiyo_brain.py`, `test_tamiyo_brain.py` |
| 5 | Update existing tests for new format | `test_tamiyo_brain.py` |
| 6 | Full test suite + visual verification | None |

**Total estimated time:** 30-45 minutes
