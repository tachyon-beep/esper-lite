# TamiyoBrain Space Utilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Maximize vertical space utilization in TamiyoBrain widget from 54% to ~82%, adding critical missing metrics and enriching existing displays.

**Architecture:** Elevate Episode Return to prime visual position, add entropy sparkline (PPO collapse detection), expand sparklines to 20 chars with trend indicators, enrich decision cards with V(s)/A(s,a)/slot/alternatives, add action sequence timeline with smart pattern detection.

**Tech Stack:** Python 3.12, Rich Text/Table, Textual TUI, dataclasses

**Reviews:** DRL Expert ✓ | UX Specialist ✓

---

## Current State Analysis

### Space Utilization (PROBLEM)
- **Current:** 13/24 rows used = **54% utilization**
- **Target:** 20/24 rows used = **~82% utilization** (per UX review - 92% too dense)
- **Wasted:** ~11 rows in vitals, ~11 rows in decisions column

### Missing Critical Metrics (per DRL review)
1. **Episode Return** - The PRIMARY RL success indicator is not displayed!
2. **Entropy Trend** - Entropy collapse is #1 PPO failure mode; need sparkline, not just current value
3. **Learning Rate** - Essential for LR scheduling visibility
4. **Value Estimate V(s)** - Shows value function calibration
5. **Advantage A(s,a)** - Shows action selection quality

### Current Sparklines (TOO SHORT)
```
──────▃▅▇█▆▄  (10 chars - insufficient for trend visibility)
```

### Target Sparklines (MEANINGFUL)
```
────────────▁▂▃▄▅▆▇█▇▆▅▄▃▂  (20 chars - shows real 20-step history)
```

---

## Target Layout (~82% utilization, Episode Return elevated)

```
┌── TAMIYO ───────────────────────────────────────────────────────────────────────────────────┐
│ [OK] LEARNING  EV:0.72 Clip:0.18 KL:0.008 Adv:+0.12±0.94 GradHP:OK batch:47/100             │
│─────────────────────────────────────────────────────────────────────────────────────────────│
│ Ep.Return  ───────────────▁▂▃▄▅▆▇█▇▆  127.3 ^      LR:3e-4  EntCoef:0.01                    │  ← PRIMARY (row 3)
│ Entropy    ───────────────████████▆▄  6.55  -                                               │  ← NEW (collapse detection)
│─────────────────────────────────────────────────────────────────────────────────────────────│
│ Expl.Var                        │ Clip Frac                         │ DECISIONS             │
│ [████████░░░░░░░░░░░░] -0.008! v│ [░░░░░░░░░░░░░░░░░░░░] 0.000  -   │ ┌─ D1 16s ──────────┐ │
│─────────────────────────────────┼───────────────────────────────────│ │ WAIT s:1 100%     │ │
│ KL Div                          │ Policy Loss                       │ │ H:25% ent:0.85    │ │
│ [░░░░░░░░░░░░░░░░░░░░] 0.008  - │ [───────────────▃▅▇█▆▄] 0.12 v    │ │ V:+0.45 A:-0.12   │ │
│─────────────────────────────────┴───────────────────────────────────│ │ -0.68→+0.00 ✓ HIT │ │
│ Value Loss ───────────────▃▅▇█▆▄ 129.4    Grad ────▃▅▇█▆▄ 0.50  ^   │ │ alt: G:12% P:8%   │ │
│ Layers     12/12 OK                                                 │ └────────────────────┘│
│─────────────────────────────────────────────────────────────────────│                       │
│ Heads: sl[███] bp[███] sy[---] te[---] at[---] as[---] ac[---]      │ ┌─ D2 14s ──────────┐ │
│─────────────────────────────────────────────────────────────────────│ │ PRUN s:2 25%      │ │
│ Actions: [██████████████████████░░░] G=11 A=01 F=00 P=09 W=64       │ │ H:18% ent:0.42    │ │
│ Recent:  W W G W W W F W W W W W                                    │ │ V:+0.32 A:+0.08   │ │
│                                                                     │ │ -0.46→+0.07 ✗ MISS│ │
│                                                                     │ │ alt: W:65% G:10%  │ │
│                                                                     │ └────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key layout changes per reviews:**
- Episode Return elevated to row 3 (prime visual real estate)
- Entropy gets its own sparkline row (collapse detection)
- Decision cards show V(s) and A(s,a)
- Trend indicators without brackets: `^` `-` `~` `v`
- Decision outcomes have text: `✓ HIT` `✗ MISS`
- Card width increased to 24 chars

---

## Task 1: Extend Sparklines to 20 Characters

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (all sparkline calls)
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

### Step 1.1: Write failing test for 20-char sparklines

```python
def test_sparklines_are_twenty_chars():
    """Sparklines should be 20 characters for meaningful trend visibility."""
    from esper.karn.sanctum.schema import make_sparkline
    from collections import deque

    values = deque([float(i) for i in range(20)], maxlen=20)
    sparkline = make_sparkline(values, width=20)

    assert len(sparkline) == 20, f"Expected 20-char sparkline, got {len(sparkline)}"
```

### Step 1.2: Run test to verify behavior

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_sparklines_are_twenty_chars -v`
Expected: PASS (make_sparkline already supports width parameter)

### Step 1.3: Update all sparkline calls to use width=20

In `tamiyo_brain.py`, add constant and update all calls:

```python
# At top of class or module
SPARKLINE_WIDTH = 20

# Update all make_sparkline() calls:
sparkline = make_sparkline(history, width=SPARKLINE_WIDTH)
```

### Step 1.4: Run tests to verify

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: PASS

### Step 1.5: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): extend sparklines to 20 characters for better trend visibility"
```

---

## Task 2: Add Trend Detection with Metric-Specific Thresholds

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add `detect_trend()`)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (display trend indicators)
- Test: `tests/karn/sanctum/test_schema.py`

**DRL Review Incorporated:**
- Window size: 10 samples (not 5) - RL metrics are noisy
- Volatility: variance ratio > 3x (not CV > 50%)
- Metric-specific thresholds

### Step 2.1: Write failing tests for trend detection

```python
def test_detect_trend_improving_loss():
    """Decreasing loss should be detected as 'improving'."""
    from esper.karn.sanctum.schema import detect_trend

    # Clear downward trend in loss
    values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    trend = detect_trend(values, metric_name="policy_loss")
    assert trend == "improving", f"Expected 'improving', got '{trend}'"


def test_detect_trend_warning_loss():
    """Increasing loss should be detected as 'warning'."""
    from esper.karn.sanctum.schema import detect_trend

    values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    trend = detect_trend(values, metric_name="policy_loss")
    assert trend == "warning", f"Expected 'warning', got '{trend}'"


def test_detect_trend_stable():
    """Flat values should be detected as 'stable'."""
    from esper.karn.sanctum.schema import detect_trend

    values = [0.5, 0.51, 0.49, 0.5, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.49]
    trend = detect_trend(values, metric_name="policy_loss")
    assert trend == "stable", f"Expected 'stable', got '{trend}'"


def test_detect_trend_volatile():
    """High recent variance vs historical should be 'volatile'."""
    from esper.karn.sanctum.schema import detect_trend

    # Stable early, then wild swings
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    trend = detect_trend(values, metric_name="policy_loss")
    assert trend == "volatile", f"Expected 'volatile', got '{trend}'"
```

### Step 2.2: Run test to verify it fails

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -k "detect_trend" -v`
Expected: FAIL with "ImportError: cannot import name 'detect_trend'"

### Step 2.3: Implement `detect_trend()` with DRL-appropriate thresholds

Add to `schema.py`:

```python
from collections import deque

# Metric-specific thresholds (per DRL review)
# Higher threshold = more change needed to trigger improving/warning
TREND_THRESHOLDS: dict[str, float] = {
    "episode_return": 0.15,   # 15% - returns vary naturally
    "entropy": 0.08,          # 8% - entropy changes are meaningful
    "policy_loss": 0.20,      # 20% - policy loss is noisy
    "value_loss": 0.20,       # 20% - value loss is noisy
    "kl_divergence": 0.30,    # 30% - KL varies widely
    "clip_fraction": 0.30,    # 30% - clip fraction is variable
    "grad_norm": 0.25,        # 25% - gradients vary batch-to-batch
    "expl_var": 0.15,         # 15% - explained variance
    "default": 0.15,          # 15% - fallback
}


def detect_trend(
    values: list[float] | deque[float],
    metric_name: str = "default",
    metric_type: str = "loss",
) -> str:
    """Detect trend pattern in metric values with RL-appropriate thresholds.

    Uses 10-sample windows (not 5) because RL metrics are inherently noisy.
    Uses variance ratio for volatility (not CV) per DRL review.

    Args:
        values: Recent metric values (oldest first).
        metric_name: Specific metric for threshold lookup.
        metric_type: "loss" (lower=better) or "accuracy" (higher=better).

    Returns:
        Trend label: "improving", "stable", "volatile", "warning"
    """
    if len(values) < 5:
        return "stable"

    values_list = list(values)

    # Use 10-sample windows for RL (per DRL review)
    window_size = min(10, len(values_list) // 2)
    if window_size < 3:
        return "stable"

    recent = values_list[-window_size:]
    older = values_list[:-window_size] if len(values_list) > window_size else values_list[:window_size]

    if not recent or not older:
        return "stable"

    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)

    # Volatility check: variance ratio > 3x (per DRL review, not CV > 50%)
    recent_var = sum((v - recent_mean) ** 2 for v in recent) / len(recent)
    older_var = sum((v - older_mean) ** 2 for v in older) / len(older)

    if older_var > 0 and recent_var / older_var > 3.0:
        return "volatile"

    # Get metric-specific threshold
    threshold_pct = TREND_THRESHOLDS.get(metric_name, TREND_THRESHOLDS["default"])
    change_threshold = threshold_pct * abs(older_mean) if older_mean != 0 else 0.01
    change = recent_mean - older_mean

    if metric_type == "loss":
        # For loss: decreasing is good, increasing is bad
        if change < -change_threshold:
            return "improving"
        elif change > change_threshold:
            return "warning"
    else:
        # For accuracy/return: increasing is good, decreasing is bad
        if change > change_threshold:
            return "improving"
        elif change < -change_threshold:
            return "warning"

    return "stable"


def trend_to_indicator(trend: str) -> tuple[str, str]:
    """Convert trend label to display indicator and style.

    Per UX review: No brackets, just single char with color.

    Returns:
        Tuple of (indicator_string, rich_style)
    """
    indicators = {
        "improving": ("^", "green"),
        "stable": ("-", "dim"),
        "volatile": ("~", "yellow"),
        "warning": ("v", "red"),
    }
    return indicators.get(trend, ("-", "dim"))
```

### Step 2.4: Run test to verify it passes

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -k "detect_trend" -v`
Expected: PASS

### Step 2.5: Add trend indicators to vitals display

In `_render_vitals_column()`, after each sparkline:

```python
from esper.karn.sanctum.schema import detect_trend, trend_to_indicator

# Example for policy loss:
trend = detect_trend(
    list(tamiyo.policy_loss_history),
    metric_name="policy_loss",
    metric_type="loss"
)
indicator, style = trend_to_indicator(trend)
# Append: f"{sparkline} {value:.3f} {indicator}"
```

### Step 2.6: Commit

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add trend detection with metric-specific thresholds per DRL review"
```

---

## Task 3: Add Episode Return and Entropy Rows (Elevated Position)

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add fields to TamiyoState)
- Modify: `src/esper/karn/sanctum/aggregator.py` (populate from telemetry)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (render rows at TOP)
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**UX Review:** Episode Return must be at row 3 (prime visual real estate)
**DRL Review:** Entropy sparkline critical for collapse detection

### Step 3.1: Write failing test for episode return at top

```python
def test_episode_return_elevated_position():
    """Episode Return should appear near the top, not buried at bottom."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from collections import deque

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            episode_return_history=deque([10.0, 20.0, 30.0, 40.0, 50.0], maxlen=20),
            current_episode_return=50.0,
            entropy_history=deque([6.0, 5.9, 5.8, 5.7, 5.6], maxlen=20),
        )
    )
    widget._snapshot = snapshot

    # Render the primary metrics section (should be at top)
    primary = widget._render_primary_metrics()
    primary_str = str(primary)

    assert "Ep.Return" in primary_str or "Episode" in primary_str, \
        "Episode Return should be in primary metrics section"
    assert "Entropy" in primary_str, \
        "Entropy sparkline should be in primary metrics section"
```

### Step 3.2: Run test to verify it fails

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_episode_return_elevated_position -v`
Expected: FAIL

### Step 3.3: Add fields to TamiyoState

In `schema.py`, update `TamiyoState`:

```python
@dataclass
class TamiyoState:
    # ... existing fields ...

    # Episode return tracking (PRIMARY RL METRIC - per DRL review)
    episode_return_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    current_episode_return: float = 0.0

    # Entropy tracking (COLLAPSE DETECTION - per DRL review)
    entropy_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )

    # Training hyperparameters (for display)
    learning_rate: float = 0.0
    entropy_coefficient: float = 0.0
```

### Step 3.4: Add `_render_primary_metrics()` method

This renders Episode Return and Entropy at the TOP of vitals:

```python
def _render_primary_metrics(self) -> Text:
    """Render primary metrics row (Episode Return + Entropy).

    Per UX review: These go at row 3, prime visual real estate.
    Per DRL review: Entropy sparkline critical for collapse detection.
    """
    from esper.karn.sanctum.schema import make_sparkline, detect_trend, trend_to_indicator

    tamiyo = self._snapshot.tamiyo
    result = Text()

    # Episode Return (PRIMARY RL METRIC)
    if tamiyo.episode_return_history:
        sparkline = make_sparkline(tamiyo.episode_return_history, width=SPARKLINE_WIDTH)
        trend = detect_trend(
            list(tamiyo.episode_return_history),
            metric_name="episode_return",
            metric_type="accuracy"  # Higher is better
        )
        indicator, style = trend_to_indicator(trend)

        result.append("Ep.Return  ", style="bold cyan")
        result.append(sparkline, style="cyan")
        result.append(f"  {tamiyo.current_episode_return:>6.1f} ", style="white")
        result.append(indicator, style=style)
        result.append(f"      LR:{tamiyo.learning_rate:.0e}", style="dim")
        result.append(f"  Ent:{tamiyo.entropy_coefficient:.2f}", style="dim")
        result.append("\n")

    # Entropy (COLLAPSE DETECTION)
    if tamiyo.entropy_history:
        sparkline = make_sparkline(tamiyo.entropy_history, width=SPARKLINE_WIDTH)
        trend = detect_trend(
            list(tamiyo.entropy_history),
            metric_name="entropy",
            metric_type="accuracy"  # Stable/high is good, low is collapse
        )
        indicator, style = trend_to_indicator(trend)

        result.append("Entropy    ", style="bold")
        result.append(sparkline, style="magenta")
        result.append(f"  {tamiyo.current_entropy:>6.2f} ", style="white")
        result.append(indicator, style=style)

    return result
```

### Step 3.5: Update aggregator to populate episode return

In `aggregator.py`, handle telemetry events. **Per UX review: avoid hasattr, use type checking:**

```python
from esper.leyline import TelemetryEventType

def _process_event(self, event: "TelemetryEvent") -> None:
    """Process telemetry event and update state."""
    # ... existing handling ...

    # Episode return from EPISODE_COMPLETED events
    if event.event_type == TelemetryEventType.EPISODE_COMPLETED:
        if event.episode_return is not None:
            self._tamiyo_state.episode_return_history.append(event.episode_return)
            self._tamiyo_state.current_episode_return = event.episode_return

    # Entropy from PPO_UPDATE events
    if event.event_type == TelemetryEventType.PPO_UPDATE:
        if event.entropy is not None:
            self._tamiyo_state.entropy_history.append(event.entropy)
            self._tamiyo_state.current_entropy = event.entropy
        if event.learning_rate is not None:
            self._tamiyo_state.learning_rate = event.learning_rate
        if event.entropy_coefficient is not None:
            self._tamiyo_state.entropy_coefficient = event.entropy_coefficient
```

### Step 3.6: Integrate into layout at TOP position

In `_render_vitals_column()`, call `_render_primary_metrics()` first:

```python
def _render_vitals_column(self) -> ...:
    # PRIMARY METRICS AT TOP (per UX review)
    primary = self._render_primary_metrics()
    # ... rest of vitals below ...
```

### Step 3.7: Run test to verify it passes

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_episode_return_elevated_position -v`
Expected: PASS

### Step 3.8: Commit

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/aggregator.py \
        src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add Episode Return and Entropy rows at top per UX/DRL reviews"
```

---

## Task 4: Add Smart Action Pattern Detection

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**DRL Review Corrections:**
- STUCK = all WAIT when dormant slots exist (not just all WAIT)
- THRASHING = germinate→prune cycles (not just action diversity)
- Add ALPHA_OSCILLATION pattern

**UX Review:** Add icons for accessibility: `⚠ STUCK`, `⚡ THRASH`

### Step 4.1: Write failing test for smart pattern detection

```python
def test_stuck_detection_checks_slot_availability():
    """STUCK should only trigger when waiting despite actionable opportunities."""
    from esper.karn.sanctum.widgets.tamiyo_brain import detect_action_patterns
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    # All WAIT, but no dormant slots = NOT stuck (correct behavior)
    decisions_no_dormant = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={"r0": "Grafted", "r1": "Grafted", "r2": "Grafted"},
            host_accuracy=80.0,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(12)
    ]
    slot_states_grafted = {"r0": "Grafted", "r1": "Grafted", "r2": "Grafted"}
    patterns = detect_action_patterns(decisions_no_dormant, slot_states_grafted)
    assert "STUCK" not in patterns, "Should NOT be stuck when all slots grafted"

    # All WAIT with dormant slot = STUCK
    decisions_with_dormant = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={"r0": "Dormant", "r1": "Grafted", "r2": "Grafted"},
            host_accuracy=80.0,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(12)
    ]
    slot_states_dormant = {"r0": "Dormant", "r1": "Grafted", "r2": "Grafted"}
    patterns = detect_action_patterns(decisions_with_dormant, slot_states_dormant)
    assert "STUCK" in patterns, "Should be STUCK when waiting with dormant slot available"


def test_thrashing_detects_germinate_prune_cycles():
    """THRASHING should detect germinate→prune cycles (wasted compute)."""
    from esper.karn.sanctum.widgets.tamiyo_brain import detect_action_patterns
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    # Germinate-Prune-Germinate-Prune cycle = THRASHING
    actions = ["GERMINATE", "PRUNE", "GERMINATE", "PRUNE", "WAIT", "WAIT", "WAIT", "WAIT"]
    decisions = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={},
            host_accuracy=80.0,
            chosen_action=actions[i] if i < len(actions) else "WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(8)
    ]
    patterns = detect_action_patterns(decisions, {})
    assert "THRASH" in patterns, "Should detect germinate-prune thrashing"
```

### Step 4.2: Run test to verify it fails

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "stuck_detection or thrashing" -v`
Expected: FAIL

### Step 4.3: Implement smart pattern detection

```python
def detect_action_patterns(
    decisions: list["DecisionSnapshot"],
    slot_states: dict[str, str],
) -> list[str]:
    """Detect problematic action patterns with DRL-informed logic.

    Per DRL review:
    - STUCK = all WAIT when dormant slots exist (actionable opportunities)
    - THRASH = germinate→prune cycles (wasted compute)
    - ALPHA_OSC = too many alpha changes without completion

    Per UX review: Returns pattern names for icon display.

    Returns:
        List of pattern names: ["STUCK"], ["THRASH"], ["ALPHA_OSC"]
    """
    patterns = []
    if not decisions:
        return patterns

    actions = [d.chosen_action for d in decisions[:12]]

    # STUCK: All WAIT when dormant slots exist (per DRL review)
    if len(actions) >= 8 and all(a == "WAIT" for a in actions[-8:]):
        has_dormant = any("Dormant" in str(s) or "Empty" in str(s) for s in slot_states.values())
        has_training = any("Training" in str(s) for s in slot_states.values())
        # Stuck = waiting when we COULD germinate (dormant exists, nothing training)
        if has_dormant and not has_training:
            patterns.append("STUCK")

    # THRASH: Germinate-Prune cycles (per DRL review)
    germ_prune_cycles = 0
    for i in range(len(actions) - 1):
        if actions[i] == "GERMINATE" and actions[i + 1] == "PRUNE":
            germ_prune_cycles += 1
    if germ_prune_cycles >= 2:
        patterns.append("THRASH")

    # ALPHA_OSC: Too many alpha changes (per DRL review)
    alpha_count = sum(1 for a in actions if a == "SET_ALPHA_TARGET")
    if alpha_count >= 4:
        patterns.append("ALPHA_OSC")

    return patterns
```

### Step 4.4: Implement `_render_action_sequence()` with pattern icons

```python
def _render_action_sequence(self) -> Text:
    """Render action sequence line with pattern detection.

    Format: "Recent:  W W G W W W F W W W W W"
    With pattern warnings: "⚠ STUCK" or "⚡ THRASH"

    Per UX review: Icons for accessibility (not just color).
    """
    decisions = self._snapshot.tamiyo.recent_decisions[:12]
    if not decisions:
        return Text("Recent:  (no actions yet)", style="dim italic")

    # Get current slot states for pattern detection
    slot_states = {}
    if decisions:
        slot_states = decisions[0].slot_states

    # Detect patterns
    patterns = detect_action_patterns(decisions, slot_states)

    # Action to single-char abbreviation with colors
    action_map = {
        "GERMINATE": ("G", "green"),
        "WAIT": ("W", "dim"),
        "FOSSILIZE": ("F", "blue"),
        "PRUNE": ("P", "red"),
        "SET_ALPHA_TARGET": ("A", "cyan"),
        "ADVANCE": ("→", "cyan"),
    }

    # Build sequence (oldest first for left-to-right reading)
    sequence = [action_map.get(d.chosen_action, ("?", "white")) for d in decisions]
    sequence.reverse()

    result = Text()
    result.append("Recent:  ", style="dim")

    # Highlight based on pattern
    is_stuck = "STUCK" in patterns
    is_thrash = "THRASH" in patterns

    for char, color in sequence:
        if is_stuck:
            result.append(char + " ", style="yellow")
        elif is_thrash:
            result.append(char + " ", style="red")
        else:
            result.append(char + " ", style=color)

    # Pattern warnings with icons (per UX review: icon + text)
    if is_stuck:
        result.append("  ⚠ STUCK", style="yellow bold")
    if is_thrash:
        result.append("  ⚡ THRASH", style="red bold")
    if "ALPHA_OSC" in patterns:
        result.append("  ↔ ALPHA", style="cyan bold")

    return result
```

### Step 4.5: Run test to verify it passes

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "stuck_detection or thrashing" -v`
Expected: PASS

### Step 4.6: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add smart action pattern detection per DRL review

- STUCK: checks slot availability, not just all-WAIT
- THRASH: detects germinate→prune cycles
- ALPHA_OSC: detects excessive alpha target changes
- Icons for accessibility: ⚠ STUCK, ⚡ THRASH, ↔ ALPHA"
```

---

## Task 5: Enrich Decision Cards with V(s), A(s,a), Outcome Text

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add fields to DecisionSnapshot)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**DRL Review:** Add V(s) and A(s,a) - shows value function calibration
**UX Review:** Card width 24 chars, add outcome text (✓ HIT, ✗ MISS)

### Step 5.1: Write failing test for enriched cards

```python
def test_decision_card_shows_value_and_advantage():
    """Decision cards should show V(s) and A(s,a) per DRL review."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={"r0": "Training 12%", "r1": "Blending 45%"},
        host_accuracy=87.0,
        chosen_action="WAIT",
        chosen_slot="r1",
        confidence=0.92,
        expected_value=0.12,
        actual_reward=0.08,
        alternatives=[("GERMINATE", 0.04), ("FOSSILIZE", 0.02)],
        decision_id="test-1",
        value_estimate=0.45,  # V(s)
        advantage=0.12,       # A(s,a)
    )

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            recent_decisions=[decision],
            current_entropy=6.55,
        )
    )
    widget._snapshot = snapshot

    card = widget._render_enriched_decision(decision, index=0)
    card_str = str(card)

    # Should show V(s) and A(s,a)
    assert "V:" in card_str, f"Card should show value estimate V(s). Got: {card_str}"
    assert "A:" in card_str, f"Card should show advantage A(s,a). Got: {card_str}"

    # Should show outcome text (per UX review)
    assert "HIT" in card_str or "MISS" in card_str, \
        f"Card should show outcome text. Got: {card_str}"
```

### Step 5.2: Run test to verify it fails

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_shows_value_and_advantage -v`
Expected: FAIL

### Step 5.3: Add fields to DecisionSnapshot

```python
@dataclass
class DecisionSnapshot:
    # ... existing fields ...

    # Value function outputs (per DRL review)
    value_estimate: float = 0.0   # V(s) - state value estimate
    advantage: float = 0.0        # A(s,a) - advantage for chosen action

    # Decision-specific entropy (per DRL review - more useful than policy entropy)
    decision_entropy: float = 0.0  # -sum(p*log(p)) for this action distribution
```

### Step 5.4: Implement `_render_enriched_decision()`

```python
# Card width per UX review
DECISION_CARD_WIDTH = 24

def _render_enriched_decision(self, decision: "DecisionSnapshot", index: int) -> Text:
    """Render an enriched 6-line decision card (24 chars wide).

    Format per DRL + UX reviews:
    ┌─ D1 16s ──────────────┐
    │ WAIT s:1 100%         │  Action, slot, confidence
    │ H:25% ent:0.85        │  Host accuracy, decision entropy
    │ V:+0.45 A:-0.12       │  Value estimate, advantage (NEW)
    │ -0.68→+0.00 ✓ HIT     │  Expected vs actual + text (NEW)
    │ alt: G:12% P:8%       │  Alternatives
    └───────────────────────┘
    """
    from datetime import datetime, timezone

    CONTENT_WIDTH = DECISION_CARD_WIDTH - 4  # "│ " + content + " │"

    now = datetime.now(timezone.utc)
    age = (now - decision.timestamp).total_seconds()
    age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"

    action_colors = {
        "GERMINATE": "green", "WAIT": "dim", "FOSSILIZE": "blue",
        "PRUNE": "red", "SET_ALPHA_TARGET": "cyan", "ADVANCE": "cyan",
    }
    action_style = action_colors.get(decision.chosen_action, "white")

    card = Text()

    # Title: ┌─ D1 16s ──────────────┐
    title = f"D{index+1} {age_str}"
    fill = DECISION_CARD_WIDTH - 4 - len(title)
    card.append(f"┌─ {title}{'─' * fill}┐\n", style="dim")

    # Line 1: ACTION s:N CONF%
    action_abbrev = decision.chosen_action[:4].upper()
    slot_num = decision.chosen_slot[-1] if decision.chosen_slot else "-"
    line1 = f"{action_abbrev} s:{slot_num} {decision.confidence:.0%}"
    card.append("│ ")
    card.append(action_abbrev, style=action_style)
    card.append(f" s:{slot_num}", style="cyan")
    card.append(f" {decision.confidence:.0%}", style="dim")
    card.append(" " * max(0, CONTENT_WIDTH - len(line1)) + " │\n")

    # Line 2: H:XX% ent:X.XX (decision entropy, per DRL review)
    line2 = f"H:{decision.host_accuracy:.0f}% ent:{decision.decision_entropy:.2f}"
    card.append("│ ")
    card.append(f"H:{decision.host_accuracy:.0f}%", style="cyan")
    card.append(f" ent:{decision.decision_entropy:.2f}", style="dim")
    card.append(" " * max(0, CONTENT_WIDTH - len(line2)) + " │\n")

    # Line 3: V:+X.XX A:+X.XX (per DRL review)
    line3 = f"V:{decision.value_estimate:+.2f} A:{decision.advantage:+.2f}"
    card.append("│ ")
    card.append(f"V:{decision.value_estimate:+.2f}", style="cyan")
    card.append(f" A:{decision.advantage:+.2f}", style="magenta")
    card.append(" " * max(0, CONTENT_WIDTH - len(line3)) + " │\n")

    # Line 4: Expected → Actual ✓ HIT / ✗ MISS (per UX review)
    card.append("│ ")
    card.append(f"{decision.expected_value:+.2f}", style="dim")
    card.append("→", style="dim")
    if decision.actual_reward is not None:
        diff = abs(decision.actual_reward - decision.expected_value)
        is_hit = diff < 0.1
        style = "green" if is_hit else "red"
        icon = "✓" if is_hit else "✗"
        text = "HIT" if is_hit else "MISS"
        card.append(f"{decision.actual_reward:+.2f}", style=style)
        card.append(f" {icon} {text}", style=style)
        line4 = f"{decision.expected_value:+.2f}→{decision.actual_reward:+.2f} {icon} {text}"
    else:
        card.append("...", style="dim italic")
        line4 = f"{decision.expected_value:+.2f}→..."
    card.append(" " * max(0, CONTENT_WIDTH - len(line4)) + " │\n")

    # Line 5: alt: G:12% P:8%
    card.append("│ ")
    if decision.alternatives:
        alt_strs = [f"{a[0]}:{p:.0%}" for a, p in decision.alternatives[:2]]
        line5 = "alt: " + " ".join(alt_strs)
        card.append("alt: ", style="dim")
        for i, (alt_action, prob) in enumerate(decision.alternatives[:2]):
            if i > 0:
                card.append(" ", style="dim")
            alt_style = action_colors.get(alt_action, "dim")
            card.append(f"{alt_action[0]}:{prob:.0%}", style=alt_style)
    else:
        line5 = "alt: -"
        card.append("alt: -", style="dim")
    card.append(" " * max(0, CONTENT_WIDTH - len(line5)) + " │\n")

    # Bottom border
    card.append("└" + "─" * (DECISION_CARD_WIDTH - 2) + "┘", style="dim")

    return card
```

### Step 5.5: Run test to verify it passes

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_card_shows_value_and_advantage -v`
Expected: PASS

### Step 5.6: Commit

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/widgets/tamiyo_brain.py \
        tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): enrich decision cards with V(s), A(s,a), outcome text

Per DRL review: value estimate and advantage for diagnostics
Per UX review: 24-char width, ✓ HIT / ✗ MISS text for accessibility"
```

---

## Task 6: Integration and Full Suite Verification

**Files:**
- All modified files from Tasks 1-5
- Test: Full test suite

### Step 6.1: Run full Sanctum test suite

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests pass

### Step 6.2: Run ruff linting

```bash
uv run ruff check src/esper/karn/sanctum/
```

Expected: No errors

### Step 6.3: Visual verification

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 10 --sanctum
```

Verify:
- Episode Return at row 3 (prime position)
- Entropy sparkline below Episode Return
- All sparklines are 20 characters wide
- Trend indicators without brackets: `^` `-` `~` `v`
- Action sequence with pattern icons: `⚠ STUCK`, `⚡ THRASH`
- Decision cards show V(s), A(s,a)
- Decision outcomes show `✓ HIT` or `✗ MISS`
- Space utilization ~82% (not overwhelming)

### Step 6.4: Final commit

```bash
git add -A
git commit -m "feat(sanctum): complete TamiyoBrain space utilization enhancement

Space utilization improved from 54% to ~82%:
- Elevated Episode Return to row 3 (primary RL metric)
- Added entropy sparkline for collapse detection
- Extended sparklines to 20 chars with metric-specific trend thresholds
- Smart pattern detection: STUCK (slot-aware), THRASH (cycle detection)
- Enriched decision cards with V(s), A(s,a), outcome text
- Accessibility: icons + text for all warnings

DRL review: APPROVED
UX review: APPROVED"
```

---

## Verification Checklist

### DRL Requirements
- [ ] Trend detection uses 10-sample window (not 5)
- [ ] Metric-specific thresholds in TREND_THRESHOLDS dict
- [ ] Volatility uses variance ratio > 3x (not CV > 50%)
- [ ] STUCK detection checks slot availability
- [ ] THRASH detection catches germinate→prune cycles
- [ ] Entropy sparkline added for collapse detection
- [ ] Decision cards show V(s) and A(s,a)
- [ ] Decision entropy shown (not just policy entropy)

### UX Requirements
- [ ] Episode Return at row 3 (prime position)
- [ ] Target ~82% utilization (not 92%)
- [ ] Trend indicators without brackets: `^` `-` `~` `v`
- [ ] Pattern icons: `⚠ STUCK`, `⚡ THRASH`, `↔ ALPHA`
- [ ] Decision outcomes: `✓ HIT`, `✗ MISS`
- [ ] Card width 24 chars
- [ ] No hasattr usage (type checking instead)

### General
- [ ] Full test suite passes
- [ ] Ruff linting passes
- [ ] Visual verification confirms layout
