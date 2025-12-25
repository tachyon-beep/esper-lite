# Expanded TamiyoBrain Widget Implementation Plan (v4)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform TamiyoBrain from a compact diagnostic widget (~50×17) into a comprehensive PPO command center (96×24) showing all P0/P1/P2 metrics with sparklines and per-head entropy visualization. Support A/B testing with color-coded policies and multi-aggregator side-by-side comparison.

**Architecture:** Six-phase incremental delivery: (1) Threshold corrections + core restructure, (2) Status banner + gauge grid, (3) Secondary metrics with sparklines, (4) Per-head entropy heatmap, (5) A/B testing color-coded Tamiyos, (6) Multi-aggregator TUI infrastructure. Each phase is independently deployable.

**Tech Stack:** Textual (TUI), Rich (rendering), Python dataclasses (schema), deque (history tracking)

---

## Pre-Implementation Addendum (2025-12-24)

### Test Fixes Completed

Before implementing this plan, 10 pre-existing test failures were fixed:

| Test File | Issue | Resolution |
|-----------|-------|------------|
| `test_event_log.py` (6 tests) | EventLog widget refactored (`_max_events` → `_max_lines`, removed helper methods) | Tests updated to match new append-only architecture |
| `test_app_integration.py` (3 tests) | Outdated `asyncio.run()` pattern, timing issues | Converted to `@pytest.mark.asyncio` async methods |
| `test_backend.py` (1 test) | Event message format changed (action now in metadata) | Test updated to check `metadata.get("action")` |

**All 216 sanctum tests now pass.**

### Task 1.1 Already Implemented

The TUIThresholds corrections specified in Task 1.1 are **already present** in `src/esper/karn/constants.py:96-151`:
- `EXPLAINED_VAR_WARNING: 0.3` ✓
- `EXPLAINED_VAR_CRITICAL: 0.0` ✓
- `KL_WARNING: 0.015` ✓
- `KL_CRITICAL: 0.03` ✓
- `ADVANTAGE_STD_WARNING/CRITICAL/LOW_WARNING/COLLAPSED` ✓

**→ SKIP Task 1.1** — proceed directly to Task 1.2 (schema fields).

### Schema Fields Needed (Task 1.2)

The following fields are missing from `TamiyoState` and must be added:
- `kl_divergence_history: deque[float]`
- `clip_fraction_history: deque[float]`
- `group_id: str | None` (for A/B testing)

---

## Review Feedback Incorporated (v2 + v3 Changes)

This revision addresses feedback from four specialist reviewers across two review rounds.

### Round 1 (v2) - DRL Expert Corrections
1. **Fixed EV thresholds:** WARNING=0.3, CRITICAL=0.0 (was 0.0/-0.5)
2. **Fixed per-head max entropy values:** Now computed dynamically from action space enums
3. **Added KL thresholds:** KL_WARNING=0.015, KL_CRITICAL=0.03
4. **Fixed advantage std logic:** Added collapsed (0.1) and critical (3.0) thresholds
5. **Status banner now includes:** Adv:± summary, GradHP: summary, episode returns

### Round 1 (v2) - UX Specialist Corrections
1. **Status banner complete:** Now includes Adv:±, GradHP:, batch:N/M with denominator
2. **Separator width:** Now 94 chars (96 - 2 for padding), not hardcoded 48
3. **CSS uses theme variables:** `$success`, `$warning`, `$error` instead of literal colors
4. **80-char fallback:** Added compact mode detection for narrow terminals
5. **Heatmap alignment:** Fixed-width formatting for column alignment

### Round 1 (v2) - Risk Assessor Recommendations
1. **Split Task 1.3:** Decision tree logic now separate from banner rendering
2. **Added checkpoints:** After Task 1.4 (decision tree) and Task 2.1 (sparklines)
3. **Missing telemetry visual distinction:** Gray bars + "awaiting data" for unpopulated heads
4. **Documented rollback:** Each phase has minimum viable deliverable

### Round 2 (v3) - DRL Expert Approval + Minor Additions
1. **All DRL values confirmed correct** ✓
2. **Added grad norm critical check** to decision tree (Task 2.1) - triggers CRITICAL when grad_norm > GRAD_NORM_CRITICAL

### Round 2 (v3) - UX Specialist Corrections
1. **Changed Group C color:** Red → Magenta (red conflicts with error semantics)
2. **Added compact mode detection task** (Task 2.7) - detects 80-char terminals, adjusts layout
3. **CSS specificity:** Moved status-* classes after group-* to ensure status colors override
4. **Added edge case tests** to sparkline renderer (Task 3.1)

### Round 2 (v3) - Risk Assessor Recommendations
1. **Added checkpoint after Task 2.5** - verify gauge grid works before wiring
2. **Defer Phase 5 until Phase 6 exists** - A/B testing needs multi-aggregator infrastructure
3. **Promoted Phase 6** to full workstream - multi-aggregator TUI is prerequisite for Phase 5
4. **Added warning header** to Phase 4 about telemetry gap

### Round 3 (v4) - Code Reviewer Critical Issues
1. **CSS specificity ordering FIXED:** Group-* classes now defined BEFORE status-* classes in CSS cascade, ensuring status colors (red=FAILING) override group colors (green=A) — see Task 5.3 CSS section
2. **Implementation order clarified:** Phase 6 (Multi-Aggregator) must be implemented BEFORE Phase 5 (Color Coding). Both phases have explicit order warnings.
3. **Task 2.7 ADDED:** Compact mode detection for 80-char terminals now has full specification
4. **Task 6.3 checkpoint ADDED:** Verify registry wiring before multi-widget layout

### Round 3 (v4) - DRL Expert Corrections
1. **Task 6.7 field names FIXED:** Changed `tamiyo.accuracy` → `snapshot.aggregate_mean_accuracy` (field doesn't exist on TamiyoState)
2. **Leader determination FIXED:** Now reward-first (primary RL objective), accuracy as tiebreaker in Task 6.6
3. **Null safety ADDED:** `event.data or {}` in Task 6.1 process_event to handle None

### Round 3 (v4) - UX Specialist Corrections
1. **CSS specificity bug FIXED:** (same as Code Reviewer #1) — see Task 5.3 CSS section
2. **Task 2.7 missing FIXED:** (same as Code Reviewer #3)
3. **3-widget edge case ADDED:** Compact mode for 3+ policies (min-width: 80) in Task 6.4

### Round 3 (v4) - Risk Assessor Recommendations
1. **Checkpoint after Task 6.3 ADDED:** Verify AggregatorRegistry integration before layout complexity
2. **Implementation order clarified:** Phase 6 before Phase 5 (infrastructure before cosmetics)
3. **Scope freeze documented:** No Phase 7 without separate design review

---

## Rollback Strategy

| Phase | If Problems Occur | Minimum Viable Deliverable |
|-------|-------------------|---------------------------|
| 1 | Revert threshold changes, keep original constants | Updated constants only |
| 2 | Keep decision tree, disable banner rendering | Decision tree logic for status |
| 3 | Remove sparklines, show static values only | Diagnostic matrix without trends |
| 4 | Hide heatmap section entirely | Phase 3 deliverable |
| 5 | Keep single-aggregator mode (revert `SanctumApp.__init__` to use `SanctumAggregator` directly) | Phase 4 deliverable |
| 6 | Disable A/B color coding, show default borders | Phase 5 deliverable (multi-TamiyoBrain without group colors) |

**Phase Dependency Note (v4 UPDATED):** Phase 5 (multi-aggregator TUI) MUST be implemented before Phase 6 (A/B color coding). Color-coding a single widget is meaningless without side-by-side comparison. This was identified as a CRITICAL ordering issue in round 3 review.

**Scope Freeze:** No Phase 7 without separate design review. If Phase 6 balloons beyond 5 tasks, abort and reassess.

---

## Phase 1: Threshold Corrections + Schema Foundation

**Objective:** Fix DRL-incorrect thresholds and prepare schema infrastructure.

---

### Task 1.1: Update TUIThresholds with DRL-Correct Values

**Files:**
- Modify: `src/esper/karn/constants.py:96-140`
- Test: `tests/karn/test_constants.py` (create if needed)

**Step 1: Write the failing test**

```python
# tests/karn/test_constants.py

def test_explained_variance_thresholds_drl_correct():
    """EV thresholds should follow DRL best practices.

    Theory: EV=0 means value function explains nothing (useless baseline).
    EV<0 means value function increases variance (actively harmful).
    WARNING at 0.3, CRITICAL at 0.0 per DRL expert review.
    """
    from esper.karn.constants import TUIThresholds

    # Warning: value function weak (not useless, but not helping much)
    assert TUIThresholds.EXPLAINED_VAR_WARNING == 0.3

    # Critical: value function useless or harmful
    assert TUIThresholds.EXPLAINED_VAR_CRITICAL == 0.0


def test_kl_thresholds_exist():
    """KL divergence should have both warning and critical thresholds."""
    from esper.karn.constants import TUIThresholds

    assert hasattr(TUIThresholds, 'KL_WARNING')
    assert hasattr(TUIThresholds, 'KL_CRITICAL')
    assert TUIThresholds.KL_WARNING == 0.015
    assert TUIThresholds.KL_CRITICAL == 0.03


def test_advantage_thresholds_exist():
    """Advantage std should have tiered thresholds."""
    from esper.karn.constants import TUIThresholds

    # Normal range: ~1.0 (normalized advantages)
    # Warning: too high (>2.0) or too low (<0.5)
    # Critical: extremely high (>3.0) or collapsed (<0.1)
    assert TUIThresholds.ADVANTAGE_STD_WARNING == 2.0
    assert TUIThresholds.ADVANTAGE_STD_CRITICAL == 3.0
    assert TUIThresholds.ADVANTAGE_STD_LOW_WARNING == 0.5
    assert TUIThresholds.ADVANTAGE_STD_COLLAPSED == 0.1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_constants.py -v`
Expected: FAIL (current thresholds are wrong)

**Step 3: Write minimal implementation**

Update `TUIThresholds` in `src/esper/karn/constants.py`:

```python
class TUIThresholds:
    """Thresholds for TUI color-coded health display.

    These control green/yellow/red status indicators.
    Entropy thresholds align with leyline for consistency.

    Explained Variance thresholds follow DRL best practices:
    - EV=1.0: Perfect value prediction
    - EV=0.0: Value function explains nothing (useless)
    - EV<0.0: Value function increases variance (harmful)
    """

    # Entropy (healthy starts near ln(4) ≈ 1.39 for 4 actions)
    ENTROPY_MAX: float = 1.39  # ln(4) for 4 actions
    ENTROPY_WARNING: float = DEFAULT_ENTROPY_WARNING_THRESHOLD
    ENTROPY_CRITICAL: float = DEFAULT_ENTROPY_COLLAPSE_THRESHOLD

    # Clip fraction (target 0.1-0.2)
    CLIP_WARNING: float = 0.25
    CLIP_CRITICAL: float = 0.3

    # Explained variance (value learning quality) - DRL CORRECTED
    # EV=0 means value function provides no advantage over REINFORCE
    EXPLAINED_VAR_WARNING: float = 0.3   # Value function weak but learning
    EXPLAINED_VAR_CRITICAL: float = 0.0   # Value function useless or harmful

    # Gradient norm
    GRAD_NORM_WARNING: float = 5.0
    GRAD_NORM_CRITICAL: float = 10.0

    # KL divergence (policy change magnitude) - ADDED per DRL review
    KL_WARNING: float = 0.015   # Mild policy drift
    KL_CRITICAL: float = 0.03   # Excessive policy change

    # Advantage normalization thresholds - ADDED per DRL review
    # Healthy advantage std is ~1.0 after normalization
    ADVANTAGE_STD_WARNING: float = 2.0      # High variance
    ADVANTAGE_STD_CRITICAL: float = 3.0     # Extreme variance
    ADVANTAGE_STD_LOW_WARNING: float = 0.5  # Too little variance
    ADVANTAGE_STD_COLLAPSED: float = 0.1    # Advantage normalization broken

    # Action distribution (WAIT dominance is suspicious)
    WAIT_DOMINANCE_WARNING: float = 0.7  # > 70% WAIT

    # Ratio statistics thresholds (PPO policy ratio should stay near 1.0)
    RATIO_MAX_CRITICAL: float = 2.0
    RATIO_MAX_WARNING: float = 1.5
    RATIO_MIN_CRITICAL: float = 0.3
    RATIO_MIN_WARNING: float = 0.5
    RATIO_STD_WARNING: float = 0.5

    # Gradient health percentage thresholds
    GRAD_HEALTH_WARNING: float = 0.8
    GRAD_HEALTH_CRITICAL: float = 0.5
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_constants.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/constants.py tests/karn/test_constants.py
git commit -m "fix(karn): correct TUIThresholds per DRL expert review

- EXPLAINED_VAR_WARNING: 0.0 -> 0.3 (weak but learning)
- EXPLAINED_VAR_CRITICAL: -0.5 -> 0.0 (useless/harmful)
- Add KL_WARNING=0.015, KL_CRITICAL=0.03
- Add ADVANTAGE_STD_* thresholds for normalization health"
```

---

### Task 1.2: Add History Deque Fields to TamiyoState

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
    assert isinstance(state.kl_divergence_history, deque)  # Added for sparklines
    assert isinstance(state.clip_fraction_history, deque)  # Added for sparklines

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
    kl_divergence_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    clip_fraction_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))

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

- Add 7 history deques (maxlen=10) for sparkline trends
- Add 6 new per-head entropy fields (style, tempo, alpha_*, op)
- Prepares schema for expanded TamiyoBrain widget"
```

---

### Task 1.3: Update Aggregator to Populate History

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
                "kl_divergence": 0.01 * (i + 1),
                "clip_fraction": 0.1 + (0.02 * i),
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
    assert len(tamiyo.kl_divergence_history) == 3
    assert len(tamiyo.clip_fraction_history) == 3

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

        kl_divergence = data.get("kl_divergence", 0.0)
        self._tamiyo.kl_divergence = kl_divergence
        self._tamiyo.kl_divergence_history.append(kl_divergence)

        clip_fraction = data.get("clip_fraction", 0.0)
        self._tamiyo.clip_fraction = clip_fraction
        self._tamiyo.clip_fraction_history.append(clip_fraction)

        # ... rest of existing code unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_populates_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): populate TamiyoState history deques in aggregator

PPO_UPDATE_COMPLETED now appends to 7 history deques for sparklines"
```

---

## Phase 1 Checkpoint

Run: `PYTHONPATH=src uv run pytest tests/karn/ -v`

Expected: All tests pass. Foundation complete:
- Corrected TUIThresholds with DRL-accurate values
- Schema has history deques and per-head entropy fields
- Aggregator populates history on each PPO update

---

## Phase 2: Status Banner + Decision Tree + Gauges

**Objective:** Status banner with complete decision tree, then 4-gauge grid.

---

### Task 2.1: Implement Decision Tree Logic (Separate from Rendering)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_tamiyo_brain.py (add to existing file)

@pytest.mark.asyncio
async def test_decision_tree_learning():
    """Decision tree should return LEARNING when all metrics healthy."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,  # > 0.3 warning threshold
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,  # Normal range
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "ok"
        assert label == "LEARNING"


@pytest.mark.asyncio
async def test_decision_tree_ev_warning():
    """Decision tree should return CAUTION when EV between 0 and 0.3."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.15,  # Between 0.0 and 0.3 = warning
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "warning"
        assert label == "CAUTION"


@pytest.mark.asyncio
async def test_decision_tree_ev_critical():
    """Decision tree should return FAILING when EV <= 0."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=-0.1,  # < 0.0 = critical
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_entropy_collapsed():
    """Decision tree should return FAILING when entropy collapsed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=0.05,  # < 0.1 = collapsed
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_kl_critical():
    """Decision tree should return FAILING when KL > 0.03."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.05,  # > 0.03 = critical
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_advantage_collapsed():
    """Decision tree should return FAILING when advantage std collapsed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=0.05,  # < 0.1 = collapsed
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_learning -v`
Expected: FAIL with "AttributeError: 'TamiyoBrain' object has no attribute '_get_overall_status'"

**Step 3: Write minimal implementation**

Add to `TamiyoBrain` class in `tamiyo_brain.py`:

```python
    def _get_overall_status(self) -> tuple[str, str, str]:
        """Get overall learning status using DRL decision tree.

        Priority order (per DRL expert review):
        1. Entropy collapse (policy dead)
        2. EV <= 0 (value harmful)
        3. Advantage std collapsed (normalization broken)
        4. KL > critical (excessive policy change)
        5. Clip > critical (too aggressive)
        6. EV < warning (value weak)
        7. KL > warning (mild drift)
        8. Clip > warning
        9. Entropy low
        10. Advantage abnormal

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

        # === CRITICAL CHECKS (immediate FAILING) ===

        # 1. Entropy collapse (policy is deterministic/dead)
        if tamiyo.entropy < 0.1:
            return "critical", "FAILING", "red bold"

        # 2. EV <= 0 (value function useless or harmful)
        if tamiyo.explained_variance <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 3. Advantage std collapsed (normalization broken)
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical", "FAILING", "red bold"

        # 4. Advantage std exploded
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 5. KL > critical (excessive policy change)
        if tamiyo.kl_divergence > TUIThresholds.KL_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 6. Clip > critical (updates too aggressive)
        if tamiyo.clip_fraction > TUIThresholds.CLIP_CRITICAL:
            return "critical", "FAILING", "red bold"

        # === WARNING CHECKS (CAUTION) ===

        # 7. EV < warning (value function weak but learning)
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning", "CAUTION", "yellow"

        # 8. Entropy low (policy converging quickly)
        if tamiyo.entropy < 0.3:
            return "warning", "CAUTION", "yellow"

        # 9. KL > warning (mild policy drift)
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            return "warning", "CAUTION", "yellow"

        # 10. Clip > warning
        if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            return "warning", "CAUTION", "yellow"

        # 11. Advantage std abnormal
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning", "CAUTION", "yellow"
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            return "warning", "CAUTION", "yellow"

        return "ok", "LEARNING", "green"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_learning tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_ev_warning tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_ev_critical tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_entropy_collapsed tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_kl_critical tests/karn/sanctum/test_tamiyo_brain.py::test_decision_tree_advantage_collapsed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement DRL-correct decision tree for PPO status

Priority order: entropy collapse > EV critical > advantage collapsed >
KL critical > clip critical > EV warning > entropy low > KL warning >
clip warning > advantage abnormal

Uses corrected thresholds from TUIThresholds"
```

---

### ⏸️ Phase 2 Checkpoint: Verify Decision Tree

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "decision_tree" -v`

This is a critical checkpoint. The decision tree is the foundation for all status displays.

---

### Task 2.2: Implement Status Banner with Complete Metrics

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_status_banner_includes_all_metrics():
    """Status banner should include EV, Clip, KL, Adv, GradHP, batch."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.65,
            clip_fraction=0.18,
            kl_divergence=0.008,
            advantage_mean=0.12,
            advantage_std=0.94,
            dead_layers=0,
            exploding_layers=0,
            ppo_data_received=True,
        )
        snapshot.current_batch = 47
        snapshot.max_batches = 100

        widget.update_snapshot(snapshot)
        banner = widget._render_status_banner()

        # Should contain all key metrics
        plain = banner.plain
        assert "EV:" in plain
        assert "Clip:" in plain
        assert "KL:" in plain
        assert "Adv:" in plain
        assert "GradHP:" in plain
        assert "batch:" in plain
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_includes_all_metrics -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    def _render_status_banner(self) -> Text:
        """Render 1-line status banner with icon and key metrics.

        Format per UX spec:
        [OK] LEARNING   EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94 GradHP:OK 12/12 batch:47/100
        """
        status, label, style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        icons = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}
        icon = icons.get(status, "?")

        banner = Text()
        banner.append(f" {icon} ", style=style)
        banner.append(f"{label}   ", style=style)

        if tamiyo.ppo_data_received:
            # EV with warning indicator
            ev_style = self._status_style(self._get_ev_status(tamiyo.explained_variance))
            banner.append(f"EV:{tamiyo.explained_variance:.2f}", style=ev_style)
            if tamiyo.explained_variance <= 0:
                banner.append("!", style="red")
            banner.append("  ")

            # Clip
            clip_style = self._status_style(self._get_clip_status(tamiyo.clip_fraction))
            banner.append(f"Clip:{tamiyo.clip_fraction:.2f}", style=clip_style)
            if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
                banner.append("!", style="yellow")
            banner.append("  ")

            # KL
            kl_style = self._status_style(self._get_kl_status(tamiyo.kl_divergence))
            banner.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
            if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
                banner.append("!", style="yellow")
            banner.append("  ")

            # Advantage summary (per UX spec)
            adv_status = self._get_advantage_status(tamiyo.advantage_std)
            adv_style = self._status_style(adv_status)
            banner.append(f"Adv:{tamiyo.advantage_mean:+.2f}±{tamiyo.advantage_std:.2f}", style=adv_style)
            if adv_status != "ok":
                banner.append("!", style=adv_style)
            banner.append("  ")

            # Gradient health summary (per UX spec)
            total_layers = 12  # Approximate
            healthy = total_layers - tamiyo.dead_layers - tamiyo.exploding_layers
            if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
                banner.append(f"GradHP:!! {tamiyo.dead_layers}D/{tamiyo.exploding_layers}E", style="red")
            else:
                banner.append(f"GradHP:OK {healthy}/{total_layers}", style="green")
            banner.append("  ")

            # Batch progress with denominator (per UX spec)
            batch = self._snapshot.current_batch
            max_batches = getattr(self._snapshot, 'max_batches', 100)
            banner.append(f"batch:{batch}/{max_batches}", style="dim")

        return banner

    def _get_kl_status(self, kl: float) -> str:
        """Get status for KL divergence."""
        if kl > TUIThresholds.KL_CRITICAL:
            return "critical"
        if kl > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _get_advantage_status(self, adv_std: float) -> str:
        """Get status for advantage normalization."""
        if adv_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning"
        if adv_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            return "warning"
        return "ok"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_includes_all_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement complete status banner per UX spec

Includes: EV, Clip, KL, Adv:±, GradHP:, batch:N/M
All metrics have warning indicators (!)"
```

---

### Task 2.3: Implement 4-Gauge Grid

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
        gauge_grid = widget._render_gauge_grid()
        assert gauge_grid is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_four_gauge_grid_rendered -v`
Expected: FAIL

**Step 3: Write minimal implementation**

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
            label=self._get_kl_label(tamiyo.kl_divergence),
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

        # Status-based color (use bright_cyan for OK per UX spec)
        bar_color = {"ok": "bright_cyan", "warning": "yellow", "critical": "red"}[status]

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
        if ev <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
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

    def _get_kl_label(self, kl: float) -> str:
        """Get descriptive label for KL divergence."""
        if kl > TUIThresholds.KL_CRITICAL:
            return "UNSTABLE!"
        elif kl > TUIThresholds.KL_WARNING:
            return "Drifting"
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
- _render_gauge_v2() with bright_cyan for OK status
- New label methods for EV, Clip, KL descriptive text"
```

---

### Task 2.4: Update CSS with Theme Variables

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

/* Use theme variables for consistency (per UX spec) */
#tamiyo-brain.status-ok {
    border: solid $success;
}

#tamiyo-brain.status-warning {
    border: solid $warning;
}

#tamiyo-brain.status-critical {
    border: solid $error;
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
git commit -m "style(sanctum): update TamiyoBrain CSS with theme variables

- min-height: 24, min-width: 96 for expanded layout
- Use \$success/\$warning/\$error for border colors (per UX spec)"
```

---

### Task 2.5: Wire Status Banner + Gauge Grid into render()

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:79-96`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Update render() method**

```python
    # Widget width for separators (96 - 2 for padding = 94)
    SEPARATOR_WIDTH = 94

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

        # Row 2: Separator (full width per UX spec)
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 3: Diagnostic Matrix (gauges left, metrics right)
        # For now, just gauges - Phase 3 adds metrics column
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
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 5: Action Distribution
        action_bar = self._render_action_distribution_bar()
        main_table.add_row(action_bar)

        # Row 6: Separator
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 7: Decision Carousel
        decisions_panel = self._render_recent_decisions()
        main_table.add_row(decisions_panel)

        return main_table
```

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py
git commit -m "feat(sanctum): wire status banner and 4-gauge grid into render()

- SEPARATOR_WIDTH=94 (full width per UX spec)
- Expanded layout with status banner at top
- 4-gauge grid replacing 3-gauge"
```

---

### Task 2.6: Dynamic Border Color Based on Status

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

        # Warning state (EV between 0 and 0.3)
        snapshot.tamiyo.explained_variance = 0.2
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-warning")
        assert not widget.has_class("status-ok")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_border_color_updates_on_status -v`
Expected: FAIL

**Step 3: Write minimal implementation**

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

Border uses theme colors: \$success (LEARNING), \$warning (CAUTION), \$error (FAILING)"
```

---

### Task 2.7: Implement Compact Mode Detection

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_compact_mode_detected_for_narrow_terminal():
    """Widget should detect 80-char terminals and switch to compact layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._is_compact_mode() is True


@pytest.mark.asyncio
async def test_full_mode_for_wide_terminal():
    """Widget should use full layout for 96+ char terminals."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._is_compact_mode() is False


@pytest.mark.asyncio
async def test_compact_mode_reduces_separator_width():
    """Compact mode should reduce separator width from 94 to 78."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_separator_width() == 78  # 80 - 2 padding

    async with app.run_test(size=(120, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_separator_width() == 94  # 96 - 2 padding
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_compact_mode_detected_for_narrow_terminal -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
    # Class constant for layout thresholds
    FULL_WIDTH = 96
    COMPACT_WIDTH = 80

    def _is_compact_mode(self) -> bool:
        """Detect if terminal is too narrow for full 96-char layout."""
        return self.size.width < self.FULL_WIDTH

    def _get_separator_width(self) -> int:
        """Get separator width based on current mode."""
        if self._is_compact_mode():
            return self.COMPACT_WIDTH - 2  # 78 chars
        return self.FULL_WIDTH - 2  # 94 chars

    def _render_separator(self) -> Text:
        """Render horizontal separator at correct width."""
        width = self._get_separator_width()
        return Text("─" * width, style="dim")

    # Compact mode degradation constants
    GAUGE_BAR_WIDTH_FULL = 10
    GAUGE_BAR_WIDTH_COMPACT = 6

    def _get_gauge_bar_width(self) -> int:
        """Get gauge bar width based on current mode."""
        if self._is_compact_mode():
            return self.GAUGE_BAR_WIDTH_COMPACT
        return self.GAUGE_BAR_WIDTH_FULL
```

**Compact Mode Degradation Behavior:**

When `_is_compact_mode()` returns True (terminal < 96 chars), the following adjustments apply:

| Component | Full Mode (96 chars) | Compact Mode (80 chars) |
|-----------|---------------------|------------------------|
| Separators | 94 chars | 78 chars |
| Gauge bars | 10 chars wide | 6 chars wide |
| Metrics labels | `Advantage   ` | `Adv:` |
| Heatmap | Bars + values line | Bars only (omit values) |
| Status banner | Full metrics | Abbreviated metrics |

**Additional compact mode methods** (used by later tasks):

```python
    # Compact label mappings
    METRIC_LABELS_FULL = {
        "advantage": "Advantage   ",
        "policy_loss": "Policy Loss ",
        "value_loss": "Value Loss  ",
    }
    METRIC_LABELS_COMPACT = {
        "advantage": "Adv:",
        "policy_loss": "PLoss:",
        "value_loss": "VLoss:",
    }

    def _get_metric_label(self, metric: str) -> str:
        """Get metric label based on current mode."""
        labels = self.METRIC_LABELS_COMPACT if self._is_compact_mode() else self.METRIC_LABELS_FULL
        return labels.get(metric, metric)

    def _should_show_heatmap_values(self) -> bool:
        """Whether to show numeric values below heatmap bars."""
        return not self._is_compact_mode()
```

**Note:** These methods are defined here but used by Tasks 3.2 (metrics column), 4.1 (heatmap). The implementation provides hooks for graceful degradation without breaking full-width rendering.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_compact_mode_detected_for_narrow_terminal tests/karn/sanctum/test_tamiyo_brain.py::test_full_mode_for_wide_terminal tests/karn/sanctum/test_tamiyo_brain.py::test_compact_mode_reduces_separator_width -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add compact mode detection with graceful degradation

Detects terminal width < 96 and provides hooks for layout adaptation:
- Separator width: 94 → 78 chars
- Gauge bar width: 10 → 6 chars
- Metric labels: full → abbreviated (e.g., 'Advantage' → 'Adv:')
- Heatmap: omit values line in compact mode

Methods defined here, consumed by Tasks 3.2 (metrics) and 4.1 (heatmap)"
```

---

## Phase 2 Complete Checkpoint

Run full test suite:
```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

Expected: All tests pass. TamiyoBrain now has:
- DRL-correct decision tree logic (10+ conditions, proper priority order)
- Complete status banner (EV, Clip, KL, Adv:±, GradHP:, batch:N/M)
- 4-gauge grid (EV, Entropy, Clip, KL) with status colors
- Dynamic border colors
- Full-width separators
- Compact mode detection for 80-char terminals (Task 2.7)

---

## Phase 3: P1/P2 Metrics with Sparklines

**Objective:** Secondary metrics column with sparkline trends.

---

### Task 3.1: Implement Sparkline Renderer

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


@pytest.mark.asyncio
async def test_sparkline_empty_history():
    """Sparkline should show placeholder for empty history."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Empty history
        sparkline = widget._render_sparkline([], width=10)
        assert len(sparkline.plain) == 10
        assert "─" in sparkline.plain
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
        style: str = "bright_cyan",
    ) -> Text:
        """Render sparkline using unicode block characters.

        Returns:
            Text with sparkline or placeholder for empty/flat data.
        """
        BLOCKS = "▁▂▃▄▅▆▇█"

        if not history:
            return Text("─" * width, style="dim")

        values = list(history)[-width:]  # Last N values
        if len(values) < width:
            # Pad with placeholder on left
            pad_count = width - len(values)
            result = Text("─" * pad_count, style="dim")
        else:
            result = Text()

        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        val_range = max_val - min_val if max_val != min_val else 1

        for v in values:
            normalized = (v - min_val) / val_range
            idx = int(normalized * (len(BLOCKS) - 1))
            idx = max(0, min(len(BLOCKS) - 1, idx))
            result.append(BLOCKS[idx], style=style)

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_sparkline_rendering tests/karn/sanctum/test_tamiyo_brain.py::test_sparkline_empty_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement sparkline renderer for trend visualization

Uses unicode blocks ▁▂▃▄▅▆▇█ with left-padding for sparse history"
```

---

### ⏸️ Phase 3 Checkpoint: Verify Sparklines

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -k "sparkline" -v`

---

### Task 3.2: Implement Secondary Metrics Column

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

        # Should contain key metrics
        plain = metrics.plain
        assert "Advantage" in plain
        assert "Ratio" in plain
        assert "Policy" in plain or "Grad" in plain
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
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        adv_style = self._status_style(adv_status)
        result.append(f" Advantage   ", style="dim")
        result.append(f"{tamiyo.advantage_mean:+.2f} ± {tamiyo.advantage_std:.2f}", style=adv_style)
        if adv_status != "ok":
            result.append(" [!]", style=adv_style)
        result.append("\n")

        # Ratio bounds
        ratio_status = self._get_ratio_status(tamiyo.ratio_min, tamiyo.ratio_max)
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
        result.append(f" {tamiyo.policy_loss:.3f}\n", style="bright_cyan")

        # Value loss with sparkline
        vl_sparkline = self._render_sparkline(tamiyo.value_loss_history)
        result.append(f" Value Loss  ", style="dim")
        result.append(vl_sparkline)
        result.append(f" {tamiyo.value_loss:.3f}\n", style="bright_cyan")

        # Grad norm with sparkline
        gn_sparkline = self._render_sparkline(tamiyo.grad_norm_history)
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        gn_style = self._status_style(gn_status)
        result.append(f" Grad Norm   ", style="dim")
        result.append(gn_sparkline)
        result.append(f" {tamiyo.grad_norm:.2f}\n", style=gn_style)

        # Layer health
        total_layers = 12
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(f" Layers      ", style="dim")
            result.append(f"!! {tamiyo.dead_layers} dead, {tamiyo.exploding_layers} exploding", style="red")
        else:
            healthy = total_layers - tamiyo.dead_layers - tamiyo.exploding_layers
            result.append(f" Layers      ", style="dim")
            result.append(f"OK {healthy}/{total_layers} healthy", style="green")

        return result

    def _get_ratio_status(self, ratio_min: float, ratio_max: float) -> str:
        """Get status for PPO ratio bounds."""
        if ratio_max > TUIThresholds.RATIO_MAX_CRITICAL or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL:
            return "critical"
        if ratio_max > TUIThresholds.RATIO_MAX_WARNING or ratio_min < TUIThresholds.RATIO_MIN_WARNING:
            return "warning"
        return "ok"
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

### Task 3.3: Wire Diagnostic Matrix (Gauges + Metrics Side-by-Side)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_diagnostic_matrix_layout():
    """Diagnostic matrix should have gauges left, metrics right."""
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
        matrix.add_column(width=2)  # Separator
        matrix.add_column(ratio=1)  # Metrics

        gauge_grid = self._render_gauge_grid()
        separator = Text("│\n│\n│\n│\n│\n│", style="dim")
        metrics_col = self._render_metrics_column()

        matrix.add_row(gauge_grid, separator, metrics_col)
        return matrix
```

Update `render()` to use diagnostic matrix instead of just gauges.

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

## Phase 3 Complete Checkpoint

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

---

## Phase 4: Per-Head Heatmap + Telemetry

**Objective:** Add per-head entropy heatmap with visual distinction for missing data.

---

### Task 4.1: Implement Per-Head Entropy Heatmap with Dynamic Max Entropy

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_per_head_entropy_heatmap():
    """Per-head heatmap should show 8 heads with correct max entropy values."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=1.2,
            head_tempo_entropy=0.9,
            head_alpha_target_entropy=0.8,
            head_alpha_speed_entropy=1.1,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.5,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        assert heatmap is not None
        # Should contain all 8 head labels
        plain = heatmap.plain
        assert "slot" in plain.lower() or "sl" in plain.lower()
        assert "bp" in plain.lower()


@pytest.mark.asyncio
async def test_per_head_heatmap_missing_data_visual():
    """Heatmap should show visual distinction for unpopulated heads."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        # Only slot and blueprint have data, others are 0.0
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=0.0,  # Missing
            head_tempo_entropy=0.0,  # Missing
            head_alpha_target_entropy=0.0,  # Missing
            head_alpha_speed_entropy=0.0,  # Missing
            head_alpha_curve_entropy=0.0,  # Missing
            head_op_entropy=0.0,  # Missing
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        plain = heatmap.plain.lower()

        # Should indicate missing/pending data visually
        # (implementation will use "n/a" or similar for zeros)
        assert "n/a" in plain or "---" in plain or "?" in plain
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_per_head_entropy_heatmap -v`
Expected: FAIL

**Step 3: Write minimal implementation with CORRECT max entropy values**

```python
    # Per-head max entropy values from factored_actions.py (DRL CORRECTED)
    # These are ln(N) where N is the number of actions for each head
    HEAD_MAX_ENTROPIES = {
        "slot": 1.099,       # ln(3) - default SlotConfig has 3 slots
        "blueprint": 2.565,  # ln(13) - BlueprintAction has 13 values
        "style": 1.386,      # ln(4) - GerminationStyle has 4 values
        "tempo": 1.099,      # ln(3) - TempoAction has 3 values
        "alpha_target": 1.099,  # ln(3) - AlphaTargetAction has 3 values
        "alpha_speed": 1.386,   # ln(4) - AlphaSpeedAction has 4 values
        "alpha_curve": 1.099,   # ln(3) - AlphaCurveAction has 3 values
        "op": 1.792,         # ln(6) - LifecycleOp has 6 values
    }

    # Heads that PPOAgent currently tracks (others awaiting neural network changes)
    TRACKED_HEADS = {"slot", "blueprint"}

    def _render_head_heatmap(self) -> Text:
        """Render per-head entropy heatmap with 8 action heads.

        Shows visual distinction for heads without telemetry data.
        Per DRL review: max entropy values computed from actual action space dimensions.
        """
        tamiyo = self._snapshot.tamiyo

        # Head config: (abbrev, field_name)
        heads = [
            ("sl", "head_slot_entropy", "slot"),
            ("bp", "head_blueprint_entropy", "blueprint"),
            ("sy", "head_style_entropy", "style"),
            ("te", "head_tempo_entropy", "tempo"),
            ("at", "head_alpha_target_entropy", "alpha_target"),
            ("as", "head_alpha_speed_entropy", "alpha_speed"),
            ("ac", "head_alpha_curve_entropy", "alpha_curve"),
            ("op", "head_op_entropy", "op"),
        ]

        result = Text()
        result.append(" Heads: ", style="dim")

        # First line: bars
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            is_tracked = head_key in self.TRACKED_HEADS

            # Check for missing data (value=0.0 for untracked heads)
            if value == 0.0 and not is_tracked:
                # Visual distinction for awaiting telemetry (per risk assessor)
                result.append(f"{abbrev}[", style="dim")
                result.append("n/a", style="dim italic")
                result.append("] ")
                continue

            # Normalize to 0-1
            fill = value / max_ent if max_ent > 0 else 0
            fill = max(0, min(1, fill))

            # 3-char bar (narrower for 80-char terminal compatibility)
            bar_width = 3
            filled = int(fill * bar_width)
            empty = bar_width - filled

            # Color based on fill level (high entropy = exploring, low = converged)
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

        # Second line: values (fixed-width for alignment per UX spec)
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            is_tracked = head_key in self.TRACKED_HEADS

            if value == 0.0 and not is_tracked:
                result.append("  ?  ", style="dim italic")
                result.append(" ")
                continue

            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            fill = value / max_ent if max_ent > 0 else 0

            # Fixed 5-char width for alignment
            if fill < 0.25:
                result.append(f"{value:5.2f}!", style="red")
            else:
                result.append(f"{value:5.2f} ", style="dim")
            result.append(" ")

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_per_head_entropy_heatmap tests/karn/sanctum/test_tamiyo_brain.py::test_per_head_heatmap_missing_data_visual -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): implement per-head entropy heatmap with correct max values

- Max entropy from factored_actions.py enums (DRL corrected)
- Visual distinction (n/a, ?) for heads awaiting telemetry
- 3-char bars for 80-char terminal compatibility"
```

---

### Task 4.2: Wire Heatmap into render()

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_heatmap_appears_in_render():
    """Heatmap should appear in widget render output."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Force render and check output
        rendered = widget.render()
        # The render should include head heatmap section
        assert rendered is not None
```

**Step 2: Update render() method**

Add heatmap section after diagnostic matrix:

```python
    def render(self):
        """Render Tamiyo content with expanded layout."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Row 1: Status Banner
        status_banner = self._render_status_banner()
        main_table.add_row(status_banner)

        # Row 2: Separator
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 3: Diagnostic Matrix (gauges + metrics)
        if self._snapshot.tamiyo.ppo_data_received:
            diagnostic_matrix = self._render_diagnostic_matrix()
            main_table.add_row(diagnostic_matrix)
        else:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            main_table.add_row(waiting_text)

        # Row 4: Separator
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 5: Per-Head Entropy Heatmap (P2 cool factor)
        if self._snapshot.tamiyo.ppo_data_received:
            head_heatmap = self._render_head_heatmap()
            main_table.add_row(head_heatmap)

            # Row 6: Separator
            main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 7: Action Distribution
        action_bar = self._render_action_distribution_bar()
        main_table.add_row(action_bar)

        # Row 8: Separator
        main_table.add_row(Text("─" * self.SEPARATOR_WIDTH, style="dim"))

        # Row 9: Decision Carousel
        decisions_panel = self._render_recent_decisions()
        main_table.add_row(decisions_panel)

        return main_table
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_heatmap_appears_in_render -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): wire per-head entropy heatmap into render()

Adds Row 5 with head entropy visualization after diagnostic matrix"
```

---

### Task 4.3: Extend Aggregator for 6 New Head Entropies

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
def test_ppo_update_populates_head_entropies():
    """PPO_UPDATE_COMPLETED should populate all 8 head entropies when available."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.5,
            # Per-head entropies (when neural network emits them)
            "head_slot_entropy": 1.0,
            "head_blueprint_entropy": 2.0,
            "head_style_entropy": 1.2,
            "head_tempo_entropy": 0.9,
            "head_alpha_target_entropy": 0.8,
            "head_alpha_speed_entropy": 1.1,
            "head_alpha_curve_entropy": 0.7,
            "head_op_entropy": 1.5,
        },
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    tamiyo = snapshot.tamiyo

    # Should have all 8 head entropies
    assert tamiyo.head_slot_entropy == 1.0
    assert tamiyo.head_blueprint_entropy == 2.0
    assert tamiyo.head_style_entropy == 1.2
    assert tamiyo.head_tempo_entropy == 0.9
    assert tamiyo.head_alpha_target_entropy == 0.8
    assert tamiyo.head_alpha_speed_entropy == 1.1
    assert tamiyo.head_alpha_curve_entropy == 0.7
    assert tamiyo.head_op_entropy == 1.5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_populates_head_entropies -v`
Expected: FAIL (aggregator doesn't extract these fields yet)

**Step 3: Write minimal implementation**

In `_handle_ppo_update` (aggregator.py), add after existing entropy handling:

```python
        # Per-head entropies (for heatmap visualization)
        # These are optional - only present when neural network emits them
        self._tamiyo.head_slot_entropy = data.get("head_slot_entropy", self._tamiyo.head_slot_entropy)
        self._tamiyo.head_blueprint_entropy = data.get("head_blueprint_entropy", self._tamiyo.head_blueprint_entropy)
        self._tamiyo.head_style_entropy = data.get("head_style_entropy", self._tamiyo.head_style_entropy)
        self._tamiyo.head_tempo_entropy = data.get("head_tempo_entropy", self._tamiyo.head_tempo_entropy)
        self._tamiyo.head_alpha_target_entropy = data.get("head_alpha_target_entropy", self._tamiyo.head_alpha_target_entropy)
        self._tamiyo.head_alpha_speed_entropy = data.get("head_alpha_speed_entropy", self._tamiyo.head_alpha_speed_entropy)
        self._tamiyo.head_alpha_curve_entropy = data.get("head_alpha_curve_entropy", self._tamiyo.head_alpha_curve_entropy)
        self._tamiyo.head_op_entropy = data.get("head_op_entropy", self._tamiyo.head_op_entropy)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_populates_head_entropies -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): extract per-head entropies in aggregator

Reads 8 head entropy fields from PPO_UPDATE_COMPLETED when available"
```

---

### Task 4.4: Document Telemetry Gap (No Code Changes)

**Note:** The neural network (`PPOAgent`) currently only emits `head_slot_entropy` and `head_blueprint_entropy`. The other 6 heads require changes to the policy network to compute per-head entropy during `get_action()`.

This is a **known limitation** documented in the plan. The TUI gracefully handles missing data by showing "n/a" for untracked heads.

**Future Work (out of scope for this plan):**
- Modify `FactoredRecurrentActorCritic.get_action()` to compute per-head entropy
- Emit all 8 head entropies in `PPO_UPDATE_COMPLETED` event

---

## Phase 4 Complete Checkpoint

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests pass. TamiyoBrain now has:
- Per-head entropy heatmap with 8 action heads
- Visual distinction for awaiting telemetry ("n/a")
- Correct max entropy values from factored_actions.py

---

## Phase 5: A/B Testing Color-Coded Tamiyos

> ⚠️ **IMPLEMENTATION ORDER:** This phase provides color coding but is only useful after Phase 6 (Multi-Aggregator) is complete. Implement Phase 6 FIRST, then return here for Phase 5.

**Objective:** When running dual-policy A/B testing, show two (or three) TamiyoBrain widgets with distinct color coding to differentiate policies.

**Context:** The `--dual-ab` CLI flag trains separate policies on separate GPUs. Each PolicyGroup has a `group_id` ("A", "B", or "C"). The Sanctum TUI should display one TamiyoBrain per group with color-coded borders.

---

### Task 5.1: Add group_id to TamiyoState

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
def test_tamiyo_state_has_group_id():
    """TamiyoState should have group_id for A/B testing identification."""
    from esper.karn.sanctum.schema import TamiyoState

    state = TamiyoState()
    assert hasattr(state, 'group_id')
    assert state.group_id is None  # Default when not A/B testing

    state_a = TamiyoState(group_id="A")
    assert state_a.group_id == "A"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_group_id -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `TamiyoState` in `schema.py`:

```python
    # A/B testing identification (None when not in A/B mode)
    group_id: str | None = None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_group_id -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add group_id to TamiyoState for A/B testing

Identifies which policy group (A, B, or C) this Tamiyo represents"
```

---

### Task 5.2: Add Color Scheme Constants for A/B/C Groups

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_ab_group_color_mapping():
    """TamiyoBrain should have color mappings for A/B/C groups."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Should have color constants for A/B/C
        assert hasattr(widget, 'GROUP_COLORS')
        assert "A" in widget.GROUP_COLORS
        assert "B" in widget.GROUP_COLORS
        assert "C" in widget.GROUP_COLORS

        # A = green, B = cyan, C = magenta (not red - conflicts with error semantics)
        assert "green" in widget.GROUP_COLORS["A"].lower()
        assert "blue" in widget.GROUP_COLORS["B"].lower() or "cyan" in widget.GROUP_COLORS["B"].lower()
        assert "magenta" in widget.GROUP_COLORS["C"].lower()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_ab_group_color_mapping -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `TamiyoBrain` class:

```python
    # A/B/C testing color scheme
    # A = Green (primary/control), B = Cyan (variant), C = Magenta (second variant)
    # NOTE: Do NOT use red for C - red is reserved for error/critical states in TUI semantics
    GROUP_COLORS: ClassVar[dict[str, str]] = {
        "A": "bright_green",
        "B": "bright_cyan",    # Blue family
        "C": "bright_magenta", # NOT red - red conflicts with error semantics
    }

    GROUP_LABELS: ClassVar[dict[str, str]] = {
        "A": "🟢 Policy A",
        "B": "🔵 Policy B",
        "C": "🟣 Policy C",  # Purple/magenta emoji - not red
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_ab_group_color_mapping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add A/B/C group color constants

A=green, B=blue/cyan, C=red for visual policy differentiation"
```

---

### Task 5.3: Apply Group Color to Border and Title

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Modify: `src/esper/karn/sanctum/styles.tcss`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_group_a_has_green_border():
    """Group A TamiyoBrain should have green border styling."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="A",
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        assert widget.has_class("group-a")


@pytest.mark.asyncio
async def test_group_b_has_blue_border():
    """Group B TamiyoBrain should have blue border styling."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="B",
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        assert widget.has_class("group-b")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_group_a_has_green_border -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `_update_status_class()` in `tamiyo_brain.py`:

```python
    def _update_status_class(self) -> None:
        """Update CSS class based on overall status and A/B group."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok", "status-warning", "status-critical")
        self.remove_class("group-a", "group-b", "group-c")

        # Add current status class
        self.add_class(f"status-{status}")

        # Add group class if in A/B mode
        if self._snapshot and self._snapshot.tamiyo.group_id:
            group = self._snapshot.tamiyo.group_id.lower()
            self.add_class(f"group-{group}")
```

Update `styles.tcss` — **CRITICAL: Insert BEFORE the `#tamiyo-brain.status-ok` block (around line 1079)**:

```css
/* A/B/C Testing Group Colors
   MUST come BEFORE status-* classes in this file!
   CSS cascade: when both .group-a and .status-critical apply,
   the LATER rule wins. We want status colors to override group colors,
   so a failing Policy A shows RED (critical), not GREEN (group-a).
*/
#tamiyo-brain.group-a {
    border: solid bright_green;
    border-title-color: bright_green;
}

#tamiyo-brain.group-b {
    border: solid bright_cyan;
    border-title-color: bright_cyan;
}

#tamiyo-brain.group-c {
    border: solid bright_magenta;
    border-title-color: bright_magenta;
}

/* Status-based styling BELOW - these override group colors when health is critical */
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_group_a_has_green_border tests/karn/sanctum/test_tamiyo_brain.py::test_group_b_has_blue_border -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): apply A/B/C group colors to TamiyoBrain borders

Green for A, Cyan for B, Magenta for C - visual policy differentiation
(Note: C uses magenta, not red, to avoid conflict with error semantics)"
```

---

### Task 5.4: Add Group Label to Status Banner

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_status_banner_shows_group_label():
    """Status banner should show group label when in A/B mode."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="A",
            entropy=1.2,
            explained_variance=0.6,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        banner = widget._render_status_banner()
        plain = banner.plain

        # Should show group indicator
        assert "Policy A" in plain or "🟢" in plain or "[A]" in plain
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_shows_group_label -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `_render_status_banner()` to prepend group label:

```python
    def _render_status_banner(self) -> Text:
        """Render 1-line status banner with icon and key metrics."""
        status, label, style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        icons = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}
        icon = icons.get(status, "?")

        banner = Text()

        # Prepend group label if in A/B mode
        if tamiyo.group_id:
            group_color = self.GROUP_COLORS.get(tamiyo.group_id, "white")
            group_label = self.GROUP_LABELS.get(tamiyo.group_id, f"[{tamiyo.group_id}]")
            banner.append(f" {group_label} ", style=group_color)
            banner.append("│ ", style="dim")

        banner.append(f"{icon} ", style=style)
        banner.append(f"{label}   ", style=style)

        # ... rest of existing banner code ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_status_banner_shows_group_label -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): show group label in status banner for A/B testing

Prepends '🟢 Policy A │' or '🔵 Policy B │' when group_id is set"
```

---

### Task 5.5: Update Aggregator to Extract group_id from Telemetry

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
def test_ppo_update_extracts_group_id():
    """PPO_UPDATE_COMPLETED should extract group_id for A/B testing."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={
            "policy_loss": 0.1,
            "group_id": "B",  # From dual-policy training
        },
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.group_id == "B"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_extracts_group_id -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `_handle_ppo_update`, add:

```python
        # A/B testing group identification
        group_id = data.get("group_id")
        if group_id:
            self._tamiyo.group_id = group_id
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_ppo_update_extracts_group_id -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(sanctum): extract group_id from telemetry for A/B testing

Maps PolicyGroup identity to TamiyoState for color-coded display"
```

---

## Phase 5 Complete Checkpoint

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests pass. A/B testing now has:
- `group_id` field on TamiyoState
- Color constants: A=green, B=cyan, C=magenta (not red - avoids error color conflict)
- Dynamic border colors based on group
- Group label in status banner
- Aggregator extracts group_id from telemetry

---

## Phase 6: Multi-Aggregator TUI Infrastructure (Full Workstream)

> ⚠️ **IMPLEMENTATION ORDER:** This phase provides the infrastructure for A/B comparison. Implement this BEFORE Phase 5 (Color Coding).

**Objective:** Enable true side-by-side A/B policy comparison by supporting multiple SanctumAggregators, each feeding its own TamiyoBrain widget. This is the prerequisite infrastructure for meaningful A/B training visualization.

**Context:** When running `--dual-ab`, each PolicyGroup trains independently on its own GPU. To compare policies visually, we need:
1. Multiple aggregators (one per PolicyGroup)
2. A registry to manage aggregator instances
3. Event routing from telemetry hub to the correct aggregator
4. A layout that displays multiple TamiyoBrains side-by-side

**Architecture Diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MULTI-AGGREGATOR SANCTUM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─ Nissa Hub ───────────────────────────────────────────────────────┐  │
│  │  TelemetryEvent(group_id="A", ...)  TelemetryEvent(group_id="B")  │  │
│  └────────────────────────┬────────────────────────┬─────────────────┘  │
│                           │                        │                     │
│                           ▼                        ▼                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   AggregatorRegistry                                │ │
│  │   ┌─────────────────┐     ┌─────────────────┐                      │ │
│  │   │ aggregators["A"]│     │ aggregators["B"]│                      │ │
│  │   │ SanctumAggregator│    │ SanctumAggregator│                     │ │
│  │   └────────┬────────┘     └────────┬────────┘                      │ │
│  └────────────┼────────────────────────┼──────────────────────────────┘ │
│               │                        │                                 │
│               ▼                        ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   SanctumApp (Textual)                              │ │
│  │   ┌─────────────────────┐     ┌─────────────────────┐              │ │
│  │   │  TamiyoBrain        │     │  TamiyoBrain        │              │ │
│  │   │  (group_id="A")     │     │  (group_id="B")     │              │ │
│  │   │  🟢 Policy A        │     │  🔵 Policy B        │              │ │
│  │   └─────────────────────┘     └─────────────────────┘              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Task 6.1: Create AggregatorRegistry

**Files:**
- Create: `src/esper/karn/sanctum/registry.py`
- Test: `tests/karn/sanctum/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_registry.py

import pytest
from esper.karn.sanctum.registry import AggregatorRegistry
from esper.karn.sanctum.aggregator import SanctumAggregator


def test_registry_creates_aggregator_on_demand():
    """Registry should create aggregator when first accessed."""
    registry = AggregatorRegistry(num_envs=4)

    # First access creates aggregator
    agg_a = registry.get_or_create("A")
    assert isinstance(agg_a, SanctumAggregator)

    # Second access returns same instance
    agg_a2 = registry.get_or_create("A")
    assert agg_a is agg_a2


def test_registry_manages_multiple_aggregators():
    """Registry should manage multiple independent aggregators."""
    registry = AggregatorRegistry(num_envs=4)

    agg_a = registry.get_or_create("A")
    agg_b = registry.get_or_create("B")

    # Different instances
    assert agg_a is not agg_b

    # Both tracked
    assert registry.group_ids == {"A", "B"}


def test_registry_list_snapshots():
    """Registry should return snapshots for all aggregators."""
    registry = AggregatorRegistry(num_envs=4)

    registry.get_or_create("A")
    registry.get_or_create("B")

    snapshots = registry.get_all_snapshots()

    assert len(snapshots) == 2
    assert "A" in snapshots
    assert "B" in snapshots
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_registry.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write minimal implementation**

```python
# src/esper/karn/sanctum/registry.py

"""Aggregator Registry for multi-policy A/B testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.karn.sanctum.schema import SanctumSnapshot


class AggregatorRegistry:
    """Manages multiple SanctumAggregators for A/B testing.

    Each PolicyGroup gets its own aggregator, keyed by group_id.
    The registry creates aggregators on-demand when first accessed.
    """

    def __init__(self, num_envs: int = 4) -> None:
        self._num_envs = num_envs
        self._aggregators: dict[str, SanctumAggregator] = {}

    def get_or_create(self, group_id: str) -> "SanctumAggregator":
        """Get existing aggregator or create new one for group."""
        if group_id not in self._aggregators:
            from esper.karn.sanctum.aggregator import SanctumAggregator
            self._aggregators[group_id] = SanctumAggregator(
                num_envs=self._num_envs
            )
        return self._aggregators[group_id]

    @property
    def group_ids(self) -> set[str]:
        """Return set of all registered group IDs."""
        return set(self._aggregators.keys())

    def get_all_snapshots(self) -> dict[str, "SanctumSnapshot"]:
        """Return snapshots from all aggregators."""
        return {
            group_id: agg.get_snapshot()
            for group_id, agg in self._aggregators.items()
        }

    def process_event(self, event) -> None:
        """Route event to appropriate aggregator based on group_id."""
        # Null safety: event.data may be None for some event types
        data = event.data or {}
        group_id = data.get("group_id", "default")
        agg = self.get_or_create(group_id)
        agg.process_event(event)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/registry.py tests/karn/sanctum/test_registry.py
git commit -m "feat(sanctum): add AggregatorRegistry for multi-policy A/B testing

Manages multiple SanctumAggregators, one per PolicyGroup, enabling
side-by-side policy comparison in the TUI."
```

---

### Task 6.2: Add Event Routing to Registry

**Files:**
- Modify: `src/esper/karn/sanctum/registry.py`
- Test: `tests/karn/sanctum/test_registry.py`

**Step 1: Write the failing test**

```python
def test_registry_routes_events_by_group_id():
    """Registry should route events to correct aggregator."""
    from esper.leyline import TelemetryEvent, TelemetryEventType

    registry = AggregatorRegistry(num_envs=4)

    # Send event for group A
    event_a = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={"group_id": "A", "policy_loss": 0.1},
    )
    registry.process_event(event_a)

    # Send event for group B
    event_b = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={"group_id": "B", "policy_loss": 0.2},
    )
    registry.process_event(event_b)

    # Verify events went to correct aggregators
    snapshots = registry.get_all_snapshots()
    assert snapshots["A"].tamiyo.policy_loss == 0.1
    assert snapshots["B"].tamiyo.policy_loss == 0.2


def test_registry_default_group_for_missing_group_id():
    """Events without group_id should go to 'default' aggregator."""
    from esper.leyline import TelemetryEvent, TelemetryEventType

    registry = AggregatorRegistry(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={"policy_loss": 0.15},  # No group_id
    )
    registry.process_event(event)

    assert "default" in registry.group_ids
    snapshots = registry.get_all_snapshots()
    assert snapshots["default"].tamiyo.policy_loss == 0.15
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_registry.py::test_registry_routes_events_by_group_id -v`
Expected: FAIL (process_event may need refinement)

**Step 3: Verify implementation (already done in Task 6.1)**

The `process_event` method was already implemented. Verify it handles the test cases.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/sanctum/test_registry.py
git commit -m "test(sanctum): add event routing tests for AggregatorRegistry

Verifies group_id-based routing and default group fallback"
```

---

### Task 6.3: Wire Registry into SanctumApp

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_sanctum_app_creates_registry():
    """SanctumApp should create AggregatorRegistry on init."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.registry import AggregatorRegistry

    app = SanctumApp()
    assert hasattr(app, '_registry')
    assert isinstance(app._registry, AggregatorRegistry)


@pytest.mark.asyncio
async def test_sanctum_app_routes_to_registry():
    """SanctumApp should route events through registry."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.leyline import TelemetryEvent, TelemetryEventType

    app = SanctumApp()

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={"group_id": "A", "policy_loss": 0.1},
    )

    app.handle_telemetry_event(event)

    snapshots = app._registry.get_all_snapshots()
    assert "A" in snapshots
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_sanctum_app_creates_registry -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `SanctumApp.__init__`:

```python
from esper.karn.sanctum.registry import AggregatorRegistry

class SanctumApp(App):
    def __init__(self, num_envs: int = 4) -> None:
        super().__init__()
        self._registry = AggregatorRegistry(num_envs=num_envs)
        # ... rest of init

    def handle_telemetry_event(self, event) -> None:
        """Route telemetry event to appropriate aggregator."""
        self._registry.process_event(event)
        self._update_widgets()

    def _update_widgets(self) -> None:
        """Update all TamiyoBrain widgets with latest snapshots."""
        snapshots = self._registry.get_all_snapshots()
        # For each registered group, update corresponding widget
        for group_id, snapshot in snapshots.items():
            widget = self._get_or_create_tamiyo_widget(group_id)
            widget.update_snapshot(snapshot)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): wire AggregatorRegistry into SanctumApp

Routes telemetry events through registry for multi-policy support"
```

---

#### ✅ Checkpoint: Phase 6 Infrastructure Integration

**Before proceeding to multi-widget layout, verify:**

1. **Registry tests pass:**
   ```bash
   PYTHONPATH=src uv run pytest tests/karn/sanctum/test_registry.py -v
   ```

2. **App integration tests pass:**
   ```bash
   PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v
   ```

3. **Single-aggregator mode still functions:**
   - Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.app import SanctumApp; print('OK')"`
   - Verify no import errors

4. **Manual sanity check (optional):**
   - Start app in test mode, send single-group events
   - Verify existing TamiyoBrain widget renders correctly

**If any checkpoint fails:** Fix before proceeding. Multi-widget layout (Tasks 6.4-6.7) depends on stable registry infrastructure.

---

### Task 6.4: Implement Multi-TamiyoBrain Layout

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Modify: `src/esper/karn/sanctum/styles.tcss`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_sanctum_app_shows_multiple_tamiyo_widgets():
    """A/B mode should show two TamiyoBrain widgets side-by-side."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.leyline import TelemetryEvent, TelemetryEventType

    app = SanctumApp()
    async with app.run_test():
        # Send events for two groups
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data={"group_id": group_id, "policy_loss": 0.1},
            )
            app.handle_telemetry_event(event)

        # Should have two TamiyoBrain widgets
        widgets = app.query(TamiyoBrain)
        assert len(widgets) == 2

        # Each should have correct group class
        group_classes = {w.classes for w in widgets}
        assert any("group-a" in c for c in group_classes)
        assert any("group-b" in c for c in group_classes)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_sanctum_app_shows_multiple_tamiyo_widgets -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add dynamic widget creation to `SanctumApp`:

```python
    def compose(self) -> ComposeResult:
        """Compose the app layout with dynamic TamiyoBrain container."""
        yield Header()
        with Horizontal(id="tamiyo-container"):
            # TamiyoBrain widgets will be added dynamically
            pass
        yield Footer()

    def _get_or_create_tamiyo_widget(self, group_id: str) -> TamiyoBrain:
        """Get existing TamiyoBrain or create new one for group."""
        widget_id = f"tamiyo-{group_id.lower()}"

        try:
            return self.query_one(f"#{widget_id}", TamiyoBrain)
        except NoMatches:
            # Create new widget
            widget = TamiyoBrain(id=widget_id)
            widget.add_class(f"group-{group_id.lower()}")

            container = self.query_one("#tamiyo-container")
            container.mount(widget)

            return widget
```

Add CSS for horizontal layout:

```tcss
/* Multi-TamiyoBrain horizontal layout */
#tamiyo-container {
    layout: horizontal;
    width: 100%;
    height: auto;
}

#tamiyo-container TamiyoBrain {
    width: 1fr;  /* Equal width for all widgets */
    margin: 0 1;
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): implement multi-TamiyoBrain horizontal layout

Dynamically creates TamiyoBrain widgets for each PolicyGroup in A/B mode"
```

---

### Task 6.5: Add Keyboard Navigation Between Policies

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_keyboard_switches_between_policies():
    """Tab key should cycle focus between policy widgets."""
    from esper.karn.sanctum.app import SanctumApp
    from textual.keys import Keys

    app = SanctumApp()
    async with app.run_test() as pilot:
        # Create two policies
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data={"group_id": group_id, "policy_loss": 0.1},
            )
            app.handle_telemetry_event(event)

        # Press Tab to switch focus
        await pilot.press("tab")

        # Focused widget should have focus class
        focused = app.query_one(":focus")
        assert focused is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_keyboard_switches_between_policies -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Make TamiyoBrain focusable:

```python
class TamiyoBrain(Widget):
    can_focus = True

    def on_focus(self) -> None:
        """Handle focus - highlight border."""
        self.add_class("focused")

    def on_blur(self) -> None:
        """Handle blur - remove highlight."""
        self.remove_class("focused")
```

Add focus CSS:

```tcss
TamiyoBrain.focused {
    border: double $accent;
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): add keyboard navigation between policy widgets

Tab cycles focus between TamiyoBrain widgets in A/B mode"
```

---

### Task 6.6: Add Comparison Metrics Header

**Files:**
- Create: `src/esper/karn/sanctum/widgets/comparison_header.py`
- Test: `tests/karn/sanctum/test_comparison_header.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_comparison_header.py

import pytest
from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader


@pytest.mark.asyncio
async def test_comparison_header_shows_delta():
    """Comparison header should show accuracy delta between policies."""
    from textual.app import App, ComposeResult

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield ComparisonHeader()

    app = TestApp()
    async with app.run_test():
        header = app.query_one(ComparisonHeader)

        header.update_comparison(
            group_a_accuracy=75.0,
            group_b_accuracy=68.0,
            group_a_reward=12.5,
            group_b_reward=10.2,
        )

        rendered = header.render()
        plain = rendered.plain if hasattr(rendered, 'plain') else str(rendered)

        # Should show delta
        assert "+7.0%" in plain or "Δ 7.0" in plain


def test_comparison_header_winner_indication():
    """Header should indicate which policy is leading."""
    header = ComparisonHeader()

    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=12.5,
        group_b_reward=10.2,
    )

    assert header.leader == "A"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_comparison_header.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write minimal implementation**

```python
# src/esper/karn/sanctum/widgets/comparison_header.py

"""Comparison Header for A/B testing - shows delta metrics."""

from __future__ import annotations

from textual.widget import Widget
from rich.text import Text


class ComparisonHeader(Widget):
    """Shows comparison metrics between A/B policies."""

    DEFAULT_CSS = """
    ComparisonHeader {
        height: 3;
        dock: top;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._group_a_accuracy = 0.0
        self._group_b_accuracy = 0.0
        self._group_a_reward = 0.0
        self._group_b_reward = 0.0
        self._leader = None

    @property
    def leader(self) -> str | None:
        """Return group ID of current leader."""
        return self._leader

    def update_comparison(
        self,
        group_a_accuracy: float,
        group_b_accuracy: float,
        group_a_reward: float,
        group_b_reward: float,
    ) -> None:
        """Update comparison metrics."""
        self._group_a_accuracy = group_a_accuracy
        self._group_b_accuracy = group_b_accuracy
        self._group_a_reward = group_a_reward
        self._group_b_reward = group_b_reward

        # Determine leader: reward-first (primary RL objective), accuracy as tiebreaker
        # Per DRL expert review: reward is the RL objective, accuracy is secondary
        reward_delta = group_a_reward - group_b_reward
        mean_reward = (abs(group_a_reward) + abs(group_b_reward)) / 2

        # Significant reward difference (>5% of mean) is decisive
        if mean_reward > 0 and abs(reward_delta) > 0.05 * mean_reward:
            self._leader = "A" if reward_delta > 0 else "B"
        # Fallback to accuracy for close reward races
        elif group_a_accuracy > group_b_accuracy:
            self._leader = "A"
        elif group_b_accuracy > group_a_accuracy:
            self._leader = "B"
        else:
            self._leader = None

        self.refresh()

    def render(self) -> Text:
        """Render comparison bar."""
        delta_acc = self._group_a_accuracy - self._group_b_accuracy
        delta_reward = self._group_a_reward - self._group_b_reward

        result = Text()
        result.append("A/B Comparison │ ")

        # Accuracy delta
        sign = "+" if delta_acc >= 0 else ""
        if abs(delta_acc) > 5:
            style = "green bold" if delta_acc > 0 else "red bold"
        else:
            style = "dim"
        result.append(f"Acc Δ: {sign}{delta_acc:.1f}% ", style=style)

        result.append("│ ")

        # Reward delta
        sign = "+" if delta_reward >= 0 else ""
        result.append(f"Reward Δ: {sign}{delta_reward:.2f} ", style="dim")

        result.append("│ ")

        # Leader indicator
        if self._leader:
            color = "green" if self._leader == "A" else "cyan"
            result.append(f"Leading: {self._leader}", style=f"{color} bold")
        else:
            result.append("Tied", style="dim italic")

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_comparison_header.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/comparison_header.py tests/karn/sanctum/test_comparison_header.py
git commit -m "feat(sanctum): add ComparisonHeader for A/B testing

Shows accuracy/reward deltas and leader indication between policies"
```

---

### Task 6.7: Wire Comparison Header into App

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_comparison_header_appears_in_ab_mode():
    """Comparison header should appear when 2+ policies exist."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader

    app = SanctumApp()
    async with app.run_test():
        # Create two policies
        for group_id in ["A", "B"]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data={"group_id": group_id, "policy_loss": 0.1},
            )
            app.handle_telemetry_event(event)

        # Should have comparison header
        headers = app.query(ComparisonHeader)
        assert len(headers) == 1


@pytest.mark.asyncio
async def test_comparison_header_hidden_in_single_mode():
    """Comparison header should be hidden with only one policy."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader

    app = SanctumApp()
    async with app.run_test():
        # Only one policy
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data={"group_id": "A", "policy_loss": 0.1},
        )
        app.handle_telemetry_event(event)

        # Header should be hidden or not exist
        headers = app.query(ComparisonHeader)
        if headers:
            assert headers[0].display is False
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_comparison_header_appears_in_ab_mode -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `SanctumApp.compose()` and `_update_widgets()`:

```python
from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader

class SanctumApp(App):
    def compose(self) -> ComposeResult:
        yield Header()
        yield ComparisonHeader(id="comparison-header")
        with Horizontal(id="tamiyo-container"):
            pass
        yield Footer()

    def _update_widgets(self) -> None:
        """Update all widgets with latest snapshots."""
        snapshots = self._registry.get_all_snapshots()

        # Update TamiyoBrain widgets
        for group_id, snapshot in snapshots.items():
            widget = self._get_or_create_tamiyo_widget(group_id)
            snapshot.tamiyo.group_id = group_id  # Ensure group_id is set
            widget.update_snapshot(snapshot)

        # Update comparison header
        header = self.query_one("#comparison-header", ComparisonHeader)
        if len(snapshots) >= 2:
            header.display = True
            # Extract comparison metrics from snapshots
            groups = list(snapshots.items())
            # Per DRL review: use snapshot aggregate fields, not TamiyoState
            # (TamiyoState has PPO metrics, not episode-level aggregates)
            header.update_comparison(
                group_a_accuracy=groups[0][1].aggregate_mean_accuracy,
                group_b_accuracy=groups[1][1].aggregate_mean_accuracy,
                group_a_reward=groups[0][1].aggregate_mean_reward,
                group_b_reward=groups[1][1].aggregate_mean_reward,
            )
        else:
            header.display = False
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): wire ComparisonHeader into SanctumApp

Shows/hides comparison header based on number of active policies"
```

---

## Phase 6 Complete Checkpoint

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests pass. Multi-aggregator infrastructure now has:
- `AggregatorRegistry` managing multiple `SanctumAggregator` instances
- Event routing by `group_id` to correct aggregator
- Dynamic TamiyoBrain widget creation per PolicyGroup
- Horizontal layout for side-by-side policy comparison
- Keyboard navigation (Tab) between policy widgets
- `ComparisonHeader` showing accuracy/reward deltas
- Automatic header visibility based on policy count

---

## Final Integration Test

**File:** `tests/karn/sanctum/test_tamiyo_integration.py`

```python
@pytest.mark.asyncio
async def test_full_tamiyo_brain_ab_mode():
    """Integration test: TamiyoBrain with A/B testing and all features."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1", "R0C2"])
        snapshot.tamiyo = TamiyoState(
            # A/B mode
            group_id="A",
            # Core PPO metrics
            entropy=1.2,
            explained_variance=0.65,
            clip_fraction=0.18,
            kl_divergence=0.008,
            # Advantage stats
            advantage_mean=0.12,
            advantage_std=0.94,
            # Losses
            policy_loss=0.025,
            value_loss=0.142,
            grad_norm=1.5,
            # Per-head entropies
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            # Gradient health
            dead_layers=0,
            exploding_layers=0,
            # Flag
            ppo_data_received=True,
        )
        # Add history
        for i in range(5):
            snapshot.tamiyo.policy_loss_history.append(0.03 - i * 0.001)
            snapshot.tamiyo.entropy_history.append(1.2 + i * 0.01)

        snapshot.current_batch = 47
        snapshot.max_batches = 100

        widget.update_snapshot(snapshot)

        # Should have group class
        assert widget.has_class("group-a")

        # Should have OK status (all metrics healthy)
        assert widget.has_class("status-ok")

        # Render should succeed
        rendered = widget.render()
        assert rendered is not None
```

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_integration.py -v`

---

### Multi-Aggregator Integration Test

```python
@pytest.mark.asyncio
async def test_full_multi_aggregator_ab_mode():
    """Integration test: Full A/B comparison with multi-aggregator TUI."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader
    from esper.leyline import TelemetryEvent, TelemetryEventType

    app = SanctumApp()
    async with app.run_test():
        # Simulate dual-policy A/B training events
        for group_id, policy_loss, accuracy in [("A", 0.025, 72.5), ("B", 0.030, 68.0)]:
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data={
                    "group_id": group_id,
                    "policy_loss": policy_loss,
                    "entropy": 1.2,
                    "explained_variance": 0.65,
                    "accuracy": accuracy,
                },
            )
            app.handle_telemetry_event(event)

        # Should have two TamiyoBrain widgets
        widgets = list(app.query(TamiyoBrain))
        assert len(widgets) == 2

        # Each should have correct group class
        classes = [list(w.classes) for w in widgets]
        assert any("group-a" in c for c in classes)
        assert any("group-b" in c for c in classes)

        # Comparison header should be visible
        header = app.query_one(ComparisonHeader)
        assert header.display is True

        # Header should show A as leader (higher accuracy)
        assert header.leader == "A"
```

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_integration.py -v`

---

## Final Summary

### What We Built (v3)

| Phase | Component | Key Changes from v1 |
|-------|-----------|---------------------|
| 1 | TUIThresholds corrections | EV: 0.3/0.0, KL: 0.015/0.03, Advantage thresholds |
| 1 | History deques | Added kl_divergence_history, clip_fraction_history |
| 2 | Decision tree | 10+ conditions with DRL-correct priority order |
| 2 | Status banner | Added Adv:±, GradHP:, batch:N/M with denominator |
| 2 | CSS | Theme variables ($success, $warning, $error) |
| 3 | Sparklines | Empty state handling, left-padding |
| 3 | Metrics column | Full sparkline integration |
| 4 | Per-head heatmap | Correct max entropy, n/a for missing data |
| 5 | A/B Testing | Green/Cyan/Magenta color-coded Tamiyos per PolicyGroup |
| 6 | Multi-Aggregator TUI | Side-by-side A/B policy comparison (full workstream) |

### Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1.1-1.3 | Threshold corrections, schema fields, aggregator history |
| 2 | 2.1-2.7 | Decision tree, status banner, 4-gauge grid, CSS, wiring, compact mode |
| 3 | 3.1-3.3 | Sparkline renderer, metrics column, diagnostic matrix |
| 4 | 4.1-4.4 | Per-head heatmap, aggregator extraction, telemetry gap docs |
| 5 | 5.1-5.5 | A/B testing: group_id, colors, borders, banner label, aggregator |
| 6 | 6.1-6.7 | Multi-aggregator: registry, routing, layout, keyboard nav, CSS |

**Total Tasks:** 29 tasks across 6 phases

### Known Limitations

1. **Telemetry Gap:** PPOAgent only emits `slot` and `blueprint` head entropies. The other 6 heads display "n/a" until neural network changes land.

2. **80-char Terminal:** Per-head heatmap uses 3-char bars (not 4) for narrower terminal compatibility.

3. **Slot Count Dynamic:** HEAD_MAX_ENTROPIES["slot"] assumes default 3-slot config. Should be computed from SlotConfig if slot count varies.

### Verification Command

```bash
PYTHONPATH=src uv run pytest tests/karn/ -v && echo "✓ All Karn tests pass"
```
