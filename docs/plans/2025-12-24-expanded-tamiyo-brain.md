# Expanded TamiyoBrain Widget Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform TamiyoBrain from a compact diagnostic widget (~50×17) into a comprehensive PPO command center (96×24) showing all P0/P1/P2 metrics with sparklines and per-head entropy visualization.

**Architecture:** Four-phase incremental delivery: (1) Threshold corrections + core restructure, (2) Status banner + gauge grid, (3) Secondary metrics with sparklines, (4) Per-head entropy heatmap. Each phase is independently deployable.

**Tech Stack:** Textual (TUI), Rich (rendering), Python dataclasses (schema), deque (history tracking)

---

## Review Feedback Incorporated (v2 Changes)

This revision addresses feedback from four specialist reviewers:

### DRL Expert Corrections
1. **Fixed EV thresholds:** WARNING=0.3, CRITICAL=0.0 (was 0.0/-0.5)
2. **Fixed per-head max entropy values:** Now computed dynamically from action space enums
3. **Added KL thresholds:** KL_WARNING=0.015, KL_CRITICAL=0.03
4. **Fixed advantage std logic:** Added collapsed (0.1) and critical (3.0) thresholds
5. **Status banner now includes:** Adv:± summary, GradHP: summary, episode returns

### UX Specialist Corrections
1. **Status banner complete:** Now includes Adv:±, GradHP:, batch:N/M with denominator
2. **Separator width:** Now 94 chars (96 - 2 for padding), not hardcoded 48
3. **CSS uses theme variables:** `$success`, `$warning`, `$error` instead of literal colors
4. **80-char fallback:** Added compact mode detection for narrow terminals
5. **Heatmap alignment:** Fixed-width formatting for column alignment

### Risk Assessor Recommendations
1. **Split Task 1.3:** Decision tree logic now separate from banner rendering
2. **Added checkpoints:** After Task 1.4 (decision tree) and Task 2.1 (sparklines)
3. **Missing telemetry visual distinction:** Gray bars + "awaiting data" for unpopulated heads
4. **Documented rollback:** Each phase has minimum viable deliverable

---

## Rollback Strategy

| Phase | If Problems Occur | Minimum Viable Deliverable |
|-------|-------------------|---------------------------|
| 1 | Revert threshold changes, keep original constants | Updated constants only |
| 2 | Keep decision tree, disable banner rendering | Decision tree logic for status |
| 3 | Remove sparklines, show static values only | Diagnostic matrix without trends |
| 4 | Hide heatmap section entirely | Phase 3 deliverable |

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

        # A = green, B = blue, C = red
        assert "green" in widget.GROUP_COLORS["A"].lower()
        assert "blue" in widget.GROUP_COLORS["B"].lower() or "cyan" in widget.GROUP_COLORS["B"].lower()
        assert "red" in widget.GROUP_COLORS["C"].lower()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_ab_group_color_mapping -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `TamiyoBrain` class:

```python
    # A/B/C testing color scheme
    # A = Green (primary/control), B = Blue (variant), C = Red (second variant)
    GROUP_COLORS: ClassVar[dict[str, str]] = {
        "A": "bright_green",
        "B": "bright_cyan",   # Blue family
        "C": "bright_red",
    }

    GROUP_LABELS: ClassVar[dict[str, str]] = {
        "A": "🟢 Policy A",
        "B": "🔵 Policy B",
        "C": "🔴 Policy C",
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

Update `styles.tcss`:

```css
/* A/B/C Testing Group Colors */
#tamiyo-brain.group-a {
    border: solid bright_green;
    border-title-color: bright_green;
}

#tamiyo-brain.group-b {
    border: solid bright_cyan;
    border-title-color: bright_cyan;
}

#tamiyo-brain.group-c {
    border: solid bright_red;
    border-title-color: bright_red;
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_group_a_has_green_border tests/karn/sanctum/test_tamiyo_brain.py::test_group_b_has_blue_border -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): apply A/B/C group colors to TamiyoBrain borders

Green for A, Blue for B, Red for C - visual policy differentiation"
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
- Color constants: A=green, B=blue, C=red
- Dynamic border colors based on group
- Group label in status banner
- Aggregator extracts group_id from telemetry

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

## Final Summary

### What We Built (v2)

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
| 5 | A/B Testing | Green/Blue/Red color-coded Tamiyos per PolicyGroup |

### Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1.1-1.3 | Threshold corrections, schema fields, aggregator history |
| 2 | 2.1-2.6 | Decision tree, status banner, 4-gauge grid, CSS, wiring |
| 3 | 3.1-3.3 | Sparkline renderer, metrics column, diagnostic matrix |
| 4 | 4.1-4.4 | Per-head heatmap, aggregator extraction, telemetry gap docs |
| 5 | 5.1-5.5 | A/B testing: group_id, colors, borders, banner label, aggregator |

**Total Tasks:** 22 tasks across 5 phases

### Known Limitations

1. **Telemetry Gap:** PPOAgent only emits `slot` and `blueprint` head entropies. The other 6 heads display "n/a" until neural network changes land.

2. **80-char Terminal:** Per-head heatmap uses 3-char bars (not 4) for narrower terminal compatibility.

3. **Slot Count Dynamic:** HEAD_MAX_ENTROPIES["slot"] assumes default 3-slot config. Should be computed from SlotConfig if slot count varies.

4. **A/B Mode Display:** Currently shows one TamiyoBrain per aggregator. For true side-by-side A/B comparison, would need multiple aggregators or a split-pane layout (future enhancement).

### Verification Command

```bash
PYTHONPATH=src uv run pytest tests/karn/ -v && echo "✓ All Karn tests pass"
```
