# Track 5: Telemetry + Observability

**Priority:** Medium (critical for validation, not blocking core functionality)
**Estimated Effort:** 1-2 days
**Dependencies:** Track 1 (L6), Track 2 (K3)

## Overview

Telemetry is essential for validating Phase 0 success criteria. This track adds anti-thrash metrics, counterfactual deltas (observational), Karn view updates, and Sanctum widgets.

---

## O1: Add Anti-Thrash Metrics to Telemetry

**File:** `src/esper/karn/collector.py`

### Ownership Note (per DRL specialist review)

**IMPORTANT:** `InternalOpTracker` should be owned by the **training loop (Simic)**, not the telemetry collector (Karn). The collector should **receive** metrics events, not compute them. This prevents coupling between training and telemetry systems.

Recommended location: `src/esper/simic/training/internal_op_tracker.py`

### DDP Behavior (per PyTorch specialist review)

Telemetry is emitted from **all ranks** (no rank-0 gating). Each rank manages its own environments and emits events for its local state. The Karn collector's JSONL backend handles multi-rank output via separate run directories or merged streams.

### Specification

Track metrics that detect thrashing behavior:

```python
from dataclasses import dataclass, field
from collections import deque


@dataclass
class InternalOpTracker:
    """Tracks internal op patterns for anti-thrash detection.

    Maintains a rolling window of internal ops per slot to compute:
    - internal_op_density: Fraction of window filled (per DRL specialist: renamed from level_change_rate)
    - net_level_change: Cumulative direction (should converge, not oscillate)
    - reversal_rate: Fraction of ops that reverse previous op (thrash signal)
    """

    window_size: int = 50
    max_slots: int = 256  # Hard cap to prevent memory leak (per PyTorch specialist review)
    _ops_by_slot: dict[str, deque] = field(default_factory=dict)

    def record_internal_op(
        self,
        slot_id: str,
        op: LifecycleOp,
        from_level: int,
        to_level: int,
    ) -> dict[str, float]:
        """Record an internal op and compute metrics.

        Returns:
            Dict of computed metrics for this slot
        """
        # Evict oldest slot if at capacity (per PyTorch specialist review)
        if slot_id not in self._ops_by_slot and len(self._ops_by_slot) >= self.max_slots:
            oldest = next(iter(self._ops_by_slot))
            del self._ops_by_slot[oldest]

        if slot_id not in self._ops_by_slot:
            self._ops_by_slot[slot_id] = deque(maxlen=self.window_size)

        # ops contains +1 (grow) or -1 (shrink) for direction comparison
        direction = 1 if op == LifecycleOp.GROW_INTERNAL else -1
        self._ops_by_slot[slot_id].append(direction)

        return self._compute_metrics(slot_id)

    def clear_slot(self, slot_id: str) -> None:
        """Call on SEED_PRUNED/FOSSILIZED to free memory (per PyTorch specialist review)."""
        self._ops_by_slot.pop(slot_id, None)

    def reset(self) -> None:
        """Call on episode boundary to prevent memory leaks (per PyTorch specialist review)."""
        self._ops_by_slot.clear()

    def _compute_metrics(self, slot_id: str) -> dict[str, float]:
        """Compute anti-thrash metrics for a slot."""
        ops = self._ops_by_slot[slot_id]

        # Need at least 2 ops to compute reversal rate (per Python specialist review)
        if len(ops) < 2:
            return {
                "internal_op_density": len(ops) / self.window_size,  # Can be 1/window_size
                "net_level_change": float(sum(ops)) if ops else 0.0,
                "reversal_rate": 0.0,
            }

        # Internal op density: fraction of window filled (renamed per DRL specialist review)
        # Measures how active the policy is at adjusting levels
        internal_op_density = len(ops) / self.window_size

        # Net level change: sum of directions
        # Positive = net growth, negative = net shrink, near-zero = oscillation
        net_level_change = sum(ops)

        # Reversal rate: fraction of consecutive ops with different directions
        # ops contains +1 (grow) or -1 (shrink), so != detects direction changes
        reversals = sum(
            1 for i in range(1, len(ops))
            if ops[i] != ops[i-1]
        )
        reversal_rate = reversals / (len(ops) - 1)  # Safe: len >= 2

        return {
            "internal_op_density": internal_op_density,
            "net_level_change": float(net_level_change),
            "reversal_rate": reversal_rate,
        }


class TelemetryCollector:
    def __init__(self):
        # ... existing init ...
        self._internal_op_tracker = InternalOpTracker()

    def on_seed_internal_level_changed(
        self,
        event: TelemetryEvent,
    ) -> None:
        """Handle SEED_INTERNAL_LEVEL_CHANGED events."""
        payload = event.payload

        # Record the op
        metrics = self._internal_op_tracker.record_internal_op(
            slot_id=payload.slot_id,
            op=LifecycleOp.GROW_INTERNAL if payload.to_level > payload.from_level
               else LifecycleOp.SHRINK_INTERNAL,
            from_level=payload.from_level,
            to_level=payload.to_level,
        )

        # Emit metrics event
        self._emit_metrics_event(
            event_type=TelemetryEventType.INTERNAL_OP_METRICS,
            payload=InternalOpMetricsPayload(
                slot_id=payload.slot_id,
                env_id=payload.env_id,
                **metrics,
            ),
        )
```

### Metrics Definitions (per DRL specialist review)

| Metric | Formula | Healthy Range | Thrash Signal |
|--------|---------|---------------|---------------|
| `internal_op_density` | `n_ops / window_size` | 0.1 - 0.4 | >0.6 (too frequent) |
| `net_level_change` | `Σ direction` | Converges to stable value | Oscillates around 0 |
| `reversal_rate` | `n_reversals / (n_ops - 1)` | <0.3 | >0.7 (constant flip-flop) |

**Threshold Notes (per DRL specialist review):**
- 0.7 reversal_rate is a "clear thrash" threshold
- 0.5-0.7 is a "caution zone" for monitoring
- Healthy exploration during early training might show 0.4-0.6 reversal_rate

### Acceptance Criteria
- [ ] `InternalOpTracker` tracks ops per slot
- [ ] `internal_op_density` computed correctly (renamed from level_change_rate)
- [ ] `net_level_change` computed correctly
- [ ] `reversal_rate` computed correctly (division-safe for len < 2)
- [ ] Metrics emitted as telemetry events
- [ ] Metrics available in Karn views
- [ ] **Lifecycle cleanup:** `reset()` called on episode boundary (per PyTorch specialist review)
- [ ] **Lifecycle cleanup:** `clear_slot()` called on SEED_PRUNED/FOSSILIZED (per PyTorch specialist review)
- [ ] **Ownership:** Tracker owned by Simic, not Karn (per DRL specialist review)

---

## O2: Add Counterfactual Delta Telemetry (Observational)

**File:** `src/esper/karn/collector.py`

### RunningMean Helper Class (per Python specialist review)

The `RunningMean` class must be defined (doesn't exist in codebase):

```python
@dataclass
class RunningMean:
    """Simple exponential moving average for scalar baselines (per DRL specialist review)."""

    alpha: float = 0.01  # EMA decay rate
    _value: float = 0.0
    _count: int = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def count(self) -> int:
        """Sample count for baseline warmth check (per DRL specialist review)."""
        return self._count

    def update(self, x: float) -> None:
        if self._count == 0:
            self._value = x
        else:
            self._value = (1 - self.alpha) * self._value + self.alpha * x
        self._count += 1

    def reset(self) -> None:
        """Reset for new episode."""
        self._value = 0.0
        self._count = 0
```

### Baseline Bias Note (per DRL specialist review)

**IMPORTANT:** The NOOP baseline approach has a known structural bias:
- NOOPs tend to occur when already near-optimal (small deltas)
- GROW/SHRINK ops tend to occur when suboptimal (larger deltas)
- This inflates "advantage" measurements

**Acceptable for Phase 0** since this is purely observational, not used for reward shaping.
For future phases, proper counterfactual estimation would require importance sampling or randomized interventions.

### Specification

Log observational data for future counterfactual analysis:

```python
@dataclass(frozen=True, slots=True)
class CounterfactualDeltaPayload:
    """Observational payload for counterfactual analysis.

    Records what happened after an internal op, along with
    counterfactual baseline (what would have happened with NOOP).

    NOTE: Phase 0 does NOT use this for reward shaping. It's purely
    observational telemetry for offline analysis.
    """
    slot_id: str
    env_id: int
    epoch: int

    # The op that was executed
    op: int  # LifecycleOp.value
    from_level: int
    to_level: int

    # State before op
    loss_before: float
    accuracy_before: float
    params_before: int

    # State after op (1 epoch later)
    loss_after: float
    accuracy_after: float
    params_after: int

    # Delta (actual - expected)
    loss_delta: float
    accuracy_delta: float

    # Baseline (what NOOP would have given, estimated)
    baseline_loss_delta: float  # Running average of NOOP deltas
    baseline_accuracy_delta: float

    # Baseline warmth indicator (per DRL specialist review)
    # Analysts should filter by baseline_sample_count >= 10 to avoid confounded early data
    baseline_sample_count: int


class CounterfactualTracker:
    """Tracks outcomes for counterfactual analysis.

    Records state before internal ops and computes deltas
    after one epoch. Also maintains a baseline estimate
    from NOOP decisions.
    """

    def __init__(self):
        self._pending_ops: dict[tuple[str, int], dict] = {}  # (slot_id, env_id) -> state
        self._baseline_loss_delta = RunningMean()
        self._baseline_accuracy_delta = RunningMean()

    def clear_env(self, env_id: int) -> None:
        """Clear pending ops for env on episode reset (per PyTorch specialist review)."""
        keys_to_remove = [k for k in self._pending_ops if k[1] == env_id]
        for key in keys_to_remove:
            del self._pending_ops[key]

    def reset(self) -> None:
        """Full reset on training restart (per PyTorch specialist review)."""
        self._pending_ops.clear()
        self._baseline_loss_delta = RunningMean()
        self._baseline_accuracy_delta = RunningMean()

    def record_op_start(
        self,
        slot_id: str,
        env_id: int,
        epoch: int,
        op: LifecycleOp,
        from_level: int,
        to_level: int,
        loss: float,
        accuracy: float,
        params: int,
    ) -> None:
        """Record state at start of internal op."""
        key = (slot_id, env_id)
        self._pending_ops[key] = {
            "epoch": epoch,
            "op": op,
            "from_level": from_level,
            "to_level": to_level,
            "loss_before": loss,
            "accuracy_before": accuracy,
            "params_before": params,
        }

    def record_op_outcome(
        self,
        slot_id: str,
        env_id: int,
        loss: float,
        accuracy: float,
        params: int,
    ) -> CounterfactualDeltaPayload | None:
        """Record outcome after one epoch, return payload if op was pending."""
        key = (slot_id, env_id)
        if key not in self._pending_ops:
            return None

        state = self._pending_ops.pop(key)

        loss_delta = loss - state["loss_before"]
        accuracy_delta = accuracy - state["accuracy_before"]

        return CounterfactualDeltaPayload(
            slot_id=slot_id,
            env_id=env_id,
            epoch=state["epoch"],
            op=state["op"].value,
            from_level=state["from_level"],
            to_level=state["to_level"],
            loss_before=state["loss_before"],
            accuracy_before=state["accuracy_before"],
            params_before=state["params_before"],
            loss_after=loss,
            accuracy_after=accuracy,
            params_after=params,
            loss_delta=loss_delta,
            accuracy_delta=accuracy_delta,
            baseline_loss_delta=self._baseline_loss_delta.value,
            baseline_accuracy_delta=self._baseline_accuracy_delta.value,
            # Include sample count so analysts can filter early data (per DRL specialist review)
            baseline_sample_count=self._baseline_loss_delta.count,
        )

    def record_noop_outcome(
        self,
        loss_delta: float,
        accuracy_delta: float,
    ) -> None:
        """Update baseline estimates from NOOP outcomes."""
        self._baseline_loss_delta.update(loss_delta)
        self._baseline_accuracy_delta.update(accuracy_delta)
```

### Usage Pattern

```python
# In training loop:

# When internal op is selected
if action.lifecycle_op in {LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL}:
    tracker.record_op_start(
        slot_id=slot_id,
        env_id=env_idx,
        epoch=current_epoch,
        op=action.lifecycle_op,
        from_level=slot.internal_level,
        to_level=slot.internal_level + (1 if GROW else -1),
        loss=current_loss,
        accuracy=current_accuracy,
        params=slot.active_param_count(),
    )

# After each epoch, check for pending outcomes
payload = tracker.record_op_outcome(
    slot_id=slot_id,
    env_id=env_idx,
    loss=new_loss,
    accuracy=new_accuracy,
    params=slot.active_param_count(),
)
if payload:
    emit_telemetry(TelemetryEventType.COUNTERFACTUAL_DELTA, payload)
```

### Acceptance Criteria
- [ ] `CounterfactualTracker` records op start state
- [ ] Outcome recorded after 1 epoch
- [ ] Baseline estimates updated from NOOP outcomes
- [ ] Payload includes all required fields including `baseline_sample_count`
- [ ] Data is **observational only** (not used for reward shaping in Phase 0)
- [ ] **Lifecycle cleanup:** `clear_env()` called on episode reset per env (per PyTorch specialist review)
- [ ] **Lifecycle cleanup:** `reset()` called on training restart (per PyTorch specialist review)
- [ ] `RunningMean` class defined in codebase (per Python specialist review)

---

## O3: Update Karn Views for Internal Level Events

**File:** `src/esper/karn/mcp/views.py`

### Specification

Add views for querying internal level telemetry:

```python
# View: seed_internal_levels
SEED_INTERNAL_LEVELS_VIEW = """
CREATE VIEW IF NOT EXISTS seed_internal_levels AS
SELECT
    run_dir,
    timestamp,
    env_id,
    slot_id,
    blueprint_id,
    internal_kind,
    from_level,
    to_level,
    max_level,
    active_params,
    to_level - from_level AS level_delta
FROM raw_events
WHERE event_type = 'SEED_INTERNAL_LEVEL_CHANGED'
ORDER BY timestamp;
"""

# View: internal_op_summary
INTERNAL_OP_SUMMARY_VIEW = """
CREATE VIEW IF NOT EXISTS internal_op_summary AS
SELECT
    run_dir,
    slot_id,
    COUNT(*) AS total_ops,
    SUM(CASE WHEN level_delta > 0 THEN 1 ELSE 0 END) AS grow_count,
    SUM(CASE WHEN level_delta < 0 THEN 1 ELSE 0 END) AS shrink_count,
    SUM(level_delta) AS net_level_change,
    AVG(active_params) AS avg_active_params,
    MAX(to_level) AS max_reached_level,
    MIN(to_level) AS min_reached_level
FROM seed_internal_levels
GROUP BY run_dir, slot_id;
"""

# View: internal_op_metrics (from O1)
INTERNAL_OP_METRICS_VIEW = """
CREATE VIEW IF NOT EXISTS internal_op_metrics AS
SELECT
    run_dir,
    timestamp,
    env_id,
    slot_id,
    internal_op_density,  -- renamed from level_change_rate per DRL specialist
    net_level_change,
    reversal_rate
FROM raw_events
WHERE event_type = 'INTERNAL_OP_METRICS'
ORDER BY timestamp;
"""

# View: counterfactual_deltas (from O2)
COUNTERFACTUAL_DELTAS_VIEW = """
CREATE VIEW IF NOT EXISTS counterfactual_deltas AS
SELECT
    run_dir,
    timestamp,
    env_id,
    slot_id,
    epoch,
    op,
    from_level,
    to_level,
    loss_delta,
    accuracy_delta,
    baseline_loss_delta,
    baseline_accuracy_delta,
    baseline_sample_count,  -- Added per DRL specialist review
    loss_delta - baseline_loss_delta AS loss_advantage,
    accuracy_delta - baseline_accuracy_delta AS accuracy_advantage
FROM raw_events
WHERE event_type = 'COUNTERFACTUAL_DELTA'
ORDER BY timestamp;
"""
```

### Example Queries

```sql
-- Check if internal ops are being used (Success Criteria #1)
SELECT
    run_dir,
    COUNT(*) as internal_ops,
    (SELECT COUNT(*) FROM decisions WHERE run_dir = s.run_dir AND op != 'NOOP') as total_decisions,
    ROUND(100.0 * COUNT(*) / NULLIF((SELECT COUNT(*) FROM decisions WHERE run_dir = s.run_dir AND op != 'NOOP'), 0), 1) as pct
FROM seed_internal_levels s
GROUP BY run_dir;

-- Check for thrash (Success Criteria #4)
SELECT
    slot_id,
    AVG(reversal_rate) as avg_reversal_rate,
    MAX(reversal_rate) as max_reversal_rate,
    COUNT(*) as samples
FROM internal_op_metrics
WHERE run_dir = ?
GROUP BY slot_id
HAVING AVG(reversal_rate) > 0.5;  -- Thrash warning threshold

-- Counterfactual analysis: do GROW ops help?
-- Filter by baseline_sample_count >= 10 to exclude early confounded data (per DRL specialist)
SELECT
    op,
    COUNT(*) as n,
    AVG(loss_advantage) as avg_loss_advantage,
    AVG(accuracy_advantage) as avg_accuracy_advantage
FROM counterfactual_deltas
WHERE run_dir = ?
  AND baseline_sample_count >= 10  -- Baseline warmth filter
GROUP BY op;
```

### Acceptance Criteria
- [ ] `seed_internal_levels` view created
- [ ] `internal_op_summary` view created
- [ ] `internal_op_metrics` view created
- [ ] `counterfactual_deltas` view created
- [ ] Views registered in MCP server
- [ ] Example queries work correctly

---

## O4: Add Sanctum Widget for Internal Level Display

**File:** `src/esper/karn/sanctum/widgets/internal_level.py` (new file)

### Integration Pattern Note (per PyTorch specialist review)

Sanctum uses a **polling model** via `_poll_and_refresh()`, not direct event handlers.
Events are processed by `AggregatorRegistry`, not directly by the app.

Follow existing pattern:
1. Widget reads from `SanctumSnapshot` during refresh
2. Aggregator handles telemetry events and updates aggregated state
3. Snapshot is read by widgets during periodic refresh

**Do NOT** add event handlers directly to `SanctumApp`. Use the aggregator pattern.

### State Type Definition (per Python specialist review)

Use a typed dataclass instead of raw dicts to avoid defensive `.get()` patterns:

```python
from dataclasses import dataclass

@dataclass
class SlotInternalState:
    """Typed state for internal level display."""
    level: int
    max_level: int
    trend: str  # "grow", "shrink", or "stable"
    reversal_rate: float
```

### Specification

Create a TUI widget showing internal level state per slot:

```python
from textual.widgets import Static
from rich.table import Table
from rich.text import Text


class InternalLevelWidget(Static):
    """Widget displaying internal level state for all slots.

    Shows:
    - Current level / max level per slot
    - Visual bar representation
    - Recent op direction (▲ grow, ▼ shrink, ─ stable)
    - Anti-thrash status (green/yellow/red)

    NOTE: Uses explicit refresh() pattern per existing codebase style (RewardHealthPanel),
    not Textual reactive descriptors (per Python specialist review).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._slot_states: dict[str, SlotInternalState] = {}

    def render(self) -> Table:
        table = Table(title="Internal Levels", box=None)
        table.add_column("Slot", style="cyan")
        table.add_column("Level", justify="center")
        table.add_column("Bar", justify="left")
        table.add_column("Trend", justify="center")
        table.add_column("Thrash", justify="center")

        for slot_id, state in sorted(self._slot_states.items()):
            # Direct attribute access, no defensive .get() (per Python specialist review)
            level = state.level
            max_level = state.max_level
            trend = state.trend
            reversal_rate = state.reversal_rate

            # Level text
            level_text = f"{level}/{max_level}"

            # Visual bar
            filled = "█" * level
            empty = "░" * (max_level - level)
            bar = Text(filled, style="green") + Text(empty, style="dim")

            # Trend indicator
            trend_icons = {
                "grow": Text("▲", style="green"),
                "shrink": Text("▼", style="red"),
                "stable": Text("─", style="dim"),
            }
            trend_text = trend_icons.get(trend, "─")

            # Thrash status
            if reversal_rate < 0.3:
                thrash_text = Text("●", style="green")
            elif reversal_rate < 0.6:
                thrash_text = Text("●", style="yellow")
            else:
                thrash_text = Text("●", style="red bold")

            table.add_row(
                slot_id,
                level_text,
                bar,
                trend_text,
                thrash_text,
            )

        return table

    def update_slot(
        self,
        slot_id: str,
        level: int,
        max_level: int,
        trend: str,
        reversal_rate: float,
    ) -> None:
        """Update state for a slot and trigger refresh (per Python specialist review)."""
        self._slot_states[slot_id] = SlotInternalState(
            level=level,
            max_level=max_level,
            trend=trend,
            reversal_rate=reversal_rate,
        )
        self.refresh()  # Explicit refresh per existing codebase pattern
```

### Widget Integration (per PyTorch specialist review)

Use the **aggregator pattern**, NOT direct event handlers in `SanctumApp`:

```python
# In src/esper/karn/sanctum/aggregator.py

class InternalLevelAggregator:
    """Aggregator for internal level telemetry events."""

    def __init__(self) -> None:
        self._slot_states: dict[str, SlotInternalState] = {}

    def handle_seed_internal_level_changed(
        self, event: TelemetryEvent
    ) -> None:
        """Handle SEED_INTERNAL_LEVEL_CHANGED events."""
        payload = event.data

        # Determine trend
        if payload.to_level > payload.from_level:
            trend = "grow"
        elif payload.to_level < payload.from_level:
            trend = "shrink"
        else:
            trend = "stable"

        # Get existing reversal_rate if present
        existing = self._slot_states.get(payload.slot_id)
        reversal_rate = existing.reversal_rate if existing else 0.0

        self._slot_states[payload.slot_id] = SlotInternalState(
            level=payload.to_level,
            max_level=payload.max_level,
            trend=trend,
            reversal_rate=reversal_rate,
        )

    def handle_internal_op_metrics(self, event: TelemetryEvent) -> None:
        """Handle INTERNAL_OP_METRICS events."""
        payload = event.data

        if payload.slot_id in self._slot_states:
            existing = self._slot_states[payload.slot_id]
            self._slot_states[payload.slot_id] = SlotInternalState(
                level=existing.level,
                max_level=existing.max_level,
                trend=existing.trend,
                reversal_rate=payload.reversal_rate,
            )

    def to_snapshot(self) -> dict[str, SlotInternalState]:
        """Return snapshot for widget consumption."""
        return dict(self._slot_states)


# In src/esper/karn/sanctum/app.py

from .widgets.internal_level import InternalLevelWidget

class SanctumApp(App):
    def compose(self) -> ComposeResult:
        # ... existing widgets ...
        yield InternalLevelWidget(id="internal-levels")

    def _poll_and_refresh(self) -> None:
        # ... existing refresh logic ...

        # Update internal levels widget from aggregator snapshot
        internal_levels = self.query_one("#internal-levels", InternalLevelWidget)
        snapshot = self._aggregator.internal_level_aggregator.to_snapshot()
        for slot_id, state in snapshot.items():
            internal_levels.update_slot(
                slot_id=slot_id,
                level=state.level,
                max_level=state.max_level,
                trend=state.trend,
                reversal_rate=state.reversal_rate,
            )
```

### Visual Design

```
┌─ Internal Levels ─────────────────────────────────┐
│ Slot   Level   Bar          Trend   Thrash       │
│ r0c0   2/4     ██░░         ▲       ●            │
│ r0c1   0/4     ░░░░         ▼       ●            │
│ r0c2   4/4     ████         ─       ●            │
│ r0c3   1/4     █░░░         ▲       ●            │
│ r0c4   3/4     ███░         ─       ●            │
└───────────────────────────────────────────────────┘
```

### Acceptance Criteria
- [ ] Widget displays level/max_level per slot
- [ ] Visual bar shows level graphically
- [ ] Trend indicator shows recent direction
- [ ] Thrash indicator shows reversal_rate status
- [ ] **Uses aggregator pattern, NOT direct event handlers** (per PyTorch specialist review)
- [ ] **Uses typed `SlotInternalState` dataclass, no defensive `.get()`** (per Python specialist review)
- [ ] **Uses explicit `refresh()` pattern** (per Python specialist review)
- [ ] Integrated into Sanctum dashboard via `SanctumSnapshot`

---

## Testing Requirements

### Unit Tests (`tests/karn/`)

**test_collector.py:**
```python
def test_internal_op_tracker_op_density():
    """Verify internal_op_density computation (renamed from level_change_rate)."""
    tracker = InternalOpTracker(window_size=10)

    # Record 5 ops
    for i in range(5):
        tracker.record_internal_op("r0c0", LifecycleOp.GROW_INTERNAL, i, i+1)

    metrics = tracker._compute_metrics("r0c0")
    assert metrics["internal_op_density"] == 0.5  # 5/10

def test_internal_op_tracker_reversal_rate():
    """Verify reversal_rate detects thrash."""
    tracker = InternalOpTracker(window_size=10)

    # Oscillate: GROW, SHRINK, GROW, SHRINK
    ops = [LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL] * 3
    for i, op in enumerate(ops):
        tracker.record_internal_op("r0c0", op, i % 2, (i + 1) % 2)

    metrics = tracker._compute_metrics("r0c0")
    assert metrics["reversal_rate"] > 0.8  # High reversal = thrash

def test_counterfactual_tracker():
    """Verify counterfactual tracking."""
    tracker = CounterfactualTracker()

    # Record op start
    tracker.record_op_start(
        slot_id="r0c0",
        env_id=0,
        epoch=10,
        op=LifecycleOp.GROW_INTERNAL,
        from_level=1,
        to_level=2,
        loss=0.5,
        accuracy=0.7,
        params=1000,
    )

    # Record outcome
    payload = tracker.record_op_outcome(
        slot_id="r0c0",
        env_id=0,
        loss=0.4,
        accuracy=0.75,
        params=1500,
    )

    assert payload is not None
    assert payload.loss_delta == pytest.approx(-0.1)
    assert payload.accuracy_delta == pytest.approx(0.05)
```

**test_views.py:**
```python
def test_seed_internal_levels_view():
    """Verify seed_internal_levels view returns expected columns."""
    conn = create_test_db_with_events()
    result = conn.execute("SELECT * FROM seed_internal_levels LIMIT 1").fetchone()

    assert "slot_id" in result.keys()
    assert "from_level" in result.keys()
    assert "to_level" in result.keys()
    assert "level_delta" in result.keys()

def test_internal_op_summary_aggregation():
    """Verify internal_op_summary aggregates correctly."""
    conn = create_test_db_with_events()

    # Insert some test events
    # ...

    result = conn.execute("""
        SELECT * FROM internal_op_summary
        WHERE slot_id = 'r0c0'
    """).fetchone()

    assert result["total_ops"] > 0
    assert result["grow_count"] >= 0
    assert result["shrink_count"] >= 0
```

**test_sanctum_widgets.py:**
```python
def test_internal_level_widget_render():
    """Verify widget renders correctly (per Python specialist: use typed state)."""
    widget = InternalLevelWidget()
    widget._slot_states = {
        "r0c0": SlotInternalState(level=2, max_level=4, trend="grow", reversal_rate=0.1),
        "r0c1": SlotInternalState(level=0, max_level=4, trend="shrink", reversal_rate=0.8),
    }

    table = widget.render()
    assert "r0c0" in str(table)
    assert "r0c1" in str(table)

def test_internal_level_widget_update():
    """Verify widget updates on new data (per Python specialist: use typed state)."""
    widget = InternalLevelWidget()

    widget.update_slot("r0c0", level=2, max_level=4, trend="grow", reversal_rate=0.1)

    assert "r0c0" in widget._slot_states
    assert widget._slot_states["r0c0"].level == 2  # Direct attribute access
```
