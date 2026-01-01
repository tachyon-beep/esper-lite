# Nest Reward Components in Analytics Payload (Hard Cutover)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate manual field-by-field copying between `RewardComponentsTelemetry` and `AnalyticsSnapshotPayload` by passing the typed dataclass directly.

**Architecture:** Add `reward_components: RewardComponentsTelemetry | None` to `AnalyticsSnapshotPayload`. Pass the object from vectorized.py through emitters. MCP views and aggregator read from nested structure. Hard cutover - no backwards compatibility.

**Tech Stack:** Python dataclasses, DuckDB JSON extraction, pytest

**Risk Acceptance:** Fix on fail or rollback. Old telemetry data won't be queryable via MCP until re-emitted.

---

## Task 1: Add Typed `reward_components` Field to Payload

**Files:**
- Modify: `src/esper/leyline/telemetry.py:1050-1060`
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write failing test for typed field**

```python
# tests/leyline/test_telemetry.py

def test_analytics_snapshot_payload_accepts_reward_components_dataclass():
    """AnalyticsSnapshotPayload should accept RewardComponentsTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=2,
        action_shaping=0.05,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        reward_components=rc,
    )

    assert payload.reward_components is rc
    assert payload.reward_components.seed_stage == 2
    assert payload.reward_components.bounded_attribution == 0.5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_analytics_snapshot_payload_accepts_reward_components_dataclass -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'reward_components'`

**Step 3: Add typed field to AnalyticsSnapshotPayload**

In `src/esper/leyline/telemetry.py`:

1. Add import at top (around line 30):
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
```

2. Add field after line 1057 (after `alpha_shock`):
```python
    # Full reward components dataclass (replaces individual fields)
    reward_components: "RewardComponentsTelemetry | None" = None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_analytics_snapshot_payload_accepts_reward_components_dataclass -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add typed reward_components field to AnalyticsSnapshotPayload"
```

---

## Task 2: Atomic Update - Emitter + Vectorized (Clean Break)

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py:187-340`
- Modify: `src/esper/simic/training/vectorized.py:2943-2976`
- Test: `tests/simic/telemetry/test_emitters.py`

**Step 1: Write test for new emitter signature**

```python
# tests/simic/telemetry/test_emitters.py

def test_on_last_action_accepts_reward_components_dataclass(mock_collector):
    """on_last_action should accept RewardComponentsTelemetry directly."""
    from esper.simic.telemetry.emitters import VectorizedEmitter
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    emitter = VectorizedEmitter(env_id=0, slot_ids=["G0", "G1", "G2"])

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=2,
        total_reward=0.35,
    )

    # Should not raise
    emitter.on_last_action(
        epoch=10,
        action_dict={"op": 0, "slot": 0, "blueprint": 0, "style": 0, "tempo": 0, "alpha_target": 0, "alpha_speed": 0, "alpha_curve": 0},
        target_slot="G0",
        masked_flags={},
        action_success=True,
        active_algo="curiosity",
        total_reward=0.35,
        value_estimate=0.3,
        host_accuracy=0.75,
        reward_components=rc,
    )

    # Verify event was emitted with typed dataclass
    from esper.karn import collector
    events = [e for e in mock_collector.events if e.event_type.name == "ANALYTICS_SNAPSHOT"]
    assert len(events) >= 1
    payload = events[-1].data
    assert payload.reward_components is rc
    assert payload.reward_components.seed_stage == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_on_last_action_accepts_reward_components_dataclass -v`

Expected: FAIL

**Step 3: Update emitter signature (REMOVE old params, ADD new)**

In `src/esper/simic/telemetry/emitters.py`, update `on_last_action` method signature.

REMOVE these parameters:
- `base_acc_delta: float | None = None`
- `bounded_attribution: float | None = None`
- `compute_rent: float | None = None`
- `stage_bonus: float | None = None`
- `ratio_penalty: float | None = None`
- `alpha_shock: float | None = None`

ADD this parameter:
```python
    reward_components: "RewardComponentsTelemetry | None" = None,
```

Add import at top:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
```

Update payload construction (around line 320) to pass the dataclass:
```python
        payload = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=self.env_id,
            total_reward=total_reward,
            action_name=op_name,
            # ... other fields ...
            reward_components=reward_components,  # Pass typed dataclass
        )
```

**Step 4: Fix `"reward_components" in locals()` code smell**

The variable `reward_components` is only created inside the `RewardFamily.CONTRIBUTION` branch.
Currently the code uses `"reward_components" in locals()` to check if we're in that branch - this is a code smell.

**Fix:** Initialize `reward_components = None` before the conditional, then use simple `is not None` checks.

In `src/esper/simic/training/vectorized.py`, find line ~2650 (before the `if reward_family_enum == RewardFamily.CONTRIBUTION:` block).

ADD this line before the conditional:
```python
                reward_components: RewardComponentsTelemetry | None = None
```

Add the import at top of file (with other TYPE_CHECKING imports):
```python
if TYPE_CHECKING:
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
```

**Step 5: Replace all `in locals()` checks with `is not None`**

Find and replace these 3 occurrences:

1. **Line ~2710** - Change:
```python
if collect_reward_summary and "reward_components" in locals():
```
To:
```python
if collect_reward_summary and reward_components is not None:
```

2. **Line ~2717** - Change:
```python
if collect_reward_summary and "reward_components" in locals():
```
To:
```python
if collect_reward_summary and reward_components is not None:
```

3. **Line ~2950** - Change:
```python
if "reward_components" in locals() and reward_components is not None:
```
To:
```python
if reward_components is not None:
```

**Step 6: Update vectorized.py to pass dataclass directly**

In `src/esper/simic/training/vectorized.py`, replace the telemetry extraction block (lines ~2943-2976).

DELETE this block:
```python
bounded_attribution_for_telemetry = None
compute_rent_for_telemetry = None
stage_bonus_for_telemetry = None
ratio_penalty_for_telemetry = None
alpha_shock_for_telemetry = None
if "reward_components" in locals() and reward_components is not None:
    bounded_attribution_for_telemetry = reward_components.bounded_attribution
    compute_rent_for_telemetry = reward_components.compute_rent
    stage_bonus_for_telemetry = reward_components.stage_bonus
    ratio_penalty_for_telemetry = reward_components.ratio_penalty
    alpha_shock_for_telemetry = reward_components.alpha_shock
emitters[env_idx].on_last_action(
    epoch,
    action_dict,
    target_slot,
    masked_flags,
    action_success,
    active_algo,
    total_reward=reward,
    value_estimate=value,
    host_accuracy=env_state.val_acc,
    slot_states=decision_slot_states,
    action_confidence=action_confidence,
    alternatives=alternatives,
    decision_entropy=decision_entropy,
    base_acc_delta=base_acc_delta_for_telemetry,
    bounded_attribution=bounded_attribution_for_telemetry,
    compute_rent=compute_rent_for_telemetry,
    stage_bonus=stage_bonus_for_telemetry,
    ratio_penalty=ratio_penalty_for_telemetry,
    alpha_shock=alpha_shock_for_telemetry,
)
```

REPLACE WITH:
```python
                    emitters[env_idx].on_last_action(
                        epoch,
                        action_dict,
                        target_slot,
                        masked_flags,
                        action_success,
                        active_algo,
                        total_reward=reward,
                        value_estimate=value,
                        host_accuracy=env_state.val_acc,
                        slot_states=decision_slot_states,
                        action_confidence=action_confidence,
                        alternatives=alternatives,
                        decision_entropy=decision_entropy,
                        reward_components=reward_components,  # Pass directly (may be None for LOSS family)
                    )
```

**Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_on_last_action_accepts_reward_components_dataclass -v`

Expected: PASS

**Step 8: Run existing tests to catch breakage**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v --tb=short -x`

Fix any failures before proceeding.

**Step 9: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py src/esper/simic/training/vectorized.py tests/simic/telemetry/test_emitters.py
git commit -m "feat(emitters): pass RewardComponentsTelemetry directly, remove individual params, fix locals() smell"
```

---

## Task 3: Update MCP View (No Fallbacks)

**Files:**
- Modify: `src/esper/karn/mcp/views.py:121-146`
- Test: `tests/karn/mcp/test_views.py`

**Step 1: Write test for nested query**

```python
# tests/karn/mcp/test_views.py

def test_rewards_view_extracts_nested_seed_stage(tmp_path):
    """rewards view should extract seed_stage from nested reward_components."""
    import json
    import duckdb
    from datetime import datetime
    from esper.karn.mcp.views import create_views

    # Create test telemetry with nested structure
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "test-1",
        "event_type": "ANALYTICS_SNAPSHOT",
        "timestamp": datetime.now().isoformat(),
        "epoch": 10,
        "data": {
            "kind": "last_action",
            "env_id": 0,
            "total_reward": 0.5,
            "reward_components": {
                "seed_stage": 2,
                "action_shaping": 0.05,
                "bounded_attribution": 0.3,
                "compute_rent": -0.1,
                "stage_bonus": 0.1,
                "ratio_penalty": 0.0,
                "terminal_bonus": 0.0,
                "base_acc_delta": 0.02,
                "val_acc": 0.75,
                "num_fossilized_seeds": 1,
            }
        }
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    result = conn.execute("""
        SELECT seed_stage, action_shaping, bounded_attribution, val_acc
        FROM rewards
    """).fetchone()

    assert result is not None
    assert result[0] == 2  # seed_stage
    assert result[1] == 0.05  # action_shaping
    assert result[2] == 0.3  # bounded_attribution
    assert result[3] == 0.75  # val_acc
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/mcp/test_views.py::test_rewards_view_extracts_nested_seed_stage -v`

Expected: FAIL (seed_stage NULL, wrong paths)

**Step 3: Update rewards view SQL (hard cutover, no COALESCE)**

In `src/esper/karn/mcp/views.py`, replace the `rewards` view (lines 121-146):

```python
    "rewards": """
        CREATE OR REPLACE VIEW rewards AS
        SELECT
            timestamp,
            epoch,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.episode')::INTEGER as episode,
            json_extract_string(data, '$.ab_group') as ab_group,
            json_extract_string(data, '$.action_name') as action_name,
            json_extract(data, '$.action_success')::BOOLEAN as action_success,
            -- All reward fields now nested under reward_components
            json_extract(data, '$.reward_components.seed_stage')::INTEGER as seed_stage,
            json_extract(data, '$.total_reward')::DOUBLE as total_reward,
            json_extract(data, '$.reward_components.base_acc_delta')::DOUBLE as base_acc_delta,
            json_extract(data, '$.reward_components.bounded_attribution')::DOUBLE as bounded_attribution,
            json_extract(data, '$.reward_components.ratio_penalty')::DOUBLE as ratio_penalty,
            json_extract(data, '$.reward_components.compute_rent')::DOUBLE as compute_rent,
            json_extract(data, '$.reward_components.stage_bonus')::DOUBLE as stage_bonus,
            json_extract(data, '$.reward_components.action_shaping')::DOUBLE as action_shaping,
            json_extract(data, '$.reward_components.terminal_bonus')::DOUBLE as terminal_bonus,
            json_extract(data, '$.reward_components.val_acc')::DOUBLE as val_acc,
            json_extract(data, '$.reward_components.num_fossilized_seeds')::INTEGER as num_fossilized_seeds,
            -- New fields now available (were missing before)
            json_extract(data, '$.reward_components.alpha_shock')::DOUBLE as alpha_shock,
            json_extract(data, '$.reward_components.hindsight_credit')::DOUBLE as hindsight_credit,
            json_extract(data, '$.reward_components.synergy_bonus')::DOUBLE as synergy_bonus,
            json_extract(data, '$.reward_components.fossilize_terminal_bonus')::DOUBLE as fossilize_terminal_bonus,
            json_extract(data, '$.reward_components.growth_ratio')::DOUBLE as growth_ratio
        FROM raw_events
        WHERE
            event_type = 'ANALYTICS_SNAPSHOT'
            AND json_extract_string(data, '$.kind') = 'last_action'
    """,
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/mcp/test_views.py::test_rewards_view_extracts_nested_seed_stage -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/mcp/views.py tests/karn/mcp/test_views.py
git commit -m "feat(mcp): update rewards view to query nested reward_components (hard cutover)"
```

---

## Task 4: Update Aggregator (Direct Field Access)

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:1166-1191`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write test for direct dataclass access**

```python
# tests/karn/sanctum/test_aggregator.py

def test_aggregator_reads_reward_components_dataclass():
    """Aggregator should read from nested RewardComponentsTelemetry."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import TelemetryEvent, AnalyticsSnapshotPayload, TelemetryEventType
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    agg = SanctumAggregator(num_envs=1)
    agg._connected = True
    agg._ensure_env(0)

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.3,
        compute_rent=-0.05,
        stage_bonus=0.1,
        ratio_penalty=0.0,
        alpha_shock=0.0,
        base_acc_delta=0.02,
        hindsight_credit=0.05,
        total_reward=0.42,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.42,
        action_name="WAIT",
        action_confidence=0.8,
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    agg.process_event(event)

    env = agg._envs[0]
    assert env.reward_components.bounded_attribution == 0.3
    assert env.reward_components.compute_rent == -0.05
    assert env.reward_components.stage_bonus == 0.1
    assert env.reward_components.hindsight_credit == 0.05
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_reads_reward_components_dataclass -v`

Expected: FAIL (values are 0.0)

**Step 3: Update aggregator to read from typed dataclass**

In `src/esper/karn/sanctum/aggregator.py`, update `_handle_analytics_snapshot` (around line 1166).

REPLACE the reward component extraction block with:

```python
            # Update reward component breakdown from typed dataclass
            rc = payload.reward_components
            if rc is not None:
                env.reward_components.base_acc_delta = rc.base_acc_delta
                env.reward_components.bounded_attribution = rc.bounded_attribution or 0.0
                env.reward_components.compute_rent = rc.compute_rent
                env.reward_components.stage_bonus = rc.stage_bonus
                env.reward_components.ratio_penalty = rc.ratio_penalty
                env.reward_components.alpha_shock = rc.alpha_shock
                env.reward_components.hindsight_credit = rc.hindsight_credit
                env.reward_components.total = rc.total_reward
                env.reward_components.last_action = action_name
                env.reward_components.env_id = env_id
                env.reward_components.val_acc = env.host_accuracy
                # New fields now available
                if rc.scaffold_count is not None:
                    env.reward_components.scaffold_count = rc.scaffold_count
                if rc.avg_scaffold_delay is not None:
                    env.reward_components.avg_scaffold_delay = rc.avg_scaffold_delay
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_reads_reward_components_dataclass -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(aggregator): read from typed RewardComponentsTelemetry dataclass"
```

---

## Task 5: Update TelemetryEvent Serialization

**Files:**
- Check/Modify: `src/esper/leyline/telemetry.py` (TelemetryEvent.to_json or serialization)
- Test: Verify JSONL output includes nested reward_components

**Step 1: Check how TelemetryEvent serializes dataclass payloads**

The `TelemetryEvent` needs to serialize `AnalyticsSnapshotPayload.reward_components` (a dataclass) to JSON.

Check if `to_dict()` is called during serialization. If not, update the serializer.

**Step 2: Test JSONL output**

```python
def test_telemetry_event_serializes_nested_reward_components():
    """TelemetryEvent should serialize reward_components dataclass to JSON."""
    import json
    from datetime import datetime, timezone
    from esper.leyline.telemetry import TelemetryEvent, AnalyticsSnapshotPayload, TelemetryEventType
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        seed_stage=2,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    json_str = event.to_json()
    parsed = json.loads(json_str)

    # Verify nested structure in JSON
    assert "reward_components" in parsed["data"]
    assert parsed["data"]["reward_components"]["seed_stage"] == 2
    assert parsed["data"]["reward_components"]["bounded_attribution"] == 0.5
```

**Step 3: If serialization doesn't handle dataclass, update it**

In `TelemetryEvent.to_json()` or the JSON encoder, ensure dataclasses are converted via `to_dict()`:

```python
def _serialize_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    # ... rest of serialization
```

**Step 4: Commit if changes needed**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "fix(telemetry): ensure reward_components dataclass serializes to JSON"
```

---

## Task 6: Integration Test - Full Pipeline

**Files:**
- Test: `tests/integration/test_reward_telemetry_flow.py`

**Step 1: Write end-to-end integration test**

```python
# tests/integration/test_reward_telemetry_flow.py

def test_reward_components_flow_end_to_end(tmp_path):
    """Verify reward_components flows from dataclass through to MCP query."""
    import json
    import duckdb
    from datetime import datetime, timezone
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
    from esper.leyline.telemetry import TelemetryEvent, AnalyticsSnapshotPayload, TelemetryEventType
    from esper.karn.mcp.views import create_views

    # 1. Create reward components (simulating compute_reward output)
    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=2,
        action_shaping=0.05,
        terminal_bonus=0.0,
        total_reward=0.45,
        val_acc=0.78,
        num_fossilized_seeds=1,
    )

    # 2. Create payload with typed dataclass
    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.45,
        action_name="WAIT",
        reward_components=rc,
    )

    # 3. Create event and serialize to JSONL
    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    # Write to JSONL
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    events_file.write_text(event.to_json() + "\n")

    # 4. Query via MCP view
    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    result = conn.execute("""
        SELECT seed_stage, action_shaping, bounded_attribution, val_acc
        FROM rewards
    """).fetchone()

    # 5. Verify all fields came through
    assert result is not None
    assert result[0] == 2  # seed_stage - WAS MISSING BEFORE
    assert result[1] == 0.05  # action_shaping - WAS MISSING BEFORE
    assert result[2] == 0.5  # bounded_attribution
    assert result[3] == 0.78  # val_acc - WAS MISSING BEFORE
```

**Step 2: Run integration test**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_reward_telemetry_flow.py -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_reward_telemetry_flow.py
git commit -m "test: add integration test for reward_components telemetry flow"
```

---

## Task 7: Full Test Suite Verification

**Step 1: Run all tests**

```bash
PYTHONPATH=src uv run pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

If tests fail due to old code expecting flat fields, update them to use nested structure.

**Step 3: Final commit if fixes needed**

```bash
git add -A
git commit -m "fix: update tests for nested reward_components structure"
```

---

## Summary

| Task | Files Modified | Change |
|------|----------------|--------|
| 1 | leyline/telemetry.py | Add `reward_components: RewardComponentsTelemetry \| None` |
| 2 | emitters.py, vectorized.py | Atomic: pass dataclass, remove old params, fix `in locals()` smell |
| 3 | mcp/views.py | Query nested paths (no fallbacks) |
| 4 | aggregator.py | Direct field access on dataclass |
| 5 | telemetry.py | Ensure dataclass serializes to JSON |
| 6 | integration test | End-to-end verification |
| 7 | Full test suite | Catch and fix any breakage |

**Fields Now Available (Were Missing Before):**
- `seed_stage`
- `action_shaping`
- `terminal_bonus`
- `hindsight_credit`
- `synergy_bonus`
- `fossilize_terminal_bonus`
- `growth_ratio`
- `val_acc`
- `num_fossilized_seeds`
- All 28 fields from `RewardComponentsTelemetry`
