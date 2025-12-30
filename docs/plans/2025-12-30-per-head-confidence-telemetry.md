# Per-Head Confidence & Entropy Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire per-head confidence values (probabilities) AND entropy from the training loop through telemetry to the Sanctum TUI's HEAD OUTPUTS panel.

**Architecture:** The training loop already computes per-head log probabilities in `head_log_probs` and entropy in the policy network. We'll:
1. Create a typed `HeadTelemetry` dataclass (not raw dict) for type safety
2. Batch all heads into a single GPU→CPU transfer using `torch.stack()`
3. Add `.detach()` before transfer for safety
4. Include per-head entropy alongside confidence
5. Use consistent naming (`curve` not `alpha_curve`)

**Tech Stack:** PyTorch (exp operation, stack), Python dataclasses, Textual TUI

**Specialist Review Notes:**
- DRL Expert: Confirmed `exp(log_prob)` correctly handles masked distributions
- PyTorch Expert: Confirmed no underflow concerns for categorical distributions
- Both: Recommended batched transfer, `.detach()`, typed dataclass, and entropy

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/esper/leyline/telemetry.py` | Add `HeadTelemetry` dataclass + 16 fields (8 confidence + 8 entropy) to `AnalyticsSnapshotPayload` |
| `src/esper/simic/telemetry/emitters.py` | Accept `HeadTelemetry` in `on_last_action()` |
| `src/esper/simic/training/vectorized.py` | Compute per-head confidence AND entropy with batched transfer |
| `src/esper/karn/sanctum/aggregator.py` | Map telemetry fields to `DecisionSnapshot` |
| `src/esper/karn/sanctum/schema.py` | Add 8 entropy fields to `DecisionSnapshot` |

---

### Task 1: Add HeadTelemetry Dataclass and Fields to Telemetry Payload

**Files:**
- Modify: `src/esper/leyline/telemetry.py`
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write the failing test**

Create test in `tests/leyline/test_telemetry.py`:

```python
def test_head_telemetry_dataclass():
    """HeadTelemetry should hold confidence and entropy for all 8 heads."""
    from esper.leyline.telemetry import HeadTelemetry

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    # Verify all confidence fields
    assert head_telem.op_confidence == 0.85
    assert head_telem.slot_confidence == 0.72
    assert head_telem.blueprint_confidence == 0.91
    assert head_telem.style_confidence == 0.65
    assert head_telem.tempo_confidence == 0.88
    assert head_telem.alpha_target_confidence == 0.77
    assert head_telem.alpha_speed_confidence == 0.69
    assert head_telem.curve_confidence == 0.82

    # Verify all entropy fields
    assert head_telem.op_entropy == 0.3
    assert head_telem.slot_entropy == 0.8
    assert head_telem.blueprint_entropy == 0.5
    assert head_telem.style_entropy == 0.6
    assert head_telem.tempo_entropy == 0.4
    assert head_telem.alpha_target_entropy == 0.55
    assert head_telem.alpha_speed_entropy == 0.45
    assert head_telem.curve_entropy == 0.35


def test_analytics_snapshot_payload_accepts_head_telemetry():
    """AnalyticsSnapshotPayload should accept HeadTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload, HeadTelemetry

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        head_telemetry=head_telem,
    )

    assert payload.head_telemetry is head_telem
    assert payload.head_telemetry.op_confidence == 0.85
    assert payload.head_telemetry.op_entropy == 0.3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_head_telemetry_dataclass -v`

Expected: FAIL with `ImportError: cannot import name 'HeadTelemetry'`

**Step 3: Add HeadTelemetry dataclass and field to AnalyticsSnapshotPayload**

In `src/esper/leyline/telemetry.py`, add the dataclass (near other telemetry dataclasses):

```python
@dataclass
class HeadTelemetry:
    """Per-head confidence and entropy values for factored action heads.

    Confidence = P(chosen_action | valid_mask) via exp(log_prob).
    This is the probability among valid actions, properly handling masking.

    Entropy measures how spread out the distribution is (higher = more uncertain).
    """
    # Per-head confidence (probability of chosen action)
    op_confidence: float = 0.0
    slot_confidence: float = 0.0
    blueprint_confidence: float = 0.0
    style_confidence: float = 0.0
    tempo_confidence: float = 0.0
    alpha_target_confidence: float = 0.0
    alpha_speed_confidence: float = 0.0
    curve_confidence: float = 0.0

    # Per-head entropy (distribution spread - higher means more uncertain)
    op_entropy: float = 0.0
    slot_entropy: float = 0.0
    blueprint_entropy: float = 0.0
    style_entropy: float = 0.0
    tempo_entropy: float = 0.0
    alpha_target_entropy: float = 0.0
    alpha_speed_entropy: float = 0.0
    curve_entropy: float = 0.0
```

Then add the field to `AnalyticsSnapshotPayload` (after `action_confidence`):

```python
    # Per-head telemetry (typed dataclass, not raw dict)
    head_telemetry: HeadTelemetry | None = None
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_head_telemetry_dataclass tests/leyline/test_telemetry.py::test_analytics_snapshot_payload_accepts_head_telemetry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add HeadTelemetry dataclass with confidence and entropy fields"
```

---

### Task 2: Update Emitter to Accept HeadTelemetry

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py`
- Test: `tests/simic/telemetry/test_emitters.py`

**Step 1: Write the failing test**

Add test in `tests/simic/telemetry/test_emitters.py`:

```python
def test_on_last_action_accepts_head_telemetry():
    """on_last_action should accept and forward HeadTelemetry."""
    from esper.simic.telemetry.emitters import PerEnvTelemetryEmitter
    from esper.karn.store import TelemetryHub
    from esper.leyline.telemetry import HeadTelemetry

    hub = TelemetryHub()
    emitter = PerEnvTelemetryEmitter(env_id=0, hub=hub)

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    emitter.on_last_action(
        epoch=1,
        action_dict={"op": 0, "slot": 0, "blueprint": 0, "style": 0, "tempo": 0,
                     "alpha_target": 0, "alpha_speed": 0, "alpha_curve": 0},
        target_slot="r0c0",
        masked={},
        success=True,
        active_alpha_algorithm=None,
        head_telemetry=head_telem,
    )

    # Check the emitted event contains head_telemetry
    events = list(hub.iter_events())
    last_action_events = [e for e in events if e.data and getattr(e.data, 'kind', None) == 'last_action']
    assert len(last_action_events) == 1
    payload = last_action_events[0].data

    assert payload.head_telemetry is not None
    assert payload.head_telemetry.op_confidence == 0.85
    assert payload.head_telemetry.op_entropy == 0.3
    assert payload.head_telemetry.curve_confidence == 0.82
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_on_last_action_accepts_head_telemetry -v`

Expected: FAIL with `TypeError: on_last_action() got an unexpected keyword argument 'head_telemetry'`

**Step 3: Update on_last_action signature and implementation**

In `src/esper/simic/telemetry/emitters.py`:

1. Add import at top:
```python
from esper.leyline.telemetry import HeadTelemetry
```

2. Add parameter to `on_last_action` signature:
```python
    def on_last_action(
        self,
        epoch: int,
        action_dict: dict[str, int],
        target_slot: str | None,
        masked: dict[str, bool],
        success: bool,
        active_alpha_algorithm: str | None,
        *,
        total_reward: float | None = None,
        value_estimate: float | None = None,
        host_accuracy: float | None = None,
        slot_states: dict[str, str] | None = None,
        action_confidence: float | None = None,
        alternatives: list[tuple[str, float]] | None = None,
        decision_entropy: float | None = None,
        reward_components: "RewardComponentsTelemetry | None" = None,
        head_telemetry: HeadTelemetry | None = None,  # NEW: typed dataclass
    ) -> dict[str, Any]:
```

3. Pass through to `AnalyticsSnapshotPayload`:
```python
        payload = AnalyticsSnapshotPayload(
            # ... existing fields ...
            head_telemetry=head_telemetry,
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_on_last_action_accepts_head_telemetry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py tests/simic/telemetry/test_emitters.py
git commit -m "feat(emitters): accept HeadTelemetry in on_last_action"
```

---

### Task 3: Compute Per-Head Confidence AND Entropy in Training Loop

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Test: Integration test (run training with telemetry)

**Step 1: Add imports and HEAD_NAMES constant**

At top of file, ensure these are available:

```python
from esper.leyline.telemetry import HeadTelemetry

# Canonical head names for consistent ordering (matches factored_actions.py)
HEAD_NAMES = ["op", "slot", "blueprint", "style", "tempo", "alpha_target", "alpha_speed", "alpha_curve"]
```

**Step 2: Add batched per-head computation after action sampling**

In `src/esper/simic/training/vectorized.py`, after the existing `op_probs_cpu` computation (around line 2548), add:

```python
            # PERF: Pre-compute per-head confidences AND entropy for telemetry.
            # Uses batched GPU->CPU transfer: stack all heads, single transfer.
            #
            # Confidence = exp(log_prob) = P(chosen_action | valid_mask)
            # This properly handles masking via MaskedCategorical.
            #
            # Entropy measures distribution spread (higher = more uncertain).
            # Already computed by policy network.
            head_confidences_cpu: np.ndarray | None = None  # [8, num_envs]
            head_entropies_cpu: np.ndarray | None = None    # [8, num_envs]

            if ops_telemetry_enabled and head_log_probs:
                # Stack all head log probs: [8, num_envs]
                stacked_log_probs = torch.stack([head_log_probs[h] for h in HEAD_NAMES])
                # Single exp + detach + transfer
                head_confidences_cpu = torch.exp(stacked_log_probs).detach().cpu().numpy()

                # Get entropy if available from action_result
                if hasattr(action_result, 'entropy') and action_result.entropy:
                    stacked_entropy = torch.stack([action_result.entropy[h] for h in HEAD_NAMES])
                    head_entropies_cpu = stacked_entropy.detach().cpu().numpy()
```

**Step 3: Build HeadTelemetry and pass to emitter**

Modify the `emitters[env_idx].on_last_action()` call:

```python
                    # Build HeadTelemetry for this env (typed dataclass, not raw dict)
                    head_telem: HeadTelemetry | None = None
                    if head_confidences_cpu is not None:
                        head_telem = HeadTelemetry(
                            op_confidence=float(head_confidences_cpu[0, env_idx]),
                            slot_confidence=float(head_confidences_cpu[1, env_idx]),
                            blueprint_confidence=float(head_confidences_cpu[2, env_idx]),
                            style_confidence=float(head_confidences_cpu[3, env_idx]),
                            tempo_confidence=float(head_confidences_cpu[4, env_idx]),
                            alpha_target_confidence=float(head_confidences_cpu[5, env_idx]),
                            alpha_speed_confidence=float(head_confidences_cpu[6, env_idx]),
                            curve_confidence=float(head_confidences_cpu[7, env_idx]),
                            # Entropy (0.0 if not available)
                            op_entropy=float(head_entropies_cpu[0, env_idx]) if head_entropies_cpu is not None else 0.0,
                            slot_entropy=float(head_entropies_cpu[1, env_idx]) if head_entropies_cpu is not None else 0.0,
                            blueprint_entropy=float(head_entropies_cpu[2, env_idx]) if head_entropies_cpu is not None else 0.0,
                            style_entropy=float(head_entropies_cpu[3, env_idx]) if head_entropies_cpu is not None else 0.0,
                            tempo_entropy=float(head_entropies_cpu[4, env_idx]) if head_entropies_cpu is not None else 0.0,
                            alpha_target_entropy=float(head_entropies_cpu[5, env_idx]) if head_entropies_cpu is not None else 0.0,
                            alpha_speed_entropy=float(head_entropies_cpu[6, env_idx]) if head_entropies_cpu is not None else 0.0,
                            curve_entropy=float(head_entropies_cpu[7, env_idx]) if head_entropies_cpu is not None else 0.0,
                        )

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
                        reward_components=reward_components,
                        head_telemetry=head_telem,  # NEW
                    )
```

**Step 4: Run existing tests to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized.py -v --tb=short -x`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "feat(training): compute per-head confidence and entropy with batched GPU transfer"
```

---

### Task 4: Add Entropy Fields to DecisionSnapshot Schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`
- Test: `tests/karn/sanctum/test_schema.py` (or inline)

**Step 1: Write the failing test**

```python
def test_decision_snapshot_has_entropy_fields():
    """DecisionSnapshot should have per-head entropy fields."""
    from esper.karn.sanctum.schema import DecisionSnapshot

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=0.9,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.85,
        expected_value=0.5,
        actual_reward=0.6,
        alternatives=[],
        decision_id="abc123",
        decision_entropy=0.4,
        env_id=0,
        value_residual=0.1,
        # Entropy fields
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    assert decision.op_entropy == 0.3
    assert decision.curve_entropy == 0.35
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_has_entropy_fields -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'op_entropy'`

**Step 3: Add entropy fields to DecisionSnapshot**

In `src/esper/karn/sanctum/schema.py`, add after the confidence fields:

```python
    # Per-head entropy values (distribution spread - higher means more uncertain)
    # Useful for diagnosing policy collapse (entropy -> 0) or exploration issues
    op_entropy: float = 0.0
    slot_entropy: float = 0.0
    blueprint_entropy: float = 0.0
    style_entropy: float = 0.0
    tempo_entropy: float = 0.0
    alpha_target_entropy: float = 0.0
    alpha_speed_entropy: float = 0.0
    curve_entropy: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_has_entropy_fields -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(schema): add per-head entropy fields to DecisionSnapshot"
```

---

### Task 5: Map HeadTelemetry in Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

Add test in `tests/karn/sanctum/test_aggregator.py`:

```python
def test_decision_snapshot_populates_from_head_telemetry():
    """DecisionSnapshot should include confidence and entropy from HeadTelemetry."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        HeadTelemetry,
        TelemetryEvent,
        TelemetryEventType,
        TrainingStartedPayload,
    )

    agg = SanctumAggregator()

    # Initialize with training started
    agg.handle_event(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        epoch=0,
        data=TrainingStartedPayload(num_envs=1),
    ))

    # Send last_action with HeadTelemetry
    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    agg.handle_event(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=1,
        timestamp=datetime.now(timezone.utc),
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            action_name="GERMINATE",
            slot_id="r0c0",
            blueprint_id="conv_light",
            style="LINEAR_ADD",
            tempo_idx=1,
            alpha_target=0.7,
            alpha_speed="MEDIUM",
            alpha_curve="COSINE",
            action_confidence=0.85,
            head_telemetry=head_telem,
        ),
    ))

    snapshot = agg.snapshot()
    decisions = snapshot.tamiyo.recent_decisions
    assert len(decisions) == 1

    decision = decisions[0]
    # Confidence
    assert decision.op_confidence == 0.85
    assert decision.slot_confidence == 0.72
    assert decision.blueprint_confidence == 0.91
    assert decision.style_confidence == 0.65
    assert decision.tempo_confidence == 0.88
    assert decision.alpha_target_confidence == 0.77
    assert decision.alpha_speed_confidence == 0.69
    assert decision.curve_confidence == 0.82
    # Entropy
    assert decision.op_entropy == 0.3
    assert decision.slot_entropy == 0.8
    assert decision.blueprint_entropy == 0.5
    assert decision.style_entropy == 0.6
    assert decision.tempo_entropy == 0.4
    assert decision.alpha_target_entropy == 0.55
    assert decision.alpha_speed_entropy == 0.45
    assert decision.curve_entropy == 0.35
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_decision_snapshot_populates_from_head_telemetry -v`

Expected: FAIL with `AssertionError: assert 0.0 == 0.85` (confidences still default)

**Step 3: Update aggregator to map HeadTelemetry**

In `src/esper/karn/sanctum/aggregator.py`, modify the `DecisionSnapshot` creation:

```python
            # Extract HeadTelemetry if present
            ht = payload.head_telemetry

            decision = DecisionSnapshot(
                timestamp=now_dt,
                slot_states=payload.slot_states or {},
                host_accuracy=env.host_accuracy,
                chosen_action=action_name,
                chosen_slot=payload.slot_id,
                confidence=payload.action_confidence,
                expected_value=value_s,
                actual_reward=total_reward,
                alternatives=payload.alternatives or [],
                decision_id=str(uuid.uuid4())[:8],
                decision_entropy=payload.decision_entropy or 0.0,
                env_id=env_id,
                value_residual=total_reward - value_s,
                td_advantage=None,
                # Head choice fields
                chosen_blueprint=payload.blueprint_id,
                chosen_tempo=chosen_tempo,
                chosen_style=payload.style,
                chosen_curve=payload.alpha_curve,
                chosen_alpha_target=chosen_alpha_target,
                chosen_alpha_speed=payload.alpha_speed,
                # Per-head confidence values (from HeadTelemetry)
                op_confidence=ht.op_confidence if ht else 0.0,
                slot_confidence=ht.slot_confidence if ht else 0.0,
                blueprint_confidence=ht.blueprint_confidence if ht else 0.0,
                style_confidence=ht.style_confidence if ht else 0.0,
                tempo_confidence=ht.tempo_confidence if ht else 0.0,
                alpha_target_confidence=ht.alpha_target_confidence if ht else 0.0,
                alpha_speed_confidence=ht.alpha_speed_confidence if ht else 0.0,
                curve_confidence=ht.curve_confidence if ht else 0.0,
                # Per-head entropy values (from HeadTelemetry)
                op_entropy=ht.op_entropy if ht else 0.0,
                slot_entropy=ht.slot_entropy if ht else 0.0,
                blueprint_entropy=ht.blueprint_entropy if ht else 0.0,
                style_entropy=ht.style_entropy if ht else 0.0,
                tempo_entropy=ht.tempo_entropy if ht else 0.0,
                alpha_target_entropy=ht.alpha_target_entropy if ht else 0.0,
                alpha_speed_entropy=ht.alpha_speed_entropy if ht else 0.0,
                curve_entropy=ht.curve_entropy if ht else 0.0,
            )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_decision_snapshot_populates_from_head_telemetry -v`

Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -v --tb=short`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(aggregator): map HeadTelemetry confidence and entropy to DecisionSnapshot"
```

---

### Task 6: Final Integration Test

**Files:**
- Test: Manual or automated integration

**Step 1: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum tests/simic/telemetry tests/leyline -v --tb=short`

Expected: All tests PASS

**Step 2: Verify HEAD OUTPUTS panel displays heat bars**

The `attention_heatmap.py` already uses `decision.op_confidence`, `decision.slot_confidence`, etc. to render heat bars. With the telemetry now populated, the bars should show real values instead of `░░░░░` (which indicates 0.0 confidence).

**Step 3: (Optional) Update attention_heatmap to show entropy**

Consider adding entropy indicator (e.g., `(H=0.3)`) next to confidence heat bars for debugging. This is a future enhancement if operators find it useful.

**Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "test: verify per-head confidence and entropy telemetry integration"
```

---

## Verification Checklist

- [ ] `HeadTelemetry` dataclass exists with 16 fields (8 confidence + 8 entropy)
- [ ] `AnalyticsSnapshotPayload.head_telemetry` accepts `HeadTelemetry`
- [ ] `on_last_action()` accepts `head_telemetry: HeadTelemetry` parameter
- [ ] Training loop uses `torch.stack()` for batched GPU→CPU transfer
- [ ] Training loop adds `.detach()` before `.cpu()`
- [ ] Per-head entropy is computed alongside confidence
- [ ] `DecisionSnapshot` has 8 entropy fields
- [ ] Aggregator maps `HeadTelemetry` to `DecisionSnapshot`
- [ ] HEAD OUTPUTS panel shows real heat bars (not all `░░░░░`)
- [ ] All existing tests still pass

---

## Specialist Review Notes (Incorporated)

| Recommendation | Status |
|----------------|--------|
| Add `.detach()` for safety | ✅ Task 3 |
| Batch GPU transfers with `torch.stack()` | ✅ Task 3 |
| Fix naming inconsistency (`curve` not `alpha_curve`) | ✅ HeadTelemetry uses `curve_*` |
| Use typed dataclass instead of dict | ✅ `HeadTelemetry` dataclass |
| Add per-head entropy | ✅ Tasks 1, 3, 4, 5 |
