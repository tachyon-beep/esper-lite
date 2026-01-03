# Telemetry Record: [TELE-800] Recent Decisions

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-800` |
| **Name** | Recent Decisions |
| **Category** | `decision` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "What decisions has Tamiyo made recently, with what confidence, and what were the outcomes?"

This provides real-time visibility into the policy's decision-making process, enabling operators to:
- Track the sequence of actions taken across environments
- Monitor per-head confidence distributions for each decision
- Detect policy collapse (deterministic choices when exploration expected)
- Correlate decisions with rewards and TD advantage estimates
- Understand the reasoning context (slot states, host accuracy, alternatives considered)

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `list[DecisionSnapshot]` |
| **Units** | N/A (composite dataclass) |
| **Range** | 0 to MAX_DECISIONS (8) elements |
| **Precision** | Varies by field (see DecisionSnapshot structure) |
| **Default** | `[]` (empty list) |

### DecisionSnapshot Structure

The `DecisionSnapshot` dataclass captures a complete snapshot of a single Tamiyo decision:

#### Core Decision Fields

| Field | Type | Description |
|-------|------|-------------|
| `decision_id` | `str` | Unique ID (8-char UUID prefix) for click-to-pin targeting |
| `timestamp` | `datetime` | When the decision was made |
| `env_id` | `int` | Environment that made this decision |
| `epoch` | `int` | Training epoch when decision was made |
| `batch` | `int` | Batch number within PPO update |
| `pinned` | `bool` | Whether decision is pinned (prevents carousel removal) |

#### Chosen Action Fields

| Field | Type | Description |
|-------|------|-------------|
| `chosen_action` | `str` | Operation: "GERMINATE", "ADVANCE", "SET_ALPHA_TARGET", "PRUNE", "FOSSILIZE", "WAIT" |
| `chosen_slot` | `str \| None` | Target slot for action (None for WAIT) |
| `chosen_blueprint` | `str \| None` | Blueprint ID (e.g., "conv_light", "attention") for GERMINATE |
| `chosen_style` | `str \| None` | Blending style: "LINEAR_ADD", "GATED_GATE", etc. |
| `chosen_tempo` | `str \| None` | Integration speed: "FAST", "STANDARD", "SLOW" |
| `chosen_alpha_target` | `str \| None` | Target alpha amplitude: "HALF", "SEVENTY", "FULL" |
| `chosen_alpha_speed` | `str \| None` | Ramp speed: "INSTANT", "FAST", "MEDIUM", "SLOW" |
| `chosen_curve` | `str \| None` | Alpha curve shape: "LINEAR", "COSINE", "SIGMOID", etc. |

#### Per-Head Confidence Fields

Probability of chosen option within each action head (0.0 to 1.0):

| Field | Type | Description |
|-------|------|-------------|
| `op_confidence` | `float` | Probability of chosen operation |
| `slot_confidence` | `float` | Probability of chosen slot |
| `blueprint_confidence` | `float` | Probability of chosen blueprint |
| `style_confidence` | `float` | Probability of chosen style |
| `tempo_confidence` | `float` | Probability of chosen tempo |
| `alpha_target_confidence` | `float` | Probability of chosen alpha target |
| `alpha_speed_confidence` | `float` | Probability of chosen alpha speed |
| `curve_confidence` | `float` | Probability of chosen curve |

#### Per-Head Entropy Fields

Distribution spread for each action head (higher = more uncertain):

| Field | Type | Description |
|-------|------|-------------|
| `op_entropy` | `float` | Entropy of operation distribution |
| `slot_entropy` | `float` | Entropy of slot distribution |
| `blueprint_entropy` | `float` | Entropy of blueprint distribution |
| `style_entropy` | `float` | Entropy of style distribution |
| `tempo_entropy` | `float` | Entropy of tempo distribution |
| `alpha_target_entropy` | `float` | Entropy of alpha target distribution |
| `alpha_speed_entropy` | `float` | Entropy of alpha speed distribution |
| `curve_entropy` | `float` | Entropy of curve distribution |

#### Value and Reward Fields

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | `float` | Overall action probability (0-1) |
| `expected_value` | `float` | V(s) estimate before action |
| `actual_reward` | `float \| None` | Reward received (None if pending) |
| `value_residual` | `float` | r - V(s): immediate reward minus value estimate |
| `td_advantage` | `float \| None` | r + gamma*V(s') - V(s): true TD(0) advantage |
| `decision_entropy` | `float` | -sum(p*log(p)) for the action distribution |

#### Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `slot_states` | `dict[str, str]` | Slot ID to state string (e.g., "Training 12%", "Empty") |
| `host_accuracy` | `float` | Host model accuracy when decision was made |
| `alternatives` | `list[tuple[str, float]]` | Top-2 alternative actions with probabilities |

### Semantic Meaning

> The `recent_decisions` list maintains a rolling window of the most recent Tamiyo decisions across all environments. It implements a "firehose model" where:
>
> 1. Decisions are added as they occur (newest first)
> 2. Maximum 8 decisions are retained (MAX_DECISIONS constant)
> 3. Decisions older than 2 minutes are expired
> 4. Pinned decisions are never auto-expired
>
> Each `DecisionSnapshot` captures the complete state at decision time, enabling reconstruction of the policy's reasoning process.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Per-head entropy > 0.3, confidence < 0.95 | Policy exploring appropriately |
| **Warning** | Per-head entropy < 0.3 with multiple valid options | Potential policy collapse |
| **Critical** | Multiple heads at entropy ~0, deterministic choices | Active policy collapse |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Per-step action in vectorized training loop |
| **File** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Function/Method** | Training loop, after action execution |
| **Line(s)** | ~3180 (calls `emitters[env_idx].on_last_action()`) |

```python
# Per-env emitter builds HeadTelemetry with per-head confidence/entropy
head_telem = HeadTelemetry(
    op_confidence=...,
    slot_confidence=...,
    blueprint_confidence=...,
    # ... all 8 heads
    op_entropy=...,
    slot_entropy=...,
    # ... all 8 entropies
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
    alternatives=top_alternatives,
    action_confidence=action_prob,
    decision_entropy=entropy,
    reward_components=reward_comp,
    head_telemetry=head_telem,
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `VectorizedEmitter.on_last_action()` | `simic/telemetry/emitters.py` |
| **2. Collection** | `ANALYTICS_SNAPSHOT` event with `kind="last_action"` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._handle_analytics_snapshot()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `TamiyoState.recent_decisions` | `karn/sanctum/schema.py` |

```
[VectorizedTrainingLoop] --on_last_action()--> [VectorizedEmitter]
    --ANALYTICS_SNAPSHOT(kind="last_action")--> [TelemetryHub]
    --event--> [SanctumAggregator._handle_analytics_snapshot()]
    --> [DecisionSnapshot created] --> [TamiyoState.recent_decisions]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `recent_decisions: list[DecisionSnapshot]` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.recent_decisions` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~968 |

### DecisionSnapshot Dataclass Location

| Property | Value |
|----------|-------|
| **Dataclass** | `DecisionSnapshot` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~1139-1206 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Decision carousel with per-head confidence heat bars |
| DecisionsColumn | `widgets/tamiyo_brain/decisions_column.py` | Vertical stack of decision cards with WHY reasoning |
| ActionDistributionPanel | `widgets/tamiyo_brain/action_distribution.py` | Decision history for action counting |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `VectorizedEmitter.on_last_action()` emits per-step decisions
- [x] **Transport works** — `ANALYTICS_SNAPSHOT(kind="last_action")` event reaches aggregator
- [x] **Schema field exists** — `TamiyoState.recent_decisions: list[DecisionSnapshot]`
- [x] **Default is correct** — Empty list `[]` appropriate before first action
- [x] **Consumer reads it** — ActionHeadsPanel and DecisionsColumn access the field
- [x] **Display is correct** — Decision cards render with confidence heat bars
- [x] **Thresholds applied** — Entropy labels (collapsing/confident/balanced/exploring) applied

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/telemetry/test_emitters.py` | (emit_last_action tests) | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_analytics_snapshot_populates_decision_head_choices` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_analytics_snapshot_wait_action_no_head_choices` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_head_telemetry_confidence_and_entropy` | `[x]` |
| Integration (backend) | `tests/karn/sanctum/test_backend.py` | `test_aggregator_processes_analytics_snapshot` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens)
3. Observe ActionHeadsPanel decision carousel (rows #1-#6)
4. Verify decisions appear with per-head confidence heat bars
5. Verify age pips transition green -> yellow -> brown -> red
6. Right-click a decision card to pin it
7. Verify pinned decisions persist while others rotate out

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO training loop | event | Only populated during active training |
| `HeadTelemetry` | dataclass | Provides per-head confidence and entropy |
| `RewardComponentsTelemetry` | dataclass | Provides reward breakdown |
| Environment state | computation | Slot states, host accuracy needed |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ActionHeadsPanel | widget | Displays decision carousel with heat bars |
| DecisionsColumn | widget | Shows decision cards with WHY reasoning |
| Decision drill-down modal | widget | Detailed view on click |
| TD advantage computation | computation | Uses pending decisions for V(s') |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-09-15 | Initial | Created with Policy V2 factored heads |
| 2024-11-20 | Refactor | Added HeadTelemetry for per-head metrics |
| 2024-12-18 | Enhancement | Added firehose model with 2-min expiry |
| 2025-01-03 | Audit | Verified wiring in telemetry audit |

---

## 8. Notes

> **Design Decision: Firehose Model**
> The decision carousel uses a "firehose" model where:
> - Decisions are grabbed as they occur, not queued
> - Age starts at 0 when added to display (not original timestamp)
> - Only the most recent decision is considered for each swap
> - This prevents backlog accumulation during high-frequency training
>
> **Design Decision: Per-Head Confidence vs Overall Confidence**
> The `confidence` field contains the overall action probability, while
> `op_confidence`, `slot_confidence`, etc. contain per-head probabilities.
> For factored action heads, per-head confidences are more diagnostic
> since they show which heads are collapsing independently.
>
> **Carousel Behavior**
> - Maximum 8 decisions displayed (MAX_DECISIONS constant)
> - Decisions expire after 2 minutes (MAX_DISPLAY_AGE_S = 120.0)
> - Swap interval is 5 seconds in ActionHeadsPanel
> - Pinned decisions are never auto-expired
>
> **TD Advantage Computation**
> The `td_advantage` field is computed asynchronously when the NEXT decision
> arrives for the same environment, since it requires V(s') from the subsequent
> state. This is stored in `_pending_decisions` on the aggregator.
