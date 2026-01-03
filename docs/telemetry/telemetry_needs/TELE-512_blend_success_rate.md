# Telemetry Record: [TELE-512] Blend Success Rate

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-512` |
| **Name** | Blend Success Rate |
| **Category** | `seed` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What proportion of seeds that reach terminal state are successfully fossilized vs pruned?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | ratio (displayed as percentage 0-100%) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Blend success rate measures the proportion of seeds that successfully integrate with the host (fossilize) versus those that fail (get pruned). Computed as:
>
> blend_success_rate = fossilize_count / (fossilize_count + prune_count)
>
> - High rate (>70%) = Germinated seeds are consistently succeeding through the lifecycle
> - Medium rate (50-70%) = Mixed outcomes, may indicate quality gate tuning needed
> - Low rate (<50%) = More seeds failing than succeeding, policy may need adjustment
>
> In the botanical lifecycle metaphor, this tracks the "survival rate" of seeds that have germinated - how many successfully graft vs how many get pruned.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `blend_success_rate >= 0.70` | Strong seed quality, most germinated seeds succeed |
| **Warning** | `0.50 <= blend_success_rate < 0.70` | Mixed outcomes, monitor for trend |
| **Critical** | `blend_success_rate < 0.50` | More seeds pruned than fossilized, investigate policy |

**Threshold Source:** `SlotsPanel` applies conditional styling: `blend_color = "green" if blend_rate >= 70 else "yellow" if blend_rate >= 50 else "red"`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Computed in aggregator from cumulative fossilize and prune counts |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator._get_snapshot_unlocked()` |
| **Line(s)** | ~509-513 |

```python
# Compute seed lifecycle stats
blend_success = (
    self._cumulative_fossilized / max(1, self._cumulative_fossilized + self._cumulative_pruned)
    if (self._cumulative_fossilized + self._cumulative_pruned) > 0
    else 0.0
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | N/A (computed from other metrics) | N/A |
| **2. Collection** | `_cumulative_fossilized` and `_cumulative_pruned` accumulated from SEED_FOSSILIZED and SEED_PRUNED events | `karn/sanctum/aggregator.py` |
| **3. Aggregation** | Computed ratio in `_get_snapshot_unlocked()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.seed_lifecycle.blend_success_rate` | `karn/sanctum/schema.py` |

```
[SEED_FOSSILIZED] --increment--> [_cumulative_fossilized]
[SEED_PRUNED]     --increment--> [_cumulative_pruned]
                                         |
                                         v
[_get_snapshot_unlocked()] --compute ratio--> [SeedLifecycleStats.blend_success_rate]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `SeedLifecycleStats` |
| **Field** | `blend_success_rate` |
| **Path from SanctumSnapshot** | `snapshot.seed_lifecycle.blend_success_rate` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 191 |

```python
@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display."""
    # Cumulative counts (entire run)
    germination_count: int = 0
    prune_count: int = 0
    fossilize_count: int = 0
    # ...
    # Quality metrics
    blend_success_rate: float = 0.0  # fossilized / (fossilized + pruned)  <-- Line 191
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| SlotsPanel | `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py` | Displays as "{rate}% success" with color-coded styling |

```python
# Lines 130-136 in slots_panel.py
blend_rate = lifecycle.blend_success_rate * 100
blend_color = "green" if blend_rate >= 70 else "yellow" if blend_rate >= 50 else "red"
result.append("Lifespan:", style="dim")
result.append(f"mu{lifecycle.avg_lifespan_epochs:.0f} eps", style="cyan")
result.append("  Blend:", style="dim")
result.append(f"{blend_rate:.0f}%", style=blend_color)
result.append(" success", style="dim")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — Computed from `_cumulative_fossilized` and `_cumulative_pruned` in aggregator
- [x] **Transport works** — Counts accumulated from SEED_FOSSILIZED and SEED_PRUNED events
- [x] **Schema field exists** — `SeedLifecycleStats.blend_success_rate: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before any terminal outcomes
- [x] **Consumer reads it** — SlotsPanel accesses `snapshot.seed_lifecycle.blend_success_rate`
- [x] **Display is correct** — Value renders as percentage with color-coded styling
- [x] **Thresholds applied** — Green (>=70%), Yellow (50-70%), Red (<50%)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | Computed metric - no direct emitter | `N/A` |
| Unit (aggregator) | N/A | No SeedLifecycleStats computation test found | `[ ]` |
| Integration (end-to-end) | N/A | No blend_success_rate integration test | `[ ]` |
| Visual (TUI snapshot) | N/A | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 100`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe SlotsPanel "Blend: X% success" display
4. Wait for seeds to complete their lifecycle (fossilize or prune)
5. Verify percentage updates as terminal outcomes occur
6. Verify color changes based on thresholds:
   - Green when >=70% success rate
   - Yellow when 50-70% success rate
   - Red when <50% success rate

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-503` fossilize_count | telemetry | Numerator of the ratio |
| `TELE-504` prune_count | telemetry | Part of denominator (with fossilize_count) |
| `SEED_FOSSILIZED` event | event | Increments fossilize count |
| `SEED_PRUNED` event | event | Increments prune count |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| SlotsPanel display | display | Shows blend success as color-coded percentage |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Claude | Initial creation during telemetry audit |

---

## 8. Notes

> **Design Decision:** Blend success rate is a derived metric computed at snapshot time rather than emitted directly. This ensures it always reflects the current ratio of cumulative counts.
>
> **Edge Case Handling:** When both fossilize_count and prune_count are zero (no terminal outcomes yet), the rate defaults to 0.0. The aggregator guards against division by zero:
> ```python
> if (self._cumulative_fossilized + self._cumulative_pruned) > 0
>     else 0.0
> ```
>
> **Display Semantics:** The widget multiplies by 100 for percentage display (`blend_rate * 100`), but the schema stores the raw ratio [0.0, 1.0].
>
> **Color Thresholds Rationale:**
> - **70%+**: Healthy - most germinated seeds are successfully integrating
> - **50-70%**: Warning - mixed outcomes, may indicate overly aggressive germination or weak seeds
> - **<50%**: Critical - more failures than successes, policy is selecting poor candidates
>
> **Test Gap:** No direct tests for blend_success_rate computation. Should test:
> - Zero-division case (no terminal outcomes)
> - Edge case with only fossilizations (100%)
> - Edge case with only prunes (0%)
> - Mixed outcomes for correct ratio calculation
