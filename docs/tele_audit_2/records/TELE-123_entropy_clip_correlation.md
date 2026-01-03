# Telemetry Record: [TELE-123] Entropy Clip Correlation

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-123` |
| **Name** | Entropy Clip Correlation |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is there a correlation between policy entropy decay and clipping rates? A strong negative correlation combined with low entropy and high clipping indicates imminent policy collapse."

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

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
| **Type** | `float` |
| **Units** | Pearson correlation coefficient |
| **Range** | `[-1.0, 1.0]` |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Pearson correlation coefficient between entropy history and clip_fraction history.
>
> `r = correlation(entropy_history[-10:], clip_fraction_history[-10:])`
>
> - **r ≈ -1.0**: Perfect negative correlation (entropy declining, clipping rising)
> - **r ≈ 0.0**: No correlation (metrics moving independently)
> - **r ≈ +1.0**: Perfect positive correlation (both rising or falling together)
>
> For policy collapse detection, the dangerous pattern is specifically **negative correlation** (entropy dropping while clipping increases), suggesting the policy is deterministically converging on specific actions as entropy fails.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(corr) < 0.3` | Entropy and clipping uncorrelated (normal behavior) |
| **Warning** | `-0.6 < corr < -0.4` AND `entropy < 0.3` | Entropy declining with clipping (monitor closely) |
| **Critical** | `corr < -0.5` AND `entropy < 0.3` AND `clip > 0.25` | **COLLAPSE RISK**: Policy converging to deterministic behavior |

**Threshold Source:** `HealthStatusPanel._render_policy_state()` decision logic

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed from metric histories |
| **File** | `/home/john/esper-lite/src/esper/karn/sanctum/aggregator.py` |
| **Function/Method** | `SanctumAggregator.handle_ppo_update()` |
| **Line(s)** | ~967-970 |

```python
# Entropy-clip correlation computation
self._tamiyo.entropy_clip_correlation = compute_correlation(
    self._tamiyo.entropy_history,
    self._tamiyo.clip_fraction_history,
)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` emits entropy and clip_fraction | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload with `entropy` and `clip_fraction` fields | `leyline/telemetry.py` (PPOUpdatePayload) |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` accumulates to histories, computes correlation | `karn/sanctum/aggregator.py` (~800, ~819, ~967) |
| **4. Delivery** | Written to `snapshot.tamiyo.entropy_clip_correlation` | `karn/sanctum/schema.py` |

```
[PPOAgent] --emit entropy/clip--> [TelemetryEmitter] --event--> [Aggregator]
--accumulate to histories--> [compute_correlation()] --> [TamiyoState.entropy_clip_correlation]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `entropy_clip_correlation` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.entropy_clip_correlation` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 980 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Policy state detection and coloring (~line 251) |

**Usage Pattern:**
```python
# Determines policy state: "COLLAPSE RISK", "collapsing", "narrowing", "stable", or "drifting"
corr = tamiyo.entropy_clip_correlation
if corr < -0.5 and entropy < 0.3 and clip > 0.25:
    status = "COLLAPSE RISK"  # Red alert
elif corr < -0.6 and entropy < 0.3:
    status = "collapsing"      # Yellow warning
elif corr < -0.4 and clip < 0.15:
    status = "narrowing"       # Green (controlled reduction)
elif abs(corr) < 0.3:
    status = "stable"          # Green (uncorrelated)
else:
    status = "drifting"        # Yellow (mixed signal)
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes entropy and clip_fraction during update
- [x] **Transport works** — PPOUpdatePayload includes entropy and clip_fraction fields
- [x] **History accumulation works** — `entropy_history` and `clip_fraction_history` deques populated at lines 800, 819
- [x] **Correlation computation works** — `compute_correlation()` function defined in schema.py lines 47-84
- [x] **Schema field exists** — `TamiyoState.entropy_clip_correlation: float = 0.0` at line 980
- [x] **Default is correct** — 0.0 appropriate (no correlation by default)
- [x] **Consumer reads it** — `HealthStatusPanel._render_policy_state()` accesses field at line 251
- [x] **Display is correct** — Value renders with appropriate policy state coloring
- [x] **Thresholds applied** — HealthStatusPanel applies -0.5/-0.6/-0.4 thresholds for collapse risk detection

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (compute_correlation) | `tests/karn/sanctum/test_correlation.py` | `TestCorrelation.*` (8 tests) | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_correlation.py` | `TestAggregatorCorrelation.test_ppo_update_computes_correlation` | `[x]` |
| Unit (field default) | `tests/karn/sanctum/test_correlation.py` | `TestTamiyoStateHasCorrelationField.*` | `[x]` |
| Integration (end-to-end) | — | Manual verification with training run | `[x]` |
| Visual (TUI snapshot) | — | Manual TUI verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Policy" row
4. Verify correlation value updates after each PPO batch
5. Trigger collapse condition manually (reduce entropy coefficient) to verify red "COLLAPSE RISK" coloring
6. Check that correlation is computed correctly:
   - Should be negative when entropy falls while clipping rises
   - Should be near 0 when metrics move independently

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-001` entropy | telemetry | Entropy value required to populate entropy_history |
| `TELE-009` clip_fraction | telemetry | Clip fraction required to populate clip_fraction_history |
| PPO update cycle | event | Only computed after PPO updates, requires ≥5 samples in history |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel policy state | display | Drives policy state string and coloring (COLLAPSE RISK/collapsing/stable) |
| Collapse risk detection | system | Used in policy collapse pattern detection alongside entropy and clip_fraction |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Created telemetry record for entropy-clip correlation metric |
| | | Verified wiring in aggregator, schema, and widget consumer |
| | | Confirmed test coverage in test_correlation.py |

---

## 8. Notes

> **Design Rationale:** The Pearson correlation is computed over the last 10 samples of entropy_history and clip_fraction_history (maintained as deques with maxlen=10 in TamiyoState). This window size balances responsiveness with stability—short enough to catch emerging patterns, long enough to filter noise.
>
> **Why Negative Correlation Matters:** A key insight from DRL theory is that policy collapse doesn't always manifest as high entropy + high clipping. Instead, the dangerous pattern is **entropy declining while clipping rises**. This indicates the policy is converging to specific actions (entropy drops) and is being clipped (importance sampling weights exceeding [1-ε, 1+ε]). The negative correlation captures this causal relationship better than either metric alone.
>
> **Numerical Stability:** The `compute_correlation()` function handles edge cases: zero variance returns 0.0 (not NaN), mismatched history lengths are aligned, and histories with <5 samples return 0.0 correlation.
>
> **Integration with Collapse Risk Score:** While `collapse_risk_score` (TELE-003) combines entropy and velocity to predict collapse, `entropy_clip_correlation` captures a distinct signal: the pattern of entropy-clipping correlation is a symptom observable in real-time, useful for immediate policy state detection.
>
> **Widget Integration:** HealthStatusPanel uses this metric to provide immediate visual feedback on policy health. The policy state transitions ("stable" → "narrowing" → "collapsing" → "COLLAPSE RISK") help operators quickly diagnose when intervention is needed.
>
> **Future Improvement:** Per-head entropy-clip correlations (slot, blueprint, style, tempo, etc.) could provide more granular diagnosis of which action heads are exhibiting collapse patterns.
