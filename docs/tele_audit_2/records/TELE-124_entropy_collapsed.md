# Telemetry Record: [TELE-124] Entropy Collapsed

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-124` |
| **Name** | Entropy Collapsed |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Has the policy entropy dropped below the critical threshold, indicating the policy has collapsed to near-deterministic behavior?"

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
| **Type** | `bool` |
| **Units** | N/A (boolean flag) |
| **Range** | `True` / `False` |
| **Precision** | N/A |
| **Default** | `False` |

### Semantic Meaning

> Boolean flag indicating whether policy entropy has fallen below the critical collapse threshold (0.1 nats).
>
> The flag is computed as: `entropy_collapsed = entropy < DEFAULT_ENTROPY_COLLAPSE_THRESHOLD`
>
> Where `DEFAULT_ENTROPY_COLLAPSE_THRESHOLD = 0.1` (defined in `leyline/__init__.py`).
>
> When `True`, the policy is near-deterministic and has effectively stopped exploring. This is a critical training failure that requires intervention (e.g., entropy coefficient boost, learning rate reduction, or training restart).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `entropy_collapsed == False` | Policy maintaining exploration |
| **Critical** | `entropy_collapsed == True` | Policy has collapsed - immediate intervention required |

**Note:** This is a binary flag derived from TELE-120 (entropy). The actual entropy value provides more granular health status with warning thresholds.

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed from entropy metric |
| **File** | `/home/john/esper-lite/src/esper/simic/telemetry/emitters.py` |
| **Function/Method** | `emit_ppo_update()` |
| **Line(s)** | ~831 |

```python
# Computed inline during PPO update emission
entropy_collapsed=metrics["entropy"] < DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` constructs `PPOUpdatePayload` with `entropy_collapsed` field | `simic/telemetry/emitters.py` |
| **2. Collection** | Event with `PPOUpdatePayload` containing `entropy_collapsed: bool` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` extracts and assigns | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.entropy_collapsed` | `karn/sanctum/schema.py` |

```
[PPOAgent] --> emit_ppo_update() --> [PPOUpdatePayload] --> [Aggregator] --> [TamiyoState.entropy_collapsed]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `entropy_collapsed` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.entropy_collapsed` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 888 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| AnomalyStrip | `widgets/anomaly_strip.py` | Sets `ppo_issues=True` when `entropy_collapsed == True` |
| PolicyDiagnostics (Vue) | `overwatch/web/src/components/PolicyDiagnostics.vue` | Displays "YES" (critical) or "OK" (good) with color coding |
| DuckDB MCP Views | `karn/mcp/views.py` | Exposed via SQL: `ppo_updates.entropy_collapsed` |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `emit_ppo_update()` computes flag from entropy metric
- [x] **Transport works** — `PPOUpdatePayload.entropy_collapsed` field exists
- [x] **Schema field exists** — `TamiyoState.entropy_collapsed: bool = False`
- [x] **Default is correct** — `False` appropriate before first PPO update
- [x] **Consumer reads it** — AnomalyStrip and PolicyDiagnostics access the field
- [x] **Display is correct** — AnomalyStrip triggers PPO issues, Vue shows YES/OK
- [x] **Thresholds applied** — Binary flag, critical styling when True

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/leyline/test_telemetry.py` | Payload serialization tests | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_anomaly_strip.py` | `test_anomaly_strip_ppo_health` | `[x]` |
| Integration (end-to-end) | — | — | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |
| Visual (Vue component) | `overwatch/web/src/components/__tests__/PolicyDiagnostics.spec.ts` | `shows entropy_collapsed prominently with critical styling when true` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI or Overwatch web dashboard
3. Observe AnomalyStrip or PolicyDiagnostics panel
4. Initially should show no PPO issues (entropy not collapsed)
5. To verify critical state: modify entropy_coef to 0.0 to force collapse, or use test fixture with `entropy_collapsed=True`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `TELE-120` entropy | telemetry | Collapse flag is computed as `entropy < 0.1` |
| PPO update cycle | event | Only populated after first PPO update completes |
| `DEFAULT_ENTROPY_COLLAPSE_THRESHOLD` | constant | Threshold value (0.1) from `leyline/__init__.py` |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| AnomalyStrip.ppo_issues | display | Sets `ppo_issues=True` when collapsed |
| PolicyDiagnostics health | display | Shows critical styling when True |
| DuckDB ppo_updates view | query | Available for SQL analysis |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-12-16 | Initial | Added in telemetry low-hanging-fruit plan |
| 2025-01-03 | Audit | Verified wiring in telemetry audit |

---

## 8. Notes

> **Design Decision:** This is a derived boolean flag, not a raw metric. It provides a simple yes/no answer for anomaly detection without requiring consumers to know the threshold value. The threshold (0.1) is defined centrally in `leyline/__init__.py` as `DEFAULT_ENTROPY_COLLAPSE_THRESHOLD`.
>
> **Relationship to TELE-120:** The raw `entropy` metric (TELE-120) provides gradual health status with warning/critical thresholds. This flag (`entropy_collapsed`) is specifically for binary anomaly detection - when True, the policy has definitively failed.
>
> **Threshold Rationale:** The 0.1 threshold assumes normalized entropy values from `MaskedCategorical.entropy()`. Values below 0.1 indicate the policy is <10% of maximum entropy, meaning it has become nearly deterministic. At this point, the agent is no longer exploring effectively.
>
> **Consumer Integration:** AnomalyStrip uses this flag alongside `kl_divergence > 0.05` to set `ppo_issues=True`. The Vue PolicyDiagnostics component shows this prominently with critical styling.
