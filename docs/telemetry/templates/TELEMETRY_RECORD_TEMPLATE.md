# Telemetry Record: [TELE-XXX] [Name]

> **Status:** `[ ] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-XXX` |
| **Name** | Human-readable name |
| **Category** | `training` / `policy` / `value` / `gradient` / `reward` / `seed` / `environment` / `infrastructure` / `decision` |
| **Priority** | `P0-critical` / `P1-important` / `P2-nice-to-have` |

## 2. Purpose

### What question does this answer?

> _Example: "Is the policy collapsing to deterministic behavior?"_

### Who needs this information?

- [ ] Training operator (real-time monitoring)
- [ ] Developer (debugging)
- [ ] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

### When is this information needed?

- [ ] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` / `int` / `bool` / `str` / `list[T]` / `dict[K,V]` / `deque[T]` |
| **Units** | e.g., "percentage (0-100)", "nats", "epochs", "seconds" |
| **Range** | e.g., "[0.0, 1.0]", "(-inf, inf)", "non-negative" |
| **Precision** | e.g., "3 decimal places", "integer" |
| **Default** | Value when unavailable |

### Semantic Meaning

> _Describe what this value represents in RL/ML terms. Include formulas if applicable._
>
> Example: "Entropy of the policy distribution, computed as H(π) = -Σ π(a|s) log π(a|s)"

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | e.g., `0.5 < value < 2.0` | Normal operating range |
| **Warning** | e.g., `value < 0.3` or `value > 3.0` | Potential issue developing |
| **Critical** | e.g., `value < 0.1` or `value > 5.0` | Immediate attention required |

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Where the raw data is computed |
| **File** | `/path/to/source/file.py` |
| **Function/Method** | `ClassName.method_name()` or `function_name()` |
| **Line(s)** | Approximate line numbers |

```python
# Emitter code snippet (key lines only)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | How data leaves source | |
| **2. Collection** | How data is gathered | |
| **3. Aggregation** | How data is processed | |
| **4. Delivery** | How data reaches schema | |

```
[Source] --> [Mechanism] --> [Collector] --> [Aggregator] --> [Schema Field]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | e.g., `TamiyoState`, `EnvState`, `SeedState` |
| **Field** | e.g., `entropy`, `advantage_std` |
| **Path from SanctumSnapshot** | e.g., `snapshot.tamiyo.entropy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | Line number of field definition |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| e.g., HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displayed with sparkline |
| e.g., StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Used for status detection |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — Code computes and emits this value
- [ ] **Transport works** — Value reaches aggregator
- [ ] **Schema field exists** — Field defined in dataclass
- [ ] **Default is correct** — Field has appropriate default
- [ ] **Consumer reads it** — Widget accesses the field
- [ ] **Display is correct** — Value renders as expected
- [ ] **Thresholds applied** — Color coding matches spec

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | | | `[ ]` |
| Unit (aggregator) | | | `[ ]` |
| Integration (end-to-end) | | | `[ ]` |
| Visual (TUI snapshot) | | | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI
3. Observe [widget name]
4. Verify [specific behavior]
5. Trigger [condition] to verify threshold coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| e.g., `TELE-001` | telemetry | Requires entropy to compute velocity |
| e.g., PPO update cycle | event | Only populated after first update |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| e.g., `TELE-045` | telemetry | Uses this for trend computation |
| e.g., Auto-intervention | system | Triggers on critical threshold |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| YYYY-MM-DD | Name | Initial creation |
| | | |

---

## 8. Notes

> _Additional context, known issues, future improvements, or design decisions._
