# Telemetry Record: [TELE-120] Entropy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-120` |
| **Name** | entropy |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the policy maintaining healthy exploration, or is it collapsing to deterministic behavior?"

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
| **Units** | nats (natural log base) |
| **Range** | `[0.0, HEAD_MAX_ENTROPIES[head]]` — varies by action head |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Policy entropy measures the "spread" of the action distribution. Computed as:
>
> H(π) = -Σ π(a|s) log π(a|s)
>
> High entropy = exploring many actions equally. Low entropy = converging on specific actions.
> For categorical distributions, max entropy = log(n_actions).
> This metric is critical for detecting policy collapse during training.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `entropy > 0.3` | Policy maintaining exploration |
| **Warning** | `0.1 < entropy <= 0.3` | Exploration declining, monitor closely |
| **Critical** | `entropy <= 0.1` | Policy collapse imminent or occurring |

**Threshold Source:** `TUIThresholds.ENTROPY_WARNING = 0.3`, `TUIThresholds.ENTROPY_CRITICAL = 0.1`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after action probability computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._handle_ppo_update()` (via PPOUpdatePayload) |
| **Line(s)** | Entropy computed in policy loss calculation, emitted via telemetry |

```python
# Entropy computed from action distribution during PPO update
entropy = dist.entropy().mean()
# Emitted as part of PPOUpdatePayload.entropy field
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `PPOUpdatePayload.entropy` field | `simic/agent/ppo.py` → `leyline/payloads.py` |
| **2. Collection** | Event payload with `entropy` field | `leyline/telemetry.py` (TelemetryEvent) |
| **3. Aggregation** | `SanctumAggregator._handle_ppo_update()` | `karn/sanctum/aggregator.py` line 798-800 |
| **4. Delivery** | Written to `snapshot.tamiyo.entropy` | `karn/sanctum/schema.py` (TamiyoState) |

```
[PPOAgent._compute_policy_loss()] --entropy--> [PPOUpdatePayload.entropy]
  --> [TelemetryEvent(PPO_UPDATE_COMPLETED)]
  --> [SanctumAggregator._handle_ppo_update()]
  --> [TamiyoState.entropy]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `entropy` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.entropy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~833 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` lines 210-215 | Entropy threshold detection for status (critical/warning) |
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` lines 145-170 | Entropy trend with velocity and collapse detection |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` lines 170-200 | Displayed as metric with sparkline and threshold coloring |
| ActionHeadsPanel | `widgets/tamiyo_brain/action_heads_panel.py` | Per-head entropy breakdown for multi-head collapse detection |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes and emits entropy during update
- [x] **Transport works** — Event includes entropy field in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.entropy: float = 0.0` (line 833)
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — All widgets access `snapshot.tamiyo.entropy` directly
- [x] **Display is correct** — Value renders with appropriate formatting and sparkline
- [x] **Thresholds applied** — StatusBanner and HealthStatusPanel use 0.1/0.3 thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_entropy_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_entropy` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_entropy_reaches_tui` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe StatusBanner entropy status detection
4. Observe HealthStatusPanel entropy row with velocity indicator
5. Verify PPOLossesPanel shows entropy with sparkline
6. Trigger entropy decline to verify warning/critical coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Action distribution | computation | Requires valid policy forward pass with valid logits |
| Entropy computation | operation | Categorical distribution entropy formula (base e) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `entropy_velocity` | derived | d(entropy)/d(batch) computed from entropy_history |
| `collapse_risk_score` | derived | Combines entropy, velocity, and thresholds for risk assessment |
| `entropy_clip_correlation` | derived | Pearson correlation with clip_fraction (collapse pattern) |
| StatusBanner status | display | Drives FAIL/WARN status when critical |
| HealthStatusPanel status | display | Shows collapse risk and velocity context |
| PPOLossesPanel color coding | display | Renders entropy with threshold-based coloring |
| Auto-intervention system | system | Could trigger on critical threshold breach |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-06-15 | Initial | Created with PPO implementation |
| 2024-09-20 | Refactor | Moved to TamiyoState from flat snapshot |
| 2025-01-03 | Audit | Verified wiring in telemetry audit (TELE-120) |

---

## 8. Notes

> **Design Decision:** Entropy is averaged across the batch rather than per-sample to reduce noise. Individual sample entropy would be too volatile for meaningful monitoring.
>
> **Known Behavior:** During warmup (first 50 batches), entropy may show artificially high values due to random initialization. The warmup period in StatusBanner accounts for this via WARMUP_BATCHES constant.
>
> **Per-Head Breakdown:** The system also tracks per-head entropy (head_slot_entropy, head_blueprint_entropy, etc.) for more granular collapse detection in multi-head policy architectures. These are separate telemetry metrics (TELE-121+) but use the same threshold logic.
>
> **Velocity Computation:** Entropy velocity is computed via linear regression on entropy_history (last 10 samples) using `compute_entropy_velocity()` from schema.py. Negative velocity indicates declining entropy (concerning), while positive indicates healthy exploration increases.
>
> **Collapse Pattern Detection:** Entropy-clip correlation combines entropy and clip_fraction to detect specific collapse pattern: low entropy + high negative correlation + high clip = policy ratio explosion during entropy collapse.
>
> **Future Improvement:** Consider adaptive thresholds based on action space size. Current fixed thresholds (0.1/0.3) assume ~4-action distributions (log(4) ≈ 1.39 max entropy).

