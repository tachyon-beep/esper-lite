# Telemetry Record: [TELE-130] PPO Clip Fraction

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-130` |
| **Name** | PPO Clip Fraction |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "How frequently is the PPO clipping mechanism activating during policy updates? Is policy drift occurring in a specific direction (upward or downward)?"

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
| **Units** | fraction (0.0-1.0) |
| **Range** | `[0.0, 1.0]` — percentage of timesteps where clipping was active |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> The PPO clipping fraction represents the proportion of timesteps where the importance sampling ratio exceeded the clipping bounds during a policy update. Computed as:
>
> clip_fraction = mean(|r - 1| > clip_ratio)
>
> where r is the joint importance sampling ratio (product across all action heads), and clip_ratio is typically 0.2.
>
> High clip fraction (>0.25) = policy is changing rapidly, or old policy and new policy diverging
> Low clip fraction (<0.1) = conservative policy updates, limited change
>
> The clip_fraction also tracks directional asymmetry via clip_fraction_positive (r > 1+ε) and clip_fraction_negative (r < 1-ε), which indicates if policy probability increases or decreases are being capped separately.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `clip_fraction < 0.25` | Moderate policy updates, clipping rare |
| **Warning** | `0.25 <= clip_fraction < 0.30` | Frequent clipping, policy drift beginning |
| **Critical** | `clip_fraction >= 0.30` | Severe clipping, policy divergence detected |

**Threshold Source:** `TUIThresholds.CLIP_WARNING = 0.25`, `TUIThresholds.CLIP_CRITICAL = 0.30` (from `/home/john/esper-lite/src/esper/karn/constants.py`)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after policy gradient clipping computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` (inner loop during policy loss computation) |
| **Line(s)** | 750-759 |

```python
# Compute clipping metrics on joint importance ratio (product across all heads)
joint_ratio = torch.exp(joint_log_ratio_clamped)  # Line 702
clip_fraction_t = ((joint_ratio - 1.0).abs() > self.clip_ratio).float().mean()  # Line 750
clip_pos = (joint_ratio > 1.0 + self.clip_ratio).float().mean()  # Line 753
clip_neg = (joint_ratio < 1.0 - self.clip_ratio).float().mean()  # Line 754
clip_metrics = torch.stack([clip_fraction_t, clip_pos, clip_neg]).cpu().tolist()  # Line 756
metrics["clip_fraction"].append(clip_metrics[0])  # Line 757
metrics["clip_fraction_positive"].append(clip_metrics[1])  # Line 758
metrics["clip_fraction_negative"].append(clip_metrics[2])  # Line 759
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` function receives metrics dict | `simic/telemetry/emitters.py` |
| **2. Collection** | `PPOUpdatePayload` dataclass with `clip_fraction` field | `leyline/telemetry.py` (line 622) |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` (line 817-819) |
| **4. Delivery** | Written to `snapshot.tamiyo.clip_fraction` | `karn/sanctum/schema.py` (line 834) |

```
[PPOAgent._update()] --metrics["clip_fraction"]--> [emit_ppo_update()] --PPOUpdatePayload.clip_fraction--> [SanctumAggregator.handle_ppo_update()] --> [TamiyoState.clip_fraction]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `clip_fraction` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.clip_fraction` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 834 |

**History field:** `clip_fraction_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))` (line 951)

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Critical/warning status detection (lines 211, 232) |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Displayed as gauge row with directional breakdown (lines 137-149) |
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Used in policy status calculation for collapse risk (line 253, 262) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPO update computes clip_fraction during policy loss computation
- [x] **Transport works** — Event includes clip_fraction field in PPOUpdatePayload
- [x] **Schema field exists** — TamiyoState.clip_fraction with default 0.0 and history deque
- [x] **Default is correct** — 0.0 appropriate before first PPO update (no clipping at init)
- [x] **Consumer reads it** — All 3 widgets access snapshot.tamiyo.clip_fraction
- [x] **Display is correct** — StatusBanner shows status, PPOLossesPanel shows gauge with directional breakdown
- [x] **Thresholds applied** — StatusBanner and PPOLossesPanel both use 0.25/0.30 thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `mutants/tests/simic/test_ppo.py` | [UNKNOWN] | `[ ]` |
| Unit (aggregator) | `mutants/tests/karn/sanctum/test_aggregator.py` | [UNKNOWN] | `[ ]` |
| Integration (end-to-end) | `mutants/tests/karn/sanctum/test_app_integration.py` | [UNKNOWN] | `[ ]` |
| Widget display | `mutants/tests/karn/sanctum/test_tamiyo_brain.py` | [UNKNOWN] | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training: `uv run esper ppo --episodes 20`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe PPOLossesPanel "Clip Frac" row showing clip_fraction value with directional breakdown (↑↓ percentages)
4. Verify value updates after each PPO update batch
5. Verify color coding: cyan (ok), yellow (warning >0.25), red (critical >0.30)
6. Check StatusBanner shows "WARN:Clip" when 0.25 < clip_fraction < 0.30
7. Check StatusBanner shows "FAIL:Clip" when clip_fraction >= 0.30

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Joint importance ratio computation | computation | Requires valid forward/backward pass with all heads |
| Policy gradient clipping enabled | configuration | clip_ratio must be > 0 |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-131` clip_fraction_positive | telemetry | Directional clipping (policy increases) |
| `TELE-132` clip_fraction_negative | telemetry | Directional clipping (policy decreases) |
| StatusBanner status | display | Drives WARN/FAIL status when thresholds exceeded |
| HealthStatusPanel policy_status | display | Used in collapse risk detection (entropy + clip correlation) |
| Auto-intervention system | system | May trigger policy reset if critical threshold sustained |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Created record with full wiring verification |

---

## 8. Notes

> **Design Decision:** Clip fraction is computed as a joint ratio (product across all 8 action heads) rather than per-head, because PPO clipping applies to the entire policy update. Per-head ratios are tracked separately (clip_fraction_positive/_negative) for directional diagnostics.
>
> **Directional Asymmetry:** The separate tracking of clip_pos and clip_neg (lines 753-754) detects asymmetric policy drift:
> - High clip_pos = many ratio > 1+ε (new policy giving higher probability) = expanding trust
> - High clip_neg = many ratio < 1-ε (new policy giving lower probability) = contracting trust
> - Balanced = symmetric policy updates
>
> **Interaction with Entropy:** When both entropy (TELE-001) and clip_fraction increase together, it indicates the agent is exploring new behaviors (high clipping due to divergence) while maintaining action diversity. When entropy decreases while clip_fraction stays high, this indicates policy collapse risk with diverging behavior.
>
> **Correlation Monitoring:** HealthStatusPanel monitors entropy_clip_correlation to detect the "collapse risk" pattern where entropy falls while clipping rises — this is the most dangerous training failure mode.
>
> **Known Limitation:** During early training (first 10-20 batches), clip_fraction may be artificially high because the old policy network and new policy network haven't converged yet. This is expected behavior and thresholds in StatusBanner account for warmup period.
>
> **Related Metrics:** The gradient_quality.clip_fraction_positive and gradient_quality.clip_fraction_negative fields in the schema provide additional breakdown for asymmetry detection. These are populated at lines 973-974 of the aggregator from the PPOUpdatePayload directional fields.
