# TamiyoBrain Widget: DRL Specialist Analysis

**Date:** 2025-12-24
**Reviewer:** DRL Expert (Claude)
**Status:** Draft findings for UX implementation

## Executive Summary

The TamiyoBrain widget displays 3 gauges + action distribution + decision carousel, but the system captures 20+ PPO metrics that are never shown. The most critical omission is **Explained Variance**, which is THE diagnostic for whether the value function is learning.

---

## 1. Current State: What We're Showing

### TamiyoState Schema Fields (schema.py:364-433)

| Field | Captured | Displayed |
|-------|----------|-----------|
| `entropy` | Yes | **Yes** (gauge) |
| `clip_fraction` | Yes | No |
| `kl_divergence` | Yes | **Yes** (gauge) |
| `explained_variance` | Yes | No |
| `policy_loss` | Yes | No |
| `value_loss` | Yes | **Yes** (gauge) |
| `entropy_loss` | Yes | No |
| `grad_norm` | Yes | No |
| `advantage_mean/std/min/max` | Yes | No |
| `ratio_mean/min/max/std` | Yes | No |
| `dead_layers/exploding_layers` | Yes | No |
| `entropy_collapsed` | Yes | No |
| `head_slot_entropy/grad_norm` | Yes | No |
| `head_blueprint_entropy/grad_norm` | Yes | No |

### Current Widget Components
1. **Action Distribution Bar** - Stacked bar: GERMINATE/SET_ALPHA_TARGET/FOSSILIZE/PRUNE/WAIT percentages
2. **Three Gauges:** Entropy, Value Loss, KL Divergence
3. **Decision Carousel** - Up to 3 pinnable decisions showing SAW/CHOSE/EXPECTED vs GOT

---

## 2. Priority 1: "Is Tamiyo Learning?" Diagnostics

### Critical Metrics (MUST ADD)

| Priority | Metric | Visualization | Rationale |
|----------|--------|---------------|-----------|
| **P0** | `explained_variance` | Gauge (0 to 1, warn <0) | THE diagnostic for value learning |
| **P0** | `clip_fraction` | Compact number with color | Policy update stability |
| **P1** | `advantage_mean/std` | Single line: "Adv: μ=X σ=Y" | Normalization health |
| **P1** | `ratio_max/min` | Alert only when critical | Policy divergence detection |
| **P2** | `policy_loss` trend | Sparkline | Actor improvement tracking |
| **P2** | Per-head entropies | Mini heatmap | Decision decomposition |
| **P3** | Gradient health | "✓ N/M layers" | Dead gradient detection |

### Healthy Ranges

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Explained Variance | > 0.5 | 0 to 0.5 | < 0 (harmful) |
| Clip Fraction | 0.1-0.25 | 0.25-0.3 | > 0.3 |
| Entropy | Controlled decay | Flat | → 0 (collapsed) |
| KL Divergence | 0.001-0.015 | > 0.015 | > 0.03 |
| Advantage Std | ~1.0 | > 2.0 | > 3.0 |
| Ratio Range | 0.8-1.2 | 0.5-2.0 | <0.3 or >2.0 |

### Diagnostic Decision Tree

```
Is Tamiyo Learning?
├─ entropy → 0?
│  └─ YES: Policy collapsed (deterministic) ⚠️
├─ explained_variance < 0?
│  └─ YES: Value function harmful, check returns computation
├─ clip_fraction > 0.3?
│  └─ YES: Policy updates too aggressive, reduce LR
├─ kl_divergence > target_kl consistently?
│  └─ YES: Early stopping too frequent, check learning rate
├─ value_loss not decreasing after batch 20+?
│  └─ YES: Value function not learning, check γ/λ/returns
└─ advantage_std >> 1 or << 1?
   └─ YES: Advantage normalization broken
```

---

## 3. Priority 2: "Cool Factor" for Demos

### High-Impact Visualizations

1. **Per-Head Entropy Display** - Shows multi-headed decision-making
   ```
   Heads: slot▓▓▓ bp░░░ sty▓░░ op▓▓░
   ```

2. **Value Prediction Accuracy** - Shows learning over time
   ```
   Prediction: ▁▂▃▄▅▆▇█ (accuracy improving)
   ```

3. **Seed Survival Metrics** - Highlights meta-learning
   - Germination rate vs fossilization rate
   - Average seed lifetime before decision
   - Seed survival rate = fossilized / (fossilized + pruned)

4. **Learning Velocity** - Rate of entropy decay vs accuracy improvement

---

## 4. Recommended Changes

### What to ADD
- Explained Variance gauge (P0)
- Clip Fraction indicator (P0)
- Advantage stats line (P1)
- Ratio bounds alert (P1)
- Policy loss sparkline (P2)
- Per-head entropy mini-display (P2)
- Gradient health summary (P3)

### What to REMOVE/REDUCE
- Reduce Decision Carousel from 3→2 slots
- Value Loss gauge - keep but reduce prominence (EV is more interpretable)

### What to CHANGE
- Entropy gauge: Add trend arrow, phase label
- KL gauge: Highlight when triggering early stopping
- Action distribution: Add delta indicators

---

## 5. Implementation Notes

**Data Availability:** All P0/P1 metrics are already in `TamiyoState` and populated from `PPO_UPDATE_COMPLETED` events. Gap is purely widget display.

**Missing from Telemetry:**
- Per-head entropies (PPOAgent computes but may not emit)
- Per-head gradient norms (PPOAgent computes but may not emit)

**Files to Modify:**
- `src/esper/karn/sanctum/widgets/tamiyo_brain.py` - Display logic
- `src/esper/karn/sanctum/schema.py` - Already has fields
- `src/esper/simic/telemetry/emitters.py` - Add per-head metrics if needed
