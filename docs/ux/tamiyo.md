# Tamiyo Panel Restructuring Design Document

**Status:** Proposed
**Authors:** UX Specialist (Lyra), DRL Expert (Yzmir), Claude
**Date:** 2026-01-08
**Location:** `src/esper/karn/sanctum/widgets/tamiyo/`

---

## Executive Summary

The Tamiyo panel in Sanctum TUI currently comprises **12 separate panels** displaying PPO training diagnostics. This design document proposes consolidating to **6 primary panels** organized around diagnostic workflows rather than data sources, while preserving all unique metrics.

### Key Changes
- Reduce panel count from 12 to 6 (plus 2 auxiliary: Decisions, EventLog)
- Eliminate 2 confirmed duplicate metrics
- Reorganize by diagnostic question ("Is my policy learning?") not data source
- Add section headers for visual grouping
- Improve accessibility with status symbols alongside colors

---

## Current State Analysis

### Panel Inventory (12 Panels)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOP ROW (height: 14)                                                        │
├───────────────────────┬───────────────────────┬─────────────────────────────┤
│   NarrativePanel      │   PPOLossesPanel      │      SlotsPanel             │
│   (variable width)    │   (56 chars)          │      (52 chars)             │
├───────────────────────┴───────────────────────┴─────────────────────────────┤
│ BOTTOM LEFT                          │ CENTER (56 chars)  │ RIGHT (52 chars)│
├──────────────────────────────────────┼────────────────────┼─────────────────┤
│   ActionHeadsPanel (1fr height)      │ HealthStatusPanel  │ ActionContext   │
│   - Head metrics (8 heads)           │ (2fr)              │ (full height)   │
│   - Decision carousel                │                    │                 │
│   - Gradient diagnostics             ├────────────────────┤                 │
├──────────────────────────────────────┤ ValueDiagnostics   │                 │
│ EpisodeMetrics  │ TorchStability     │ (1fr)              │                 │
│ (9 rows)        │ (9 rows)           ├────────────────────┤                 │
│                 │                    │ CriticCalibration  │                 │
│                 │                    │ (9 rows)           │                 │
├─────────────────┴────────────────────┴────────────────────┼─────────────────┤
│                                                           │ DecisionsColumn │
│                                                           │ (scrollable)    │
├───────────────────────────────────────────────────────────┼─────────────────┤
│                                                           │ EventLog        │
│                                                           │ (12 rows)       │
└───────────────────────────────────────────────────────────┴─────────────────┘
```

### Problems Identified

#### 1. Information Architecture Fragmentation
Metrics are organized by **data source** (where they come from in code) rather than **diagnostic workflow** (what questions they answer).

| Diagnostic Question | Currently Scattered Across |
|---------------------|---------------------------|
| "Is my value function calibrated?" | HealthStatusPanel, ValueDiagnosticsPanel, CriticCalibrationPanel |
| "Are gradients healthy?" | PPOLossesPanel, HealthStatusPanel, ActionHeadsPanel |
| "Is entropy appropriate?" | NarrativePanel, HealthStatusPanel, EpisodeMetricsPanel, ActionHeadsPanel |

#### 2. Confirmed Duplicate Metrics

| Metric | Location 1 | Location 2 | Resolution |
|--------|-----------|-----------|------------|
| Explained Variance | PPOLossesPanel | CriticCalibrationPanel | Keep in CriticCalibration only |
| Value Range/Span | HealthStatusPanel | CriticCalibrationPanel | Keep in CriticCalibration only |

#### 3. Cognitive Overload
- 12 panels exceed typical terminal viewport (24-50 lines)
- Requires scrolling to see all metrics
- No visual grouping of related panels
- All borders same weight—no hierarchy

#### 4. Accessibility Issues
- Status indicated by color only (green/yellow/red)
- "Dim" text may fail contrast requirements
- Sparkline Unicode blocks (▁▂▃▅▆▇█) inconsistent across terminals

---

## Proposed Design

### Panel Structure (6 + 2 Auxiliary)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TAMIYO ─ A                                      │
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━ POLICY HEALTH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├─────────────────────────┬───────────────────────┬───────────────────────────┤
│      NARRATIVE          │  POLICY OPTIMIZATION  │     SEED LIFECYCLE        │
│   (NOW/WHY/NEXT)        │                       │                           │
│                         │                       │                           │
├─────────────────────────┴───────────────────────┴───────────────────────────┤
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━ DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├─────────────────────────────────────┬───────────────────────────────────────┤
│     VALUE FUNCTION QUALITY          │      ACTION SPACE EXPLORATION         │
│                                     │                                       │
│                                     │                                       │
├─────────────────────────────────────┤                                       │
│     GRADIENT & ADVANTAGE HEALTH     │                                       │
│                                     │                                       │
├─────────────────────────────────────┴───────────────────────────────────────┤
│━━━━━━━━━━━━━━━━━━━━━━━━━━ INFRASTRUCTURE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├─────────────────────────────────────┬───────────────────────────────────────┤
│         INFRASTRUCTURE              │           DECISIONS                   │
│                                     ├───────────────────────────────────────┤
│                                     │           EVENT LOG                   │
└─────────────────────────────────────┴───────────────────────────────────────┘
```

### Detailed Panel Specifications

---

#### Panel 1: NARRATIVE (Unchanged)

**Purpose:** High-level system interpreter; answers "What's happening right now?"

**Location:** Top-left, variable width (min 38 chars)
**Height:** 14 rows

**Metrics:**
| Metric | Format | Source |
|--------|--------|--------|
| Group ID | `A` / `B` / `C` with color | SanctumSnapshot.group_id |
| Training Status | Spinner during warmup, status text | Computed from ~20 criteria |
| Overall Health | OK / WARNING / CRITICAL | Aggregated from all panels |
| KL Divergence | Value + trend arrow | PPOUpdatePayload.approx_kl |
| Round Progress | `current/max` | SanctumSnapshot.episodes_completed |
| Memory Usage | Percentage (if >0%) | TorchStabilityPayload.cuda_memory_* |
| NaN/Inf Count | `⚠ NaN:5 Inf:0` (MOVED HERE) | PPOUpdatePayload.nan_count, inf_count |

**NOW/WHY/NEXT Framework:**
- **NOW:** Current system state in one line
- **WHY:** Top 3 issues blocking progress (ranked)
- **NEXT:** Contextual recovery guidance

**Changes from Current:**
- ADD: NaN/Inf counts (moved from ActionHeadsPanel for critical visibility)
- No other changes—this panel is well-designed

---

#### Panel 2: POLICY OPTIMIZATION (Consolidated)

**Purpose:** Answers "Is my policy learning? Is it learning stably?"

**Location:** Top-center, 56 chars
**Height:** 14 rows

**Metrics (from PPOLossesPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Policy Loss | Sparkline + value + trend | Primary optimization signal |
| KL Divergence | Sparkline (12 chars) | Trust region; >0.02 = instability |
| Clip Fraction | `↑0.12 ↓0.08` breakdown | PPO clipping; >0.3 = too aggressive |
| Joint Ratio Max | Single value | Multi-head coordination |

**Metrics (absorbed from HealthStatusPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Entropy | Value + trend arrow | Exploration level; collapse = premature convergence |
| Policy State | `WARMUP/STABLE/DRIFT/RISK` | High-level diagnostic |

**Metrics (absorbed from EpisodeMetricsPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Yield Rate | Percentage + trend | Task-specific performance |
| Slot Utilization | `6/8 (75%)` | Resource efficiency |

**Removed (duplicate):**
- ~~Explained Variance~~ → moved to VALUE FUNCTION QUALITY

**Layout:**
```
┌─ POLICY OPTIMIZATION ─────────────────────────┐
│ Policy Loss  ▁▂▃▂▄▅▃▂  -0.023 ↘              │
│ KL Diverge   ▁▁▂▂▃▃▂▁   0.008 ✓              │
│ Clip Frac    ↑ 0.12    ↓ 0.08   [0.20 total] │
│ Ratio Max    1.82                             │
│ ─────────────────────────────────────────────│
│ Entropy      [████░░░░]  2.14 ↗  ✓ exploring │
│ Policy State STABLE                           │
│ ─────────────────────────────────────────────│
│ Yield Rate   73% ↗      Slot Util  6/8 (75%) │
│ Steps/germ   45         Steps/prune 12       │
└───────────────────────────────────────────────┘
```

---

#### Panel 3: SEED LIFECYCLE (Relocated)

**Purpose:** Answers "What's the state of the seed population?"

**Location:** Top-right, 52 chars (moved from scattered location)
**Height:** 14 rows

**Metrics (from SlotsPanel):**
| Metric | Format | Description |
|--------|--------|-------------|
| Stage Distribution | Proportional bars | DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILIZED |
| Active Count | Cyan number | Currently active seeds |
| Germination Count | Cumulative | Total germinations this run |
| Prune Count | Red if > fossils | Total prunes (warning if pruning > fossilizing) |
| Fossilization Count | Cumulative | Successfully integrated seeds |
| Per-Episode Rates | With trend arrows | Germination/prune/fossil rate per episode |
| Lifespan (μ) | Average epochs | Mean seed lifespan |
| Blend Success Rate | Percentage | Successful blends / attempted blends |

**Layout:**
```
┌─ SEED LIFECYCLE ──────────────────────────────┐
│ DORMANT     [██░░░░░░░░░░░░░░░░░░░░░░]  2    │
│ GERMINATED  [████░░░░░░░░░░░░░░░░░░░░]  1    │
│ TRAINING    [████████████░░░░░░░░░░░░]  3    │
│ BLENDING    [████░░░░░░░░░░░░░░░░░░░░]  1    │
│ HOLDING     [░░░░░░░░░░░░░░░░░░░░░░░░]  0    │
│ FOSSILIZED  [████░░░░░░░░░░░░░░░░░░░░]  1    │
│ ─────────────────────────────────────────────│
│ Active: 5      Germ: 12    Prune: 3   Foss: 8│
│ Rates/ep  germ: 0.4↗  prune: 0.1→  foss: 0.3↗│
│ Lifespan  μ=45 epochs    Blend success: 85%  │
└───────────────────────────────────────────────┘
```

---

#### Panel 4: VALUE FUNCTION QUALITY (Consolidated)

**Purpose:** Answers "Is my critic well-calibrated? Can I trust the advantages?"

**Location:** Middle-left, 56 chars
**Height:** 18 rows (2fr equivalent)

**Metrics (from CriticCalibrationPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Explained Variance | Bar gauge + value | THE key metric; <0.5 = critic is noise |
| V-Return Correlation | Bar gauge + icon | Should be >0.8; low = value not tracking returns |
| TD Error | Mean ± std | Mean ~0; high std = inconsistent |
| Bellman Error | Single value | Temporal consistency |
| Calibration Summary | `OK/WEAK/BAD` | Aggregate assessment |

**Metrics (from ValueDiagnosticsPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Return Percentiles | p10, p50, p90 | Distribution shape for advantage normalization |
| Return σ | Single value | Reward scale |
| Return Skewness | Single value | Distribution asymmetry |
| Return Mean | Single value | Central tendency |
| Trend Indicator | Arrow | Overall trajectory |

**Metrics (from PPOLossesPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Value Loss | Sparkline + trend | Critic optimization signal |
| Lv/Lp Ratio | Single value | Critic-actor learning balance; >10 may need rebalancing |

**Metrics (absorbed from HealthStatusPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Value Range | min/max (DEDUPLICATED) | Kept here, removed from HealthStatus |

**Layout:**
```
┌─ VALUE FUNCTION QUALITY ──────────────────────┐
│ Explained Var  [██████████░░]  0.72 ✓        │
│ V-Return Corr  [█████████░░░]  0.85 ↗        │
│ TD Error       μ = -0.02      σ = 0.31       │
│ Bellman Error  0.04                           │
│ Calibration    ✓ OK                           │
│ ─────────────────────────────────────────────│
│ Returns   p10: -12    p50: +34    p90: +78   │
│           σ = 8.2     skew = -0.1   ↗        │
│           μ = +28.4   range: [-45, +120]     │
│ ─────────────────────────────────────────────│
│ Value Loss   ▁▂▃▂▄▃▂▁  0.052 ↘               │
│ Lv/Lp Ratio  3.2  (actor-critic balanced)    │
└───────────────────────────────────────────────┘
```

---

#### Panel 5: GRADIENT & ADVANTAGE HEALTH (Reorganized)

**Purpose:** Answers "Are gradients flowing correctly? Are advantages well-behaved?"

**Location:** Middle-left (below Value Function), 56 chars
**Height:** 12 rows (1fr equivalent)

**Metrics (from HealthStatusPanel):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Advantage Mean ± Std | `μ=0.00 σ=1.02` | Mean ~0 after normalization; std ~1 |
| Advantage Skewness | Single value | Heavy tails = outlier returns |
| Advantage Kurtosis | Single value | Distribution shape |
| Advantage Positive Ratio | Percentage | ~50% is healthy |
| Gradient Norm | Sparkline + value | Stability; spikes = instability |
| Log Prob Extremes | `[min, max]` | Numerical stability; very negative = near-zero probs |
| Observation Health | `OK/WARNING` | Input distribution sanity |

**Metrics (from ActionHeadsPanel footer):**
| Metric | Format | Why It Matters |
|--------|--------|----------------|
| Gradient Flow CV | Single value | Coefficient of variation across layers |
| Dead Layers | Count | Layers with zero gradients |
| Exploding Layers | Count | Layers with extreme gradients |
| Gradient Clip Fraction | Percentage | Gradient clipping activation |

**Removed (moved to NARRATIVE for visibility):**
- ~~NaN/Inf Counts~~ → critical metric, moved to top-level NARRATIVE panel

**Layout:**
```
┌─ GRADIENT & ADVANTAGE HEALTH ─────────────────┐
│ Advantage  μ = 0.00   σ = 1.02   pos = 51%   │
│            skew = -0.1    kurtosis = 3.2     │
│ ─────────────────────────────────────────────│
│ Grad Norm  ▁▂▃▂▁▂▃▂▁  1.24 ✓                 │
│ Log Prob   range: [-12.3, -0.1]  ✓           │
│ Obs Health ✓ OK                               │
│ ─────────────────────────────────────────────│
│ Gradient Flow  CV: 0.15   Dead: 0  Explode: 0│
│ Grad Clip Frac 0.02                           │
└───────────────────────────────────────────────┘
```

---

#### Panel 6: ACTION SPACE EXPLORATION (Reorganized)

**Purpose:** Answers "Is the policy exploring appropriately across all action dimensions?"

**Location:** Middle-right, 52 chars
**Height:** Full remaining height (~30 rows)

**Metrics (from ActionHeadsPanel):**
| Metric | Format | Description |
|--------|--------|-------------|
| Per-Head Entropy | 8 values with bars | Which heads are collapsing? |
| Per-Head Gradient Norm | 8 values | Gradient distribution across heads |
| Per-Head Ratio | 8 values (π_new/π_old) | Which heads changing most aggressively? |
| Head State Indicators | `●/○/◐/◇/▲` | Synthesized health per head |
| Decision Carousel | 5 recent decisions | Recent action sequence |
| Gradient Flow CV | Single value | Overall head coordination |

**Metrics (absorbed from ActionContext):**
| Metric | Format | Description |
|--------|--------|-------------|
| Action Sequence | 12-step history | Pattern detection (STUCK/THRASH/ALPHA_OSC) |
| Action Distribution | Bar chart | Round vs run-to-date distribution |

**Layout:**
```
┌─ ACTION SPACE EXPLORATION ────────────────────┐
│ HEAD         ENTROPY    GRAD     RATIO  STATE│
│ op           [████░]    0.02     1.01    ●   │
│ slot         [███░░]    0.03     0.98    ●   │
│ blueprint    [█████]    0.01     1.02    ●   │
│ style        [████░]    0.02     0.99    ●   │
│ tempo        [████░]    0.02     1.00    ●   │
│ alpha_target [███░░]    0.03     0.97    ○   │
│ alpha_speed  [████░]    0.02     1.01    ●   │
│ alpha_curve  [█████]    0.01     1.02    ●   │
│ ─────────────────────────────────────────────│
│ RECENT DECISIONS                              │
│ ┌─────┬────┬─────┬─────┬─────┬──────┬──────┐ │
│ │ OP  │SLOT│BLUEP│STYLE│TEMPO│α_TGT │α_SPD │ │
│ ├─────┼────┼─────┼─────┼─────┼──────┼──────┤ │
│ │GERM │ r0 │ CNN │ add │ med │ 0.5  │ slow │ │
│ │WAIT │ -- │ --  │ --  │ --  │ --   │ --   │ │
│ │...  │    │     │     │     │      │      │ │
│ └─────┴────┴─────┴─────┴─────┴──────┴──────┘ │
│ ─────────────────────────────────────────────│
│ SEQUENCE ✓✓✓✗✓✓✓✓✓✓✗✓  Pattern: STABLE       │
│ Gradient Flow CV: 0.15                        │
└───────────────────────────────────────────────┘
```

---

#### Panel 7: INFRASTRUCTURE (Unchanged)

**Purpose:** Answers "Are there system-level issues affecting training?"

**Location:** Bottom-left, variable width
**Height:** 9 rows

**Metrics (from TorchStabilityPanel):**
| Metric | Format | Description |
|--------|--------|-------------|
| torch.compile Status | `backend:mode` or `EAGER` | JIT compilation state |
| CUDA Memory | `allocated/reserved` (GB) + % | Memory pressure |
| CUDA Peak Memory | GB | Maximum allocation |
| CUDA Fragmentation | Percentage | Memory efficiency |
| DataLoader Wait Ratio | Decimal | I/O bottleneck detection |
| PPO Update Time | Milliseconds | Compute efficiency |

**Layout:**
```
┌─ INFRASTRUCTURE ──────────────────────────────┐
│ torch.compile  inductor:max-autotune         │
│ CUDA Memory    2.1G / 4.0G (52%)  peak: 3.8G │
│ Fragmentation  8%                             │
│ DataLoader     wait ratio: 0.02 ✓            │
│ PPO Update     142ms                          │
└───────────────────────────────────────────────┘
```

---

#### Panel 8: DECISIONS (Unchanged)

**Purpose:** Scrollable history of recent policy decisions

**Location:** Bottom-right, 52 chars
**Height:** Variable (scrollable)

No changes to current implementation.

---

#### Panel 9: EVENT LOG (Unchanged)

**Purpose:** Raw telemetry event stream for debugging

**Location:** Bottom-right (below Decisions), 52 chars
**Height:** 12 rows (consider reducing to 8)

No changes to current implementation.

---

## Complete Metrics Inventory

### Preserved Metrics (45 unique)

| Category | Count | Metrics |
|----------|-------|---------|
| **Narrative** | 7 | Group ID, Training Status, Overall Health, KL trend, Round Progress, Memory %, NaN/Inf counts |
| **Policy Optimization** | 10 | Policy Loss, KL Divergence, Clip Fraction (↑/↓), Joint Ratio Max, Entropy, Policy State, Yield Rate, Slot Utilization, Steps/germ, Steps/prune |
| **Seed Lifecycle** | 10 | Stage Distribution (6), Active Count, Germ/Prune/Foss Counts, Rates (3), Lifespan, Blend Success |
| **Value Function** | 12 | EV, V-Return Corr, TD Error (μ/σ), Bellman Error, Calibration, Return p10/p50/p90, Return σ/skew/mean, Value Loss, Lv/Lp Ratio, Value Range |
| **Gradient & Advantage** | 10 | Adv μ/σ/skew/kurt/pos%, Grad Norm, Log Prob extremes, Obs Health, Grad Flow CV, Dead/Exploding layers, Clip Frac |
| **Action Space** | 26 | Per-head Entropy (8), Per-head Grad (8), Per-head Ratio (8), Head States, Decision Carousel, Sequence Pattern |
| **Infrastructure** | 6 | torch.compile, CUDA mem/peak/frag, DataLoader wait, PPO time |

### Removed Metrics (2 duplicates)

| Metric | Removed From | Kept In |
|--------|-------------|---------|
| Explained Variance | PPOLossesPanel | VALUE FUNCTION QUALITY |
| Value Range | HealthStatusPanel | VALUE FUNCTION QUALITY |

---

## Accessibility Improvements

### Color + Symbol Redundancy

All status indicators must include both color AND symbol:

| Status | Color | Symbol | Example |
|--------|-------|--------|---------|
| OK/Healthy | Green | ✓ | `✓ 0.72` |
| Warning | Yellow | ⚠ | `⚠ 0.35` |
| Critical | Red | ✗ | `✗ NaN` |
| Neutral/Info | Cyan | ● | `● 1.02` |

### Label Width Standardization

All panels must use **13-character label columns** for vertical alignment:

```
# Good
Explained Var  [██████] 0.72
V-Return Corr  [█████░] 0.85
TD Error       μ=-0.02

# Bad (inconsistent widths)
EV        [██████] 0.72
V-Return Correlation [█████░] 0.85
TD Err    μ=-0.02
```

### Section Headers

Add horizontal dividers with labels between panel groups:

```python
# In tamiyo.py compose() method
yield Static("━━━━━━━━━━━━ POLICY HEALTH ━━━━━━━━━━━━", classes="section-header")
yield Horizontal(narrative_panel, policy_opt_panel, seed_lifecycle_panel)
yield Static("━━━━━━━━━━━━ DIAGNOSTICS ━━━━━━━━━━━━━━", classes="section-header")
# ...
```

### Terminal Fallbacks

Provide ASCII alternatives for Unicode-dependent elements:

| Unicode | ASCII Fallback |
|---------|---------------|
| `▁▂▃▄▅▆▇█` (sparkline) | `_.-=+*#@` |
| `━` (divider) | `-` |
| `✓✗⚠●○` (status) | `[OK][XX][!!][**][ ]` |

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)

**Risk:** Low
**Changes:**

1. Remove duplicate EV from `ppo_losses_panel.py`
2. Remove duplicate value range from `health_status_panel.py`
3. Add section headers in `tamiyo.py` compose()
4. Add status symbols to all `_status_style()` methods
5. Move NaN/Inf display to `narrative_panel.py`

### Phase 2: Panel Consolidation (4-6 hours)

**Risk:** Medium
**Changes:**

1. Create new `value_function_panel.py`:
   - Merge `value_diagnostics_panel.py` + `critic_calibration_panel.py`
   - Add Value Loss and Lv/Lp from PPOLossesPanel

2. Create new `gradient_health_panel.py`:
   - Extract advantage/gradient metrics from `health_status_panel.py`
   - Move gradient footer from `action_heads_panel.py`

3. Update `policy_optimization_panel.py` (rename from PPOLossesPanel):
   - Add entropy from HealthStatusPanel
   - Add yield/utilization from EpisodeMetricsPanel

4. Delete deprecated panels:
   - `value_diagnostics_panel.py`
   - `critic_calibration_panel.py`
   - `episode_metrics_panel.py`
   - `health_status_panel.py`

### Phase 3: Layout Restructure (2-3 hours)

**Risk:** Medium
**Changes:**

1. Update `tamiyo.py` layout:
   - Implement 3-section layout (Policy Health, Diagnostics, Infrastructure)
   - Adjust CSS for new panel arrangement

2. Update `tamiyo.tcss`:
   - Add `.section-header` styling
   - Adjust grid fractions for consolidated panels

3. Move SlotsPanel to right column (above Decisions)

### Phase 4: Testing & Polish (2 hours)

**Risk:** Low
**Changes:**

1. Update all tests referencing renamed/deleted panels
2. Manual testing with live training run
3. Screenshot comparison (before/after)
4. Update any documentation referencing old panel names

---

## Migration Guide

### For Existing Users

The consolidation preserves all unique metrics. Here's where to find metrics that moved:

| Old Location | New Location |
|-------------|--------------|
| PPOLossesPanel → Explained Variance | VALUE FUNCTION QUALITY |
| PPOLossesPanel → Value Loss | VALUE FUNCTION QUALITY |
| HealthStatusPanel → Entropy | POLICY OPTIMIZATION |
| HealthStatusPanel → Advantage stats | GRADIENT & ADVANTAGE HEALTH |
| HealthStatusPanel → Value range | VALUE FUNCTION QUALITY |
| EpisodeMetricsPanel → Yield/Util | POLICY OPTIMIZATION |
| ActionHeadsPanel → NaN/Inf | NARRATIVE (NOW line) |
| ActionHeadsPanel → Grad footer | GRADIENT & ADVANTAGE HEALTH |
| ValueDiagnosticsPanel → All | VALUE FUNCTION QUALITY |
| CriticCalibrationPanel → All | VALUE FUNCTION QUALITY |

---

## Appendix: Full Wireframe

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    TAMIYO ─ A                                        │
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ POLICY HEALTH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├────────────────────────────┬────────────────────────────┬───────────────────────────┤
│ NARRATIVE                  │ POLICY OPTIMIZATION        │ SEED LIFECYCLE            │
│ ─────────────────────────  │ ──────────────────────     │ ────────────────────────  │
│ NOW: Training stable ✓     │ Policy Loss ▁▂▃▂▄ -0.02 ↘ │ DORMANT   [██░░░░░░]  2   │
│ WHY: -                     │ KL Diverge  ▁▁▂▂▃  0.01 ✓ │ GERMINATE [████░░░░]  1   │
│ NEXT: Continue training    │ Clip Frac  ↑0.12 ↓0.08    │ TRAINING  [████████░]  3   │
│                            │ Ratio Max  1.82            │ BLENDING  [████░░░░░]  1   │
│ Group: A                   │ ────────────────────────── │ HOLDING   [░░░░░░░░░]  0   │
│ Round: 45/100              │ Entropy [████░░] 2.1 ↗ ✓  │ FOSSIL    [████░░░░░]  1   │
│ Memory: 52%                │ Policy State: STABLE       │ ────────────────────────  │
│ ⚠ NaN: 0  Inf: 0          │ ────────────────────────── │ Active: 5   Blend: 85%    │
│                            │ Yield: 73%↗  Util: 6/8    │ Rates: g:0.4 p:0.1 f:0.3  │
├────────────────────────────┴────────────────────────────┴───────────────────────────┤
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ DIAGNOSTICS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├──────────────────────────────────────────┬──────────────────────────────────────────┤
│ VALUE FUNCTION QUALITY                   │ ACTION SPACE EXPLORATION                 │
│ ────────────────────────────────────     │ ──────────────────────────────────────   │
│ Explained Var [██████████░░] 0.72 ✓     │ HEAD        ENTROPY   GRAD   RATIO STATE │
│ V-Return Corr [█████████░░░] 0.85 ↗     │ op          [████░]   0.02   1.01   ●    │
│ TD Error      μ=-0.02    σ=0.31         │ slot        [███░░]   0.03   0.98   ●    │
│ Bellman Error 0.04                       │ blueprint   [█████]   0.01   1.02   ●    │
│ Calibration   ✓ OK                       │ style       [████░]   0.02   0.99   ●    │
│ ────────────────────────────────────     │ tempo       [████░]   0.02   1.00   ●    │
│ Returns  p10:-12  p50:+34  p90:+78      │ alpha_tgt   [███░░]   0.03   0.97   ○    │
│          σ=8.2  skew=-0.1   ↗           │ alpha_spd   [████░]   0.02   1.01   ●    │
│ Value Loss  ▁▂▃▂▄▃▂  0.052 ↘            │ alpha_crv   [█████]   0.01   1.02   ●    │
│ Lv/Lp Ratio 3.2                          │ ──────────────────────────────────────   │
├──────────────────────────────────────────┤ RECENT DECISIONS                         │
│ GRADIENT & ADVANTAGE HEALTH              │ ┌────┬────┬─────┬─────┬─────┬─────────┐ │
│ ────────────────────────────────────     │ │ OP │SLOT│BLUEP│STYLE│TEMPO│α_TARGET │ │
│ Advantage μ=0.00  σ=1.02  pos=51%       │ ├────┼────┼─────┼─────┼─────┼─────────┤ │
│           skew=-0.1   kurtosis=3.2      │ │GERM│ r0 │ CNN │ add │ med │   0.5   │ │
│ ────────────────────────────────────     │ │WAIT│ -- │  -- │  -- │  -- │    --   │ │
│ Grad Norm   ▁▂▃▂▁▂▃▂▁  1.24 ✓           │ │WAIT│ -- │  -- │  -- │  -- │    --   │ │
│ Log Prob    [-12.3, -0.1] ✓             │ └────┴────┴─────┴─────┴─────┴─────────┘ │
│ Obs Health  ✓ OK                         │ ──────────────────────────────────────   │
│ ────────────────────────────────────     │ SEQUENCE ✓✓✓✗✓✓✓✓✓✓✗✓  STABLE          │
│ Grad Flow CV:0.15 Dead:0 Explode:0      │ Gradient Flow CV: 0.15                   │
├──────────────────────────────────────────┴──────────────────────────────────────────┤
│━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ INFRASTRUCTURE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
├──────────────────────────────────────────┬──────────────────────────────────────────┤
│ INFRASTRUCTURE                           │ DECISIONS (scrollable)                   │
│ ────────────────────────────────────     │ ┌────────────────────────────────────┐  │
│ torch.compile  inductor:max-autotune    │ │ [GERM] r0c0 CNN/add ep:42 env:0   │  │
│ CUDA Memory    2.1G/4.0G (52%) pk:3.8G  │ │ [WAIT] -- ep:41 env:0             │  │
│ Fragmentation  8%                        │ │ [FOSS] r0c1 ep:40 env:2           │  │
│ DataLoader     wait: 0.02 ✓             │ └────────────────────────────────────┘  │
│ PPO Update     142ms                     ├──────────────────────────────────────────┤
│                                          │ EVENT LOG                                │
│                                          │ 14:23:05 BATCH_COMPLETED batch=42       │
│                                          │ 14:23:04 PPO_UPDATE kl=0.008 ev=0.72    │
│                                          │ 14:23:03 SEED_LIFECYCLE GERMINATE r0c0  │
└──────────────────────────────────────────┴──────────────────────────────────────────┘
```

---

---

## Part 2: Multi-Mode TUI Architecture

**Status:** Design Review Complete
**Date:** 2026-01-08
**Reviewers:** UX Designer (Lyra), PyTorch Expert (Yzmir), DRL Expert (Yzmir)

### Original Proposal

The original proposal suggested 3 full-screen modes:

1. **Event View** - Full screen event log
2. **Tamiyo Mode** - Training diagnostics with thin event wedge
3. **Kasmina Mode** - Environment card view (Top 5, Bottom 5, Interesting envs)

Additional modes considered:
- **PyTorch Mode** - Infrastructure monitoring
- **DRL Mode** - Deep RL diagnostics
- **Curriculum Mode** - Staged difficulty progression

### Specialist Verdict: Single-Mode with View Switchers

**All three specialists converged on the same recommendation:** Do NOT implement full-screen modes. Instead, enhance the single-mode layout with progressive disclosure.

| Proposal | Verdict | Reasoning |
|----------|---------|-----------|
| Event View | ❌ | Current EventLog already has click-to-detail. Full-screen adds complexity without benefit. |
| Tamiyo Mode | ✅ | Already exists and is well-designed. Add drill-down modal for deep analysis. |
| Kasmina Mode | ⚠️ | Cards don't scale to 256 envs. Use Grid/Cards/List view switcher instead. |
| PyTorch Mode | ❌ | Infrastructure metrics are lightweight status indicators. Current `TorchStabilityPanel` is sufficient. |
| DRL Mode | ❌ | Overlaps significantly with Tamiyo. Add DRL drill-down modal instead. |
| Curriculum Mode | ⏸️ | Defer until Phase 3 TinyStories adds explicit curriculum. |

### Recommended Architecture

```
┌─ Run Header ─────────────────────────────────────────────────────────┐
│ Ep 47 │ 120/150 │ Batch 12/50 │ Runtime: 3h 14m │ [A/B: A selected] │
└──────────────────────────────────────────────────────────────────────┘
┌─ Anomaly Strip (ALWAYS VISIBLE - critical DRL + PyTorch alerts) ────┐
│ ⚠ Env 3 stalled (15 epochs) │ 🔥 Entropy: 0.24 │ cuda:0 95% ⚠      │
└──────────────────────────────────────────────────────────────────────┘
┌─ Environment Overview ───────────────────────────────────────────────┐
│ View: [Grid] [Cards] [List]  ← cycle with 'v' key                   │
│                                                                       │
│ Grid (256 envs):  ▓▓▒░░░░▒▓█▓▓▒░░░░▒▓█  (colored by status)        │
│ Cards: Top 5 + Bottom 5 + Flagged envs                               │
│ List: Full table, scrollable (current implementation)                │
└──────────────────────────────────────────────────────────────────────┘
┌─ Training Health (Tamiyo) ───────────────────────────────────────────┐
│ Tab: [PPO] [Health] [Actions] [Events]  ← cycle with Tab key        │
│                                                                       │
│ Press 'd' for DRL deep dive modal                                    │
│ Press 'a' for A/B comparison panel (when --dual-ab active)           │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Enhancements

#### 1. Kasmina Progressive Disclosure (View Switcher)

Instead of a separate Kasmina Mode, add a view switcher to the Environment Overview panel:

| View | Scale | Use Case |
|------|-------|----------|
| **Grid** | 8-256 envs | Anomaly detection (colored cells, one char per env) |
| **Cards** | Top 5 + Bottom 5 + Flagged | Detailed status of interesting envs |
| **List** | All envs (scrollable) | Current implementation, full table |

**Interaction:**
- `v` key cycles through views
- Grid cells colored by status: green (ok), yellow (warning), red (critical)
- Click/Enter on grid cell → expand to card detail

**"Interesting" Environment Definition:**
```python
interesting = (
    env.status == "stalled" or
    abs(env.reward - median_reward) > 2 * std_reward or
    env.state_changed_last_epoch
)
```

#### 2. A/B Comparison Widget

When `--dual-ab` is active, show comparison metrics:

```
┌─ A/B COMPARISON ── shaped (A) vs simplified (B) ────────────────────┐
│           │  Policy A (shaped)  │  Policy B (simplified)  │ Better  │
│ Accuracy  │      72.3%          │       74.1%             │   B     │
│ Param ROI │      0.12%/1K       │       0.18%/1K          │   B     │
│ Entropy   │      0.45 ↓         │       0.62 →            │   B     │
│ Clip Frac │      0.18           │       0.12              │   B     │
│ Yield     │      33%            │       50%               │   B     │
└─────────────────────────────────────────────────────────────────────┘
```

**Interaction:**
- `a` key toggles A/B comparison panel visibility
- `t` key (existing) switches which policy's detailed view is shown

#### 3. DRL Deep Dive Modal

Press `d` to open a modal with detailed analysis not suitable for the main dashboard:

- Per-PPO-epoch loss breakdown
- Return distribution histogram (ASCII)
- Advantage histogram (ASCII)
- Trust region utilization gauge
- Value function calibration scatter (bucketized)

#### 4. Event Log Enhancements

- `Space` key toggles autoscroll (shows `[Auto ✓]` or `[Paused]`)
- `/` key opens filter (by event type, env, severity)
- Persistent event strip in all views (3 most recent)

### Keyboard Navigation Map

```
Global (work in any view):
  q       Quit
  ?       Help
  r       Refresh
  1-9,0   Focus env N (persistent)
  t       Toggle A/B policy (when --dual-ab)

Environment Overview:
  v       Cycle view (Grid → Cards → List)
  j/k     Navigate list/cards
  h/l     Navigate grid columns
  Enter   Expand focused env to detail modal
  i       Toggle "interesting" flag on focused env

Training Health (Tamiyo):
  Tab     Cycle panels (PPO → Health → Actions → Events)
  d       Open DRL deep dive modal
  a       Toggle A/B comparison panel

Event Log:
  Space   Toggle autoscroll
  /       Filter events
  Esc     Clear filter
```

---

## Bug Fix: Multi-GPU Infrastructure Metrics

**Priority:** HIGH
**Status:** Not Implemented

### Problem

Current infrastructure metrics (`TorchStabilityPanel`, `InfrastructureMetrics`) assume single GPU:

```python
# Current schema (single values)
cuda_memory_allocated_gb: float = 0.0
cuda_memory_reserved_gb: float = 0.0
cuda_memory_fragmentation: float = 0.0
```

With `--devices cuda:0 cuda:1`, each agent collects its own device's memory, but the TUI displays only one device's stats (last one to emit wins).

### Impact

- Users cannot see memory pressure on individual GPUs
- OOM on cuda:1 may be invisible while cuda:0 looks healthy
- Fragmentation per-device is critical for multi-GPU debugging

### Proposed Fix

#### Schema Changes (`karn/sanctum/schema.py`)

```python
@dataclass
class PerDeviceMetrics:
    """Per-CUDA-device infrastructure metrics."""
    device: str  # "cuda:0", "cuda:1", etc.
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    memory_peak_gb: float = 0.0
    memory_fragmentation: float = 0.0
    temperature_celsius: float | None = None  # If available via pynvml

@dataclass
class InfrastructureMetrics:
    # ... existing fields ...

    # NEW: Per-device metrics
    devices: dict[str, PerDeviceMetrics] = field(default_factory=dict)

    # Aggregate properties (for backward compatibility)
    @property
    def cuda_memory_allocated_gb(self) -> float:
        """Total allocated across all devices."""
        return sum(d.memory_allocated_gb for d in self.devices.values())

    @property
    def worst_device_memory_usage(self) -> tuple[str, float]:
        """Device with highest memory pressure."""
        if not self.devices:
            return ("none", 0.0)
        worst = max(self.devices.values(),
                    key=lambda d: d.memory_allocated_gb / max(d.memory_reserved_gb, 0.001))
        return (worst.device, worst.memory_allocated_gb / max(worst.memory_reserved_gb, 0.001))
```

#### TUI Changes (`torch_stability_panel.py`)

```python
def render(self) -> Text:
    # ... existing code ...

    # Multi-device display
    devices = infra.devices
    if len(devices) > 1:
        # Show per-device breakdown
        for device_name, metrics in sorted(devices.items()):
            usage = metrics.memory_allocated_gb / max(metrics.memory_reserved_gb, 0.001)
            style = "red" if usage > 0.90 else "yellow" if usage > 0.75 else "green"
            self._render_label(result, device_name)
            result.append(f"{metrics.memory_allocated_gb:.1f}G {usage:.0%}", style=style)
            result.append("\n")
    else:
        # Single device (current behavior)
        # ... existing code ...
```

#### Anomaly Strip Enhancement

Show worst-case device in the always-visible header:

```
│ 🔥 cuda:1 95% ⚠ │  (instead of just "Mem: 95%")
```

---

## Implementation Priority

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Multi-GPU bug fix | HIGH | Medium | Schema + emitter + TUI changes |
| Kasmina view switcher (Grid/Cards/List) | HIGH | Medium | New rendering logic |
| A/B comparison widget | MEDIUM | Low | Already have data, just display |
| DRL deep dive modal | MEDIUM | Medium | New modal component |
| Event log autoscroll toggle | LOW | Low | Simple state toggle |
| Curriculum Mode | DEFERRED | High | Blocked on Phase 3 TinyStories |

---

## References

- **UX Review:** Lyra UX Designer (ux-critic agent)
- **DRL Review:** Yzmir DRL Expert (drl-expert agent)
- **PyTorch Review:** Yzmir PyTorch Expert (pytorch-code-reviewer agent)
- **Current Implementation:** `src/esper/karn/sanctum/widgets/tamiyo/`
- **CSS Styling:** `src/esper/karn/sanctum/widgets/tamiyo/tamiyo.tcss`
