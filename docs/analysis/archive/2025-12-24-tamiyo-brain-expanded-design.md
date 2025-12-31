# TamiyoBrain Widget: Expanded Design Specification

**Date:** 2025-12-24
**Designer:** UX Specialist (Claude)
**Based On:** DRL Expert Review + Previous UX Design
**Status:** Expanded design with increased space budget

---

## Design Philosophy

The expanded TamiyoBrain transforms from a compact diagnostic widget into a **command center for PPO training observation**. With the additional space budget, we move from "answering 3 questions" to "providing complete situational awareness."

The design follows the **Air Traffic Control mental model**: operators scan for anomalies (green = safe to ignore, red = attend immediately), with progressive disclosure for details.

---

## Space Allocation

**Target Dimensions:** 96 characters wide × 24 lines tall (was ~50×17)

This represents roughly **100% more horizontal space** and **40% more vertical space**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Line allocation:                                                                             │
│   Status Banner:           1 line                                                            │
│   Diagnostic Matrix:       6 lines (2×2 gauges left, metrics column right)                   │
│   Per-Head Heatmap:        2 lines                                                           │
│   Separator:               1 line                                                            │
│   Action Distribution:     1 line                                                            │
│   Separator:               1 line                                                            │
│   Decision Carousel:       10 lines (active decision + 2 compact pinned)                     │
│   Bottom border:           1 line                                                            │
│   TOTAL:                   24 lines                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Information Architecture

```
TAMIYO BRAIN (Expanded)
├── STATUS BANNER (1 line)
│   └── Overall health: LEARNING / CAUTION / FAILING + key metric summary
│
├── DIAGNOSTIC MATRIX (6 lines)
│   ├── LEFT: Primary Gauges (2×2 grid)
│   │   ├── Explained Variance [P0] - THE diagnostic for value learning
│   │   ├── Entropy [P0] - Policy exploration level
│   │   ├── Clip Fraction [P0] - Policy stability
│   │   └── KL Divergence [P0] - Policy change rate
│   │
│   └── RIGHT: Secondary Metrics (compact column)
│       ├── Advantage Stats [P1] - mean/std normalization health
│       ├── Ratio Bounds [P1] - min/max policy ratio (alert only)
│       ├── Policy Loss Trend [P2] - sparkline
│       ├── Value Loss Trend [P2] - sparkline
│       └── Gradient Health [P3] - "OK N/M layers" or warning
│
├── PER-HEAD ENTROPY HEATMAP (2 lines) [P2 - "Cool Factor"]
│   └── Visual breakdown: slot|bp|sty|tem|a_t|a_s|a_c|op
│
├── ACTION DISTRIBUTION BAR (1 line)
│   └── [GGGAAAFFPPWWWWWWWWWWWWW] + legend + anomaly indicator
│
└── DECISION CAROUSEL (8 lines)
    ├── ACTIVE DECISION (detailed)
    │   ├── SAW: slot states + host accuracy
    │   ├── CHOSE: action + confidence + alternatives
    │   └── EXPECTED vs GOT: value prediction accuracy
    │
    └── PINNED/RECENT DECISIONS (compact list)
```

---

## ASCII Mockups

### HEALTHY STATE

```
┌─ TAMIYO ─────────────────────────────────────────────────────────────────────────────────────┐
│ [OK] LEARNING   EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94   GradHP:OK 12/12   batch:47/100   │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│  Explained Var      Entropy           ││ Advantage   +0.12 ± 0.94  [OK]                      │
│  [████████░░] 0.72  [██████░░░░] 0.89  ││ Ratio       0.85 < r < 1.12 [OK]                    │
│   "Learning!"        "Exploring"       ││ Policy Loss ▁▂▃▂▃▄▃▂▃▄ -0.024                       │
│  ─────────────────────────────────────────────────────────────────────────────               │
│  Clip Fraction      KL Divergence     ││ Value Loss  ▅▄▃▃▂▂▂▁▁▁  0.142                       │
│  [████░░░░░░] 0.18  [██░░░░░░░░] 0.008 ││ Grad Norm   ▂▂▃▂▂▃▂▂▂▂  1.24                        │
│   "Stable"           "Stable"          ││ Layers      OK 12/12 healthy                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Heads: slot[████] bp[███░] sty[██░░] tem[███░] a_t[██░░] a_s[██░░] a_c[███░] op[████]        │
│        1.21       0.89      0.62      0.95      0.71      0.68      0.91     1.38            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Actions: [▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░] G=32 A=18 F=12 P=08 W=30  dist:balanced                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ DECISION (3.2s ago)                                                              [click pin] │
│ SAW:  r0c0:TRAIN@45% | r0c1:BLEND@0.8 | r0c2:HOLD | r1c0:DORM    Host: 78.4%                 │
│ CHOSE: GERMINATE r1c0 conv_light (92%)                    Also: WAIT(5%) FOSS r0c1(3%)      │
│ VALUE: expected +0.12  -->  actual +0.08  [OK delta=0.04]                                    │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│ [pin] (12s) FOSSILIZE r0c0  exp:+0.18 got:+0.22 [OK]                                         │
│       (28s) WAIT            exp:+0.02 got:+0.01 [OK]                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### WARNING STATE (Clip Fraction High + Advantage Normalization Issue)

```
┌─ TAMIYO ─────────────────────────────────────────────────────────────────────────────────────┐
│ [!] CAUTION   EV:0.31 Clip:0.28! KL:0.018 Adv:0.45±2.31!  GradHP:OK 12/12   batch:72/100    │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│  Explained Var      Entropy           ││ Advantage   +0.45 ± 2.31  [!] std>>1                │
│  [████░░░░░░] 0.31  [████░░░░░░] 0.62  ││ Ratio       0.72 < r < 1.38 [!] wide                │
│   "Improving"        "Focusing"        ││ Policy Loss ▁▂▃▄▅▆▇█▇▆ -0.089 [!] rising           │
│  ─────────────────────────────────────────────────────────────────────────────               │
│  Clip Fraction      KL Divergence     ││ Value Loss  ▃▃▄▄▅▅▆▆▆▇  0.892 [!] rising            │
│  [████████░░] 0.28! [████░░░░░░] 0.018 ││ Grad Norm   ▂▃▄▅▆▇▆▅▄▃  4.21                        │
│   "Aggressive!"      "Fast"            ││ Layers      OK 12/12 healthy                        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Heads: slot[████] bp[██░░] sty[█░░░] tem[██░░] a_t[█░░░] a_s[█░░░] a_c[██░░] op[███░]        │
│        1.18       0.58!     0.31!     0.54      0.28!     0.25!     0.52     0.89            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Actions: [░░░░░░░░░░░░░░░░░░░░▓▓▓▓] G=02 A=01 F=00 P=01 W=96! dist:WAIT-DOMINATED!          │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ DECISION (1.8s ago)                                                              [click pin] │
│ SAW:  r0c0:TRAIN@12% | r0c1:DORM | r0c2:DORM | r1c0:DORM         Host: 45.2%                 │
│ CHOSE: WAIT (96%)                                         Also: GERM r0c1(3%) GERM r0c2(1%) │
│ VALUE: expected +0.02  -->  actual -0.12  [!] value overestimated by 0.14                    │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│       (8s)  WAIT            exp:+0.01 got:-0.08 [!]                                          │
│       (15s) WAIT            exp:+0.03 got:-0.05 [!]                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### CRITICAL STATE (Value Function Harmful + Policy Collapsed)

```
┌─ TAMIYO ─────────────────────────────────────────────────────────────────────────────────────┐
│ [X] FAILING   EV:-0.42! Clip:0.35! KL:0.042! Adv:-0.8±0.12  GradHP:!! 8/12   batch:89/100   │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│  Explained Var      Entropy           ││ Advantage   -0.80 ± 0.12  [X] mean<0                │
│  [░░░░░░░░░░] -0.42 [█░░░░░░░░░] 0.08  ││ Ratio       0.45! < r < 2.1! [X] extreme            │
│   "HARMFUL!"         "COLLAPSED!"      ││ Policy Loss ▆▇█████████ -0.312 [X] diverging        │
│  ─────────────────────────────────────────────────────────────────────────────               │
│  Clip Fraction      KL Divergence     ││ Value Loss  ▅▆▇███████  2.891 [X] exploding         │
│  [██████████] 0.35! [████████░░] 0.042!││ Grad Norm   ▃▅▇████▇▅▃  8.92! [X] unstable          │
│   "TOO AGGRESSIVE!"  "DIVERGING!"      ││ Layers      !! 4 dead, 0 exploding                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Heads: slot[█░░░] bp[░░░░] sty[░░░░] tem[░░░░] a_t[░░░░] a_s[░░░░] a_c[░░░░] op[█░░░]        │
│        0.12!      0.02!     0.01!     0.03!     0.01!     0.01!     0.02!    0.15!           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Actions: [▓░░░░░░░░░░░░░░░░░░░░░░░] G=00 A=00 F=00 P=100! W=00  dist:PRUNE-ONLY!             │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ DECISION (0.4s ago)                                                              [click pin] │
│ SAW:  r0c0:DORM | r0c1:DORM | r0c2:DORM | r1c0:DORM              Host: 22.1%                 │
│ CHOSE: PRUNE r0c0 (99%)                                   Also: PRUNE r0c1(1%)               │
│ VALUE: expected +0.45  -->  actual -0.89  [X] value VERY wrong, predicting wrong sign!       │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│       (2s)  PRUNE r0c1      exp:+0.38 got:-0.72 [X]                                          │
│       (5s)  PRUNE r0c2      exp:+0.41 got:-0.81 [X]                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Status Banner

**Purpose:** Immediate health assessment at a glance.

**Layout:** Single line with:
- Status icon: `[OK]` / `[!]` / `[X]` with redundant color + text
- Status word: `LEARNING` / `CAUTION` / `FAILING`
- Key metrics summary (most critical numbers)
- Batch progress

**Status Determination Logic:**
```python
def determine_status(tamiyo: TamiyoState) -> str:
    # P0 Critical checks (immediate FAILING)
    if tamiyo.explained_variance < -0.5:
        return "FAILING"  # Value function actively harmful
    if tamiyo.entropy < 0.1:
        return "FAILING"  # Policy collapsed
    if tamiyo.clip_fraction > 0.35:
        return "FAILING"  # Policy updates too aggressive

    # P0/P1 Warning checks (CAUTION)
    if tamiyo.explained_variance < 0.0:
        return "CAUTION"  # Value function not helping
    if tamiyo.entropy < 0.3:
        return "CAUTION"  # Policy focusing too fast
    if tamiyo.clip_fraction > 0.25:
        return "CAUTION"  # Clipping often
    if tamiyo.kl_divergence > 0.02:
        return "CAUTION"  # Policy changing fast
    if tamiyo.advantage_std > 2.0 or tamiyo.advantage_std < 0.5:
        return "CAUTION"  # Advantage normalization off

    return "LEARNING"  # All systems nominal
```

### 2. Diagnostic Matrix (Primary Gauges + Secondary Metrics)

**Left Column: 4 Primary Gauges (2×2 grid)**

Each gauge occupies 22 chars × 3 lines:
```
Label (12 chars)
[██████████] value
 "descriptor"
```

| Gauge | Range | Healthy | Warning | Critical |
|-------|-------|---------|---------|----------|
| Explained Var | -1 to 1 | > 0.5 | 0 to 0.5 | < 0 |
| Entropy | 0 to 2 | > 0.5 | 0.3 to 0.5 | < 0.1 |
| Clip Fraction | 0 to 0.5 | < 0.2 | 0.2 to 0.3 | > 0.3 |
| KL Divergence | 0 to 0.1 | < 0.015 | 0.015 to 0.03 | > 0.03 |

**Right Column: Secondary Metrics (compact)**

| Metric | Display | Alert Condition |
|--------|---------|-----------------|
| Advantage | `+0.12 ± 0.94` | std > 2 or < 0.5, mean < -0.5 |
| Ratio | `0.85 < r < 1.12` | max > 1.5 or min < 0.5 |
| Policy Loss | sparkline + value | rising trend (last 5 > first 5) |
| Value Loss | sparkline + value | rising trend or > 1.0 late |
| Grad Norm | sparkline + value | > 5.0 |
| Layers | `OK 12/12` or `!! N dead` | any dead/exploding |

### 3. Per-Head Entropy Heatmap

**Purpose:** Show which action heads are collapsing (the "cool factor" visualization).

**Layout:**
```
Heads: slot[████] bp[███░] sty[██░░] tem[███░] a_t[██░░] a_s[██░░] a_c[███░] op[████]
       1.21       0.89      0.62      0.95      0.71      0.68      0.91     1.38
```

**Head Abbreviations:**
| Full Name | Abbrev | Max Entropy (uniform) |
|-----------|--------|----------------------|
| slot | slot | ln(num_slots) |
| blueprint | bp | ln(12) ≈ 2.48 |
| style | sty | ln(3) ≈ 1.10 |
| tempo | tem | ln(3) ≈ 1.10 |
| alpha_target | a_t | ln(5) ≈ 1.61 |
| alpha_speed | a_s | ln(3) ≈ 1.10 |
| alpha_curve | a_c | ln(3) ≈ 1.10 |
| op | op | ln(5) ≈ 1.61 |

**Bar Fill Logic:**
- 4 chars per head
- Fill = entropy / max_entropy_for_head
- Color: green if > 50%, yellow if 25-50%, red if < 25%

### 4. Action Distribution Bar

**Purpose:** Show what Tamiyo is actually doing (action mix).

**Layout:**
```
Actions: [▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░] G=32 A=18 F=12 P=08 W=30  dist:balanced
```

**Anomaly Detection:**
- `WAIT-DOMINATED!` if WAIT > 70%
- `PRUNE-ONLY!` if PRUNE > 50%
- `NO-GERMINATION!` if GERMINATE = 0 for 20+ actions
- `balanced` if distribution is healthy

### 5. Decision Carousel (Expanded)

**Purpose:** Show the decision-making process in detail.

**Active Decision (5 lines):**
```
DECISION (3.2s ago)                                                              [click pin]
SAW:  r0c0:TRAIN@45% | r0c1:BLEND@0.8 | r0c2:HOLD | r1c0:DORM    Host: 78.4%
CHOSE: GERMINATE r1c0 conv_light (92%)                    Also: WAIT(5%) FOSS r0c1(3%)
VALUE: expected +0.12  -->  actual +0.08  [OK delta=0.04]
```

**Compact Pinned/Recent (2 lines each):**
```
[pin] (12s) FOSSILIZE r0c0  exp:+0.18 got:+0.22 [OK]
      (28s) WAIT            exp:+0.02 got:+0.01 [OK]
```

**Value Prediction Accuracy Indicators:**
- `[OK]` - prediction within 0.1 of actual
- `[!]` - prediction off by 0.1-0.3
- `[X]` - prediction off by > 0.3 or wrong sign

---

## Color Palette

| Element | OK/Green | Warning/Yellow | Critical/Red |
|---------|----------|----------------|--------------|
| Status Icon | `[OK]` bright_green | `[!]` yellow | `[X]` red bold |
| Gauge Fill | cyan/bright_cyan | yellow | red |
| Metric Value | cyan | yellow bold | red bold |
| Descriptor | dim italic | yellow italic | red italic |
| Alert Tag | - | `[!]` yellow | `[X]` red bold |

**Accessibility Notes:**
- All status indicators use text + icon + color (triple redundancy)
- Gauge bars use fill level (visual) + numeric value (text)
- Alert conditions always have bracketed indicators like `[!]` or `[X]`

---

## Keyboard Navigation

| Key | Action |
|-----|--------|
| `j/k` | Navigate decisions (if focused) |
| `p` | Toggle pin on selected decision |
| `Enter` | Expand decision to modal |
| `?` | Show help overlay |
| `r` | Reset pinned decisions |

---

## CSS Styling (Textual)

```css
TamiyoBrain {
    height: 24;
    min-width: 96;
    border: solid $accent;
    border-title-color: $primary;
    padding: 0 1;
}

TamiyoBrain .status-ok {
    color: $success;
}

TamiyoBrain .status-warning {
    color: $warning;
}

TamiyoBrain .status-critical {
    color: $error;
    text-style: bold;
}

TamiyoBrain .gauge-bar {
    color: $primary-lighten-2;
}

TamiyoBrain .gauge-empty {
    color: $surface-darken-2;
}

TamiyoBrain .metric-label {
    color: $text-muted;
}

TamiyoBrain .metric-value {
    color: $primary;
}

TamiyoBrain .sparkline {
    color: $secondary;
}

TamiyoBrain .head-healthy {
    color: $success;
}

TamiyoBrain .head-warning {
    color: $warning;
}

TamiyoBrain .head-collapsed {
    color: $error;
}

TamiyoBrain .decision-active {
    border: solid $accent;
}

TamiyoBrain .decision-pinned {
    border: solid $warning;
}
```

---

## Implementation Notes

### New Widget Structure

The expanded TamiyoBrain should be split into composable sub-widgets:

```
TamiyoBrain (Static, 96×24)
├── TamiyoStatusBanner (Static, 96×1)
├── TamiyoDiagnosticMatrix (Static, 96×7)
│   ├── TamiyoGaugeGrid (Static, 48×6) - left side
│   └── TamiyoMetricColumn (Static, 46×6) - right side
├── TamiyoHeadHeatmap (Static, 96×2)
├── TamiyoActionBar (Static, 96×1)
└── TamiyoDecisionCarousel (Static, 96×10)
```

### Data Requirements from TamiyoState

All required fields already exist in `TamiyoState`:

**P0 (Already Displayed):**
- `entropy` ✓
- `kl_divergence` ✓
- `value_loss` ✓ (used, but should add sparkline history)

**P0 (Must Add to Display):**
- `explained_variance` - available but not shown!
- `clip_fraction` - available but not shown!

**P1 (Must Add to Display):**
- `advantage_mean`, `advantage_std` - available but not shown!
- `ratio_max`, `ratio_min` - available but not shown!

**P2 (Must Add to Display):**
- `policy_loss` - available but not shown (need history for sparkline)
- `head_slot_entropy`, `head_blueprint_entropy` - available but not shown!

**P3 (Must Add to Display):**
- `dead_layers`, `exploding_layers` - available but not shown!
- `layer_gradient_health` - available but not shown!

### History Tracking for Sparklines

Add to `TamiyoState`:
```python
# History for trend sparklines (last 10 values)
policy_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
value_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
grad_norm_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
```

Update aggregator to append to these deques on each PPO update.

### Missing Telemetry Data

The following per-head metrics need to be emitted from `PPOAgent`:
- Per-head entropies for all 8 action heads (currently only slot/blueprint)
- Per-head gradient norms (computed but not emitted)

Files to modify:
- `src/esper/simic/telemetry/emitters.py` - Add per-head metrics emission
- `src/esper/karn/sanctum/aggregator.py` - Capture new per-head fields

---

## Summary

This expanded design:

1. **Uses the space budget fully** - 96×24 vs original ~50×17
2. **Shows ALL P0/P1/P2 metrics** from the DRL analysis
3. **Includes "cool factor" visualizations:**
   - Per-head entropy heatmap (8 heads with bar visualization)
   - Sparkline trends for losses/gradients
   - Gradient layer health indicator
   - Value prediction accuracy feedback
4. **Maintains diagnostic clarity** with the decision tree status banner
5. **Preserves accessibility** with text+icon+color redundancy throughout

The widget answers:
- **"Is Tamiyo learning?"** - Status banner + diagnostic matrix (< 2 seconds to assess)
- **"What is she doing?"** - Action distribution + head heatmap
- **"What did she decide?"** - Decision carousel with value accuracy feedback
- **"What's going wrong?"** - Alert indicators with specific diagnostics
