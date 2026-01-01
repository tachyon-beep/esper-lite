# TamiyoBrain Widget Redesign: UX Specification

**Date:** 2025-12-24
**Designer:** UX Specialist (Claude)
**Based On:** DRL Expert Review (2025-12-24-tamiyo-diagnostics-drl-review.md)
**Status:** Design specification ready for implementation

---

## 1. Executive Summary

The current TamiyoBrain widget displays 3 gauges (Entropy, Value Loss, KL Divergence) + action distribution + decision carousel. Based on the DRL expert review, the most critical gap is **Explained Variance** - the diagnostic that tells you whether the value function is actually learning. Several other P0/P1 metrics are captured but never displayed.

### Design Goals

1. **Answer "Is Tamiyo Learning?"** in < 2 seconds of glance time
2. **Surface anomalies automatically** - problems should be visually loud
3. **Maintain information density** within fixed height constraint
4. **Preserve decision carousel** - still valuable for understanding decisions

---

## 2. Information Architecture

### Hierarchy (Top to Bottom)

```
TIER 1: LEARNING STATUS     (1 line)  - Binary: Learning? Yes/No + why
TIER 2: VITALS GAUGES       (4 lines) - 4 key metrics with sparklines
TIER 3: ACTION DISTRIBUTION (1 line)  - What is Tamiyo doing?
TIER 4: LAST DECISION       (5 lines) - Most recent decision detail
```

### What Changes From Current

| Current | Proposed | Rationale |
|---------|----------|-----------|
| 3 gauges: Entropy, Value Loss, KL | 4 gauges: **Explained Var**, Entropy, Clip Frac, KL | EV is P0, Value Loss demoted |
| Value Loss gauge prominent | Value Loss → secondary (shown inline) | EV is more interpretable |
| No status summary | Add 1-line "Learning Status" banner | Instant diagnostic |
| 3 decision panels | 2 decision panels (1 current + 1 pinned) | More space for vitals |
| Gauges side-by-side | Gauges 2x2 with compact sparklines | Better use of height |

---

## 3. Detailed ASCII Mockups

### Layout Dimensions

- Widget width: ~50-60 characters (40% of typical 140-char terminal)
- Widget height: Fixed at ~17 lines (based on current CSS)

### 3.1 HEALTHY State (Everything Working)

```
┌─ TAMIYO ─────────────────────────────────────────┐
│ ● LEARNING  EV:0.72 Clip:0.18 KL:0.008           │
├──────────────────────────────────────────────────┤
│ Expl.Var    Entropy     Clip Frac   KL Div      │
│ ███████░░░  ████░░░░░░  ████░░░░░░  ██░░░░░░░░  │
│ 0.72 ▁▂▃▅▆  0.89 ▆▅▄▃▂  0.18        0.008       │
│ "Learning"  "Focusing"  "Stable"    "Stable"    │
├──────────────────────────────────────────────────┤
│ Actions: [▓▓▓░░░░░░░░░░░░] G=12 A=05 F=02 P=01 W=80 │
├──────────────────────────────────────────────────┤
│ DECISION (3s ago)                                │
│ SAW: r0c0:Train 45% | r0c1:Blend 0.8 | Host:78%  │
│ CHOSE: WAIT (92%)         Also: GERM(5%) FOSS(3%)│
│ EXPECTED: +0.12  →  GOT: +0.08 ✓                 │
└──────────────────────────────────────────────────┘
```

**Key Visual Elements:**

- **Status Line**: Green filled circle (●) + "LEARNING" + 3 key metrics inline
- **4 Gauges in 2x2 Grid**: Each with bar + value + sparkline + label
- **Action Bar**: Compact stacked bar with fixed-width legend
- **Single Decision Panel**: Collapsed from 3 to save space

### 3.2 WARNING State (Something Needs Attention)

```
┌─ TAMIYO ─────────────────────────────────────────┐
│ ◐ CAUTION  EV:0.12 Clip:0.28! KL:0.022!          │
├──────────────────────────────────────────────────┤
│ Expl.Var    Entropy     Clip Frac   KL Div      │
│ █░░░░░░░░░  ████░░░░░░  ███████░░░  █████░░░░░  │
│ 0.12 ▁▁▁▂▂  0.89 ▆▅▄▃▂  0.28 !      0.022 !     │
│ "Slow"      "Focusing"  "High"      "Drifting"  │
├──────────────────────────────────────────────────┤
│ Actions: [▓▓▓▓▓▓▓▓▓▓▓▓▓░░] G=02 A=01 F=00 P=00 W=97 │
│          ^ WAIT > 90% - policy may be stuck     │
├──────────────────────────────────────────────────┤
│ DECISION (8s ago)                                │
│ SAW: r0c0:Train 12% | r0c1:Empty | Host:71%      │
│ CHOSE: WAIT (97%)         Also: GERM(2%) ALPH(1%)│
│ EXPECTED: +0.05  →  GOT: -0.02 ✗                 │
└──────────────────────────────────────────────────┘
```

**Warning Indicators:**

- **Status Line**: Half-filled circle (◐) + "CAUTION" in yellow + exclamation marks on problem metrics
- **Gauge Labels**: "High", "Drifting", "Slow" replace normal labels
- **Action Warning**: Inline annotation when WAIT > 90%
- **Decision Mismatch**: Red X when expected vs got diverge significantly

### 3.3 CRITICAL State (Something Is Wrong)

```
┌─ TAMIYO ─────────────────────────────────────────┐
│ ○ FAILING  EV:-0.32! Ent:0.08! Clip:0.41!        │
├──────────────────────────────────────────────────┤
│ Expl.Var    Entropy     Clip Frac   KL Div      │
│ ██████████  ░░░░░░░░░░  ██████████  ████░░░░░░  │
│ -0.32 !     0.08 !!!    0.41 !!!    0.018       │
│ "HARMFUL"   "COLLAPSED" "UNSTABLE"  "Stable"    │
│                                                  │
│ ! Value function making predictions WORSE       │
│ ! Policy deterministic - no exploration         │
├──────────────────────────────────────────────────┤
│ Actions: [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] G=00 A=00 F=00 P=00 W=100│
│          !! ONLY WAITING - training stalled     │
├──────────────────────────────────────────────────┤
│ DECISION (45s ago) ⚠ STALE                       │
│ SAW: r0c0:Empty | r0c1:Empty | Host:65%          │
│ CHOSE: WAIT (100%)                               │
│ EXPECTED: +0.00  →  GOT: +0.00 ─                 │
└──────────────────────────────────────────────────┘
```

**Critical Indicators:**

- **Status Line**: Empty circle (○) + "FAILING" in red + all problem metrics with !
- **Diagnostic Messages**: 2-line alert box explaining what's wrong
- **Stale Decision**: Warning when decision is > 30s old
- **Multiple Exclamation Marks**: Severity scaling (! warning, !! critical, !!! severe)

---

## 4. Gauge Design Specification

### 4.1 Gauge Layout (Each Gauge)

```
Width: 12 chars per gauge (4 gauges in ~50 char widget = reasonable)

Layout:
┌────────────┐
│ Label      │  <- 12 chars, left-aligned
│ ████░░░░░░ │  <- 10-char bar with brackets
│ 0.72 ▁▂▃▅▆ │  <- value + 5-char sparkline
│ "Learning" │  <- Interpretive label in quotes
└────────────┘
```

### 4.2 Gauge Specifications

| Gauge | Range | Green | Yellow | Red | Sparkline Shows |
|-------|-------|-------|--------|-----|-----------------|
| **Explained Var** | -1 to 1 | > 0.5 | 0 to 0.5 | < 0 | Trend (rising = good) |
| **Entropy** | 0 to ln(N) | Contextual decay | Flat | Near 0 | Expected decay pattern |
| **Clip Fraction** | 0 to 0.5 | 0.1-0.2 | 0.25-0.3 | > 0.3 | Should be stable |
| **KL Divergence** | 0 to 0.1 | 0.001-0.015 | > 0.015 | > 0.03 | Should be stable |

### 4.3 Interpretive Labels (Human-Readable Status)

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Explained Var | "Learning", "Accurate" | "Slow", "Uncertain" | "HARMFUL", "BLIND" |
| Entropy | "Exploring", "Focusing" | "Decisive", "Narrow" | "COLLAPSED", "STUCK" |
| Clip Fraction | "Stable", "Controlled" | "High", "Aggressive" | "UNSTABLE", "CLIPPING" |
| KL Divergence | "Stable", "Smooth" | "Drifting", "Fast" | "DIVERGING", "UNSTABLE" |

---

## 5. Status Banner Design

The top line provides instant "at a glance" health:

### 5.1 Status States

| Icon | Text | Color | When |
|------|------|-------|------|
| `●` | `LEARNING` | Green | All metrics healthy |
| `◐` | `CAUTION` | Yellow | Any metric in warning range |
| `○` | `FAILING` | Red | Any metric in critical range |
| `?` | `WAITING` | Dim | No PPO data received yet |

### 5.2 Status Line Format

```
<icon> <status>  EV:<val> Clip:<val> KL:<val>
```

- Width: ~45 chars
- Exclamation marks (!) appended to problem values
- Most critical metric values shown inline for quick scanning

---

## 6. Action Distribution Bar

### 6.1 Current vs Proposed

Current:
```
[▓▓▓░░░░░░░░░░░░░░░░░░░░░░] G=09 A=02 F=00 P=06 W=60
```

Proposed (with anomaly detection):
```
Actions: [▓▓▓▓▓▓▓▓░░░░░░░] G=12 A=05 F=02 P=01 W=80
         ^ warning annotation when anomalous
```

### 6.2 Anomaly Detection Rules

| Condition | Annotation | Severity |
|-----------|------------|----------|
| WAIT > 90% | "policy may be stuck" | Warning |
| WAIT = 100% | "training stalled" | Critical |
| PRUNE > 30% | "high prune rate" | Warning |
| GERMINATE = 0 after batch 20 | "not creating seeds" | Warning |

---

## 7. Decision Panel (Simplified)

### 7.1 From 3 to 1 (or 2 with Pin)

Current: 3 decision panels, each ~5 lines = 15 lines
Proposed: 1 active decision + 1 pinned slot = 5-10 lines

### 7.2 Decision Panel Format

```
DECISION (Xs ago) [PIN if pinned]
SAW: slot_states | Host:XX%
CHOSE: ACTION (confidence%)    Also: ALT1(%) ALT2(%)
EXPECTED: +X.XX  →  GOT: +X.XX [✓|✗|─]
```

### 7.3 Staleness Indicator

- If decision > 30s old: Add `STALE` warning
- If decision > 60s old: Dim the entire panel

---

## 8. Keyboard Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| `e` | Toggle expanded view | Show all 20+ metrics in popup |
| `p` | Pin current decision | Keep decision visible |
| `h` | Show head entropies | Per-head breakdown popup |
| `?` | Toggle help | Show keyboard shortcuts |

---

## 9. Color Palette (Textual)

| Element | Good | Warning | Critical | Dim |
|---------|------|---------|----------|-----|
| Status text | `green` | `yellow` | `red bold` | `dim` |
| Gauge bar fill | `cyan` | `yellow` | `red` | `dim` |
| Gauge bar empty | `dim` | `dim` | `dim` | `dim` |
| Sparkline | `cyan` | `yellow` | `red` | `dim` |
| Labels | `italic dim` | `italic yellow` | `italic red` | `dim` |
| Borders | `magenta` (normal) | `yellow` (warning) | `red` (critical) | - |

---

## 10. Implementation Specification

### 10.1 Widget Structure (Textual Containers)

```python
class TamiyoBrain(Static):
    def render(self):
        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Row 1: Status Banner (1 line)
        main_table.add_row(self._render_status_banner())

        # Row 2: Vitals Gauges (4 lines)
        main_table.add_row(self._render_vitals_grid())

        # Row 3: Action Distribution (1-2 lines)
        main_table.add_row(self._render_action_bar())

        # Row 4: Decision Panel (5 lines)
        main_table.add_row(self._render_decision())

        return main_table
```

### 10.2 Key Methods to Implement

```python
def _render_status_banner(self) -> Text:
    """Render 1-line status: ● LEARNING  EV:0.72 Clip:0.18 KL:0.008"""

def _render_vitals_grid(self) -> Table:
    """Render 2x2 gauge grid with sparklines."""

def _render_gauge(self, label: str, value: float,
                  min_val: float, max_val: float,
                  history: list[float],
                  thresholds: tuple[float, float]) -> Text:
    """Render single gauge: bar + value + sparkline + label."""

def _get_learning_status(self) -> tuple[str, str, str]:
    """Returns (icon, text, style) for status banner."""

def _get_action_anomaly(self) -> str | None:
    """Check for action distribution anomalies."""
```

### 10.3 Data Dependencies

All required data is already in `TamiyoState` (schema.py:364-433):

- `explained_variance` - P0, not currently displayed
- `clip_fraction` - P0, not currently displayed
- `entropy` - Currently displayed
- `kl_divergence` - Currently displayed
- `value_loss` - Currently displayed (demote to secondary)
- `advantage_mean/std` - P1, for inline display
- `ratio_min/max` - P1, for alert detection

No telemetry changes required.

### 10.4 Character Width Allocation

| Section | Width | Notes |
|---------|-------|-------|
| Status banner | 48 chars | Full width |
| Gauge label | 11 chars | "Expl.Var", "Entropy", etc. |
| Gauge bar | 10 chars | `[████░░░░░░]` |
| Gauge value | 6 chars | "0.72", "-0.32" |
| Gauge sparkline | 5 chars | `▁▂▃▅▆` |
| Gauge row total | 48 chars | 4 gauges @ 12 each |

---

## 11. Accessibility Checklist

- [x] All status conveyed through icon + text + color (● LEARNING, not just green)
- [x] Focus indicator via border change (double border when focused)
- [x] No keyboard traps (no modal states without Esc exit)
- [x] High contrast labels for critical states (bold red)
- [x] Sparklines use height variation, not color alone
- [x] Numeric values always displayed alongside visual bars

---

## 12. Empty State

When no PPO data has been received yet:

```
┌─ TAMIYO ─────────────────────────────────────────┐
│ ? WAITING  Collecting PPO data...                │
├──────────────────────────────────────────────────┤
│                                                  │
│     ⏳ Waiting for first PPO update              │
│                                                  │
│     Progress: 150/500 epochs                     │
│     PPO updates typically start after epoch 200  │
│                                                  │
├──────────────────────────────────────────────────┤
│ Actions: [░░░░░░░░░░░░░░░] No actions yet        │
├──────────────────────────────────────────────────┤
│ No decisions captured yet                        │
│ Decisions appear after first reward computation  │
└──────────────────────────────────────────────────┘
```

---

## 13. Summary of Changes

### Additions
1. **Status banner** (1 line) - instant health check
2. **Explained Variance gauge** (P0) - the key diagnostic
3. **Clip Fraction gauge** (P0) - policy update stability
4. **Sparklines on gauges** - trend visualization
5. **Action anomaly detection** - inline warnings
6. **Decision staleness indicator** - data freshness

### Removals/Demotions
1. **Value Loss gauge** - demoted to inline secondary display
2. **3 decision panels** - reduced to 1 active + 1 pinned
3. **"Waiting for PPO vitals" in gauge area** - moved to status banner

### Modifications
1. **Gauge layout** - 3 side-by-side to 2x2 grid
2. **Gauge labels** - static to contextual ("Learning", "COLLAPSED")
3. **Border color** - static magenta to dynamic (green/yellow/red)

---

This specification provides sufficient detail for implementation. The key insight from the DRL review is that **Explained Variance is the single most important metric** for determining if the value function is learning - it should be the first gauge the eye lands on, not buried in unused schema fields.
