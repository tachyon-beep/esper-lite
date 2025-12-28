# TAMIYO Dashboard UX Redesign Briefing

## Project Context

TAMIYO is a real-time training monitor for Esper, a morphogenetic neural network platform. The dashboard displays PPO (Proximal Policy Optimisation) training metrics, attention head states, slot lifecycle management, and decision-making processes for a multi-agent reinforcement learning system.

The current implementation is functional but has accumulated organic complexity. This briefing outlines specific issues and proposed solutions for each panel, along with overarching design principles.

---

## Design Principles

These principles should guide all layout decisions:

1. **Group by conceptual function** — Related metrics belong together, regardless of implementation history
2. **Single status indicator per scope** — If a state applies to a whole section, show it once at section level
3. **Value-first, bar-second** — Users read the number; the bar provides visual confirmation of magnitude
4. **Right-align terminal states** — Status indicators (MISS, ✓, !, warnings) should align to the right edge for rapid scanning
5. **Reduce redundancy** — Repeated identical values (e.g., "WARMING UP" ×4) should consolidate
6. **Consistent column widths** — Within logical groups, maintain alignment for scannability
7. **Visual separators** — Use box-drawing or whitespace to delineate major sections
8. **Trend indicators are cheap** — Arrows (↗↘→) add information density at minimal visual cost
9. **Colour conveys state, not decoration** — Reserve colour for semantic meaning (healthy/warning/error/state)

---

## Panel-by-Panel Analysis

### 1. Header Bar

**Current:**

```
TAMIYO
[~] WARMING UP (5/50)  EV:0.00  Clip:0.00  KL:0.000  Adv:+0.00±1.00  GradHP:OK 12/12  batch:5/100
```

**Issues:**

- `[~]` is a static placeholder; could be animated during warmup
- "OK" is redundant when 12/12 already indicates full health
- Metrics run together without clear grouping

**Proposed:**

```
TAMIYO ─── WARMING UP [5/50] ──────────────────────────────────────────────────
 EV:0.00  Clip:0.00  KL:0.000  Adv:+0.00±1.00  GradHP:12/12 ✓  batch:5/100
```

**Recommendations:**

- Use spinner characters (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) during warmup phase for visual activity feedback
- Replace "OK" with checkmark; use colour (green) to indicate health
- Consider `GradHP:▓▓▓▓▓▓▓▓▓▓▓▓` visual bar when partial failures occur
- Title and phase integrated into single header line

---

### 2. Sparklines (Ep.Return / Entropy)

**Current:**

```
Ep.Return ──────────────────────────────  -2.0 -    LR:1e-04  EntCoef:0.08
Entropy   ──────────────────────────────  7.91 -
```

**Issues:**

- Right-side values feel disconnected from their sparklines
- No trend indication
- Hyperparameters (LR, EntCoef) mixed with time-series metrics

**Proposed:**

```
Ep.Return  ▁▂▁▃▂▄▃▅▄▆▅▇▆█  -2.0 ↘    LR:1e-04
Entropy    █▇▆▇▅▆▄▅▃▄▂▃▁▂   7.91 →    EntCoef:0.08
```

**Recommendations:**

- Tighter coupling: sparkline → value → trend arrow → related hyperparam
- Trend arrows computed from last N points: ↗ (improving), ↘ (declining), → (stable)
- Consider colour-coding the trend arrow (green ↗ for return, context-dependent for entropy)

---

### 3. Training Metrics Grid (Expl.Var / Clip Frac / Entropy / KL Div)

**Current:**

```
Expl.Var                    Entropy
[████      ] 0.004 -        [██████████] 7.91 -
"WARMING UP"                "WARMING UP"

Clip Frac                   KL Div
[          ] 0.000 -        [          ] 0.000 -
"WARMING UP"                "WARMING UP"
```

**Issues:**

- "WARMING UP" repeated 4 times — this is panel-level state, not per-metric
- Bars lack scale context (what constitutes 100% for Expl.Var?)
- Status text consumes a full line per metric
- Vertical space inefficient

**Proposed Option A — Boxed Panel:**

```
┌─ Training Metrics ──────────────────────────────── WARMING UP ─┐
│  Expl.Var   ▓░░░░░░░░░  0.004    Entropy    ▓▓▓▓▓▓▓▓▓▓  7.91  │
│  Clip Frac  ░░░░░░░░░░  0.000    KL Div     ░░░░░░░░░░  0.000 │
└────────────────────────────────────────────────────────────────┘
```

**Proposed Option B — Inline Status Glyphs:**

```
  Expl.Var   ▓░░░░  0.004  ~    Entropy    ▓▓▓▓▓  7.91  ~
  Clip Frac  ░░░░░  0.000  ~    KL Div     ░░░░░  0.000 ~
```

Where: `~` = warming, `✓` = stable, `!` = concerning, `✗` = error

**Recommendations:**

- Consolidate status to panel header or use single-character per-metric indicators
- 2×2 grid layout saves vertical space
- Document bar scales (either in tooltip, help, or scale markers)
- Consider showing target/healthy ranges as background zones on bars

---

### 4. PPO Core Statistics

**Current:**

```
Advantage    +0.00 ± 1.00
Ratio        0.98 < r < 1.02
Policy Loss        █      -0.390 -
Value Loss         ████   35.800
Grad Norm                 1.00 -
Layers       OK 12/12 healthy
```

**Issues:**

- Mixed alignment — some values left, some right, bars inconsistent
- Bars before values (harder to read)
- "OK" and "healthy" are redundant
- Ratio shown as text range rather than visual position

**Proposed:**

```
┌─ PPO ──────────────────────────────────────────────────────────┐
│  Advantage   +0.00 ± 1.00          Policy Loss  -0.390  ▓░░░░  │
│  Ratio       [0.98 ──●── 1.02]     Value Loss   35.800  ▓▓▓▓░  │
│  Grad Norm   1.00                  Layers       12/12 ✓        │
└────────────────────────────────────────────────────────────────┘
```

**Recommendations:**

- Ratio as visual range indicator: `[min ──●── max]` where ● shows current r position within clipping bounds
- Group losses together (conceptually related)
- Bars after values consistently
- Replace "OK 12/12 healthy" with "12/12 ✓" — the fraction implies health
- Consider colour thresholds: ratio approaching clip bounds → yellow/red

---

### 5. Attention Heads Panel

**Current:**

```
Heads: slot[█] bpnt~[░] styl~[░] temp~[░] atgt~[▓] aspd~[░] acrv~[░] op[█]
            1.00   1.00~   0.22~   1.00~   0.61~   1.00~   1.00~  0.91

Grads: slot[█] bpnt~[░] styl~[░] temp~[░] atgt~[░] aspd~[░] acrv~[░] op[░]
            0.20   0.38    0.29    0.13    0.22    0.15    0.29   0.16
```

**Issues:**

- Difficult to visually match head name to its value (horizontal distance)
- `~` suffix meaning unclear without documentation
- Two parallel rows hard to cross-reference
- Inline mini-bars `[█]` don't scale well

**Proposed Option A — Vertical Pairing (Dense):**

```
┌─ Attention Heads ──────────────────────────────────────────────┐
│        slot   bpnt   styl   temp   atgt   aspd   acrv   op    │
│ Head   1.00   1.00~  0.22~  1.00~  0.61~  1.00~  1.00~  0.91  │
│        ████   ████   ▓░░░   ████   ▓▓░░   ████   ████   ▓▓▓░  │
│ Grad   0.20   0.38   0.29   0.13   0.22   0.15   0.29   0.16  │
│        ▓░░░   ▓▓░░   ▓░░░   ░░░░   ▓░░░   ░░░░   ▓░░░   ░░░░  │
└────────────────────────────────────────────────────────────────┘
```

**Proposed Option B — Side-by-Side Lists:**

```
Attention Heads                          Gradients
 slot  ████████████  1.00                 slot  ▓▓░░░░░░░░  0.20
 bpnt  ████████████  1.00~                bpnt  ▓▓▓▓░░░░░░  0.38
 styl  ▓▓░░░░░░░░░░  0.22~                styl  ▓▓▓░░░░░░░  0.29
 temp  ████████████  1.00~                temp  ▓░░░░░░░░░  0.13
 atgt  ▓▓▓▓▓▓░░░░░░  0.61~                atgt  ▓▓░░░░░░░░  0.22
 aspd  ████████████  1.00~                aspd  ▓░░░░░░░░░  0.15
 acrv  ████████████  1.00~                acrv  ▓▓▓░░░░░░░  0.29
 op    ▓▓▓▓▓▓▓▓▓░░░  0.91                 op    ▓░░░░░░░░░  0.16
```

**Recommendations:**

- Document the `~` suffix (presumably indicates learnable/unfrozen heads)
- Vertical layout improves name→value association
- Consider highlighting anomalous gradients (very low or very high)
- If heads have expected ranges, show deviation from expected
- Option A better for comparing heads-to-grads; Option B better for scanning within category

---

### 6. Episode Progress Bar

**Current:**

```
[██████████████████████████████                    ] G=18 A=04 F=00 P=17 W=40
Recent: W W → W W P G W
```

**Assessment:** This is already well-designed. Minor refinements only.

**Proposed:**

```
[██████████████████████████████░░░░░░░░░░░░░░░░░░░░] 18G 4A 0F 17P 40W
 Recent: W W → W W P G W
```

**Recommendations:**

- Slightly more compact stat format (drop `=`)
- Colour-code the Recent sequence by outcome type:
  - W (Wait/default): grey/dim
  - G (Germinate): green
  - P (Prune): magenta/red
  - A (Advance): cyan/blue
- Consider showing progress percentage or ETA if computable
- The `→` marker showing "current position" is a nice touch — keep it

---

### 7. Returns History

**Current:**

```
Returns: Ep0:-2.0  Ep-1:-6.6  Ep-2:-4.8  Ep-3:-3.1  Ep-4:-12.0
```

**Assessment:** Functional but could benefit from visual enhancement.

**Proposed:**

```
Returns: ▃█▅▄▁  Ep0:-2.0  Ep-1:-6.6  Ep-2:-4.8  Ep-3:-3.1  Ep-4:-12.0
```

**Recommendations:**

- Prepend mini sparkline showing the same data visually
- Consider colour gradient (red for worst, green for best in window)
- Highlight current episode (Ep0) with emphasis
- Optional: show rolling average or trend

---

### 8. Slots Summary Panel

**Current:**

```
SLOTS (96 across 48 envs)
DORM:45 ████  GERM:16 ██  TRAIN:7 █  BLEND:19 ███  HOLD:0  FOSS:9 █
Foss:38  Prune:2578  Rate:1%  AvgAge:8.5 epochs
```

**Issues:**

- Bar lengths don't correspond to consistent scale
- No percentage context for quick assessment
- "Foss" appears twice (count and state) — potentially confusing

**Proposed Option A — Detailed Breakdown:**

```
┌─ SLOTS ─────────────────── 96 across 48 envs ──────────────────┐
│  DORM   ████████████████████████████████████░░░░░░░░  45 (47%) │
│  GERM   ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  16 (17%) │
│  TRAIN  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7  (7%) │
│  BLEND  ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  19 (20%) │
│  HOLD   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0  (0%) │
│  FOSS   ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   9  (9%) │
├─────────────────────────────────────────────────────────────────┤
│  Fossils: 38   Pruned: 2578   Rate: 1%   AvgAge: 8.5 epochs    │
└─────────────────────────────────────────────────────────────────┘
```

**Proposed Option B — Compact Stacked Bar:**

```
SLOTS (96 across 48 envs)
 [████████████████████░░░░░░██░░░████████░░░░░░░░░░░███░░░░░░░░░]
  DORM:45       GERM:16  TR:7   BLEND:19    HOLD:0    FOSS:9

 Fossils:38  Pruned:2578  Rate:1%  AvgAge:8.5ep
```

**Recommendations:**

- Use consistent scale (all bars relative to total slots or max category)
- Add percentages for quick proportional understanding
- Clarify "Foss" vs "FOSS" vs "Fossils" terminology (state vs cumulative count)
- Colour-code by lifecycle stage (e.g., DORM=grey, GERM=green, TRAIN=yellow, etc.)
- Option A when vertical space available; Option B for compact view

---

### 9. Decisions Panel

**Current:**

```
DECISIONS
─────────────────────────────────────────────────────────────────
 #1 ADVANCE                                                   29s
    slot:1  confidence:24%
    host:29%  entropy:1.39 [exploring]                      [MISS]
    expect:-1.17 reward:-0.00  TD:+0.01
    also: WAIT 25%  GERMINATE 25%

 #2 GERMINATE                                                 56s
    slot:1  confidence:50%
    blueprint:conv_heavy ▸▸ linear×mul                          -
    host:28%  entropy:0.69 [confident]                      [MISS]
    expect:-1.12 reward:+0.00  TD:+0.09
    also: WAIT 50%  SET_ALPHA_TARGET 0%

 #3 ADVANCE                                                  1:28
    slot:1  confidence:32%
    host:28%  entropy:1.10 [balanced]                       [MISS]
    expect:-1.06 reward:+0.23  TD:+0.26
    also: WAIT 34%  PRUNE 34%
```

**Issues:**

- `slot:1` repeated on every entry — redundant if always 1
- `host:%` nearly identical across entries (28-29%) — could aggregate
- Status tag (MISS) alignment is ragged
- Information hierarchy is flat
- Timing right-aligned but visually disconnected from decision

**Proposed:**

```
DECISIONS                                              3 pending
┌────────────────────────────────────────────────────────────────┐
│ 1 ▌ADVANCE     24%  ░░▓░░░░░░░  29s   [exploring]        MISS │
│   H:1.39  expect:-1.17  r:-0.00  TD:+0.01                      │
│   also: WAIT 25%  GERMINATE 25%                                │
│                                                                │
│ 2 ▌GERMINATE   50%  ░░░░░▓░░░░  56s   [confident]        MISS │
│   conv_heavy ▸▸ linear×mul                                     │
│   H:0.69  expect:-1.12  r:+0.00  TD:+0.09                      │
│   also: WAIT 50%                                               │
│                                                                │
│ 3 ▌ADVANCE     32%  ░░░▓░░░░░░  1:28  [balanced]         MISS │
│   H:1.10  expect:-1.06  r:+0.23  TD:+0.26                      │
│   also: WAIT 34%  PRUNE 34%                                    │
└────────────────────────────────────────────────────────────────┘
```

**Recommendations:**

- Add mini confidence bar for instant visual assessment
- Remove `slot:1` when uniform (or show only on change)
- Aggregate `host:%` to panel header if uniform (e.g., "host avg: 28%")
- Right-align status indicators (MISS/HIT) consistently
- Consistent line semantics:
  - L1: Decision identity (type, confidence, age, state, outcome)
  - L2: Blueprint mutation (if applicable, otherwise skip)
  - L3: Metrics (entropy, expect, reward, TD)
  - L4: Alternative actions considered
- Consider colour-coding confidence bars:
  - <25%: red (low confidence)
  - 25-50%: yellow (moderate)
  - >50%: green (high confidence)
- Age/timer could colour-shift when stale (e.g., >60s → yellow, >120s → red)

---

## Colour Semantics (Proposed Palette)

Establish consistent colour meanings across all panels:

| Colour | Meaning | Example Usage |
|--------|---------|---------------|
| Green | Healthy, positive, success | GradHP ✓, improving trend ↗, HIT |
| Yellow | Warning, moderate, transitional | Approaching limits, moderate confidence |
| Red | Error, negative, critical | Failures, declining trend ↘, MISS |
| Cyan | Active, in-progress | TRAIN state, current episode |
| Magenta | Pruning, removal | PRUNE action, fossils |
| Grey/Dim | Dormant, waiting, neutral | DORM state, WAIT action |
| White | Normal text, values | Default metric values |

---

## Typography and Spacing

**Box Drawing Characters:**

- Use `┌┐└┘├┤┬┴┼` for clean panel borders
- Use `─` for horizontal rules, `│` for vertical
- Use `━` and `┃` for emphasis (e.g., main panel borders)

**Progress/Bar Characters:**

- Filled: `█▓▒░` (full → empty gradient)
- Alternative: `▰▱` (filled/empty blocks)
- Sparkline: `▁▂▃▄▅▆▇█` (8-level vertical)

**Status Glyphs:**

- Success: `✓` or `✔`
- Failure: `✗` or `✘`
- Warning: `!` or `⚠`
- Pending: `~` or `⋯`
- Spinner: `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` (braille rotation)

**Spacing Rules:**

- Minimum 1 space between distinct data elements
- 2 spaces between logical groups on same line
- Blank line between major sections within a panel
- Consistent indentation (2 spaces) for hierarchical data

---

## Implementation Priority

Suggested order for implementation:

1. **Decisions Panel** — Highest complexity, most user attention
2. **Header Bar** — Sets visual tone, seen constantly
3. **Training Metrics Grid** — Redundancy removal gives quick win
4. **PPO Core Stats** — Ratio visualisation adds value
5. **Attention Heads** — Layout choice impacts information density
6. **Slots Summary** — Important but lower interaction frequency
7. **Sparklines** — Refinement rather than redesign
8. **Episode Progress** — Already good, minor polish

---

## Open Questions for Implementation

1. **Responsive behaviour:** How should panels adapt when terminal width changes?
2. **Detail levels:** Should there be a "compact" vs "expanded" view toggle?
3. **Highlighting:** Should anomalous values auto-highlight, or only on threshold breach?
4. **History depth:** How many decisions/episodes to retain visible?
5. **Interactivity:** Any plans for drill-down on click/keypress?
6. **Refresh rate:** Does layout need to account for high-frequency updates?

---

## Reference: Full Layout Mockup

```
┏━ TAMIYO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ⠋ WARMING UP [5/50]          LR:1e-04  EntCoef:0.08          GradHP:12/12 ✓     batch:5/100   ┃
┠──────────────────────────────────────────────────────────────────────────────────────────────────┨
┃                                                                                                  ┃
┃  Ep.Return  ▁▂▁▃▂▄▃▅▄▆▅▇▆█  -2.0 ↘        │  Advantage   +0.00 ± 1.00                           ┃
┃  Entropy    █▇▆▇▅▆▄▅▃▄▂▃▁▂   7.91 →       │  Ratio       [0.98 ──●── 1.02]                      ┃
┃                                            │  Policy Loss  -0.390    Value Loss  35.80          ┃
┃  Expl.Var  ▓░░░░  0.004   Clip  ░░░░░  0  │  Grad Norm     1.00     Layers      12/12 ✓        ┃
┃  Entropy   ▓▓▓▓▓  7.91    KL    ░░░░░  0  │                                                     ┃
┃                                                                                                  ┃
┠──────────────────────────────────────────────────────────────────────────────────────────────────┨
┃  Attention Heads                                   Gradients                                     ┃
┃   slot ████  1.00    bpnt ████  1.00~              slot ▓▓░░  0.20    bpnt ▓▓▓░  0.38           ┃
┃   styl ▓░░░  0.22~   temp ████  1.00~              styl ▓▓░░  0.29    temp ▓░░░  0.13           ┃
┃   atgt ▓▓▓░  0.61~   aspd ████  1.00~              atgt ▓░░░  0.22    aspd ▓░░░  0.15           ┃
┃   acrv ████  1.00~   op   ▓▓▓░  0.91               acrv ▓▓░░  0.29    op   ▓░░░  0.16           ┃
┃                                                                                                  ┃
┠──────────────────────────────────────────────────────────────────────────────────────────────────┨
┃  [██████████████████████████████░░░░░░░░░░░░░░░░░░░░]  18G 4A 0F 17P 40W                        ┃
┃   Recent: W W → W W P G W              Returns: ▃█▅▄▁  -2.0  -6.6  -4.8  -3.1  -12.0            ┃
┃                                                                                                  ┃
┠──────────────────────────────────────────────────────────────────────────────────────────────────┨
┃  SLOTS (96/48 envs)  [████████████████████░░░░░░██░████████░░░░░░░███░░░]                        ┃
┃                       DORM:45    GERM:16  TR:7  BLEND:19  H:0  FOSS:9                            ┃
┃                       Fossils:38  Pruned:2578  Rate:1%  AvgAge:8.5ep                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Document History

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2025-01-XX | Claude (UX Review) | Initial briefing based on screenshot analysis |

---

*End of briefing*
