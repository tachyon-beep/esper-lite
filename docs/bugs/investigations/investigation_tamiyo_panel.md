# Investigation: Tamiyo Panel Bug in Sanctum UX

**Date:** 2026-01-02
**Investigator:** Gemini Agent
**Status:** In Progress

## Objective
Fact-finding for a reported bug in the Tamiyo panel of the Sanctum UX within the Karn subsystem.

## Context
- **Subsystem:** Karn (`src/esper/karn`)
- **Component:** Sanctum UX (`src/esper/karn/sanctum`)
- **Target:** Tamiyo Panel (`src/esper/karn/sanctum/widgets/tamiyo_brain`)

## Findings

### 1. Layout Overflow (Critical)
The `HeadsPanel` and `AttentionHeatmapPanel` require significantly more horizontal space than allocated, leading to likely truncation or wrapping (breaking the ASCII table alignment).

**Calculations:**
- **Required Width:** Both panels use a fixed-width ASCII table layout requiring **103 characters**.
  - `HeadsPanel` columns: Label(6)+Gutter(5)+Op(7)+Gutter(4)+Slot(7)+Gutter(4)+Blueprint(10)+Gutter(5)+Style(9)+Gutter(2)+Tempo(9)+Gutter(3)+αTarget(9)+Gutter(3)+αSpeed(9)+Gutter(2)+Curve(9) = **94 chars**? 
  - *Correction on HeadsPanel calculation:*
    - Row label: 6 chars ("Entr  ")
    - `_PRE_OP_GUTTER`: 5 chars
    - `Op`: 7 chars + 4 gutter = 11
    - `Slot`: 7 chars + 4 gutter = 11
    - `Blueprint`: 10 chars + 5 gutter = 15
    - `Style`: 9 chars + 2 gutter = 11
    - `Tempo`: 9 chars + 3 gutter = 12
    - `αTarget`: 9 chars + 3 gutter = 12
    - `αSpeed`: 9 chars + 2 gutter = 11
    - `Curve`: 9 chars (no trailing gutter)
    - **Total:** 6+5 + 11+11+15+11+12+12+11+9 = **103 chars**.
- **Allocated Width:**
  - `TamiyoBrain` widget width = 80% of screen width (single group).
  - `HeadsPanel` width = 68% of `TamiyoBrain` width.
  - Net width = 0.80 * 0.68 = **54.4%** of screen width.
- **Scenario: Minimum Terminal (120 chars)**
  - Available: 120 * 0.544 = **65.28 chars**.
  - Required: **103 chars**.
  - **Result:** ~37 chars of overflow/truncation.
- **Scenario: Recommended Terminal (140 chars)**
  - Available: 140 * 0.544 = **76.16 chars**.
  - Required: **103 chars**.
  - **Result:** ~27 chars of overflow/truncation.
- **Scenario: Wide Terminal (190 chars)**
  - Available: 190 * 0.544 = **103.36 chars**.
  - **Result:** Barely fits.

**Conclusion:** The layout is broken for all but extremely wide terminals (>190 cols). The "Recommended" size of 140x50 is insufficient.

### 2. Card Swapping Logic (Potential Race)
In `DecisionsColumn._render_cards`:
```python
            # Remove ALL existing cards (use list() to avoid mutation during iteration)
            for card in list(container.query(DecisionCard)):
                card.remove()
            ...
            # Mount new cards
            for i, decision in enumerate(self._displayed_decisions):
                ...
                container.mount(card)
```
`card.remove()` is async/scheduled. Mounting immediately after might cause ID collisions or ordering issues if Textual doesn't guarantee atomic swap in this manner. Though `_render_generation` suffix on IDs mitigates collision, visual artifacts might occur.

### 3. Missing Data Handling
`HeadsPanel` manual alignment relies on `AttentionHeatmapPanel` column widths, but they have different internal column definitions (`HEAD_CONFIG` widths vs `AttentionHeatmapPanel` `COL_*` constants). While they align mathematically now, any change to one will break the other.

## Suggested Fixes

### Short-term (Quick Fix)
1. **Enable Horizontal Scrolling:**
   - Update `styles.tcss` for `#heads-panel` and `#attention-heatmap` to include `overflow-x: auto`.
   - Ensure the parent container allows scrolling or the widget expands to its content width.
   - Note: This might degrade UX by requiring scrolling to see all heads.

### Medium-term (Better UX)
2. **Compact Layout:**
   - Redesign `AttentionHeatmapPanel` and `HeadsPanel` to use a more compact representation.
   - Example: Group related heads (e.g., αTarget, αSpeed, Curve) or use abbreviated headers.
   - Reduce gutter sizes. Currently `Op` has 4 chars gutter, `Blueprint` has 5. Reducing gutters to 1-2 chars could save ~20 chars.
   - Current: 103 chars. Target: ~80 chars (fits in 120-width terminal with 68% split -> 81 chars).
   - **Calculation:** Saving 20 chars gets us to 83 chars. Almost there.

3. **Responsive/Adaptive Layout:**
   - Detect terminal width and switch to a "compact" mode (hiding less critical heads or using shorter abbreviations) if width < 150.

## Next Steps
1. Verify if `Static` widgets wrap lines by default (breaking the table).
2. Confirm if the user considers this the "bug" (highly likely given "Sanctum UX").
3. Propose a layout fix (e.g., horizontal scrolling, reducing columns, or changing the 68/32 split).