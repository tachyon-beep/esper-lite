# Tamiyo Restructuring: Micro-Step Implementation Plan

**Philosophy:** Each step should be:
1. **Independently testable** - run the TUI, visually verify
2. **Reversible** - easy to revert if something looks wrong
3. **Non-breaking** - old functionality works until explicitly removed
4. **Small** - ideally < 20 lines changed per step

**Testing approach:** After each step:
```bash
uv run python -m esper.scripts.train ppo --sanctum --telemetry-level debug --envs 2 --episodes 5
```
Look at Tamiyo panel, verify nothing broke, verify change is visible.

---

## Phase A: Infrastructure Prep (No Visual Changes)

These steps add code but change nothing visible. Safe to batch if desired.

### A1: Add section header CSS class
**File:** `tamiyo.tcss`
**Change:** Add new CSS class (unused until later)
```css
.section-header {
    text-style: bold;
    color: $text-muted;
    padding: 0 1;
    height: 1;
}
```
**Test:** TUI loads, no visual change

### A2: Add status symbol constants
**File:** `src/esper/karn/sanctum/widgets/tamiyo/__init__.py` (or new `constants.py`)
**Change:** Add constants (unused until later)
```python
STATUS_SYMBOLS = {
    "ok": "✓",
    "warning": "⚠",
    "critical": "✗",
    "info": "●",
}
```
**Test:** TUI loads, no visual change

### A3: Add label width constant
**File:** `src/esper/karn/sanctum/widgets/tamiyo/__init__.py`
**Change:** Add constant
```python
STANDARD_LABEL_WIDTH = 13
```
**Test:** TUI loads, no visual change

---

## Phase B: Additive Changes (Add Without Removing)

Each step adds something new without removing the old version.

### B1: Add NaN/Inf to NarrativePanel (keep in ActionHeads too)
**File:** `narrative_panel.py`
**Change:** Add NaN/Inf display to NOW line (data already available in snapshot)
**Visual:** NOW line shows `⚠ NaN:0 Inf:0`
**Test:** Both NarrativePanel AND ActionHeadsPanel show NaN/Inf counts

### B2: Add status symbol to ONE metric in NarrativePanel
**File:** `narrative_panel.py`
**Change:** Add ✓/⚠/✗ prefix to KL divergence display
**Visual:** KL shows `✓ 0.008` instead of just `0.008`
**Test:** Symbol appears, color still works

### B3: Add status symbol to ONE metric in PPOLossesPanel
**File:** `ppo_losses_panel.py`
**Change:** Add symbol to Explained Variance
**Test:** Symbol appears alongside gauge

### B4-B8: Repeat B3 for other panels (one commit each)
- B4: HealthStatusPanel - add symbol to grad norm
- B5: ValueDiagnosticsPanel - add symbol to trend
- B6: CriticCalibrationPanel - add symbol to calibration summary
- B7: TorchStabilityPanel - add symbol to memory %
- B8: ActionHeadsPanel - add symbol to head states

### B9: Add first section header
**File:** `tamiyo.py`
**Change:** Add `Static("━━ POLICY HEALTH ━━", classes="section-header")` before NarrativePanel
**Test:** Header visible, panels unchanged

### B10: Add second section header
**File:** `tamiyo.py`
**Change:** Add `━━ DIAGNOSTICS ━━` header before HealthStatusPanel
**Test:** Both headers visible

### B11: Add third section header
**File:** `tamiyo.py`
**Change:** Add `━━ INFRASTRUCTURE ━━` header before TorchStabilityPanel
**Test:** All three headers visible

---

## Phase C: Create Empty Shell Panels

Create new consolidated panels as empty/minimal shells, add to layout but keep old panels too.

### C1: Create empty ValueFunctionPanel file
**File:** NEW `value_function_panel.py`
**Change:** Create minimal panel with just a title, no content
```python
class ValueFunctionPanel(Static):
    def compose(self):
        yield Static("VALUE FUNCTION QUALITY")
        yield Static("(migrating...)")
```
**Test:** File exists, not yet in layout

### C2: Add ValueFunctionPanel to layout (alongside old panels)
**File:** `tamiyo.py`
**Change:** Add ValueFunctionPanel below the old panels (temporary location)
**Test:** New empty panel visible at bottom, old panels unchanged

### C3: Create empty GradientHealthPanel file
**File:** NEW `gradient_health_panel.py`
**Change:** Minimal shell
**Test:** File exists, not in layout yet

### C4: Add GradientHealthPanel to layout
**File:** `tamiyo.py`
**Test:** Empty panel visible, old panels unchanged

---

## Phase D: Migrate Metrics One-by-One

Move ONE metric at a time from old panel to new panel. Old panel keeps working with remaining metrics.

### Value Function Panel Migration

#### D1: Move Explained Variance from CriticCalibration → ValueFunction
**Files:** `value_function_panel.py`, `critic_calibration_panel.py`
**Change:**
- Add EV display to ValueFunctionPanel
- Keep EV in CriticCalibrationPanel (duplicate temporarily)
**Test:** EV shows in BOTH panels

#### D2: Remove EV from CriticCalibrationPanel
**File:** `critic_calibration_panel.py`
**Change:** Remove EV display
**Test:** EV only in ValueFunctionPanel, CriticCalibration still has other metrics

#### D3: Move V-Return Correlation → ValueFunction
**Test:** V-Corr in new panel

#### D4: Move TD Error → ValueFunction
**Test:** TD Error in new panel

#### D5: Move Bellman Error → ValueFunction
**Test:** Bellman in new panel

#### D6: Move Calibration Summary → ValueFunction
**Test:** Summary in new panel

#### D7: Verify CriticCalibrationPanel is empty
**Test:** Panel should show no metrics (or placeholder)

#### D8: Move Return Percentiles from ValueDiagnostics → ValueFunction
**Test:** Percentiles in new panel

#### D9: Move Return σ/skew → ValueFunction
**Test:** Stats in new panel

#### D10: Move Return Mean/Trend → ValueFunction
**Test:** Mean in new panel

#### D11: Verify ValueDiagnosticsPanel is empty
**Test:** Panel should be empty

#### D12: Move Value Loss from PPOLosses → ValueFunction
**Test:** Value Loss sparkline in new panel

#### D13: Move Lv/Lp Ratio from PPOLosses → ValueFunction
**Test:** Ratio in new panel

#### D14: Remove EV gauge from PPOLossesPanel (was duplicate)
**File:** `ppo_losses_panel.py`
**Change:** Remove the duplicate EV gauge
**Test:** EV only in ValueFunctionPanel now

### Gradient Health Panel Migration

#### D15: Move Advantage stats from HealthStatus → GradientHealth
**Test:** Advantage μ/σ in new panel

#### D16: Move Advantage skew/kurtosis → GradientHealth
**Test:** Shape stats in new panel

#### D17: Move Advantage positive ratio → GradientHealth
**Test:** Pos% in new panel

#### D18: Move Gradient Norm from HealthStatus → GradientHealth
**Test:** Grad norm + sparkline in new panel

#### D19: Move Log Prob extremes → GradientHealth
**Test:** Log prob range in new panel

#### D20: Move Observation Health → GradientHealth
**Test:** Obs health in new panel

#### D21: Move Gradient Flow metrics from ActionHeadsPanel → GradientHealth
**Test:** CV, dead, exploding in new panel

### Policy Optimization Panel Consolidation

#### D22: Move Entropy from HealthStatus → PolicyOptimization (rename of PPOLosses)
**Test:** Entropy in PPOLosses area

#### D23: Move Policy State from HealthStatus → PolicyOptimization
**Test:** Policy state in PPOLosses area

#### D24: Move Yield Rate from EpisodeMetrics → PolicyOptimization
**Test:** Yield in PPOLosses area

#### D25: Move Slot Utilization from EpisodeMetrics → PolicyOptimization
**Test:** Utilization in PPOLosses area

---

## Phase E: Remove Empty Panels

Only after all metrics migrated and verified.

### E1: Remove CriticCalibrationPanel from layout
**File:** `tamiyo.py`
**Change:** Remove from compose()
**Test:** Panel gone, ValueFunctionPanel has all its metrics

### E2: Remove ValueDiagnosticsPanel from layout
**File:** `tamiyo.py`
**Test:** Panel gone

### E3: Remove EpisodeMetricsPanel from layout
**File:** `tamiyo.py`
**Test:** Panel gone, metrics now in PolicyOptimization

### E4: Remove HealthStatusPanel from layout
**File:** `tamiyo.py`
**Test:** Panel gone, metrics split between GradientHealth and PolicyOptimization

### E5: Delete empty panel files
**Files:** Delete `critic_calibration_panel.py`, `value_diagnostics_panel.py`, `episode_metrics_panel.py`, `health_status_panel.py`
**Test:** No import errors, TUI still works

---

## Phase F: Layout Adjustments

Now adjust the layout to match the target design.

### F1: Rename PPOLossesPanel → PolicyOptimizationPanel
**Files:** Rename file, update imports
**Test:** Same panel, new name

### F2: Move ValueFunctionPanel to center column
**File:** `tamiyo.py`
**Change:** Adjust CSS grid placement
**Test:** Panel in correct location

### F3: Move GradientHealthPanel below ValueFunction
**File:** `tamiyo.py`
**Test:** Panels stacked correctly

### F4: Move SlotsPanel to right column
**File:** `tamiyo.py`
**Test:** SlotsPanel above Decisions

### F5: Adjust panel heights
**File:** `tamiyo.tcss`
**Test:** Panels sized appropriately

### F6: Remove NaN/Inf from ActionHeadsPanel (now in Narrative)
**File:** `action_heads_panel.py`
**Test:** No duplicate, Narrative has it

---

## Phase G: Polish

### G1: Standardize label widths in ValueFunctionPanel
**Test:** Labels aligned

### G2: Standardize label widths in GradientHealthPanel
**Test:** Labels aligned

### G3: Standardize label widths in PolicyOptimizationPanel
**Test:** Labels aligned

### G4: Update panel border titles
**Test:** Titles match design doc

### G5: Final visual review
**Test:** Compare to wireframe in design doc

---

## Step Count Summary

| Phase | Steps | Risk | Description |
|-------|-------|------|-------------|
| A | 3 | None | Infrastructure prep |
| B | 11 | Low | Additive changes |
| C | 4 | Low | Empty shell panels |
| D | 25 | Medium | Metric migration |
| E | 5 | Medium | Remove empty panels |
| F | 6 | Medium | Layout adjustments |
| G | 5 | Low | Polish |
| **Total** | **59** | | |

Each step is ~5-20 lines of code changes. At 5-10 minutes per step (including testing), this is roughly 5-10 hours of work spread across multiple sessions.

---

## Checkpoint Strategy

**Save points** (good places to pause work):
- After Phase A: Infrastructure ready
- After Phase B: Visual improvements done, no structural changes
- After D14: ValueFunctionPanel complete
- After D21: GradientHealthPanel complete
- After D25: All metrics migrated
- After Phase E: Old panels removed
- After Phase F: Layout finalized

**Rollback strategy:** Each phase can be reverted independently. If Phase D causes issues, revert to end of Phase C and old panels still work.

---

## Automation Opportunities

Some steps could be batched with a script:

```bash
# Run after each step
./scripts/test-tamiyo.sh

# Contents:
#!/bin/bash
uv run python -c "from esper.karn.sanctum.widgets.tamiyo import *; print('Imports OK')"
echo "Manual test: run sanctum and verify visually"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Metric displays incorrectly in new location | Keep old panel until verified |
| Layout breaks on different terminal sizes | Test at 80col, 120col, 200col |
| Performance regression from duplicate panels | Phase D is temporary, E removes dupes |
| Merge conflicts if main branch changes | Keep commits small, rebase frequently |
