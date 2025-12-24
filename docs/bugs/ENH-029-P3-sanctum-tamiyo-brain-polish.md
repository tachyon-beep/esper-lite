# ENH-029: TamiyoBrain A/B Testing Polish

- **Title:** TamiyoBrain A/B Testing Polish - Follow-up from Expanded TamiyoBrain Plan
- **Context:** Sanctum TUI / TamiyoBrain widget / A/B testing visualization
- **Impact:** Low - cosmetic and defensive improvements, no functional gaps
- **Environment:** Post docs/plans/2025-12-24-expanded-tamiyo-brain.md implementation
- **Status:** Open

## Background

During subagent-driven implementation of the Expanded TamiyoBrain plan, three specialist reviewers (DRL Expert, UX Specialist, Code Reviewer) identified minor polish items that were not blocking but would improve robustness and UX quality.

## Recommended Enhancements

### 1. Filter "default" group_id in Aggregator

**Source:** Code Reviewer (Task 5.5)

**Issue:** `TelemetryEvent.group_id` defaults to `"default"` for single-policy training. Current code sets `tamiyo.group_id = "default"` which causes `[default]` label to appear in status banner.

**Fix:**
```python
# In aggregator.py _handle_ppo_update
group_id = event.group_id
if group_id and group_id != "default":
    self._tamiyo.group_id = group_id
```

**Files:** `src/esper/karn/sanctum/aggregator.py`

---

### 2. Add Edge Case Tests for group_id

**Source:** Code Reviewer, DRL Expert (Tasks 5.3, 5.4, 5.5)

**Missing test coverage:**
- `group_id=None` - verify no group label appears
- `group_id="default"` - verify treated as non-A/B mode
- Group ID change (A→B transition) - verify old class removed
- Group C - only A and B tested, not C
- Unknown group_id (e.g., "D") - verify fallback `[D]` format

**Files:** `tests/karn/sanctum/test_tamiyo_brain.py`, `tests/karn/sanctum/test_aggregator.py`

---

### 3. Upstream Telemetry Gap: emit_ppo_update_event

**Source:** DRL Expert (Task 5.5)

**Issue:** `emit_ppo_update_event()` in `emitters.py` does not propagate `group_id` to TelemetryEvent. Events use default `"default"` value.

**Fix:** Update `emit_ppo_update_event` to accept and propagate `group_id` parameter:
```python
def emit_ppo_update_event(hub, group_id: str = "default", ...):
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        group_id=group_id,
        ...
    ))
```

**Files:** `src/esper/nissa/emitters.py`, `src/esper/simic/training/vectorized.py`

---

### 4. Add group_id to border_title for Accessibility

**Source:** UX Specialist (Tasks 5.3, 5.4)

**Issue:** Widget `border_title` is hardcoded to `"TAMIYO"`. Screen readers and colorblind users benefit from redundant group identification.

**Fix:**
```python
# In update_snapshot or _update_status_class
if self._snapshot and self._snapshot.tamiyo.group_id:
    self.border_title = f"TAMIYO [{self._snapshot.tamiyo.group_id}]"
else:
    self.border_title = "TAMIYO"
```

**Files:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`

---

### 5. Strengthen CSS Separator for Group Label

**Source:** UX Specialist (Task 5.4)

**Issue:** The dim pipe `│` separator between group label and status may be insufficient in dense multi-widget views.

**Suggestion:** Consider double pipe `││` or additional spacing:
```python
banner.append(" ││ ", style="dim")  # Heavier separator
# or
banner.append("  │  ", style="dim")  # More spacing
```

**Files:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py` (`_render_status_banner`)

---

### 6. CSS Focus State Duplication

**Source:** Code Reviewer (Task 6.5)

**Issue:** Both `:focus` pseudo-class AND `.focused` class apply the same border style. Choose one approach.

**Options:**
- Keep CSS `:focus` only (if Textual handles focus state automatically)
- Keep `.focused` class only (if manual control preferred)

**Files:** `src/esper/karn/sanctum/styles.tcss`, `src/esper/karn/sanctum/widgets/tamiyo_brain.py`

---

### ~~7. Reduce EnvsList and BestRuns Height for Layout Balance~~

**Status:** RESOLVED

**Resolution:** ComparisonHeader was consolidated into RunHeader, freeing up 3 rows of vertical space. Height reduction is no longer necessary.

**Commit:** (pending) - Moved A/B comparison display into RunHeader, deleted ComparisonHeader widget.

---

## Validation Plan

1. Run full Sanctum test suite: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
2. Manual verification with `--dual-ab` flag once upstream telemetry gap is fixed
3. Visual inspection of single-policy mode (no `[default]` label)
4. Screen reader testing for border_title accessibility

## Priority

P3 - Polish items. All functional requirements from the plan are complete and tested.

## Links

- Plan: `docs/plans/2025-12-24-expanded-tamiyo-brain.md`
- Branch: `sanctum-tamiyo-ux-enhancement`
- Commits: ae0d147..HEAD (Phase 5 implementation + ComparisonHeader consolidation)
