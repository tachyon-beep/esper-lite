# ENH-029: TamiyoBrain A/B Testing Polish

- **Title:** TamiyoBrain A/B Testing Polish - Follow-up from Expanded TamiyoBrain Plan
- **Context:** Sanctum TUI / TamiyoBrain widget / A/B testing visualization
- **Impact:** Low - cosmetic and defensive improvements, no functional gaps
- **Environment:** Post docs/plans/2025-12-24-expanded-tamiyo-brain.md implementation
- **Status:** Closed (2025-12-24)

## Background

During subagent-driven implementation of the Expanded TamiyoBrain plan, three specialist reviewers (DRL Expert, UX Specialist, Code Reviewer) identified minor polish items that were not blocking but would improve robustness and UX quality.

## Recommended Enhancements

### ~~1. Filter "default" group_id in Aggregator~~

**Status:** RESOLVED (commit 1c0c38e)

**Source:** Code Reviewer (Task 5.5)

**Resolution:** Added `and group_id != "default"` condition to filter out the default value in single-policy mode.

**Files:** `src/esper/karn/sanctum/aggregator.py`

---

### ~~2. Add Edge Case Tests for group_id~~

**Status:** RESOLVED (commit 8906182)

**Source:** Code Reviewer, DRL Expert (Tasks 5.3, 5.4, 5.5)

**Resolution:** Added 7 edge case tests covering: None, "default", A→B transition, group C, and unknown identifiers (e.g., "experiment_42").

**Files:** `tests/karn/sanctum/test_tamiyo_brain.py`, `tests/karn/sanctum/test_aggregator.py`

---

### ~~3. Upstream Telemetry Gap: emit_ppo_update_event~~

**Status:** RESOLVED (commit 50f189a)

**Source:** DRL Expert (Task 5.5)

**Resolution:** Added `group_id: str = "default"` parameter to `emit_ppo_update_event()` and propagated to TelemetryEvent. Backward compatible via default parameter.

**Files:** `src/esper/simic/telemetry/emitters.py`

---

### ~~4. Add group_id to border_title for Accessibility~~

**Status:** RESOLVED (commit f0846d8)

**Source:** UX Specialist (Tasks 5.3, 5.4)

**Resolution:** Updated `update_snapshot()` to set border_title dynamically: `TAMIYO` for single-policy, `TAMIYO [A]` for A/B mode. Properly escapes bracket for Rich markup.

**Files:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`

---

### ~~5. Strengthen CSS Separator for Group Label~~

**Status:** RESOLVED (commit fc6df24)

**Source:** UX Specialist (Task 5.4)

**Resolution:** Changed separator from light bar (`│`) to heavy bar (`┃`) with spacing for better visual separation in A/B mode.

**Files:** `src/esper/karn/sanctum/widgets/tamiyo_brain.py`

---

### ~~6. CSS Focus State Duplication~~

**Status:** RESOLVED (commit 91d37ee)

**Source:** Code Reviewer (Task 6.5)

**Resolution:** Removed duplicate `.focused` class and manual `on_focus`/`on_blur` handlers. Kept Textual's built-in `:focus` pseudo-class only.

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
