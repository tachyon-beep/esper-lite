# Growth Threshold Configuration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose growth ratio display thresholds as configurable constants in leyline, replacing hardcoded values in UI widgets.

**Architecture:** Add DEFAULT_GROWTH_RATIO_* constants to leyline following existing patterns. Import these in env_overview.py and scoreboard.py. Tests verify the widgets respect the constants.

**Tech Stack:** Python dataclasses, leyline constants, pytest

---

## Task 1: Add Growth Threshold Constants to Leyline

**Files:**
- Modify: `src/esper/leyline/__init__.py` (add after line ~250, near other display thresholds)

**Step 1: Add the constants**

Add these constants to `src/esper/leyline/__init__.py` in the "Display Thresholds" section (create section if needed):

```python
# =============================================================================
# Display Thresholds (Karn UI)
# =============================================================================

# Growth ratio: (host+fossilized_params) / host_params
# Controls color coding in env_overview and scoreboard widgets
DEFAULT_GROWTH_RATIO_GREEN_MAX = 2.0   # <2x = green (efficient)
DEFAULT_GROWTH_RATIO_YELLOW_MAX = 5.0  # 2-5x = yellow (moderate), >5x = red (heavy)
```

**Step 2: Export the constants**

Add to the `__all__` list in `src/esper/leyline/__init__.py`:

```python
    "DEFAULT_GROWTH_RATIO_GREEN_MAX",
    "DEFAULT_GROWTH_RATIO_YELLOW_MAX",
```

**Step 3: Verify import works**

Run: `PYTHONPATH=src python -c "from esper.leyline import DEFAULT_GROWTH_RATIO_GREEN_MAX; print(DEFAULT_GROWTH_RATIO_GREEN_MAX)"`

Expected: `2.0`

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): add growth ratio display threshold constants"
```

---

## Task 2: Update EnvOverview Widget to Use Constants

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_overview.py:422-441`
- Test: `tests/karn/sanctum/test_env_overview.py` (add new test)

**Step 1: Write the failing test**

Add to `tests/karn/sanctum/test_env_overview.py`:

```python
def test_growth_ratio_respects_leyline_thresholds():
    """Growth ratio coloring uses leyline constants, not hardcoded values."""
    from esper.leyline import DEFAULT_GROWTH_RATIO_GREEN_MAX, DEFAULT_GROWTH_RATIO_YELLOW_MAX

    # Verify constants are what we expect (guards against accidental changes)
    assert DEFAULT_GROWTH_RATIO_GREEN_MAX == 2.0
    assert DEFAULT_GROWTH_RATIO_YELLOW_MAX == 5.0

    # Create widget and test formatting
    from esper.karn.sanctum.widgets.env_overview import EnvOverview
    from esper.karn.sanctum.schema import EnvState

    widget = EnvOverview(num_envs=1)

    # Test green threshold (just under 2.0)
    env_green = EnvState(env_id=0, growth_ratio=1.99)
    result = widget._format_growth_ratio(env_green)
    assert "[green]" in result

    # Test yellow threshold (at 2.0, just under 5.0)
    env_yellow = EnvState(env_id=0, growth_ratio=2.0)
    result = widget._format_growth_ratio(env_yellow)
    assert "[yellow]" in result

    env_yellow2 = EnvState(env_id=0, growth_ratio=4.99)
    result = widget._format_growth_ratio(env_yellow2)
    assert "[yellow]" in result

    # Test red threshold (at 5.0 and above)
    env_red = EnvState(env_id=0, growth_ratio=5.0)
    result = widget._format_growth_ratio(env_red)
    assert "[red]" in result
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py::test_growth_ratio_respects_leyline_thresholds -v`

Expected: FAIL (test file may not exist or test function doesn't exist yet)

**Step 3: Update env_overview.py to use constants**

Modify `src/esper/karn/sanctum/widgets/env_overview.py`:

Add import at top (around line 12):
```python
from esper.leyline import (
    DEFAULT_GROWTH_RATIO_GREEN_MAX,
    DEFAULT_GROWTH_RATIO_YELLOW_MAX,
)
```

Replace the `_format_growth_ratio` method (lines 422-441):
```python
    def _format_growth_ratio(self, env: "EnvState") -> str:
        """Format growth ratio: (host+fossilized)/host.

        Shows how much larger the mutated model is vs baseline.
        - 1.0x = no growth (baseline or no fossilized seeds)
        - Green if < DEFAULT_GROWTH_RATIO_GREEN_MAX
        - Yellow if < DEFAULT_GROWTH_RATIO_YELLOW_MAX
        - Red otherwise

        Thresholds are configurable via leyline constants. Generous defaults
        because small host models can easily double with a single attention seed.
        """
        ratio = env.growth_ratio
        if ratio <= 1.0:
            return "[dim]1.0x[/dim]"
        elif ratio < DEFAULT_GROWTH_RATIO_GREEN_MAX:
            return f"[green]{ratio:.2f}x[/green]"
        elif ratio < DEFAULT_GROWTH_RATIO_YELLOW_MAX:
            return f"[yellow]{ratio:.2f}x[/yellow]"
        else:
            return f"[red]{ratio:.2f}x[/red]"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py::test_growth_ratio_respects_leyline_thresholds -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_overview.py tests/karn/sanctum/test_env_overview.py
git commit -m "refactor(sanctum): use leyline constants for growth ratio thresholds"
```

---

## Task 3: Update Scoreboard Widget for Consistency (Optional)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/scoreboard.py:241-248`

**Context:** The scoreboard uses a different color scheme (cyan intensity) for growth ratio since it shows "best runs" where any growth is positive. This task is optional - only proceed if you want consistent threshold-based coloring.

**Step 1: Review current behavior**

The scoreboard currently uses:
- `<= 1.0` → dim
- `< 1.1` → cyan
- `>= 1.1` → bold cyan

This is intentionally different from env_overview (celebrating growth, not warning about it).

**Decision point:** Keep scoreboard as-is (recommended) or align with leyline thresholds.

**If keeping as-is:** Skip to Task 4.

**If aligning:** Update similarly to Task 2, but consider whether warning colors make sense for a "best runs" display.

---

## Task 4: Verify All Tests Pass

**Step 1: Run full test suite for affected modules**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests PASS

**Step 2: Run import verification**

Run: `PYTHONPATH=src python -c "from esper.karn.sanctum.widgets.env_overview import EnvOverview; print('OK')"`

Expected: `OK`

**Step 3: Final commit if any fixups needed**

```bash
git add -A
git commit -m "test(sanctum): verify growth threshold configuration"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Add leyline constants | `leyline/__init__.py` |
| 2 | Update EnvOverview | `env_overview.py`, `test_env_overview.py` |
| 3 | (Optional) Update Scoreboard | `scoreboard.py` |
| 4 | Verify tests | - |

**Total estimated time:** 15-20 minutes

**Future enhancement:** If thresholds need to be runtime-configurable (not just code constants), consider adding them to `KarnConfig` or a new `DisplayConfig` dataclass. For now, leyline constants provide compile-time configurability which matches the existing pattern.
