# Seed Stage Color Standardization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize seed lifecycle stage colors across all Sanctum widgets by defining canonical color mappings in leyline.

**Architecture:** Define `STAGE_COLORS` and `STAGE_ABBREVIATIONS` dictionaries in leyline's Display Thresholds section. Update 6 widget files to import from leyline instead of defining local mappings. Update existing test to import from new location.

**Tech Stack:** Python, Rich markup colors, leyline constants pattern

---

## Background: Current Inconsistencies

| Stage | env_detail | env_overview | tamiyo_brain | esper_status | run_header | scoreboard |
|-------|------------|--------------|--------------|--------------|------------|------------|
| DORMANT | dim | white | dim | — | — | — |
| GERMINATED | bright_blue | white | green | green | — | — |
| TRAINING | cyan | cyan | cyan | yellow | yellow | — |
| HOLDING | magenta | magenta | bright_cyan | blue | (cyan) | yellow |
| BLENDING | yellow | yellow | yellow | cyan | cyan | magenta |
| FOSSILIZED | green | green | blue | magenta | magenta | green |
| PRUNED | red | red | — | red | — | — |
| EMBARGOED | bright_red | bright_red | — | red | — | — |

**Canonical colors chosen:** Based on `env_detail_screen.py` which has the most complete mapping.

---

## Task 1: Add Canonical Stage Constants to Leyline

**Files:**
- Modify: `src/esper/leyline/__init__.py:281-282` (add after growth ratio constants)

**Step 1: Add the STAGE_COLORS dictionary**

Add these constants to `src/esper/leyline/__init__.py` after line 281 (after `DEFAULT_GROWTH_RATIO_YELLOW_MAX`):

```python
# Seed lifecycle stage colors (Rich markup)
# Used across all Sanctum widgets for consistent visual language
STAGE_COLORS: dict[str, str] = {
    "DORMANT": "dim",
    "GERMINATED": "bright_blue",
    "TRAINING": "cyan",
    "HOLDING": "magenta",
    "BLENDING": "yellow",
    "FOSSILIZED": "green",
    "PRUNED": "red",
    "EMBARGOED": "bright_red",
    "RESETTING": "dim",
}

# Stage abbreviations for compact display
STAGE_ABBREVIATIONS: dict[str, str] = {
    "DORMANT": "Dorm",
    "GERMINATED": "Germ",
    "TRAINING": "Train",
    "HOLDING": "Hold",
    "BLENDING": "Blend",
    "FOSSILIZED": "Foss",
    "PRUNED": "Prune",
    "EMBARGOED": "Embar",
    "RESETTING": "Reset",
}
```

**Step 2: Add to __all__ exports**

Find the `__all__` list and add:

```python
    "STAGE_COLORS",
    "STAGE_ABBREVIATIONS",
```

**Step 3: Verify import works**

Run: `PYTHONPATH=src python -c "from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS; print(STAGE_COLORS['TRAINING'])"`

Expected: `cyan`

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): add canonical STAGE_COLORS and STAGE_ABBREVIATIONS"
```

---

## Task 2: Update env_detail_screen.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_detail_screen.py:29-40`
- Test: `tests/karn/sanctum/test_env_detail_screen.py:14,364-372`

**Step 1: Update imports**

At the top of `env_detail_screen.py`, add import (around line 23, after other imports):

```python
from esper.leyline import STAGE_COLORS
```

**Step 2: Remove local STAGE_COLORS**

Delete lines 29-40 (the local `STAGE_COLORS` dictionary). The `STAGE_CSS_CLASSES` on lines 42-53 should remain - it's for CSS styling, not colors.

**Step 3: Update test import**

In `tests/karn/sanctum/test_env_detail_screen.py`, change line 14:

From:
```python
from esper.karn.sanctum.widgets.env_detail_screen import (
    EnvDetailScreen,
    SeedCard,
    STAGE_COLORS,
)
```

To:
```python
from esper.karn.sanctum.widgets.env_detail_screen import (
    EnvDetailScreen,
    SeedCard,
)
from esper.leyline import STAGE_COLORS
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_detail_screen.py -v -k "stage"`

Expected: PASS (test_stage_colors_defined should pass)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_detail_screen.py tests/karn/sanctum/test_env_detail_screen.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in env_detail_screen"
```

---

## Task 3: Update env_overview.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_overview.py:600-609`

**Step 1: Add import**

At the top of `env_overview.py`, add to the leyline import (should already have one for growth thresholds):

```python
from esper.leyline import (
    DEFAULT_GROWTH_RATIO_GREEN_MAX,
    DEFAULT_GROWTH_RATIO_YELLOW_MAX,
    STAGE_COLORS,
)
```

**Step 2: Replace local style_map**

In the `_format_slot_cell` method (around line 600), replace:

```python
        # Stage-specific styling
        style_map = {
            "TRAINING": "cyan",
            "BLENDING": "yellow",
            "HOLDING": "magenta",
            "FOSSILIZED": "green",
            "PRUNED": "red",
            "EMBARGOED": "bright_red",
            "RESETTING": "dim",
        }
        style = style_map.get(seed.stage, "white")
```

With:

```python
        # Stage-specific styling from leyline
        style = STAGE_COLORS.get(seed.stage, "white")
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_overview.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in env_overview"
```

---

## Task 4: Update tamiyo_brain.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py:606-621`

**Step 1: Add import**

At the top of `tamiyo_brain.py`, add:

```python
from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS
```

**Step 2: Replace local dictionaries**

In the `_render_slot_distribution` method (around line 606), replace:

```python
        stage_colors = {
            "DORMANT": "dim",
            "GERMINATED": "green",
            "TRAINING": "cyan",
            "BLENDING": "yellow",
            "HOLDING": "bright_cyan",
            "FOSSILIZED": "blue",
        }
        stage_abbrevs = {
            "DORMANT": "DORM",
            "GERMINATED": "GERM",
            "TRAINING": "TRAIN",
            "BLENDING": "BLEND",
            "HOLDING": "HOLD",
            "FOSSILIZED": "FOSS",
        }
```

With:

```python
        # Use leyline constants (uppercase abbrevs for this widget's style)
        stage_abbrevs = {k: v.upper() for k, v in STAGE_ABBREVIATIONS.items()}
```

Then update the usage at line 627 from `stage_colors[stage]` to `STAGE_COLORS.get(stage, "dim")`.

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in tamiyo_brain"
```

---

## Task 5: Update esper_status.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/esper_status.py:21-44`

**Step 1: Add import**

At the top of `esper_status.py`, add:

```python
from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS
```

**Step 2: Remove local dictionaries**

Delete lines 21-44 (both `_STAGE_SHORT` and `_STAGE_STYLES` dictionaries).

**Step 3: Update usage in render method**

At line 92-93, replace:

```python
                short = _STAGE_SHORT.get(stage, stage[:4])
                style = _STAGE_STYLES.get(stage, "dim")
```

With:

```python
                short = STAGE_ABBREVIATIONS.get(stage, stage[:4])
                style = STAGE_COLORS.get(stage, "dim")
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v -k "esper_status or EsperStatus"`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/esper_status.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in esper_status"
```

---

## Task 6: Update run_header.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/run_header.py:184-189`

**Step 1: Add import**

At the top of `run_header.py`, add:

```python
from esper.leyline import STAGE_COLORS
```

**Step 2: Update inline colors in _get_seed_stage_counts**

In the `_get_seed_stage_counts` method (lines 184-189), replace the inline color assignments:

From:
```python
        if training > 0:
            parts.append(f"[yellow]T:{training}[/]")
        if blending > 0:
            parts.append(f"[cyan]B:{blending}[/]")
        if fossilized > 0:
            parts.append(f"[magenta]F:{fossilized}[/]")
```

To:
```python
        if training > 0:
            parts.append(f"[{STAGE_COLORS['TRAINING']}]T:{training}[/]")
        if blending > 0:
            parts.append(f"[{STAGE_COLORS['BLENDING']}]B:{blending}[/]")
        if fossilized > 0:
            parts.append(f"[{STAGE_COLORS['FOSSILIZED']}]F:{fossilized}[/]")
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_run_header.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/run_header.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in run_header"
```

---

## Task 7: Update scoreboard.py

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/scoreboard.py:220-222`

**Step 1: Add import**

At the top of `scoreboard.py`, add:

```python
from esper.leyline import STAGE_COLORS
```

**Step 2: Remove local stage_colors**

In the `_format_seeds` method (around line 221), replace:

```python
        stage_order = {"FOSSILIZED": 0, "BLENDING": 1, "HOLDING": 2}
        stage_colors = {"FOSSILIZED": "green", "BLENDING": "magenta", "HOLDING": "yellow"}
```

With:

```python
        stage_order = {"FOSSILIZED": 0, "BLENDING": 1, "HOLDING": 2}
        # Use leyline STAGE_COLORS for consistency
```

Then update line 227 from `stage_colors.get(seed.stage, "dim")` to `STAGE_COLORS.get(seed.stage, "dim")`.

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_remaining_widgets.py -v -k "scoreboard"`

Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/scoreboard.py
git commit -m "refactor(sanctum): use leyline STAGE_COLORS in scoreboard"
```

---

## Task 8: Final Verification

**Step 1: Run full sanctum test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests PASS

**Step 2: Verify visual consistency**

Run: `PYTHONPATH=src python -c "
from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS
print('=== Canonical Stage Colors ===')
for stage, color in STAGE_COLORS.items():
    abbrev = STAGE_ABBREVIATIONS.get(stage, stage[:4])
    print(f'{abbrev:8} ({stage:12}): {color}')
"`

Expected output:
```
=== Canonical Stage Colors ===
Dorm     (DORMANT     ): dim
Germ     (GERMINATED  ): bright_blue
Train    (TRAINING    ): cyan
Hold     (HOLDING     ): magenta
Blend    (BLENDING    ): yellow
Foss     (FOSSILIZED  ): green
Prune    (PRUNED      ): red
Embar    (EMBARGOED   ): bright_red
Reset    (RESETTING   ): dim
```

**Step 3: Verify imports work in all widgets**

Run: `PYTHONPATH=src python -c "
from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen
from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
from esper.karn.sanctum.widgets.esper_status import EsperStatus
from esper.karn.sanctum.widgets.run_header import RunHeader
from esper.karn.sanctum.widgets.scoreboard import Scoreboard
print('All widgets import successfully')
"`

Expected: `All widgets import successfully`

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Add leyline constants | `leyline/__init__.py` |
| 2 | Update env_detail_screen | `env_detail_screen.py`, `test_env_detail_screen.py` |
| 3 | Update env_overview | `env_overview.py` |
| 4 | Update tamiyo_brain | `tamiyo_brain.py` |
| 5 | Update esper_status | `esper_status.py` |
| 6 | Update run_header | `run_header.py` |
| 7 | Update scoreboard | `scoreboard.py` |
| 8 | Final verification | — |

**Total estimated time:** 30-40 minutes

**Note:** `event_log.py` is intentionally NOT updated. It uses event-semantic colors (bright_yellow for germination *event*, bright_green for fossilization *event*) which emphasize the transition moment rather than the stage itself. This is a deliberate UX choice.

**Future enhancement:** If stage colors need to be runtime-configurable, they can be moved to a `DisplayConfig` dataclass. For now, leyline constants provide compile-time configurability matching existing patterns.
