# Sanctum Hindsight Credit UX Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire scaffold hindsight credit telemetry to Sanctum's EnvDetailScreen with a refactored multi-row reward breakdown layout.

**Architecture:** Add hindsight credit fields to Sanctum's `RewardComponents` schema, wire aggregator to populate from telemetry, refactor `EnvDetailScreen._render_metrics()` to semantic grouping (Signals/Credits/Warnings), and delete the orphaned `RewardComponents` widget.

**Tech Stack:** Textual TUI, Rich text formatting, dataclasses

---

## Background

Scaffold hindsight credit telemetry is now emitted via `AnalyticsSnapshotPayload` with three fields:
- `hindsight_credit: float` - Credit amount applied (post-cap)
- `scaffold_count: int` - Number of scaffolds that contributed
- `avg_scaffold_delay: float` - Average epochs since scaffolding interactions

The UX specialist recommended a multi-row semantic layout:
```
Reward Total     │ +0.150
  Signals        │ ΔAcc: +0.050  Rent: -0.020  Shock: -0.010
  Credits        │ Attr: +0.120  Hind: +0.080 (3x, 12.5e)  Foss: +0.500
  Warnings       │ Blend: -0.010
```

---

## Task 1: Add hindsight credit fields to RewardComponents schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:539-584`

**Step 1: Add fields to RewardComponents dataclass**

Add after line 578 (`holding_warning: float = 0.0`):

```python
    # Hindsight credit (scaffold contribution bonus) - Phase 3.2
    hindsight_credit: float = 0.0
    scaffold_count: int = 0  # Number of scaffolds that contributed
    avg_scaffold_delay: float = 0.0  # Average epochs since scaffolding
```

**Step 2: Update docstring**

Add to the docstring (around line 555):
```
    - hindsight_credit: Retroactive credit when beneficiary fossilizes (blue bonus)
    - scaffold_count: Number of scaffolds that contributed (debugging)
    - avg_scaffold_delay: Average epochs since scaffolding interactions (debugging)
```

**Step 3: Verify no tests break**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v -x`
Expected: All tests pass (schema is additive)

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/schema.py
git commit -m "feat(sanctum): add hindsight credit fields to RewardComponents schema"
```

---

## Task 2: Wire aggregator to populate hindsight credit from telemetry

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:1102-1122`

**Step 1: Add hindsight credit population**

After line 1116 (`env.reward_components.alpha_shock = payload.alpha_shock`), add:

```python
            # Hindsight credit (Phase 3.2 scaffold credit)
            if payload.hindsight_credit is not None:
                env.reward_components.hindsight_credit = payload.hindsight_credit
            if payload.scaffold_count is not None:
                env.reward_components.scaffold_count = payload.scaffold_count
            if payload.avg_scaffold_delay is not None:
                env.reward_components.avg_scaffold_delay = payload.avg_scaffold_delay
```

**Step 2: Verify no tests break**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -v -x`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): wire hindsight credit from telemetry to RewardComponents"
```

---

## Task 3: Refactor EnvDetailScreen reward display to multi-row layout

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_detail_screen.py:490-505`

**Step 1: Replace inline reward rendering with semantic grouping**

Replace lines 490-505 with:

```python
        # Reward breakdown (semantic grouping per UX review)
        rc = env.reward_components
        if rc.total != 0:
            # Total (standalone row for emphasis)
            total_style = "bold green" if rc.total >= 0 else "bold red"
            table.add_row("Reward Total", Text(f"{rc.total:+.3f}", style=total_style))

            # Step-based signals
            signals = Text()
            if rc.base_acc_delta != 0:
                style = "green" if rc.base_acc_delta > 0 else "red"
                signals.append(f"ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
            if rc.compute_rent != 0:
                signals.append(f"  Rent: {rc.compute_rent:.3f}", style="red")
            if rc.alpha_shock != 0:
                signals.append(f"  Shock: {rc.alpha_shock:.3f}", style="red")
            if rc.ratio_penalty != 0:
                signals.append(f"  Ratio: {rc.ratio_penalty:.3f}", style="red")
            if signals.plain:
                table.add_row("  Signals", signals)

            # Event-based credits/bonuses
            credits = Text()
            if rc.bounded_attribution != 0:
                style = "green" if rc.bounded_attribution > 0 else "red"
                credits.append(f"Attr: {rc.bounded_attribution:+.3f}", style=style)
            if rc.hindsight_credit != 0:
                hind_str = f"Hind: {rc.hindsight_credit:+.3f}"
                # Append scaffold context only when credit is active
                if rc.scaffold_count > 0:
                    hind_str += f" ({rc.scaffold_count}x, {rc.avg_scaffold_delay:.1f}e)"
                credits.append(f"  {hind_str}", style="blue")
            if rc.stage_bonus != 0:
                credits.append(f"  Stage: {rc.stage_bonus:+.3f}", style="blue")
            if rc.fossilize_terminal_bonus != 0:
                credits.append(f"  Foss: {rc.fossilize_terminal_bonus:+.3f}", style="blue")
            if credits.plain:
                table.add_row("  Credits", credits)

            # Warnings (if any active)
            warnings = Text()
            if rc.blending_warning < 0:
                warnings.append(f"Blend: {rc.blending_warning:.3f}", style="yellow")
            if rc.holding_warning < 0:
                warnings.append(f"  Hold: {rc.holding_warning:.3f}", style="yellow")
            if warnings.plain:
                table.add_row("  Warnings", warnings)
```

**Step 2: Verify widget renders correctly**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_detail_screen.py -v -x`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_detail_screen.py
git commit -m "refactor(sanctum): multi-row semantic reward breakdown in EnvDetailScreen

Per UX review:
- Signals: ΔAcc, Rent, Shock, Ratio (step-based costs)
- Credits: Attr, Hind, Stage, Foss (event-based bonuses)
- Warnings: Blend, Hold (yellow alerts)

Hindsight credit shows scaffold context (3x, 12.5e) when active."
```

---

## Task 4: Delete orphaned RewardComponents widget

**Files:**
- Delete: `src/esper/karn/sanctum/widgets/reward_components.py`
- Modify: `src/esper/karn/sanctum/widgets/__init__.py`

**Step 1: Remove from __init__.py exports**

In `src/esper/karn/sanctum/widgets/__init__.py`, remove:
- Import line: `from .reward_components import RewardComponents`
- Export in `__all__`: `"RewardComponents",`

**Step 2: Delete the widget file**

```bash
rm src/esper/karn/sanctum/widgets/reward_components.py
```

**Step 3: Check for any remaining references**

Run: `grep -r "RewardComponents" src/esper/karn/sanctum/`
Expected: Only `schema.py` references (the dataclass, not the widget)

**Step 4: Run tests to verify no breakage**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v -x`
Expected: All tests pass (widget was never used)

**Step 5: Commit**

```bash
git add -A
git commit -m "chore(sanctum): remove orphaned RewardComponents widget

Widget was defined but never mounted in app layout.
EnvDetailScreen renders reward breakdown inline instead.
Per No Legacy Code Policy."
```

---

## Task 5: Add test for hindsight credit display

**Files:**
- Modify: `tests/karn/sanctum/test_env_detail_screen.py`

**Step 1: Add test for hindsight credit rendering**

```python
def test_reward_breakdown_shows_hindsight_credit():
    """Hindsight credit displays with scaffold context when active."""
    from esper.karn.sanctum.schema import EnvSnapshot, RewardComponents

    env = EnvSnapshot(
        env_id=0,
        reward_components=RewardComponents(
            total=0.25,
            bounded_attribution=0.10,
            hindsight_credit=0.08,
            scaffold_count=3,
            avg_scaffold_delay=12.5,
        ),
    )

    # Create screen and render metrics
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()

    # Convert to string and check for hindsight credit
    rendered = str(table)
    assert "Hind:" in rendered
    assert "0.08" in rendered or "+0.080" in rendered
    assert "(3x, 12.5e)" in rendered
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_detail_screen.py::test_reward_breakdown_shows_hindsight_credit -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/karn/sanctum/test_env_detail_screen.py
git commit -m "test(sanctum): add hindsight credit display test"
```

---

## Task 6: Final verification

**Step 1: Run full Sanctum test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
Expected: All tests pass

**Step 2: Manual smoke test (optional)**

If training is running, open Sanctum and press `d` on an environment to verify the new layout renders correctly.

**Step 3: Final commit if any cleanup needed**

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add hindsight fields to schema | schema.py |
| 2 | Wire aggregator | aggregator.py |
| 3 | Refactor to multi-row layout | env_detail_screen.py |
| 4 | Delete orphaned widget | reward_components.py, __init__.py |
| 5 | Add display test | test_env_detail_screen.py |
| 6 | Final verification | - |

**Total: ~80 lines changed, 150 lines deleted (orphaned widget)**
