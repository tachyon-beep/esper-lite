# Per-Head NaN/Inf Indicator Lights Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NaN and Inf indicator rows to the Action Heads panel that latch red when detected and stay red until the run ends.

**Architecture:** Detect NaN/Inf per action head in the PPO training loop (already partially exists), emit per-head flags in telemetry, OR-latch in aggregator (once set, stays set for entire run), display as indicator lights (○ clear, ● red when latched).

**Tech Stack:** PyTorch (NaN detection), Python dataclasses (telemetry/schema), Textual/Rich (TUI display)

**Reviewed by:** PyTorch specialist (performance), Code review agent (CLAUDE.md compliance)

---

## Background

The PPO training loop already checks for non-finite values per head (ppo.py:590-593):
```python
for key in HEAD_NAMES:
    if not torch.isfinite(log_probs[key]).all():
        nonfinite_count = (~torch.isfinite(log_probs[key])).sum().item()
        nonfinite_sources.append(f"log_probs[{key}]: {nonfinite_count} non-finite")
```

We need to:
1. Separate NaN from Inf detection per head
2. Emit per-head flags in `PPOUpdatePayload`
3. OR-latch in aggregator (persists for entire run)
4. Display as two rows (NaN, Inf) under the State row

## HEAD_NAMES Reference

From `leyline/factored_actions.py`:
```python
ACTION_HEAD_NAMES = ("slot", "blueprint", "style", "tempo", "alpha_target", "alpha_speed", "alpha_curve", "op")
```

**IMPORTANT:** Keys are lowercase WITHOUT `head_` prefix: `"op"`, `"slot"`, `"blueprint"`, etc.

Display abbreviations: Op, Slot, Blueprint, Style, Tempo, αTarget, αSpeed, Curve

## Performance Considerations (PyTorch Specialist Review)

**Issue:** Naive approach would add 16 GPU syncs per epoch (2 per head × 8 heads).

**Solution:** Keep `isfinite` fast-path, only drill down to NaN/Inf when non-finite found:
```python
# Fast path: single isfinite check per head (most common case: no issues)
for key in HEAD_NAMES:
    lp = log_probs[key]
    if not torch.isfinite(lp).all():
        # Slow path: distinguish NaN from Inf (only when needed)
        if torch.isnan(lp).any():
            head_nan_detected[key] = True
        if torch.isinf(lp).any():
            head_inf_detected[key] = True
        nonfinite_found = True
```

This preserves 0 syncs in the happy path (no NaN/Inf), only adding syncs when issues are actually found.

## CLAUDE.md Compliance Notes

**No defensive `.get()` usage:** Latch dicts are pre-populated with all HEAD_NAMES keys set to False. Display code accesses keys directly without `.get()` fallbacks - if a key is missing, that's a bug to be caught, not hidden.

**No duplicate code:** The display name → leyline key mapping is extracted to a module-level constant `DISPLAY_TO_LEYLINE_KEY`.

---

### Task 1: Add per-head NaN/Inf flags to PPOUpdatePayload

**Files:**
- Modify: `src/esper/leyline/telemetry.py:613-680` (PPOUpdatePayload class)
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write the failing test**

```python
def test_ppo_update_payload_has_per_head_nan_inf_flags():
    """PPOUpdatePayload should have per-head NaN/Inf flag dicts."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.0,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.1,
        nan_grad_count=0,
        pre_clip_grad_norm=0.5,
        head_nan_detected={"op": True, "slot": False},
        head_inf_detected={"op": False, "slot": True},
    )

    assert payload.head_nan_detected == {"op": True, "slot": False}
    assert payload.head_inf_detected == {"op": False, "slot": True}
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_per_head_nan_inf_flags -v`
Expected: FAIL with "unexpected keyword argument 'head_nan_detected'"

**Step 3: Add fields to PPOUpdatePayload**

In `src/esper/leyline/telemetry.py`, add after line ~665 (after `entropy_collapsed`):

```python
    # Per-head NaN/Inf detection (for indicator lights with latch behavior)
    # Keys are HEAD_NAMES: op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve
    head_nan_detected: dict[str, bool] | None = None
    head_inf_detected: dict[str, bool] | None = None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_per_head_nan_inf_flags -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add per-head NaN/Inf flags to PPOUpdatePayload"
```

---

### Task 2: Emit per-head NaN/Inf flags from PPO training

**Files:**
- Modify: `src/esper/simic/agent/ppo.py:586-600` (non-finite detection loop)
- Modify: `src/esper/simic/agent/ppo.py:950-960` (aggregated_result)
- Modify: `src/esper/simic/telemetry/emitters.py:830-860` (emit_ppo_update_event)
- Test: `tests/simic/agent/test_ppo_nan_detection.py` (new file)

**Step 1: Write the failing tests**

Create `tests/simic/agent/test_ppo_nan_detection.py`:

```python
"""Tests for per-head NaN/Inf detection in PPO training."""

import pytest
import torch
from esper.leyline import HEAD_NAMES


def test_detect_nan_per_head():
    """Verify NaN detection separates NaN from Inf per head."""
    # Simulate log_probs with NaN in one head, Inf in another
    log_probs = {head: torch.zeros(10) for head in HEAD_NAMES}
    log_probs["op"][0] = float("nan")
    log_probs["slot"][0] = float("inf")

    head_nan_detected = {head: False for head in HEAD_NAMES}
    head_inf_detected = {head: False for head in HEAD_NAMES}

    # Fast-path pattern: only drill down when isfinite fails
    for head in HEAD_NAMES:
        lp = log_probs[head]
        if not torch.isfinite(lp).all():
            if torch.isnan(lp).any():
                head_nan_detected[head] = True
            if torch.isinf(lp).any():
                head_inf_detected[head] = True

    assert head_nan_detected["op"] is True
    assert head_nan_detected["slot"] is False
    assert head_inf_detected["op"] is False
    assert head_inf_detected["slot"] is True


def test_detect_nan_and_inf_same_tensor():
    """A tensor can have both NaN and Inf - both flags should be set."""
    lp = torch.tensor([float("nan"), float("inf"), 1.0])

    has_nan = bool(torch.isnan(lp).any().item())
    has_inf = bool(torch.isinf(lp).any().item())

    assert has_nan is True
    assert has_inf is True


def test_detect_negative_inf():
    """Negative infinity is common for log_probs of impossible actions."""
    lp = torch.tensor([float("-inf"), 1.0, 2.0])

    # torch.isinf detects both +inf and -inf
    assert bool(torch.isinf(lp).any().item()) is True
    assert bool(torch.isnan(lp).any().item()) is False


def test_empty_tensor_returns_false():
    """Empty tensors should not trigger NaN/Inf detection."""
    lp = torch.zeros(0)

    assert bool(torch.isnan(lp).any().item()) is False
    assert bool(torch.isinf(lp).any().item()) is False


def test_clean_tensor_fast_path():
    """Clean tensors should pass isfinite check without drilling down."""
    lp = torch.tensor([0.1, 0.2, 0.3])

    # Fast path: isfinite.all() returns True, no need to check nan/inf
    assert torch.isfinite(lp).all().item() is True
```

**Step 2: Run tests to verify they pass** (these validate our detection approach)

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo_nan_detection.py -v`
Expected: All PASS

**Step 3: Modify PPO training loop to track per-head NaN/Inf**

In `src/esper/simic/agent/ppo.py`, after line ~523 (head_ratio_max initialization), add:

```python
        # Per-head NaN/Inf tracking (for indicator lights)
        # OR across all epochs - once detected, stays detected for this update
        head_nan_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}
        head_inf_detected: dict[str, bool] = {head: False for head in HEAD_NAMES}
```

In the epoch loop (~line 590-593), replace the existing non-finite check with:

```python
            # Check new log_probs - separate NaN from Inf per head
            # Fast path: only drill down when isfinite fails (preserves 0 syncs in happy path)
            for key in HEAD_NAMES:
                lp = log_probs[key]
                if not torch.isfinite(lp).all():
                    # Slow path: distinguish NaN from Inf
                    if torch.isnan(lp).any():
                        head_nan_detected[key] = True
                        nonfinite_sources.append(f"log_probs[{key}]: NaN detected")
                    if torch.isinf(lp).any():
                        head_inf_detected[key] = True
                        nonfinite_sources.append(f"log_probs[{key}]: Inf detected")
                    nonfinite_found = True
```

In aggregated_result (~line 957), add:

```python
        # Add per-head NaN/Inf flags (for indicator lights)
        aggregated_result["head_nan_detected"] = head_nan_detected
        aggregated_result["head_inf_detected"] = head_inf_detected
```

**Step 4: Modify emitters.py to include flags in payload**

In `src/esper/simic/telemetry/emitters.py`, in `emit_ppo_update_event()` payload construction (~line 860), add:

```python
            # Per-head NaN/Inf flags (for indicator lights)
            head_nan_detected=metrics.get("head_nan_detected"),
            head_inf_detected=metrics.get("head_inf_detected"),
```

Note: Using `.get()` here is legitimate - older training runs may not have these fields, and `None` (the payload default) correctly signals "no data" vs "all False".

**Step 5: Run full PPO tests to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/ -v --tb=short -x`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/agent/ppo.py src/esper/simic/telemetry/emitters.py tests/simic/agent/test_ppo_nan_detection.py
git commit -m "feat(simic): emit per-head NaN/Inf flags from PPO training"
```

---

### Task 3: Add per-head NaN/Inf latch to TamiyoState schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:826-832` (TamiyoState gradient health section)
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
def test_tamiyo_state_has_nan_inf_latch_fields():
    """TamiyoState should have per-head NaN/Inf latch dicts pre-populated."""
    from esper.leyline import HEAD_NAMES
    from esper.karn.sanctum.schema import TamiyoState

    state = TamiyoState()

    # Should have latch dicts pre-populated with all heads set to False
    assert hasattr(state, "head_nan_latch")
    assert hasattr(state, "head_inf_latch")

    # All HEAD_NAMES keys should exist (no .get() needed in display code)
    for head in HEAD_NAMES:
        assert head in state.head_nan_latch
        assert head in state.head_inf_latch
        assert state.head_nan_latch[head] is False
        assert state.head_inf_latch[head] is False
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_nan_inf_latch_fields -v`
Expected: FAIL with "TamiyoState has no attribute 'head_nan_latch'"

**Step 3: Add latch fields to TamiyoState**

In `src/esper/karn/sanctum/schema.py`, first add import at top if not present:

```python
from esper.leyline import HEAD_NAMES
```

Then after line ~830 (inf_grad_count), add:

```python
    # Per-head NaN/Inf latch (indicator lights - once True, stays True for entire run)
    # Pre-populated with all HEAD_NAMES keys to enable direct access without .get()
    # Keys: op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve
    head_nan_latch: dict[str, bool] = field(
        default_factory=lambda: {head: False for head in HEAD_NAMES}
    )
    head_inf_latch: dict[str, bool] = field(
        default_factory=lambda: {head: False for head in HEAD_NAMES}
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_nan_inf_latch_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(schema): add pre-populated per-head NaN/Inf latch fields to TamiyoState"
```

---

### Task 4: Implement OR-latch in aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:858-864` (PPO update handling)
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing tests**

```python
def test_aggregator_latches_per_head_nan_inf():
    """Aggregator should OR-latch per-head NaN/Inf flags (once set, stays set)."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import PPOUpdatePayload, TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator()

    # First update: NaN in op head
    event1 = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1, value_loss=0.2, entropy=1.0, grad_norm=0.5,
            kl_divergence=0.01, clip_fraction=0.1, nan_grad_count=1,
            pre_clip_grad_norm=0.5,
            head_nan_detected={"op": True, "slot": False},
            head_inf_detected={},
        ),
    )
    agg.process_event(event1)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.head_nan_latch["op"] is True
    assert snapshot.tamiyo.head_nan_latch["slot"] is False

    # Second update: No NaN, but latch should persist
    event2 = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1, value_loss=0.2, entropy=1.0, grad_norm=0.5,
            kl_divergence=0.01, clip_fraction=0.1, nan_grad_count=0,
            pre_clip_grad_norm=0.5,
            head_nan_detected={"op": False, "slot": False},
            head_inf_detected={},
        ),
    )
    agg.process_event(event2)

    snapshot = agg.get_snapshot()
    # op should STILL be latched (OR-latch behavior)
    assert snapshot.tamiyo.head_nan_latch["op"] is True


def test_aggregator_latches_both_nan_and_inf_same_head():
    """A head can have both NaN and Inf detected - both latches should set."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import PPOUpdatePayload, TelemetryEvent, TelemetryEventType

    agg = SanctumAggregator()

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1, value_loss=0.2, entropy=1.0, grad_norm=0.5,
            kl_divergence=0.01, clip_fraction=0.1, nan_grad_count=1,
            pre_clip_grad_norm=0.5,
            head_nan_detected={"op": True},
            head_inf_detected={"op": True},  # Same head has both!
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.head_nan_latch["op"] is True
    assert snapshot.tamiyo.head_inf_latch["op"] is True
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_latches_per_head_nan_inf -v`
Expected: FAIL (latch not implemented)

**Step 3: Implement OR-latch in aggregator**

In `src/esper/karn/sanctum/aggregator.py`, in `_handle_ppo_update()` after line ~862, add:

```python
        # Per-head NaN/Inf OR-latch (once True, stays True for entire run)
        if payload.head_nan_detected:
            for head, detected in payload.head_nan_detected.items():
                if detected:
                    self._tamiyo.head_nan_latch[head] = True
        if payload.head_inf_detected:
            for head, detected in payload.head_inf_detected.items():
                if detected:
                    self._tamiyo.head_inf_latch[head] = True
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_latches_per_head_nan_inf tests/karn/sanctum/test_aggregator.py::test_aggregator_latches_both_nan_and_inf_same_head -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(aggregator): OR-latch per-head NaN/Inf flags"
```

---

### Task 5: Add NaN/Inf indicator rows to Action Heads panel

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py:380-450` (heads section rendering)
- Test: `tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py`

**Step 1: Write the failing tests**

```python
def test_heads_panel_shows_nan_inf_indicator_rows():
    """HeadsPanel should show NaN and Inf indicator rows below State row."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import HeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    # Latch NaN on op head, Inf on slot head
    snapshot.tamiyo.head_nan_latch["op"] = True
    snapshot.tamiyo.head_inf_latch["slot"] = True

    panel = HeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should have NaN and Inf row labels
    assert "NaN" in content_str
    assert "Inf" in content_str
    # Should show filled circle (●) for latched heads
    assert "●" in content_str


def test_heads_panel_shows_all_clear_when_no_nan_inf():
    """HeadsPanel should show all empty indicators when no NaN/Inf latched."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from esper.karn.sanctum.widgets.tamiyo_brain.action_heads_panel import HeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    # All latches are pre-populated with False (default)

    panel = HeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should have NaN and Inf row labels
    assert "NaN" in content_str
    assert "Inf" in content_str
    # Should show only empty circles (no filled circles)
    assert "○" in content_str
    # Count filled circles in NaN/Inf rows only (not State row)
    # The NaN and Inf rows should have zero filled circles
    lines = content_str.split("\n")
    nan_line = next((l for l in lines if l.strip().startswith("NaN")), "")
    inf_line = next((l for l in lines if l.strip().startswith("Inf")), "")
    assert "●" not in nan_line
    assert "●" not in inf_line
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py::test_heads_panel_shows_nan_inf_indicator_rows -v`
Expected: FAIL (NaN/Inf rows not implemented)

**Step 3: Add display-to-leyline key mapping constant**

In `src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py`, add near the top (after HEAD_CONFIG):

```python
# Map display names (from HEAD_CONFIG) to leyline HEAD_NAMES keys
# Used for NaN/Inf indicator rows to look up latch status
DISPLAY_TO_LEYLINE_KEY: dict[str, str] = {
    "Op": "op",
    "Slot": "slot",
    "Blueprint": "blueprint",
    "Style": "style",
    "Tempo": "tempo",
    "αTarget": "alpha_target",
    "αSpeed": "alpha_speed",
    "Curve": "alpha_curve",
}
```

**Step 4: Implement NaN/Inf indicator rows**

In `src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py`, in `_render_heads_section()`, after the State row rendering (~line 440), add:

```python
        # NaN indicator row
        result.append("NaN  ", style="dim")
        for head_key, _, _, width, _ in HEAD_CONFIG:
            leyline_key = DISPLAY_TO_LEYLINE_KEY[head_key]
            latched = self._snapshot.tamiyo.head_nan_latch[leyline_key]
            indicator = "●" if latched else "○"
            style = "red bold" if latched else "dim"
            gutter = self._column_gutter(head_key)
            result.append(f"{indicator:>{width}}", style=style)
            result.append(" " * gutter)
        result.append("\n")

        # Inf indicator row
        result.append("Inf  ", style="dim")
        for head_key, _, _, width, _ in HEAD_CONFIG:
            leyline_key = DISPLAY_TO_LEYLINE_KEY[head_key]
            latched = self._snapshot.tamiyo.head_inf_latch[leyline_key]
            indicator = "●" if latched else "○"
            style = "red bold" if latched else "dim"
            gutter = self._column_gutter(head_key)
            result.append(f"{indicator:>{width}}", style=style)
            result.append(" " * gutter)
        result.append("\n")
```

**Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py::test_heads_panel_shows_nan_inf_indicator_rows tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py::test_heads_panel_shows_all_clear_when_no_nan_inf -v`
Expected: PASS

**Step 6: Run all Action Heads panel tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py tests/karn/sanctum/widgets/tamiyo_brain/test_action_heads_panel.py
git commit -m "feat(sanctum): add NaN/Inf indicator rows to Action Heads panel"
```

---

### Task 6: Visual verification and cleanup

**Files:**
- None (manual testing)

**Step 1: Run Sanctum TUI manually**

```bash
PYTHONPATH=src uv run python -m esper.karn.sanctum.app --demo
```

**Step 2: Verify layout**

Expected display (under State row):
```
             Op       Slot     Blueprint  Style    Tempo    αTarget  αSpeed   Curve
...
State          ○          ○           ○        ○        ○         ○        ○       ○
NaN            ○          ○           ○        ○        ○         ○        ○       ○
Inf            ○          ○           ○        ○        ○         ○        ○       ○
─────────────────────────────────────────────────────────────────────────────────────
```

When NaN/Inf detected (simulated), indicator should be bright red ●.

**Step 3: Run full test suite**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v --tb=short
```
Expected: All PASS

**Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: polish NaN/Inf indicator implementation"
```

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Add fields to PPOUpdatePayload | leyline/telemetry.py |
| 2 | Emit per-head flags from PPO | simic/agent/ppo.py, simic/telemetry/emitters.py |
| 3 | Add pre-populated latch fields to TamiyoState | karn/sanctum/schema.py |
| 4 | Implement OR-latch in aggregator | karn/sanctum/aggregator.py |
| 5 | Add indicator rows to display | karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py |
| 6 | Visual verification | Manual testing |

## CLAUDE.md Compliance Checklist

- [x] No defensive `.get()` in display code (latch dicts pre-populated)
- [x] No duplicate code (key mapping extracted to `DISPLAY_TO_LEYLINE_KEY`)
- [x] Tests use public API (`process_event()` not `_handle_ppo_update()`)
- [x] All edge cases tested (all-clear, both-latched, persistence)

## Expected Final Display

```
             Op       Slot     Blueprint  Style    Tempo    αTarget  αSpeed   Curve
Entr        0.893     1.000       1.000   0.215    1.000     0.571    1.000   1.000
            ▓▓▓░░     █████       █████   ▓░░░░    █████     ▓▓░░░    █████   █████
Grad        0.131→    0.135↗      0.211→  0.275↘   0.192→    0.101↗   0.213→  0.124→
            ▓░░░░     ▓░░░░       ▓░░░░   ▓░░░░    ▓░░░░     ▓░░░░    ▓░░░░   ▓░░░░
Ratio       1.000     1.000       1.000   1.000    1.000     1.000    1.000   1.000
            █████     █████       █████   █████    █████     █████    █████   █████
State         ●         ●           ●       ○        ●         ●        ●       ●
NaN           ○         ○           ○       ●        ○         ○        ○       ○
Inf           ○         ○           ○       ○        ○         ○        ○       ○
```

- `○` = Clear (dim gray) - no NaN/Inf ever detected for this head
- `●` = Latched (bright red) - NaN/Inf was detected at some point during this run
