# Alpha Curve UX Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Flow the alpha_curve field from the training layer through telemetry to display in Sanctum TUI and Overwatch web dashboard.

**Architecture:** Add `alpha_curve: str` field to telemetry payloads, propagate through aggregator to SeedState schema, then render as curve glyphs (╱∿⌒∫⊐) in Sanctum widgets and Overwatch Vue components. Display is stage-aware: show glyph only during BLENDING (when curve is causally active).

**Tech Stack:** Python dataclasses (telemetry), Textual/Rich (Sanctum TUI), TypeScript/Vue (Overwatch web)

**Prerequisites:** The sigmoid steepness feature has been implemented - AlphaController now has `alpha_steepness`, SeedSlot passes curve/steepness to retarget(), and AlphaCurveAction has `to_curve()` and `to_steepness()` methods.

---

## Design Rationale: Two-Layer Masking

The policy uses **two-layer masking** for head relevance:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Action Masking** | `src/esper/tamiyo/policy/action_masks.py` | Prevent physically impossible actions (logits → -∞) |
| **Advantage Masking** | `src/esper/simic/agent/advantages.py` | Zero gradient for causally irrelevant heads |

**For `alpha_curve`:**
- Action mask: Always all-ones (all curves always valid)
- Advantage mask: Zero when `op ∉ {SET_ALPHA_TARGET, PRUNE}`

This means the policy **always samples a curve**, but only receives learning signal when the curve causally affects the outcome. The sampled-but-unused curve represents "what the policy would have chosen" - useful for diagnostics but not affecting training.

**UX implication:** The curve value is always present in telemetry, but we only display it when the stage is BLENDING (when it's causally active). For other stages, the curve is sampled but not used.

---

## Curve Glyph Reference

| AlphaCurveAction | Glyph | Description |
|------------------|-------|-------------|
| LINEAR | `╱` | Constant rate transition |
| COSINE | `∿` | Smooth ease-in/ease-out |
| SIGMOID_GENTLE | `⌒` | Gradual S-curve (k=6) |
| SIGMOID | `∫` | Standard S-curve (k=12) |
| SIGMOID_SHARP | `⊐` | Steep near-step (k=24) |

---

## Task 1: Add alpha_curve to Telemetry Payloads

> **REVIEW COMPLETE:** DRL and PyTorch specialists approved. The curve is always sampled (advantage masking handles causal attribution), so we always emit it.

**Files:**
- Modify: `src/esper/leyline/telemetry.py` - SeedTelemetry (class at ~line 174, add field after `blending_velocity` at line 223)
- Modify: `src/esper/leyline/telemetry.py` - SeedGerminatedPayload (class at ~line 725, add field after `blend_tempo_epochs` at line 746)
- Modify: `src/esper/leyline/telemetry.py` - SeedStageChangedPayload (class at ~line 765, add field after `has_exploding` at line 786)
- Test: `tests/leyline/test_telemetry_alpha_curve.py`

**Step 1: Write the failing test**

Create `tests/leyline/test_telemetry_alpha_curve.py`:

```python
"""Tests for alpha_curve field in telemetry payloads."""

import pytest
from esper.leyline.telemetry import (
    SeedTelemetry,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
)


class TestSeedTelemetryAlphaCurve:
    """Test alpha_curve field in SeedTelemetry."""

    def test_default_alpha_curve_is_linear(self):
        """SeedTelemetry should default to LINEAR curve."""
        telemetry = SeedTelemetry(seed_id="test", blueprint_id="conv_l", layer_id="l0")
        assert telemetry.alpha_curve == "LINEAR"

    def test_to_dict_includes_alpha_curve(self):
        """to_dict() should include alpha_curve field."""
        telemetry = SeedTelemetry(
            seed_id="test",
            blueprint_id="conv_l",
            layer_id="l0",
            alpha_curve="SIGMOID",
        )
        data = telemetry.to_dict()
        assert "alpha_curve" in data
        assert data["alpha_curve"] == "SIGMOID"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        original = SeedTelemetry(
            seed_id="test",
            blueprint_id="conv_l",
            layer_id="l0",
            alpha_curve="SIGMOID_SHARP",
        )
        data = original.to_dict()
        restored = SeedTelemetry.from_dict(data)
        assert restored.alpha_curve == "SIGMOID_SHARP"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "seed_id": "test",
            "blueprint_id": "conv_l",
            "layer_id": "l0",
            "gradient_norm": 1.0,
            "gradient_health": 1.0,
            "has_vanishing": False,
            "has_exploding": False,
            "accuracy": 0.0,
            "accuracy_delta": 0.0,
            "epochs_in_stage": 0,
            "stage": 1,
            "alpha": 0.0,
            "alpha_target": 0.0,
            "alpha_mode": 0,
            "alpha_steps_total": 0,
            "alpha_steps_done": 0,
            "time_to_target": 0,
            "alpha_velocity": 0.0,
            "alpha_algorithm": 0,
            "epoch": 0,
            "max_epochs": 25,
            "blend_tempo_epochs": 5,
            "blending_velocity": 0.0,
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedTelemetry.from_dict(incomplete_data)


class TestSeedGerminatedPayloadAlphaCurve:
    """Test alpha_curve field in SeedGerminatedPayload."""

    def test_default_alpha_curve_is_linear(self):
        """SeedGerminatedPayload should default to LINEAR curve."""
        payload = SeedGerminatedPayload(
            slot_id="slot_0",
            env_id=0,
            blueprint_id="conv_l",
            params=1000,
        )
        assert payload.alpha_curve == "LINEAR"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "blueprint_id": "conv_l",
            "params": 1000,
            "alpha_curve": "COSINE",
        }
        payload = SeedGerminatedPayload.from_dict(data)
        assert payload.alpha_curve == "COSINE"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "blueprint_id": "conv_l",
            "params": 1000,
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedGerminatedPayload.from_dict(incomplete_data)


class TestSeedStageChangedPayloadAlphaCurve:
    """Test alpha_curve field in SeedStageChangedPayload.

    Note: alpha_curve is always present (not None) because the policy always
    samples a curve. However, the curve only causally affects outcomes during
    BLENDING (SET_ALPHA_TARGET/PRUNE operations). The advantage masking in
    simic/agent/advantages.py handles causal attribution - see design rationale in plan.
    """

    def test_default_alpha_curve_is_linear(self):
        """SeedStageChangedPayload should default to LINEAR curve."""
        payload = SeedStageChangedPayload(
            slot_id="slot_0",
            env_id=0,
            from_stage="TRAINING",
            to_stage="BLENDING",
        )
        assert payload.alpha_curve == "LINEAR"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "from": "TRAINING",
            "to": "BLENDING",
            "alpha_curve": "SIGMOID_GENTLE",
        }
        payload = SeedStageChangedPayload.from_dict(data)
        assert payload.alpha_curve == "SIGMOID_GENTLE"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "from": "TRAINING",
            "to": "BLENDING",
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedStageChangedPayload.from_dict(incomplete_data)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry_alpha_curve.py -v`
Expected: FAIL with "no attribute 'alpha_curve'"

**Step 3: Add alpha_curve to SeedTelemetry**

In `src/esper/leyline/telemetry.py`, after line 223 (`blending_velocity: float = 0.0`), add:

```python
    # Alpha curve shape (from AlphaCurveAction enum name).
    # Always present because policy always samples a curve; causal relevance
    # is handled by advantage masking in simic/agent/advantages.py.
    alpha_curve: str = "LINEAR"
```

**Step 4: Update SeedTelemetry.to_dict()**

In `src/esper/leyline/telemetry.py`, in the `to_dict()` method (around line 316), add after `"blending_velocity"`:

```python
            "alpha_curve": self.alpha_curve,
```

**Step 5: Update SeedTelemetry.from_dict()**

In `src/esper/leyline/telemetry.py`, in the `from_dict()` method, add to the constructor call:

```python
            alpha_curve=data["alpha_curve"],
```

**Step 6: Add alpha_curve to SeedGerminatedPayload**

In `src/esper/leyline/telemetry.py`, after line 746 (`blend_tempo_epochs: int = 5`), add:

```python
    alpha_curve: str = "LINEAR"
```

**Step 7: Update SeedGerminatedPayload.from_dict()**

In the `from_dict()` method, add to the constructor call:

```python
            alpha_curve=data["alpha_curve"],
```

**Step 8: Add alpha_curve to SeedStageChangedPayload**

In `src/esper/leyline/telemetry.py`, after line 786 (`has_exploding: bool = False`), add:

```python
    # Alpha curve - always present (policy always samples), but only causally
    # relevant during BLENDING. See simic/agent/advantages.py for causal masking.
    alpha_curve: str = "LINEAR"
```

**Step 9: Update SeedStageChangedPayload.from_dict()**

In the `from_dict()` method, add to the constructor call:

```python
            alpha_curve=data["alpha_curve"],
```

**Step 10: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry_alpha_curve.py -v`
Expected: All tests PASS

**Step 11: Run existing telemetry tests**

Run: `PYTHONPATH=src uv run pytest tests/leyline/ -v -k "telemetry"`
Expected: All tests PASS (backward compatible)

**Step 12: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry_alpha_curve.py
git commit -m "feat(leyline): add alpha_curve field to telemetry payloads

- Add alpha_curve to SeedTelemetry, SeedGerminatedPayload, SeedStageChangedPayload
- Always present (policy always samples); causal relevance via advantage masking
- Strict deserialization: missing field raises KeyError (no silent defaults)"
```

---

## Task 2: Add alpha_curve to Sanctum Schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:127` (SeedState)
- Test: `tests/karn/sanctum/test_schema_alpha_curve.py`

**Step 1: Write the failing test**

Create `tests/karn/sanctum/test_schema_alpha_curve.py`:

```python
"""Tests for alpha_curve field in Sanctum schema."""

from esper.karn.sanctum.schema import SeedState


class TestSeedStateAlphaCurve:
    """Test alpha_curve field in SeedState."""

    def test_default_alpha_curve_is_linear(self):
        """SeedState should default to LINEAR curve."""
        seed = SeedState(slot_id="slot_0")
        assert seed.alpha_curve == "LINEAR"

    def test_alpha_curve_can_be_set(self):
        """SeedState should accept alpha_curve parameter."""
        seed = SeedState(slot_id="slot_0", alpha_curve="SIGMOID")
        assert seed.alpha_curve == "SIGMOID"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema_alpha_curve.py -v`
Expected: FAIL with "unexpected keyword argument 'alpha_curve'"

**Step 3: Add alpha_curve to SeedState**

In `src/esper/karn/sanctum/schema.py`, after line 127 (`blend_tempo_epochs: int = 5`), add:

```python
    # Alpha curve shape - always present, but only displayed during BLENDING
    # (when the curve is causally active). See design rationale in plan.
    alpha_curve: str = "LINEAR"
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema_alpha_curve.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema_alpha_curve.py
git commit -m "feat(sanctum): add alpha_curve field to SeedState schema"
```

---

## Task 3: Wire alpha_curve through Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py` - SEED_GERMINATED handler (starts ~line 741, add after `blend_tempo_epochs` assignment at line 767)
- Modify: `src/esper/karn/sanctum/aggregator.py` - SEED_STAGE_CHANGED handler (starts ~line 779, add after alpha block ~line 805)
- Test: `tests/karn/sanctum/test_aggregator_alpha_curve.py`

**Step 1: Write the failing test**

Create `tests/karn/sanctum/test_aggregator_alpha_curve.py`:

```python
"""Tests for alpha_curve propagation through aggregator."""

import pytest
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    TelemetryEvent,
)


class TestAggregatorAlphaCurve:
    """Test alpha_curve flows from payloads to SeedState."""

    def test_germinated_payload_sets_alpha_curve(self):
        """SEED_GERMINATED should copy alpha_curve to SeedState."""
        aggregator = SanctumAggregator()

        event = TelemetryEvent(
            event_type="SEED_GERMINATED",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="conv_l",
                params=1000,
                alpha_curve="SIGMOID",
            ),
        )
        aggregator.process_event(event)

        env = aggregator.get_env(0)
        assert env is not None
        seed = env.seeds.get("slot_0")
        assert seed is not None
        assert seed.alpha_curve == "SIGMOID"

    def test_stage_changed_payload_updates_alpha_curve(self):
        """SEED_STAGE_CHANGED should update alpha_curve."""
        aggregator = SanctumAggregator()

        # First germinate the seed
        germinate_event = TelemetryEvent(
            event_type="SEED_GERMINATED",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="conv_l",
                params=1000,
                alpha_curve="LINEAR",
            ),
        )
        aggregator.process_event(germinate_event)

        # Then change stage with new curve
        stage_event = TelemetryEvent(
            event_type="SEED_STAGE_CHANGED",
            data=SeedStageChangedPayload(
                slot_id="slot_0",
                env_id=0,
                from_stage="TRAINING",
                to_stage="BLENDING",
                alpha_curve="SIGMOID_SHARP",
            ),
        )
        aggregator.process_event(stage_event)

        env = aggregator.get_env(0)
        seed = env.seeds.get("slot_0")
        assert seed.alpha_curve == "SIGMOID_SHARP"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator_alpha_curve.py -v`
Expected: FAIL with assertion error (alpha_curve not being set)

**Step 3: Update SEED_GERMINATED handler**

In `src/esper/karn/sanctum/aggregator.py`, after line 767 (`seed.blend_tempo_epochs = germinated_payload.blend_tempo_epochs`), add:

```python
            seed.alpha_curve = germinated_payload.alpha_curve
```

**Step 4: Update SEED_STAGE_CHANGED handler**

In `src/esper/karn/sanctum/aggregator.py`, after the alpha update block (around line 805), add:

```python
            seed.alpha_curve = stage_changed_payload.alpha_curve
```

**Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator_alpha_curve.py -v`
Expected: All tests PASS

**Step 6: Run existing aggregator tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v -k "aggregator"`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator_alpha_curve.py
git commit -m "feat(sanctum): wire alpha_curve through aggregator

- Copy alpha_curve from SeedGerminatedPayload to SeedState
- Update alpha_curve from SeedStageChangedPayload"
```

---

## Task 4: Display Curve Glyph in SeedCard (BLENDING only)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_detail_screen.py:125-136` (blend tempo display)
- Test: Manual visual verification

**Step 1: Add curve glyph mapping**

In `src/esper/karn/sanctum/widgets/env_detail_screen.py`, after the imports (around line 24), add:

```python
# Curve glyph mapping for visual display.
# Only displayed during BLENDING when curve is causally active.
CURVE_GLYPHS = {
    "LINEAR": "╱",
    "COSINE": "∿",
    "SIGMOID_GENTLE": "⌒",
    "SIGMOID": "∫",
    "SIGMOID_SHARP": "⊐",
}
```

**Step 2: Update blend tempo display in SeedCard**

Replace lines 125-136 with:

```python
        # Blend info (shows tempo and curve ONLY during BLENDING when causally active)
        if seed.stage == "BLENDING" and seed.blend_tempo_epochs is not None:
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            curve_glyph = CURVE_GLYPHS.get(seed.alpha_curve, "")
            lines.append(Text(f"Blend: {tempo_arrows} {tempo_name} {curve_glyph} ({tempo} epochs)"))
        elif seed.stage == "FOSSILIZED" and seed.blend_tempo_epochs is not None:
            # Past tense for fossilized - no curve (historical detail)
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            lines.append(Text(f"Blended: {tempo_arrows} {tempo_name}", style="dim"))
        else:
            lines.append(Text("Blend: --", style="dim"))
```

**Step 3: Run existing widget tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_detail_screen.py
git commit -m "feat(sanctum): display curve glyph in SeedCard during BLENDING

- Add CURVE_GLYPHS mapping (╱∿⌒∫⊐)
- Show curve glyph only during BLENDING (when causally active)
- FOSSILIZED shows tempo only (curve is historical detail)"
```

---

## Task 5: Display Curve Glyph in EnvOverview Table (BLENDING only)

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_overview.py:610-616` (slot cell formatting)

**Step 1: Add curve glyph mapping**

In `src/esper/karn/sanctum/widgets/env_overview.py`, add after imports:

```python
# Curve glyph mapping for compact display (BLENDING only)
CURVE_GLYPHS_COMPACT = {
    "LINEAR": "╱",
    "COSINE": "∿",
    "SIGMOID_GENTLE": "⌒",
    "SIGMOID": "∫",
    "SIGMOID_SHARP": "⊐",
}
```

**Step 2: Update BLENDING slot cell format**

Replace lines 612-616 with:

```python
        if seed.stage == "BLENDING" and seed.alpha > 0:
            tempo = seed.blend_tempo_epochs
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            curve_glyph = CURVE_GLYPHS_COMPACT.get(seed.alpha_curve, "")
            base = f"[{style}]{stage_short}:{blueprint} {tempo_arrows}{curve_glyph} {seed.alpha:.1f}[/{style}]"
            return f"{base}{grad_indicator}" if grad_indicator else base
```

**Step 3: Run existing tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/test_env_overview.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/karn/sanctum/widgets/env_overview.py
git commit -m "feat(sanctum): display curve glyph in EnvOverview table

Format: Blend:conv_l ▸▸▸∫ 0.3 (tempo + curve + alpha)
Only shown during BLENDING when curve is causally active"
```

---

## Task 6: Add alpha_curve to Overwatch TypeScript Types

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/types/sanctum.ts:43` (SeedState interface)

**Step 1: Add alpha_curve to SeedState interface**

In `src/esper/karn/overwatch/web/src/types/sanctum.ts`, after line 43 (`blend_tempo_epochs: number;`), add:

```typescript
  alpha_curve: string;
```

**Step 2: Run TypeScript compilation**

Run: `cd src/esper/karn/overwatch/web && npm run build`
Expected: Compilation succeeds (may have warnings about unused field until component updated)

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/web/src/types/sanctum.ts
git commit -m "feat(overwatch): add alpha_curve to SeedState TypeScript interface"
```

---

## Task 7: Display Curve Glyph in SeedChip Component (BLENDING only)

**Files:**
- Modify: `src/esper/karn/overwatch/web/src/components/SeedChip.vue`
- Modify: `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:43` (default state)

**Step 1: Update SeedChip props and computed**

Replace the script section of `SeedChip.vue`:

```vue
<script setup lang="ts">
import { computed } from 'vue'
import type { SeedStage } from '../types/sanctum'

const props = defineProps<{
  slotId: string
  stage: SeedStage
  alpha?: number
  alphaCurve?: string
  hasWarning?: boolean
}>()

// Curve glyph mapping - only shown during BLENDING when causally active
const CURVE_GLYPHS: Record<string, string> = {
  'LINEAR': '╱',
  'COSINE': '∿',
  'SIGMOID_GENTLE': '⌒',
  'SIGMOID': '∫',
  'SIGMOID_SHARP': '⊐',
}

const abbreviatedSlotId = computed(() => {
  const match = props.slotId.match(/slot_(\d+)/)
  return match ? `S${match[1]}` : props.slotId
})

const stageClass = computed(() => `stage-${props.stage.toLowerCase()}`)

const showAlpha = computed(() => {
  return props.alpha !== undefined && (props.stage === 'BLENDING' || props.stage === 'HOLDING')
})

const alphaPercent = computed(() => {
  if (props.alpha === undefined) return ''
  return `${Math.round(props.alpha * 100)}%`
})

// Only show curve glyph during BLENDING (when causally active)
const curveGlyph = computed(() => {
  if (props.stage !== 'BLENDING' || !props.alphaCurve) return ''
  return CURVE_GLYPHS[props.alphaCurve] || ''
})

const tooltip = computed(() => `${props.slotId}: ${props.stage}`)
</script>
```

**Step 2: Update SeedChip template**

Replace the template section:

```vue
<template>
  <span
    class="seed-chip"
    :class="stageClass"
    :title="tooltip"
    data-testid="seed-chip"
  >
    <span class="slot-id" data-testid="slot-id">{{ abbreviatedSlotId }}</span>
    <span v-if="showAlpha" class="alpha" data-testid="alpha">{{ alphaPercent }}</span>
    <span v-if="curveGlyph" class="curve" data-testid="curve-glyph">{{ curveGlyph }}</span>
    <span v-if="hasWarning" class="warning" data-testid="warning-indicator">!</span>
  </span>
</template>
```

**Step 3: Add curve style**

In the style section, add:

```css
.curve {
  font-size: 10px;
  opacity: 0.8;
  margin-left: 2px;
}
```

**Step 4: Update SeedSwimlane default state**

In `SeedSwimlane.vue`, update the default SeedState (around line 43) to include:

```typescript
    alpha_curve: 'LINEAR'
```

**Step 5: Run Playwright tests**

Run: `cd src/esper/karn/overwatch/web && npx playwright test`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/overwatch/web/src/components/SeedChip.vue \
        src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue
git commit -m "feat(overwatch): display curve glyph in SeedChip during BLENDING

Format: S1 45% ∫ (slot + alpha + curve)
Only shown during BLENDING when curve is causally active"
```

---

## Task 8: Verify Historical Env Detail (Sanctum)

**Files:**
- Check: `src/esper/karn/sanctum/widgets/historical_env_detail.py`

**Step 1: Verify SeedCard reuse**

Historical env detail reuses `SeedCard` from `env_detail_screen.py`. Verify no additional changes needed.

Run: `grep -n "SeedCard" src/esper/karn/sanctum/widgets/historical_env_detail.py`

If SeedCard is imported and used, no changes needed - it will inherit the curve glyph display logic (showing curve only during BLENDING).

**Step 2: Commit (if changes needed)**

If no changes: Skip this commit.

---

## Summary

| Task | Files | Tests | Display Rule |
|------|-------|-------|--------------|
| 1. Telemetry payloads | telemetry.py | test_telemetry_alpha_curve.py | Always emit |
| 2. Sanctum schema | schema.py | test_schema_alpha_curve.py | Always store |
| 3. Aggregator wiring | aggregator.py | test_aggregator_alpha_curve.py | Always copy |
| 4. SeedCard display | env_detail_screen.py | visual | BLENDING only |
| 5. EnvOverview display | env_overview.py | existing tests | BLENDING only |
| 6. TypeScript types | sanctum.ts | npm build | Always present |
| 7. SeedChip display | SeedChip.vue, SeedSwimlane.vue | Playwright | BLENDING only |
| 8. Historical detail | historical_env_detail.py | visual | Inherits SeedCard |

**Total estimated time:** 45-60 minutes

**Key Design Decisions:**
1. **Always emit** - curve is always sampled by policy (advantage masking handles causal attribution)
2. **Display BLENDING only** - curve glyph shown only when causally active
3. **Strict deserialization** - missing `alpha_curve` field raises KeyError (no silent defaults)

---

## Review Status

- [x] DRL specialist approved (two-layer masking design is correct)
- [x] PyTorch specialist approved (serialization pattern is fine for schema migration)
- [x] Head masking audit complete (all heads correctly configured)
- [x] **Post-compaction validation complete** (2025-12-28)

### Errors Fixed During Validation

| Error | Fix Applied |
|-------|-------------|
| Wrong path: `advantages.py` | Corrected to `src/esper/simic/agent/advantages.py` |
| Misleading line refs in Task 1 | Clarified: class location vs field location |
| Misleading line refs in Task 3 | Clarified: handler start vs assignment location |
| **Used `.get()` with defaults** | Changed to strict `data["alpha_curve"]` - missing field = KeyError |

### Verified Correct (No Changes Needed)

- SeedGerminatedPayload/SeedStageChangedPayload only have `from_dict()` (not `to_dict()`) - plan tests correctly only test `from_dict()` for these classes
- SeedTelemetry has both `to_dict()` and `from_dict()` - plan tests are correct
- Line 127 in schema.py is `blend_tempo_epochs` field - correct
- Line 43 in sanctum.ts is `blend_tempo_epochs` field - correct
- SeedChip.vue and SeedSwimlane.vue exist with expected structure
- Tests now enforce strict deserialization (KeyError on missing field, no `.get()` defaults)

Ready for implementation.
