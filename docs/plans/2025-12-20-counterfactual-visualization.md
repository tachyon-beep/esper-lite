# Counterfactual Matrix Visualization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Display full factorial counterfactual analysis in the EnvDetailScreen modal, showing each seed's marginal impact and interaction effects (synergy detection).

**Architecture:** Emit full counterfactual matrix as single telemetry event, capture in Sanctum aggregator, render as waterfall visualization in modal.

**Tech Stack:** Python dataclasses, Textual/Rich widgets, existing telemetry pipeline

---

## Design Decisions

1. **Full factorial only** - If matrix unavailable, show "detailed counterfactual analysis unavailable"
2. **Dedicated section** - New section below metrics in EnvDetailScreen
3. **Waterfall visualization** - Show baseline → individuals → pairs → combined with synergy calculation
4. **Scalable layout:**
   - 1-3 seeds: Show all 2^n configurations
   - 4+ seeds: Show individuals → top 5 combinations by synergy → combined

---

## Task 1: Add Event Type

**Files:**
- Modify: `src/esper/leyline/telemetry.py`

**Step 1: Add event type to TelemetryEventType enum**

After `COUNTERFACTUAL_COMPUTED` (~line 81), add:

```python
    COUNTERFACTUAL_MATRIX_COMPUTED = auto()  # Full factorial matrix for env
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED)"`

---

## Task 2: Add Schema Dataclasses

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Add CounterfactualConfig dataclass**

Add after imports, before other dataclasses (~line 20):

```python
@dataclass
class CounterfactualConfig:
    """Single configuration result from factorial evaluation.

    Represents one row in the counterfactual matrix:
    e.g., seed_mask=(True, False, True) means slots 0 and 2 enabled.
    """
    seed_mask: tuple[bool, ...]  # Which seeds are enabled
    accuracy: float = 0.0  # Validation accuracy for this config
```

**Step 2: Add CounterfactualSnapshot dataclass**

Add after CounterfactualConfig:

```python
@dataclass
class CounterfactualSnapshot:
    """Full factorial counterfactual matrix for an environment.

    Contains all 2^n configurations for n active seeds.
    Used to compute marginal contributions and interaction terms.
    """
    slot_ids: tuple[str, ...] = ()  # ("r0c0", "r0c1", "r0c2")
    configs: list[CounterfactualConfig] = field(default_factory=list)
    strategy: str = "unavailable"  # "full_factorial" or "unavailable"
    compute_time_ms: float = 0.0

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy with all seeds disabled."""
        for cfg in self.configs:
            if not any(cfg.seed_mask):
                return cfg.accuracy
        return 0.0

    @property
    def combined_accuracy(self) -> float:
        """Accuracy with all seeds enabled."""
        for cfg in self.configs:
            if all(cfg.seed_mask):
                return cfg.accuracy
        return 0.0

    def get_accuracy(self, mask: tuple[bool, ...]) -> float | None:
        """Get accuracy for a specific seed configuration."""
        for cfg in self.configs:
            if cfg.seed_mask == mask:
                return cfg.accuracy
        return None

    def individual_contributions(self) -> dict[str, float]:
        """Compute each seed's solo contribution over baseline."""
        baseline = self.baseline_accuracy
        result = {}
        n = len(self.slot_ids)
        for i, slot_id in enumerate(self.slot_ids):
            mask = tuple(j == i for j in range(n))
            acc = self.get_accuracy(mask)
            if acc is not None:
                result[slot_id] = acc - baseline
        return result

    def pair_contributions(self) -> dict[tuple[str, str], float]:
        """Compute each pair's contribution over baseline."""
        baseline = self.baseline_accuracy
        result = {}
        n = len(self.slot_ids)
        for i in range(n):
            for j in range(i + 1, n):
                mask = tuple(k == i or k == j for k in range(n))
                acc = self.get_accuracy(mask)
                if acc is not None:
                    pair = (self.slot_ids[i], self.slot_ids[j])
                    result[pair] = acc - baseline
        return result

    def total_synergy(self) -> float:
        """Compute total synergy: combined - baseline - sum(individual contributions)."""
        baseline = self.baseline_accuracy
        combined = self.combined_accuracy
        individuals = self.individual_contributions()
        expected = baseline + sum(individuals.values())
        return combined - expected
```

**Step 3: Add field to EnvState**

In `EnvState` dataclass, add after `reward_components`:

```python
    counterfactual_matrix: CounterfactualSnapshot = field(
        default_factory=CounterfactualSnapshot
    )
```

**Step 4: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.schema import CounterfactualSnapshot, EnvState"`

---

## Task 3: Emit Matrix from Training Loop

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`

**Context:** The `CounterfactualHelper.compute_contributions()` is called during epoch completion and stores the matrix in `last_matrix`. We need to emit the full matrix as a single event after Shapley computation.

**Step 1: Find the Shapley computation section**

Search for `compute_contributions` in vectorized.py (~line 2370). This is inside the epoch completion logic.

**Step 2: After the compute_contributions call, emit the full matrix**

After the existing `compute_contributions()` call and its logging, add:

```python
                                # Emit full counterfactual matrix for Sanctum visualization
                                counterfactual_matrix = env_state.counterfactual_helper.last_matrix
                                if hub and counterfactual_matrix is not None and counterfactual_matrix.configs:
                                    matrix_data = {
                                        "env_id": env_idx,
                                        "slot_ids": list(counterfactual_matrix.configs[0].slot_ids),
                                        "configs": [
                                            {
                                                "seed_mask": list(cfg.config),
                                                "accuracy": cfg.val_accuracy,
                                            }
                                            for cfg in counterfactual_matrix.configs
                                        ],
                                        "strategy": counterfactual_matrix.strategy_used,
                                        "compute_time_ms": counterfactual_matrix.compute_time_seconds * 1000,
                                    }
                                    hub.emit(TelemetryEvent(
                                        event_type=TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED,
                                        data=matrix_data,
                                    ))
```

**Note:** The matrix is accessed via `env_state.counterfactual_helper.last_matrix` which is populated by `compute_contributions()`.

**Step 3: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.simic.training.vectorized import VectorizedTrainer"`

---

## Task 4: Capture Matrix in Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Add import**

Add to schema imports:

```python
from esper.karn.sanctum.schema import (
    ...
    CounterfactualConfig,
    CounterfactualSnapshot,
)
```

**Step 2: Add handler registration**

In `_register_handlers` method, add:

```python
        self._handlers[TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED] = self._handle_counterfactual_matrix
```

**Step 3: Add handler method**

Add after `_handle_batch_completed`:

```python
    def _handle_counterfactual_matrix(self, event: "TelemetryEvent") -> None:
        """Handle COUNTERFACTUAL_MATRIX_COMPUTED event."""
        data = event.data or {}
        env_id = data.get("env_id")

        if env_id is None:
            return

        env = self._envs.get(env_id)
        if env is None:
            return

        # Parse configs
        slot_ids = tuple(data.get("slot_ids", []))
        raw_configs = data.get("configs", [])

        configs = [
            CounterfactualConfig(
                seed_mask=tuple(cfg.get("seed_mask", [])),
                accuracy=cfg.get("accuracy", 0.0),
            )
            for cfg in raw_configs
        ]

        env.counterfactual_matrix = CounterfactualSnapshot(
            slot_ids=slot_ids,
            configs=configs,
            strategy=data.get("strategy", "unavailable"),
            compute_time_ms=data.get("compute_time_ms", 0.0),
        )
```

**Step 4: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.aggregator import SanctumAggregator"`

---

## Task 5: Add Waterfall Visualization Widget

**Files:**
- Create: `src/esper/karn/sanctum/widgets/counterfactual_panel.py`

**Step 1: Create the widget file**

```python
"""CounterfactualPanel - Waterfall visualization of factorial counterfactual analysis.

Shows baseline → individuals → pairs → combined with synergy calculation.
Displays "detailed counterfactual analysis unavailable" if no full factorial data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import CounterfactualSnapshot


class CounterfactualPanel(Static):
    """Waterfall visualization of counterfactual analysis."""

    def __init__(self, matrix: "CounterfactualSnapshot", **kwargs) -> None:
        super().__init__(**kwargs)
        self._matrix = matrix

    def update_matrix(self, matrix: "CounterfactualSnapshot") -> None:
        """Update the matrix and refresh display."""
        self._matrix = matrix
        self.refresh()

    def render(self) -> Panel:
        """Render the counterfactual analysis panel."""
        if self._matrix.strategy == "unavailable" or not self._matrix.configs:
            return self._render_unavailable()
        return self._render_waterfall()

    def _render_unavailable(self) -> Panel:
        """Render unavailable state."""
        content = Text("Detailed counterfactual analysis unavailable", style="dim italic")
        return Panel(content, title="Counterfactual Analysis", border_style="dim")

    def _render_waterfall(self) -> Panel:
        """Render the waterfall visualization."""
        lines = []

        baseline = self._matrix.baseline_accuracy
        combined = self._matrix.combined_accuracy
        individuals = self._matrix.individual_contributions()
        pairs = self._matrix.pair_contributions()
        synergy = self._matrix.total_synergy()
        n_seeds = len(self._matrix.slot_ids)

        # Baseline
        lines.append(self._make_bar_line("Baseline (Host only)", baseline, baseline, combined))
        lines.append(Text(""))

        # Individuals section
        if individuals:
            lines.append(Text("Individual:", style="bold"))
            for slot_id, contrib in individuals.items():
                acc = baseline + contrib
                label = f"  {slot_id} alone"
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib))

        # Pairs section (only for 2-3 seeds, or top 5 for 4+)
        if pairs and n_seeds <= 3:
            lines.append(Text(""))
            lines.append(Text("Pairs:", style="bold"))
            for (s1, s2), contrib in pairs.items():
                acc = baseline + contrib
                label = f"  {s1} + {s2}"
                # Calculate pair synergy
                ind1 = individuals.get(s1, 0)
                ind2 = individuals.get(s2, 0)
                pair_synergy = contrib - ind1 - ind2
                style = "green" if pair_synergy > 0.5 else None
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style))
        elif pairs and n_seeds > 3:
            # Show top 5 by synergy
            lines.append(Text(""))
            lines.append(Text("Top Combinations (by synergy):", style="bold"))

            # Calculate synergy for each pair
            pair_synergies = []
            for (s1, s2), contrib in pairs.items():
                ind1 = individuals.get(s1, 0)
                ind2 = individuals.get(s2, 0)
                pair_synergy = contrib - ind1 - ind2
                pair_synergies.append(((s1, s2), contrib, pair_synergy))

            # Sort by synergy descending, take top 5
            pair_synergies.sort(key=lambda x: x[2], reverse=True)
            for (s1, s2), contrib, pair_syn in pair_synergies[:5]:
                acc = baseline + contrib
                label = f"  {s1} + {s2}"
                style = "green" if pair_syn > 0.5 else None
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style))

        # Combined
        lines.append(Text(""))
        lines.append(Text("Combined:", style="bold"))
        improvement = combined - baseline
        lines.append(self._make_bar_line("  All seeds", combined, baseline, combined, improvement))

        # Synergy summary
        lines.append(Text(""))
        expected = sum(individuals.values())
        lines.append(Text(f"Expected (sum of solo): +{expected:.1f}%", style="dim"))
        lines.append(Text(f"Actual improvement:     +{improvement:.1f}%", style="dim"))

        # Interference is MORE critical to surface than synergy - seeds hurting each other
        # Use loud visual treatment for negative cases
        if synergy < -0.5:
            # INTERFERENCE: Seeds are hurting each other - make this LOUD
            lines.append(Text(""))
            lines.append(Text("✗ INTERFERENCE DETECTED", style="bold red reverse"))
            lines.append(Text(f"  Seeds are hurting each other by {synergy:.1f}%", style="red"))
        elif synergy > 0.5:
            # Synergy: Seeds working together
            lines.append(Text(f"✓ Synergy:              +{synergy:.1f}%", style="bold green"))
        else:
            # Neutral: Seeds are independent
            lines.append(Text(f"  Interaction:          {synergy:+.1f}%", style="dim"))

        content = Group(*lines)
        return Panel(content, title="Counterfactual Analysis", border_style="cyan")

    def _make_bar_line(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        delta: float | None = None,
        highlight: str | None = None,
    ) -> Text:
        """Create a bar line with label, visual bar, and value."""
        # Normalize to 0-1 range
        range_val = max_val - min_val if max_val > min_val else 1.0
        normalized = (value - min_val) / range_val if range_val > 0 else 0.0
        normalized = max(0.0, min(1.0, normalized))

        bar_width = 30
        filled = int(normalized * bar_width)
        empty = bar_width - filled

        line = Text()
        line.append(f"{label:20s} ", style=highlight or "white")
        line.append("█" * filled, style="cyan")
        line.append("░" * empty, style="dim")
        line.append(f" {value:5.1f}%", style="white")

        if delta is not None:
            delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
            line.append(f"  ({delta:+.1f})", style=delta_style)

        return line
```

**Step 2: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel"`

---

## Task 6: Integrate Panel into EnvDetailScreen

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/env_detail_screen.py`

**Step 1: Add import**

```python
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel
```

**Step 2: Add panel to compose method**

In `compose()`, after the metrics section (~line 260), add:

```python
            # Counterfactual analysis section
            with Vertical(classes="counterfactual-section"):
                yield CounterfactualPanel(
                    self._env.counterfactual_matrix,
                    id="counterfactual-panel"
                )
```

**Step 3: Add CSS for the section**

In `DEFAULT_CSS`, add:

```css
    EnvDetailScreen .counterfactual-section {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }
```

**Step 4: Update panel in update_env_state method**

In `update_env_state()`, add:

```python
        # Update counterfactual panel
        try:
            cf_panel = self.query_one("#counterfactual-panel", CounterfactualPanel)
            cf_panel.update_matrix(env_state.counterfactual_matrix)
        except Exception:
            pass
```

**Step 5: Verify syntax**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen"`

---

## Task 7: Add Tests

**Files:**
- Create: `tests/karn/sanctum/test_counterfactual_panel.py`

**Step 1: Create test file**

```python
"""Tests for CounterfactualPanel widget."""
import pytest
from esper.karn.sanctum.schema import CounterfactualConfig, CounterfactualSnapshot
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel


class TestCounterfactualSnapshot:
    """Test CounterfactualSnapshot dataclass methods."""

    def test_baseline_accuracy(self):
        """Baseline is config with all False."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        assert snapshot.baseline_accuracy == 25.0

    def test_combined_accuracy(self):
        """Combined is config with all True."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        assert snapshot.combined_accuracy == 65.0

    def test_individual_contributions(self):
        """Individual contribution is solo accuracy minus baseline."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=30.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        contribs = snapshot.individual_contributions()
        assert contribs["r0c0"] == 10.0  # 35 - 25
        assert contribs["r0c1"] == 5.0   # 30 - 25

    def test_total_synergy_positive(self):
        """Synergy when combined > sum of individuals."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),   # +40 total
            ],
            strategy="full_factorial",
        )
        # Expected: 25 + 10 + 10 = 45, Actual: 65, Synergy: 20
        assert snapshot.total_synergy() == 20.0

    def test_total_synergy_negative(self):
        """Interference when combined < sum of individuals."""
        snapshot = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),  # +10
                CounterfactualConfig(seed_mask=(True, True), accuracy=40.0),   # +15 total (interference!)
            ],
            strategy="full_factorial",
        )
        # Expected: 25 + 10 + 10 = 45, Actual: 40, Synergy: -5
        assert snapshot.total_synergy() == -5.0


class TestCounterfactualPanel:
    """Test CounterfactualPanel widget rendering."""

    def test_renders_unavailable_when_no_data(self):
        """Panel shows unavailable message when strategy is unavailable."""
        matrix = CounterfactualSnapshot(strategy="unavailable")
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()
        assert "unavailable" in str(rendered.renderable).lower()

    def test_renders_waterfall_with_data(self):
        """Panel renders waterfall when data available."""
        matrix = CounterfactualSnapshot(
            slot_ids=("r0c0", "r0c1"),
            configs=[
                CounterfactualConfig(seed_mask=(False, False), accuracy=25.0),
                CounterfactualConfig(seed_mask=(True, False), accuracy=35.0),
                CounterfactualConfig(seed_mask=(False, True), accuracy=35.0),
                CounterfactualConfig(seed_mask=(True, True), accuracy=65.0),
            ],
            strategy="full_factorial",
        )
        panel = CounterfactualPanel(matrix)
        rendered = panel.render()
        # Should contain key elements
        content = str(rendered.renderable)
        assert "Baseline" in content or "baseline" in content.lower()
```

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_counterfactual_panel.py -v`

---

## Task 8: Run Full Test Suite

**Step 1: Run all Sanctum tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`

Expected: All tests pass

**Step 2: Verify imports work end-to-end**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.app import SanctumApp; print('OK')"`

---

## Task 9: Commit

```bash
git add src/esper/leyline/telemetry.py \
        src/esper/karn/sanctum/schema.py \
        src/esper/karn/sanctum/aggregator.py \
        src/esper/karn/sanctum/widgets/counterfactual_panel.py \
        src/esper/karn/sanctum/widgets/env_detail_screen.py \
        src/esper/simic/training/vectorized.py \
        tests/karn/sanctum/test_counterfactual_panel.py

git commit -m "feat(sanctum): add counterfactual matrix visualization

Display full factorial counterfactual analysis in EnvDetailScreen modal:

Telemetry:
- Add COUNTERFACTUAL_MATRIX_COMPUTED event type
- Emit full matrix after counterfactual computation in training loop

Schema:
- CounterfactualConfig: single configuration result
- CounterfactualSnapshot: full matrix with utility methods
  - baseline_accuracy, combined_accuracy
  - individual_contributions(), pair_contributions()
  - total_synergy() for emergence detection

Visualization:
- CounterfactualPanel widget with waterfall display
- Shows baseline → individuals → pairs → combined
- Highlights synergy/interference between seeds
- Graceful 'unavailable' state when no full factorial data

Tests:
- Unit tests for snapshot methods
- Widget rendering tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After implementation, verify:

| Check | Command |
|-------|---------|
| Event type exists | `python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED)"` |
| Schema imports | `python -c "from esper.karn.sanctum.schema import CounterfactualSnapshot"` |
| Widget imports | `python -c "from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel"` |
| Tests pass | `pytest tests/karn/sanctum/test_counterfactual_panel.py -v` |
| Full suite | `pytest tests/karn/sanctum/ -v` |

---

## PyTorch/CUDA Considerations

**Reviewed by:** pytorch-expert agent (2025-12-20)

**Status:** ✅ APPROVED

### Findings:

1. **CUDA synchronization: SAFE**
   - The `evaluate_fn` used by `CounterfactualHelper.compute_contributions()` is pure Python
   - It looks up cached baselines from Python dicts, no GPU operations run
   - Emission location is safe

2. **Tensor serialization: SAFE**
   - All data in `CounterfactualMatrix` is already Python primitives
   - `config`: `tuple[bool, ...]`
   - `val_accuracy`: `float`
   - `slot_ids`: `tuple[str, ...]`
   - No `.detach().cpu().item()` calls needed

3. **Memory pressure: NEGLIGIBLE**
   - Max 16 configs × ~550 bytes = <10KB per env per epoch
   - No concern holding this in Python during GPU operations

4. **Thread safety: SAFE**
   - `hub.emit()` is already used extensively in this code path
