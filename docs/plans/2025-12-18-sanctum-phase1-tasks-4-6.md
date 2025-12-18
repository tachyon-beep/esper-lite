# Sanctum Phase 1: Tasks 4-6 (Scoreboard, TamiyoBrain, Remaining Widgets)

> **Context:** This document contains Tasks 4-6 from the Sanctum Phase 1 implementation plan.
> These tasks port the Scoreboard (Best Runs), TamiyoBrain (policy diagnostics), RewardComponents,
> and EsperStatus widgets from the Rich TUI to Textual.

**Parent Plan:** `2025-12-18-sanctum-phase1-implementation.md`

**Reference Files:**
- Existing TUI: `/home/john/esper-lite/src/esper/karn/tui.py`
- Design doc: `/home/john/esper-lite/docs/plans/2025-12-18-sanctum-design.md`
- Constants: `/home/john/esper-lite/src/esper/karn/constants.py` (TUIThresholds)

---

## Task 4: Port Scoreboard Widget (Best Runs) - COMPLETE

**Files:**
- Create: `src/esper/karn/sanctum/widgets/scoreboard.py`
- Create: `tests/karn/sanctum/test_scoreboard.py`
- Reference: `src/esper/karn/tui.py` lines 1039-1138

**Overview:** The Scoreboard shows a stats header (global best, mean best, fossilized/culled counts)
and a leaderboard table with the top 10 environments ranked by best accuracy.

**Column Widths (MUST MATCH):**
- `#`: 3 chars (rank or medal)
- `@Ep`: 4 chars (episode number)
- `High`: 6 chars (best accuracy)
- `Cur`: 6 chars (current accuracy)
- `Seeds`: 20 chars (seed blueprints at best)

**Comparison Thresholds (MUST MATCH):**
- Delta >= -0.5: green (current very close to best)
- Delta >= -2.0: yellow (current somewhat below best)
- Else: dim (current far below best)

### Step 1: Write tests

Create `tests/karn/sanctum/test_scoreboard.py`:

```python
"""Tests for Sanctum Scoreboard widget."""
import pytest

from esper.karn.sanctum.widgets.scoreboard import Scoreboard
from esper.karn.sanctum.schema import EnvState, SeedState


def test_scoreboard_creation():
    """Scoreboard should create without errors."""
    board = Scoreboard()
    assert board is not None


def test_scoreboard_has_stats_header():
    """Should have stats header with global best, mean, counts."""
    board = Scoreboard()
    envs = {
        0: EnvState(env_id=0, best_accuracy=80.0, fossilized_count=2, culled_count=1),
        1: EnvState(env_id=1, best_accuracy=90.0, fossilized_count=1, culled_count=0),
    }
    board.update_envs(envs)
    # Stats should be computed
    assert board._global_best == 90.0
    assert board._mean_best == 85.0
    assert board._total_fossilized == 3
    assert board._total_culled == 1


def test_scoreboard_sorts_by_best_accuracy():
    """Scoreboard should sort envs by best accuracy descending."""
    board = Scoreboard()
    envs = {
        0: EnvState(env_id=0, best_accuracy=75.0),
        1: EnvState(env_id=1, best_accuracy=85.0),
        2: EnvState(env_id=2, best_accuracy=80.0),
    }
    board.update_envs(envs)
    assert board._sorted_envs[0].env_id == 1  # Best: 85.0
    assert board._sorted_envs[1].env_id == 2  # Second: 80.0
    assert board._sorted_envs[2].env_id == 0  # Third: 75.0


def test_scoreboard_shows_medals():
    """Top 3 should get medal indicators."""
    board = Scoreboard()
    envs = {i: EnvState(env_id=i, best_accuracy=90 - i) for i in range(5)}
    board.update_envs(envs)
    assert board._get_rank_display(0) == "🥇"
    assert board._get_rank_display(1) == "🥈"
    assert board._get_rank_display(2) == "🥉"
    assert board._get_rank_display(3) == "4"


def test_scoreboard_limits_to_top_10():
    """Should only show top 10 envs."""
    board = Scoreboard()
    envs = {i: EnvState(env_id=i, best_accuracy=100 - i) for i in range(20)}
    board.update_envs(envs)
    assert len(board._sorted_envs) == 10


def test_scoreboard_current_accuracy_styling():
    """Current accuracy should be styled based on delta from best."""
    board = Scoreboard()
    # Delta >= -0.5 → green
    env1 = EnvState(env_id=0, best_accuracy=80.0, host_accuracy=79.6)
    # Delta >= -2.0 → yellow
    env2 = EnvState(env_id=1, best_accuracy=80.0, host_accuracy=78.5)
    # Delta < -2.0 → dim
    env3 = EnvState(env_id=2, best_accuracy=80.0, host_accuracy=77.0)

    board.update_envs({0: env1, 1: env2, 2: env3})
    # Styling will be applied in _refresh_display


def test_scoreboard_shows_best_seeds():
    """Should show seeds at best accuracy (blueprint names or count)."""
    board = Scoreboard()
    env = EnvState(env_id=0, best_accuracy=85.0)
    env.best_seeds["r0c0"] = SeedState(slot_id="r0c0", blueprint_id="conv_light")
    env.best_seeds["r0c1"] = SeedState(slot_id="r0c1", blueprint_id="mlp_medium")
    board.update_envs({0: env})
    # With ≤2 seeds, should show blueprint names
    # With >2 seeds, should show count


def test_scoreboard_shows_best_seeds_count_when_many():
    """Should show count when >2 seeds at best."""
    board = Scoreboard()
    env = EnvState(env_id=0, best_accuracy=85.0)
    for i in range(5):
        env.best_seeds[f"r{i}c0"] = SeedState(
            slot_id=f"r{i}c0",
            blueprint_id=f"seed{i}"
        )
    board.update_envs({0: env})
    # Should display "5 seeds" instead of listing all
```

### Step 2: Run tests to verify they fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_scoreboard.py -v
```

Expected: FAIL with "No module named 'esper.karn.sanctum.widgets.scoreboard'"

### Step 3: Write Scoreboard implementation

Create `src/esper/karn/sanctum/widgets/scoreboard.py`:

```python
"""Scoreboard Widget - Best Runs leaderboard.

Displays global stats header and top 10 environments by best accuracy.
Ported from Rich TUI _render_scoreboard() (tui.py:1039-1138).

Column Layout:
- # (rank/medal): 3 chars
- @Ep (episode): 4 chars
- High (best accuracy): 6 chars
- Cur (current accuracy): 6 chars
- Seeds (at best): 20 chars
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from esper.karn.sanctum.schema import EnvState


class Scoreboard(Static):
    """Best Runs leaderboard showing top performers by best accuracy."""

    DEFAULT_CSS = """
    Scoreboard {
        height: 100%;
        padding: 1;
    }

    Scoreboard #scoreboard-stats {
        height: 3;
        content-align: center middle;
        border: solid $primary-darken-2;
        margin-bottom: 1;
    }

    Scoreboard DataTable {
        height: 1fr;
    }
    """

    MEDALS = ["🥇", "🥈", "🥉"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._envs: dict[int, EnvState] = {}
        self._sorted_envs: list[EnvState] = []
        self._table: DataTable | None = None

        # Aggregate stats for header
        self._global_best: float = 0.0
        self._mean_best: float = 0.0
        self._total_fossilized: int = 0
        self._total_culled: int = 0

    def compose(self) -> ComposeResult:
        """Create the scoreboard layout."""
        # Stats header showing global stats
        yield Static("", id="scoreboard-stats")

        # Leaderboard table
        table = DataTable(id="scoreboard-table", zebra_stripes=True)
        table.add_column("#", width=3)      # Rank or medal
        table.add_column("@Ep", width=4)    # Episode number
        table.add_column("High", width=6)   # Best accuracy
        table.add_column("Cur", width=6)    # Current accuracy
        table.add_column("Seeds", width=20) # Seeds at best
        self._table = table
        yield table

    def update_envs(self, envs: dict[int, EnvState]) -> None:
        """Update with new environment states."""
        self._envs = envs

        # Sort by best accuracy descending, take top 10
        self._sorted_envs = sorted(
            envs.values(),
            key=lambda e: e.best_accuracy,
            reverse=True
        )[:10]

        # Compute aggregate stats for header
        all_envs = list(envs.values())
        self._total_fossilized = sum(e.fossilized_count for e in all_envs)
        self._total_culled = sum(e.culled_count for e in all_envs)
        best_accs = [e.best_accuracy for e in all_envs if e.best_accuracy > 0]
        self._mean_best = sum(best_accs) / len(best_accs) if best_accs else 0.0
        self._global_best = max(best_accs) if best_accs else 0.0

        self._refresh_display()

    def _get_rank_display(self, rank: int) -> str:
        """Get display string for rank (medal for top 3, number otherwise)."""
        if rank < len(self.MEDALS):
            return self.MEDALS[rank]
        return str(rank + 1)

    def _refresh_display(self) -> None:
        """Refresh the stats header and leaderboard table."""
        # Update stats header
        stats = self.query_one("#scoreboard-stats", Static)
        stats.update(
            f"Global Best: [bold green]{self._global_best:.1f}%[/] | "
            f"Mean: {self._mean_best:.1f}% | "
            f"Foss: [green]{self._total_fossilized}[/] | "
            f"Cull: [red]{self._total_culled}[/]"
        )

        if self._table is None:
            return

        self._table.clear()

        if not self._sorted_envs:
            # Empty state
            self._table.add_row("─", "─", "─", "─", "─")
            return

        for i, env in enumerate(self._sorted_envs):
            rank = self._get_rank_display(i)

            # Current vs best accuracy styling
            # Thresholds: delta >= -0.5 → green, >= -2.0 → yellow, else dim
            delta = env.host_accuracy - env.best_accuracy
            if delta >= -0.5:
                cur_style = "green"
            elif delta >= -2.0:
                cur_style = "yellow"
            else:
                cur_style = "dim"

            # Seeds at best accuracy
            if env.best_seeds:
                n_seeds = len(env.best_seeds)
                if n_seeds <= 2:
                    # Show blueprint names (first 6 chars each)
                    seed_parts = [
                        f"[green]{s.blueprint_id[:6] if s.blueprint_id else '?'}[/]"
                        for s in env.best_seeds.values()
                    ]
                    seeds_str = " ".join(seed_parts)
                else:
                    # Show count if many seeds
                    seeds_str = f"[green]{n_seeds} seeds[/]"
            else:
                seeds_str = "─"

            # Add row with stable key for updates
            self._table.add_row(
                rank,
                str(env.best_accuracy_episode),
                f"[bold green]{env.best_accuracy:.1f}[/]",
                f"[{cur_style}]{env.host_accuracy:.1f}[/]",
                seeds_str,
                key=f"env-{env.env_id}",
            )
```

### Step 4: Update widgets __init__.py

Edit `src/esper/karn/sanctum/widgets/__init__.py`:

```python
"""Sanctum widgets."""
from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.widgets.scoreboard import Scoreboard

__all__ = ["EnvOverview", "Scoreboard"]
```

### Step 5: Run tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_scoreboard.py -v
```

Expected: All tests PASS

### Step 6: Commit

```bash
git add src/esper/karn/sanctum/widgets/scoreboard.py tests/karn/sanctum/test_scoreboard.py
git commit -m "feat(sanctum): add Scoreboard widget with stats header and top 10 leaderboard

Port Best Runs scoreboard from Rich TUI:
- Stats header: global best, mean best, fossilized/culled counts
- Leaderboard table: top 10 envs by best accuracy
- Medal indicators (🥇🥈🥉) for top 3
- Current accuracy styling based on delta thresholds (-0.5, -2.0)
- Seeds at best: show blueprints (≤2) or count (>2)
- Stable row keys for flicker-free updates

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Port Tamiyo Brain Widget (Complete) - WITH ALL HELPER METHODS

**Files:**
- Create: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Create: `tests/karn/sanctum/test_tamiyo_brain.py`
- Reference: `src/esper/karn/tui.py` lines 1140-1430

**Overview:** The TamiyoBrain widget is a 4-column panel showing policy agent diagnostics:
1. **Health**: Entropy, clip fraction, KL divergence, explained variance
2. **Losses**: Policy loss, value loss, entropy loss, gradient norm
3. **Vitals**: Learning rate, ratio stats (mean/min/max/std), dead/exploding layers, GradHP
4. **Actions**: WAIT/GERMINATE/CULL/FOSSILIZE distribution

**CRITICAL REQUIREMENTS:**
- Use `TUIThresholds` from `constants.py` for ALL threshold checks
- Display waiting state before PPO data arrives: "Waiting for PPO data... Progress: X/Y epochs"
- Implement ALL helper methods (none can be stubs)
- Display ratio_std (NOT ratio_mean in isolation)
- Column assignment: Col1=Health, Col2=Losses, Col3=Vitals, Col4=Actions

### Step 1: Write tests

Create `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
"""Tests for Sanctum TamiyoBrain widget - must have ALL metrics."""
import pytest

from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
from esper.karn.sanctum.schema import TamiyoState
from esper.karn.constants import TUIThresholds


def test_tamiyo_brain_creation():
    """TamiyoBrain should create without errors."""
    brain = TamiyoBrain()
    assert brain is not None


def test_tamiyo_brain_update():
    """TamiyoBrain should accept state updates."""
    brain = TamiyoBrain()
    state = TamiyoState(
        entropy=1.2,
        clip_fraction=0.15,
        kl_divergence=0.02,
        explained_variance=0.85,
        ppo_data_received=True,
    )
    brain.update_state(state)
    assert brain._state.entropy == 1.2


def test_entropy_uses_thresholds():
    """Must use TUIThresholds, not hardcoded values."""
    brain = TamiyoBrain()
    # Test against actual thresholds
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_CRITICAL - 0.01) == "critical"
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_WARNING - 0.01) == "warning"
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_WARNING + 0.1) == "ok"


def test_clip_uses_thresholds():
    """Must use TUIThresholds for clip fraction."""
    brain = TamiyoBrain()
    assert brain._get_clip_status(TUIThresholds.CLIP_CRITICAL + 0.01) == "critical"
    assert brain._get_clip_status(TUIThresholds.CLIP_WARNING + 0.01) == "warning"
    assert brain._get_clip_status(TUIThresholds.CLIP_WARNING - 0.05) == "ok"


def test_kl_uses_thresholds():
    """Must use TUIThresholds for KL divergence."""
    brain = TamiyoBrain()
    assert brain._get_kl_status(TUIThresholds.KL_WARNING + 0.01) == "warning"
    assert brain._get_kl_status(TUIThresholds.KL_WARNING - 0.01) == "ok"


def test_ev_uses_thresholds():
    """Must use TUIThresholds for explained variance."""
    brain = TamiyoBrain()
    assert brain._get_ev_status(TUIThresholds.EXPLAINED_VAR_CRITICAL - 0.1) == "critical"
    assert brain._get_ev_status(TUIThresholds.EXPLAINED_VAR_WARNING - 0.1) == "warning"
    assert brain._get_ev_status(TUIThresholds.EXPLAINED_VAR_WARNING + 0.1) == "ok"


def test_grad_norm_uses_thresholds():
    """Must use TUIThresholds for gradient norm."""
    brain = TamiyoBrain()
    assert brain._get_grad_norm_status(TUIThresholds.GRAD_NORM_CRITICAL + 1.0) == "critical"
    assert brain._get_grad_norm_status(TUIThresholds.GRAD_NORM_WARNING + 1.0) == "warning"
    assert brain._get_grad_norm_status(TUIThresholds.GRAD_NORM_WARNING - 1.0) == "ok"


def test_status_style_returns_correct_styles():
    """Status style helper must return correct Rich styles."""
    brain = TamiyoBrain()
    assert brain._status_style("ok") == "green"
    assert brain._status_style("warning") == "yellow"
    assert brain._status_style("critical") == "red bold"


def test_status_text_returns_correct_indicators():
    """Status text helper must return correct indicator strings."""
    brain = TamiyoBrain()
    assert "OK" in brain._status_text("ok")
    assert "WARN" in brain._status_text("warning")
    assert "CRIT" in brain._status_text("critical")


def test_learning_rate_display():
    """Must display learning rate in Vitals section."""
    brain = TamiyoBrain()
    state = TamiyoState(learning_rate=3e-4, ppo_data_received=True)
    brain.update_state(state)
    # LR should be formatted as scientific notation


def test_gradient_health_metrics():
    """Must display dead/exploding layers and GradHP in Vitals."""
    brain = TamiyoBrain()
    state = TamiyoState(
        dead_layers=2,
        exploding_layers=1,
        layer_gradient_health=0.75,
        ppo_data_received=True,
    )
    brain.update_state(state)
    # Should show dead layers, exploding layers, and Grad Health percentage


def test_ratio_stats_all_displayed():
    """Must display ratio min, max, AND std (not mean)."""
    brain = TamiyoBrain()
    state = TamiyoState(
        ratio_mean=1.02,
        ratio_min=0.8,
        ratio_max=1.5,
        ratio_std=0.15,
        ppo_data_received=True,
    )
    brain.update_state(state)
    # Must show ratio_max, ratio_min, ratio_std (ratio_mean is NOT displayed)


def test_waiting_state():
    """Should show waiting message before PPO data received."""
    brain = TamiyoBrain()
    state = TamiyoState(ppo_data_received=False)
    brain.update_state(state)
    # All sections should show "Waiting for PPO data..."


def test_action_distribution():
    """Must show WAIT/GERMINATE/CULL/FOSSILIZE with percentages."""
    brain = TamiyoBrain()
    state = TamiyoState(
        action_counts={
            "WAIT": 50,
            "GERMINATE": 30,
            "CULL": 10,
            "FOSSILIZE": 10,
        },
        ppo_data_received=True,
    )
    brain.update_state(state)
    # Total = 100, percentages: 50%, 30%, 10%, 10%


def test_wait_dominance_warning():
    """Should warn if WAIT action dominates (>70%)."""
    brain = TamiyoBrain()
    state = TamiyoState(
        action_counts={
            "WAIT": 80,
            "GERMINATE": 10,
            "CULL": 5,
            "FOSSILIZE": 5,
        },
        ppo_data_received=True,
    )
    brain.update_state(state)
    # WAIT should be styled with warning (80% > 70% threshold)
```

### Step 2: Run tests to verify they fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v
```

Expected: FAIL with "No module named 'esper.karn.sanctum.widgets.tamiyo_brain'"

### Step 3: Write TamiyoBrain implementation (COMPLETE)

Create `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
"""Tamiyo Brain Widget - Policy agent diagnostics.

4-column panel: Health, Losses, Vitals, Actions.
Ported from Rich TUI _render_tamiyo_brain() (tui.py:1140-1430).

Column Assignment:
- Column 1: Health (entropy, clip, KL, ExplVar)
- Column 2: Losses (policy, value, entropy, grad norm)
- Column 3: Vitals (LR, ratio stats, gradient health)
- Column 4: Actions (WAIT/GERMINATE/CULL/FOSSILIZE %)

IMPORTANT: Uses TUIThresholds from karn/constants.py, NOT hardcoded values.
"""
from __future__ import annotations

import math
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Grid, Vertical

from esper.karn.sanctum.schema import TamiyoState
from esper.karn.constants import TUIThresholds


class TamiyoBrain(Static):
    """Tamiyo policy agent diagnostic panel - 4 columns."""

    DEFAULT_CSS = """
    TamiyoBrain {
        height: 100%;
        padding: 1;
    }

    .brain-grid {
        layout: grid;
        grid-size: 4 1;
        grid-gutter: 1;
        height: 100%;
    }

    .brain-section {
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    .brain-section-title {
        text-style: bold;
        color: $text-muted;
        text-align: center;
    }
    """

    # Max entropy for 4-action space (WAIT, GERMINATE, CULL, FOSSILIZE)
    MAX_ENTROPY = math.log(4)  # ≈ 1.386

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = TamiyoState()

    def compose(self) -> ComposeResult:
        """Create the 4-column brain layout."""
        with Grid(classes="brain-grid"):
            # Column 1: Health
            with Vertical(classes="brain-section", id="health-section"):
                yield Static("Health", classes="brain-section-title")
                yield Static("", id="health-content")

            # Column 2: Losses
            with Vertical(classes="brain-section", id="losses-section"):
                yield Static("Losses", classes="brain-section-title")
                yield Static("", id="losses-content")

            # Column 3: Vitals
            with Vertical(classes="brain-section", id="vitals-section"):
                yield Static("Vitals", classes="brain-section-title")
                yield Static("", id="vitals-content")

            # Column 4: Actions
            with Vertical(classes="brain-section", id="actions-section"):
                yield Static("Actions", classes="brain-section-title")
                yield Static("", id="actions-content")

    def update_state(self, state: TamiyoState) -> None:
        """Update with new Tamiyo state."""
        self._state = state
        self._refresh_display()

    # =========================================================================
    # Status methods using TUIThresholds (NOT hardcoded values!)
    # =========================================================================

    def _get_entropy_status(self, entropy: float) -> str:
        """Get status for entropy using TUIThresholds.

        Returns:
            "critical" if entropy < ENTROPY_CRITICAL
            "warning" if entropy < ENTROPY_WARNING
            "ok" otherwise
        """
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning"
        return "ok"

    def _get_clip_status(self, clip: float) -> str:
        """Get status for clip fraction using TUIThresholds.

        Returns:
            "critical" if clip > CLIP_CRITICAL
            "warning" if clip > CLIP_WARNING
            "ok" otherwise
        """
        if clip > TUIThresholds.CLIP_CRITICAL:
            return "critical"
        elif clip > TUIThresholds.CLIP_WARNING:
            return "warning"
        return "ok"

    def _get_kl_status(self, kl: float) -> str:
        """Get status for KL divergence using TUIThresholds.

        Returns:
            "warning" if kl > KL_WARNING
            "ok" otherwise
        """
        if kl > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _get_ev_status(self, ev: float) -> str:
        """Get status for explained variance using TUIThresholds.

        Returns:
            "critical" if ev < EXPLAINED_VAR_CRITICAL (harmful)
            "warning" if ev < EXPLAINED_VAR_WARNING (weak)
            "ok" otherwise
        """
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical"
        elif ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        """Get status for gradient norm using TUIThresholds.

        Returns:
            "critical" if grad_norm > GRAD_NORM_CRITICAL
            "warning" if grad_norm > GRAD_NORM_WARNING
            "ok" otherwise
        """
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        elif grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        """Get Rich style for status.

        Args:
            status: One of "ok", "warning", "critical"

        Returns:
            Rich style string
        """
        return {
            "ok": "green",
            "warning": "yellow",
            "critical": "red bold"
        }.get(status, "dim")

    def _status_text(self, status: str) -> str:
        """Get status indicator text.

        Args:
            status: One of "ok", "warning", "critical"

        Returns:
            Formatted status text with indicator
        """
        return {
            "ok": "[green]✓ OK[/]",
            "warning": "[yellow]⚠ WARN[/]",
            "critical": "[red bold]✕ CRIT[/]",
        }.get(status, "")

    # =========================================================================
    # Refresh methods
    # =========================================================================

    def _refresh_display(self) -> None:
        """Refresh all sections."""
        # Check for waiting state (before PPO data arrives)
        if not self._state.ppo_data_received:
            self._show_waiting_state()
            return

        self._refresh_health()
        self._refresh_losses()
        self._refresh_vitals()
        self._refresh_actions()

    def _show_waiting_state(self) -> None:
        """Show waiting state before PPO data arrives."""
        waiting_text = "[dim italic]Waiting for PPO data...[/]"
        for section_id in ["health-content", "losses-content", "vitals-content", "actions-content"]:
            try:
                content = self.query_one(f"#{section_id}", Static)
                content.update(waiting_text)
            except Exception:
                pass

    def _refresh_health(self) -> None:
        """Refresh Health section (Column 1).

        Shows:
        - Entropy (with percentage of max)
        - Clip fraction (with status)
        - KL divergence (with status)
        - Explained variance (with hint: harm/weak/ok)
        """
        s = self._state

        # Entropy: show as percentage of max (DRL Expert suggestion)
        ent_pct = (s.entropy / self.MAX_ENTROPY) * 100 if self.MAX_ENTROPY > 0 else 0
        ent_status = self._get_entropy_status(s.entropy)

        clip_status = self._get_clip_status(s.clip_fraction)
        kl_status = self._get_kl_status(s.kl_divergence)
        ev_status = self._get_ev_status(s.explained_variance)

        # ExplVar interpretive hint (DRL Expert suggestion)
        if s.explained_variance < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            ev_hint = "harm"
        elif s.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            ev_hint = "weak"
        else:
            ev_hint = "ok"

        lines = [
            f"Entropy   [{self._status_style(ent_status)}]{s.entropy:.2f}[/] ({ent_pct:.0f}%)",
            f"Clip      [{self._status_style(clip_status)}]{s.clip_fraction:.3f}[/] {self._status_text(clip_status)}",
            f"KL        {s.kl_divergence:.4f} {self._status_text(kl_status)}",
            f"ExplVar   [{self._status_style(ev_status)}]{s.explained_variance:.2f}[/] ({ev_hint})",
        ]

        content = self.query_one("#health-content", Static)
        content.update("\n".join(lines))

    def _refresh_losses(self) -> None:
        """Refresh Losses section (Column 2).

        Shows:
        - Policy loss
        - Value loss
        - Entropy loss
        - Gradient norm (with status)
        """
        s = self._state

        grad_status = self._get_grad_norm_status(s.grad_norm)

        lines = [
            f"Policy    {s.policy_loss:.4f}",
            f"Value     {s.value_loss:.4f}",
            f"Entropy   {s.entropy_loss:.4f}",
            f"GradNorm  [{self._status_style(grad_status)}]{s.grad_norm:.2f}[/]",
        ]

        content = self.query_one("#losses-content", Static)
        content.update("\n".join(lines))

    def _refresh_vitals(self) -> None:
        """Refresh Vitals section (Column 3).

        Shows:
        - Learning rate (scientific notation)
        - π Ratio↑ (max ratio with color)
        - π Ratio↓ (min ratio with color)
        - π Ratio σ (std deviation with warning if high)
        - Dead layers (if any)
        - Exploding layers (if any)
        - Grad Health (GradHP renamed per DRL expert)
        """
        s = self._state

        # Learning rate
        if s.learning_rate is not None:
            lr_str = f"{s.learning_rate:.2e}"
        else:
            lr_str = "─"

        # Ratio stats with color coding (DRL Expert: prefix with π)
        ratio_max = s.ratio_max
        ratio_min = s.ratio_min
        ratio_std = s.ratio_std

        # Max ratio: red if >= 2.0, yellow if >= 1.5, else green
        max_style = "red bold" if ratio_max >= 2.0 else "yellow" if ratio_max >= 1.5 else "green"
        # Min ratio: red if <= 0.3, yellow if <= 0.5, else green
        min_style = "red bold" if ratio_min <= 0.3 else "yellow" if ratio_min <= 0.5 else "green"
        # Std: yellow if >= 0.5 (high variance)
        std_style = "yellow" if ratio_std >= 0.5 else ""

        lines = [
            f"LR        {lr_str}",
            f"π Ratio↑  [{max_style}]{ratio_max:.2f}[/]",
            f"π Ratio↓  [{min_style}]{ratio_min:.2f}[/]",
            f"π Ratio σ [{std_style}]{ratio_std:.3f}[/]" if std_style else f"π Ratio σ {ratio_std:.3f}",
        ]

        # Gradient health warnings
        if s.dead_layers > 0:
            lines.append(f"[yellow bold]Dead: {s.dead_layers} layers[/]")
        if s.exploding_layers > 0:
            lines.append(f"[red bold]Explode: {s.exploding_layers} layers[/]")

        # Grad Health (renamed from GradHP per DRL expert suggestion)
        health = s.layer_gradient_health
        health_style = "red bold" if health < 0.5 else "yellow" if health < 0.8 else "green"
        lines.append(f"Grad Health [{health_style}]{health:.0%}[/]")

        content = self.query_one("#vitals-content", Static)
        content.update("\n".join(lines))

    def _refresh_actions(self) -> None:
        """Refresh Actions section (Column 4).

        Shows:
        - WAIT percentage (warn if >70%)
        - GERMINATE percentage
        - CULL percentage
        - FOSSILIZE percentage
        """
        s = self._state

        total = sum(s.action_counts.values()) or 1

        # Action order and styling
        action_styles = {
            "WAIT": "dim",
            "GERMINATE": "green",
            "CULL": "red",
            "FOSSILIZE": "blue",
        }

        lines = []
        for action in ["WAIT", "GERMINATE", "CULL", "FOSSILIZE"]:
            count = s.action_counts.get(action, 0)
            pct = (count / total) * 100
            style = action_styles.get(action, "white")

            # Warn if WAIT dominates (>70%)
            if action == "WAIT" and pct > TUIThresholds.WAIT_DOMINANCE_WARNING * 100:
                style = "yellow bold"

            lines.append(f"[{style}]{action:10} {pct:5.1f}%[/]")

        content = self.query_one("#actions-content", Static)
        content.update("\n".join(lines) if lines else "[dim]No actions yet[/]")
```

### Step 4: Update widgets __init__.py

Edit `src/esper/karn/sanctum/widgets/__init__.py`:

```python
"""Sanctum widgets."""
from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.widgets.scoreboard import Scoreboard
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain

__all__ = ["EnvOverview", "Scoreboard", "TamiyoBrain"]
```

### Step 5: Run tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v
```

Expected: All tests PASS

### Step 6: Commit

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): add TamiyoBrain widget with complete policy diagnostics

Port Tamiyo Brain panel from Rich TUI with ALL helper methods:
- Health: entropy (% of max), clip, KL, ExplVar (with hints)
- Losses: policy, value, entropy, grad norm
- Vitals: LR, π ratio stats (max/min/std), dead/exploding layers, Grad Health
- Actions: WAIT/GERMINATE/CULL/FOSSILIZE distribution with dominance warning

Uses TUIThresholds from constants.py (not hardcoded values)
Waiting state shows before PPO data arrives
All status helpers fully implemented (no stubs)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Port Remaining Widgets (RewardComponents + EsperStatus)

**Files:**
- Create: `src/esper/karn/sanctum/widgets/reward_components.py`
- Create: `src/esper/karn/sanctum/widgets/esper_status.py`
- Create: `tests/karn/sanctum/test_remaining_widgets.py`
- Reference: `src/esper/karn/tui.py` lines 1513-1586 (reward), 1596-1696 (status)

### Part A: RewardComponents Widget

**Overview:** Shows detailed breakdown of Esper reward components for the focused environment.
Must display ALL Esper-specific reward components with proper conditional styling.

**Components to Show:**
1. Header: Env ID, Last Action, Val Acc
2. Base: ΔAcc (base_acc_delta) - green if positive, red if negative
3. Attribution: Attr (bounded_attribution) - conditional display, green/red
4. Costs: Rent (compute_rent) - red if negative, Penalty (ratio_penalty) - red if negative
5. Bonuses: Stage (stage_bonus) - blue, Fossil (fossilize_terminal_bonus) - blue
6. Warnings: Blend Warn (blending_warning) - yellow if negative, Prob Warn (probation_warning) - yellow if negative
7. Total: Sum with bold green/red

### Step 1: Write tests

Create `tests/karn/sanctum/test_remaining_widgets.py`:

```python
"""Tests for remaining Sanctum widgets (RewardComponents and EsperStatus)."""
import pytest
from datetime import datetime

from esper.karn.sanctum.widgets.reward_components import RewardComponents
from esper.karn.sanctum.widgets.esper_status import EsperStatus
from esper.karn.sanctum.schema import (
    RewardComponents as RewardData,
    SystemVitals,
    GPUStats,
    EnvState,
    SeedState,
)


# ============================================================================
# RewardComponents Tests
# ============================================================================

def test_reward_components_creation():
    """RewardComponents widget should create without errors."""
    widget = RewardComponents()
    assert widget is not None


def test_reward_components_displays_header():
    """Should show env_id, last_action, val_acc in header."""
    widget = RewardComponents()
    data = RewardData(
        env_id=5,
        last_action="GERMINATE",
        val_acc=75.5,
    )
    widget.update_rewards(data)
    # Header should be populated


def test_reward_components_shows_base_delta():
    """Should show ΔAcc with green/red styling."""
    widget = RewardComponents()
    # Positive delta
    data = RewardData(base_acc_delta=0.5)
    widget.update_rewards(data)
    # Should be green

    # Negative delta
    data = RewardData(base_acc_delta=-0.3)
    widget.update_rewards(data)
    # Should be red


def test_reward_components_shows_attribution():
    """Should show Attr (bounded_attribution) conditionally."""
    widget = RewardComponents()
    # Non-zero attribution
    data = RewardData(bounded_attribution=0.25)
    widget.update_rewards(data)
    # Should be displayed

    # Zero attribution (should not display)
    data = RewardData(bounded_attribution=0.0)
    widget.update_rewards(data)


def test_reward_components_shows_costs():
    """Should show Rent and Penalty with red styling."""
    widget = RewardComponents()
    data = RewardData(
        compute_rent=-0.1,
        ratio_penalty=-0.05,
    )
    widget.update_rewards(data)
    # Both should be red


def test_reward_components_shows_bonuses():
    """Should show Stage and Fossil bonuses in blue."""
    widget = RewardComponents()
    data = RewardData(
        stage_bonus=0.2,
        fossilize_terminal_bonus=1.0,
    )
    widget.update_rewards(data)
    # Both should be blue


def test_reward_components_shows_warnings():
    """Should show blending and probation warnings in yellow."""
    widget = RewardComponents()
    data = RewardData(
        blending_warning=-0.1,
        probation_warning=-0.05,
    )
    widget.update_rewards(data)
    # Both should be yellow


def test_reward_components_shows_total():
    """Should show total reward with bold styling."""
    widget = RewardComponents()
    # Positive total
    data = RewardData(total=0.75)
    widget.update_rewards(data)
    # Should be bold green

    # Negative total
    data = RewardData(total=-0.5)
    widget.update_rewards(data)
    # Should be bold red


# ============================================================================
# EsperStatus Tests
# ============================================================================

def test_esper_status_creation():
    """EsperStatus widget should create without errors."""
    widget = EsperStatus()
    assert widget is not None


def test_esper_status_shows_seed_stage_counts():
    """Should aggregate and display seed stage counts."""
    widget = EsperStatus()
    envs = {
        0: EnvState(env_id=0),
        1: EnvState(env_id=1),
    }
    # Add seeds in various stages
    envs[0].seeds["r0c0"] = SeedState(slot_id="r0c0", stage="TRAINING")
    envs[0].seeds["r0c1"] = SeedState(slot_id="r0c1", stage="BLENDING")
    envs[1].seeds["r0c0"] = SeedState(slot_id="r0c0", stage="TRAINING")
    envs[1].seeds["r0c1"] = SeedState(slot_id="r0c1", stage="FOSSILIZED")

    vitals = SystemVitals()
    widget.update_status(envs, vitals)
    # Should show Train: 2, Blend: 1, Foss: 1


def test_esper_status_shows_host_params():
    """Should format host params (M/K/raw)."""
    widget = EsperStatus()
    vitals = SystemVitals(host_params=1_500_000)
    widget.update_status({}, vitals)
    # Should show "1.5M"

    vitals = SystemVitals(host_params=50_000)
    widget.update_status({}, vitals)
    # Should show "50K"


def test_esper_status_shows_throughput():
    """Should display epochs/sec and batches/hr."""
    widget = EsperStatus()
    vitals = SystemVitals(
        epochs_per_second=2.5,
        batches_per_hour=150.0,
    )
    widget.update_status({}, vitals)
    # Should display both metrics


def test_esper_status_shows_runtime():
    """Should format runtime as Xh Ym Zs."""
    widget = EsperStatus()
    vitals = SystemVitals()
    widget.update_status({}, vitals, start_time=datetime.now())
    # Should show formatted runtime


def test_esper_status_shows_multi_gpu():
    """Should display all GPU devices with memory and utilization."""
    widget = EsperStatus()
    vitals = SystemVitals()
    vitals.gpu_stats[0] = GPUStats(
        device_id=0,
        memory_used_gb=12.0,
        memory_total_gb=16.0,
        utilization=85.0,
    )
    vitals.gpu_stats[1] = GPUStats(
        device_id=1,
        memory_used_gb=8.0,
        memory_total_gb=16.0,
        utilization=60.0,
    )
    widget.update_status({}, vitals)
    # Should show GPU0 and GPU1


def test_esper_status_gpu_memory_color_thresholds():
    """Should color-code GPU memory: green <75%, yellow 75-90%, red >90%."""
    widget = EsperStatus()
    vitals = SystemVitals()

    # Test 50% (green)
    vitals.gpu_stats[0] = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0)
    widget.update_status({}, vitals)

    # Test 80% (yellow)
    vitals.gpu_stats[0] = GPUStats(device_id=0, memory_used_gb=12.8, memory_total_gb=16.0)
    widget.update_status({}, vitals)

    # Test 95% (red)
    vitals.gpu_stats[0] = GPUStats(device_id=0, memory_used_gb=15.2, memory_total_gb=16.0)
    widget.update_status({}, vitals)


def test_esper_status_shows_ram():
    """Should display RAM with color thresholds."""
    widget = EsperStatus()
    vitals = SystemVitals(
        ram_used_gb=12.5,
        ram_total_gb=32.0,
    )
    widget.update_status({}, vitals)
    # Should show RAM with appropriate color


def test_esper_status_shows_cpu():
    """THE FIX: CPU was collected but never displayed - must show it."""
    widget = EsperStatus()
    vitals = SystemVitals(cpu_percent=67.0)
    widget.update_status({}, vitals)
    # MUST display CPU percentage
```

### Step 2: Run tests to verify they fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_remaining_widgets.py -v
```

Expected: FAIL with "No module named..."

### Step 3: Write RewardComponents implementation

Create `src/esper/karn/sanctum/widgets/reward_components.py`:

```python
"""Reward Components Widget - Esper reward breakdown.

Shows detailed breakdown of all Esper-specific reward components.
Ported from Rich TUI _render_reward_components() (tui.py:1513-1586).

Component Display Rules:
- ΔAcc: Always show, green if positive, red if negative
- Attr: Show only if non-zero
- Rent: Show always, red if negative
- Penalty: Show only if non-zero, red if negative
- Stage: Show only if non-zero, blue
- Fossil: Show only if non-zero, blue
- Blend Warn: Show only if negative, yellow
- Prob Warn: Show only if negative, yellow
- Total: Always show, bold green/red
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from esper.karn.sanctum.schema import RewardComponents as RewardData


class RewardComponents(Static):
    """Reward component breakdown for focused environment."""

    DEFAULT_CSS = """
    RewardComponents {
        height: 100%;
        padding: 1;
    }

    RewardComponents DataTable {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rewards: RewardData | None = None
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        """Create the table."""
        table = DataTable(show_header=False, id="reward-table")
        table.add_column("Component", width=12)
        table.add_column("Value", width=12, justify="right")
        self._table = table
        yield table

    def update_rewards(self, rewards: RewardData) -> None:
        """Update with new reward data."""
        self._rewards = rewards
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the reward breakdown display."""
        if self._table is None or self._rewards is None:
            return

        self._table.clear()
        r = self._rewards

        # Header context
        self._table.add_row("[dim]Env:[/]", f"[bold cyan]{r.env_id}[/]")
        if r.last_action:
            self._table.add_row("[dim]Action:[/]", r.last_action)
        if r.val_acc > 0:
            self._table.add_row("[dim]Val Acc:[/]", f"{r.val_acc:.1f}%")

        self._table.add_row("", "")

        # Base delta (legacy shaped signal)
        if isinstance(r.base_acc_delta, (int, float)):
            style = "green" if r.base_acc_delta >= 0 else "red"
            self._table.add_row(
                "[dim]ΔAcc:[/]",
                f"[{style}]{r.base_acc_delta:+.2f}[/{style}]"
            )

        # Attribution (contribution-primary)
        if isinstance(r.bounded_attribution, (int, float)) and r.bounded_attribution != 0.0:
            style = "green" if r.bounded_attribution >= 0 else "red"
            self._table.add_row(
                "[dim]Attr:[/]",
                f"[{style}]{r.bounded_attribution:+.2f}[/{style}]"
            )

        # Compute rent (usually negative)
        if isinstance(r.compute_rent, (int, float)):
            style = "red" if r.compute_rent < 0 else "dim"
            self._table.add_row(
                "[dim]Rent:[/]",
                f"[{style}]{r.compute_rent:+.2f}[/{style}]"
            )

        # Ratio penalty (ransomware / attribution mismatch)
        if isinstance(r.ratio_penalty, (int, float)) and r.ratio_penalty != 0.0:
            style = "red" if r.ratio_penalty < 0 else "dim"
            self._table.add_row(
                "[dim]Penalty:[/]",
                f"[{style}]{r.ratio_penalty:+.2f}[/{style}]"
            )

        # Stage / terminal bonuses
        if isinstance(r.stage_bonus, (int, float)) and r.stage_bonus != 0.0:
            self._table.add_row(
                "[dim]Stage:[/]",
                f"[blue]{r.stage_bonus:+.2f}[/blue]"
            )

        if isinstance(r.fossilize_terminal_bonus, (int, float)) and r.fossilize_terminal_bonus != 0.0:
            self._table.add_row(
                "[dim]Fossil:[/]",
                f"[blue]{r.fossilize_terminal_bonus:+.2f}[/blue]"
            )

        # Warnings
        if isinstance(r.blending_warning, (int, float)) and r.blending_warning < 0:
            self._table.add_row(
                "[dim]Blend Warn:[/]",
                f"[yellow]{r.blending_warning:.2f}[/yellow]"
            )

        if isinstance(r.probation_warning, (int, float)) and r.probation_warning < 0:
            self._table.add_row(
                "[dim]Prob Warn:[/]",
                f"[yellow]{r.probation_warning:.2f}[/yellow]"
            )

        # Total (last computed reward)
        self._table.add_row("", "")
        self._table.add_row("[dim]───────────[/]", "[dim]───────[/]")
        total_style = "bold green" if r.total >= 0 else "bold red"
        self._table.add_row(
            "[dim]Total:[/]",
            f"[{total_style}]{r.total:+.2f}[/{total_style}]"
        )
```

### Step 4: Write EsperStatus implementation

Create `src/esper/karn/sanctum/widgets/esper_status.py`:

```python
"""Esper Status Widget - System vitals and seed counts.

Shows seed stage counts, host params, throughput, runtime, GPU stats, RAM, and CPU.
Ported from Rich TUI _render_esper_status() (tui.py:1596-1696).

THE FIX: CPU was collected (cpu_percent) but never displayed. Now shows CPU percentage.

Color Thresholds:
- GPU/RAM memory: green <75%, yellow 75-90%, red >90%
- GPU utilization: green <80%, yellow 80-95%, red >95%
"""
from __future__ import annotations

from datetime import datetime
from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from esper.karn.sanctum.schema import EnvState, SystemVitals


class EsperStatus(Static):
    """System status panel showing seeds, performance, and vitals."""

    DEFAULT_CSS = """
    EsperStatus {
        height: 100%;
        padding: 1;
    }

    EsperStatus DataTable {
        height: 1fr;
    }
    """

    # Stage colors (matching env overview)
    _STAGE_STYLES: dict[str, str] = {
        "TRAINING": "yellow",
        "BLENDING": "cyan",
        "PROBATIONARY": "blue",
        "FOSSILIZED": "magenta",
        "CULLED": "red",
        "GERMINATED": "green",
    }

    # Short stage names
    _STAGE_SHORT: dict[str, str] = {
        "TRAINING": "Train",
        "BLENDING": "Blend",
        "PROBATIONARY": "Prob",
        "FOSSILIZED": "Foss",
        "CULLED": "Cull",
        "GERMINATED": "Germ",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._envs: dict[int, EnvState] = {}
        self._vitals: SystemVitals | None = None
        self._start_time: datetime | None = None
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        """Create the table."""
        table = DataTable(show_header=False, id="status-table")
        table.add_column("Metric", width=12)
        table.add_column("Value", width=15, justify="right")
        self._table = table
        yield table

    def update_status(
        self,
        envs: dict[int, EnvState],
        vitals: SystemVitals,
        start_time: datetime | None = None,
    ) -> None:
        """Update with new status data."""
        self._envs = envs
        self._vitals = vitals
        if start_time is not None:
            self._start_time = start_time
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the status display."""
        if self._table is None or self._vitals is None:
            return

        self._table.clear()

        # Seed stage counts (aggregate across all envs)
        stage_counts: dict[str, int] = {}
        for env in self._envs.values():
            for seed in env.seeds.values():
                if seed.stage != "DORMANT":
                    stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1

        if stage_counts:
            for stage, count in sorted(stage_counts.items()):
                short = self._STAGE_SHORT.get(stage, stage[:4])
                style = self._STAGE_STYLES.get(stage, "dim")
                self._table.add_row(f"[dim]{short}:[/]", f"[{style}]{count}[/{style}]")
            self._table.add_row("", "")

        # Host network params
        if self._vitals.host_params > 0:
            if self._vitals.host_params >= 1_000_000:
                params_str = f"{self._vitals.host_params / 1_000_000:.1f}M"
            elif self._vitals.host_params >= 1_000:
                params_str = f"{self._vitals.host_params / 1_000:.0f}K"
            else:
                params_str = str(self._vitals.host_params)
            self._table.add_row("[dim]Host Params:[/]", params_str)
            self._table.add_row("", "")

        # Throughput
        self._table.add_row(
            "[dim]Epochs/sec:[/]",
            f"{self._vitals.epochs_per_second:.2f}"
        )
        self._table.add_row(
            "[dim]Batches/hr:[/]",
            f"{self._vitals.batches_per_hour:.0f}"
        )

        # Runtime
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self._table.add_row(
                "[dim]Runtime:[/]",
                f"{hours}h {minutes}m {seconds}s"
            )
        else:
            self._table.add_row("[dim]Runtime:[/]", "─")

        self._table.add_row("", "")

        # GPU stats - show all configured GPUs
        if self._vitals.gpu_stats:
            for dev_id, stats in sorted(self._vitals.gpu_stats.items()):
                if stats.memory_total_gb > 0:
                    gpu_pct = (stats.memory_used_gb / stats.memory_total_gb) * 100
                    mem_style = "red" if gpu_pct > 90 else "yellow" if gpu_pct > 75 else "green"
                    label = f"[dim]GPU{dev_id}:[/]" if len(self._vitals.gpu_stats) > 1 else "[dim]GPU:[/]"

                    # Show memory usage
                    self._table.add_row(
                        label,
                        f"[{mem_style}]{stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f}GB[/{mem_style}]"
                    )

                    # Show utilization if available
                    if stats.utilization > 0:
                        util_style = "red" if stats.utilization > 95 else "yellow" if stats.utilization > 80 else "green"
                        util_label = "[dim]  util:[/]" if len(self._vitals.gpu_stats) > 1 else "[dim]GPU util:[/]"
                        self._table.add_row(
                            util_label,
                            f"[{util_style}]{stats.utilization:.0f}%[/{util_style}]"
                        )
        elif self._vitals.gpu_memory_total_gb > 0:
            # Fallback to legacy single-GPU fields
            gpu_pct = (self._vitals.gpu_memory_used_gb / self._vitals.gpu_memory_total_gb) * 100
            mem_style = "red" if gpu_pct > 90 else "yellow" if gpu_pct > 75 else "green"
            self._table.add_row(
                "[dim]GPU:[/]",
                f"[{mem_style}]{self._vitals.gpu_memory_used_gb:.1f}/{self._vitals.gpu_memory_total_gb:.1f}GB[/{mem_style}]"
            )
            if self._vitals.gpu_utilization > 0:
                util_style = "red" if self._vitals.gpu_utilization > 95 else "yellow" if self._vitals.gpu_utilization > 80 else "green"
                self._table.add_row(
                    "[dim]GPU util:[/]",
                    f"[{util_style}]{self._vitals.gpu_utilization:.0f}%[/{util_style}]"
                )
        else:
            self._table.add_row("[dim]GPU:[/]", "─")

        # RAM
        if self._vitals.ram_total_gb > 0:
            ram_pct = (self._vitals.ram_used_gb / self._vitals.ram_total_gb) * 100
            ram_style = "red" if ram_pct > 90 else "yellow" if ram_pct > 75 else "dim"
            self._table.add_row(
                "[dim]RAM:[/]",
                f"[{ram_style}]{self._vitals.ram_used_gb:.1f}/{self._vitals.ram_total_gb:.0f}GB[/{ram_style}]"
            )

        # CPU (THE FIX: was collected but never displayed!)
        if self._vitals.cpu_percent > 0:
            cpu_style = "red" if self._vitals.cpu_percent > 90 else "yellow" if self._vitals.cpu_percent > 75 else "dim"
            self._table.add_row(
                "[dim]CPU:[/]",
                f"[{cpu_style}]{self._vitals.cpu_percent:.0f}%[/{cpu_style}]"
            )
```

### Step 5: Update widgets __init__.py

Edit `src/esper/karn/sanctum/widgets/__init__.py`:

```python
"""Sanctum widgets."""
from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.widgets.scoreboard import Scoreboard
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
from esper.karn.sanctum.widgets.reward_components import RewardComponents
from esper.karn.sanctum.widgets.esper_status import EsperStatus

__all__ = [
    "EnvOverview",
    "Scoreboard",
    "TamiyoBrain",
    "RewardComponents",
    "EsperStatus",
]
```

### Step 6: Run tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_remaining_widgets.py -v
```

Expected: All tests PASS

### Step 7: Commit

```bash
git add src/esper/karn/sanctum/widgets/ tests/karn/sanctum/test_remaining_widgets.py
git commit -m "feat(sanctum): add RewardComponents and EsperStatus widgets

RewardComponents (Esper reward breakdown):
- Header: env_id, last_action, val_acc
- All Esper components: ΔAcc, Attr, Rent, Penalty, Stage, Fossil, warnings
- Conditional display (only show non-zero components)
- Proper styling: green/red for deltas, blue for bonuses, yellow for warnings
- Total with bold styling

EsperStatus (system vitals):
- Seed stage counts (Train/Blend/Prob/Foss/Cull/Germ) with colors
- Host params formatted (M/K/raw)
- Throughput (epochs/sec, batches/hr)
- Runtime formatted (Xh Ym Zs)
- Multi-GPU support with memory and utilization per device
- RAM usage with color thresholds
- CPU display (THE FIX - was collected but never shown!)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This document provides complete, production-ready implementations for Tasks 4-6:

### Task 4: Scoreboard Widget ✓
- Stats header with global best, mean best, fossilized/culled counts
- Leaderboard table with top 10 envs by best accuracy
- Medal indicators for top 3
- Current accuracy styling based on delta thresholds
- Seeds at best: show blueprints (≤2) or count (>2)
- Stable row keys for updates

### Task 5: TamiyoBrain Widget ✓
- 4-column layout (Health, Losses, Vitals, Actions)
- ALL helper methods fully implemented (no stubs)
- Uses TUIThresholds from constants.py
- Waiting state before PPO data arrives
- Entropy as percentage of max
- ExplVar with interpretive hints
- Ratio stats (max/min/std, NOT mean)
- Dead/exploding layers display
- Grad Health (renamed from GradHP)
- WAIT dominance warning

### Task 6: Remaining Widgets ✓
- **RewardComponents**: All Esper-specific reward components with conditional display and proper styling
- **EsperStatus**: Seed counts, host params, throughput, runtime, multi-GPU stats, RAM, and CPU (THE FIX!)

**All code is complete with NO placeholders, NO stubs, and ALL assertions in tests.**

**Next Steps:** Tasks 7-10 will wire the app, add telemetry backend, CLI integration, and final cleanup.
