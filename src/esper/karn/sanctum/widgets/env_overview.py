"""EnvOverview widget - Per-environment overview table.

This is a TABLE, NOT a card grid. Direct port of tui.py _render_env_overview().
Shows a row per environment with metrics, slot states, and status.

Reference: src/esper/karn/tui.py lines 1777-1959 (_render_env_overview method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import DataTable, Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EnvState, SanctumSnapshot, SeedState


# A/B test cohort styling - colored pips for reward modes
_AB_STYLES: dict[str, tuple[str, str]] = {
    "shaped": ("●", "bright_blue"),      # Blue pip for shaped
    "simplified": ("●", "bright_yellow"), # Yellow pip for simplified
    "sparse": ("●", "bright_cyan"),       # Cyan pip for sparse
}


class EnvOverview(Static):
    """Per-environment overview table.

    Shows a row per environment with:
    - Env ID (with A/B test cohort pip)
    - Accuracy (with color coding)
    - Reward (current and avg)
    - Sparklines for accuracy and reward history
    - Reward component breakdown (ΔAcc, Seed Δ, Rent)
    - Dynamic slot columns (stage:blueprint with gradient indicators)
    - Last action taken
    - Status (with color coding)

    CRITICAL FIXES:
    1. Accuracy: Green if at best, yellow if stagnant >5 epochs
    2. Reward: >0 green, <-0.5 red, else white
    3. A/B cohort: Colored pip (●) next to env ID based on reward_mode
    """

    def __init__(self, num_envs: int = 16, **kwargs) -> None:
        """Initialize EnvOverview widget.

        Args:
            num_envs: Number of training environments.
        """
        super().__init__(**kwargs)
        self._num_envs = num_envs
        self.table = DataTable(zebra_stripes=True, cursor_type="row")
        self._snapshot: SanctumSnapshot | None = None

    def compose(self):
        """Compose the widget."""
        yield self.table

    def on_mount(self) -> None:
        """Setup table columns on mount."""
        self._setup_columns()

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update table with new snapshot data."""
        self._snapshot = snapshot
        self._setup_columns()
        self._refresh_table()

    def _setup_columns(self) -> None:
        """Setup table columns based on snapshot."""
        if self._snapshot is None:
            return

        self.table.clear(columns=True)

        # Fixed columns
        self.table.add_column("Env", key="env")
        self.table.add_column("Acc", key="acc")
        self.table.add_column("Reward", key="reward")
        self.table.add_column("Acc▁▃▅", key="acc_spark")
        self.table.add_column("Rwd▁▃▅", key="rwd_spark")
        self.table.add_column("ΔAcc", key="delta_acc")
        self.table.add_column("Seed Δ", key="seed_delta")
        self.table.add_column("Rent", key="rent")

        # Dynamic slot columns
        for slot_id in self._snapshot.slot_ids:
            self.table.add_column(slot_id, key=f"slot_{slot_id}")

        # Last action and status
        self.table.add_column("Last", key="last")
        self.table.add_column("Status", key="status")

    def _refresh_table(self) -> None:
        """Refresh table rows with current snapshot data."""
        if self._snapshot is None:
            return

        # Clear existing rows
        self.table.clear()

        # Sort envs by ID for stable display
        sorted_envs = sorted(
            self._snapshot.envs.values(),
            key=lambda e: e.env_id
        )

        # Add row for each environment
        for env in sorted_envs:
            self._add_env_row(env)

        # Add aggregate row if multiple envs
        if len(self._snapshot.envs) > 1:
            self._add_separator_row()
            self._add_aggregate_row()

    def _add_env_row(self, env: "EnvState") -> None:
        """Add a single environment row."""
        # Env ID with A/B test cohort pip
        env_id_cell = self._format_env_id(env)

        # Accuracy with color coding
        acc_cell = self._format_accuracy(env)

        # Reward (current and average)
        reward_cell = self._format_reward(env)

        # Sparklines
        acc_spark = self._make_sparkline(env.accuracy_history)
        rwd_spark = self._make_sparkline(env.reward_history)

        # Reward components
        delta_acc_cell = self._format_delta_acc(env)
        seed_delta_cell = self._format_seed_delta(env)
        rent_cell = self._format_rent(env)

        # Dynamic slot cells
        slot_cells = []
        for slot_id in self._snapshot.slot_ids:
            slot_cells.append(self._format_slot_cell(env, slot_id))

        # Last action
        last_action = self._format_last_action(env)

        # Status
        status_cell = self._format_status(env)

        # Build row
        row = [
            env_id_cell,
            acc_cell,
            reward_cell,
            acc_spark,
            rwd_spark,
            delta_acc_cell,
            seed_delta_cell,
            rent_cell,
        ] + slot_cells + [
            last_action,
            status_cell,
        ]

        # Add row with key=env_id for row selection event handling
        self.table.add_row(*row, key=str(env.env_id))

    def _add_separator_row(self) -> None:
        """Add separator row before aggregate."""
        num_cols = len(self.table.columns)
        separator = ["─" * 2] * num_cols
        self.table.add_row(*separator)

    def _add_aggregate_row(self) -> None:
        """Add aggregate row at bottom."""
        if self._snapshot is None or not self._snapshot.envs:
            return

        # Calculate aggregates (reward_components is now a RewardComponents dataclass)
        deltas = [
            float(e.reward_components.base_acc_delta)
            for e in self._snapshot.envs.values()
            if isinstance(e.reward_components.base_acc_delta, (int, float))
        ]
        rents = [
            float(e.reward_components.compute_rent)
            for e in self._snapshot.envs.values()
            if isinstance(e.reward_components.compute_rent, (int, float))
        ]

        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        mean_rent = sum(rents) / len(rents) if rents else 0.0

        # Build aggregate row
        agg_row = [
            "[bold]Σ[/bold]",
            f"[bold]{self._snapshot.aggregate_mean_accuracy:.1f}%[/bold]",
            f"[bold]{self._snapshot.aggregate_mean_reward:+.2f}[/bold]",
            "",  # Acc sparkline
            "",  # Rwd sparkline
            f"[dim]{mean_delta:+.2f}[/dim]" if deltas else "─",
            "─",  # Seed Δ aggregate not meaningful
            f"[dim]{mean_rent:.2f}[/dim]" if rents else "─",
        ]

        # Empty slot columns
        for _ in self._snapshot.slot_ids:
            agg_row.append("")

        # Last action column (empty)
        agg_row.append("")

        # Status column shows best accuracy from any env
        best_acc = max((e.best_accuracy for e in self._snapshot.envs.values()), default=0.0)
        agg_row.append(f"[dim]{best_acc:.1f}%[/dim]")

        self.table.add_row(*agg_row)

    def _format_env_id(self, env: "EnvState") -> str:
        """Format env ID with A/B test cohort pip."""
        if env.reward_mode and env.reward_mode in _AB_STYLES:
            pip, color = _AB_STYLES[env.reward_mode]
            return f"[{color}]{pip}[/{color}]{env.env_id}"
        return str(env.env_id)

    def _format_accuracy(self, env: "EnvState") -> str:
        """Format accuracy with color coding.

        Green if at best, yellow if stagnant >5 epochs.
        """
        acc_str = f"{env.host_accuracy:.1f}%"

        if env.best_accuracy > 0:
            if env.host_accuracy >= env.best_accuracy:
                return f"[green]{acc_str}[/green]"
            elif env.epochs_since_improvement > 5:
                return f"[yellow]{acc_str}[/yellow]"

        return acc_str

    def _format_reward(self, env: "EnvState") -> str:
        """Format reward (current and average).

        >0 green, <-0.5 red, else white.
        """
        reward_val = env.current_reward
        mean_val = env.mean_reward

        # Color based on current reward
        if reward_val > 0:
            r_style = "green"
        elif reward_val < -0.5:
            r_style = "red"
        else:
            r_style = "white"

        return f"[{r_style}]{reward_val:+.2f}[/{r_style}] [dim]({mean_val:+.2f})[/dim]"

    def _format_delta_acc(self, env: "EnvState") -> str:
        """Format base accuracy delta component."""
        # env.reward_components is now a RewardComponents dataclass
        base_delta = env.reward_components.base_acc_delta
        if isinstance(base_delta, (int, float)):
            style = "green" if float(base_delta) >= 0 else "red"
            return f"[{style}]{float(base_delta):+.2f}[/{style}]"
        return "─"

    def _format_seed_delta(self, env: "EnvState") -> str:
        """Format seed contribution component."""
        # env.reward_components is now a RewardComponents dataclass
        seed_contrib = env.reward_components.seed_contribution
        bounded_attr = env.reward_components.bounded_attribution

        if isinstance(seed_contrib, (int, float)) and seed_contrib != 0:
            style = "green" if seed_contrib > 0 else "red"
            return f"[{style}]{seed_contrib:+.1f}%[/{style}]"
        elif isinstance(bounded_attr, (int, float)) and bounded_attr != 0:
            style = "green" if bounded_attr > 0 else "red"
            return f"[{style}]{bounded_attr:+.2f}[/{style}]"

        return "─"

    def _format_rent(self, env: "EnvState") -> str:
        """Format compute rent component."""
        # env.reward_components is now a RewardComponents dataclass
        compute_rent = env.reward_components.compute_rent
        if isinstance(compute_rent, (int, float)) and compute_rent != 0:
            return f"[red]{compute_rent:.2f}[/red]"
        return "─"

    def _format_slot_cell(self, env: "EnvState", slot_id: str) -> str:
        """Format slot cell with stage:blueprint and gradient indicators.

        Format:
        - BLENDING: "Blend:conv_l 0.3" (shows alpha)
        - TRAINING: "Train:dense_m e5" (shows epochs)
        - Gradient health: ▼ (vanishing), ▲ (exploding)
        - DORMANT: "─"
        """
        seed = env.seeds.get(slot_id)
        if not seed or seed.stage == "DORMANT":
            return "─"

        # Stage abbreviations
        stage_short = {
            "TRAINING": "Train",
            "BLENDING": "Blend",
            "PROBATIONARY": "Prob",
            "FOSSILIZED": "Foss",
            "CULLED": "Cull",
        }.get(seed.stage, seed.stage[:5])

        blueprint = seed.blueprint_id or "?"
        if len(blueprint) > 6:
            blueprint = blueprint[:6]

        # Stage-specific styling
        style_map = {
            "TRAINING": "cyan",
            "BLENDING": "yellow",
            "PROBATIONARY": "magenta",
            "FOSSILIZED": "green",
            "CULLED": "red",
        }
        style = style_map.get(seed.stage, "white")

        # Gradient health indicator
        grad_indicator = ""
        if seed.has_exploding:
            grad_indicator = "[red]▲[/red]"
        elif seed.has_vanishing:
            grad_indicator = "[yellow]▼[/yellow]"

        # BLENDING shows alpha
        if seed.stage == "BLENDING" and seed.alpha > 0:
            base = f"[{style}]{stage_short}:{blueprint} {seed.alpha:.1f}[/{style}]"
            return f"{base}{grad_indicator}" if grad_indicator else base

        # Active seeds show epochs in stage
        epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
        base = f"[{style}]{stage_short}:{blueprint}{epochs_str}[/{style}]"
        return f"{base}{grad_indicator}" if grad_indicator else base

    def _format_last_action(self, env: "EnvState") -> str:
        """Format last action taken."""
        if not env.action_history:
            return "—"

        last_action = env.action_history[-1]

        # Shorten action names for compact display
        action_short = {
            "WAIT": "WAIT",
            "GERMINATE": "GERM",
            "FOSSILIZE": "FOSS",
            "CULL": "CULL",
        }.get(last_action, last_action[:4] if last_action else "—")

        return action_short

    def _format_status(self, env: "EnvState") -> str:
        """Format status with color coding."""
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }

        status_short = {
            "excellent": "EXCL",
            "healthy": "OK",
            "initializing": "INIT",
            "stalled": "STAL",
            "degraded": "DEGR",
        }.get(env.status, env.status[:4].upper())

        status_style = status_styles.get(env.status, "white")
        status_str = f"[{status_style}]{status_short}[/{status_style}]"

        # Show epochs since improvement if stagnating (>5 epochs)
        if env.epochs_since_improvement > 5:
            stale_style = "red" if env.epochs_since_improvement > 15 else "yellow"
            status_str += f"[{stale_style}]({env.epochs_since_improvement})[/{stale_style}]"

        return status_str

    def _make_sparkline(self, values, width: int = 8) -> str:
        """Create sparkline from values using schema module."""
        from esper.karn.sanctum.schema import make_sparkline
        return make_sparkline(values, width)
