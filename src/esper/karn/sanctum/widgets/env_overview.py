"""EnvOverview widget - Per-environment overview table.

This is a TABLE, NOT a card grid. Direct port of tui.py _render_env_overview().
Shows a row per environment with metrics, slot states, and status.

Reference: src/esper/karn/tui.py lines 1777-1959 (_render_env_overview method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from textual.widgets import DataTable, Static

from esper.leyline import (
    ALPHA_CURVE_GLYPHS,
    DEFAULT_GROWTH_RATIO_GREEN_MAX,
    DEFAULT_GROWTH_RATIO_YELLOW_MAX,
    STAGE_COLORS,
)

if TYPE_CHECKING:
    from datetime import datetime

    from esper.karn.sanctum.schema import EnvState, SanctumSnapshot


# A/B test cohort styling - colored pips for reward modes
# Note: cyan reserved for informational data; sparse uses white for distinction
_AB_STYLES: dict[str, tuple[str, str]] = {
    "shaped": ("●", "bright_blue"),      # Blue pip for shaped
    "simplified": ("●", "bright_yellow"), # Yellow pip for simplified
    "sparse": ("●", "bright_white"),      # White pip for sparse (cyan reserved for info)
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

    def __init__(self, num_envs: int = 16, **kwargs: Any) -> None:
        """Initialize EnvOverview widget.

        Args:
            num_envs: Number of training environments.
        """
        super().__init__(**kwargs)
        self._num_envs = num_envs
        self.table: DataTable[Any] = DataTable[Any](zebra_stripes=True, cursor_type="row")
        self._snapshot: SanctumSnapshot | None = None
        self._current_slot_ids: list[str] = []  # Track slot_ids to detect column changes
        self._filter_text: str = ""  # Filter text for env rows

    def set_filter(self, text: str) -> None:
        """Set filter text and refresh display.

        Filter matches:
        - Env ID (number): "3" matches env 3
        - Status: "stall" matches stalled envs
        - Empty string: shows all envs

        Args:
            text: Filter text.
        """
        self._filter_text = text.lower().strip()
        self._refresh_table()

    def _matches_filter(self, env: "EnvState") -> bool:
        """Check if env matches current filter.

        Args:
            env: Environment state to check.

        Returns:
            True if env matches filter or filter is empty.
        """
        if not self._filter_text:
            return True

        # Match by env ID
        if self._filter_text.isdigit():
            return str(env.env_id) == self._filter_text

        # Match by status
        if self._filter_text in env.status.lower():
            return True

        return False

    def compose(self) -> Iterator[DataTable[Any]]:
        """Compose the widget."""
        yield self.table

    def on_mount(self) -> None:
        """Setup table columns on mount."""
        self._setup_columns()

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update table with new snapshot data.

        Only rebuilds columns if slot_ids changed. Preserves cursor position.
        """
        self._snapshot = snapshot

        # Only rebuild columns if slot_ids changed
        if snapshot.slot_ids != self._current_slot_ids:
            self._setup_columns()
            self._current_slot_ids = list(snapshot.slot_ids)

        self._refresh_table()

    def _setup_columns(self) -> None:
        """Setup table columns based on snapshot."""
        self.table.clear(columns=True)
        slot_ids = self._snapshot.slot_ids if self._snapshot is not None else self._current_slot_ids

        # Fixed columns - ordered: Identity → Performance → Trends → Reward breakdown
        self.table.add_column("Env", key="env")
        self.table.add_column("Acc", key="acc")
        self.table.add_column("∑Rwd", key="cum_rwd")  # Cumulative reward for episode
        self.table.add_column("Loss", key="loss")  # Host loss (for overfitting detection)
        self.table.add_column("CF", key="cf")  # Counterfactual: synergy/interference
        self.table.add_column("Growth", key="growth")  # growth_ratio: (host+foss)/host
        self.table.add_column("Reward", key="reward")
        self.table.add_column("Acc▁▃▅", key="acc_spark")
        self.table.add_column("Rwd▁▃▅", key="rwd_spark")
        self.table.add_column("ΔAcc", key="delta_acc")
        self.table.add_column("Seed Δ", key="seed_delta")
        self.table.add_column("Rent", key="rent")

        # Dynamic slot columns
        for slot_id in slot_ids:
            self.table.add_column(slot_id, key=f"slot_{slot_id}")

        # Last action, momentum, status, and telemetry staleness
        self.table.add_column("Last", key="last")
        self.table.add_column("Momentum", key="momentum")
        self.table.add_column("Status", key="status")
        self.table.add_column("Stale", key="stale")

    def _refresh_table(self) -> None:
        """Refresh table rows with current snapshot data.

        Preserves cursor position AND scroll position across refresh cycles.
        Respects filter if set.
        Applies visual quieting to OK/STALLED envs unless top 5 acc or has fossilized.
        """
        if self._snapshot is None:
            return

        # Save cursor and scroll position before clearing
        saved_cursor_row = self.table.cursor_row
        saved_scroll_y = self.table.scroll_y

        # Clear existing rows
        self.table.clear()

        # Sort envs by ID for stable display, then filter
        sorted_envs = sorted(
            self._snapshot.envs.values(),
            key=lambda e: e.env_id
        )
        filtered_envs = [e for e in sorted_envs if self._matches_filter(e)]

        # Show placeholder if filter matches nothing
        if self._filter_text and not filtered_envs:
            self._add_no_matches_row()
            return

        # Compute top 5 accuracy env IDs for visual quieting
        all_envs = list(self._snapshot.envs.values())
        top5_env_ids = self._compute_top5_accuracy_ids(all_envs)

        # Add row for each environment
        for env in filtered_envs:
            should_dim = self._should_dim_row(env, top5_env_ids)
            self._add_env_row(env, dim=should_dim)

        # Add aggregate row if multiple filtered envs and no filter
        # (aggregate doesn't make sense for filtered subset)
        if not self._filter_text and len(self._snapshot.envs) > 1:
            self._add_separator_row()
            self._add_aggregate_row()

        # Restore cursor position (clamped to valid range)
        if self.table.row_count > 0 and saved_cursor_row is not None:
            target_row = min(saved_cursor_row, self.table.row_count - 1)
            self.table.move_cursor(row=target_row)

        # Restore scroll position after layout is computed
        # Direct assignment to scroll_y doesn't work before layout pass completes
        if saved_scroll_y > 0:
            # Capture value in default arg to avoid closure issues
            self.table.call_after_refresh(
                lambda y=saved_scroll_y: setattr(self.table, "scroll_y", y)
            )

    def _compute_top5_accuracy_ids(self, envs: list["EnvState"]) -> set[int]:
        """Compute env IDs of top 5 by current accuracy.

        Args:
            envs: All environments.

        Returns:
            Set of env_ids for top 5 accuracy envs.
        """
        if len(envs) <= 5:
            return {e.env_id for e in envs}

        # Sort by accuracy descending, take top 5
        sorted_by_acc = sorted(envs, key=lambda e: e.host_accuracy, reverse=True)
        return {e.env_id for e in sorted_by_acc[:5]}

    def _has_fossilized_seed(self, env: "EnvState") -> bool:
        """Check if env has any fossilized seed.

        Args:
            env: Environment to check.

        Returns:
            True if any seed is in FOSSILIZED stage.
        """
        for seed in env.seeds.values():
            if seed and seed.stage == "FOSSILIZED":
                return True
        return False

    def _should_dim_row(self, env: "EnvState", top5_ids: set[int]) -> bool:
        """Determine if row should be visually dimmed.

        Visual quieting rules:
        - Dim rows with status OK or STALLED
        - EXCEPT: if env is in top 5 accuracy
        - EXCEPT: if env has a fossilized seed

        Args:
            env: Environment to check.
            top5_ids: Set of env IDs in top 5 accuracy.

        Returns:
            True if row should be dimmed.
        """
        # Only dim OK or STALLED statuses
        if env.status not in ("healthy", "stalled"):
            return False

        # Don't dim if in top 5 accuracy
        if env.env_id in top5_ids:
            return False

        # Don't dim if has fossilized seed
        if self._has_fossilized_seed(env):
            return False

        return True

    def _add_env_row(self, env: "EnvState", dim: bool = False) -> None:
        """Add a single environment row.

        Args:
            env: Environment state to display.
            dim: If True, apply dim styling for visual quieting.
        """
        # Check for rollback state - show red alert instead of normal row
        if env.rolled_back:
            self._add_rollback_alert_row(env)
            return

        # Env ID with A/B test cohort pip and action target indicator
        last_action_env_id = self._snapshot.last_action_env_id if self._snapshot else None
        last_action_timestamp = self._snapshot.last_action_timestamp if self._snapshot else None
        env_id_cell = self._format_env_id(
            env,
            last_action_env_id=last_action_env_id,
            last_action_timestamp=last_action_timestamp,
        )

        # Accuracy with color coding
        acc_cell = self._format_accuracy(env)

        # Cumulative reward
        cum_rwd_cell = self._format_cumulative_reward(env)

        # Host loss (for overfitting detection)
        loss_cell = self._format_host_loss(env)

        # Counterfactual: synergy/interference indicator
        cf_cell = self._format_counterfactual(env)

        # Growth ratio: (host+fossilized)/host
        growth_cell = self._format_growth_ratio(env)

        # Reward (current and average)
        reward_cell = self._format_reward(env)

        # Sparklines
        acc_spark = self._make_sparkline(list(env.accuracy_history))
        rwd_spark = self._make_sparkline(list(env.reward_history))

        # Reward components
        delta_acc_cell = self._format_delta_acc(env)
        seed_delta_cell = self._format_seed_delta(env)
        rent_cell = self._format_rent(env)

        # Dynamic slot cells
        slot_cells = []
        if self._snapshot is not None:
            for slot_id in self._snapshot.slot_ids:
                slot_cells.append(self._format_slot_cell(env, slot_id))

        # Last action
        last_action = self._format_last_action(env)

        # Momentum (epochs since improvement)
        momentum_cell = self._format_momentum_epochs(env)

        # Status
        status_cell = self._format_status(env)

        # Telemetry staleness (time since last env update)
        staleness_cell = self._format_row_staleness(env)

        # Build row - order matches column definition
        row = [
            env_id_cell,
            acc_cell,
            cum_rwd_cell,
            loss_cell,
            cf_cell,
            growth_cell,
            reward_cell,
            acc_spark,
            rwd_spark,
            delta_acc_cell,
            seed_delta_cell,
            rent_cell,
        ] + slot_cells + [
            last_action,
            momentum_cell,
            status_cell,
            staleness_cell,
        ]

        # Apply visual quieting (dim) if needed
        if dim:
            row = [self._apply_dim(cell) for cell in row]

        # Add row with key=env_id for row selection event handling
        self.table.add_row(*row, key=str(env.env_id))

    def _add_rollback_alert_row(self, env: "EnvState") -> None:
        """Add a red alert row for an env that has rolled back.

        Shows a prominent CATASTROPHIC FAILURE message instead of normal metrics.
        This row persists until training resumes for this env.

        Args:
            env: Environment state with rolled_back=True.
        """
        # Format reason for display
        reason_display = {
            "governor_nan": "NaN DETECTED",
            "governor_lobotomy": "LOBOTOMY",
            "governor_divergence": "DIVERGENCE",
        }.get(env.rollback_reason, env.rollback_reason.upper() if env.rollback_reason else "UNKNOWN")

        # Build the alert row
        # First cell: env ID
        env_id_cell = f"[bold red]{env.env_id}[/bold red]"

        # Calculate how many columns we need to fill
        num_cols = len(self.table.columns)

        # Alert message spans most columns
        alert_msg = f"[bold white on red] ⚠ CATASTROPHIC FAILURE - ROLLED BACK ({reason_display}) [/bold white on red]"

        # Build row: env_id, then alert message, then empty cells
        row = [env_id_cell, alert_msg]

        # Fill remaining columns with empty cells styled red
        for _ in range(num_cols - 2):
            row.append("[on red] [/on red]")

        self.table.add_row(*row, key=str(env.env_id))

    def _add_separator_row(self) -> None:
        """Add separator row before aggregate."""
        num_cols = len(self.table.columns)
        separator = ["─" * 2] * num_cols
        self.table.add_row(*separator)

    def _add_no_matches_row(self) -> None:
        """Add placeholder row when filter matches no environments."""
        num_cols = len(self.table.columns)
        # First cell shows message, rest are empty
        row = [f"[dim italic]No environments match '{self._filter_text}'[/dim italic]"]
        row.extend([""] * (num_cols - 1))
        self.table.add_row(*row)

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

        # Calculate mean loss
        losses = [e.host_loss for e in self._snapshot.envs.values() if e.host_loss > 0]
        mean_loss = sum(losses) / len(losses) if losses else 0.0

        # Calculate mean growth ratio
        growth_ratios = [e.growth_ratio for e in self._snapshot.envs.values()]
        mean_growth = sum(growth_ratios) / len(growth_ratios) if growth_ratios else 1.0

        # Calculate total cumulative reward across all envs
        total_cum_rwd = sum(e.cumulative_reward for e in self._snapshot.envs.values())

        # Build aggregate row - order must match column definition
        agg_row = [
            "[bold]Σ[/bold]",
            f"[bold]{self._snapshot.aggregate_mean_accuracy:.1f}%[/bold]",
            f"[bold]{total_cum_rwd:+.1f}[/bold]",  # Total cumulative reward
            f"[dim]{mean_loss:.3f}[/dim]" if mean_loss > 0 else "─",  # Mean loss
            "",  # CF - not aggregated
            f"[dim]{mean_growth:.2f}x[/dim]",  # Growth ratio mean
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

        # Momentum column - show mean epochs since improvement
        stales = [e.epochs_since_improvement for e in self._snapshot.envs.values()]
        mean_stale = sum(stales) / len(stales) if stales else 0
        agg_row.append(f"[dim]μ{mean_stale:.0f}[/dim]")

        # Status column shows best accuracy from any env
        best_acc = max((e.best_accuracy for e in self._snapshot.envs.values()), default=0.0)
        agg_row.append(f"[dim]{best_acc:.1f}%[/dim]")

        # Stale column - show mean telemetry age
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        ages = [
            (now - e.last_update).total_seconds()
            for e in self._snapshot.envs.values()
            if e.last_update is not None
        ]
        mean_age = sum(ages) / len(ages) if ages else 0.0
        agg_row.append(f"[dim]μ{mean_age:.0f}s[/dim]" if ages else "[dim]─[/dim]")

        self.table.add_row(*agg_row)

    def _format_env_id(
        self,
        env: "EnvState",
        last_action_env_id: int | None = None,
        last_action_timestamp: "datetime | None" = None,
    ) -> str:
        """Format env ID with A/B test cohort pip and action target indicator.

        Args:
            env: Environment state.
            last_action_env_id: ID of env that received last action (for highlighting).
            last_action_timestamp: When the last action occurred (for hysteresis).

        Returns:
            Formatted env ID string with indicators.
        """
        from datetime import datetime, timezone

        # Action target indicator (cyan ▶ prefix) - per UX accessibility review
        # Only show if action was within last 5 seconds (hysteresis prevents jitter)
        action_pip = ""
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0  # 5-second hysteresis
            if show_indicator:
                action_pip = "[cyan]▶[/cyan]"

        # A/B cohort pip (existing logic)
        if env.reward_mode and env.reward_mode in _AB_STYLES:
            pip, color = _AB_STYLES[env.reward_mode]
            return f"{action_pip}[{color}]{pip}[/{color}]{env.env_id}"
        return f"{action_pip}{env.env_id}"

    def _format_host_loss(self, env: "EnvState") -> str:
        """Format host loss with color coding for overfitting detection.

        Loss color coding:
        - Green: loss < 0.1 (very low, good convergence)
        - White: 0.1 <= loss < 0.5 (normal training range)
        - Yellow: 0.5 <= loss < 1.0 (elevated, might be overfitting)
        - Red: loss >= 1.0 (high, possible issues)
        """
        loss = env.host_loss
        if loss <= 0:
            return "[dim]─[/dim]"
        elif loss < 0.1:
            return f"[green]{loss:.3f}[/green]"
        elif loss < 0.5:
            return f"{loss:.3f}"
        elif loss < 1.0:
            return f"[yellow]{loss:.3f}[/yellow]"
        else:
            return f"[red]{loss:.3f}[/red]"

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

    def _format_accuracy(self, env: "EnvState") -> str:
        """Format accuracy with color coding and trend arrow.

        Trend arrows: ↑ (improving), ↓ (declining), → (stable)
        Color: Green if at best, yellow if stagnant >5 epochs.
        """
        acc_str = f"{env.host_accuracy:.1f}%"

        # Compute trend from last 5 epochs of accuracy history
        trend_arrow = self._compute_trend_arrow(list(env.accuracy_history))

        if env.best_accuracy > 0:
            if env.host_accuracy >= env.best_accuracy:
                return f"[green]{trend_arrow}{acc_str}[/green]"
            elif env.epochs_since_improvement > 5:
                return f"[yellow]{trend_arrow}{acc_str}[/yellow]"

        return f"{trend_arrow}{acc_str}"

    def _format_cumulative_reward(self, env: "EnvState") -> str:
        """Format cumulative reward with color coding.

        Color based on value:
        - Green: > 0 (positive overall)
        - Red: < -5 (significantly negative)
        - White: near zero
        """
        cum_rwd = env.cumulative_reward
        if cum_rwd > 0:
            return f"[green]{cum_rwd:+.1f}[/green]"
        elif cum_rwd < -5:
            return f"[red]{cum_rwd:+.1f}[/red]"
        else:
            return f"{cum_rwd:+.1f}"

    def _compute_trend_arrow(self, history: list[float], window: int = 5) -> str:
        """Compute trend arrow from history.

        Args:
            history: List of values (newest at end).
            window: Number of values to consider.

        Returns:
            ↑ if improving by >0.5%, ↓ if declining by >0.5%, → if stable.
        """
        if len(history) < 2:
            return ""

        # Compare recent average to earlier average
        recent = history[-min(window, len(history)):]
        if len(recent) < 2:
            return ""

        # Compare end vs start of window
        delta = recent[-1] - recent[0]

        if delta > 0.5:
            return "↑"
        elif delta < -0.5:
            return "↓"
        else:
            return "→"

    def _format_counterfactual(self, env: "EnvState") -> str:
        """Format counterfactual synergy/interference indicator.

        Shows:
        - '+' (green): Synergy detected (seeds help each other)
        - '-' (red): Interference detected (seeds hurt each other) - LOUD
        - '.' (dim): Neutral or unavailable
        """
        cf = env.counterfactual_matrix
        if cf is None or cf.strategy == "unavailable" or len(cf.slot_ids) < 2:
            return "[dim]·[/dim]"

        synergy = cf.total_synergy()

        if synergy > 0.5:
            # Positive synergy: seeds working together
            return "[green]+[/green]"
        elif synergy < -0.5:
            # Negative synergy: interference!
            return "[bold red]-[/bold red]"
        else:
            # Neutral
            return "[dim]·[/dim]"

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
            "GERMINATED": "Germi",
            "TRAINING": "Train",
            "BLENDING": "Blend",
            "HOLDING": "Hold",
            "FOSSILIZED": "Foss",
            "PRUNED": "Prune",
            "EMBARGOED": "Embar",
            "RESETTING": "Reset",
        }.get(seed.stage, seed.stage[:5])

        blueprint = seed.blueprint_id or "?"
        if len(blueprint) > 6:
            blueprint = blueprint[:6]

        # Stage-specific styling from leyline
        style = STAGE_COLORS.get(seed.stage, "white")

        # Gradient health indicator
        grad_indicator = ""
        if seed.has_exploding:
            grad_indicator = "[red]▲[/red]"
        elif seed.has_vanishing:
            grad_indicator = "[yellow]▼[/yellow]"

        # Curve glyph: shown for BLENDING/HOLDING/FOSSILIZED, dim "-" for other stages
        # Position: after stage:blueprint with space (e.g., "Blend:conv_l ⌢")
        # Always use raw glyph character - dim styling applied in format strings where needed
        curve_glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "−") if seed.stage in ("BLENDING", "HOLDING", "FOSSILIZED") else "−"

        # Helper: tempo arrows based on blend_tempo_epochs
        # Tempo: ▸▸▸ = FAST (3), ▸▸ = STANDARD (5), ▸ = SLOW (8)
        def _tempo_arrows(tempo: int | None) -> str:
            if tempo is None:
                return ""
            return "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")

        # BLENDING shows tempo arrows and alpha
        if seed.stage == "BLENDING" and seed.alpha > 0:
            tempo_arrows = _tempo_arrows(seed.blend_tempo_epochs)
            base = f"[{style}]{stage_short}:{blueprint} {curve_glyph} {tempo_arrows} {seed.alpha:.1f}[/{style}]"
            return f"{base}{grad_indicator}"

        # HOLDING shows tempo arrows + alpha (blend tempo was used, still relevant)
        if seed.stage == "HOLDING":
            tempo_arrows = _tempo_arrows(seed.blend_tempo_epochs)
            base = f"[{style}]{stage_short}:{blueprint} {curve_glyph} {tempo_arrows} {seed.alpha:.1f}[/{style}]"
            return f"{base}{grad_indicator}"

        # FOSSILIZED shows historical tempo + curve (how it blended)
        if seed.stage == "FOSSILIZED":
            tempo_arrows = _tempo_arrows(seed.blend_tempo_epochs)
            base = f"[{style}]{stage_short}:{blueprint} [dim]{curve_glyph}[/dim] {tempo_arrows}[/{style}]"
            return f"{base}{grad_indicator}"

        # Other stages show epochs in stage with dim "-" for curve
        epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
        base = f"[{style}]{stage_short}:{blueprint} [dim]{curve_glyph}[/dim]{epochs_str}[/{style}]"
        return f"{base}{grad_indicator}"

    def _format_last_action(self, env: "EnvState") -> str:
        """Format last action taken."""
        if not env.action_history:
            return "—"

        last_action = env.action_history[-1]

        # Shorten action names for compact display
        action_short = {
            "WAIT": "WAIT",
            "GERMINATE": "GERM",
            "SET_ALPHA_TARGET": "ALPH",
            "FOSSILIZE": "FOSS",
            "PRUNE": "PRUN",
        }.get(last_action, last_action[:4] if last_action else "—")

        return action_short

    def _format_momentum_epochs(self, env: "EnvState") -> str:
        """Format epochs since improvement with context-aware color coding.

        Momentum is only concerning when stuck in a BAD state. Being stable
        at "excellent" or "healthy" for many epochs is good, not bad.

        Returns the number of epochs since the last improvement, colored:
        - Green: 0 (currently improving) with + prefix
        - For healthy/excellent status: white numbers (stability is good)
        - For stalled/degraded status: yellow/red warnings (stuck in bad state)
        - For initializing: dim (neutral, still warming up)

        Uses fixed-width ASCII prefixes for consistent column alignment.
        """
        epochs = env.epochs_since_improvement
        status = env.status

        # Currently improving - always green
        if epochs == 0:
            return "[green]+0[/green]"

        # Good states: stability is a feature, not a bug
        if status in ("excellent", "healthy"):
            return f"[green] {epochs}[/green]"

        # Initializing: neutral, still warming up
        if status == "initializing":
            return f"[dim] {epochs}[/dim]"

        # Bad states (stalled, degraded): staleness is concerning
        if epochs <= 5:
            return f"[white] {epochs}[/white]"
        elif epochs <= 15:
            return f"[yellow]!{epochs}[/yellow]"
        else:
            return f"[red]x{epochs}[/red]"

    def _format_row_staleness(self, env: "EnvState") -> str:
        """Format staleness as time since last env update."""
        from datetime import datetime, timezone

        if env.last_update is None:
            return "[red]●BAD[/red]"

        age_s = (datetime.now(timezone.utc) - env.last_update).total_seconds()
        if age_s < 2.0:
            return "[green]●OK[/green]"
        if age_s <= 5.0:
            return "[yellow]●WARN[/yellow]"
        return "[red]●BAD[/red]"

    def _format_status(self, env: "EnvState") -> str:
        """Format status with color coding."""
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }

        # Icons provide color-independent status indication (accessibility)
        # Hierarchy: ★ (excellent) > ● (ok) > ◐ (stall/init) > ○ (degraded)
        status_short = {
            "excellent": "★EXCL",
            "healthy": "●OK",
            "initializing": "◐INIT",
            "stalled": "◐STALL",
            "degraded": "○DEGR",
        }.get(env.status, env.status[:4].upper())

        status_style = status_styles.get(env.status, "white")
        return f"[{status_style}]{status_short}[/{status_style}]"

    def _make_sparkline(self, values: list[float], width: int = 8) -> str:
        """Create sparkline from values using schema module."""
        from esper.karn.sanctum.schema import make_sparkline
        return make_sparkline(values, width)

    def _apply_dim(self, text: str) -> str:
        """Apply dim styling to text for visual quieting.

        Wraps the entire cell content in [dim]...[/dim] markup.
        This works with Rich markup - dim reduces contrast on all colors.

        Args:
            text: Cell content (may contain Rich markup).

        Returns:
            Dimmed version of the text.
        """
        if not text or text == "─":
            return text
        return f"[dim]{text}[/dim]"
