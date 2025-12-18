"""TamiyoBrain widget - Policy agent diagnostics.

Port of tui.py _render_tamiyo_brain() (lines 1192-1237).
Shows 4-column layout with policy health, losses, vitals, and actions.

Reference: src/esper/karn/tui.py lines 1192-1237 (_render_tamiyo_brain method)
          src/esper/karn/tui.py lines 1373-1483 (helper table methods)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TamiyoBrain(Static):
    """TamiyoBrain widget - Policy agent diagnostics.

    Shows:
    1. Health: entropy, clip, KL, explained variance with status indicators
    2. Losses: policy/value/entropy loss, gradient norm
    3. Vitals: LR, ratio stats (min/max/std), gradient health (dead/exploding, GradHP)
    4. Actions: WAIT/GERMINATE/CULL/FOSSILIZE distribution with percentages

    Before PPO data arrives, shows waiting state with progress indicator.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize TamiyoBrain widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Panel:
        """Render the TamiyoBrain panel."""
        if self._snapshot is None:
            return Panel("No data", title="TAMIYO BRAIN (Policy Agent)", border_style="magenta")

        # Show waiting state if no PPO data received yet
        if not self._snapshot.tamiyo.ppo_data_received:
            waiting_text = Text(justify="center")
            waiting_text.append("⏳ Waiting for PPO data...\n", style="dim italic")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            return Panel(
                waiting_text,
                title="[bold magenta]TAMIYO BRAIN (Policy Agent)[/bold magenta]",
                border_style="magenta dim",
            )

        # Create a grid layout for the brain panel (4 columns)
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Column 1: Health
        health_table = self._render_policy_health_table()

        # Column 2: Losses
        losses_table = self._render_losses_table()

        # Column 3: Vitals
        vitals_table = self._render_vitals_table()

        # Column 4: Actions
        actions_table = self._render_actions_table()

        # Use fixed height for uniform sub-panels (7 rows + border)
        panel_height = 9
        grid.add_row(
            Panel(health_table, title="Health", border_style="dim", height=panel_height),
            Panel(losses_table, title="Losses", border_style="dim", height=panel_height),
            Panel(vitals_table, title="Vitals", border_style="dim", height=panel_height),
            Panel(actions_table, title="Actions", border_style="dim", height=panel_height),
        )

        return Panel(
            grid,
            title="[bold magenta]TAMIYO BRAIN (Policy Agent)[/bold magenta]",
            border_style="magenta",
        )

    def _render_policy_health_table(self) -> Table:
        """Render policy health as a table (Health column)."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim", justify="left", width=10)
        table.add_column("Val", justify="right", width=8)
        table.add_column("St", justify="center", width=6)  # "✕ CRIT" is 6 chars

        tamiyo = self._snapshot.tamiyo

        # Entropy
        entropy_status = self._get_entropy_status(tamiyo.entropy)
        table.add_row("Entropy", f"{tamiyo.entropy:.2f}", self._status_text(entropy_status))

        # Clip fraction
        clip_status = self._get_clip_status(tamiyo.clip_fraction)
        table.add_row("Clip", f"{tamiyo.clip_fraction:.2f}", self._status_text(clip_status))

        # KL divergence
        kl_status = self._get_kl_status(tamiyo.kl_divergence)
        table.add_row("KL", f"{tamiyo.kl_divergence:.3f}", self._status_text(kl_status))

        # Explained variance
        ev_status = self._get_ev_status(tamiyo.explained_variance)
        table.add_row("ExplVar", f"{tamiyo.explained_variance:.2f}", self._status_text(ev_status))

        return table

    def _render_losses_table(self) -> Table:
        """Render losses as a table (Losses column)."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Loss", style="dim", justify="left", width=10)
        table.add_column("Value", justify="right", width=10)

        tamiyo = self._snapshot.tamiyo

        table.add_row("Policy", f"{tamiyo.policy_loss:.4f}")
        table.add_row("Value", f"{tamiyo.value_loss:.4f}")
        table.add_row("Entropy", f"{tamiyo.entropy_loss:.4f}")

        # Gradient norm with status
        grad_status = self._get_grad_norm_status(tamiyo.grad_norm)
        grad_style = self._status_style(grad_status)
        table.add_row("GradNorm", Text(f"{tamiyo.grad_norm:.2f}", style=grad_style))

        return table

    def _render_vitals_table(self) -> Table:
        """Render training vitals as a table (Vitals column)."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim", justify="left", width=10)
        table.add_column("Value", justify="right", width=10)

        tamiyo = self._snapshot.tamiyo

        # Learning rate (scientific notation)
        if tamiyo.learning_rate is not None:
            lr_str = f"{tamiyo.learning_rate:.2e}"
        else:
            lr_str = "─"
        table.add_row("LR", lr_str)

        # Ratio statistics (PPO policy ratio - should stay near 1.0)
        ratio_max = tamiyo.ratio_max
        ratio_min = tamiyo.ratio_min
        ratio_std = tamiyo.ratio_std

        # Color code ratio_max: green if <1.5, yellow if <2.0, red if >=2.0
        if ratio_max >= TUIThresholds.RATIO_MAX_CRITICAL:
            max_style = "red bold"
        elif ratio_max >= TUIThresholds.RATIO_MAX_WARNING:
            max_style = "yellow"
        else:
            max_style = "green"
        table.add_row("Ratio↑", Text(f"{ratio_max:.2f}", style=max_style))

        # Color code ratio_min: green if >0.5, yellow if >0.3, red if <=0.3
        if ratio_min <= TUIThresholds.RATIO_MIN_CRITICAL:
            min_style = "red bold"
        elif ratio_min <= TUIThresholds.RATIO_MIN_WARNING:
            min_style = "yellow"
        else:
            min_style = "green"
        table.add_row("Ratio↓", Text(f"{ratio_min:.2f}", style=min_style))

        # Ratio std - higher means more variance in updates
        if ratio_std >= TUIThresholds.RATIO_STD_WARNING:
            std_style = "yellow"
        else:
            std_style = ""
        table.add_row("Ratio σ", Text(f"{ratio_std:.3f}", style=std_style))

        # Gradient health: dead/exploding layers
        dead = tamiyo.dead_layers
        exploding = tamiyo.exploding_layers
        health = tamiyo.layer_gradient_health

        # Show gradient issues with color
        if dead > 0:
            table.add_row("Dead", Text(f"{dead} layers", style="yellow bold"))
        if exploding > 0:
            table.add_row("Explode", Text(f"{exploding} layers", style="red bold"))

        # Overall gradient health (1.0 = perfect)
        if health < TUIThresholds.GRAD_HEALTH_CRITICAL:
            health_style = "red bold"
        elif health < TUIThresholds.GRAD_HEALTH_WARNING:
            health_style = "yellow"
        else:
            health_style = "green"
        table.add_row("GradHP", Text(f"{health:.0%}", style=health_style))

        return table

    def _render_actions_table(self) -> Table:
        """Render action distribution as a table (Actions column)."""
        table = Table(show_header=False, box=None, padding=(0, 0))
        table.add_column("Action", style="dim", width=8)
        table.add_column("Bar", width=8)
        table.add_column("%", justify="right", width=4)

        tamiyo = self._snapshot.tamiyo
        total = tamiyo.total_actions

        if total == 0:
            table.add_row("─", "─", "─")
            return table

        # Calculate percentages
        percentages = {}
        for action, count in tamiyo.action_counts.items():
            percentages[action] = (count / total) * 100

        # Display actions in descending order by percentage
        for action, pct in sorted(percentages.items(), key=lambda x: -x[1]):
            action_style = {
                "WAIT": "dim",
                "GERMINATE": "green",
                "CULL": "red",
                "FOSSILIZE": "blue",
            }.get(action, "white")

            # Warn if WAIT > 70%
            pct_style = (
                "yellow bold"
                if action == "WAIT" and pct > TUIThresholds.WAIT_DOMINANCE_WARNING * 100
                else ""
            )

            table.add_row(
                Text(action, style=action_style),
                self._make_bar(pct, width=8),
                Text(f"{pct:.0f}%", style=pct_style),
            )

        return table

    # ========================================================================
    # Status Helpers (using TUIThresholds)
    # ========================================================================

    def _get_entropy_status(self, entropy: float) -> str:
        """Get health status for entropy."""
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning"
        return "ok"

    def _get_clip_status(self, clip_fraction: float) -> str:
        """Get health status for clip fraction."""
        if clip_fraction > TUIThresholds.CLIP_CRITICAL:
            return "critical"
        elif clip_fraction > TUIThresholds.CLIP_WARNING:
            return "warning"
        return "ok"

    def _get_kl_status(self, kl_div: float) -> str:
        """Get health status for KL divergence."""
        if kl_div > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _get_ev_status(self, ev: float) -> str:
        """Get health status for explained variance."""
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical"
        elif ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        """Get health status for gradient norm."""
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        elif grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        """Get Rich style string for status."""
        return {"ok": "green", "warning": "yellow", "critical": "red bold"}[status]

    def _status_text(self, status: str) -> str:
        """Get status text with color markup."""
        if status == "ok":
            return "[green]✓ OK[/]"
        elif status == "warning":
            return "[yellow]⚠ WARN[/]"
        else:
            return "[red bold]✕ CRIT[/]"

    def _make_bar(self, percentage: float, width: int = 8) -> Text:
        """Create a simple bar chart for percentage (0-100)."""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "─" * (width - filled)
        return Text(bar, style="dim")
