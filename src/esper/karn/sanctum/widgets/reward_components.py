"""RewardComponents widget - Esper reward breakdown.

Port of tui.py _render_reward_components() (lines 1565-1637).
Shows reward component breakdown for the focused environment.

Reference: src/esper/karn/tui.py lines 1565-1637 (_render_reward_components method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class RewardComponents(Static):
    """RewardComponents widget - Esper reward breakdown.

    Shows reward component breakdown for the focused environment:
    - Header: env_id, last_action, val_acc
    - ΔAcc (base_acc_delta): Always show, green if ≥0, red if <0
    - Attr (bounded_attribution): Show only if non-zero, green/red
    - Rent (compute_rent): Always show, red if <0
    - Penalty (ratio_penalty): Show only if non-zero, red if <0
    - Stage (stage_bonus): Show only if non-zero, blue
    - Fossil (fossilize_terminal_bonus): Show only if non-zero, blue
    - Blend Warn: Show only if <0, yellow
    - Prob Warn: Show only if <0, yellow
    - Total: Always show, bold green if ≥0, bold red if <0

    Display rules follow Esper-specific reward component semantics.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize RewardComponents widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._override_env_id: int | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot", env_id: int | None = None) -> None:
        """Update widget with new snapshot data.

        This method intentionally extends the SanctumWidget protocol by accepting an
        optional env_id parameter. When provided, it overrides the focused_env_id in
        the snapshot for rendering. This allows the reward components to display data
        for a specific environment when explicitly requested (e.g., when user selects
        an environment via keyboard shortcut).

        Args:
            snapshot: The current telemetry snapshot.
            env_id: Optional environment ID to focus. If None, uses snapshot's focused_env_id.
        """
        self._snapshot = snapshot
        self._override_env_id = env_id
        self.refresh()

    def render(self) -> Panel:
        """Render the reward components panel."""
        if self._snapshot is None:
            return Panel("No data", title="REWARD COMPONENTS", border_style="cyan")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim")
        table.add_column("Value", justify="right")

        # Get focused env (use override if provided)
        env_id = self._override_env_id if self._override_env_id is not None else self._snapshot.focused_env_id
        env_state = self._snapshot.envs.get(env_id)

        if env_state is None:
            return Panel(
                "No env selected",
                title="[bold]REWARD COMPONENTS[/bold]",
                border_style="cyan",
            )

        # Use snapshot.rewards (RewardComponents dataclass) instead of
        # env_state.reward_components (dict). The aggregator populates
        # snapshot.rewards from the focused env's telemetry.
        components = self._snapshot.rewards

        # Header context
        table.add_row("Env:", Text(str(env_id), style="bold cyan"))
        if env_state.action_history:
            table.add_row("Action:", env_state.action_history[-1])
        if isinstance(components.val_acc, (int, float)) and components.val_acc > 0:
            table.add_row("Val Acc:", f"{components.val_acc:.1f}%")

        table.add_row("", "")

        # Base delta (legacy shaped signal)
        if isinstance(components.base_acc_delta, (int, float)):
            style = "green" if components.base_acc_delta >= 0 else "red"
            table.add_row("ΔAcc:", Text(f"{components.base_acc_delta:+.2f}", style=style))

        # Attribution (contribution-primary)
        if isinstance(components.bounded_attribution, (int, float)) and components.bounded_attribution != 0.0:
            style = "green" if components.bounded_attribution >= 0 else "red"
            table.add_row("Attr:", Text(f"{components.bounded_attribution:+.2f}", style=style))

        # Compute rent (usually negative)
        if isinstance(components.compute_rent, (int, float)):
            style = "red" if components.compute_rent < 0 else "dim"
            table.add_row("Rent:", Text(f"{components.compute_rent:+.2f}", style=style))

        # Ratio penalty (ransomware / attribution mismatch)
        if isinstance(components.ratio_penalty, (int, float)) and components.ratio_penalty != 0.0:
            style = "red" if components.ratio_penalty < 0 else "dim"
            table.add_row("Penalty:", Text(f"{components.ratio_penalty:+.2f}", style=style))

        # Stage / terminal bonuses
        if isinstance(components.stage_bonus, (int, float)) and components.stage_bonus != 0.0:
            table.add_row("Stage:", Text(f"{components.stage_bonus:+.2f}", style="blue"))

        if isinstance(components.fossilize_terminal_bonus, (int, float)) and components.fossilize_terminal_bonus != 0.0:
            table.add_row("Fossil:", Text(f"{components.fossilize_terminal_bonus:+.2f}", style="blue"))

        # Warnings
        if isinstance(components.blending_warning, (int, float)) and components.blending_warning < 0:
            table.add_row("Blend Warn:", Text(f"{components.blending_warning:.2f}", style="yellow"))

        if isinstance(components.probation_warning, (int, float)) and components.probation_warning < 0:
            table.add_row("Prob Warn:", Text(f"{components.probation_warning:.2f}", style="yellow"))

        # Total (last computed reward for this env)
        table.add_row("", "")
        table.add_row("───────────", "───────")
        total = components.total
        style = "bold green" if total >= 0 else "bold red"
        table.add_row("Total:", Text(f"{total:+.2f}", style=style))

        return Panel(
            table,
            title=f"[bold]REWARD COMPONENTS (env {env_id})[/bold]",
            border_style="cyan",
        )
