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

        components = env_state.reward_components

        # Header context
        table.add_row("Env:", Text(str(env_id), style="bold cyan"))
        if env_state.action_history:
            table.add_row("Action:", env_state.action_history[-1])
        if "val_acc" in components and isinstance(components.get("val_acc"), (int, float)):
            table.add_row("Val Acc:", f"{float(components['val_acc']):.1f}%")

        table.add_row("", "")

        # Base delta (legacy shaped signal)
        base = components.get("base_acc_delta")
        if isinstance(base, (int, float)):
            style = "green" if float(base) >= 0 else "red"
            table.add_row("ΔAcc:", Text(f"{float(base):+.2f}", style=style))

        # Attribution (contribution-primary)
        bounded = components.get("bounded_attribution")
        if isinstance(bounded, (int, float)) and float(bounded) != 0.0:
            style = "green" if float(bounded) >= 0 else "red"
            table.add_row("Attr:", Text(f"{float(bounded):+.2f}", style=style))

        # Compute rent (usually negative)
        rent = components.get("compute_rent")
        if isinstance(rent, (int, float)):
            style = "red" if float(rent) < 0 else "dim"
            table.add_row("Rent:", Text(f"{float(rent):+.2f}", style=style))

        # Ratio penalty (ransomware / attribution mismatch)
        ratio_penalty = components.get("ratio_penalty")
        if isinstance(ratio_penalty, (int, float)) and float(ratio_penalty) != 0.0:
            style = "red" if float(ratio_penalty) < 0 else "dim"
            table.add_row("Penalty:", Text(f"{float(ratio_penalty):+.2f}", style=style))

        # Stage / terminal bonuses
        stage_bonus = components.get("stage_bonus")
        if isinstance(stage_bonus, (int, float)) and float(stage_bonus) != 0.0:
            table.add_row("Stage:", Text(f"{float(stage_bonus):+.2f}", style="blue"))

        fossil_bonus = components.get("fossilize_terminal_bonus")
        if isinstance(fossil_bonus, (int, float)) and float(fossil_bonus) != 0.0:
            table.add_row("Fossil:", Text(f"{float(fossil_bonus):+.2f}", style="blue"))

        # Warnings
        blending_warn = components.get("blending_warning")
        if isinstance(blending_warn, (int, float)) and float(blending_warn) < 0:
            table.add_row("Blend Warn:", Text(f"{float(blending_warn):.2f}", style="yellow"))

        probation_warn = components.get("probation_warning")
        if isinstance(probation_warn, (int, float)) and float(probation_warn) < 0:
            table.add_row("Prob Warn:", Text(f"{float(probation_warn):.2f}", style="yellow"))

        # Total (last computed reward for this env)
        table.add_row("", "")
        table.add_row("───────────", "───────")
        total = env_state.current_reward
        style = "bold green" if total >= 0 else "bold red"
        table.add_row("Total:", Text(f"{total:+.2f}", style=style))

        return Panel(
            table,
            title=f"[bold]REWARD COMPONENTS (env {env_id})[/bold]",
            border_style="cyan",
        )
