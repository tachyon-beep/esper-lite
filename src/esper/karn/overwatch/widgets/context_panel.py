"""Context Panel Widget.

Displays detailed context for the selected environment:
- "Why Flagged" anomaly reasons
- Slot details (stage, blueprint, alpha, gate history)
- Environment metrics (throughput, reward, task metric)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary


class ContextPanel(Container):
    """Widget displaying context for the selected environment.

    Shows:
    - Environment header (ID, status, anomaly score)
    - "Why Flagged" bullet list of anomaly reasons
    - Slot details for each slot in the env
    - Environment metrics
    """

    DEFAULT_CSS = """
    ContextPanel {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ContextPanel .section-header {
        text-style: bold;
        margin-top: 1;
    }

    ContextPanel .anomaly-reason {
        color: $warning;
    }

    ContextPanel .healthy {
        color: $success;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the context panel."""
        super().__init__(**kwargs)
        self._env: EnvSummary | None = None

    def render_content(self) -> str:
        """Render the panel content."""
        if self._env is None:
            return "[dim]Select an environment (j/k to navigate)[/dim]"

        lines = []
        env = self._env

        # Header
        status_color = self._status_color(env.status)
        lines.append(f"[bold]Env {env.env_id}[/bold] [{status_color}]{env.status}[/{status_color}]")
        lines.append(f"Device: GPU {env.device_id} | Anomaly: {env.anomaly_score:.2f}")
        lines.append("")

        # Why Flagged section
        lines.append("[bold]Why Flagged[/bold]")
        if env.anomaly_reasons:
            for reason in env.anomaly_reasons:
                lines.append(f"  [yellow]•[/yellow] {reason}")
        else:
            lines.append("  [green]No issues detected - Healthy[/green]")
        lines.append("")

        # Metrics section
        lines.append("[bold]Metrics[/bold]")
        lines.append(f"  Throughput: {env.throughput_fps:.1f} fps")
        lines.append(f"  Last Reward: {env.reward_last:.3f}")
        lines.append(f"  Task Metric: {env.task_metric:.3f}")
        lines.append("")

        # Slots section
        if env.slots:
            lines.append("[bold]Slots[/bold]")
            for slot_id, slot in sorted(env.slots.items()):
                gate_str = ""
                if slot.gate_last:
                    gate_icon = "✓" if slot.gate_passed else "✗"
                    gate_str = f" | {slot.gate_last}{gate_icon}"
                lines.append(
                    f"  [{slot.slot_id}] {slot.stage} | {slot.blueprint_id} | "
                    f"α={slot.alpha:.2f}{gate_str}"
                )

        return "\n".join(lines)

    def _status_color(self, status: str) -> str:
        """Get Rich color for status."""
        colors = {
            "OK": "green",
            "INFO": "blue",
            "WARN": "yellow",
            "CRIT": "red",
        }
        return colors.get(status, "white")

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static(self.render_content(), id="context-content")

    def update_env(self, env: EnvSummary | None) -> None:
        """Update with selected environment."""
        self._env = env
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#context-content", Static).update(self.render_content())
        except Exception:
            pass  # Widget not mounted yet
