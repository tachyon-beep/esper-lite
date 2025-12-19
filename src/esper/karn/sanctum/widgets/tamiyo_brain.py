"""TamiyoBrain widget - Policy agent diagnostics (Redesigned).

New layout focuses on answering:
- "What is Tamiyo doing?" (Action distribution bar)
- "Is she learning?" (Entropy, Value Loss gauges)
- "What did she just decide?" (Last Decision snapshot)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.message import Message
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TamiyoBrain(Static):
    """TamiyoBrain widget - Policy agent diagnostics.

    New two-section layout:
    1. LEARNING VITALS - Action distribution bar + gauges (entropy, value loss, advantage)
    2. LAST DECISION - What Tamiyo saw, chose, and got

    Click on a decision panel to pin it (prevents replacement).
    """

    class DecisionPinToggled(Message):
        """Posted when user clicks a decision to toggle pin status."""

        def __init__(self, decision_id: str) -> None:
            super().__init__()
            self.decision_id = decision_id

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._decision_ids: list[str] = []  # IDs of currently displayed decisions
        self.border_title = "TAMIYO"  # Top-left title like EventLog

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def on_click(self, event) -> None:
        """Handle click to toggle decision pin.

        Decisions are in the bottom section of the widget.
        Each decision panel is ~5 lines tall (border + 3 content + border).
        We estimate which decision was clicked based on Y coordinate.
        """
        if not self._decision_ids:
            return

        # The RECENT DECISIONS section starts after LEARNING VITALS
        # LEARNING VITALS is roughly: title(1) + action bar(1) + gauges(3) + padding(1) = 6 lines
        # Then RECENT DECISIONS title(1), then each decision panel(~5 lines)
        vitals_height = 7  # Approximate height of Learning Vitals section
        decision_height = 5  # Each decision panel height

        y = event.y
        if y < vitals_height:
            return  # Click was in Learning Vitals, not decisions

        # Calculate which decision was clicked
        decision_y = y - vitals_height
        decision_index = decision_y // decision_height

        if 0 <= decision_index < len(self._decision_ids):
            decision_id = self._decision_ids[decision_index]
            self.post_message(self.DecisionPinToggled(decision_id))

    def render(self):
        """Render Tamiyo content (border provided by CSS, not Rich Panel)."""
        from rich.console import Group

        if self._snapshot is None:
            return Text("No data", style="dim")

        if not self._snapshot.tamiyo.ppo_data_received:
            waiting_text = Text(justify="center")
            waiting_text.append("⏳ Waiting for PPO data...\n", style="dim italic")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            return waiting_text

        # Main layout: two sections stacked
        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Section 1: Learning Vitals
        vitals_panel = self._render_learning_vitals()
        main_table.add_row(vitals_panel)

        # Section 2: Recent Decisions (up to 3, each visible for 10s minimum)
        decisions_panel = self._render_recent_decisions()
        main_table.add_row(decisions_panel)

        return main_table

    def _render_learning_vitals(self) -> Panel:
        """Render Learning Vitals section with action bar and gauges."""
        tamiyo = self._snapshot.tamiyo

        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        # Row 1: Action distribution bar
        action_bar = self._render_action_distribution_bar()
        content.add_row(action_bar)

        # Row 2: Gauges (Entropy, Value Loss, Advantage)
        gauges = Table.grid(expand=True)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)

        entropy_gauge = self._render_gauge(
            "Entropy", tamiyo.entropy, 0, 2.0,
            self._get_entropy_label(tamiyo.entropy)
        )
        value_gauge = self._render_gauge(
            "Value Loss", tamiyo.value_loss, 0, 1.0,
            self._get_value_loss_label(tamiyo.value_loss)
        )
        advantage_gauge = self._render_gauge(
            "Advantage", tamiyo.advantage_mean, -1.0, 1.0,
            self._get_advantage_label(tamiyo.advantage_mean)
        )

        gauges.add_row(entropy_gauge, value_gauge, advantage_gauge)
        content.add_row(gauges)

        return Panel(content, title="LEARNING VITALS", border_style="dim")

    def _render_action_distribution_bar(self) -> Text:
        """Render horizontal stacked bar for action distribution."""
        tamiyo = self._snapshot.tamiyo
        total = tamiyo.total_actions

        if total == 0:
            return Text("[no data]", style="dim")

        # Calculate percentages
        pcts = {a: (c / total) * 100 for a, c in tamiyo.action_counts.items()}

        # Build stacked bar (width 25 chars for narrower display)
        bar_width = 25
        bar = Text("[")

        # Color mapping
        colors = {
            "GERMINATE": "green",
            "WAIT": "dim",
            "BLEND": "cyan",  # Blending includes FOSSILIZE transitions
            "FOSSILIZE": "blue",
            "CULL": "red",
        }

        for action in ["GERMINATE", "WAIT", "FOSSILIZE", "CULL"]:
            pct = pcts.get(action, 0)
            width = int((pct / 100) * bar_width)
            if width > 0:
                bar.append("▓" * width, style=colors.get(action, "white"))

        bar.append("] ")

        # Compact legend: G=50 W=25 F=25
        abbrevs = {"GERMINATE": "G", "WAIT": "W", "FOSSILIZE": "F", "CULL": "C"}
        for action in ["GERMINATE", "WAIT", "FOSSILIZE", "CULL"]:
            pct = pcts.get(action, 0)
            if pct > 0:
                bar.append(f"{abbrevs[action]}={pct:.0f} ", style=colors.get(action, "white"))

        return bar

    def _render_gauge(self, label: str, value: float, min_val: float, max_val: float, description: str) -> Text:
        """Render a single gauge with label and description on separate lines."""
        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        normalized = max(0, min(1, normalized))

        # Build gauge bar (width 12)
        gauge_width = 12
        filled = int(normalized * gauge_width)
        empty = gauge_width - filled

        gauge = Text()
        # Line 1: Label
        gauge.append(f"{label}:\n", style="dim")
        # Line 2: Bar and value
        gauge.append("[")
        gauge.append("█" * filled, style="cyan")
        gauge.append("░" * empty, style="dim")
        gauge.append("]")
        gauge.append(f" {value:.2f}\n", style="cyan")
        # Line 3: Description
        gauge.append(f'"{description}"', style="italic dim")

        return gauge

    def _get_entropy_label(self, entropy: float) -> str:
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "Collapsed!"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            return "Getting decisive"
        else:
            return "Exploring"

    def _get_value_loss_label(self, value_loss: float) -> str:
        if value_loss < 0.1:
            return "Learning well"
        elif value_loss < 0.5:
            return "Still learning"
        else:
            return "Struggling"

    def _get_advantage_label(self, advantage: float) -> str:
        if advantage > 0.2:
            return "Choices working"
        elif advantage > 0:
            return "Slight edge"
        else:
            return "Needs improvement"

    def _render_recent_decisions(self) -> Panel:
        """Render Recent Decisions section (up to 3, each visible for 30s minimum).

        Stable carousel behavior:
        - Each decision stays visible for at least 30 seconds
        - Only the oldest unpinned decision can be replaced
        - Click on a decision to pin it (📌 shown in title)
        """
        from datetime import datetime, timezone
        from rich.console import Group

        tamiyo = self._snapshot.tamiyo
        decisions = tamiyo.recent_decisions

        if not decisions:
            return Panel(
                Text("No decisions captured yet\n[dim]Click to pin decisions[/dim]", style="dim italic"),
                title="RECENT DECISIONS",
                border_style="dim",
            )

        now = datetime.now(timezone.utc)
        decision_panels = []

        # Store decision IDs for click handling
        self._decision_ids = [d.decision_id for d in decisions[:3]]

        for i, decision in enumerate(decisions[:3]):
            age = (now - decision.timestamp).total_seconds()
            age_str = f"{age:.1f}s ago" if age < 60 else f"{age/60:.0f}m ago"

            # Build full decision display (like the original single panel)
            content = Table.grid(expand=True)
            content.add_column(ratio=1)

            # SAW line
            saw_line = Text()
            saw_line.append("SAW:  ", style="bold")
            for slot_id, state in decision.slot_states.items():
                saw_line.append(f"{slot_id}: {state} │ ", style="dim")
            saw_line.append(f"Host: {decision.host_accuracy:.0f}%", style="cyan")
            content.add_row(saw_line)

            # CHOSE line (with Also alternatives on same line, tab-separated)
            chose_line = Text()
            chose_line.append("CHOSE: ", style="bold")
            action_colors = {
                "GERMINATE": "green bold",
                "WAIT": "dim",
                "FOSSILIZE": "blue bold",
                "CULL": "red bold",
            }
            chose_line.append(f"{decision.chosen_action}", style=action_colors.get(decision.chosen_action, "white"))
            if decision.chosen_slot:
                chose_line.append(f" {decision.chosen_slot}", style="cyan")
            chose_line.append(f" ({decision.confidence:.0%})", style="dim")
            # Add alternatives on same line
            if decision.alternatives:
                chose_line.append("\t\tAlso: ", style="dim")
                for action, prob in decision.alternatives[:2]:
                    chose_line.append(f"{action} ({prob:.0%}) ", style="dim")
            content.add_row(chose_line)

            # EXPECTED vs GOT line
            result_line = Text()
            result_line.append("EXPECTED: ", style="dim")
            result_line.append(f"{decision.expected_value:+.2f}", style="cyan")
            result_line.append("  →  GOT: ", style="dim")
            if decision.actual_reward is not None:
                diff = decision.actual_reward - decision.expected_value
                style = "green" if abs(diff) < 0.1 else ("yellow" if diff > 0 else "red")
                result_line.append(f"{decision.actual_reward:+.2f} ", style=style)
                result_line.append("✓" if abs(diff) < 0.1 else "✗", style=style)
            else:
                result_line.append("pending...", style="dim italic")
            content.add_row(result_line)

            # Show pinned status in title
            pin_icon = "📌 " if decision.pinned else ""
            title = f"{pin_icon}DECISION {i+1} ({age_str})"
            border = "cyan" if decision.pinned else "dim"

            decision_panels.append(Panel(content, title=title, border_style=border))

        return Panel(Group(*decision_panels), title=f"RECENT DECISIONS ({len(decisions)}) [dim]click to pin[/dim]", border_style="dim")

    # ========================================================================
    # Legacy status helpers (kept for backward compatibility with existing tests)
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
