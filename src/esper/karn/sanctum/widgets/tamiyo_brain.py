"""TamiyoBrain widget - Policy agent diagnostics (Redesigned).

New layout focuses on answering:
- "What is Tamiyo doing?" (Action distribution bar)
- "Is she learning?" (Entropy, Value Loss, KL gauges)
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

    # Widget width for separators (96 - 2 for padding = 94)
    SEPARATOR_WIDTH = 94

    # Layout width constants for compact mode detection
    FULL_WIDTH = 96
    COMPACT_WIDTH = 80
    COMPACT_THRESHOLD = 85

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
        self._update_status_class()
        self.refresh()

    def _is_compact_mode(self) -> bool:
        """Detect if terminal is too narrow for full 96-char layout."""
        return self.size.width < self.COMPACT_THRESHOLD

    def _get_separator_width(self) -> int:
        """Get separator width based on current mode."""
        if self._is_compact_mode():
            return self.COMPACT_WIDTH - 2  # 78 chars
        return self.FULL_WIDTH - 2  # 94 chars

    def _render_separator(self) -> Text:
        """Render horizontal separator at correct width."""
        width = self._get_separator_width()
        return Text("─" * width, style="dim")

    def _update_status_class(self) -> None:
        """Update CSS class based on overall status."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok")
        self.remove_class("status-warning")
        self.remove_class("status-critical")

        # Add current status class
        self.add_class(f"status-{status}")

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
        """Render Tamiyo content with expanded layout."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        # Main layout: stacked sections
        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Row 1: Status Banner (1 line)
        status_banner = self._render_status_banner()
        main_table.add_row(status_banner)

        # Row 2: Separator (full width per UX spec)
        main_table.add_row(self._render_separator())

        # Row 3: Diagnostic Matrix (gauges left, metrics right)
        # For now, just gauges - Phase 3 adds metrics column
        if self._snapshot.tamiyo.ppo_data_received:
            gauge_grid = self._render_gauge_grid()
            main_table.add_row(gauge_grid)
        else:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            main_table.add_row(waiting_text)

        # Row 4: Separator
        main_table.add_row(self._render_separator())

        # Row 5: Action Distribution
        action_bar = self._render_action_distribution_bar()
        main_table.add_row(action_bar)

        # Row 6: Separator
        main_table.add_row(self._render_separator())

        # Row 7: Decision Carousel
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

        if not tamiyo.ppo_data_received:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            content.add_row(waiting_text)
            return Panel(content, title="LEARNING VITALS", border_style="dim")

        # Row 2: Gauges (Entropy, Value Loss, KL)
        gauges = Table.grid(expand=True)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)

        batch = self._snapshot.current_batch
        entropy_gauge = self._render_gauge(
            "Entropy", tamiyo.entropy, 0, 2.0,
            self._get_entropy_label(tamiyo.entropy, batch)
        )
        value_gauge = self._render_gauge(
            "Value Loss", tamiyo.value_loss, 0, 1.0,
            self._get_value_loss_label(tamiyo.value_loss, batch)
        )
        kl_gauge = self._render_gauge(
            "KL", tamiyo.kl_divergence, 0.0, 0.1,
            self._get_kl_label(tamiyo.kl_divergence, batch)
        )

        gauges.add_row(entropy_gauge, value_gauge, kl_gauge)
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
            "SET_ALPHA_TARGET": "cyan",
            "WAIT": "dim",
            "FOSSILIZE": "blue",
            "PRUNE": "red",
        }

        for action in ["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "WAIT"]:
            pct = pcts.get(action, 0)
            width = int((pct / 100) * bar_width)
            if width > 0:
                bar.append("▓" * width, style=colors.get(action, "white"))

        bar.append("]")

        # Fixed-width legend: G=09 A=02 F=00 P=06 W=60
        # Always show all actions with 2-digit zero-padded percentages for stability
        abbrevs = {
            "GERMINATE": "G",
            "SET_ALPHA_TARGET": "A",
            "WAIT": "W",
            "FOSSILIZE": "F",
            "PRUNE": "P",
        }
        # Build legend as separate Text for right-justification
        legend_parts = []
        for action in ["GERMINATE", "SET_ALPHA_TARGET", "FOSSILIZE", "PRUNE", "WAIT"]:
            pct = pcts.get(action, 0)
            legend_parts.append((f"{abbrevs[action]}={pct:02.0f}", colors.get(action, "white")))

        # Add spacing then fixed-width legend
        bar.append(" ")
        for i, (text, color) in enumerate(legend_parts):
            bar.append(text, style=color)
            if i < len(legend_parts) - 1:
                bar.append(" ", style="dim")

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
        # Use more precision for small values (like KL divergence in 0.001-0.015 range)
        if max_val < 1.0 and value < 0.1:
            gauge.append(f" {value:.4f}\n", style="cyan")
        else:
            gauge.append(f" {value:.2f}\n", style="cyan")
        # Line 3: Description
        gauge.append(f'"{description}"', style="italic dim")

        return gauge

    def _get_entropy_label(self, entropy: float, batch: int = 0) -> str:
        """Batch-aware entropy label."""
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "Collapsed!"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            if batch < 20:
                return "Focusing early"
            return "Getting decisive"
        else:
            if batch < 10:
                return "Warming up"
            return "Exploring"

    def _get_value_loss_label(self, value_loss: float, batch: int = 0) -> str:
        """Batch-aware value loss label.

        Early training: High loss is expected (value function learning)
        Mid training: Should be decreasing
        Late training: Should be stable and low
        """
        # Early training (batch < 20): lenient thresholds
        if batch < 20:
            if value_loss < 1.0:
                return "Good for early"
            elif value_loss < 5.0:
                return "Learning (expected)"
            else:
                return "High (normal early)"

        # Mid training (batch 20-100): moderate expectations
        elif batch < 100:
            if value_loss < 0.5:
                return "Learning well"
            elif value_loss < 2.0:
                return "Progressing"
            elif value_loss < 5.0:
                return "Slow progress"
            else:
                return "Check rewards"

        # Late training (batch 100+): strict expectations
        else:
            if value_loss < 0.1:
                return "Excellent"
            elif value_loss < 0.5:
                return "Learning well"
            elif value_loss < 1.0:
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

    def _get_kl_label(self, kl_divergence: float, batch: int = 0) -> str:
        """Batch-aware KL divergence label.

        KL = 0 early is expected; later it should be small but non-zero.
        """
        if kl_divergence > TUIThresholds.KL_WARNING:
            return "Too fast"
        elif kl_divergence < 0.0001:
            if batch < 5:
                return "Starting up"
            else:
                return "Not updating?"
        elif kl_divergence < 0.005:
            return "Very stable"
        else:
            return "Stable"

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
                "PRUNE": "red bold",
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

    def _render_status_banner(self) -> Text:
        """Render 1-line status banner with icon and key metrics.

        Format per UX spec:
        [OK] LEARNING   EV:0.72 Clip:0.18 KL:0.008 Adv:0.12±0.94 GradHP:OK 12/12 batch:47/100
        """
        status, label, style = self._get_overall_status()
        tamiyo = self._snapshot.tamiyo

        icons = {"ok": "[OK]", "warning": "[!]", "critical": "[X]"}
        icon = icons.get(status, "?")

        banner = Text()
        banner.append(f" {icon} ", style=style)
        banner.append(f"{label}   ", style=style)

        if tamiyo.ppo_data_received:
            # EV with warning indicator
            ev_style = self._status_style(self._get_ev_status(tamiyo.explained_variance))
            banner.append(f"EV:{tamiyo.explained_variance:.2f}", style=ev_style)
            if tamiyo.explained_variance <= 0:
                banner.append("!", style="red")
            banner.append("  ")

            # Clip
            clip_style = self._status_style(self._get_clip_status(tamiyo.clip_fraction))
            banner.append(f"Clip:{tamiyo.clip_fraction:.2f}", style=clip_style)
            if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
                banner.append("!", style=clip_style)
            banner.append("  ")

            # KL
            kl_style = self._status_style(self._get_kl_status(tamiyo.kl_divergence))
            banner.append(f"KL:{tamiyo.kl_divergence:.3f}", style=kl_style)
            if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
                banner.append("!", style=kl_style)
            banner.append("  ")

            # Advantage summary (per UX spec)
            adv_status = self._get_advantage_status(tamiyo.advantage_std)
            adv_style = self._status_style(adv_status)
            banner.append(f"Adv:{tamiyo.advantage_mean:+.2f}±{tamiyo.advantage_std:.2f}", style=adv_style)
            if adv_status != "ok":
                banner.append("!", style=adv_style)
            banner.append("  ")

            # Gradient health summary (per UX spec)
            total_layers = 12  # Approximate
            healthy = total_layers - tamiyo.dead_layers - tamiyo.exploding_layers
            if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
                banner.append(f"GradHP:!! {tamiyo.dead_layers}D/{tamiyo.exploding_layers}E", style="red")
            else:
                banner.append(f"GradHP:OK {healthy}/{total_layers}", style="green")
            banner.append("  ")

            # Batch progress with denominator (per UX spec)
            batch = self._snapshot.current_batch
            max_batches = self._snapshot.max_batches
            banner.append(f"batch:{batch}/{max_batches}", style="dim")

        return banner

    # ========================================================================
    # Decision Tree Logic
    # ========================================================================

    def _get_overall_status(self) -> tuple[str, str, str]:
        """Get overall learning status using DRL decision tree.

        Priority order (per DRL expert review):
        1. Entropy collapse (policy dead)
        2. EV <= 0 (value harmful)
        3. Advantage std collapsed (normalization broken)
        4. KL > critical (excessive policy change)
        5. Clip > critical (too aggressive)
        6. Grad norm > critical (gradient explosion)
        7. EV < warning (value weak)
        8. KL > warning (mild drift)
        9. Clip > warning
        10. Entropy low
        11. Advantage abnormal
        12. Grad norm > warning

        Returns:
            Tuple of (status, label, style) where:
            - status: "ok", "warning", or "critical"
            - label: "LEARNING", "CAUTION", or "FAILING"
            - style: Rich style string for coloring
        """
        if self._snapshot is None:
            return "ok", "WAITING", "dim"

        tamiyo = self._snapshot.tamiyo

        if not tamiyo.ppo_data_received:
            return "ok", "WAITING", "dim"

        # === CRITICAL CHECKS (immediate FAILING) ===

        # 1. Entropy collapse (policy is deterministic/dead)
        if tamiyo.entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 2. EV <= 0 (value function useless or harmful)
        if tamiyo.explained_variance <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 3. Advantage std collapsed (normalization broken)
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical", "FAILING", "red bold"

        # 4. Advantage std exploded
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 5. KL > critical (excessive policy change)
        if tamiyo.kl_divergence > TUIThresholds.KL_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 6. Clip > critical (updates too aggressive)
        if tamiyo.clip_fraction > TUIThresholds.CLIP_CRITICAL:
            return "critical", "FAILING", "red bold"

        # 7. Grad norm > critical (gradient explosion)
        if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical", "FAILING", "red bold"

        # === WARNING CHECKS (CAUTION) ===

        # 8. EV < warning (value function weak but learning)
        if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning", "CAUTION", "yellow"

        # 9. Entropy low (policy converging quickly)
        if tamiyo.entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning", "CAUTION", "yellow"

        # 10. KL > warning (mild policy drift)
        if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
            return "warning", "CAUTION", "yellow"

        # 11. Clip > warning
        if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
            return "warning", "CAUTION", "yellow"

        # 12. Advantage std abnormal
        if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning", "CAUTION", "yellow"
        if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            return "warning", "CAUTION", "yellow"

        # 13. Grad norm > warning
        if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning", "CAUTION", "yellow"

        return "ok", "LEARNING", "green"

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
        if kl_div > TUIThresholds.KL_CRITICAL:
            return "critical"
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

    def _get_advantage_status(self, adv_std: float) -> str:
        """Get status for advantage normalization."""
        if adv_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning"
        if adv_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
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

    def _render_gauge_grid(self) -> Table:
        """Render 2x2 gauge grid: EV, Entropy, Clip, KL."""
        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Row 1: Explained Variance | Entropy
        ev_gauge = self._render_gauge_v2(
            "Expl.Var",
            tamiyo.explained_variance,
            min_val=-1.0,
            max_val=1.0,
            status=self._get_ev_status(tamiyo.explained_variance),
            label_text=self._get_ev_label(tamiyo.explained_variance),
        )
        entropy_gauge = self._render_gauge_v2(
            "Entropy",
            tamiyo.entropy,
            min_val=0.0,
            max_val=2.0,
            status=self._get_entropy_status(tamiyo.entropy),
            label_text=self._get_entropy_label(tamiyo.entropy, batch),
        )
        grid.add_row(ev_gauge, entropy_gauge)

        # Row 2: Clip Fraction | KL Divergence
        clip_gauge = self._render_gauge_v2(
            "Clip Frac",
            tamiyo.clip_fraction,
            min_val=0.0,
            max_val=0.5,
            status=self._get_clip_status(tamiyo.clip_fraction),
            label_text=self._get_clip_label(tamiyo.clip_fraction),
        )
        kl_gauge = self._render_gauge_v2(
            "KL Div",
            tamiyo.kl_divergence,
            min_val=0.0,
            max_val=0.1,
            status=self._get_kl_status(tamiyo.kl_divergence),
            label_text=self._get_kl_label(tamiyo.kl_divergence, batch),
        )
        grid.add_row(clip_gauge, kl_gauge)

        return grid

    def _render_gauge_v2(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        status: str,
        label_text: str,
    ) -> Text:
        """Render a gauge with status-colored bar."""
        # Normalize to 0-1
        if max_val != min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        normalized = max(0, min(1, normalized))

        gauge_width = 10
        filled = int(normalized * gauge_width)
        empty = gauge_width - filled

        # Status-based color (use bright_cyan for OK per UX spec)
        bar_color = {"ok": "bright_cyan", "warning": "yellow", "critical": "red"}[status]

        gauge = Text()
        gauge.append(f" {label}\n", style="dim")
        gauge.append(" [")
        gauge.append("█" * filled, style=bar_color)
        gauge.append("░" * empty, style="dim")
        gauge.append("] ")

        # Value with precision based on magnitude
        if abs(value) < 0.1:
            gauge.append(f"{value:.3f}", style=bar_color)
        else:
            gauge.append(f"{value:.2f}", style=bar_color)

        if status == "critical":
            gauge.append("!", style="red bold")

        gauge.append(f'\n  "{label_text}"', style="italic dim")

        return gauge

    def _get_ev_label(self, ev: float) -> str:
        """Get descriptive label for explained variance."""
        if ev <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "HARMFUL!"
        elif ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "Uncertain"
        elif ev < 0.5:
            return "Improving"
        else:
            return "Learning!"

    def _get_clip_label(self, clip: float) -> str:
        """Get descriptive label for clip fraction."""
        if clip > TUIThresholds.CLIP_CRITICAL:
            return "TOO AGGRESSIVE!"
        elif clip > TUIThresholds.CLIP_WARNING:
            return "Aggressive"
        elif clip < 0.1:
            return "Very stable"
        else:
            return "Stable"

    def _render_sparkline(
        self,
        history: list[float] | "deque[float]",
        width: int = 10,
        style: str = "bright_cyan",
    ) -> Text:
        """Render sparkline using unicode block characters.

        Args:
            history: Historical values to visualize
            width: Maximum width in characters
            style: Rich style for the blocks

        Returns:
            Text with sparkline or placeholder for empty/flat data.
        """
        BLOCKS = "▁▂▃▄▅▆▇█"

        if not history:
            return Text("─" * width, style="dim")

        values = list(history)[-width:]  # Last N values
        if len(values) < width:
            # Pad with placeholder on left
            pad_count = width - len(values)
            result = Text("─" * pad_count, style="dim")
        else:
            result = Text()

        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        val_range = max_val - min_val if max_val != min_val else 1

        for v in values:
            normalized = (v - min_val) / val_range
            idx = int(normalized * (len(BLOCKS) - 1))
            idx = max(0, min(len(BLOCKS) - 1, idx))
            result.append(BLOCKS[idx], style=style)

        return result
