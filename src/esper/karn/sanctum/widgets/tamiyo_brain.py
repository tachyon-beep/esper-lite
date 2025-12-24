"""TamiyoBrain widget - Policy agent diagnostics (Redesigned).

New layout focuses on answering:
- "What is Tamiyo doing?" (Action distribution bar)
- "Is she learning?" (Entropy, Value Loss, KL gauges)
- "What did she just decide?" (Last Decision snapshot)
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, ClassVar

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

    # Enable keyboard focus for Tab navigation between policy widgets
    can_focus = True

    # Widget width for separators (96 - 2 for padding = 94)
    SEPARATOR_WIDTH = 94

    # Layout width constants for compact mode detection
    FULL_WIDTH = 96
    COMPACT_WIDTH = 80
    COMPACT_THRESHOLD = 85

    # Neural network architecture constant
    _TOTAL_LAYERS = 12

    # Prediction accuracy thresholds for decision cards
    PREDICTION_EXCELLENT_THRESHOLD = 0.1  # Green checkmark: |actual - expected| < 0.1
    PREDICTION_ACCEPTABLE_THRESHOLD = 0.3  # Yellow warning: |actual - expected| < 0.3

    # Per-head max entropy values from factored_actions.py (DRL CORRECTED)
    # These are ln(N) where N is the number of actions for each head
    HEAD_MAX_ENTROPIES = {
        "slot": 1.099,       # ln(3) - default SlotConfig has 3 slots
        "blueprint": 2.565,  # ln(13) - BlueprintAction has 13 values
        "style": 1.386,      # ln(4) - GerminationStyle has 4 values
        "tempo": 1.099,      # ln(3) - TempoAction has 3 values
        "alpha_target": 1.099,  # ln(3) - AlphaTargetAction has 3 values
        "alpha_speed": 1.386,   # ln(4) - AlphaSpeedAction has 4 values
        "alpha_curve": 1.099,   # ln(3) - AlphaCurveAction has 3 values
        "op": 1.792,         # ln(6) - LifecycleOp has 6 values
    }

    # Heads that PPOAgent currently tracks (others awaiting neural network changes)
    TRACKED_HEADS = {"slot", "blueprint"}

    # A/B/C testing color scheme
    # A = Green (primary/control), B = Cyan (variant), C = Magenta (second variant)
    # NOTE: Do NOT use red for C - red is reserved for error/critical states
    GROUP_COLORS: ClassVar[dict[str, str]] = {
        "A": "bright_green",
        "B": "bright_cyan",
        "C": "bright_magenta",
    }

    GROUP_LABELS: ClassVar[dict[str, str]] = {
        "A": "🟢 Policy A",
        "B": "🔵 Policy B",
        "C": "🟣 Policy C",
    }

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
        # Update border_title for accessibility (screen readers, colorblind users)
        if snapshot.tamiyo.group_id:
            # Escape opening bracket to prevent Rich markup interpretation
            self.border_title = f"TAMIYO \\[{snapshot.tamiyo.group_id}]"
        else:
            self.border_title = "TAMIYO"
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
        """Update CSS class based on overall status and A/B group."""
        status, _, _ = self._get_overall_status()

        # Remove all status classes
        self.remove_class("status-ok", "status-warning", "status-critical")

        # Add current status class
        self.add_class(f"status-{status}")

        # Add group class if in A/B mode
        if self._snapshot and self._snapshot.tamiyo.group_id:
            # Remove all group classes before adding the current one
            self.remove_class("group-a", "group-b", "group-c")
            group = self._snapshot.tamiyo.group_id.lower()
            self.add_class(f"group-{group}")

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
        if self._snapshot.tamiyo.ppo_data_received:
            diagnostic_matrix = self._render_diagnostic_matrix()
            main_table.add_row(diagnostic_matrix)
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

        # Row 5: Per-Head Entropy Heatmap (P2 cool factor)
        if self._snapshot.tamiyo.ppo_data_received:
            head_heatmap = self._render_head_heatmap()
            main_table.add_row(head_heatmap)

            # Row 6: Separator
            main_table.add_row(self._render_separator())

        # Row 7: Action Distribution
        action_bar = self._render_action_distribution_bar()
        main_table.add_row(action_bar)

        # Row 8: Separator
        main_table.add_row(self._render_separator())

        # Row 9: Decision Carousel
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

    def _render_compact_decision(self, decision: "DecisionSnapshot", index: int) -> Text:
        """Render a compact 4-line decision card with fixed 20-char width.

        Format (exactly 20 chars per line):
        ┌─ D1 12s ─────────┐
        │ WAIT 92% H:87% [P]│
        │ +0.12→+0.08 ✓    │
        └──────────────────┘

        Args:
            decision: The decision snapshot to render.
            index: 0-indexed position (0=most recent).

        Returns:
            Rich Text with fixed-width compact decision card.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        age = (now - decision.timestamp).total_seconds()
        age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"

        # Action colors
        action_colors = {
            "GERMINATE": "green",
            "WAIT": "dim",
            "FOSSILIZE": "blue",
            "PRUNE": "red",
            "SET_ALPHA_TARGET": "cyan",
            "ADVANCE": "cyan",
        }
        action_style = action_colors.get(decision.chosen_action, "white")

        # Build card with fixed width (20 chars total per line)
        card = Text()

        # Title line: Fixed width using left-padded content
        # Format: "┌─ D1 12s ─────────┐" (20 chars)
        title_content = f"D{index+1} {age_str}"
        title_fill = 20 - 4 - len(title_content)  # 20 total - "┌─ " - "┐"
        card.append(f"┌─ {title_content}{'─' * title_fill}┐\n", style="dim")

        # Line 1: ACTION PROB% H:XX% P (fixed 20 chars)
        # Format: "│ WAIT 92% H:87%P  │" or "│ WAIT 92% H:87%   │"
        # Structure: "│ " (2) + content (16) + " │" (2) = 20 total
        # When pinned, drop % from host and append P to fit in 16 chars
        action_abbrev = decision.chosen_action[:4].upper()  # WAIT, GERM, FOSS, PRUN, ADVA

        if decision.pinned:
            # Format: "GERM 100% H:100P" (exactly 16 chars worst case)
            line1_content = f"{action_abbrev} {decision.confidence:.0%} H:{decision.host_accuracy:.0f}P"
        else:
            # Format: "GERM 100% H:100%" (exactly 16 chars worst case)
            line1_content = f"{action_abbrev} {decision.confidence:.0%} H:{decision.host_accuracy:.0f}%"

        card.append("│ ")
        # Append styled segments
        card.append(f"{action_abbrev}", style=action_style)
        card.append(f" {decision.confidence:.0%}", style="dim")
        if decision.pinned:
            card.append(f" H:{decision.host_accuracy:.0f}", style="cyan")
            card.append("P", style="yellow")
        else:
            card.append(f" H:{decision.host_accuracy:.0f}%", style="cyan")

        # Padding: need to reach 16 chars total content, then add " │"
        content_len = len(line1_content)
        padding_needed = 16 - content_len
        if padding_needed > 0:
            card.append(" " * padding_needed)
        card.append(" │\n")

        # Line 2: +0.12→+0.08 ✓ (fixed 20 chars)
        # Format: "│ +0.12→+0.08 ✓   │"
        # Structure: "│ " (2) + content (16) + " │" (2) = 20 total
        card.append("│ ")
        card.append(f"{decision.expected_value:+.2f}", style="dim")
        card.append("→", style="dim")
        if decision.actual_reward is not None:
            diff = abs(decision.actual_reward - decision.expected_value)
            style = "green" if diff < self.PREDICTION_EXCELLENT_THRESHOLD else (
                "yellow" if diff < self.PREDICTION_ACCEPTABLE_THRESHOLD else "red"
            )
            icon = "✓" if diff < self.PREDICTION_EXCELLENT_THRESHOLD else "✗"
            card.append(f"{decision.actual_reward:+.2f}", style=style)
            card.append(f" {icon}", style=style)
            # Content: "+0.12" (5) + "→" (1) + "+0.08" (5) + " " (1) + "✓" (1) = 13
            content_len = 5 + 1 + 5 + 1 + 1  # 13 chars
            padding = 16 - content_len
            card.append(" " * padding)
        else:
            card.append("...", style="dim italic")
            # Content: "+0.12" (5) + "→" (1) + "..." (3) = 9 chars
            content_len = 5 + 1 + 3  # 9 chars
            padding = 16 - content_len
            card.append(" " * padding)
        card.append(" │\n")

        # Bottom border (20 chars)
        card.append("└──────────────────┘", style="dim")

        return card

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

        # Prepend group label if in A/B mode
        if tamiyo.group_id:
            group_color = self.GROUP_COLORS.get(tamiyo.group_id, "white")
            group_label = self.GROUP_LABELS.get(tamiyo.group_id, f"[{tamiyo.group_id}]")
            banner.append(f" {group_label} ", style=group_color)
            banner.append(" ┃ ", style="dim")  # Heavy vertical bar with spacing

        banner.append(f"{icon} ", style=style)
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
            healthy = self._TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
            if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
                banner.append(f"GradHP:!! {tamiyo.dead_layers}D/{tamiyo.exploding_layers}E", style="red")
            else:
                banner.append(f"GradHP:OK {healthy}/{self._TOTAL_LAYERS}", style="green")
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
        history: list[float] | deque[float],
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

    def _render_metrics_column(self) -> Text:
        """Render secondary metrics column with sparklines."""
        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Advantage stats
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        adv_style = self._status_style(adv_status)
        result.append(f" Advantage   ", style="dim")
        result.append(f"{tamiyo.advantage_mean:+.2f} ± {tamiyo.advantage_std:.2f}", style=adv_style)
        if adv_status != "ok":
            result.append(" [!]", style=adv_style)
        result.append("\n")

        # Ratio bounds
        ratio_status = self._get_ratio_status(tamiyo.ratio_min, tamiyo.ratio_max)
        ratio_style = self._status_style(ratio_status)
        result.append(f" Ratio       ", style="dim")
        result.append(f"{tamiyo.ratio_min:.2f} < r < {tamiyo.ratio_max:.2f}", style=ratio_style)
        if ratio_status != "ok":
            result.append(" [!]", style=ratio_style)
        result.append("\n")

        # Policy loss with sparkline
        pl_sparkline = self._render_sparkline(tamiyo.policy_loss_history)
        result.append(f" Policy Loss ", style="dim")
        result.append(pl_sparkline)
        result.append(f" {tamiyo.policy_loss:.3f}\n", style="bright_cyan")

        # Value loss with sparkline
        vl_sparkline = self._render_sparkline(tamiyo.value_loss_history)
        result.append(f" Value Loss  ", style="dim")
        result.append(vl_sparkline)
        result.append(f" {tamiyo.value_loss:.3f}\n", style="bright_cyan")

        # Grad norm with sparkline
        gn_sparkline = self._render_sparkline(tamiyo.grad_norm_history)
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        gn_style = self._status_style(gn_status)
        result.append(f" Grad Norm   ", style="dim")
        result.append(gn_sparkline)
        result.append(f" {tamiyo.grad_norm:.2f}\n", style=gn_style)

        # Layer health
        result.append(f" Layers      ", style="dim")
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(f"!! {tamiyo.dead_layers} dead, {tamiyo.exploding_layers} exploding", style="red")
        else:
            healthy = self._TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
            result.append(f"OK {healthy}/{self._TOTAL_LAYERS} healthy", style="green")

        return result

    def _get_ratio_status(self, ratio_min: float, ratio_max: float) -> str:
        """Get status for PPO ratio bounds."""
        if ratio_max > TUIThresholds.RATIO_MAX_CRITICAL or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL:
            return "critical"
        if ratio_max > TUIThresholds.RATIO_MAX_WARNING or ratio_min < TUIThresholds.RATIO_MIN_WARNING:
            return "warning"
        return "ok"

    def _render_diagnostic_matrix(self) -> Table:
        """Render diagnostic matrix: gauges left, metrics right."""
        matrix = Table.grid(expand=True)
        matrix.add_column(ratio=1)  # Gauges
        matrix.add_column(width=2)  # Separator
        matrix.add_column(ratio=1)  # Metrics

        gauge_grid = self._render_gauge_grid()
        separator = Text("│\n│\n│\n│\n│\n│", style="dim")
        metrics_col = self._render_metrics_column()

        matrix.add_row(gauge_grid, separator, metrics_col)
        return matrix

    def _render_head_heatmap(self) -> Text:
        """Render per-head entropy heatmap with 8 action heads.

        Shows visual distinction for heads without telemetry data.
        Per DRL review: max entropy values computed from actual action space dimensions.
        """
        tamiyo = self._snapshot.tamiyo

        # Head config: (abbrev, field_name, head_key)
        heads = [
            ("sl", "head_slot_entropy", "slot"),
            ("bp", "head_blueprint_entropy", "blueprint"),
            ("sy", "head_style_entropy", "style"),
            ("te", "head_tempo_entropy", "tempo"),
            ("at", "head_alpha_target_entropy", "alpha_target"),
            ("as", "head_alpha_speed_entropy", "alpha_speed"),
            ("ac", "head_alpha_curve_entropy", "alpha_curve"),
            ("op", "head_op_entropy", "op"),
        ]

        result = Text()
        result.append(" Heads: ", style="dim")

        # First line: bars
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            is_tracked = head_key in self.TRACKED_HEADS

            # Check for missing data (value=0.0 for untracked heads)
            if value == 0.0 and not is_tracked:
                # Visual distinction for awaiting telemetry (per risk assessor)
                # Using "---" for consistent 8-char width
                result.append(f"{abbrev}[", style="dim")
                result.append("---", style="dim italic")
                result.append("] ")
                continue

            # Normalize to 0-1
            fill = value / max_ent if max_ent > 0 else 0
            fill = max(0, min(1, fill))

            # 3-char bar (narrower for 80-char terminal compatibility)
            bar_width = 3
            filled = int(fill * bar_width)
            empty = bar_width - filled

            # Color based on fill level (high entropy = exploring, low = converged)
            if fill > 0.5:
                color = "green"
            elif fill > 0.25:
                color = "yellow"
            else:
                color = "red"

            result.append(f"{abbrev}[")
            result.append("█" * filled, style=color)
            result.append("░" * empty, style="dim")
            result.append("] ")

        result.append("\n        ")

        # Second line: values (8-char width to match bars)
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            is_tracked = head_key in self.TRACKED_HEADS

            if value == 0.0 and not is_tracked:
                # 8-char segment: "  n/a   " (2 spaces + n/a + 3 spaces)
                result.append("  n/a   ", style="dim italic")
                continue

            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            fill = value / max_ent if max_ent > 0 else 0

            # 8-char segment with indicators: critical (!), warning (*), normal
            if fill < 0.25:
                # Critical: " 1.00! " (6 chars) + 2 padding = 8 total
                result.append(f" {value:4.2f}! ", style="red")
            elif fill < 0.5:
                # Warning: " 1.00* " (6 chars) + 2 padding = 8 total
                result.append(f" {value:4.2f}* ", style="yellow")
            else:
                # Normal: " 1.00  " (6 chars) + 2 padding = 8 total
                result.append(f" {value:4.2f}  ", style="dim")

        return result

    def _render_decisions_column(self) -> Text:
        """Render vertical stack of 3 compact decision cards for right column.

        Returns:
            Rich Text with stacked compact decision cards.
        """
        tamiyo = self._snapshot.tamiyo
        decisions = tamiyo.recent_decisions

        if not decisions:
            result = Text()
            result.append("DECISIONS\n", style="dim bold")
            result.append("No decisions yet\n", style="dim italic")
            result.append("Waiting for\n", style="dim")
            result.append("agent actions...", style="dim")
            return result

        # Store decision IDs for click handling
        self._decision_ids = [d.decision_id for d in decisions[:3]]

        result = Text()
        result.append("DECISIONS\n", style="dim bold")

        for i, decision in enumerate(decisions[:3]):
            card = self._render_compact_decision(decision, index=i)
            result.append(card)
            if i < 2:  # Add spacing between cards
                result.append("\n")

        return result

    def _render_vitals_column(self) -> Table:
        """Render left 2/3 column with all learning vitals.

        Contains (top to bottom):
        - Diagnostic matrix (gauges + metrics)
        - Separator
        - Head heatmap
        - Separator
        - Action distribution bar

        Returns:
            Rich Table with vertically stacked vitals components.
        """
        tamiyo = self._snapshot.tamiyo

        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        if not tamiyo.ppo_data_received:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            content.add_row(waiting_text)
            return content

        # Row 1: Diagnostic matrix (gauges left, metrics right)
        diagnostic_matrix = self._render_diagnostic_matrix()
        content.add_row(diagnostic_matrix)

        # Row 2: Separator
        content.add_row(self._render_separator())

        # Row 3: Head heatmap
        head_heatmap = self._render_head_heatmap()
        content.add_row(head_heatmap)

        # Row 4: Separator
        content.add_row(self._render_separator())

        # Row 5: Action distribution bar
        action_bar = self._render_action_distribution_bar()
        content.add_row(action_bar)

        return content
