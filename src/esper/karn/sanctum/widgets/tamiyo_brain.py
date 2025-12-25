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
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


def detect_action_patterns(
    decisions: list["DecisionSnapshot"],
    slot_states: dict[str, str],
) -> list[str]:
    """Detect problematic action patterns with DRL-informed logic.

    Per DRL review:
    - STUCK = all WAIT when dormant slots exist (actionable opportunities)
    - THRASH = germinate→prune cycles (wasted compute)
    - ALPHA_OSC = too many alpha changes without completion

    Per UX review: Returns pattern names for icon display.

    NOTE: decisions list is newest-first (aggregator uses insert(0, decision)).
    We reverse to chronological order for pattern analysis so that:
    - actions[-8:] checks the 8 MOST RECENT actions
    - THRASH detection finds GERMINATE→PRUNE (not PRUNE→GERMINATE)

    Returns:
        List of pattern names: ["STUCK"], ["THRASH"], ["ALPHA_OSC"]
    """
    patterns = []
    if not decisions:
        return patterns

    # Reverse to chronological order: oldest first, newest last
    # So actions[-8:] = most recent 8, and i→i+1 = earlier→later
    actions = [d.chosen_action for d in reversed(decisions[:12])]

    # STUCK: All WAIT when dormant slots exist (per DRL review)
    if len(actions) >= 8 and all(a == "WAIT" for a in actions[-8:]):
        has_dormant = any(
            "Dormant" in str(s) or "Empty" in str(s) for s in slot_states.values()
        )
        has_training = any("Training" in str(s) for s in slot_states.values())
        # Stuck = waiting when we COULD germinate (dormant exists, nothing training)
        if has_dormant and not has_training:
            patterns.append("STUCK")

    # THRASH: Germinate-Prune cycles (per DRL review)
    germ_prune_cycles = 0
    for i in range(len(actions) - 1):
        if actions[i] == "GERMINATE" and actions[i + 1] == "PRUNE":
            germ_prune_cycles += 1
    if germ_prune_cycles >= 2:
        patterns.append("THRASH")

    # ALPHA_OSC: Too many alpha changes (per DRL review)
    alpha_count = sum(1 for a in actions if a == "SET_ALPHA_TARGET")
    if alpha_count >= 4:
        patterns.append("ALPHA_OSC")

    return patterns


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

    # Layout mode thresholds
    HORIZONTAL_THRESHOLD = 96  # Full side-by-side
    COMPACT_HORIZONTAL_THRESHOLD = 85  # Compressed side-by-side

    # Neural network architecture constant
    _TOTAL_LAYERS = 12

    # Sparkline width for trend visibility (Task 1)
    SPARKLINE_WIDTH = 35

    # Prediction accuracy thresholds for decision cards
    PREDICTION_EXCELLENT_THRESHOLD = 0.1  # Green checkmark: |actual - expected| < 0.1
    PREDICTION_ACCEPTABLE_THRESHOLD = 0.3  # Yellow warning: |actual - expected| < 0.3

    # Enriched decision card width (Task 5)
    DECISION_CARD_WIDTH = 30

    # Decision card height for dynamic count calculation
    DECISION_CARD_HEIGHT = 7  # 6 lines (title + 5 content) + 1 gap

    # Per-head max entropy values from factored_actions.py (DRL CORRECTED)
    # These are ln(N) where N is the number of actions for each head
    HEAD_MAX_ENTROPIES = {
        "slot": 1.099,  # ln(3) - default SlotConfig has 3 slots
        "blueprint": 2.565,  # ln(13) - BlueprintAction has 13 values
        "style": 1.386,  # ln(4) - GerminationStyle has 4 values
        "tempo": 1.099,  # ln(3) - TempoAction has 3 values
        "alpha_target": 1.099,  # ln(3) - AlphaTargetAction has 3 values
        "alpha_speed": 1.386,  # ln(4) - AlphaSpeedAction has 4 values
        "alpha_curve": 1.099,  # ln(3) - AlphaCurveAction has 3 values
        "op": 1.792,  # ln(6) - LifecycleOp has 6 values
    }

    # All 8 action heads are tracked by PPO (per-head entropy from factored policy)
    TRACKED_HEADS = {
        "slot",
        "blueprint",
        "style",
        "tempo",
        "alpha_target",
        "alpha_speed",
        "alpha_curve",
        "op",
    }

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

    # Decision card display throttling: ONE card swap per interval, maximum
    CARD_SWAP_INTERVAL = 30.0  # Minimum seconds between card replacements

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._decision_ids: list[str] = []  # IDs of currently displayed decisions
        self.border_title = "TAMIYO"  # Top-left title like EventLog

        # Display throttling state: stable buffer updated ONE card at a time
        self._displayed_decisions: list["DecisionSnapshot"] = []
        self._last_card_swap_time: float = 0.0

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

    def _get_max_decision_cards(self) -> int:
        """Calculate how many decision cards fit in available height.

        Returns:
            Number of decision cards that fit (clamped between 3 and 8).
        """
        # Get widget content height (approximate based on min-height)
        # min-height is 24 in CSS, but actual may be larger
        available_height = max(24, self.size.height - 2)  # -2 for borders

        # Reserve space for "DECISIONS" header (1 line)
        content_height = available_height - 1

        # Calculate how many cards fit
        max_cards = content_height // self.DECISION_CARD_HEIGHT

        # Clamp between 3 and 8 (reasonable bounds)
        return max(3, min(8, max_cards))

    def _get_layout_mode(self) -> str:
        """Determine layout mode based on terminal width.

        Returns:
            - "horizontal": Full side-by-side (≥96 chars)
            - "compact-horizontal": Compressed side-by-side (85-95 chars)
            - "stacked": Vertical stack fallback (<85 chars)
        """
        width = self.size.width
        if width >= self.HORIZONTAL_THRESHOLD:
            return "horizontal"
        elif width >= self.COMPACT_HORIZONTAL_THRESHOLD:
            return "compact-horizontal"
        else:
            return "stacked"

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

        In horizontal layout: decisions are in right 1/3 column
        In stacked layout: decisions are at bottom after vitals
        """
        if not self._decision_ids:
            return

        layout_mode = self._get_layout_mode()

        if layout_mode in ("horizontal", "compact-horizontal"):
            # In horizontal layout, check if click is in right 1/3
            widget_width = self.size.width
            decision_column_start = int(widget_width * 2 / 3)

            if event.x < decision_column_start:
                return  # Click was in vitals column, not decisions

            # Calculate which decision card based on Y position
            # Each compact card is ~4 lines (title + 2 content + gap)
            header_height = 2  # Status banner + separator
            card_height = 4
            decision_y = event.y - header_height
            decision_index = max(0, decision_y // card_height)

        else:
            # Stacked layout: original click handling
            vitals_height = 7  # Approximate height of Learning Vitals section
            decision_height = 5  # Each full decision panel height

            y = event.y
            if y < vitals_height:
                return  # Click was in Learning Vitals

            decision_y = y - vitals_height
            decision_index = decision_y // decision_height

        if 0 <= decision_index < len(self._decision_ids):
            decision_id = self._decision_ids[decision_index]
            self.post_message(self.DecisionPinToggled(decision_id))

    def render(self):
        """Render Tamiyo content with responsive layout.

        Layout modes:
        - horizontal (≥96 chars): Side-by-side [vitals 2/3 | decisions 1/3]
        - compact-horizontal (85-95 chars): Compressed side-by-side
        - stacked (<85 chars): Vertical stack (legacy layout)
        """
        if self._snapshot is None:
            return Text("No data", style="dim")

        layout_mode = self._get_layout_mode()

        # Main layout container
        main_table = Table.grid(expand=True)

        # Row 1: Status Banner (always full width)
        main_table.add_column(ratio=1)
        status_banner = self._render_status_banner()
        main_table.add_row(status_banner)

        # Row 2: Separator
        main_table.add_row(self._render_separator())

        # Row 3: Content (layout-dependent)
        if layout_mode in ("horizontal", "compact-horizontal"):
            # Side-by-side: vitals left (expanding), decisions right (fixed width)
            # Decisions column is fixed at card width + padding to avoid whitespace
            # Extra space goes to vitals (sparklines, gauges) not empty decisions area
            content_table = Table.grid(expand=True)
            content_table.add_column(ratio=1)  # Vitals (expands to fill)
            content_table.add_column(width=1)  # Separator
            content_table.add_column(
                width=self.DECISION_CARD_WIDTH + 4
            )  # Decisions (fixed)

            vitals_col = self._render_vitals_column()
            separator = Text("│\n" * 15, style="dim")  # Vertical separator
            decisions_col = self._render_decisions_column()

            content_table.add_row(vitals_col, separator, decisions_col)
            main_table.add_row(content_table)
        else:
            # Stacked layout (legacy)
            main_table.add_row(self._render_stacked_content())

        return main_table

    def _render_stacked_content(self) -> Table:
        """Render legacy stacked layout for narrow terminals.

        Preserves original vertical stack behavior for <85 char terminals.
        """
        tamiyo = self._snapshot.tamiyo

        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        # Diagnostic Matrix (gauges + metrics)
        if tamiyo.ppo_data_received:
            diagnostic_matrix = self._render_diagnostic_matrix()
            content.add_row(diagnostic_matrix)
        else:
            waiting_text = Text(style="dim italic")
            waiting_text.append("⏳ Waiting for PPO vitals\n")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            content.add_row(waiting_text)

        # Separator
        content.add_row(self._render_separator())

        # Head heatmap
        if tamiyo.ppo_data_received:
            content.add_row(self._render_head_heatmap())
            content.add_row(self._render_separator())

        # Action bar
        content.add_row(self._render_action_distribution_bar())
        content.add_row(self._render_separator())

        # Decision cards (uses dynamic count and enriched format)
        content.add_row(self._render_decisions_column())

        return content

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
            "Entropy",
            tamiyo.entropy,
            0,
            2.0,
            self._get_entropy_label(tamiyo.entropy, batch),
        )
        value_gauge = self._render_gauge(
            "Value Loss",
            tamiyo.value_loss,
            0,
            1.0,
            self._get_value_loss_label(tamiyo.value_loss, batch),
        )
        kl_gauge = self._render_gauge(
            "KL",
            tamiyo.kl_divergence,
            0.0,
            0.1,
            self._get_kl_label(tamiyo.kl_divergence, batch),
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
            legend_parts.append(
                (f"{abbrevs[action]}={pct:02.0f}", colors.get(action, "white"))
            )

        # Add spacing then fixed-width legend
        bar.append(" ")
        for i, (text, color) in enumerate(legend_parts):
            bar.append(text, style=color)
            if i < len(legend_parts) - 1:
                bar.append(" ", style="dim")

        return bar

    def _render_return_history(self) -> Text:
        """Render recent episode returns for quick reference.

        Format: Returns: Ep47:+12.3  Ep46:+8.1  Ep45:-2.4  ...
        """
        tamiyo = self._snapshot.tamiyo
        history = list(tamiyo.episode_return_history)

        if not history:
            return Text("Returns: (no episodes yet)", style="dim italic")

        result = Text()
        result.append("Returns: ", style="dim")

        # Show last 6 returns (most recent first), with episode numbers
        current_ep = tamiyo.current_episode

        # Take last 6 values, reverse to show most recent first
        recent_returns = history[-6:][::-1]

        for i, ret in enumerate(recent_returns):
            ep_num = current_ep - i
            # Color code: green for positive, red for negative
            style = "green" if ret >= 0 else "red"
            result.append(f"Ep{ep_num}:", style="dim")
            result.append(f"{ret:+.1f}", style=style)
            if i < len(recent_returns) - 1:
                result.append("  ", style="dim")  # spacing between entries

        return result

    def _render_slot_summary(self) -> Text:
        """Render aggregate slot state across all environments.

        Shows distribution of slots by lifecycle stage to explain Tamiyo's actions.
        Format:
            SLOTS (96 across 32 envs)
            ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
            │ DORMANT  │GERMINATED│ TRAINING │ BLENDING │ HOLDING  │FOSSILIZED│
            │    42    │     8    │    31    │    12    │     3    │     0    │
            └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
            Fossilized: 14  Pruned: 6  Rate: 70%  AvgAge: 8.2 epochs
        """
        snapshot = self._snapshot
        counts = snapshot.slot_stage_counts
        total = snapshot.total_slots

        if total == 0:
            return Text("SLOTS: (no environments)", style="dim italic")

        result = Text()

        # Header line
        n_envs = len(snapshot.envs) if snapshot.envs else 0
        result.append(f"SLOTS ({total} across {n_envs} envs)\n", style="bold")

        # Stage distribution with mini-bars
        # Calculate proportions for visual bars (max 8 chars per stage)
        stages = [
            "DORMANT",
            "GERMINATED",
            "TRAINING",
            "BLENDING",
            "HOLDING",
            "FOSSILIZED",
        ]
        stage_colors = {
            "DORMANT": "dim",
            "GERMINATED": "green",
            "TRAINING": "cyan",
            "BLENDING": "yellow",
            "HOLDING": "bright_cyan",
            "FOSSILIZED": "blue",
        }
        stage_abbrevs = {
            "DORMANT": "DORM",
            "GERMINATED": "GERM",
            "TRAINING": "TRAIN",
            "BLENDING": "BLEND",
            "HOLDING": "HOLD",
            "FOSSILIZED": "FOSS",
        }

        # Build distribution line: "DORM:48 ░░░░ │ GERM:4 ░ │ TRAIN:28 ████ │ ..."
        for i, stage in enumerate(stages):
            count = counts.get(stage, 0)
            abbrev = stage_abbrevs[stage]
            color = stage_colors[stage]

            # Proportional bar (max 4 chars)
            bar_width = min(4, int((count / max(1, total)) * 16)) if total > 0 else 0
            bar_char = "█" if stage != "DORMANT" else "░"
            bar = bar_char * bar_width

            result.append(f"{abbrev}:", style="dim")
            result.append(f"{count}", style=color)
            if bar:
                result.append(f" {bar}", style=color)

            if i < len(stages) - 1:
                result.append("  ", style="dim")

        result.append("\n")

        # Summary line with lifecycle stats
        foss = snapshot.cumulative_fossilized
        pruned = snapshot.cumulative_pruned
        rate = (foss / max(1, foss + pruned)) * 100 if (foss + pruned) > 0 else 0
        avg_epochs = snapshot.avg_epochs_in_stage

        result.append("Foss:", style="dim")
        result.append(f"{foss}", style="blue")
        result.append("  Prune:", style="dim")
        result.append(f"{pruned}", style="red" if pruned > foss else "dim")
        result.append("  Rate:", style="dim")
        rate_color = "green" if rate >= 70 else "yellow" if rate >= 50 else "red"
        result.append(f"{rate:.0f}%", style=rate_color)
        result.append("  AvgAge:", style="dim")
        result.append(f"{avg_epochs:.1f}", style="cyan")
        result.append(" epochs", style="dim")

        # Constraint insight line (explains WHY Tamiyo behaves this way)
        dormant = counts.get("DORMANT", 0)
        if dormant == 0:
            result.append("\n")
            result.append(
                "→ No DORMANT slots: GERMINATE blocked", style="yellow italic"
            )
        elif dormant >= total * 0.5:
            result.append("\n")
            result.append(
                f"→ {dormant} DORMANT: GERMINATE opportunities available",
                style="green italic",
            )

        return result

    def _render_action_sequence(self) -> Text:
        """Render action sequence with two rows for more history.

        Format:
            Recent:  W W G W W W F W W W W W  (actions 0-11, most recent)
            Prior:   G G F P W W W G F W W W  (actions 12-23, older)

        With pattern warnings: "⚠ STUCK" or "⚡ THRASH"

        Per UX review: Icons for accessibility (not just color).
        """
        decisions = self._snapshot.tamiyo.recent_decisions[:24]
        if not decisions:
            return Text("Recent:  (no actions yet)", style="dim italic")

        # Get current slot states for pattern detection
        slot_states = {}
        if decisions:
            slot_states = decisions[0].slot_states

        # Detect patterns on recent 12 actions
        patterns = detect_action_patterns(decisions[:12], slot_states)

        # Action to single-char abbreviation with colors
        action_map = {
            "GERMINATE": ("G", "green"),
            "WAIT": ("W", "dim"),
            "FOSSILIZE": ("F", "blue"),
            "PRUNE": ("P", "red"),
            "SET_ALPHA_TARGET": ("A", "cyan"),
            "ADVANCE": ("→", "cyan"),
        }

        # Highlight based on pattern
        is_stuck = "STUCK" in patterns
        is_thrash = "THRASH" in patterns

        result = Text()

        # Recent row (most recent 12)
        recent_decisions = decisions[:12]
        recent_actions = [
            action_map.get(d.chosen_action, ("?", "white")) for d in recent_decisions
        ]
        recent_actions.reverse()  # Oldest first for left-to-right reading

        result.append("Recent:  ", style="dim")
        for char, color in recent_actions:
            style = "yellow" if is_stuck else ("red" if is_thrash else color)
            result.append(char + " ", style=style)

        # Pattern warnings
        if is_stuck:
            result.append(" ⚠ STUCK", style="yellow bold")
        if is_thrash:
            result.append(" ⚡ THRASH", style="red bold")
        if "ALPHA_OSC" in patterns:
            result.append(" ↔ ALPHA", style="cyan bold")

        result.append("\n")

        # Prior row (older 12)
        prior_decisions = decisions[12:24]
        if prior_decisions:
            prior_actions = [
                action_map.get(d.chosen_action, ("?", "white")) for d in prior_decisions
            ]
            prior_actions.reverse()

            result.append("Prior:   ", style="dim")
            for char, color in prior_actions:
                result.append(
                    char + " ", style=color
                )  # No pattern highlighting on prior

        return result

    def _render_gauge(
        self, label: str, value: float, min_val: float, max_val: float, description: str
    ) -> Text:
        """Render a single gauge with label and description on separate lines."""
        # Normalize to 0-1
        normalized = (
            (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        )
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

    def _render_enriched_decision(
        self,
        decision: "DecisionSnapshot",
        index: int,
        total_cards: int = 3,
    ) -> Text:
        """Render an enriched 6-line decision card (24 chars wide).

        Age-based border colors (per UX review):
        - Newest (index 0): cyan border - fresh, actionable
        - Middle: dim grey border - intermediate
        - Oldest (index == total-1): yellow border - aging out soon

        Format per DRL + UX reviews:
        ┌─ D1 16s ──────────────┐
        │ WAIT s:1 100%         │  Action, slot, confidence
        │ H:25% ent:0.85        │  Host accuracy, decision entropy
        │ V:+0.45 A:-0.12       │  Value estimate, advantage (NEW)
        │ -0.68→+0.00 ✓ HIT     │  Expected vs actual + text (NEW)
        │ alt: G:12% P:8%       │  Alternatives
        └───────────────────────┘
        """
        from datetime import datetime, timezone

        CONTENT_WIDTH = self.DECISION_CARD_WIDTH - 4  # "│ " + content + " │"

        # Age-based border colors (per UX review):
        # - Newest (index 0): cyan - fresh, actionable
        # - Middle: dim grey - intermediate
        # - Oldest (index == total-1): yellow - aging out soon
        if total_cards <= 1:
            border_style = "cyan"  # Single card is always "newest"
        elif index == 0:
            border_style = "cyan"  # Newest
        elif index == total_cards - 1:
            border_style = "yellow"  # Oldest
        else:
            border_style = "dim"  # Middle

        now = datetime.now(timezone.utc)
        age = (now - decision.timestamp).total_seconds()
        # Precise formatting: "45s" for <60s, "1:35" for >=60s (no rounding confusion)
        if age < 60:
            age_str = f"{age:.0f}s"
        else:
            mins = int(age // 60)
            secs = int(age % 60)
            age_str = f"{mins}:{secs:02d}"

        action_colors = {
            "GERMINATE": "green",
            "WAIT": "dim",
            "FOSSILIZE": "blue",
            "PRUNE": "red",
            "SET_ALPHA_TARGET": "cyan",
            "ADVANCE": "cyan",
        }
        action_style = action_colors.get(decision.chosen_action, "white")

        card = Text()

        # Title: ┌─ D1 16s ──────────────┐
        title = f"D{index + 1} {age_str}"
        fill = self.DECISION_CARD_WIDTH - 4 - len(title)
        card.append(f"┌─ {title}{'─' * fill}┐\n", style=border_style)

        # Line 1: ACTION s:N CONF%
        action_abbrev = decision.chosen_action[:4].upper()
        slot_num = decision.chosen_slot[-1] if decision.chosen_slot else "-"
        line1 = f"{action_abbrev} s:{slot_num} {decision.confidence:.0%}"
        card.append("│", style=border_style)
        card.append(" ")
        card.append(action_abbrev, style=action_style)
        card.append(f" s:{slot_num}", style="cyan")
        card.append(f" {decision.confidence:.0%}", style="dim")
        card.append(" " * max(0, CONTENT_WIDTH - len(line1)) + " ")
        card.append("│", style=border_style)
        card.append("\n")

        # Line 2: H:XX% ent:X.XX (decision entropy, per DRL review)
        line2 = f"H:{decision.host_accuracy:.0f}% ent:{decision.decision_entropy:.2f}"
        card.append("│", style=border_style)
        card.append(" ")
        card.append(f"H:{decision.host_accuracy:.0f}%", style="cyan")
        card.append(f" ent:{decision.decision_entropy:.2f}", style="dim")
        card.append(" " * max(0, CONTENT_WIDTH - len(line2)) + " ")
        card.append("│", style=border_style)
        card.append("\n")

        # Line 3: V:+X.XX δ:+X.XX (per DRL review)
        # V = expected_value = V(s) value function estimate
        # δ = value_residual = r - V(s) = prediction error
        line3 = (
            f"V:{decision.expected_value:+.2f} \u03b4:{decision.value_residual:+.2f}"
        )
        card.append("│", style=border_style)
        card.append(" ")
        card.append(f"V:{decision.expected_value:+.2f}", style="cyan")
        card.append(f" \u03b4:{decision.value_residual:+.2f}", style="magenta")
        card.append(" " * max(0, CONTENT_WIDTH - len(line3)) + " ")
        card.append("│", style=border_style)
        card.append("\n")

        # Line 4: r:+X.XX TD:+X.XX ✓ HIT / ✗ MISS (per DRL + UX review)
        # r = actual_reward = immediate reward received
        # TD = td_advantage = r + γV(s') - V(s) = true TD(0) advantage
        # HIT/MISS based on prediction accuracy |r - V(s)|
        card.append("│", style=border_style)
        card.append(" ")
        if decision.actual_reward is not None:
            diff = abs(decision.actual_reward - decision.expected_value)
            is_hit = diff < self.PREDICTION_EXCELLENT_THRESHOLD
            style = "green" if is_hit else "red"
            icon = "✓" if is_hit else "✗"
            # Show reward
            card.append(f"r:{decision.actual_reward:+.2f}", style=style)
            # Show TD advantage if computed (requires next step's V(s'))
            if decision.td_advantage is not None:
                card.append(f" TD:{decision.td_advantage:+.2f}", style="bright_cyan")
                line4 = f"r:{decision.actual_reward:+.2f} TD:{decision.td_advantage:+.2f} {icon}"
            else:
                card.append(" TD:...", style="dim italic")
                line4 = f"r:{decision.actual_reward:+.2f} TD:... {icon}"
            card.append(f" {icon}", style=style)
        else:
            card.append("r:... TD:...", style="dim italic")
            line4 = "r:... TD:..."
        card.append(" " * max(0, CONTENT_WIDTH - len(line4)) + " ")
        card.append("│", style=border_style)
        card.append("\n")

        # Line 5: alt: G:12% P:8%
        card.append("│", style=border_style)
        card.append(" ")
        if decision.alternatives:
            alt_strs = [f"{a[0]}:{p:.0%}" for a, p in decision.alternatives[:2]]
            line5 = "alt: " + " ".join(alt_strs)
            card.append("alt: ", style="dim")
            for i, (alt_action, prob) in enumerate(decision.alternatives[:2]):
                if i > 0:
                    card.append(" ", style="dim")
                alt_style = action_colors.get(alt_action, "dim")
                card.append(f"{alt_action[0]}:{prob:.0%}", style=alt_style)
        else:
            line5 = "alt: -"
            card.append("alt: -", style="dim")
        card.append(" " * max(0, CONTENT_WIDTH - len(line5)) + " ")
        card.append("│", style=border_style)
        card.append("\n")

        # Bottom border
        card.append(
            "└" + "─" * (self.DECISION_CARD_WIDTH - 2) + "┘", style=border_style
        )

        return card

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
            ev_style = self._status_style(
                self._get_ev_status(tamiyo.explained_variance)
            )
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
            banner.append(
                f"Adv:{tamiyo.advantage_mean:+.2f}±{tamiyo.advantage_std:.2f}",
                style=adv_style,
            )
            if adv_status != "ok":
                banner.append("!", style=adv_style)
            banner.append("  ")

            # Gradient health summary (per UX spec)
            healthy = self._TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
            if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
                banner.append(
                    f"GradHP:!! {tamiyo.dead_layers}D/{tamiyo.exploding_layers}E",
                    style="red",
                )
            else:
                banner.append(
                    f"GradHP:OK {healthy}/{self._TOTAL_LAYERS}", style="green"
                )
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
        """Render 2x2 gauge grid: EV, Entropy, Clip, KL with trend indicators."""
        from esper.karn.sanctum.schema import detect_trend, trend_to_indicator

        tamiyo = self._snapshot.tamiyo
        batch = self._snapshot.current_batch

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Detect trends for each metric
        ev_trend = detect_trend(
            list(tamiyo.explained_variance_history),
            metric_name="expl_var",
            metric_type="accuracy",
        )
        ev_indicator, ev_indicator_style = trend_to_indicator(ev_trend)

        entropy_trend = detect_trend(
            list(tamiyo.entropy_history), metric_name="entropy", metric_type="accuracy"
        )
        entropy_indicator, entropy_indicator_style = trend_to_indicator(entropy_trend)

        clip_trend = detect_trend(
            list(tamiyo.clip_fraction_history),
            metric_name="clip_fraction",
            metric_type="loss",
        )
        clip_indicator, clip_indicator_style = trend_to_indicator(clip_trend)

        kl_trend = detect_trend(
            list(tamiyo.kl_divergence_history),
            metric_name="kl_divergence",
            metric_type="loss",
        )
        kl_indicator, kl_indicator_style = trend_to_indicator(kl_trend)

        # Row 1: Explained Variance | Entropy
        ev_gauge = self._render_gauge_v2(
            "Expl.Var",
            tamiyo.explained_variance,
            min_val=-1.0,
            max_val=1.0,
            status=self._get_ev_status(tamiyo.explained_variance),
            label_text=self._get_ev_label(tamiyo.explained_variance),
            trend_indicator=ev_indicator,
            trend_style=ev_indicator_style,
        )
        entropy_gauge = self._render_gauge_v2(
            "Entropy",
            tamiyo.entropy,
            min_val=0.0,
            max_val=2.0,
            status=self._get_entropy_status(tamiyo.entropy),
            label_text=self._get_entropy_label(tamiyo.entropy, batch),
            trend_indicator=entropy_indicator,
            trend_style=entropy_indicator_style,
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
            trend_indicator=clip_indicator,
            trend_style=clip_indicator_style,
        )
        kl_gauge = self._render_gauge_v2(
            "KL Div",
            tamiyo.kl_divergence,
            min_val=0.0,
            max_val=0.1,
            status=self._get_kl_status(tamiyo.kl_divergence),
            label_text=self._get_kl_label(tamiyo.kl_divergence, batch),
            trend_indicator=kl_indicator,
            trend_style=kl_indicator_style,
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
        trend_indicator: str = "-",
        trend_style: str = "dim",
    ) -> Text:
        """Render a gauge with status-colored bar and trend indicator."""
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
        bar_color = {"ok": "bright_cyan", "warning": "yellow", "critical": "red"}[
            status
        ]

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

        # Add trend indicator (per Task 2)
        gauge.append(" ", style="dim")
        gauge.append(trend_indicator, style=trend_style)

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
        width: int | None = None,
        style: str = "bright_cyan",
    ) -> Text:
        """Render sparkline using unicode block characters.

        Args:
            history: Historical values to visualize
            width: Maximum width in characters (defaults to SPARKLINE_WIDTH)
            style: Rich style for the blocks

        Returns:
            Text with sparkline or placeholder for empty/flat data.
        """
        if width is None:
            width = self.SPARKLINE_WIDTH

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
        """Render secondary metrics column with sparklines and trend indicators."""
        from esper.karn.sanctum.schema import detect_trend, trend_to_indicator

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Advantage stats
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        adv_style = self._status_style(adv_status)
        result.append(" Advantage   ", style="dim")
        result.append(
            f"{tamiyo.advantage_mean:+.2f} ± {tamiyo.advantage_std:.2f}",
            style=adv_style,
        )
        if adv_status != "ok":
            result.append(" [!]", style=adv_style)
        result.append("\n")

        # Ratio bounds
        ratio_status = self._get_ratio_status(tamiyo.ratio_min, tamiyo.ratio_max)
        ratio_style = self._status_style(ratio_status)
        result.append(" Ratio       ", style="dim")
        result.append(
            f"{tamiyo.ratio_min:.2f} < r < {tamiyo.ratio_max:.2f}", style=ratio_style
        )
        if ratio_status != "ok":
            result.append(" [!]", style=ratio_style)
        result.append("\n")

        # Policy loss with sparkline and trend indicator
        pl_sparkline = self._render_sparkline(tamiyo.policy_loss_history)
        pl_trend = detect_trend(
            list(tamiyo.policy_loss_history),
            metric_name="policy_loss",
            metric_type="loss",
        )
        pl_indicator, pl_indicator_style = trend_to_indicator(pl_trend)
        result.append(" Policy Loss ", style="dim")
        result.append(pl_sparkline)
        result.append(f" {tamiyo.policy_loss:.3f} ", style="bright_cyan")
        result.append(pl_indicator, style=pl_indicator_style)
        result.append("\n")

        # Value loss with sparkline and trend indicator
        vl_sparkline = self._render_sparkline(tamiyo.value_loss_history)
        vl_trend = detect_trend(
            list(tamiyo.value_loss_history),
            metric_name="value_loss",
            metric_type="loss",
        )
        vl_indicator, vl_indicator_style = trend_to_indicator(vl_trend)
        result.append(" Value Loss  ", style="dim")
        result.append(vl_sparkline)
        result.append(f" {tamiyo.value_loss:.3f} ", style="bright_cyan")
        result.append(vl_indicator, style=vl_indicator_style)
        result.append("\n")

        # Grad norm with sparkline and trend indicator
        gn_sparkline = self._render_sparkline(tamiyo.grad_norm_history)
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        gn_style = self._status_style(gn_status)
        gn_trend = detect_trend(
            list(tamiyo.grad_norm_history), metric_name="grad_norm", metric_type="loss"
        )
        gn_indicator, gn_indicator_style = trend_to_indicator(gn_trend)
        result.append(" Grad Norm   ", style="dim")
        result.append(gn_sparkline)
        result.append(f" {tamiyo.grad_norm:.2f} ", style=gn_style)
        result.append(gn_indicator, style=gn_indicator_style)
        result.append("\n")

        # Layer health
        result.append(" Layers      ", style="dim")
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(
                f"!! {tamiyo.dead_layers} dead, {tamiyo.exploding_layers} exploding",
                style="red",
            )
        else:
            healthy = self._TOTAL_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
            result.append(f"OK {healthy}/{self._TOTAL_LAYERS} healthy", style="green")

        return result

    def _get_ratio_status(self, ratio_min: float, ratio_max: float) -> str:
        """Get status for PPO ratio bounds."""
        if (
            ratio_max > TUIThresholds.RATIO_MAX_CRITICAL
            or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL
        ):
            return "critical"
        if (
            ratio_max > TUIThresholds.RATIO_MAX_WARNING
            or ratio_min < TUIThresholds.RATIO_MIN_WARNING
        ):
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

    # Head segment width: 10 chars per head for alignment
    # Format: "abbr[███] " = 4-char abbrev + "[" + 3-char bar + "] " = 10 chars
    HEAD_SEGMENT_WIDTH = 10

    def _render_head_heatmap(self) -> Text:
        """Render per-head entropy heatmap with 8 action heads.

        Shows visual distinction for heads without telemetry data.
        Per DRL review: max entropy values computed from actual action space dimensions.

        Layout uses fixed-width segments (HEAD_SEGMENT_WIDTH=9 chars) for alignment:
        - Row 1: "abbr[███] " - 4-char label + bar
        - Row 2: "  0.00!  " - centered value with indicator

        Per UX review: expanded abbreviations for readability.
        Per Code review: explicit width accounting prevents drift.
        """
        tamiyo = self._snapshot.tamiyo

        # Head config: (abbrev, field_name, head_key)
        # Expanded abbreviations per UX review for readability
        heads = [
            ("slot", "head_slot_entropy", "slot"),
            ("bpnt", "head_blueprint_entropy", "blueprint"),
            ("styl", "head_style_entropy", "style"),
            ("temp", "head_tempo_entropy", "tempo"),
            ("atgt", "head_alpha_target_entropy", "alpha_target"),
            ("aspd", "head_alpha_speed_entropy", "alpha_speed"),
            ("acrv", "head_alpha_curve_entropy", "alpha_curve"),
            (
                "op",
                "head_op_entropy",
                "op",
            ),  # No need to pad this unless it stops being the last one.
        ]

        result = Text()
        result.append(" Heads: ", style="dim")

        # First line: bars (9-char segments)
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            is_tracked = head_key in self.TRACKED_HEADS

            # Check for missing data (value=0.0 for untracked heads)
            if value == 0.0 and not is_tracked:
                # Visual distinction for awaiting telemetry (per risk assessor)
                # 9-char segment: "abbr[---] "
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

            # 9-char segment: "abbr[███] " = 4 + 1 + 3 + 1 + space = 9
            result.append(f"{abbrev}[")
            result.append("█" * filled, style=color)
            result.append("░" * empty, style="dim")
            result.append("] ")

        result.append("\n        ")

        # Second line: values (10-char segments to match bars)
        for abbrev, field, head_key in heads:
            value = getattr(tamiyo, field, 0.0)
            is_tracked = head_key in self.TRACKED_HEADS

            if value == 0.0 and not is_tracked:
                # 10-char segment: "   n/a    " (centered)
                result.append("   n/a    ", style="dim italic")
                continue

            max_ent = self.HEAD_MAX_ENTROPIES[head_key]
            fill = value / max_ent if max_ent > 0 else 0

            # 10-char segment with indicators: critical (!), warning (*), normal
            # Format: "  X.XX!   " = 2 spaces + 4.2f (4 chars) + indicator + 3 spaces = 10
            if fill < 0.25:
                # Critical: entropy collapsed (< 25% of max)
                result.append(f"  {value:4.2f}!   ", style="red")
            elif fill < 0.5:
                # Warning: entropy converging (< 50% of max)
                result.append(f"  {value:4.2f}*   ", style="yellow")
            else:
                # Normal: healthy exploration (>= 50% of max)
                result.append(f"  {value:4.2f}    ", style="dim")

        return result

    def _render_decisions_column(self) -> Text:
        """Render vertical stack of compact decision cards for right column.

        CRITICAL INVARIANT: ONE card swap per CARD_SWAP_INTERVAL (30s), maximum.
        This provides visual stability - users need time to read each card.

        The display buffer (_displayed_decisions) is separate from the incoming
        data firehose. It updates slowly and predictably:
        - Growing: Add ONE card per interval until max_cards reached
        - Replacing: Swap ONE oldest card for ONE newest card per interval

        NEVER replace 2+ cards simultaneously. That is a bug.

        Returns:
            Rich Text with stacked compact decision cards.
        """
        import time

        tamiyo = self._snapshot.tamiyo
        incoming = tamiyo.recent_decisions
        max_cards = self._get_max_decision_cards()
        now = time.time()

        # === THROTTLED DISPLAY BUFFER UPDATE ===
        # Rule: ONE card change every CARD_SWAP_INTERVAL, maximum.

        if not incoming:
            # No data - clear display and show waiting message
            self._displayed_decisions = []
            result = Text()
            result.append("DECISIONS\n", style="dim bold")
            result.append("No decisions yet\n", style="dim italic")
            result.append("Waiting for\n", style="dim")
            result.append("agent actions...", style="dim")
            return result

        # Build set of displayed IDs for deduplication
        # NOTE: We do NOT remove displayed decisions when they expire from incoming.
        # The display buffer is independent - cards stay until replaced by the
        # throttled replacement logic (ONE card per 30s interval).

        # Time check: has enough time passed since last card change?
        time_since_swap = now - self._last_card_swap_time
        can_change = time_since_swap >= self.CARD_SWAP_INTERVAL

        # Find the newest incoming decision not already in display
        # EXPLICITLY sort by timestamp to guarantee we get the freshest decision,
        # regardless of incoming list order
        displayed_ids = {d.decision_id for d in self._displayed_decisions}
        candidates = [d for d in incoming if d.decision_id not in displayed_ids]
        new_decision = max(candidates, key=lambda d: d.timestamp) if candidates else None

        if new_decision and can_change:
            # We have a new decision and enough time has passed
            if len(self._displayed_decisions) < max_cards:
                # GROWING: Add ONE card to display
                self._displayed_decisions.insert(0, new_decision)
            else:
                # REPLACING: Remove oldest (last), add newest (first)
                self._displayed_decisions.pop()  # Remove ONE oldest
                self._displayed_decisions.insert(0, new_decision)  # Add ONE newest

            self._last_card_swap_time = now

        # Clamp to max_cards (in case max_cards shrunk due to resize)
        self._displayed_decisions = self._displayed_decisions[:max_cards]

        # Store decision IDs for click handling
        self._decision_ids = [d.decision_id for d in self._displayed_decisions]

        # === RENDER THE STABLE DISPLAY BUFFER ===
        result = Text()
        result.append("DECISIONS\n", style="dim bold")

        if not self._displayed_decisions:
            # Buffer empty but incoming has data - waiting for first interval
            result.append("Loading...\n", style="dim italic")
            return result

        total_cards = len(self._displayed_decisions)
        for i, decision in enumerate(self._displayed_decisions):
            card = self._render_enriched_decision(
                decision, index=i, total_cards=total_cards
            )
            result.append(card)
            if i < total_cards - 1:  # Add spacing between cards
                result.append("\n")

        return result

    def _render_primary_metrics(self) -> Text:
        """Render primary metrics row (Episode Return + Entropy).

        Per UX review: These go at row 3, prime visual real estate.
        Per DRL review: Entropy sparkline critical for collapse detection.
        """
        from esper.karn.sanctum.schema import detect_trend, trend_to_indicator

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Episode Return (PRIMARY RL METRIC)
        if tamiyo.episode_return_history:
            sparkline = self._render_sparkline(
                tamiyo.episode_return_history, width=self.SPARKLINE_WIDTH
            )
            trend = detect_trend(
                list(tamiyo.episode_return_history),
                metric_name="episode_return",
                metric_type="accuracy",  # Higher is better
            )
            indicator, style = trend_to_indicator(trend)

            result.append("Ep.Return  ", style="bold cyan")
            result.append(sparkline)
            result.append(f"  {tamiyo.current_episode_return:>6.1f} ", style="white")
            result.append(indicator, style=style)
            result.append(
                f"      LR:{tamiyo.learning_rate:.0e}"
                if tamiyo.learning_rate
                else "      LR:n/a",
                style="dim",
            )
            result.append(f"  EntCoef:{tamiyo.entropy_coef:.2f}", style="dim")
            result.append("\n")

        # Entropy (COLLAPSE DETECTION)
        if tamiyo.entropy_history:
            sparkline = self._render_sparkline(
                tamiyo.entropy_history, width=self.SPARKLINE_WIDTH
            )
            trend = detect_trend(
                list(tamiyo.entropy_history),
                metric_name="entropy",
                metric_type="accuracy",  # Stable/high is good, low is collapse
            )
            indicator, style = trend_to_indicator(trend)

            result.append("Entropy    ", style="bold")
            result.append(sparkline)
            result.append(f"  {tamiyo.entropy:>6.2f} ", style="white")
            result.append(indicator, style=style)

        return result

    def _render_vitals_column(self) -> Table:
        """Render left 2/3 column with all learning vitals.

        Contains (top to bottom):
        - Primary metrics (Episode Return + Entropy) at TOP
        - Separator
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

        # Row 0: PRIMARY METRICS AT TOP (per UX review)
        primary = self._render_primary_metrics()
        content.add_row(primary)

        # Row 1: Separator
        content.add_row(self._render_separator())

        # Row 2: Diagnostic matrix (gauges left, metrics right)
        diagnostic_matrix = self._render_diagnostic_matrix()
        content.add_row(diagnostic_matrix)

        # Row 3: Separator
        content.add_row(self._render_separator())

        # Row 4: Head heatmap
        head_heatmap = self._render_head_heatmap()
        content.add_row(head_heatmap)

        # Row 5: Separator
        content.add_row(self._render_separator())

        # Row 6: Action distribution bar
        action_bar = self._render_action_distribution_bar()
        content.add_row(action_bar)

        # Row 7: Action sequence with pattern detection
        action_sequence = self._render_action_sequence()
        content.add_row(action_sequence)

        # Row 8: Episode return history
        return_history = self._render_return_history()
        content.add_row(return_history)

        # Row 9: Separator before slot summary
        content.add_row(self._render_separator())

        # Row 10: Aggregate slot summary (explains Tamiyo's action constraints)
        slot_summary = self._render_slot_summary()
        content.add_row(slot_summary)

        return content
