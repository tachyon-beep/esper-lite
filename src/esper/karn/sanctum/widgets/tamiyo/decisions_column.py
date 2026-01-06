"""DecisionsColumn - Vertical stack of decision cards.

Manages the throttled display of decision cards to provide visual stability.
Cards pop in ONE at a time every 5 seconds (both growing and steady state).
First card appears immediately, then one every 5s until MAX_CARDS reached.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from .action_display import ACTION_COLORS
from esper.leyline import ALPHA_CURVE_GLYPHS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


class DecisionDetailRequested(Message):
    """Request to open a drill-down view for a decision."""

    bubble = True

    def __init__(self, *, group_id: str, decision: "DecisionSnapshot") -> None:
        super().__init__()
        self.group_id = group_id
        self.decision = decision


class DecisionCard(Static):
    """Individual decision card widget with CSS-driven styling."""

    CARD_WIDTH: ClassVar[int] = 46

    def __init__(
        self,
        decision: "DecisionSnapshot",
        index: int,
        total_cards: int,
        group_id: str,
        display_timestamp: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decision = decision
        self.index = index
        self.total_cards = total_cards
        self._group_id = group_id
        self._display_timestamp = display_timestamp or datetime.now(timezone.utc)
        self._update_classes()

    def _update_classes(self) -> None:
        """Update CSS classes based on card position."""
        self.remove_class("newest", "oldest")

        if self.index == 0:
            self.add_class("newest")
        elif self.index == self.total_cards - 1:
            self.add_class("oldest")

    def on_click(self, event: events.Click) -> None:
        """Open drill-down view for a decision."""
        self.post_message(DecisionDetailRequested(group_id=self._group_id, decision=self.decision))

    def render(self) -> Text:
        """Render the decision card content."""
        decision = self.decision
        result = Text()

        # Calculate age (from when card was added to display, not decision timestamp)
        now = datetime.now(timezone.utc)
        age = (now - self._display_timestamp).total_seconds()
        age_str = f"{age:.0f}s" if age < 60 else f"{int(age // 60)}:{int(age % 60):02d}"

        action_style = ACTION_COLORS.get(decision.chosen_action, "white")

        # Line 1: Action + Age
        result.append(f"#{self.index + 1} ", style="bright_white")
        result.append(decision.chosen_action, style=action_style)
        padding = self.CARD_WIDTH - len(f"#{self.index + 1} {decision.chosen_action}") - len(age_str) - 2
        result.append(" " * max(0, padding))
        result.append(age_str, style="dim")
        result.append("\n")

        # Line 2: Training context (epoch, env, round)
        result.append(f"epoch:{decision.epoch}", style="dim")
        result.append("  ", style="dim")
        result.append(f"env:{decision.env_id}", style="cyan")
        result.append("  ", style="dim")
        result.append(f"round:{decision.batch}", style="dim")
        result.append("\n")

        # Line 3: Slot + Confidence
        slot_num = decision.chosen_slot[-1] if decision.chosen_slot else "-"
        result.append(f"slot:{slot_num}", style="cyan")
        result.append("  ", style="dim")
        result.append(f"confidence:{decision.confidence:.0%}", style="dim")
        result.append("\n")

        # Line 3: Blueprint info (GERMINATE) or context note
        if decision.chosen_action == "GERMINATE" and decision.chosen_blueprint:
            bp_name = decision.chosen_blueprint[:10]
            bp_conf = f"({decision.blueprint_confidence:.0%})" if decision.blueprint_confidence else ""

            tempo_map = {"FAST": "▸▸▸", "STANDARD": "▸▸", "SLOW": "▸"}
            tempo_display = tempo_map.get(decision.chosen_tempo or "", "-")

            style_map = {
                "LINEAR_ADD": "lin+add",
                "LINEAR_MULTIPLY": "lin×mul",
                "SIGMOID_ADD": "sig+add",
                "GATED_GATE": "gate⊙",
            }
            style_display = style_map.get(decision.chosen_style or "", "-")

            curve_glyph = ALPHA_CURVE_GLYPHS.get(decision.chosen_curve or "", "-")

            result.append("blueprint:", style="dim")
            result.append(bp_name, style="cyan")
            result.append(bp_conf, style="dim")
            result.append("  ", style="dim")
            result.append(tempo_display, style="magenta")
            result.append("  ", style="dim")
            result.append(style_display, style="yellow")
            result.append("  ", style="dim")
            result.append(curve_glyph, style="green")
        else:
            context = self._action_context_note(decision)
            if context:
                result.append(context, style="dim italic")
        result.append("\n")

        # Separator
        result.append("─" * self.CARD_WIDTH, style="dim")
        result.append("\n")

        # Line 4: WHY - inferred reasoning
        why_text = self._infer_why(decision)
        result.append("WHY: ", style="bold yellow")
        result.append(why_text, style="italic")
        result.append("\n")

        # Line 5: Host + Entropy + Badge
        entropy_label, entropy_style = self._entropy_label(decision)
        outcome_badge, badge_style = self._outcome_badge(
            decision.expected_value, decision.actual_reward
        )

        result.append(f"host:{decision.host_accuracy:.0f}%", style="cyan")
        result.append("  ", style="dim")
        result.append(f"entropy:{decision.decision_entropy:.2f}", style="dim")
        result.append(" ", style="dim")
        result.append(entropy_label, style=entropy_style)

        # Right-align badge
        content_so_far = f"host:{decision.host_accuracy:.0f}%  entropy:{decision.decision_entropy:.2f} {entropy_label}"
        badge_padding = self.CARD_WIDTH - len(content_so_far) - len(outcome_badge) - 4
        result.append(" " * max(1, badge_padding))
        result.append(outcome_badge, style=badge_style)
        result.append("\n")

        # Line 5: Expect + Reward + TD
        if decision.actual_reward is not None:
            result.append(f"expect:{decision.expected_value:+.2f}", style="cyan")
            result.append("  ", style="dim")
            result.append(f"reward:{decision.actual_reward:+.2f}", style="magenta")
            result.append("  ", style="dim")
            if decision.td_advantage is not None:
                result.append(f"TD:{decision.td_advantage:+.2f}", style="bright_cyan")
            else:
                result.append("TD:—", style="dim italic")
        else:
            result.append(f"expect:{decision.expected_value:+.2f}", style="cyan")
            result.append("  ", style="dim")
            result.append("reward:—", style="dim italic")
            result.append("  ", style="dim")
            result.append("TD:—", style="dim italic")
        result.append("\n")

        # Line 6: Alternatives
        if decision.alternatives:
            result.append("also: ", style="dim")
            for i, (alt_action, prob) in enumerate(decision.alternatives[:2]):
                if i > 0:
                    result.append("  ", style="dim")
                alt_style = ACTION_COLORS.get(alt_action, "dim")
                result.append(alt_action, style=alt_style)
                result.append(f" {prob:.0%}", style="dim")
        else:
            result.append("also: -", style="dim")

        return result

    def _action_context_note(self, decision: "DecisionSnapshot") -> str:
        """Return contextual note for non-GERMINATE actions."""
        action = decision.chosen_action
        slot = decision.chosen_slot or "?"

        if action == "GERMINATE":
            return ""
        elif action == "WAIT":
            return "(waiting for training progress)"
        elif action == "PRUNE":
            return f"(removing underperformer from {slot})"
        elif action == "FOSSILIZE":
            return f"(fusing trained module in {slot})"
        elif action == "SET_ALPHA_TARGET":
            return "(adjusting blend parameters)"
        return ""

    def _infer_why(self, decision: "DecisionSnapshot") -> str:
        """Infer the reasoning behind a decision from context.

        Returns a short explanation of WHY Tamiyo made this choice.
        """
        action = decision.chosen_action
        slot_states = decision.slot_states
        confidence = decision.confidence
        entropy = decision.decision_entropy
        host_acc = decision.host_accuracy

        # Count slot states
        dormant_count = sum(1 for s in slot_states.values() if "Dormant" in s or "Empty" in s)
        training_count = sum(1 for s in slot_states.values() if "Training" in s)

        if action == "GERMINATE":
            reasons = []
            if dormant_count > 0:
                reasons.append("dormant slot available")
            if host_acc < 30:
                reasons.append("host accuracy low")
            elif host_acc < 50:
                reasons.append("host needs help")
            if confidence > 0.8:
                reasons.append("high confidence")
            return " + ".join(reasons[:2]) if reasons else "opportunity to grow"

        elif action == "WAIT":
            if training_count > 0:
                return f"{training_count} slot{'s' if training_count > 1 else ''} still training"
            if entropy > 1.0:
                return "high uncertainty, gathering data"
            if dormant_count == 0:
                return "all slots occupied"
            if confidence > 0.7:
                return "deliberate pause"
            return "monitoring progress"

        elif action == "PRUNE":
            if confidence > 0.8:
                return "clear underperformer"
            return "removing low contributor"

        elif action == "FOSSILIZE":
            slot = decision.chosen_slot or "?"
            slot_state = slot_states.get(slot, "")
            if "Blending" in slot_state:
                return "module ready to fuse"
            if "Holding" in slot_state:
                return "stable, ready to commit"
            return "matured module"

        elif action == "SET_ALPHA_TARGET":
            if confidence > 0.8:
                return "alpha adjustment needed"
            return "tuning blend ratio"

        return "policy decision"

    def _entropy_label(self, decision: "DecisionSnapshot") -> tuple[str, str]:
        """Return (label, style) for entropy value.

        Context-aware: distinguishes between legitimate low-entropy decisions
        (deterministic valid actions) and concerning policy collapse.
        """
        entropy = decision.decision_entropy
        action = decision.chosen_action
        confidence = decision.confidence
        slot_states = decision.slot_states

        # Count slot states for context
        dormant_count = sum(1 for s in slot_states.values() if "Dormant" in s or "Empty" in s)
        training_count = sum(1 for s in slot_states.values() if "Training" in s)

        # Determine if low entropy is legitimate
        legitimate_low_entropy = False
        if entropy < 0.3:
            # WAIT is legitimate when no valid actions exist
            if action == "WAIT" and (dormant_count == 0 or training_count > 0):
                legitimate_low_entropy = True
            # High-confidence GERMINATE with available slot is legitimate
            elif action == "GERMINATE" and confidence > 0.8 and dormant_count > 0:
                legitimate_low_entropy = True
            # High-confidence FOSSILIZE/PRUNE are legitimate
            elif action in ("FOSSILIZE", "PRUNE") and confidence > 0.8:
                legitimate_low_entropy = True

        # Apply labels
        if entropy < 0.3:
            if legitimate_low_entropy:
                return "✓", "green"  # Deterministic valid decision
            return "[collapsing]", "red"  # Actual policy collapse
        elif entropy < 0.7:
            return "[confident]", "yellow"
        elif entropy < 1.2:
            return "[balanced]", "green"
        return "[exploring]", "cyan"

    def _outcome_badge(self, expect: float, reward: float | None) -> tuple[str, str]:
        """Return (badge, style) for prediction accuracy."""
        if reward is None:
            return "[...]", "dim"
        diff = abs(reward - expect)
        if diff < 0.1:
            return "[HIT]", "bright_green"
        elif diff < 0.3:
            return "[~OK]", "yellow"
        return "[MISS]", "red"


class DecisionsColumn(Container):
    """Vertical stack of decision cards with throttled updates."""

    CARD_SWAP_INTERVAL: ClassVar[float] = 5.0
    MAX_CARDS: ClassVar[int] = 3

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._displayed_decisions: list["DecisionSnapshot"] = []
        self._display_timestamps: dict[str, datetime] = {}  # decision_id -> when added to display
        self._last_card_swap_time: float = 0.0
        self._last_env_id: int | None = None
        self._rendering: bool = False  # Guard against concurrent renders
        self._render_generation: int = 0  # Unique ID suffix for each render
        self._group_id: str = "default"
        self.border_title = "DECISIONS"

    def compose(self) -> ComposeResult:
        """Compose the decisions column."""
        yield VerticalScroll(id="cards-container")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data using throttled card replacement.

        "Firehose" model: when it's time to add/swap, grab the MOST RECENT
        decision available - no backlog, no queueing. Age starts at 0 when
        a decision goes on display.
        """
        self._snapshot = snapshot
        self._group_id = snapshot.tamiyo.group_id or "default"
        now = time.time()

        incoming = snapshot.tamiyo.recent_decisions

        # If no incoming decisions and we have none displayed, show placeholder
        if not incoming and not self._displayed_decisions:
            self._render_cards()
            return

        # If no incoming decisions but we have displayed ones, just refresh ages
        if not incoming:
            self._refresh_cards()
            return

        # Firehose: always grab the MOST RECENT decision (no backlog)
        # Exclude ALL currently displayed decisions to prevent duplicates
        displayed_ids = {d.decision_id for d in self._displayed_decisions if d.decision_id}
        candidates = [d for d in incoming if d.decision_id and d.decision_id not in displayed_ids]
        newest = None
        if candidates:
            if self._last_env_id is None:
                newest = max(candidates, key=lambda d: d.timestamp)
            else:
                rotated = [d for d in candidates if d.env_id != self._last_env_id]
                newest = max(rotated, key=lambda d: d.timestamp) if rotated else max(
                    candidates, key=lambda d: d.timestamp
                )

        if not newest:
            # No new decision available, refresh existing cards
            self._refresh_cards()
            return

        # Throttled update logic
        is_growing = len(self._displayed_decisions) < self.MAX_CARDS
        is_empty = len(self._displayed_decisions) == 0
        time_since_swap = now - self._last_card_swap_time
        can_add = is_empty or time_since_swap >= self.CARD_SWAP_INTERVAL

        if not can_add:
            # Throttled - just refresh ages
            self._refresh_cards()
            return

        # Time to add/swap - use the newest decision, age starts at 0
        display_now = datetime.now(timezone.utc)

        if is_growing:
            # Growing phase: add to front
            self._displayed_decisions.insert(0, newest)
        else:
            # Steady state: swap oldest for newest
            old_decision = self._displayed_decisions.pop()
            if old_decision.decision_id in self._display_timestamps:
                del self._display_timestamps[old_decision.decision_id]
            self._displayed_decisions.insert(0, newest)

        # Age starts at 0 - record when this decision went on display
        if newest.decision_id:
            self._display_timestamps[newest.decision_id] = display_now

        self._last_env_id = newest.env_id
        self._last_card_swap_time = now
        self._render_cards()

    def _render_cards(self) -> None:
        """Re-render all decision cards."""
        # Guard against concurrent renders
        if self._rendering:
            return
        self._rendering = True
        self._render_generation += 1
        gen = self._render_generation

        try:
            container = self.query_one("#cards-container", VerticalScroll)

            # Remove ALL existing cards (use list() to avoid mutation during iteration)
            for card in list(container.query(DecisionCard)):
                card.remove()

            # Remove ALL placeholders and other statics
            for widget in list(container.query(Static)):
                widget.remove()

            if not self._displayed_decisions:
                # Show placeholder with unique ID per generation
                container.mount(
                    Static(
                        "Waiting for decisions...",
                        id=f"placeholder-{gen}",
                        classes="dim italic",
                    )
                )
                return

            # Mount new cards with unique IDs (generation suffix prevents collisions)
            total = len(self._displayed_decisions)
            for i, decision in enumerate(self._displayed_decisions):
                # Get display timestamp (when card was added to display)
                display_ts = self._display_timestamps.get(decision.decision_id) if decision.decision_id else None
                card = DecisionCard(
                    decision=decision,
                    index=i,
                    total_cards=total,
                    group_id=self._group_id,
                    display_timestamp=display_ts,
                    id=f"card-{decision.decision_id}-{gen}",
                )
                container.mount(card)
        finally:
            self._rendering = False

    def _refresh_cards(self) -> None:
        """Refresh card content without structural changes (updates ages)."""
        container = self.query_one("#cards-container", VerticalScroll)
        for card in container.query(DecisionCard):
            card.refresh()
