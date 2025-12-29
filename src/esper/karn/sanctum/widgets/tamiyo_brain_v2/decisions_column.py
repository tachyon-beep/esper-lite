"""DecisionsColumn - Vertical stack of decision cards.

Manages the throttled display of decision cards to provide visual stability.
Cards update ONE at a time every 30 seconds maximum.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Static

from esper.leyline import ALPHA_CURVE_GLYPHS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


# Action colors for decision cards
ACTION_COLORS: dict[str, str] = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
    "ADVANCE": "cyan",
}


class DecisionCard(Static):
    """Individual decision card widget with CSS-driven styling."""

    CARD_WIDTH: ClassVar[int] = 42

    # Enable keyboard focus for card navigation
    can_focus = True

    class Pinned(Message):
        """Posted when user clicks to toggle pin status."""

        def __init__(self, decision_id: str) -> None:
            super().__init__()
            self.decision_id = decision_id

    def __init__(
        self,
        decision: "DecisionSnapshot",
        index: int,
        total_cards: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.decision = decision
        self.index = index
        self.total_cards = total_cards
        self._update_classes()

    def _update_classes(self) -> None:
        """Update CSS classes based on card position."""
        self.remove_class("newest", "oldest", "pinned")

        if self.decision.pinned:
            self.add_class("pinned")
        elif self.index == 0:
            self.add_class("newest")
        elif self.index == self.total_cards - 1:
            self.add_class("oldest")

    def on_click(self) -> None:
        """Handle click to toggle pin."""
        self.post_message(self.Pinned(self.decision.decision_id))

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard input on focused card."""
        if event.key == "p":
            self.post_message(self.Pinned(self.decision.decision_id))
            event.stop()

    def render(self) -> Text:
        """Render the decision card content."""
        decision = self.decision
        result = Text()

        # Calculate age
        now = datetime.now(timezone.utc)
        age = (now - decision.timestamp).total_seconds()
        age_str = f"{age:.0f}s" if age < 60 else f"{int(age // 60)}:{int(age % 60):02d}"

        action_style = ACTION_COLORS.get(decision.chosen_action, "white")

        # Line 1: Action + Age
        result.append(f"#{self.index + 1} ", style="bright_white")
        result.append(decision.chosen_action, style=action_style)
        padding = self.CARD_WIDTH - len(f"#{self.index + 1} {decision.chosen_action}") - len(age_str) - 2
        result.append(" " * max(0, padding))
        result.append(age_str, style="dim")
        result.append("\n")

        # Line 2: Slot + Confidence
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
        entropy_label, entropy_style = self._entropy_label(decision.decision_entropy)
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
                result.append("TD:...", style="dim italic")
        else:
            result.append(f"expect:{decision.expected_value:+.2f}", style="cyan")
            result.append("  ", style="dim")
            result.append("reward:...", style="dim italic")
            result.append("  ", style="dim")
            result.append("TD:...", style="dim italic")
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
        blending_count = sum(1 for s in slot_states.values() if "Blending" in s or "Holding" in s)

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

    def _entropy_label(self, entropy: float) -> tuple[str, str]:
        """Return (label, style) for entropy value."""
        if entropy < 0.3:
            return "[collapsed]", "red"
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

    CARD_SWAP_INTERVAL: ClassVar[float] = 30.0
    MAX_CARDS: ClassVar[int] = 3

    BINDINGS = [
        Binding("j", "focus_next", "Next card", show=False),
        Binding("k", "focus_prev", "Previous card", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._displayed_decisions: list["DecisionSnapshot"] = []
        self._last_card_swap_time: float = 0.0
        self._rendering: bool = False  # Guard against concurrent renders
        self._render_generation: int = 0  # Unique ID suffix for each render

    def compose(self) -> ComposeResult:
        """Compose the decisions column."""
        yield Static("DECISIONS [j/k:nav p:pin]", id="decisions-header", classes="decisions-header")
        yield Vertical(id="cards-container")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data using throttled card replacement."""
        self._snapshot = snapshot
        now = time.time()

        incoming = snapshot.tamiyo.recent_decisions

        # If no incoming decisions and we have none displayed, ensure placeholder shows
        if not incoming and not self._displayed_decisions:
            self._render_cards()
            return

        # If no incoming decisions but we have displayed ones, keep them stable
        if not incoming:
            return

        # Throttled update logic:
        # - Growing phase (< MAX_CARDS): add cards immediately, no throttle
        # - Steady state (= MAX_CARDS): swap one card every CARD_SWAP_INTERVAL
        displayed_ids = {d.decision_id for d in self._displayed_decisions}
        candidates = [d for d in incoming if d.decision_id not in displayed_ids]
        new_decision = max(candidates, key=lambda d: d.timestamp) if candidates else None

        if not new_decision:
            # No new decisions, but refresh existing cards to update ages
            self._refresh_cards()
            return

        is_growing = len(self._displayed_decisions) < self.MAX_CARDS
        time_since_swap = now - self._last_card_swap_time
        can_swap = time_since_swap >= self.CARD_SWAP_INTERVAL

        if is_growing:
            # Growing phase: add cards immediately
            self._displayed_decisions.insert(0, new_decision)
            self._render_cards()
        elif can_swap:
            # Steady state: swap oldest for newest, respecting throttle
            self._displayed_decisions.pop()
            self._displayed_decisions.insert(0, new_decision)
            self._last_card_swap_time = now
            self._render_cards()
        else:
            # New decision waiting, but throttled - refresh to update ages
            self._refresh_cards()

    def _render_cards(self) -> None:
        """Re-render all decision cards."""
        # Guard against concurrent renders
        if self._rendering:
            return
        self._rendering = True
        self._render_generation += 1
        gen = self._render_generation

        try:
            container = self.query_one("#cards-container", Vertical)

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
                card = DecisionCard(
                    decision=decision,
                    index=i,
                    total_cards=total,
                    id=f"card-{decision.decision_id}-{gen}",
                )
                container.mount(card)
        finally:
            self._rendering = False

    def _refresh_cards(self) -> None:
        """Refresh card content without structural changes (updates ages)."""
        container = self.query_one("#cards-container", Vertical)
        for card in container.query(DecisionCard):
            card.refresh()

    def on_decision_card_pinned(self, event: DecisionCard.Pinned) -> None:
        """Handle pin toggle from decision card."""
        # Toggle pin state in data
        for decision in self._displayed_decisions:
            if decision.decision_id == event.decision_id:
                decision.pinned = not decision.pinned
                break

        # Update card styling in-place (avoid remove/mount cycle)
        container = self.query_one("#cards-container", Vertical)
        for card in container.query(DecisionCard):
            if card.decision.decision_id == event.decision_id:
                card._update_classes()
                card.refresh()
                break

    def action_focus_next(self) -> None:
        """Move focus to next decision card."""
        cards = list(self.query(DecisionCard))
        if not cards:
            return
        focused = self.app.focused
        if focused in cards:
            idx = cards.index(focused)
            next_idx = (idx + 1) % len(cards)
            cards[next_idx].focus()
        else:
            cards[0].focus()

    def action_focus_prev(self) -> None:
        """Move focus to previous decision card."""
        cards = list(self.query(DecisionCard))
        if not cards:
            return
        focused = self.app.focused
        if focused in cards:
            idx = cards.index(focused)
            prev_idx = (idx - 1) % len(cards)
            cards[prev_idx].focus()
        else:
            cards[-1].focus()
