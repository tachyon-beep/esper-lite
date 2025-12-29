"""HeadsPanel - Attention head entropy and gradient heatmaps.

Uses vertical layout to ensure values align directly under their labels:

    ┌─ ATTENTION HEADS ──────────────────────────────────────────┐
    │       slot   bpnt~  styl~  temp~  atgt~  aspd~  acrv~  op  │
    │ Entr  1.00   1.00   0.21   1.00   0.57   1.00   1.00  0.89 │
    │       ████   ████   ▓░░░   ████   ▓▓░░   ████   ████  ▓▓▓░ │
    │ Grad  0.13   0.21   0.27   0.19   0.10   0.21   0.12  0.13 │
    │       ▓░░░   ▓░░░   ▓░░░   ▓░░░   ▓░░░   ▓░░░   ▓░░░  ▓░░░ │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


# Head configuration
HEAD_CONFIG: list[tuple[str, str, str]] = [
    ("slot", "head_slot_entropy", "head_slot_grad_norm"),
    ("bpnt", "head_blueprint_entropy", "head_blueprint_grad_norm"),
    ("styl", "head_style_entropy", "head_style_grad_norm"),
    ("temp", "head_tempo_entropy", "head_tempo_grad_norm"),
    ("atgt", "head_alpha_target_entropy", "head_alpha_target_grad_norm"),
    ("aspd", "head_alpha_speed_entropy", "head_alpha_speed_grad_norm"),
    ("acrv", "head_alpha_curve_entropy", "head_alpha_curve_grad_norm"),
    ("op", "head_op_entropy", "head_op_grad_norm"),
]

# Heads that are conditional (only relevant for certain ops)
CONDITIONAL_HEADS = {"styl", "bpnt", "temp", "atgt", "aspd", "acrv"}

# Max entropy per head (ln(N) where N = action space size)
HEAD_MAX_ENTROPIES: dict[str, float] = {
    "slot": 1.099,      # ln(3)
    "bpnt": 2.565,      # ln(13)
    "styl": 1.386,      # ln(4)
    "temp": 1.099,      # ln(3)
    "atgt": 1.099,      # ln(3)
    "aspd": 1.386,      # ln(4)
    "acrv": 1.609,      # ln(5) - LINEAR, COSINE, SIGMOID_{GENTLE,STD,SHARP}
    "op": 1.792,        # ln(6)
}


class HeadsPanel(Static):
    """Attention head entropy and gradient display panel.

    Extends Static directly (like DecisionCard) to eliminate Container
    layout overhead that causes whitespace issues.
    """

    CELL_WIDTH: ClassVar[int] = 7  # Width per head column
    BAR_WIDTH: ClassVar[int] = 4   # Width of mini-bar

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "ATTENTION HEADS"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()  # Trigger render()

    def render(self) -> Text:
        """Render the heads grid."""
        return self._render_heads()

    def _render_heads(self) -> Text:
        """Render the heads grid with vertical alignment."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Row 0: Header labels
        result.append("      ", style="dim")  # Indent for row label
        for abbrev, _, _ in HEAD_CONFIG:
            # Add ~ suffix for conditional heads
            label = f"{abbrev}~" if abbrev in CONDITIONAL_HEADS else abbrev
            result.append(f"{label:^{self.CELL_WIDTH}}", style="dim bold")
        result.append("\n")

        # Row 1: Entropy values
        # Note: getattr without default - AttributeError if HEAD_CONFIG has typo
        result.append("Entr  ", style="dim")
        for abbrev, ent_field, _ in HEAD_CONFIG:
            entropy: float = getattr(tamiyo, ent_field)
            color = self._entropy_color(abbrev, entropy)
            result.append(f"{entropy:^{self.CELL_WIDTH}.2f}", style=color)
        result.append("\n")

        # Row 2: Entropy bars
        result.append("      ", style="dim")  # Indent
        for abbrev, ent_field, _ in HEAD_CONFIG:
            entropy = getattr(tamiyo, ent_field)
            bar = self._render_entropy_bar(abbrev, entropy)
            # Center the bar in the cell
            padding = (self.CELL_WIDTH - self.BAR_WIDTH) // 2
            result.append(" " * padding)
            result.append(bar)
            result.append(" " * (self.CELL_WIDTH - self.BAR_WIDTH - padding))
        result.append("\n")

        # Row 3: Gradient values
        result.append("Grad  ", style="dim")
        for abbrev, _, grad_field in HEAD_CONFIG:
            grad: float = getattr(tamiyo, grad_field)
            color = self._gradient_color(grad)
            result.append(f"{grad:^{self.CELL_WIDTH}.2f}", style=color)
        result.append("\n")

        # Row 4: Gradient bars
        result.append("      ", style="dim")  # Indent
        for abbrev, _, grad_field in HEAD_CONFIG:
            grad = getattr(tamiyo, grad_field)
            bar = self._render_gradient_bar(grad)
            padding = (self.CELL_WIDTH - self.BAR_WIDTH) // 2
            result.append(" " * padding)
            result.append(bar)
            result.append(" " * (self.CELL_WIDTH - self.BAR_WIDTH - padding))
        result.append("\n")

        # Row 5: Head state indicators (per DRL expert recommendation)
        result.append("State ", style="dim")
        for abbrev, ent_field, grad_field in HEAD_CONFIG:
            entropy = getattr(tamiyo, ent_field)
            grad = getattr(tamiyo, grad_field)
            state, style = self._head_state(abbrev, entropy, grad)
            result.append(f"{state:^{self.CELL_WIDTH}}", style=style)

        return result

    def _entropy_color(self, head: str, entropy: float) -> str:
        """Get color for entropy value based on normalized level."""
        max_ent = HEAD_MAX_ENTROPIES.get(head, 1.0)
        normalized = entropy / max_ent if max_ent > 0 else 0

        if normalized > 0.5:
            return "green"  # Healthy exploration
        elif normalized > 0.25:
            return "yellow"  # Getting focused
        else:
            return "red"  # Collapsed or near-collapsed

    def _render_entropy_bar(self, head: str, entropy: float) -> Text:
        """Render mini-bar for entropy."""
        max_ent = HEAD_MAX_ENTROPIES.get(head, 1.0)
        normalized = entropy / max_ent if max_ent > 0 else 0
        normalized = max(0, min(1, normalized))

        filled = int(normalized * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled

        color = self._entropy_color(head, entropy)
        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        return bar

    def _gradient_color(self, grad_norm: float) -> str:
        """Get color for gradient norm value.

        Healthy range: 0.1 - 2.0
        Warning: 0.01-0.1 (vanishing) or 2.0-5.0 (strong)
        Critical: <0.01 (dead) or >5.0 (exploding)
        """
        if grad_norm < 0.01:
            return "red"  # Vanishing
        elif grad_norm < 0.1:
            return "yellow"  # Weak
        elif grad_norm <= 2.0:
            return "green"  # Healthy
        elif grad_norm <= 5.0:
            return "yellow"  # Strong
        else:
            return "red"  # Exploding

    def _head_state(self, head: str, entropy: float, grad_norm: float) -> tuple[str, str]:
        """Classify head state based on entropy and gradient.

        States (per DRL expert):
        - ● healthy: Active learning (moderate entropy, normal gradients)
        - ○ dead: Collapsed and not learning (low entropy + vanishing gradients)
        - ◐ confused: Can't discriminate (very high entropy, normal gradients)
        - ◇ deterministic: Converged to specific choice (low entropy, normal gradients)

        Returns:
            Tuple of (indicator_char, style).
        """
        max_ent = HEAD_MAX_ENTROPIES.get(head, 1.0)
        normalized_ent = entropy / max_ent if max_ent > 0 else 0

        # Gradient health check
        grad_is_dead = grad_norm < 0.01
        grad_is_normal = 0.1 <= grad_norm <= 2.0
        grad_is_exploding = grad_norm > 5.0

        # State classification
        if grad_is_dead and normalized_ent < 0.1:
            # Dead: collapsed entropy AND vanishing gradients
            return "○", "red"

        if grad_is_exploding:
            # Exploding gradients - always bad
            return "▲", "red bold"

        if normalized_ent < 0.1:
            # Very low entropy with normal gradients = deterministic (may be OK)
            # Conditional heads often have low entropy when their op isn't selected
            if head in CONDITIONAL_HEADS:
                return "◇", "dim"  # Expected for conditional heads
            return "◇", "yellow"  # Concerning for always-active heads

        if normalized_ent > 0.9 and grad_is_normal:
            # High entropy with normal gradients = confused (can't decide)
            return "◐", "yellow"

        if 0.3 <= normalized_ent <= 0.7 and grad_is_normal:
            # Ideal: moderate entropy with healthy gradients
            return "●", "green"

        # Default: acceptable but not ideal
        return "●", "dim"

    def _render_gradient_bar(self, grad_norm: float) -> Text:
        """Render mini-bar for gradient norm.

        Uses log scale to handle wide range of values.
        """
        bar = Text()

        if grad_norm == 0:
            bar.append("░" * self.BAR_WIDTH, style="dim")
            return bar

        # Map gradient to fill level using log scale
        if grad_norm < 0.01:
            fill = 0.1
        elif grad_norm < 0.1:
            fill = 0.2
        elif grad_norm <= 2.0:
            # Healthy range maps to 0.3-1.0
            log_val = math.log10(grad_norm)
            fill = 0.5 + (log_val + 1) * 0.35
            fill = max(0.3, min(1.0, fill))
        elif grad_norm <= 5.0:
            fill = 0.8
        else:
            fill = 1.0

        filled = int(fill * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled

        color = self._gradient_color(grad_norm)
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        return bar
