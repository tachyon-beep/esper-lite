"""HeadsPanel - Action head entropy and gradient heatmaps.

Uses vertical layout to ensure values align directly under their labels:

    ┌─ ACTION HEADS ──────────────────────────────────────────────────────────────────────────────────┐
    │              Op        Slot    Blueprint        Style      Tempo     αTarget      αSpeed      Curve │
    │ Entr       0.89        1.00         1.00         0.21       1.00        0.57        1.00       1.00 │
    │            ▓▓▓░        ████         ████         ▓░░░       ████        ▓▓░░        ████       ████ │
    │ Grad       0.13        0.13         0.21         0.27       0.19        0.10        0.21       0.12 │
    │            ▓░░░        ▓░░░         ▓░░░         ▓░░░       ▓░░░        ▓░░░        ▓░░░       ▓░░░ │
    │ State         ●           ◇            ◇            ○          ◇           ●           ◇          ◇ │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.leyline import (
    DEFAULT_HOST_LSTM_LAYERS,
    HEAD_MAX_ENTROPIES,
    is_head_relevant,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


# Head configuration (label, entropy_field, grad_norm_field, width, entropy_coef)
# Order and widths match HEAD OUTPUTS panel for vertical alignment
# entropy_coef: Differential entropy coefficient from Policy V2 (1.3x for sparse heads)
HEAD_CONFIG: list[tuple[str, str, str, int, float]] = [
    ("Op", "head_op_entropy", "head_op_grad_norm", 7, 1.0),
    ("Slot", "head_slot_entropy", "head_slot_grad_norm", 7, 1.0),
    ("Blueprint", "head_blueprint_entropy", "head_blueprint_grad_norm", 10, 1.3),
    ("Style", "head_style_entropy", "head_style_grad_norm", 9, 1.2),
    ("Tempo", "head_tempo_entropy", "head_tempo_grad_norm", 9, 1.3),
    ("αTarget", "head_alpha_target_entropy", "head_alpha_target_grad_norm", 9, 1.2),
    ("αSpeed", "head_alpha_speed_entropy", "head_alpha_speed_grad_norm", 9, 1.2),
    ("Curve", "head_alpha_curve_entropy", "head_alpha_curve_grad_norm", 9, 1.2),
]


def _get_head_key(entropy_field: str) -> str:
    """Extract lowercase head key from entropy field name.

    Maps HEAD_CONFIG field names to Leyline's HEAD_MAX_ENTROPIES keys.
    Example: "head_alpha_target_entropy" -> "alpha_target"
    """
    return entropy_field.removeprefix("head_").removesuffix("_entropy")


# Drift detection: ensure HEAD_CONFIG matches Leyline's HEAD_MAX_ENTROPIES
_CONFIG_HEAD_KEYS = {_get_head_key(ent_field) for _, ent_field, *_ in HEAD_CONFIG}
_LEYLINE_HEAD_KEYS = set(HEAD_MAX_ENTROPIES.keys())
assert _CONFIG_HEAD_KEYS == _LEYLINE_HEAD_KEYS, (
    f"HEAD_CONFIG and Leyline's HEAD_MAX_ENTROPIES must have matching keys. "
    f"Config has: {_CONFIG_HEAD_KEYS}, Leyline has: {_LEYLINE_HEAD_KEYS}"
)


class HeadsPanel(Static):
    """Action head entropy and gradient display panel.

    Extends Static directly (like DecisionCard) to eliminate Container
    layout overhead that causes whitespace issues.
    """

    BAR_WIDTH: ClassVar[int] = 5  # Width of mini-bar
    # Alignment shim: match column boundaries with AttentionHeatmapPanel
    # ("ACTION HEAD OUTPUTS") so scanning between the two panels is effortless.
    #
    # Reference widths in AttentionHeatmapPanel:
    # Dec=5, Op=13, Slot=11, Blueprint=14, Style=14, Tempo=11, αTarget=12, αSpeed=12, Curve=11
    _PRE_OP_GUTTER: ClassVar[int] = 5
    _GUTTER_BY_LABEL: ClassVar[dict[str, int]] = {
        "Op": 4,
        "Slot": 4,
        "Blueprint": 5,
        "Style": 2,
        "Tempo": 3,
        "αTarget": 3,
        "αSpeed": 2,
        "Curve": 2,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "ACTION HEADS"

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
        last_col = len(HEAD_CONFIG) - 1

        # Row 0: Header labels (plain text, right-aligned like HEAD OUTPUTS)
        result.append("      ", style="dim")  # Indent for row label
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, _, width, _) in enumerate(HEAD_CONFIG):
            result.append(f"{label:>{width}}", style="dim bold")
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 1: Entropy values with coefficient markers (Policy V2)
        # Note: getattr without default - AttributeError if HEAD_CONFIG has typo
        result.append("Entr  ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, coef) in enumerate(HEAD_CONFIG):
            entropy: float = getattr(tamiyo, ent_field)
            head_key = _get_head_key(ent_field)
            color = self._entropy_color(head_key, entropy)

            # Show coefficient for sparse heads (>1.0)
            # Each column is exactly `width` chars, with 2-space gutter between columns
            if coef > 1.0:
                # Format: "1.3×0.22" (coefficient first, then value)
                coef_prefix = f"{coef:.1f}×"  # "1.3×" (4 chars)
                value = f"{entropy:.2f}"  # "0.22" (4 chars)
                combined = coef_prefix + value  # e.g., "1.3×0.22" (8 chars)
                padding = width - len(combined)
                result.append(" " * padding, style="dim")
                result.append(coef_prefix, style="cyan dim")
                result.append(value, style=color)
            else:
                result.append(f"{entropy:>{width}.2f}", style=color)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 2: Entropy bars
        result.append("      ", style="dim")  # Indent
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, _) in enumerate(HEAD_CONFIG):
            entropy = getattr(tamiyo, ent_field)
            head_key = _get_head_key(ent_field)
            bar = self._render_entropy_bar(head_key, entropy)
            # Right-align the bar in the cell (like HEAD OUTPUTS heat bars)
            padding = width - self.BAR_WIDTH
            result.append(" " * padding)
            result.append(bar)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 3: Gradient values with trend arrows (Policy V2)
        result.append("Grad  ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, grad_field, width, _) in enumerate(HEAD_CONFIG):
            grad: float = getattr(tamiyo, grad_field)
            grad_prev: float = getattr(tamiyo, f"{grad_field}_prev")
            trend = self._gradient_trend(grad, grad_prev)
            color = self._gradient_color(grad)

            # Value + trend arrow (1 char) = exactly `width` chars total
            # e.g., " 0.35↗" (6 chars for width=7: 5 for value + 1 for arrow)
            value = f"{grad:.2f}"  # "0.35" (4 chars)
            combined = value + trend  # e.g., "0.35↗" (5 chars)
            padding = width - len(combined)
            result.append(" " * padding, style="dim")
            result.append(value, style=color)
            result.append(trend, style=self._gradient_trend_style(grad, grad_prev))
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 4: Gradient bars
        result.append("      ", style="dim")  # Indent
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, grad_field, width, _) in enumerate(HEAD_CONFIG):
            grad = getattr(tamiyo, grad_field)
            bar = self._render_gradient_bar(grad)
            # Right-align the bar
            padding = width - self.BAR_WIDTH
            result.append(" " * padding)
            result.append(bar)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 5: Head state indicators (per DRL expert + causal relevance)
        result.append("State ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        last_op = tamiyo.last_action_op
        for col_idx, (label, ent_field, grad_field, width, _) in enumerate(HEAD_CONFIG):
            entropy = getattr(tamiyo, ent_field)
            grad = getattr(tamiyo, grad_field)
            head_key = _get_head_key(ent_field)
            state, style_str = self._head_state(head_key, entropy, grad, last_op)
            # Center state indicator under the 5-char bar (bars are right-aligned)
            # Bar center is at width - 3, so state should be at that position
            # For width=7: 4 spaces + indicator + 2 spaces = 7 chars total
            # For width=9: 6 spaces + indicator + 2 spaces = 9 chars total
            # For width=10: 7 spaces + indicator + 2 spaces = 10 chars total
            left_pad = width - 3
            right_pad = 2
            result.append(" " * left_pad, style="dim")
            result.append(state, style=style_str)
            result.append(" " * right_pad, style="dim")
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 6-7: Gradient Flow Footer (split across 2 lines for readability)
        result.append("Flow: ", style="dim")

        # Line 1: CV and layer health
        cv = tamiyo.gradient_quality.gradient_cv
        cv_status = "stable" if cv < 0.5 else ("warn" if cv < 2.0 else "BAD")
        cv_style = "green" if cv < 0.5 else ("yellow" if cv < 2.0 else "red")
        result.append(f"CV:{cv:.2f} ", style=cv_style)
        result.append(f"{cv_status}   ", style="dim")

        # Dead/Exploding layers (use existing TamiyoState fields)
        dead = tamiyo.dead_layers
        exploding = tamiyo.exploding_layers
        total = DEFAULT_HOST_LSTM_LAYERS
        layers_style = "green" if (dead == 0 and exploding == 0) else "red"
        result.append(
            f"Dead:{dead}/{total}   Exploding:{exploding}/{total}", style=layers_style
        )

        # Line 2: Directional clip
        result.append("\n")
        result.append("      ", style="dim")  # Indent to align with "Flow:"
        clip_pos = tamiyo.gradient_quality.clip_fraction_positive
        clip_neg = tamiyo.gradient_quality.clip_fraction_negative
        result.append(f"Clip:\u2191{clip_pos:.0%}/\u2193{clip_neg:.0%}", style="dim")

        return result

    def _column_gutter(self, label: str) -> int:
        """Return number of spaces after a column label.

        Used to align selected columns with the ACTION HEAD OUTPUTS panel.
        """
        return self._GUTTER_BY_LABEL[label]

    def _entropy_color(self, head_key: str, entropy: float) -> str:
        """Get color for entropy value based on normalized level.

        Args:
            head_key: Lowercase head key matching Leyline's HEAD_MAX_ENTROPIES
                      (e.g., "slot", "alpha_target")
        """
        max_ent = HEAD_MAX_ENTROPIES[head_key]  # Direct access - fail fast on drift
        normalized = entropy / max_ent if max_ent > 0 else 0

        if normalized > 0.5:
            return "green"  # Healthy exploration
        elif normalized > 0.25:
            return "yellow"  # Getting focused
        else:
            return "red"  # Collapsed or near-collapsed

    def _render_entropy_bar(self, head_key: str, entropy: float) -> Text:
        """Render mini-bar for entropy.

        Args:
            head_key: Lowercase head key matching Leyline's HEAD_MAX_ENTROPIES
        """
        max_ent = HEAD_MAX_ENTROPIES[head_key]  # Direct access - fail fast on drift
        normalized = entropy / max_ent if max_ent > 0 else 0
        normalized = max(0, min(1, normalized))

        filled = int(normalized * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled

        color = self._entropy_color(head_key, entropy)
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

    def _gradient_trend(self, grad: float, grad_prev: float) -> str:
        """Get trend arrow for gradient health (Policy V2).

        Enables distinguishing transient spikes (noisy batch) from sustained issues.
        """
        EPSILON = 0.01
        delta = grad - grad_prev

        if abs(delta) < EPSILON:
            return "→"  # Stable
        elif delta > 0:
            return "↗"  # Rising
        else:
            return "↘"  # Falling

    def _gradient_trend_style(self, grad: float, grad_prev: float) -> str:
        """Style for gradient trend arrow based on health context."""
        # Rising gradients are good if in vanishing range, bad if in exploding range
        # Falling gradients are good if in exploding range, bad if in vanishing range
        delta = grad - grad_prev
        EPSILON = 0.01

        if abs(delta) < EPSILON:
            return "dim"  # Stable

        # Context-aware coloring
        if grad < 0.1:  # Vanishing range
            return "green" if delta > 0 else "red"  # Rising = good, falling = bad
        elif grad > 2.0:  # Exploding range
            return "red" if delta > 0 else "green"  # Rising = bad, falling = good
        else:  # Healthy range
            return "dim"  # Changes are neutral in healthy range

    def _head_state(
        self, head_key: str, entropy: float, grad_norm: float, last_op: str
    ) -> tuple[str, str]:
        """Classify head state based on entropy, gradient, and causal relevance.

        Args:
            head_key: Lowercase head key (e.g., "slot", "alpha_target")
            entropy: Current entropy value for the head
            grad_norm: Current gradient norm for the head
            last_op: Last LifecycleOp name (e.g., "WAIT", "GERMINATE")

        States (per DRL expert + causal relevance):
        - · irrelevant: Head not causally relevant for current op (dimmed)
        - ● healthy: Active learning (moderate entropy, normal gradients)
        - ○ dead: Collapsed and not learning (low entropy + vanishing gradients)
        - ◐ confused: Can't discriminate (very high entropy, normal gradients)
        - ◇ deterministic: Converged to specific choice (low entropy, normal gradients)

        Returns:
            Tuple of (indicator_char, style).
        """
        # Check causal relevance using Leyline's single source of truth
        if not is_head_relevant(last_op, head_key):
            # Head not causally relevant for current op - dim, not dead
            return "·", "dim"

        max_ent = HEAD_MAX_ENTROPIES[head_key]  # Direct access - fail fast on drift
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
            # Very low entropy with normal gradients = deterministic
            # This is concerning - the head IS relevant but collapsed
            return "◇", "yellow"

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
