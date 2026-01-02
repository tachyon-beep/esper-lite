"""ActionHeadsPanel - Unified action head health and output visualization.

Combines head entropy/gradient/ratio metrics with per-decision output heatmaps.

Layout:
    ┌─ ACTION HEADS ─────────────────────────────────────────────────────────────────────────────────────────┐
    │               Op        Slot     Blueprint        Style       Tempo     αTarget      αSpeed       Curve │
    │ Entr       0.893       1.000        1.000        0.215       1.000       0.571       1.000       1.000 │
    │            ▓▓▓░        ████         ████         ▓░░░        ████        ▓▓░░        ████        ████ │
    │ Grad      0.131→     0.135↗       0.211→       0.275↘      0.192→      0.101↗      0.213→      0.124→ │
    │            ▓░░░        ▓░░░         ▓░░░         ▓░░░        ▓░░░        ▓░░░        ▓░░░        ▓░░░ │
    │ Ratio      1.023       0.987        1.142        1.312       0.891       1.056       1.001       0.956 │
    │            ░░░░        ░░░░         ░░░░         █░░░        ░░░░        ░░░░        ░░░░        ░░░░ │
    │ State         ●           ●            ●            ○           ●           ●           ●           ● │
    │─────────────────────────────────────────────────────────────────────────────────────────────────────────│
    │  Dec          Op        Slot     Blueprint       Style       Tempo   αTarget    αSpeed   Curve         │
    │   #1   GERM█████   r0c0█████   conv_lt████   LIN_ADD███   STD███░░   70%████   MED████   COS██         │
    │   #2   WAIT█████           -             -            -          -         -         -       -         │
    │   ...                                                                                                   │
    │─────────────────────────────────────────────────────────────────────────────────────────────────────────│
    │ Flow: CV:0.123 stable   Dead:0/4   Exploding:0/4                                                       │
    │       Clip:↑2.5%/↓3.1%                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

State indicators synthesize entropy + gradient health:
  ● healthy (green): Moderate entropy + normal gradients
  ○ dead (red): Collapsed entropy + vanishing gradients
  ◐ confused (yellow): Very high entropy
  ◇ deterministic (dim): Expected - e.g., slot head with single slot
  ◇ deterministic (yellow): Concerning - policy collapsed with multiple choices
  ▲ exploding (red): Gradient explosion

Ratio row shows π_new/π_old max per head:
  Green (0.8-1.2): Within PPO clip range, stable updates
  Yellow (0.5-1.5): Moderate policy change, monitor closely
  Red (<0.5 or >1.5): Large policy shift, possible instability
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.leyline import (
    DEFAULT_HOST_LSTM_LAYERS,
    HEAD_MAX_ENTROPIES,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


# =============================================================================
# Head Configuration
# =============================================================================

# Head configuration (label, entropy_field, grad_norm_field, width, entropy_coef)
# Order and widths match decision output columns for vertical alignment
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


# =============================================================================
# Decision Output Abbreviations
# =============================================================================

# Action abbreviations (4 chars max)
ACTION_ABBREVS = {
    "GERMINATE": "GERM",
    "SET_ALPHA_TARGET": "ALPH",
    "FOSSILIZE": "FOSS",
    "PRUNE": "PRUN",
    "WAIT": "WAIT",
    "ADVANCE": "ADVN",
}

# Action colors
ACTION_COLORS = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
    "ADVANCE": "cyan",
}

# Tempo abbreviations
TEMPO_ABBREVS = {
    "FAST": "FST",
    "STANDARD": "STD",
    "SLOW": "SLW",
}

# Style abbreviations (blending algorithm)
STYLE_ABBREVS = {
    "LINEAR_ADD": "LIN_ADD",
    "GATED_GATE": "GATED",
    "ATTENTION_BLEND": "ATTN_BL",
    "RESIDUAL_ADD": "RESID",
}

# Curve abbreviations (alpha easing function)
CURVE_ABBREVS = {
    "LINEAR": "LIN",
    "COSINE": "COS",
    "SIGMOID_GENTLE": "SIG_G",
    "SIGMOID": "SIG",
    "SIGMOID_SHARP": "SIG_H",
}

# Alpha target abbreviations
ALPHA_TARGET_ABBREVS = {
    "HALF": "50%",
    "SEVENTY": "70%",
    "FULL": "100%",
}

# Alpha speed abbreviations
ALPHA_SPEED_ABBREVS = {
    "INSTANT": "INST",
    "FAST": "FST",
    "MEDIUM": "MED",
    "SLOW": "SLW",
}


def abbrev_blueprint(bp: str | None) -> str:
    """Abbreviate blueprint name for display."""
    if not bp:
        return "-"
    # Standard abbreviations matching leyline/factored_actions.py BlueprintAction
    abbrevs = {
        "conv_light": "conv_lt",
        "conv_small": "conv_sm",
        "conv_heavy": "conv_hv",
        "attention": "attn",
        "flex_attention": "flex_at",
        "norm": "norm",
        "depthwise": "depthw",
        "bottleneck": "btlneck",
        "lora": "lora",
        "lora_large": "lora_lg",
        "mlp_small": "mlp_sm",
        "mlp": "mlp",
        "noop": "noop",
    }
    return abbrevs.get(bp, bp[:7])


# =============================================================================
# Panel Implementation
# =============================================================================


class ActionHeadsPanel(Static):
    """Unified action head health and output visualization panel.

    Combines:
    - Head entropy/gradient metrics with state indicators
    - Per-decision output heatmaps with confidence heat bars
    - Gradient flow footer (CV, dead/exploding layers, clip fractions)
    """

    # Head health display
    BAR_WIDTH: ClassVar[int] = 5  # Width of mini-bar

    # Alignment shim: match column boundaries for visual scanning
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

    # Decision carousel (8-row with 5s staggering)
    MAX_ROWS: ClassVar[int] = 8
    SWAP_INTERVAL_S: ClassVar[float] = 5.0
    MAX_DISPLAY_AGE_S: ClassVar[float] = 40.0
    AGE_PIP_CHAR: ClassVar[str] = "●"
    AGE_PIP_EMPTY: ClassVar[str] = "○"

    # Decision column widths
    COL_DEC: ClassVar[int] = 5
    COL_OP: ClassVar[int] = 13
    COL_SLOT: ClassVar[int] = 11
    COL_BLUEPRINT: ClassVar[int] = 14
    COL_STYLE: ClassVar[int] = 14
    COL_TEMPO: ClassVar[int] = 11
    COL_ALPHA_TGT: ClassVar[int] = 12
    COL_ALPHA_SPD: ClassVar[int] = 12
    COL_CURVE: ClassVar[int] = 11

    # Separator width
    SEPARATOR_WIDTH: ClassVar[int] = 103

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._displayed_decisions: list[DecisionSnapshot] = []
        self._last_swap_ts: float = 0.0
        self.classes = "panel"
        self.border_title = "ACTION HEADS"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot

        # Update decision carousel
        incoming = list(snapshot.tamiyo.recent_decisions)
        if incoming:
            self._refresh_carousel(incoming)

        self.refresh()  # Trigger render()

    def render(self) -> Text:
        """Render the unified panel: heads section + decisions + footer."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        result = Text()

        # Section 1: Head health metrics
        result.append(self._render_heads_section())

        # Separator
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 2: Decision outputs
        result.append(self._render_decision_table())

        # Separator
        result.append("\n")
        result.append("─" * self.SEPARATOR_WIDTH + "\n", style="dim")

        # Section 3: Gradient flow footer
        result.append(self._render_flow_footer())

        return result

    # =========================================================================
    # Section 1: Head Health Metrics
    # =========================================================================

    def _render_heads_section(self) -> Text:
        """Render the heads grid with vertical alignment."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()
        last_col = len(HEAD_CONFIG) - 1

        # Row 0: Header labels (plain text, right-aligned)
        result.append("      ", style="dim")  # Indent for row label
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, _, width, _) in enumerate(HEAD_CONFIG):
            result.append(f"{label:>{width}}", style="dim bold")
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 1: Entropy values with coefficient markers (Policy V2)
        result.append("Entr  ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, coef) in enumerate(HEAD_CONFIG):
            entropy: float = getattr(tamiyo, ent_field)
            head_key = _get_head_key(ent_field)
            color = self._entropy_color(head_key, entropy)

            if coef > 1.0:
                coef_prefix = f"{coef:.1f}×"
                value = f"{entropy:.3f}"
                combined = coef_prefix + value
                padding = width - len(combined)
                result.append(" " * padding, style="dim")
                result.append(coef_prefix, style="cyan dim")
                result.append(value, style=color)
            else:
                result.append(f"{entropy:>{width}.3f}", style=color)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 2: Entropy bars
        result.append("      ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, _) in enumerate(HEAD_CONFIG):
            entropy = getattr(tamiyo, ent_field)
            head_key = _get_head_key(ent_field)
            bar = self._render_entropy_bar(head_key, entropy)
            padding = width - self.BAR_WIDTH
            result.append(" " * padding)
            result.append(bar)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 3: Gradient values with trend arrows (shifted right so arrow hangs over bar)
        result.append("Grad  ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, grad_field, width, _) in enumerate(HEAD_CONFIG):
            grad: float = getattr(tamiyo, grad_field)
            grad_prev: float = getattr(tamiyo, f"{grad_field}_prev")
            trend = self._gradient_trend(grad, grad_prev)
            color = self._gradient_color(grad)

            # Shift value+arrow 1 char right so arrow hangs over the bar below
            value = f"{grad:.3f}"
            combined = value + trend
            padding = width - len(combined) + 1
            result.append(" " * padding, style="dim")
            result.append(value, style=color)
            result.append(trend, style=self._gradient_trend_style(grad, grad_prev))
            if col_idx != last_col:
                # Reduce gutter by 1 to compensate for right shift
                result.append(" " * (self._column_gutter(label) - 1))
        result.append("\n")

        # Row 4: Gradient bars
        result.append("      ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, _, grad_field, width, _) in enumerate(HEAD_CONFIG):
            grad = getattr(tamiyo, grad_field)
            bar = self._render_gradient_bar(grad)
            padding = width - self.BAR_WIDTH
            result.append(" " * padding)
            result.append(bar)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 5: Ratio values (π_new/π_old max)
        result.append("Ratio ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, _) in enumerate(HEAD_CONFIG):
            head_key = _get_head_key(ent_field)
            ratio_field = f"head_{head_key}_ratio_max"
            ratio: float = getattr(tamiyo, ratio_field, 1.0)
            color = self._ratio_color(ratio)

            result.append(f"{ratio:>{width}.3f}", style=color)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 6: Ratio bars
        result.append("      ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        for col_idx, (label, ent_field, _, width, _) in enumerate(HEAD_CONFIG):
            head_key = _get_head_key(ent_field)
            ratio_field = f"head_{head_key}_ratio_max"
            ratio = getattr(tamiyo, ratio_field, 1.0)
            bar = self._render_ratio_bar(ratio)
            padding = width - self.BAR_WIDTH
            result.append(" " * padding)
            result.append(bar)
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        # Row 7: Head state indicators
        result.append("State ", style="dim")
        result.append(" " * self._PRE_OP_GUTTER, style="dim")
        n_slots = len(self._snapshot.slot_ids) if self._snapshot else 0
        for col_idx, (label, ent_field, grad_field, width, _) in enumerate(HEAD_CONFIG):
            entropy = getattr(tamiyo, ent_field)
            grad = getattr(tamiyo, grad_field)
            head_key = _get_head_key(ent_field)
            state, style_str = self._head_state(head_key, entropy, grad, n_slots)
            left_pad = width - 3
            right_pad = 2
            result.append(" " * left_pad, style="dim")
            result.append(state, style=style_str)
            result.append(" " * right_pad, style="dim")
            if col_idx != last_col:
                result.append(" " * self._column_gutter(label))
        result.append("\n")

        return result

    # =========================================================================
    # Section 2: Decision Outputs
    # =========================================================================

    def _refresh_carousel(self, incoming: list["DecisionSnapshot"]) -> None:
        """Update the displayed rows with a 5s-staggered carousel."""
        displayed_ids = {d.decision_id for d in self._displayed_decisions if d.decision_id}
        candidates = [d for d in incoming if d.decision_id and d.decision_id not in displayed_ids]
        if not candidates:
            return

        candidates.sort(key=lambda d: d.timestamp, reverse=True)
        now = time.monotonic()

        # Growing phase: add immediately until full
        if len(self._displayed_decisions) < self.MAX_ROWS:
            needed = self.MAX_ROWS - len(self._displayed_decisions)
            to_add = candidates[:needed]
            for decision in reversed(to_add):
                self._displayed_decisions.insert(0, decision)
            if len(self._displayed_decisions) == self.MAX_ROWS and self._last_swap_ts == 0.0:
                self._last_swap_ts = now
            return

        # Steady state: swap at most once per interval
        if now - self._last_swap_ts < self.SWAP_INTERVAL_S:
            return

        self._displayed_decisions.insert(0, candidates[0])
        if len(self._displayed_decisions) > self.MAX_ROWS:
            self._displayed_decisions.pop()
        self._last_swap_ts = now

    def _render_decision_table(self) -> Text:
        """Render the decision output table."""
        decisions = self._displayed_decisions
        if not decisions:
            return self._render_decision_placeholder()

        now_dt = datetime.now(timezone.utc)
        result = Text()

        # Header row
        self._rjust_cell(result, "Dec", self.COL_DEC, "dim bold")
        self._rjust_cell(result, "Op", self.COL_OP, "dim bold")
        self._rjust_cell(result, "Slot", self.COL_SLOT, "dim bold")
        self._rjust_cell(result, "Blueprint", self.COL_BLUEPRINT, "dim bold")
        self._rjust_cell(result, "Style", self.COL_STYLE, "dim bold")
        self._rjust_cell(result, "Tempo", self.COL_TEMPO, "dim bold")
        self._rjust_cell(result, "αTarget", self.COL_ALPHA_TGT, "dim bold")
        self._rjust_cell(result, "αSpeed", self.COL_ALPHA_SPD, "dim bold")
        self._rjust_cell(result, "Curve", self.COL_CURVE, "dim bold")
        result.append("\n")

        # Decision rows
        for i in range(self.MAX_ROWS):
            decision = decisions[i] if i < len(decisions) else None
            if decision is not None:
                self._render_decision_row(result, i, decision, now_dt=now_dt)
            else:
                self._render_empty_row(result, i, now_dt=now_dt)
            if i < self.MAX_ROWS - 1:
                result.append("\n")

        return result

    def _render_decision_placeholder(self) -> Text:
        """Render placeholder when no decision data is available."""
        result = Text()

        # Header row
        self._rjust_cell(result, "Dec", self.COL_DEC, "dim bold")
        self._rjust_cell(result, "Op", self.COL_OP, "dim bold")
        self._rjust_cell(result, "Slot", self.COL_SLOT, "dim bold")
        self._rjust_cell(result, "Blueprint", self.COL_BLUEPRINT, "dim bold")
        self._rjust_cell(result, "Style", self.COL_STYLE, "dim bold")
        self._rjust_cell(result, "Tempo", self.COL_TEMPO, "dim bold")
        self._rjust_cell(result, "αTarget", self.COL_ALPHA_TGT, "dim bold")
        self._rjust_cell(result, "αSpeed", self.COL_ALPHA_SPD, "dim bold")
        self._rjust_cell(result, "Curve", self.COL_CURVE, "dim bold")
        result.append("\n")

        for i in range(self.MAX_ROWS):
            self._render_dec_cell(result, row_index=i, decision=None, now_dt=datetime.now(timezone.utc))
            self._rjust_cell(result, "---", self.COL_OP, "dim")
            self._rjust_cell(result, "---", self.COL_SLOT, "dim")
            self._rjust_cell(result, "---", self.COL_BLUEPRINT, "dim")
            self._rjust_cell(result, "---", self.COL_STYLE, "dim")
            self._rjust_cell(result, "---", self.COL_TEMPO, "dim")
            self._rjust_cell(result, "---", self.COL_ALPHA_TGT, "dim")
            self._rjust_cell(result, "---", self.COL_ALPHA_SPD, "dim")
            self._rjust_cell(result, "---", self.COL_CURVE, "dim")
            if i < self.MAX_ROWS - 1:
                result.append("\n")
        return result

    def _render_empty_row(self, result: Text, index: int, *, now_dt: datetime) -> None:
        """Render an empty decision row."""
        self._render_dec_cell(result, row_index=index, decision=None, now_dt=now_dt)
        self._rjust_cell(result, "---", self.COL_OP, "dim")
        self._rjust_cell(result, "---", self.COL_SLOT, "dim")
        self._rjust_cell(result, "---", self.COL_BLUEPRINT, "dim")
        self._rjust_cell(result, "---", self.COL_STYLE, "dim")
        self._rjust_cell(result, "---", self.COL_TEMPO, "dim")
        self._rjust_cell(result, "---", self.COL_ALPHA_TGT, "dim")
        self._rjust_cell(result, "---", self.COL_ALPHA_SPD, "dim")
        self._rjust_cell(result, "---", self.COL_CURVE, "dim")

    def _render_decision_row(
        self, result: Text, index: int, decision: "DecisionSnapshot", *, now_dt: datetime
    ) -> None:
        """Render a single decision row with head choices."""
        self._render_dec_cell(result, row_index=index, decision=decision, now_dt=now_dt)

        # Op (action) with confidence heat
        action = decision.chosen_action
        action_abbrev = ACTION_ABBREVS.get(action, action[:4])
        action_color = ACTION_COLORS.get(action, "white")
        op_conf = decision.op_confidence if decision.op_confidence > 0 else decision.confidence
        heat = self._confidence_heat(op_conf)
        op_content = f"{action_abbrev}{heat}"
        self._rjust_cell(result, op_content, self.COL_OP, action_color)

        # Slot
        if decision.chosen_slot:
            slot_abbrev = decision.chosen_slot[:4]
            slot_conf = decision.slot_confidence if decision.slot_confidence > 0 else decision.confidence
            slot_heat = self._confidence_heat(slot_conf)
            slot_content = f"{slot_abbrev}{slot_heat}"
            self._rjust_cell(result, slot_content, self.COL_SLOT, "cyan")
        else:
            self._rjust_cell(result, "-", self.COL_SLOT, "dim")

        # Blueprint
        if decision.chosen_blueprint:
            bp = abbrev_blueprint(decision.chosen_blueprint)
            bp_heat = self._confidence_heat(decision.blueprint_confidence)
            bp_content = f"{bp}{bp_heat}"
            self._rjust_cell(result, bp_content, self.COL_BLUEPRINT, "green")
        else:
            self._rjust_cell(result, "-", self.COL_BLUEPRINT, "dim")

        # Style
        if decision.chosen_style:
            style = STYLE_ABBREVS.get(decision.chosen_style, decision.chosen_style[:7])
            style_heat = self._confidence_heat(decision.style_confidence)
            style_content = f"{style}{style_heat}"
            self._rjust_cell(result, style_content, self.COL_STYLE, "blue")
        else:
            self._rjust_cell(result, "-", self.COL_STYLE, "dim")

        # Tempo
        if decision.chosen_tempo:
            tempo = TEMPO_ABBREVS.get(decision.chosen_tempo, decision.chosen_tempo[:3])
            tempo_heat = self._confidence_heat(decision.tempo_confidence)
            tempo_content = f"{tempo}{tempo_heat}"
            self._rjust_cell(result, tempo_content, self.COL_TEMPO, "yellow")
        else:
            self._rjust_cell(result, "-", self.COL_TEMPO, "dim")

        # Alpha Target
        if decision.chosen_alpha_target:
            alpha_tgt = ALPHA_TARGET_ABBREVS.get(
                decision.chosen_alpha_target, decision.chosen_alpha_target[:4]
            )
            alpha_tgt_heat = self._confidence_heat(decision.alpha_target_confidence)
            alpha_tgt_content = f"{alpha_tgt}{alpha_tgt_heat}"
            self._rjust_cell(result, alpha_tgt_content, self.COL_ALPHA_TGT, "bright_cyan")
        else:
            self._rjust_cell(result, "-", self.COL_ALPHA_TGT, "dim")

        # Alpha Speed
        if decision.chosen_alpha_speed:
            alpha_spd = ALPHA_SPEED_ABBREVS.get(
                decision.chosen_alpha_speed, decision.chosen_alpha_speed[:4]
            )
            alpha_spd_heat = self._confidence_heat(decision.alpha_speed_confidence)
            alpha_spd_content = f"{alpha_spd}{alpha_spd_heat}"
            self._rjust_cell(result, alpha_spd_content, self.COL_ALPHA_SPD, "bright_yellow")
        else:
            self._rjust_cell(result, "-", self.COL_ALPHA_SPD, "dim")

        # Curve
        if decision.chosen_curve:
            curve = CURVE_ABBREVS.get(decision.chosen_curve, decision.chosen_curve[:3])
            curve_heat = self._confidence_heat(decision.curve_confidence)
            curve_content = f"{curve}{curve_heat}"
            self._rjust_cell(result, curve_content, self.COL_CURVE, "magenta")
        else:
            self._rjust_cell(result, "-", self.COL_CURVE, "dim")

    def _rjust_cell(self, result: Text, content: str, width: int, style: str) -> None:
        """Append right-justified content to result."""
        padding = max(0, width - len(content))
        result.append(" " * padding + content, style=style)

    def _render_dec_cell(
        self,
        result: Text,
        *,
        row_index: int,
        decision: "DecisionSnapshot | None",
        now_dt: datetime,
    ) -> None:
        """Render the decision number cell with age pip."""
        label = f"#{row_index + 1}"
        if decision is None:
            content = f"{self.AGE_PIP_EMPTY}{label}"
            self._rjust_cell(result, content, self.COL_DEC, "dim")
            return

        age_s = max(0.0, (now_dt - decision.timestamp).total_seconds())
        padding = max(0, self.COL_DEC - (len(self.AGE_PIP_CHAR) + len(label)))
        result.append(" " * padding, style="dim")
        result.append(self.AGE_PIP_CHAR, style=self._age_pip_style(age_s))
        label_style = "bright_white" if row_index == 0 else "dim"
        result.append(label, style=label_style)

    def _age_pip_style(self, age_s: float) -> str:
        """Return Rich style for age pip (green → yellow → brown → red)."""
        if self.MAX_DISPLAY_AGE_S <= 0:
            return "dim"
        ratio = age_s / self.MAX_DISPLAY_AGE_S
        if ratio < 0.40:
            return "green"
        if ratio < 0.70:
            return "yellow"
        if ratio < 0.90:
            return "#8B4513"  # brown (saddle brown)
        return "red"

    def _confidence_heat(self, confidence: float) -> str:
        """Convert confidence to heat indicator (5 chars)."""
        if confidence <= 0.0:
            return "░░░░░"
        if confidence >= 1.0:
            return "█████"
        if confidence >= 0.75:
            return "████░"
        if confidence >= 0.5:
            return "███░░"
        if confidence >= 0.25:
            return "██░░░"
        return "█░░░░"

    # =========================================================================
    # Section 3: Gradient Flow Footer
    # =========================================================================

    def _render_flow_footer(self) -> Text:
        """Render the gradient flow footer (moved from heads section)."""
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Line 1: CV and layer health
        result.append("Flow: ", style="dim")

        cv = tamiyo.gradient_quality.gradient_cv
        cv_status = "stable" if cv < 0.5 else ("warn" if cv < 2.0 else "BAD")
        cv_style = "green" if cv < 0.5 else ("yellow" if cv < 2.0 else "red")
        result.append(f"CV:{cv:.3f} ", style=cv_style)
        result.append(f"{cv_status}   ", style="dim")

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
        result.append(f"Clip:↑{clip_pos:.1%}/↓{clip_neg:.1%}", style="dim")

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _column_gutter(self, label: str) -> int:
        """Return number of spaces after a column label."""
        return self._GUTTER_BY_LABEL[label]

    def _entropy_color(self, head_key: str, entropy: float) -> str:
        """Get color for entropy value based on normalized level."""
        max_ent = HEAD_MAX_ENTROPIES[head_key]
        normalized = entropy / max_ent if max_ent > 0 else 0

        if normalized > 0.5:
            return "green"
        elif normalized > 0.25:
            return "yellow"
        else:
            return "red"

    def _render_entropy_bar(self, head_key: str, entropy: float) -> Text:
        """Render mini-bar for entropy."""
        max_ent = HEAD_MAX_ENTROPIES[head_key]
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
        """Get color for gradient norm value."""
        if grad_norm < 0.01:
            return "red"
        elif grad_norm < 0.1:
            return "yellow"
        elif grad_norm <= 2.0:
            return "green"
        elif grad_norm <= 5.0:
            return "yellow"
        else:
            return "red"

    def _gradient_trend(self, grad: float, grad_prev: float) -> str:
        """Get trend arrow for gradient health."""
        EPSILON = 0.01
        delta = grad - grad_prev

        if abs(delta) < EPSILON:
            return "→"
        elif delta > 0:
            return "↗"
        else:
            return "↘"

    def _gradient_trend_style(self, grad: float, grad_prev: float) -> str:
        """Style for gradient trend arrow based on health context."""
        delta = grad - grad_prev
        EPSILON = 0.01

        if abs(delta) < EPSILON:
            return "dim"

        if grad < 0.1:
            return "green" if delta > 0 else "red"
        elif grad > 2.0:
            return "red" if delta > 0 else "green"
        else:
            return "dim"

    def _head_state(
        self, head_key: str, entropy: float, grad_norm: float, n_slots: int = 0
    ) -> tuple[str, str]:
        """Classify head state based on entropy and gradient health."""
        max_ent = HEAD_MAX_ENTROPIES[head_key]
        normalized_ent = entropy / max_ent if max_ent > 0 else 0

        grad_is_dead = grad_norm < 0.01
        grad_is_normal = 0.1 <= grad_norm <= 2.0
        grad_is_exploding = grad_norm > 5.0

        if grad_is_dead and normalized_ent < 0.1:
            return "○", "red"

        if grad_is_exploding:
            return "▲", "red bold"

        if normalized_ent < 0.1:
            if head_key == "slot" and n_slots == 1:
                return "◇", "dim"
            return "◇", "yellow"

        if normalized_ent > 0.9 and grad_is_normal:
            return "◐", "yellow"

        if 0.3 <= normalized_ent <= 0.7 and grad_is_normal:
            return "●", "green"

        return "●", "dim"

    def _render_gradient_bar(self, grad_norm: float) -> Text:
        """Render mini-bar for gradient norm using log scale."""
        bar = Text()

        if grad_norm == 0:
            bar.append("░" * self.BAR_WIDTH, style="dim")
            return bar

        if grad_norm < 0.01:
            fill = 0.1
        elif grad_norm < 0.1:
            fill = 0.2
        elif grad_norm <= 2.0:
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

    def _ratio_color(self, ratio: float) -> str:
        """Get color for probability ratio based on PPO clip range.

        PPO uses clip range ε ≈ 0.2, so ideal ratio is in [0.8, 1.2].
        Larger deviations indicate significant policy changes.
        """
        if 0.8 <= ratio <= 1.2:
            return "green"  # Within clip range
        elif 0.5 <= ratio <= 1.5:
            return "yellow"  # Moderate deviation
        else:
            return "red"  # Large policy shift

    def _render_ratio_bar(self, ratio: float) -> Text:
        """Render mini-bar for ratio showing deviation from 1.0.

        Bar fills based on distance from unity: 1.0 = empty, edges = full.
        Uses asymmetric log scale since ratio > 1 and < 1 have different ranges.
        """
        bar = Text()

        # Deviation from unity (log scale for symmetry around 1.0)
        if ratio <= 0:
            fill = 1.0
        else:
            log_ratio = abs(math.log(ratio))  # 0 at ratio=1, grows for deviations
            # Map: log(0.5)≈0.69, log(1.5)≈0.41, log(2.0)≈0.69
            fill = min(1.0, log_ratio / 0.7)  # Full at ratio=0.5 or ratio=2.0

        filled = int(fill * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled

        color = self._ratio_color(ratio)
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        return bar


# Backwards compatibility alias (temporary - remove after layout update)
HeadsPanel = ActionHeadsPanel
