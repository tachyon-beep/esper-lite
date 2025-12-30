"""AttentionHeatmapPanel - Per-decision policy head outputs visualization.

Shows the output of each policy head for Tamiyo's recent decisions,
with confidence indicated by heat bar intensity.

All 8 factored action heads are displayed (see leyline/factored_actions.py):
- Op: Operation type (GERM, ALPH, FOSS, PRUN, WAIT, ADVN)
- Slot: Target slot position (r0c0, r1c2, etc.)
- Blueprint: Module type for GERMINATE (conv_lt, attn, mlp, etc.)
- Style: Blending algorithm (LIN_ADD, GATED, ATTN_BL, RESID)
- Tempo: Learning speed (FST, STD, SLW)
- αTarget: Target alpha amplitude (50%, 70%, 100%)
- αSpeed: Alpha ramp speed (INST, FST, MED, SLW)
- Curve: Alpha easing function (LIN, COS, SIG, SIG_G, SIG_H)

Layout:
    ┌─ HEAD OUTPUTS ─────────────────────────────────────────────────────────────────────────────────┐
    │  Dec          Op        Slot     Blueprint       Style       Tempo   αTarget    αSpeed   Curve │
    │   #1   GERM█████   r0c0█████   conv_lt████   LIN_ADD███   STD███░░   70%████   MED████   COS██ │
    │   #2   WAIT█████           -             -            -          -         -         -       - │
    └────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot


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

# Blueprint display names (use standard leyline names, abbreviated for display)
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


class AttentionHeatmapPanel(Static):
    """Per-decision head choices with confidence heat.

    Shows the actual choices Tamiyo made for each head, with
    color intensity indicating confidence level.
    """

    MAX_ROWS: ClassVar[int] = 5

    # Column widths (expanded for wider heat indicators)
    COL_DEC: ClassVar[int] = 5
    COL_OP: ClassVar[int] = 13
    COL_SLOT: ClassVar[int] = 11
    COL_BLUEPRINT: ClassVar[int] = 14
    COL_STYLE: ClassVar[int] = 14
    COL_TEMPO: ClassVar[int] = 11
    COL_ALPHA_TGT: ClassVar[int] = 12
    COL_ALPHA_SPD: ClassVar[int] = 12
    COL_CURVE: ClassVar[int] = 11

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "HEAD OUTPUTS"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def _rjust_cell(self, result: Text, content: str, width: int, style: str) -> None:
        """Append right-justified content to result."""
        padding = max(0, width - len(content))
        result.append(" " * padding + content, style=style)

    def render(self) -> Text:
        """Render the head choices table."""
        if self._snapshot is None:
            return self._render_placeholder()

        decisions = self._snapshot.tamiyo.recent_decisions[:self.MAX_ROWS]
        if not decisions:
            return self._render_placeholder()

        result = Text()

        # Header row (right-aligned)
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
        for i, decision in enumerate(decisions):
            self._render_decision_row(result, i, decision)
            if i < len(decisions) - 1:
                result.append("\n")

        # Pad to minimum rows for consistent height
        rows_shown = len(decisions)
        total_width = (
            self.COL_DEC + self.COL_OP + self.COL_SLOT + self.COL_BLUEPRINT +
            self.COL_STYLE + self.COL_TEMPO + self.COL_ALPHA_TGT +
            self.COL_ALPHA_SPD + self.COL_CURVE
        )
        for _ in range(rows_shown, self.MAX_ROWS):
            result.append("\n")
            result.append("-" * total_width, style="dim")

        # Extra lines for visual breathing room
        result.append("\n\n")

        return result

    def _render_placeholder(self) -> Text:
        """Render placeholder when no decision data is available."""
        result = Text()

        # Header row (right-aligned)
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

        # Show placeholder rows
        for i in range(self.MAX_ROWS):
            self._rjust_cell(result, f"#{i + 1}", self.COL_DEC, "dim")
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

    def _render_decision_row(
        self, result: Text, index: int, decision: "DecisionSnapshot"
    ) -> None:
        """Render a single decision row with head choices."""
        # Row label (right-aligned)
        label_style = "bright_white" if index == 0 else "dim"
        self._rjust_cell(result, f"#{index + 1}", self.COL_DEC, label_style)

        # Op (action) with confidence heat (right-aligned)
        action = decision.chosen_action
        action_abbrev = ACTION_ABBREVS.get(action, action[:4])
        action_color = ACTION_COLORS.get(action, "white")
        op_conf = decision.op_confidence if decision.op_confidence > 0 else decision.confidence
        heat = self._confidence_heat(op_conf)
        op_content = f"{action_abbrev}{heat}"
        self._rjust_cell(result, op_content, self.COL_OP, action_color)

        # Slot (right-aligned)
        if decision.chosen_slot:
            slot_abbrev = decision.chosen_slot[:4]  # "r0c0", etc.
            slot_conf = decision.slot_confidence if decision.slot_confidence > 0 else decision.confidence
            slot_heat = self._confidence_heat(slot_conf)
            slot_content = f"{slot_abbrev}{slot_heat}"
            self._rjust_cell(result, slot_content, self.COL_SLOT, "cyan")
        else:
            self._rjust_cell(result, "-", self.COL_SLOT, "dim")

        # Blueprint (right-aligned, only for GERMINATE)
        if decision.chosen_blueprint:
            bp = abbrev_blueprint(decision.chosen_blueprint)
            bp_heat = self._confidence_heat(decision.blueprint_confidence)
            bp_content = f"{bp}{bp_heat}"
            self._rjust_cell(result, bp_content, self.COL_BLUEPRINT, "green")
        else:
            self._rjust_cell(result, "-", self.COL_BLUEPRINT, "dim")

        # Style (right-aligned, blending algorithm)
        if decision.chosen_style:
            style = STYLE_ABBREVS.get(decision.chosen_style, decision.chosen_style[:7])
            style_heat = self._confidence_heat(decision.style_confidence)
            style_content = f"{style}{style_heat}"
            self._rjust_cell(result, style_content, self.COL_STYLE, "blue")
        else:
            self._rjust_cell(result, "-", self.COL_STYLE, "dim")

        # Tempo (right-aligned, only for GERMINATE)
        if decision.chosen_tempo:
            tempo = TEMPO_ABBREVS.get(decision.chosen_tempo, decision.chosen_tempo[:3])
            tempo_heat = self._confidence_heat(decision.tempo_confidence)
            tempo_content = f"{tempo}{tempo_heat}"
            self._rjust_cell(result, tempo_content, self.COL_TEMPO, "yellow")
        else:
            self._rjust_cell(result, "-", self.COL_TEMPO, "dim")

        # Alpha Target (right-aligned, for GERMINATE or SET_ALPHA_TARGET)
        if decision.chosen_alpha_target:
            alpha_tgt = ALPHA_TARGET_ABBREVS.get(
                decision.chosen_alpha_target, decision.chosen_alpha_target[:4]
            )
            alpha_tgt_heat = self._confidence_heat(decision.alpha_target_confidence)
            alpha_tgt_content = f"{alpha_tgt}{alpha_tgt_heat}"
            self._rjust_cell(result, alpha_tgt_content, self.COL_ALPHA_TGT, "bright_cyan")
        else:
            self._rjust_cell(result, "-", self.COL_ALPHA_TGT, "dim")

        # Alpha Speed (right-aligned, for SET_ALPHA_TARGET or PRUNE)
        if decision.chosen_alpha_speed:
            alpha_spd = ALPHA_SPEED_ABBREVS.get(
                decision.chosen_alpha_speed, decision.chosen_alpha_speed[:4]
            )
            alpha_spd_heat = self._confidence_heat(decision.alpha_speed_confidence)
            alpha_spd_content = f"{alpha_spd}{alpha_spd_heat}"
            self._rjust_cell(result, alpha_spd_content, self.COL_ALPHA_SPD, "bright_yellow")
        else:
            self._rjust_cell(result, "-", self.COL_ALPHA_SPD, "dim")

        # Curve (right-aligned, for SET_ALPHA_TARGET or PRUNE)
        if decision.chosen_curve:
            curve = CURVE_ABBREVS.get(decision.chosen_curve, decision.chosen_curve[:3])
            curve_heat = self._confidence_heat(decision.curve_confidence)
            curve_content = f"{curve}{curve_heat}"
            self._rjust_cell(result, curve_content, self.COL_CURVE, "magenta")
        else:
            self._rjust_cell(result, "-", self.COL_CURVE, "dim")

    def _confidence_heat(self, confidence: float) -> str:
        """Convert confidence to heat indicator (5 chars)."""
        if confidence >= 0.9:
            return "█████"
        elif confidence >= 0.8:
            return "████░"
        elif confidence >= 0.7:
            return "███░░"
        elif confidence >= 0.5:
            return "██░░░"
        elif confidence >= 0.3:
            return "█░░░░"
        else:
            return "░░░░░"
