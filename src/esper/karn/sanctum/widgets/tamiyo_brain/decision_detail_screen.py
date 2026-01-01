"""DecisionDetailScreen - Drill-down view for a single Tamiyo decision."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import DecisionSnapshot


class DecisionDetailScreen(ModalScreen[None]):
    """Modal showing full details for one DecisionSnapshot."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    DEFAULT_CSS = """
    DecisionDetailScreen {
        align: center middle;
        background: $surface-darken-1 80%;
    }

    DecisionDetailScreen > #detail-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    DecisionDetailScreen #detail-title {
        height: 1;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    DecisionDetailScreen #detail-scroll {
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
    }
    """

    def __init__(self, *, decision: "DecisionSnapshot", group_id: str) -> None:
        super().__init__()
        self._decision = decision
        self._group_id = group_id

    def compose(self) -> ComposeResult:
        with Container(id="detail-container"):
            yield Static(self._render_title(), id="detail-title")
            with VerticalScroll(id="detail-scroll"):
                yield Static(self._render_detail())

    def _render_title(self) -> Text:
        title = Text()
        title.append("TAMIYO DECISION DETAIL", style="bold")
        title.append("  ", style="dim")
        title.append(f"[{self._group_id}]", style="dim")
        return title

    def _render_detail(self) -> Text:
        d = self._decision
        now = datetime.now(timezone.utc)
        age_s = max(0.0, (now - d.timestamp).total_seconds())
        age_str = f"{age_s:.0f}s" if age_s < 60 else f"{int(age_s // 60)}:{int(age_s % 60):02d}"

        t = Text()

        t.append("Summary\n", style="bold cyan")
        t.append(f"  Decision ID: {d.decision_id or '(missing)'}\n", style="dim")
        t.append(f"  Time:       {d.timestamp.isoformat()}\n", style="dim")
        t.append(f"  Age:        {age_str}\n", style="dim")
        t.append(f"  Env:        {d.env_id}\n", style="dim")
        t.append(f"  Epoch:      {d.epoch}\n", style="dim")
        t.append(f"  Batch:      {d.batch}\n", style="dim")
        t.append("\n")

        t.append("Action\n", style="bold cyan")
        t.append(f"  Op:         {d.chosen_action}\n", style="dim")
        t.append(f"  Slot:       {d.chosen_slot or '—'}\n", style="dim")
        t.append(f"  Confidence: {d.confidence:.0%}\n", style="dim")
        t.append(f"  Entropy:    {d.decision_entropy:.3f}\n", style="dim")
        t.append("\n")

        t.append("Values\n", style="bold cyan")
        t.append(f"  V(s):       {d.expected_value:+.3f}\n", style="dim")
        t.append(
            f"  Reward:     {d.actual_reward:+.3f}\n" if d.actual_reward is not None else "  Reward:     pending\n",
            style="dim",
        )
        t.append(f"  Residual:   {d.value_residual:+.3f}\n", style="dim")
        t.append(
            f"  TD(0):      {d.td_advantage:+.3f}\n"
            if d.td_advantage is not None
            else "  TD(0):      pending\n",
            style="dim",
        )
        t.append("\n")

        t.append("Factored Heads\n", style="bold cyan")
        t.append(f"  Blueprint:  {d.chosen_blueprint or '—'}  (p={d.blueprint_confidence:.0%}, H={d.blueprint_entropy:.3f})\n", style="dim")
        t.append(f"  Tempo:      {d.chosen_tempo or '—'}  (p={d.tempo_confidence:.0%}, H={d.tempo_entropy:.3f})\n", style="dim")
        t.append(f"  Style:      {d.chosen_style or '—'}  (p={d.style_confidence:.0%}, H={d.style_entropy:.3f})\n", style="dim")
        t.append(f"  Curve:      {d.chosen_curve or '—'}  (p={d.curve_confidence:.0%}, H={d.curve_entropy:.3f})\n", style="dim")
        t.append(f"  α Target:   {d.chosen_alpha_target or '—'}  (p={d.alpha_target_confidence:.0%}, H={d.alpha_target_entropy:.3f})\n", style="dim")
        t.append(f"  α Speed:    {d.chosen_alpha_speed or '—'}  (p={d.alpha_speed_confidence:.0%}, H={d.alpha_speed_entropy:.3f})\n", style="dim")
        t.append("\n")

        t.append("Slot State\n", style="bold cyan")
        if d.slot_states:
            for slot_id, state in sorted(d.slot_states.items()):
                t.append(f"  {slot_id}: {state}\n", style="dim")
        else:
            t.append("  (no slot state captured)\n", style="dim")
        t.append("\n")

        t.append("Alternatives\n", style="bold cyan")
        if d.alternatives:
            for action, prob in d.alternatives:
                t.append(f"  {action:<18} {prob:.0%}\n", style="dim")
        else:
            t.append("  (none captured)\n", style="dim")

        return t

