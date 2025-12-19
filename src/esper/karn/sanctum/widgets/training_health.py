"""TrainingHealth widget - PPO training diagnostics at a glance.

Compact display of training health indicators:
- Entropy status (collapsed/warning/ok)
- Gradient health (dead/exploding layers)
- Action distribution summary
- Seed stage summary
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


# Threshold constants for training health indicators
ENTROPY_WARNING = 0.1  # Below this shows "LOW"
ENTROPY_CRITICAL = 0.01  # Below this shows "COLLAPSED"
KL_WARNING = 0.02  # Above this shows yellow


def _entropy_indicator(entropy: float, collapsed: bool) -> Text:
    """Get entropy status indicator.

    Returns a compact indicator like:
    - "Entropy: OK" (green)
    - "Entropy: LOW" (yellow)
    - "Entropy: COLLAPSED" (red)
    """
    text = Text()
    text.append("Entropy: ", style="dim")

    if collapsed or entropy < ENTROPY_CRITICAL:
        text.append("COLLAPSED", style="bold red")
    elif entropy < ENTROPY_WARNING:
        text.append("LOW", style="yellow")
    else:
        text.append(f"{entropy:.3f}", style="green")

    return text


def _gradient_indicator(dead: int, exploding: int, health: float) -> Text:
    """Get gradient health indicator.

    Returns compact indicator like:
    - "Gradients: OK" (green)
    - "Gradients: 2 dead" (yellow)
    - "Gradients: 3 exploding" (red)
    """
    text = Text()
    text.append("Gradients: ", style="dim")

    if exploding > 0:
        text.append(f"{exploding} exploding", style="bold red")
    elif dead > 0:
        text.append(f"{dead} dead", style="yellow")
    elif health < 0.5:
        text.append("UNHEALTHY", style="yellow")
    else:
        text.append("OK", style="green")

    return text


def _action_distribution(counts: dict[str, int], total: int) -> Text:
    """Get compact action distribution summary.

    Shows top 3 actions with percentages, e.g.:
    "Actions: WAIT 45% GERM 30% FOSS 25%"
    """
    text = Text()
    text.append("Actions: ", style="dim")

    if not counts or total == 0:
        text.append("--", style="dim")
        return text

    # Sort by count descending
    sorted_actions = sorted(counts.items(), key=lambda x: -x[1])

    # Abbreviations for compact display
    abbrev = {
        "WAIT": "WAIT",
        "GERMINATE": "GERM",
        "FOSSILIZE": "FOSS",
        "CULL": "CULL",
    }

    # Show top 3
    parts = []
    for action, count in sorted_actions[:3]:
        if count > 0:
            pct = (count / total) * 100
            name = abbrev.get(action, action[:4])
            parts.append((name, pct))

    for i, (name, pct) in enumerate(parts):
        if i > 0:
            text.append(" ", style="dim")
        # Color based on action type and percentage
        if name == "WAIT" and pct > 80:
            style = "yellow"  # Too much waiting
        elif name in ("GERM", "FOSS") and pct > 0:
            style = "cyan"  # Active decisions
        else:
            style = "white"
        text.append(f"{name}", style=style)
        text.append(f" {pct:.0f}%", style="dim")

    return text


def _seed_stage_summary(snapshot: "SanctumSnapshot") -> Text:
    """Get compact seed stage counts.

    Shows: "Seeds: T:5 B:2 F:8"
    """
    text = Text()
    text.append("Seeds: ", style="dim")

    # Count stages across all envs
    stages = {"TRAINING": 0, "BLENDING": 0, "PROBATIONARY": 0, "FOSSILIZED": 0, "CULLED": 0}
    for env in snapshot.envs.values():
        for seed in env.seeds.values():
            if seed.stage in stages:
                stages[seed.stage] += 1

    parts = []
    if stages["TRAINING"] > 0:
        parts.append(("T", stages["TRAINING"], "yellow"))
    if stages["BLENDING"] > 0 or stages["PROBATIONARY"] > 0:
        blend_count = stages["BLENDING"] + stages["PROBATIONARY"]
        parts.append(("B", blend_count, "cyan"))
    if stages["FOSSILIZED"] > 0:
        parts.append(("F", stages["FOSSILIZED"], "magenta"))
    if stages["CULLED"] > 0:
        parts.append(("X", stages["CULLED"], "red"))

    if not parts:
        text.append("--", style="dim")
    else:
        for i, (abbr, count, style) in enumerate(parts):
            if i > 0:
                text.append(" ", style="dim")
            text.append(f"{abbr}:", style="dim")
            text.append(str(count), style=style)

    return text


class TrainingHealth(Static):
    """Training health dashboard widget.

    Compact display of key training diagnostics:
    - Entropy status with collapse detection
    - Gradient health (dead/exploding layers)
    - Action distribution summary
    - Seed stage counts

    Uses color-coded indicators:
    - Green: Healthy
    - Yellow: Warning
    - Red: Critical
    """

    def __init__(self, **kwargs) -> None:
        """Initialize TrainingHealth widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Group:
        """Render training health indicators."""
        if self._snapshot is None:
            return Group(Text("Waiting for data...", style="dim"))

        tamiyo = self._snapshot.tamiyo
        lines = []

        # Entropy indicator
        lines.append(_entropy_indicator(tamiyo.entropy, tamiyo.entropy_collapsed))

        # Gradient health
        lines.append(_gradient_indicator(
            tamiyo.dead_layers,
            tamiyo.exploding_layers,
            tamiyo.layer_gradient_health,
        ))

        # Action distribution
        lines.append(_action_distribution(tamiyo.action_counts, tamiyo.total_actions))

        # Seed stage summary
        lines.append(_seed_stage_summary(self._snapshot))

        # KL divergence if available
        if tamiyo.kl_divergence > 0:
            kl_text = Text()
            kl_text.append("KL: ", style="dim")
            if tamiyo.kl_divergence > KL_WARNING:
                kl_text.append(f"{tamiyo.kl_divergence:.4f}", style="yellow")
            else:
                kl_text.append(f"{tamiyo.kl_divergence:.4f}", style="green")
            lines.append(kl_text)

        return Group(*lines)
