"""EnvDetailScreen - Full-screen modal for detailed environment and seed inspection.

Displays comprehensive diagnostics for a single environment including:
- Per-seed cards with stage, blueprint, alpha, gradient health
- Environment metrics: accuracy history, action distribution, reward breakdown
- Triggered by 'D' key or Enter on DataTable row
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EnvState, SeedState


# Stage color mapping
STAGE_COLORS = {
    "DORMANT": "dim",
    "GERMINATED": "bright_blue",
    "TRAINING": "cyan",
    "PROBATIONARY": "magenta",
    "BLENDING": "yellow",
    "FOSSILIZED": "green",
    "CULLED": "red",
}

# Stage border styles for CSS classes
STAGE_CSS_CLASSES = {
    "DORMANT": "dormant",
    "GERMINATED": "training",  # Use training style for germinated
    "TRAINING": "training",
    "PROBATIONARY": "blending",  # Use blending style
    "BLENDING": "blending",
    "FOSSILIZED": "fossilized",
    "CULLED": "culled",
}


class SeedCard(Static):
    """Individual seed card showing detailed seed state."""

    def __init__(self, seed: "SeedState | None", slot_id: str, **kwargs) -> None:
        """Initialize seed card.

        Args:
            seed: SeedState for this slot, or None if dormant.
            slot_id: Slot identifier (e.g., "r0c0").
        """
        super().__init__(**kwargs)
        self._seed = seed
        self._slot_id = slot_id

    def compose(self):
        """No child widgets - we render directly."""
        yield from []

    def render(self) -> Panel:
        """Render the seed card as a Rich Panel."""
        if self._seed is None or self._seed.stage == "DORMANT":
            return self._render_dormant()
        return self._render_active()

    def _render_dormant(self) -> Panel:
        """Render dormant/empty slot."""
        content = Text()
        content.append(f"{self._slot_id}\n", style="dim")
        content.append("DORMANT", style="dim italic")
        return Panel(
            content,
            title=self._slot_id,
            border_style="dim",
            height=10,
        )

    def _render_active(self) -> Panel:
        """Render active seed with all metrics."""
        seed = self._seed
        stage_color = STAGE_COLORS.get(seed.stage, "white")

        lines = []

        # Stage with color
        lines.append(Text(f"{seed.stage}", style=f"bold {stage_color}"))

        # Blueprint
        blueprint = seed.blueprint_id or "unknown"
        lines.append(Text(f"Blueprint: {blueprint}", style="white"))

        # Parameters
        if seed.seed_params > 0:
            if seed.seed_params >= 1_000_000:
                params_str = f"{seed.seed_params / 1_000_000:.1f}M"
            elif seed.seed_params >= 1_000:
                params_str = f"{seed.seed_params / 1_000:.1f}K"
            else:
                params_str = str(seed.seed_params)
            lines.append(Text(f"Params: {params_str}", style="dim"))

        # Alpha (blending progress)
        if seed.alpha > 0 or seed.stage in ("BLENDING", "PROBATIONARY"):
            alpha_bar = self._make_alpha_bar(seed.alpha)
            lines.append(Text(f"Alpha: {seed.alpha:.2f} {alpha_bar}"))

        # Accuracy delta
        if seed.accuracy_delta != 0:
            delta_style = "green" if seed.accuracy_delta > 0 else "red"
            lines.append(Text(f"Acc Δ: {seed.accuracy_delta:+.2f}%", style=delta_style))
        else:
            lines.append(Text("Acc Δ: --", style="dim"))

        # Gradient health
        grad_text = Text("Grad: ")
        if seed.has_exploding:
            grad_text.append("▲ EXPLODING", style="bold red")
        elif seed.has_vanishing:
            grad_text.append("▼ VANISHING", style="bold yellow")
        elif seed.grad_ratio > 0:
            grad_text.append(f"ratio={seed.grad_ratio:.2f}", style="green")
        else:
            grad_text.append("OK", style="green")
        lines.append(grad_text)

        # Epochs in stage
        lines.append(Text(f"Epochs: {seed.epochs_in_stage}", style="dim"))

        # Combine into panel content
        content = Text("\n").join(lines)

        return Panel(
            content,
            title=f"[{stage_color}]{self._slot_id}[/{stage_color}]",
            border_style=stage_color,
            height=10,
        )

    def _make_alpha_bar(self, alpha: float, width: int = 10) -> str:
        """Create a text-based progress bar for alpha."""
        filled = int(alpha * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"


class EnvDetailScreen(ModalScreen[None]):
    """Full-screen modal for detailed environment and seed inspection.

    Shows comprehensive diagnostics for a single environment:
    - Header with env summary
    - Seed grid with per-slot cards
    - Environment metrics section
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    EnvDetailScreen {
        align: center middle;
        background: $surface-darken-1 90%;
    }

    EnvDetailScreen > #modal-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    EnvDetailScreen .header-bar {
        height: 3;
        padding: 0 1;
        background: $surface-lighten-1;
        margin-bottom: 1;
    }

    EnvDetailScreen .seed-grid {
        height: auto;
        min-height: 12;
    }

    EnvDetailScreen .metrics-section {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }

    EnvDetailScreen .footer-hint {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        env_state: "EnvState",
        slot_ids: list[str],
        **kwargs,
    ) -> None:
        """Initialize the detail screen.

        Args:
            env_state: The environment state to display.
            slot_ids: List of slot IDs for the seed grid.
        """
        super().__init__(**kwargs)
        self._env = env_state
        self._slot_ids = slot_ids

    def compose(self):
        """Compose the modal layout."""
        with Container(id="modal-container"):
            # Header bar
            yield Static(self._render_header(), classes="header-bar")

            # Seed grid
            with Horizontal(classes="seed-grid"):
                for slot_id in self._slot_ids:
                    seed = self._env.seeds.get(slot_id)
                    yield SeedCard(seed, slot_id)

            # Metrics section
            with Vertical(classes="metrics-section"):
                yield Static(self._render_metrics())

            # Footer hint
            yield Static(
                "[dim]Press ESC or Q to close[/dim]",
                classes="footer-hint",
            )

    def _render_header(self) -> Text:
        """Render the header bar with env summary."""
        env = self._env

        header = Text()

        # Env ID
        header.append(f"Environment {env.env_id}", style="bold")
        header.append("  │  ")

        # Status
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }
        status_style = status_styles.get(env.status, "white")
        header.append(env.status.upper(), style=status_style)
        header.append("  │  ")

        # Best accuracy
        header.append(f"Best: {env.best_accuracy:.1f}%", style="cyan")
        header.append("  │  ")

        # Current accuracy
        header.append(f"Current: {env.host_accuracy:.1f}%", style="white")
        header.append("  │  ")

        # Epochs since improvement
        if env.epochs_since_improvement > 0:
            stale_style = "red" if env.epochs_since_improvement > 10 else "yellow"
            header.append(
                f"Stale: {env.epochs_since_improvement} epochs",
                style=stale_style,
            )
        else:
            header.append("Improving", style="green")

        return header

    def _render_metrics(self) -> Table:
        """Render environment metrics section."""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", style="white")

        env = self._env

        # Accuracy history sparkline
        from esper.karn.sanctum.schema import make_sparkline
        acc_spark = make_sparkline(env.accuracy_history, width=40)
        table.add_row("Accuracy History", acc_spark)

        # Reward history sparkline
        rwd_spark = make_sparkline(env.reward_history, width=40)
        table.add_row("Reward History", rwd_spark)

        # Seed counts
        seed_counts = Text()
        seed_counts.append(f"Active: {env.active_seed_count}", style="cyan")
        seed_counts.append("  ")
        seed_counts.append(f"Fossilized: {env.fossilized_count}", style="green")
        seed_counts.append("  ")
        seed_counts.append(f"Culled: {env.culled_count}", style="red")
        table.add_row("Seed Counts", seed_counts)

        # Fossilized params
        if env.fossilized_params > 0:
            if env.fossilized_params >= 1_000_000:
                params_str = f"{env.fossilized_params / 1_000_000:.2f}M"
            elif env.fossilized_params >= 1_000:
                params_str = f"{env.fossilized_params / 1_000:.1f}K"
            else:
                params_str = str(env.fossilized_params)
            table.add_row("Fossilized Params", params_str)

        # Action distribution
        if env.total_actions > 0:
            action_text = Text()
            for action, count in sorted(env.action_counts.items()):
                pct = (count / env.total_actions) * 100
                # Color coding by action type
                action_colors = {
                    "WAIT": "dim",
                    "GERMINATE": "cyan",
                    "FOSSILIZE": "green",
                    "CULL": "red",
                }
                color = action_colors.get(action, "white")
                action_text.append(f"{action}: {pct:.0f}%", style=color)
                action_text.append("  ")
            table.add_row("Action Distribution", action_text)

        # Reward components
        rc = env.reward_components
        if rc.total != 0:
            reward_text = Text()
            reward_text.append(f"Total: {rc.total:+.3f}", style="bold")
            if rc.base_acc_delta != 0:
                style = "green" if rc.base_acc_delta > 0 else "red"
                reward_text.append(f"  ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
            if rc.compute_rent != 0:
                reward_text.append(f"  Rent: {rc.compute_rent:.3f}", style="red")
            if rc.bounded_attribution != 0:
                style = "green" if rc.bounded_attribution > 0 else "red"
                reward_text.append(f"  Attr: {rc.bounded_attribution:+.3f}", style=style)
            table.add_row("Reward Breakdown", reward_text)

        # Recent actions
        if env.action_history:
            recent = " → ".join(list(env.action_history)[-5:])
            table.add_row("Recent Actions", recent)

        return table
