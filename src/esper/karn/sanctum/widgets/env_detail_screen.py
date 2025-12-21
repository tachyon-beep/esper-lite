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

from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EnvState, SeedState


# Stage color mapping
STAGE_COLORS = {
    "DORMANT": "dim",
    "GERMINATED": "bright_blue",
    "TRAINING": "cyan",
    "HOLDING": "magenta",
    "BLENDING": "yellow",
    "FOSSILIZED": "green",
    "PRUNED": "red",
}

# Stage border styles for CSS classes
STAGE_CSS_CLASSES = {
    "DORMANT": "dormant",
    "GERMINATED": "training",  # Use training style for germinated
    "TRAINING": "training",
    "HOLDING": "blending",  # Use blending style
    "BLENDING": "blending",
    "FOSSILIZED": "fossilized",
    "PRUNED": "pruned",
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

    def update_seed(self, seed: "SeedState | None") -> None:
        """Update the seed state and refresh display.

        Args:
            seed: New SeedState for this slot, or None if dormant.
        """
        self._seed = seed
        self.refresh()

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
        if seed.seed_params and seed.seed_params > 0:
            if seed.seed_params >= 1_000_000:
                params_str = f"{seed.seed_params / 1_000_000:.1f}M"
            elif seed.seed_params >= 1_000:
                params_str = f"{seed.seed_params / 1_000:.1f}K"
            else:
                params_str = str(seed.seed_params)
            lines.append(Text(f"Params: {params_str}", style="dim"))

        # Alpha (blending progress)
        if (seed.alpha and seed.alpha > 0) or seed.stage in ("BLENDING", "HOLDING"):
            alpha_bar = self._make_alpha_bar(seed.alpha)
            lines.append(Text(f"Alpha: {seed.alpha:.2f} {alpha_bar}"))

        # Blend tempo (shown during BLENDING and for FOSSILIZED to show how they were blended)
        if seed.stage in ("BLENDING", "FOSSILIZED") and seed.blend_tempo_epochs is not None:
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            # For fossilized, show "was blended" in past tense
            if seed.stage == "FOSSILIZED":
                lines.append(Text(f"Blended: {tempo_arrows} {tempo_name}", style="dim"))
            else:
                lines.append(Text(f"Tempo: {tempo_arrows} {tempo_name} ({tempo} epochs)"))

        # Accuracy delta (stage-aware display)
        # TRAINING/GERMINATED seeds have alpha=0 and cannot affect output
        if seed.stage in ("TRAINING", "GERMINATED"):
            lines.append(Text("Acc Δ: 0.0 (learning)", style="dim italic"))
        elif seed.accuracy_delta is not None and seed.accuracy_delta != 0:
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
        elif seed.grad_ratio is not None and seed.grad_ratio > 0:
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

    EnvDetailScreen SeedCard {
        width: 1fr;
        height: auto;
        min-height: 10;
    }

    EnvDetailScreen .metrics-section {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }

    EnvDetailScreen .counterfactual-section {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }

    EnvDetailScreen .graveyard-section {
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
        self._env_id = env_state.env_id  # Track env_id for updates

    @property
    def env_id(self) -> int:
        """Return the environment ID this screen is showing."""
        return self._env_id

    def compose(self):
        """Compose the modal layout."""
        with Container(id="modal-container"):
            # Header bar
            yield Static(self._render_header(), id="detail-header", classes="header-bar")

            # Seed grid
            with Horizontal(classes="seed-grid"):
                for slot_id in self._slot_ids:
                    seed = self._env.seeds.get(slot_id)
                    yield SeedCard(seed, slot_id, id=f"seed-card-{slot_id}")

            # Metrics section
            with Vertical(classes="metrics-section"):
                yield Static(self._render_metrics(), id="detail-metrics")

            # Counterfactual analysis section
            with Vertical(classes="counterfactual-section"):
                yield CounterfactualPanel(
                    self._env.counterfactual_matrix,
                    id="counterfactual-panel"
                )

            # Seed graveyard section
            with Vertical(classes="graveyard-section"):
                yield Static(self._render_graveyard(), id="seed-graveyard")

            # Footer hint
            yield Static(
                "[dim]Press ESC or Q to close[/dim]",
                classes="footer-hint",
            )

    def update_env_state(self, env_state: "EnvState") -> None:
        """Update the displayed environment state.

        Called by the app during refresh to keep modal in sync with live data.

        Args:
            env_state: Updated environment state.
        """
        self._env = env_state

        # Update header
        try:
            header = self.query_one("#detail-header", Static)
            header.update(self._render_header())
        except Exception:
            pass  # Widget may not be mounted yet

        # Update metrics
        try:
            metrics = self.query_one("#detail-metrics", Static)
            metrics.update(self._render_metrics())
        except Exception:
            pass  # Widget may not be mounted yet

        # Update counterfactual panel
        try:
            cf_panel = self.query_one("#counterfactual-panel", CounterfactualPanel)
            cf_panel.update_matrix(env_state.counterfactual_matrix)
        except Exception:
            pass

        # Update each seed card
        for slot_id in self._slot_ids:
            try:
                seed = self._env.seeds.get(slot_id)
                card = self.query_one(f"#seed-card-{slot_id}", SeedCard)
                card.update_seed(seed)
            except Exception:
                pass  # Widget may not be mounted yet

        # Update graveyard
        try:
            graveyard = self.query_one("#seed-graveyard", Static)
            graveyard.update(self._render_graveyard())
        except Exception:
            pass

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
        epochs_stale = env.epochs_since_improvement or 0
        if epochs_stale > 0:
            stale_style = "red" if epochs_stale > 10 else "yellow"
            header.append(
                f"Stale: {epochs_stale} epochs",
                style=stale_style,
            )
        else:
            header.append("Improving", style="green")

        # Host params, seed params, and growth ratio
        header.append("  │  ")

        # Format params in human-readable form
        def _format_params(p: int) -> str:
            if p >= 1_000_000:
                return f"{p / 1_000_000:.1f}M"
            elif p >= 1_000:
                return f"{p / 1_000:.1f}K"
            return str(p)

        host_str = _format_params(env.host_params or 0)
        fossilized = env.fossilized_params or 0
        seed_str = _format_params(fossilized)
        growth = env.growth_ratio or 1.0

        header.append(f"Host: {host_str}", style="dim")
        header.append(f"  +Seed: {seed_str}", style="green" if fossilized > 0 else "dim")

        # Growth ratio with color coding
        # 1.0x = no growth (dim), 1.0-1.2x = normal (green), >1.2x = significant (yellow)
        if growth > 1.2:
            growth_style = "yellow"
        elif growth > 1.0:
            growth_style = "green"
        else:
            growth_style = "dim"
        header.append(f"  = {growth:.2f}x", style=growth_style)

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
        seed_counts.append(f"Pruned: {env.pruned_count}", style="red")
        table.add_row("Seed Counts", seed_counts)

        # Fossilized params
        foss_params = env.fossilized_params or 0
        if foss_params > 0:
            if foss_params >= 1_000_000:
                params_str = f"{foss_params / 1_000_000:.2f}M"
            elif foss_params >= 1_000:
                params_str = f"{foss_params / 1_000:.1f}K"
            else:
                params_str = str(foss_params)
            table.add_row("Fossilized Params", params_str)

        # Action distribution
        total_actions = env.total_actions or 0
        if total_actions > 0:
            action_text = Text()
            for action, count in sorted(env.action_counts.items()):
                pct = (count / total_actions) * 100
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
        if rc.total is not None and rc.total != 0:
            reward_text = Text()
            reward_text.append(f"Total: {rc.total:+.3f}", style="bold")
            if rc.base_acc_delta is not None and rc.base_acc_delta != 0:
                style = "green" if rc.base_acc_delta > 0 else "red"
                reward_text.append(f"  ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
            if rc.compute_rent is not None and rc.compute_rent != 0:
                reward_text.append(f"  Rent: {rc.compute_rent:.3f}", style="red")
            if rc.bounded_attribution is not None and rc.bounded_attribution != 0:
                style = "green" if rc.bounded_attribution > 0 else "red"
                reward_text.append(f"  Attr: {rc.bounded_attribution:+.3f}", style=style)
            table.add_row("Reward Breakdown", reward_text)

        # Recent actions
        if env.action_history:
            recent = " → ".join(list(env.action_history)[-5:])
            table.add_row("Recent Actions", recent)

        return table

    def _render_graveyard(self) -> Panel:
        """Render the seed graveyard showing per-blueprint lifecycle stats.

        Shows how many seeds of each blueprint type have been:
        - Spawned (germinated)
        - Fossilized (successfully integrated)
        - Pruned (removed due to poor performance)
        """
        env = self._env

        # Combine all blueprints seen across spawns, fossilized, pruned
        all_blueprints = set(env.blueprint_spawns.keys())
        all_blueprints.update(env.blueprint_fossilized.keys())
        all_blueprints.update(env.blueprint_prunes.keys())

        if not all_blueprints:
            content = Text("No seeds germinated yet", style="dim italic")
            return Panel(content, title="Seed Graveyard", border_style="dim")

        # Build graveyard display
        lines = []
        for blueprint in sorted(all_blueprints):
            spawned = env.blueprint_spawns.get(blueprint, 0)
            fossilized = env.blueprint_fossilized.get(blueprint, 0)
            pruned = env.blueprint_prunes.get(blueprint, 0)

            line = Text()
            line.append(f"{blueprint:15s}", style="white")
            line.append(f"  spawn:{spawned:2d}", style="cyan")
            line.append(f"  foss:{fossilized:2d}", style="green")
            line.append(f"  prun:{pruned:2d}", style="red")

            # Calculate success rate if any have terminated
            terminated = fossilized + pruned
            if terminated > 0:
                success_rate = fossilized / terminated * 100
                rate_style = "green" if success_rate >= 50 else "yellow" if success_rate >= 25 else "red"
                line.append(f"  ({success_rate:.0f}% success)", style=rate_style)

            lines.append(line)

        content = Text("\n").join(lines) if lines else Text("No activity", style="dim")
        return Panel(content, title="Seed Graveyard", border_style="dim")
