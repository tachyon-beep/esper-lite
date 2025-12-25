"""HistoricalEnvDetail - Modal for viewing frozen env state from Best Runs.

Shows the complete environment state as it was when the env achieved its
peak accuracy and was added to the leaderboard. Unlike EnvDetailScreen,
this is a static snapshot (no live updates).

Triggered by left-clicking a row in the Best Runs scoreboard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel
from esper.karn.sanctum.widgets.env_detail_screen import SeedCard, STAGE_COLORS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import BestRunRecord


class HistoricalEnvDetail(ModalScreen[None]):
    """Modal for viewing historical env snapshot from Best Runs leaderboard.

    Shows the complete environment state at the moment it achieved peak
    accuracy and was added to the leaderboard. This is a frozen view -
    no live updates occur.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HistoricalEnvDetail {
        align: center middle;
        background: $surface-darken-1 90%;
    }

    HistoricalEnvDetail > #modal-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $secondary;
        padding: 1 2;
    }

    HistoricalEnvDetail .header-bar {
        height: 4;
        padding: 0 1;
        background: $surface-lighten-1;
        margin-bottom: 1;
    }

    HistoricalEnvDetail .seed-grid {
        height: auto;
        min-height: 12;
    }

    HistoricalEnvDetail SeedCard {
        width: 1fr;
        height: auto;
        min-height: 10;
    }

    HistoricalEnvDetail .metrics-section {
        height: auto;
        margin-top: 1;
        border-top: solid $secondary-lighten-2;
        padding-top: 1;
    }

    HistoricalEnvDetail .counterfactual-section {
        height: auto;
        margin-top: 1;
        border-top: solid $secondary-lighten-2;
        padding-top: 1;
    }

    HistoricalEnvDetail .graveyard-section {
        height: auto;
        margin-top: 1;
        border-top: solid $secondary-lighten-2;
        padding-top: 1;
    }

    HistoricalEnvDetail .footer-hint {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, record: "BestRunRecord", **kwargs: Any) -> None:
        """Initialize the historical detail screen.

        Args:
            record: The BestRunRecord to display (frozen snapshot).
        """
        super().__init__(**kwargs)
        self._record = record

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="modal-container") as container:
            # Header bar
            yield Static(self._render_header(), id="detail-header", classes="header-bar")

            # Seed grid - show all slots (including DORMANT)
            # Uses slot_ids if available, otherwise falls back to seeds.keys()
            slot_ids = self._record.slot_ids or sorted(self._record.seeds.keys())
            if slot_ids:
                with Horizontal(classes="seed-grid"):
                    for slot_id in slot_ids:
                        seed = self._record.seeds.get(slot_id)  # None for DORMANT
                        yield SeedCard(seed, slot_id, id=f"seed-card-{slot_id}")

            # Metrics section
            with Vertical(classes="metrics-section"):
                yield Static(self._render_metrics(), id="detail-metrics")

            # Counterfactual analysis section (if available)
            if self._record.counterfactual_matrix is not None:
                with Vertical(classes="counterfactual-section"):
                    yield CounterfactualPanel(
                        self._record.counterfactual_matrix,
                        id="counterfactual-panel"
                    )

            # Seed graveyard section
            with Vertical(classes="graveyard-section"):
                yield Static(self._render_graveyard(), id="seed-graveyard")

            # Footer hint
            pin_status = "📌 Pinned" if self._record.pinned else "Not pinned (right-click to pin)"
            yield Static(
                f"[dim]Press ESC or Q to close  │  {pin_status}[/dim]",
                classes="footer-hint",
            )

        yield container

    def _render_header(self) -> Text:
        """Render the header bar with record summary."""
        record = self._record

        header = Text()

        # Historical banner
        header.append("📜 HISTORICAL VIEW", style="bold yellow")
        header.append("  │  ")

        # Episode number
        header.append(f"Episode {record.episode + 1}", style="bold")
        header.append("  │  ")

        # Env ID
        header.append(f"Env {record.env_id}", style="dim")
        header.append("  │  ")

        # Peak accuracy (the hero metric)
        header.append(f"Peak: {record.peak_accuracy:.1f}%", style="bold green")
        header.append("  │  ")

        # Final accuracy (what it ended at)
        if record.final_accuracy != record.peak_accuracy:
            header.append(f"Final: {record.final_accuracy:.1f}%", style="cyan")
            header.append("  │  ")

        # Growth ratio
        growth = record.growth_ratio or 1.0
        if growth > 1.0:
            growth_style = "yellow" if growth > 1.2 else "green"
            header.append(f"Growth: {growth:.2f}x", style=growth_style)
        else:
            header.append("Growth: 1.00x", style="dim")

        # A/B cohort indicator (if available)
        if record.reward_mode:
            header.append("  │  ")
            cohort_color = "cyan" if record.reward_mode == "A" else "magenta"
            header.append(f"Cohort {record.reward_mode}", style=cohort_color)

        # Pin status
        if record.pinned:
            header.append("  │  ")
            header.append("📌 PINNED", style="bold cyan")

        # Second line: parameter info
        header.append("\n")

        def _format_params(p: int) -> str:
            if p >= 1_000_000:
                return f"{p / 1_000_000:.1f}M"
            elif p >= 1_000:
                return f"{p / 1_000:.1f}K"
            return str(p)

        header.append(f"Host: {_format_params(record.host_params)}", style="dim")
        header.append("  │  ")
        header.append(f"Fossilized: {record.fossilized_count}", style="green")
        header.append("  │  ")
        header.append(f"Pruned: {record.pruned_count}", style="red")

        return header

    def _render_metrics(self) -> Table:
        """Render environment metrics section with historical data."""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", style="white")

        record = self._record

        # Accuracy history sparkline
        if record.accuracy_history:
            from esper.karn.sanctum.schema import make_sparkline
            acc_spark = make_sparkline(record.accuracy_history, width=40)
            table.add_row("Accuracy History", acc_spark)

        # Reward history sparkline
        if record.reward_history:
            from esper.karn.sanctum.schema import make_sparkline
            rwd_spark = make_sparkline(record.reward_history, width=40)
            table.add_row("Reward History", rwd_spark)

        # Reward components (if captured)
        rc = record.reward_components
        if rc is not None and rc.total is not None and rc.total != 0:
            reward_text = Text()
            reward_text.append(f"Total: {rc.total:+.3f}", style="bold")
            if rc.base_acc_delta is not None and rc.base_acc_delta != 0:
                style = "green" if rc.base_acc_delta > 0 else "red"
                reward_text.append(f"  ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
            if rc.compute_rent is not None and rc.compute_rent != 0:
                reward_text.append(f"  Rent: {rc.compute_rent:.3f}", style="red")
            if rc.alpha_shock is not None and rc.alpha_shock != 0:
                reward_text.append(f"  Shock: {rc.alpha_shock:.3f}", style="red")
            if rc.bounded_attribution is not None and rc.bounded_attribution != 0:
                style = "green" if rc.bounded_attribution > 0 else "red"
                reward_text.append(f"  Attr: {rc.bounded_attribution:+.3f}", style=style)
            table.add_row("Reward Breakdown", reward_text)

        # Recent actions (at time of snapshot)
        if record.action_history:
            recent = " → ".join(list(record.action_history)[-5:])
            table.add_row("Final Actions", recent)

        # Seed composition summary
        if record.seeds:
            seed_summary = Text()
            stages: dict[str, int] = {}
            for seed in record.seeds.values():
                stage = seed.stage if seed else "DORMANT"
                stages[stage] = stages.get(stage, 0) + 1

            for stage, count in sorted(stages.items()):
                color = STAGE_COLORS.get(stage, "dim")
                seed_summary.append(f"{stage}: {count}  ", style=color)
            table.add_row("Seed States", seed_summary)

        # Host loss (if available)
        if record.host_loss and record.host_loss > 0:
            table.add_row("Host Loss", f"{record.host_loss:.4f}")

        return table

    def _render_graveyard(self) -> Panel:
        """Render the seed graveyard showing per-blueprint lifecycle stats.

        Shows how many seeds of each blueprint type were:
        - Spawned (germinated)
        - Fossilized (successfully integrated)
        - Pruned (removed due to poor performance)
        """
        record = self._record

        # Combine all blueprints seen across spawns, fossilized, pruned
        all_blueprints = set(record.blueprint_spawns.keys())
        all_blueprints.update(record.blueprint_fossilized.keys())
        all_blueprints.update(record.blueprint_prunes.keys())

        if not all_blueprints:
            content = Text("No seeds germinated in this episode", style="dim italic")
            return Panel(content, title="Seed Graveyard", border_style="dim")

        # Build graveyard display
        lines = []
        for blueprint in sorted(all_blueprints):
            spawned = record.blueprint_spawns.get(blueprint, 0)
            fossilized = record.blueprint_fossilized.get(blueprint, 0)
            pruned = record.blueprint_prunes.get(blueprint, 0)

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
