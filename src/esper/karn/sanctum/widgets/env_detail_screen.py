"""EnvDetailScreen - Full-screen modal for detailed environment and seed inspection.

Displays comprehensive diagnostics for a single environment including:
- Per-seed cards with stage, blueprint, alpha, gradient health
- Environment metrics: accuracy history, action distribution, reward breakdown
- Triggered by 'D' key or Enter on DataTable row
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from collections.abc import Iterable

from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Static

from esper.karn.constants import DisplayThresholds
from esper.karn.sanctum.formatting import format_params
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel
from esper.karn.sanctum.widgets.shapley_panel import ShapleyPanel
from esper.leyline import ALPHA_CURVE_GLYPHS, STAGE_COLORS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EnvState, SeedState


# Stage border styles for CSS classes (different from leyline STAGE_COLORS - these map to Textual CSS)
STAGE_CSS_CLASSES = {
    "DORMANT": "dormant",
    "GERMINATED": "training",  # Use training style for germinated
    "TRAINING": "training",
    "HOLDING": "blending",  # Use blending style
    "BLENDING": "blending",
    "FOSSILIZED": "fossilized",
    "PRUNED": "pruned",
    "EMBARGOED": "embargoed",
    "RESETTING": "resetting",
}


class SeedCard(Static):
    """Individual seed card showing detailed seed state."""

    def __init__(self, seed: "SeedState | None", slot_id: str, **kwargs: Any) -> None:
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

    def compose(self) -> Iterable[Widget]:
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
            height=12,
        )

    def _render_active(self) -> Panel:
        """Render active seed with all metrics."""
        seed = self._seed
        if seed is None:
            return self._render_dormant()

        stage_color = STAGE_COLORS.get(seed.stage, "white")

        lines = []

        # Stage with color
        lines.append(Text(f"{seed.stage}", style=f"bold {stage_color}"))

        # Blueprint
        blueprint = seed.blueprint_id or "unknown"
        lines.append(Text(f"Blueprint: {blueprint}", style="white"))

        # Parameters (always visible)
        if seed.seed_params and seed.seed_params > 0:
            lines.append(Text(f"Params: {format_params(seed.seed_params)}", style="dim"))
        else:
            lines.append(Text("Params: --", style="dim"))

        # Alpha (always visible - shows progress bar when blending, placeholder otherwise)
        if (seed.alpha and seed.alpha > 0) or seed.stage in ("BLENDING", "HOLDING"):
            alpha_bar = self._make_alpha_bar(seed.alpha)
            lines.append(Text(f"Alpha: {seed.alpha:.2f} {alpha_bar}"))
        else:
            lines.append(Text("Alpha: --", style="dim"))

        # Blend tempo + curve (always visible - UX policy: data points don't disappear)
        # Curve glyph: shown for BLENDING/HOLDING/FOSSILIZED, dim "-" otherwise
        if seed.stage in ("BLENDING", "HOLDING") and seed.blend_tempo_epochs is not None:
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            curve_glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "−")
            lines.append(Text(f"Tempo: {tempo_arrows} {tempo_name} ({tempo} epochs) {curve_glyph}"))
        elif seed.stage == "FOSSILIZED" and seed.blend_tempo_epochs is not None:
            # Historical - show what was used, dimmed (but curve still visible for curiosity)
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = "▸▸▸" if tempo <= 3 else ("▸▸" if tempo <= 5 else "▸")
            curve_glyph = ALPHA_CURVE_GLYPHS.get(seed.alpha_curve, "−")
            lines.append(Text(f"Blended: {tempo_arrows} {tempo_name} {curve_glyph}", style="dim"))
        else:
            # Not yet blending - show placeholder with dim "-"
            lines.append(Text("Tempo: -- −", style="dim"))

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

        # Inter-slot interaction metrics (always visible, greyed out when not applicable)
        interaction_text = Text("Synergy: ")
        if seed.stage in ("BLENDING", "HOLDING") and (
            seed.interaction_sum != 0
            or seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD
        ):
            if seed.interaction_sum > DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD:
                interaction_text.append(f"+{seed.interaction_sum:.1f}", style="green")
            elif seed.interaction_sum < -DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD:
                interaction_text.append(f"{seed.interaction_sum:.1f}", style="red")
            else:
                interaction_text.append(f"{seed.interaction_sum:.1f}", style="dim")
            # Add boost indicator if significant
            if seed.boost_received > DisplayThresholds.BOOST_RECEIVED_THRESHOLD:
                interaction_text.append(f" (↗{seed.boost_received:.1f})", style="cyan")
        else:
            interaction_text.append("--", style="dim")
        lines.append(interaction_text)

        # Contribution velocity (trend indicator - always visible)
        trend_text = Text("Trend: ")
        if seed.contribution_velocity > DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON:
            trend_text.append("↗ improving", style="green")
        elif seed.contribution_velocity < -DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON:
            trend_text.append("↘ declining", style="yellow")
        else:
            trend_text.append("--", style="dim")
        lines.append(trend_text)

        # Combine into panel content
        content = Text("\n").join(lines)

        return Panel(
            content,
            title=f"[{stage_color}]{self._slot_id}[/{stage_color}]",
            border_style=stage_color,
            height=12,  # Increased height to accommodate new metrics
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
        min-height: 12;
    }

    EnvDetailScreen .metrics-graveyard-row {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }

    EnvDetailScreen .metrics-section {
        width: 1fr;
        height: auto;
        padding-right: 1;
    }

    EnvDetailScreen .graveyard-section {
        width: 1fr;
        height: auto;
        padding-left: 1;
        border-left: solid $primary-lighten-2;
    }

    EnvDetailScreen .attribution-section {
        height: auto;
        margin-top: 1;
        border-top: solid $primary-lighten-2;
        padding-top: 1;
    }

    EnvDetailScreen .attribution-section CounterfactualPanel {
        width: 2fr;
    }

    EnvDetailScreen .attribution-section ShapleyPanel {
        width: 1fr;
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
        **kwargs: Any,
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

    def compose(self) -> Iterable[Widget]:
        """Compose the modal layout."""
        with Container(id="modal-container"):
            # Header bar
            yield Static(self._render_header(), id="detail-header", classes="header-bar")

            # Seed grid
            with Horizontal(classes="seed-grid"):
                for slot_id in self._slot_ids:
                    seed = self._env.seeds.get(slot_id)
                    yield SeedCard(seed, slot_id, id=f"seed-card-{slot_id}")

            # Metrics + Graveyard side by side
            with Horizontal(classes="metrics-graveyard-row"):
                with Vertical(classes="metrics-section"):
                    yield Static(self._render_metrics(), id="detail-metrics")
                with Vertical(classes="graveyard-section"):
                    yield Static(self._render_graveyard(), id="seed-graveyard")

            # Attribution section: Counterfactual + Shapley side by side
            with Horizontal(classes="attribution-section"):
                yield CounterfactualPanel(
                    self._env.counterfactual_matrix,
                    seeds=self._env.seeds,
                    id="counterfactual-panel",
                )
                yield ShapleyPanel(
                    self._env.shapley_snapshot,
                    seeds=self._env.seeds,
                    id="shapley-panel",
                )

            # Footer hint
            yield Static(
                "[dim]Press ESC, Q, or click to close[/dim]",
                classes="footer-hint",
            )

    def on_click(self) -> None:
        """Dismiss modal on click."""
        self.dismiss()

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
        except NoMatches:
            pass  # Widget may not be mounted yet

        # Update metrics
        try:
            metrics = self.query_one("#detail-metrics", Static)
            metrics.update(self._render_metrics())
        except NoMatches:
            pass  # Widget may not be mounted yet

        # Update counterfactual panel
        try:
            cf_panel = self.query_one("#counterfactual-panel", CounterfactualPanel)
            cf_panel.update_matrix(env_state.counterfactual_matrix, seeds=env_state.seeds)
        except NoMatches:
            pass  # Widget may not be mounted yet

        # Update Shapley panel
        try:
            shapley_panel = self.query_one("#shapley-panel", ShapleyPanel)
            shapley_panel.update_snapshot(env_state.shapley_snapshot, seeds=env_state.seeds)
        except NoMatches:
            pass  # Widget may not be mounted yet

        # Update each seed card
        for slot_id in self._slot_ids:
            try:
                seed = self._env.seeds.get(slot_id)
                card = self.query_one(f"#seed-card-{slot_id}", SeedCard)
                card.update_seed(seed)
            except NoMatches:
                pass  # Widget may not be mounted yet

        # Update graveyard
        try:
            graveyard = self.query_one("#seed-graveyard", Static)
            graveyard.update(self._render_graveyard())
        except NoMatches:
            pass  # Widget may not be mounted yet

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

        # Epochs since improvement (momentum)
        momentum_epochs = env.epochs_since_improvement or 0
        if momentum_epochs > 0:
            momentum_style = (
                "red" if momentum_epochs > DisplayThresholds.MOMENTUM_STALL_THRESHOLD else "yellow"
            )
            header.append(
                f"Momentum: {momentum_epochs} epochs",
                style=momentum_style,
            )
        else:
            header.append("Improving", style="green")

        # Host params, seed params, and growth ratio
        header.append("  │  ")

        host_str = format_params(env.host_params or 0)
        fossilized = env.fossilized_params or 0
        seed_str = format_params(fossilized)
        growth = env.growth_ratio or 1.0

        header.append(f"Host: {host_str}", style="dim")
        header.append(f"  +Seed: {seed_str}", style="green" if fossilized > 0 else "dim")

        # Growth ratio with color coding
        # 1.0x = no growth (dim), 1.0-1.2x = normal (green), >1.2x = significant (yellow)
        if growth > DisplayThresholds.GROWTH_RATIO_WARNING:
            growth_style = "yellow"
        elif growth > 1.0:
            growth_style = "green"
        else:
            growth_style = "dim"
        header.append(f"  = {growth:.2f}x", style=growth_style)

        return header

    def _render_metrics(self) -> Table:
        """Render environment metrics section.

        All rows are always visible to prevent jarring layout shifts.
        Empty/zero values display as dim "--" placeholders.
        """
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", style="white")

        env = self._env
        rc = env.reward_components
        dim_placeholder = Text("--", style="dim")

        # Accuracy history sparkline
        from esper.karn.sanctum.schema import make_sparkline
        acc_spark = make_sparkline(env.accuracy_history, width=40)
        table.add_row("Accuracy History", acc_spark if acc_spark else dim_placeholder)

        # Reward history sparkline
        rwd_spark = make_sparkline(env.reward_history, width=40)
        table.add_row("Reward History", rwd_spark if rwd_spark else dim_placeholder)

        # Seed counts (always has structure)
        seed_counts = Text()
        seed_counts.append(f"Active: {env.active_seed_count}", style="cyan")
        seed_counts.append("  ")
        seed_counts.append(f"Fossilized: {env.fossilized_count}", style="green")
        seed_counts.append("  ")
        seed_counts.append(f"Pruned: {env.pruned_count}", style="red")
        table.add_row("Seed Counts", seed_counts)

        # Fossilized params (always visible)
        foss_params = env.fossilized_params or 0
        if foss_params > 0:
            table.add_row("Fossilized Params", format_params(foss_params))
        else:
            table.add_row("Fossilized Params", dim_placeholder)

        # Action distribution (always visible)
        total_actions = env.total_actions or 0
        if total_actions > 0:
            action_text = Text()
            action_colors = {
                "WAIT": "dim",
                "GERMINATE": "cyan",
                "SET_ALPHA_TARGET": "yellow",
                "FOSSILIZE": "green",
                "PRUNE": "red",
            }
            for action, count in sorted(env.action_counts.items()):
                pct = (count / total_actions) * 100
                color = action_colors.get(action, "white")
                action_text.append(f"{action}: {pct:.0f}%", style=color)
                action_text.append("  ")
            table.add_row("Action Distribution", action_text)
        else:
            table.add_row("Action Distribution", dim_placeholder)

        # Reward Total with PBRS fraction (always visible)
        reward_text = Text()
        if rc.total != 0:
            total_style = "bold green" if rc.total >= 0 else "bold red"
            reward_text.append(f"{rc.total:+.3f}", style=total_style)
            # Add PBRS fraction
            pbrs_fraction = abs(rc.stage_bonus) / abs(rc.total) if rc.total != 0 else 0.0
            pbrs_healthy = (
                DisplayThresholds.PBRS_HEALTHY_MIN
                <= pbrs_fraction
                <= DisplayThresholds.PBRS_HEALTHY_MAX
            )
            pbrs_icon = "✓" if pbrs_healthy else "⚠" if pbrs_fraction > 0 else ""
            pbrs_style = "green" if pbrs_healthy else "yellow"
            reward_text.append(f"  PBRS: {pbrs_fraction:.0%} ", style="dim")
            if pbrs_icon:
                reward_text.append(pbrs_icon, style=pbrs_style)
        else:
            reward_text.append("0.000", style="dim")
            reward_text.append("  PBRS: --", style="dim")
        table.add_row("Reward Total", reward_text)

        # Signals with gaming rate (always visible)
        signals = Text()
        has_signals = False

        if rc.base_acc_delta != 0:
            style = "green" if rc.base_acc_delta > 0 else "red"
            signals.append(f"ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
            has_signals = True
        if rc.compute_rent != 0:
            if has_signals:
                signals.append("  ")
            signals.append(f"Rent: {rc.compute_rent:.3f}", style="red")
            has_signals = True
        if rc.alpha_shock != 0:
            if has_signals:
                signals.append("  ")
            signals.append(f"Shock: {rc.alpha_shock:.3f}", style="red")
            has_signals = True
        if rc.ratio_penalty != 0:
            if has_signals:
                signals.append("  ")
            signals.append(f"Ratio: {rc.ratio_penalty:.3f}", style="red")
            has_signals = True

        # Add gaming rate
        if has_signals:
            signals.append("  │  ")
        gaming_rate = env.gaming_rate
        gaming_active = rc.ratio_penalty != 0 or rc.alpha_shock != 0
        gaming_healthy = gaming_rate < DisplayThresholds.GAMING_RATE_HEALTHY_MAX
        if gaming_active:
            gaming_state = "SHOCK" if rc.alpha_shock != 0 else "RATIO"
            signals.append(f"Gaming: {gaming_rate:.1%} ({gaming_state})", style="red")
        else:
            gaming_style = "green" if gaming_healthy else "yellow"
            signals.append(f"Gaming: {gaming_rate:.1%} (CLEAN)", style=gaming_style)

        table.add_row("  Signals", signals)

        # Credits (always visible)
        credits = Text()
        has_credits = False

        if rc.bounded_attribution != 0:
            style = "green" if rc.bounded_attribution > 0 else "red"
            label = "EscΔ" if env.reward_mode == "escrow" else "Attr"
            credits.append(f"{label}: {rc.bounded_attribution:+.3f}", style=style)
            has_credits = True
        if env.reward_mode == "escrow":
            if rc.stable_val_acc is not None:
                if has_credits:
                    credits.append("  ")
                credits.append(f"StAcc: {rc.stable_val_acc:.1f}%", style="cyan")
                has_credits = True

            if has_credits:
                credits.append("  ")
            credits.append(
                f"Esc: {rc.escrow_credit_prev:.2f}→{rc.escrow_credit_next:.2f} (tgt {rc.escrow_credit_target:.2f})",
                style="cyan",
            )
            has_credits = True

            if rc.escrow_forfeit != 0:
                if has_credits:
                    credits.append("  ")
                style = "green" if rc.escrow_forfeit > 0 else "red"
                credits.append(f"Forf: {rc.escrow_forfeit:+.3f}", style=style)
                has_credits = True
        if rc.hindsight_credit != 0:
            hind_str = f"Hind: {rc.hindsight_credit:+.3f}"
            if rc.scaffold_count > 0:
                hind_str += f" ({rc.scaffold_count}x, {rc.avg_scaffold_delay:.1f}e)"
            if has_credits:
                credits.append("  ")
            credits.append(hind_str, style="blue")
            has_credits = True
        if rc.stage_bonus != 0:
            if has_credits:
                credits.append("  ")
            credits.append(f"Stage: {rc.stage_bonus:+.3f}", style="blue")
            has_credits = True
        if rc.fossilize_terminal_bonus != 0:
            if has_credits:
                credits.append("  ")
            credits.append(f"Foss: {rc.fossilize_terminal_bonus:+.3f}", style="blue")
            has_credits = True

        table.add_row("  Credits", credits if has_credits else dim_placeholder)

        # Warnings (always visible)
        warnings = Text()
        has_warnings = False

        if rc.blending_warning < 0:
            warnings.append(f"Blend: {rc.blending_warning:.3f}", style="yellow")
            has_warnings = True
        if rc.holding_warning < 0:
            if has_warnings:
                warnings.append("  ")
            warnings.append(f"Hold: {rc.holding_warning:.3f}", style="yellow")
            has_warnings = True

        table.add_row("  Warnings", warnings if has_warnings else dim_placeholder)

        # Recent actions (always visible)
        if env.action_history:
            recent = " → ".join(list(env.action_history)[-5:])
            table.add_row("Recent Actions", recent)
        else:
            table.add_row("Recent Actions", dim_placeholder)

        return table

    def _render_graveyard(self) -> Panel:
        """Render the seed graveyard showing per-blueprint lifecycle stats.

        All rows are always visible to prevent jarring layout shifts.
        Shows how many seeds of each blueprint type have been:
        - Spawned (germinated)
        - Fossilized (successfully integrated)
        - Pruned (removed due to poor performance)
        """
        env = self._env
        lines = []

        # Header row (always visible)
        header = Text()
        header.append("Blueprint       ", style="dim")
        header.append("  spawn", style="dim")
        header.append("  foss", style="dim")
        header.append("  prun", style="dim")
        header.append("  rate", style="dim")
        lines.append(header)

        # Combine all blueprints seen across spawns, fossilized, pruned
        all_blueprints = set(env.blueprint_spawns.keys())
        all_blueprints.update(env.blueprint_fossilized.keys())
        all_blueprints.update(env.blueprint_prunes.keys())

        if not all_blueprints:
            # Placeholder row when no seeds spawned (matches column structure)
            placeholder = Text()
            placeholder.append("(none)          ", style="dim italic")
            placeholder.append("     --", style="dim")
            placeholder.append("    --", style="dim")
            placeholder.append("    --", style="dim")
            placeholder.append("    --", style="dim")
            lines.append(placeholder)
        else:
            # Build graveyard display (no prefixes - header has column labels)
            for blueprint in sorted(all_blueprints):
                spawned = env.blueprint_spawns.get(blueprint, 0)
                fossilized = env.blueprint_fossilized.get(blueprint, 0)
                pruned = env.blueprint_prunes.get(blueprint, 0)

                line = Text()
                line.append(f"{blueprint:15s} ", style="white")
                line.append(f"    {spawned:2d}", style="cyan")
                line.append(f"    {fossilized:2d}", style="green")
                line.append(f"    {pruned:2d}", style="red")

                # Calculate success rate if any have terminated
                terminated = fossilized + pruned
                if terminated > 0:
                    success_rate = fossilized / terminated
                    if success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_GREEN:
                        rate_style = "green"
                    elif success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW:
                        rate_style = "yellow"
                    else:
                        rate_style = "red"
                    line.append(f"  {success_rate * 100:3.0f}%", style=rate_style)
                else:
                    line.append("    --", style="dim")

                lines.append(line)

        content = Text("\n").join(lines)
        return Panel(content, title="Seed Graveyard", border_style="dim")
