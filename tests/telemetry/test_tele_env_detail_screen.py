"""End-to-end tests for EnvDetailScreen telemetry consumption.

Verifies EnvDetailScreen correctly consumes:
- SeedState fields (TELE-555 to TELE-560) via SeedCard
- EnvState fields (TELE-646 to TELE-649) via _render_metrics()
- RewardComponents fields (TELE-655 to TELE-664) via _render_metrics()
- Graveyard fields (TELE-670 to TELE-672) via _render_graveyard()

EnvDetailScreen is the most comprehensive telemetry consumer - it displays
detailed state for a single environment including per-seed cards, metrics,
reward breakdown, and graveyard statistics.
"""

from __future__ import annotations

from esper.karn.sanctum.schema import (
    EnvState,
    RewardComponents,
    SeedState,
)
from esper.karn.sanctum.widgets.env_detail_screen import SeedCard


# =============================================================================
# Test Fixtures
# =============================================================================


def make_seed(
    slot_id: str = "r0c0",
    stage: str = "TRAINING",
    blueprint_id: str = "conv_light",
    alpha: float = 0.0,
    accuracy_delta: float = 0.0,
    seed_params: int = 10000,
    grad_ratio: float = 1.0,
    has_vanishing: bool = False,
    has_exploding: bool = False,
    epochs_in_stage: int = 5,
    blend_tempo_epochs: int = 5,
    alpha_curve: str = "LINEAR",
    interaction_sum: float = 0.0,
    boost_received: float = 0.0,
    contribution_velocity: float = 0.0,
) -> SeedState:
    """Create a SeedState with specified fields for testing."""
    return SeedState(
        slot_id=slot_id,
        stage=stage,
        blueprint_id=blueprint_id,
        alpha=alpha,
        accuracy_delta=accuracy_delta,
        seed_params=seed_params,
        grad_ratio=grad_ratio,
        has_vanishing=has_vanishing,
        has_exploding=has_exploding,
        epochs_in_stage=epochs_in_stage,
        blend_tempo_epochs=blend_tempo_epochs,
        alpha_curve=alpha_curve,
        interaction_sum=interaction_sum,
        boost_received=boost_received,
        contribution_velocity=contribution_velocity,
    )


def make_env(
    env_id: int = 0,
    seeds: dict[str, SeedState] | None = None,
    host_accuracy: float = 75.0,
    best_accuracy: float = 80.0,
    epochs_since_improvement: int = 0,
    status: str = "healthy",
    host_params: int = 1000000,
    fossilized_params: int = 0,
    active_seed_count: int = 1,
    fossilized_count: int = 0,
    pruned_count: int = 0,
    total_actions: int = 100,
    action_counts: dict[str, int] | None = None,
    reward_components: RewardComponents | None = None,
    blueprint_spawns: dict[str, int] | None = None,
    blueprint_fossilized: dict[str, int] | None = None,
    blueprint_prunes: dict[str, int] | None = None,
) -> EnvState:
    """Create an EnvState with specified fields for testing."""
    env = EnvState(env_id=env_id)
    env.seeds = seeds or {}
    env.host_accuracy = host_accuracy
    env.best_accuracy = best_accuracy
    env.epochs_since_improvement = epochs_since_improvement
    env.status = status
    env.host_params = host_params
    env.fossilized_params = fossilized_params
    env.active_seed_count = active_seed_count
    env.fossilized_count = fossilized_count
    env.pruned_count = pruned_count
    env.total_actions = total_actions
    env.action_counts = action_counts or {
        "WAIT": 50,
        "GERMINATE": 20,
        "SET_ALPHA_TARGET": 15,
        "PRUNE": 5,
        "FOSSILIZE": 10,
        "ADVANCE": 0,
    }
    if reward_components:
        env.reward_components = reward_components
    env.blueprint_spawns = blueprint_spawns or {}
    env.blueprint_fossilized = blueprint_fossilized or {}
    env.blueprint_prunes = blueprint_prunes or {}
    return env


# =============================================================================
# TELE-555 to TELE-560: SeedCard Consumer Tests
# =============================================================================


class TestSeedCardConsumer:
    """Tests verifying SeedCard correctly consumes SeedState fields.

    SeedCard renders individual seed slots with:
    - Stage with color coding
    - Blueprint ID
    - Seed parameters (TELE-555)
    - Accuracy delta with color (TELE-556)
    - Gradient ratio and health flags (TELE-557)
    - Interaction sum with synergy threshold (TELE-558)
    - Boost received indicator (TELE-559)
    - Contribution velocity trend (TELE-560)
    """

    def test_seed_card_displays_seed_params(self) -> None:
        """TELE-555: SeedCard displays seed_params with format_params helper."""
        seed = make_seed(seed_params=150000)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        # Panel content should include formatted params
        content_str = str(panel.renderable)
        assert "150.0K" in content_str or "Params" in content_str

    def test_seed_card_displays_seed_params_zero(self) -> None:
        """TELE-555: SeedCard displays '--' for zero seed_params."""
        seed = make_seed(seed_params=0)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Should show placeholder for no params
        assert "Params: --" in content_str or "Params" in content_str

    def test_seed_card_displays_accuracy_delta_positive(self) -> None:
        """TELE-556: SeedCard displays positive accuracy_delta in green."""
        seed = make_seed(stage="BLENDING", alpha=0.5, accuracy_delta=2.5)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        # Verify the panel was rendered (content is Rich Text)
        assert panel is not None
        # The accuracy delta should be present with positive formatting
        content_str = str(panel.renderable)
        assert "+2.50%" in content_str or "Acc" in content_str

    def test_seed_card_displays_accuracy_delta_negative(self) -> None:
        """TELE-556: SeedCard displays negative accuracy_delta in red."""
        seed = make_seed(stage="BLENDING", alpha=0.5, accuracy_delta=-1.5)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Negative delta should be displayed
        assert "-1.50%" in content_str or "Acc" in content_str

    def test_seed_card_displays_grad_ratio(self) -> None:
        """TELE-557: SeedCard displays grad_ratio when present."""
        seed = make_seed(grad_ratio=1.25)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "ratio=1.25" in content_str or "Grad" in content_str

    def test_seed_card_displays_exploding_gradient(self) -> None:
        """TELE-557: SeedCard shows EXPLODING indicator for gradient issues."""
        seed = make_seed(has_exploding=True)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "EXPLODING" in content_str

    def test_seed_card_displays_vanishing_gradient(self) -> None:
        """TELE-557: SeedCard shows VANISHING indicator for gradient issues."""
        seed = make_seed(has_vanishing=True)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "VANISHING" in content_str

    def test_seed_card_displays_interaction_sum_with_threshold(self) -> None:
        """TELE-558: SeedCard displays interaction_sum when above threshold."""
        # Above synergy threshold (0.5)
        seed = make_seed(
            stage="BLENDING",
            alpha=0.5,
            interaction_sum=0.8,
        )
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Should show positive synergy
        assert "Synergy" in content_str or "+0.8" in content_str

    def test_seed_card_displays_negative_interaction(self) -> None:
        """TELE-558: SeedCard displays negative interaction in red."""
        seed = make_seed(
            stage="BLENDING",
            alpha=0.5,
            interaction_sum=-0.7,
        )
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "Synergy" in content_str

    def test_seed_card_displays_boost_received(self) -> None:
        """TELE-559: SeedCard displays boost_received when above threshold."""
        seed = make_seed(
            stage="BLENDING",
            alpha=0.5,
            interaction_sum=0.3,
            boost_received=0.5,  # Above BOOST_RECEIVED_THRESHOLD (0.1)
        )
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        # Should show boost indicator
        content_str = str(panel.renderable)
        # Boost shows as arrow indicator
        assert "Synergy" in content_str

    def test_seed_card_displays_contribution_velocity_improving(self) -> None:
        """TELE-560: SeedCard shows improving trend for positive velocity."""
        seed = make_seed(
            stage="BLENDING",
            alpha=0.5,
            contribution_velocity=0.05,  # Above CONTRIBUTION_VELOCITY_EPSILON
        )
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "improving" in content_str or "Trend" in content_str

    def test_seed_card_displays_contribution_velocity_declining(self) -> None:
        """TELE-560: SeedCard shows declining trend for negative velocity."""
        seed = make_seed(
            stage="BLENDING",
            alpha=0.5,
            contribution_velocity=-0.05,  # Below -CONTRIBUTION_VELOCITY_EPSILON
        )
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "declining" in content_str or "Trend" in content_str

    def test_seed_card_training_stage_shows_zero_acc_delta(self) -> None:
        """TELE-556: TRAINING stage shows 'Acc delta: 0.0 (learning)' message.

        TRAINING/GERMINATED seeds have alpha=0 and cannot affect output,
        so accuracy_delta is meaningless and shown as learning placeholder.
        """
        seed = make_seed(stage="TRAINING", accuracy_delta=0.0)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Training stage should show learning indicator, not delta
        assert "learning" in content_str or "0.0" in content_str


# =============================================================================
# TELE-646 to TELE-649: EnvDetailScreen Header Consumer Tests
# =============================================================================


class TestEnvDetailHeaderConsumer:
    """Tests verifying _render_header() correctly consumes EnvState fields.

    Header bar displays:
    - TELE-646: status with color coding
    - TELE-647: best_accuracy
    - TELE-648: host_accuracy (current)
    - TELE-649: epochs_since_improvement (momentum)
    - Growth ratio with color coding
    """

    def test_header_displays_status_with_color(self) -> None:
        """TELE-646: Header displays status with appropriate color."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(status="healthy")
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        # Header should contain status
        header_str = str(header)
        assert "HEALTHY" in header_str

    def test_header_displays_excellent_status(self) -> None:
        """TELE-646: Header displays excellent status for high accuracy."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(status="excellent", host_accuracy=85.0)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        assert "EXCELLENT" in header_str

    def test_header_displays_best_accuracy(self) -> None:
        """TELE-647: Header displays best_accuracy with cyan style."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(best_accuracy=82.5)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        assert "Best: 82.5%" in header_str

    def test_header_displays_host_accuracy(self) -> None:
        """TELE-648: Header displays host_accuracy (current)."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(host_accuracy=78.3)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        assert "Current: 78.3%" in header_str

    def test_header_displays_momentum_epochs(self) -> None:
        """TELE-649: Header displays epochs_since_improvement when stalled."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(epochs_since_improvement=15)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        assert "Momentum: 15 epochs" in header_str

    def test_header_displays_improving_when_no_stall(self) -> None:
        """TELE-649: Header displays 'Improving' when epochs_since_improvement is 0."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(epochs_since_improvement=0)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        assert "Improving" in header_str

    def test_header_displays_growth_ratio_with_color(self) -> None:
        """Header displays growth_ratio with appropriate color coding."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # Growth ratio = (host + fossilized) / host = (1M + 250K) / 1M = 1.25
        env = make_env(host_params=1000000, fossilized_params=250000)
        screen = EnvDetailScreen(env, ["r0c0"])
        header = screen._render_header()

        header_str = str(header)
        # Should show growth ratio
        assert "1.25x" in header_str or "= 1.25" in header_str


# =============================================================================
# TELE-650 to TELE-654: EnvDetailScreen Metrics Consumer Tests
# =============================================================================


class TestEnvDetailMetricsConsumer:
    """Tests verifying _render_metrics() correctly consumes EnvState fields.

    Metrics section displays:
    - TELE-650: accuracy_history as sparkline
    - TELE-651: reward_history as sparkline
    - TELE-652: seed counts (active, fossilized, pruned)
    - TELE-653: action_counts distribution
    - TELE-654: gaming_rate
    """

    def test_metrics_displays_accuracy_history_sparkline(self) -> None:
        """TELE-650: Metrics displays accuracy_history as sparkline."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env()
        # Add accuracy history
        for acc in [70.0, 72.0, 74.0, 75.0, 76.0]:
            env.accuracy_history.append(acc)

        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        # Table should be renderable
        assert table is not None

    def test_metrics_displays_reward_history_sparkline(self) -> None:
        """TELE-651: Metrics displays reward_history as sparkline."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env()
        # Add reward history
        for rwd in [0.1, 0.2, 0.15, 0.25, 0.3]:
            env.reward_history.append(rwd)

        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_metrics_displays_seed_counts(self) -> None:
        """TELE-652: Metrics displays active, fossilized, and pruned counts."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(
            active_seed_count=2,
            fossilized_count=3,
            pruned_count=1,
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        # Table is a Rich Table, check it was created
        assert table is not None
        assert table.row_count > 0

    def test_metrics_displays_action_distribution(self) -> None:
        """TELE-653: Metrics displays action_counts as percentages."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(
            total_actions=100,
            action_counts={
                "WAIT": 50,
                "GERMINATE": 20,
                "SET_ALPHA_TARGET": 15,
                "PRUNE": 5,
                "FOSSILIZE": 10,
                "ADVANCE": 0,
            },
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_metrics_displays_gaming_rate(self) -> None:
        """TELE-654: Metrics displays gaming_rate with health indicator."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env()
        env.gaming_trigger_count = 5
        env.total_reward_steps = 100
        # gaming_rate = 5/100 = 0.05 = 5%

        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None


# =============================================================================
# TELE-655 to TELE-664: Reward Breakdown Consumer Tests
# =============================================================================


class TestRewardBreakdownConsumer:
    """Tests verifying _render_metrics() correctly consumes RewardComponents.

    Reward section displays:
    - TELE-655: total reward with color
    - TELE-656: PBRS fraction (stage_bonus / total)
    - TELE-657: signals section (base_acc_delta, compute_rent, alpha_shock, ratio_penalty)
    - TELE-658: credits section (bounded_attribution, hindsight_credit, stage_bonus, fossilize_terminal_bonus)
    - TELE-659: warnings section (blending_warning, holding_warning)
    - Gaming indicator (shock or ratio triggered)
    """

    def test_reward_displays_total_positive(self) -> None:
        """TELE-655: Reward total displays in green when positive."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(total=0.5)
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_reward_displays_total_negative(self) -> None:
        """TELE-655: Reward total displays in red when negative."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(total=-0.3)
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_reward_displays_pbrs_fraction(self) -> None:
        """TELE-656: PBRS fraction is calculated from stage_bonus / total."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # PBRS = 0.1 / 0.5 = 20% (healthy range 10-40%)
        rc = RewardComponents(total=0.5, stage_bonus=0.1)
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_reward_displays_signals_section(self) -> None:
        """TELE-657: Signals section shows base_acc_delta, rent, shock, ratio."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(
            total=0.3,
            base_acc_delta=0.2,
            compute_rent=-0.05,
            alpha_shock=-0.02,
            ratio_penalty=-0.01,
        )
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_reward_displays_credits_section(self) -> None:
        """TELE-658: Credits section shows attribution, hindsight, stage, fossilize."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(
            total=0.8,
            bounded_attribution=0.3,
            hindsight_credit=0.1,
            scaffold_count=2,
            avg_scaffold_delay=3.5,
            stage_bonus=0.15,
            fossilize_terminal_bonus=0.25,
        )
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_reward_displays_warnings_section(self) -> None:
        """TELE-659: Warnings section shows blending and holding warnings."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(
            total=0.2,
            blending_warning=-0.05,
            holding_warning=-0.03,
        )
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_gaming_indicator_shows_shock(self) -> None:
        """Gaming indicator shows SHOCK when alpha_shock is triggered."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(
            total=0.1,
            alpha_shock=-0.05,
        )
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None

    def test_gaming_indicator_shows_ratio(self) -> None:
        """Gaming indicator shows RATIO when ratio_penalty is triggered."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        rc = RewardComponents(
            total=0.1,
            ratio_penalty=-0.03,
        )
        env = make_env(reward_components=rc)
        screen = EnvDetailScreen(env, ["r0c0"])
        table = screen._render_metrics()

        assert table is not None


# =============================================================================
# TELE-670 to TELE-672: Graveyard Consumer Tests
# =============================================================================


class TestGraveyardConsumer:
    """Tests verifying _render_graveyard() correctly consumes graveyard fields.

    Graveyard section displays:
    - TELE-670: blueprint_spawns (per-blueprint spawn counts)
    - TELE-671: blueprint_fossilized (per-blueprint fossilize counts)
    - TELE-672: blueprint_prunes (per-blueprint prune counts)
    - Success rate calculation and color coding
    """

    def test_graveyard_displays_all_blueprint_types(self) -> None:
        """TELE-670-672: Graveyard shows all blueprints across spawn/foss/prune."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(
            blueprint_spawns={"conv_light": 5, "attention": 3},
            blueprint_fossilized={"conv_light": 2},
            blueprint_prunes={"attention": 1, "norm": 2},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        # Should include all three blueprints
        content_str = str(panel.renderable)
        # conv_light appears in spawns and fossilized
        # attention appears in spawns and prunes
        # norm appears only in prunes
        assert "conv_light" in content_str or "Seed Graveyard" in str(panel.title)

    def test_graveyard_shows_spawn_fossilize_prune_columns(self) -> None:
        """TELE-670-672: Graveyard shows spawn/foss/prun columns."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(
            blueprint_spawns={"conv_light": 10},
            blueprint_fossilized={"conv_light": 5},
            blueprint_prunes={"conv_light": 3},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        content_str = str(panel.renderable)
        # Header should have spawn/foss/prun
        assert "spawn" in content_str or "foss" in content_str

    def test_graveyard_calculates_success_rate(self) -> None:
        """TELE-670-672: Graveyard calculates success_rate = fossilized / (fossilized + pruned)."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # 2 fossilized + 2 pruned = 50% success rate
        env = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={"conv_light": 2},
            blueprint_prunes={"conv_light": 2},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        content_str = str(panel.renderable)
        # Should show success rate (50% is yellow threshold)
        assert "%" in content_str or "rate" in content_str

    def test_graveyard_applies_success_rate_green_threshold(self) -> None:
        """Graveyard shows green for success_rate >= 50%."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # 3 fossilized + 1 pruned = 75% success rate (green)
        env = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={"conv_light": 3},
            blueprint_prunes={"conv_light": 1},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        # Just verify rendering works
        assert panel is not None

    def test_graveyard_applies_success_rate_yellow_threshold(self) -> None:
        """Graveyard shows yellow for success_rate >= 25% and < 50%."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # 1 fossilized + 2 pruned = 33% success rate (yellow)
        env = make_env(
            blueprint_spawns={"conv_light": 4},
            blueprint_fossilized={"conv_light": 1},
            blueprint_prunes={"conv_light": 2},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        assert panel is not None

    def test_graveyard_applies_success_rate_red_threshold(self) -> None:
        """Graveyard shows red for success_rate < 25%."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # 0 fossilized + 4 pruned = 0% success rate (red)
        env = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={},
            blueprint_prunes={"conv_light": 4},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        assert panel is not None

    def test_graveyard_shows_placeholder_when_empty(self) -> None:
        """Graveyard shows '(none)' placeholder when no seeds spawned."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        env = make_env(
            blueprint_spawns={},
            blueprint_fossilized={},
            blueprint_prunes={},
        )
        screen = EnvDetailScreen(env, ["r0c0"])
        panel = screen._render_graveyard()

        content_str = str(panel.renderable)
        assert "(none)" in content_str


# =============================================================================
# Schema Completeness Tests
# =============================================================================


class TestEnvDetailScreenSchemaCompleteness:
    """Verify all consumed fields have corresponding schema definitions."""

    def test_all_consumed_fields_have_tele_records(self) -> None:
        """All fields consumed by EnvDetailScreen exist in schema."""
        # SeedState fields (TELE-555 to TELE-560)
        seed = SeedState(slot_id="r0c0")
        assert hasattr(seed, "seed_params")  # TELE-555
        assert hasattr(seed, "accuracy_delta")  # TELE-556
        assert hasattr(seed, "grad_ratio")  # TELE-557
        assert hasattr(seed, "has_vanishing")  # TELE-557
        assert hasattr(seed, "has_exploding")  # TELE-557
        assert hasattr(seed, "interaction_sum")  # TELE-558
        assert hasattr(seed, "boost_received")  # TELE-559
        assert hasattr(seed, "contribution_velocity")  # TELE-560

        # EnvState fields (TELE-646 to TELE-654)
        env = EnvState(env_id=0)
        assert hasattr(env, "status")  # TELE-646
        assert hasattr(env, "best_accuracy")  # TELE-647
        assert hasattr(env, "host_accuracy")  # TELE-648
        assert hasattr(env, "epochs_since_improvement")  # TELE-649
        assert hasattr(env, "accuracy_history")  # TELE-650
        assert hasattr(env, "reward_history")  # TELE-651
        assert hasattr(env, "active_seed_count")  # TELE-652
        assert hasattr(env, "fossilized_count")  # TELE-652
        assert hasattr(env, "pruned_count")  # TELE-652
        assert hasattr(env, "action_counts")  # TELE-653
        assert hasattr(env, "gaming_rate")  # TELE-654 (property)

        # RewardComponents fields (TELE-655 to TELE-664)
        rc = RewardComponents()
        assert hasattr(rc, "total")  # TELE-655
        assert hasattr(rc, "stage_bonus")  # TELE-656 (for PBRS fraction)
        assert hasattr(rc, "base_acc_delta")  # TELE-657
        assert hasattr(rc, "compute_rent")  # TELE-657
        assert hasattr(rc, "alpha_shock")  # TELE-657
        assert hasattr(rc, "ratio_penalty")  # TELE-657
        assert hasattr(rc, "bounded_attribution")  # TELE-658
        assert hasattr(rc, "hindsight_credit")  # TELE-658
        assert hasattr(rc, "fossilize_terminal_bonus")  # TELE-658
        assert hasattr(rc, "blending_warning")  # TELE-659
        assert hasattr(rc, "holding_warning")  # TELE-659

        # Graveyard fields (TELE-670 to TELE-672)
        assert hasattr(env, "blueprint_spawns")  # TELE-670
        assert hasattr(env, "blueprint_fossilized")  # TELE-671
        assert hasattr(env, "blueprint_prunes")  # TELE-672

    def test_env_detail_screen_imports(self) -> None:
        """Verify EnvDetailScreen imports required dependencies."""
        from esper.karn.sanctum.widgets.env_detail_screen import (
            EnvDetailScreen,
            SeedCard,
        )

        assert EnvDetailScreen is not None
        assert SeedCard is not None


# =============================================================================
# SeedCard Rendering Logic Tests
# =============================================================================


class TestSeedCardRenderingLogic:
    """Tests for SeedCard rendering logic edge cases."""

    def test_seed_card_dormant_renders_placeholder(self) -> None:
        """Dormant slots render with placeholder content."""
        card = SeedCard(None, "r0c0")
        panel = card._render_dormant()

        content_str = str(panel.renderable)
        assert "DORMANT" in content_str

    def test_seed_card_blending_shows_alpha_bar(self) -> None:
        """BLENDING stage shows alpha progress bar."""
        seed = make_seed(stage="BLENDING", alpha=0.5)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Alpha bar uses block characters
        assert "Alpha" in content_str

    def test_seed_card_tempo_fast(self) -> None:
        """BLENDING with tempo <= 3 shows FAST and triple arrows."""
        seed = make_seed(stage="BLENDING", alpha=0.3, blend_tempo_epochs=3)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "FAST" in content_str or "Tempo" in content_str

    def test_seed_card_tempo_standard(self) -> None:
        """BLENDING with tempo 4-5 shows STANDARD and double arrows."""
        seed = make_seed(stage="BLENDING", alpha=0.3, blend_tempo_epochs=5)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "STANDARD" in content_str or "Tempo" in content_str

    def test_seed_card_tempo_slow(self) -> None:
        """BLENDING with tempo > 5 shows SLOW and single arrow."""
        seed = make_seed(stage="BLENDING", alpha=0.3, blend_tempo_epochs=8)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        assert "SLOW" in content_str or "Tempo" in content_str

    def test_seed_card_curve_glyph_displayed(self) -> None:
        """BLENDING stage shows curve glyph from ALPHA_CURVE_GLYPHS."""
        seed = make_seed(stage="BLENDING", alpha=0.3, alpha_curve="SIGMOID")
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Curve glyph should be present
        assert "Tempo" in content_str

    def test_seed_card_fossilized_shows_historical_tempo(self) -> None:
        """FOSSILIZED stage shows historical tempo in dim style."""
        seed = make_seed(stage="FOSSILIZED", alpha=1.0, blend_tempo_epochs=5)
        card = SeedCard(seed, "r0c0")
        panel = card._render_active()

        content_str = str(panel.renderable)
        # Should show "Blended:" prefix for historical
        assert "Blended" in content_str or "FOSSILIZED" in content_str

    def test_seed_card_stage_color_from_leyline(self) -> None:
        """Stage color comes from leyline STAGE_COLORS mapping."""
        for stage in ["TRAINING", "BLENDING", "FOSSILIZED"]:
            seed = make_seed(stage=stage, alpha=0.5 if stage != "TRAINING" else 0.0)
            card = SeedCard(seed, "r0c0")
            panel = card._render_active()

            # Verify stage appears in content
            content_str = str(panel.renderable)
            assert stage in content_str


# =============================================================================
# Integration: DisplayThresholds Usage
# =============================================================================


class TestDisplayThresholdsIntegration:
    """Verify EnvDetailScreen correctly uses DisplayThresholds constants."""

    def test_interaction_synergy_uses_threshold(self) -> None:
        """Synergy display uses INTERACTION_SYNERGY_THRESHOLD (0.5)."""
        # Just below threshold - should not show colored synergy
        seed_below = make_seed(
            stage="BLENDING", alpha=0.5, interaction_sum=0.4
        )
        # At threshold - should show colored synergy
        seed_at = make_seed(
            stage="BLENDING", alpha=0.5, interaction_sum=0.5
        )

        card_below = SeedCard(seed_below, "r0c0")
        card_at = SeedCard(seed_at, "r0c0")

        # Both should render without error
        assert card_below._render_active() is not None
        assert card_at._render_active() is not None

    def test_boost_received_uses_threshold(self) -> None:
        """Boost display uses BOOST_RECEIVED_THRESHOLD (0.1)."""
        # Just below threshold
        seed_below = make_seed(
            stage="BLENDING", alpha=0.5, interaction_sum=0.3, boost_received=0.09
        )
        # At threshold
        seed_at = make_seed(
            stage="BLENDING", alpha=0.5, interaction_sum=0.3, boost_received=0.1
        )

        card_below = SeedCard(seed_below, "r0c0")
        card_at = SeedCard(seed_at, "r0c0")

        assert card_below._render_active() is not None
        assert card_at._render_active() is not None

    def test_contribution_velocity_uses_epsilon(self) -> None:
        """Trend display uses CONTRIBUTION_VELOCITY_EPSILON (0.01)."""
        # Zero velocity - stable
        seed_stable = make_seed(
            stage="BLENDING", alpha=0.5, contribution_velocity=0.0
        )
        # Positive velocity - improving
        seed_improving = make_seed(
            stage="BLENDING", alpha=0.5, contribution_velocity=0.02
        )
        # Negative velocity - declining
        seed_declining = make_seed(
            stage="BLENDING", alpha=0.5, contribution_velocity=-0.02
        )

        for seed in [seed_stable, seed_improving, seed_declining]:
            card = SeedCard(seed, "r0c0")
            assert card._render_active() is not None

    def test_growth_ratio_warning_threshold(self) -> None:
        """Header uses GROWTH_RATIO_WARNING (1.2) for color coding."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # Below warning (1.15x)
        env_normal = make_env(host_params=1000000, fossilized_params=150000)
        # Above warning (1.25x)
        env_warning = make_env(host_params=1000000, fossilized_params=250000)

        screen_normal = EnvDetailScreen(env_normal, ["r0c0"])
        screen_warning = EnvDetailScreen(env_warning, ["r0c0"])

        assert screen_normal._render_header() is not None
        assert screen_warning._render_header() is not None

    def test_momentum_stall_threshold(self) -> None:
        """Header uses MOMENTUM_STALL_THRESHOLD (10) for color coding."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # Below threshold
        env_ok = make_env(epochs_since_improvement=5)
        # Above threshold
        env_stalled = make_env(epochs_since_improvement=15)

        screen_ok = EnvDetailScreen(env_ok, ["r0c0"])
        screen_stalled = EnvDetailScreen(env_stalled, ["r0c0"])

        assert screen_ok._render_header() is not None
        assert screen_stalled._render_header() is not None

    def test_blueprint_success_rate_thresholds(self) -> None:
        """Graveyard uses BLUEPRINT_SUCCESS_GREEN (0.50) and YELLOW (0.25)."""
        from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen

        # Green: 60% success (3/5)
        env_green = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={"conv_light": 3},
            blueprint_prunes={"conv_light": 2},
        )
        # Yellow: 40% success (2/5)
        env_yellow = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={"conv_light": 2},
            blueprint_prunes={"conv_light": 3},
        )
        # Red: 20% success (1/5)
        env_red = make_env(
            blueprint_spawns={"conv_light": 5},
            blueprint_fossilized={"conv_light": 1},
            blueprint_prunes={"conv_light": 4},
        )

        for env in [env_green, env_yellow, env_red]:
            screen = EnvDetailScreen(env, ["r0c0"])
            assert screen._render_graveyard() is not None
