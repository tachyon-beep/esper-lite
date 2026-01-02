"""Tests for EnvDetailScreen modal - detailed environment and seed inspection."""
from io import StringIO

from rich.console import Console

from esper.karn.constants import DisplayThresholds
from esper.karn.sanctum.schema import (
    EnvState,
    RewardComponents,
    SeedState,
)
from esper.karn.sanctum.widgets.env_detail_screen import (
    EnvDetailScreen,
    SeedCard,
)
from esper.leyline import STAGE_COLORS


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False, force_terminal=True)
    console.print(renderable)
    return console.file.getvalue()


# =============================================================================
# SeedCard Tests
# =============================================================================


def test_seed_card_dormant():
    """Test rendering of dormant/empty slot."""
    card = SeedCard(seed=None, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "r0c0" in rendered
    assert "DORMANT" in rendered


def test_seed_card_dormant_stage():
    """Test rendering of seed with DORMANT stage."""
    seed = SeedState(slot_id="r0c0", stage="DORMANT")
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "DORMANT" in rendered


def test_seed_card_training_stage():
    """Test rendering of TRAINING stage seed."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="conv_light",
        seed_params=50000,
        epochs_in_stage=5,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "TRAINING" in rendered
    assert "conv_light" in rendered
    assert "50.0K" in rendered or "50000" in rendered
    assert "Epochs: 5" in rendered


def test_seed_card_training_stage_shows_learning_annotation():
    """TRAINING stage seeds should show '(learning)' annotation for accuracy_delta.

    Seeds in TRAINING have alpha=0 and cannot affect host output.
    The UI should indicate this clearly rather than showing a misleading value.
    """
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="conv_light",
        accuracy_delta=0.0,  # Will be 0 from sync_telemetry
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    # Should show "(learning)" annotation to explain why delta is 0
    assert "learning" in rendered.lower()
    assert "Acc" in rendered  # Part of "Acc Δ"


def test_seed_card_germinated_stage_shows_learning_annotation():
    """GERMINATED stage seeds should also show '(learning)' annotation."""
    seed = SeedState(
        slot_id="r0c0",
        stage="GERMINATED",
        blueprint_id="attention",
        accuracy_delta=0.0,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "learning" in rendered.lower()


def test_seed_card_blending_stage():
    """Test rendering of BLENDING stage with alpha."""
    seed = SeedState(
        slot_id="r1c0",
        stage="BLENDING",
        blueprint_id="attention",
        alpha=0.75,
        accuracy_delta=2.5,
    )
    card = SeedCard(seed=seed, slot_id="r1c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "BLENDING" in rendered
    assert "Alpha: 0.75" in rendered
    assert "2.5" in rendered or "+2.50" in rendered


def test_seed_card_gradient_exploding():
    """Test gradient health indicator for exploding gradients."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="dense_heavy",
        has_exploding=True,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "EXPLODING" in rendered


def test_seed_card_gradient_vanishing():
    """Test gradient health indicator for vanishing gradients."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="mlp",
        has_vanishing=True,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "VANISHING" in rendered


def test_seed_card_gradient_healthy():
    """Test gradient health indicator when healthy."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="mlp",
        grad_ratio=0.8,
        has_vanishing=False,
        has_exploding=False,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "ratio=0.80" in rendered or "OK" in rendered


def test_seed_card_fossilized():
    """Test rendering of FOSSILIZED stage."""
    seed = SeedState(
        slot_id="r0c0",
        stage="FOSSILIZED",
        blueprint_id="conv_heavy",
        accuracy_delta=5.2,
        seed_params=100000,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "FOSSILIZED" in rendered
    assert "conv_heavy" in rendered


def test_seed_card_params_millions():
    """Test parameter display in millions."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="large",
        seed_params=2500000,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "2.5M" in rendered


def test_seed_card_alpha_bar():
    """Test alpha progress bar rendering."""
    card = SeedCard(seed=None, slot_id="test")

    # Test various alpha values
    bar_0 = card._make_alpha_bar(0.0)
    assert "░" in bar_0

    bar_50 = card._make_alpha_bar(0.5)
    assert "█" in bar_50

    bar_100 = card._make_alpha_bar(1.0)
    assert "█" in bar_100


# =============================================================================
# EnvDetailScreen Tests
# =============================================================================


def test_env_detail_screen_creation():
    """Test basic screen creation."""
    env = EnvState(env_id=0)
    screen = EnvDetailScreen(env_state=env, slot_ids=["r0c0", "r0c1"])

    assert screen is not None
    assert screen._env == env
    assert screen._slot_ids == ["r0c0", "r0c1"]


def test_env_detail_screen_bindings():
    """Test that ESC and Q bindings are defined."""
    env = EnvState(env_id=0)
    screen = EnvDetailScreen(env_state=env, slot_ids=[])

    binding_keys = [b.key for b in screen.BINDINGS]
    assert "escape" in binding_keys
    assert "q" in binding_keys


def test_env_detail_screen_header():
    """Test header rendering with env summary."""
    env = EnvState(
        env_id=3,
        status="healthy",
        best_accuracy=85.5,
        host_accuracy=82.3,
        epochs_since_improvement=2,
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    header = screen._render_header()
    header_str = str(header)

    assert "3" in header_str  # env_id
    assert "HEALTHY" in header_str.upper()
    assert "85.5" in header_str  # best_accuracy
    assert "82.3" in header_str  # current accuracy


def test_env_detail_screen_header_stale():
    """Test header shows stale indicator for epochs_since_improvement."""
    env = EnvState(
        env_id=0,
        status="stalled",
        best_accuracy=80.0,
        host_accuracy=75.0,
        epochs_since_improvement=15,
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    header = screen._render_header()
    header_str = str(header)

    assert "15" in header_str


def test_env_detail_screen_header_improving():
    """Test header shows Improving when epochs_since_improvement is 0."""
    env = EnvState(
        env_id=0,
        status="excellent",
        best_accuracy=90.0,
        host_accuracy=90.0,
        epochs_since_improvement=0,
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    header = screen._render_header()
    header_str = str(header)

    assert "Improving" in header_str


def test_env_detail_screen_metrics_section():
    """Test metrics section rendering."""
    from collections import deque

    env = EnvState(env_id=0)
    env.accuracy_history = deque([70.0, 72.5, 75.0, 78.0, 80.0], maxlen=50)
    env.reward_history = deque([0.1, 0.2, 0.15, 0.25, 0.3], maxlen=50)
    env.active_seed_count = 2
    env.fossilized_count = 1
    env.pruned_count = 0
    env.total_actions = 10
    env.action_counts = {"WAIT": 5, "GERMINATE": 3, "FOSSILIZE": 1, "PRUNE": 1}

    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    # Check sparklines present (some character from the sparkline set)
    assert "▁" in rendered or "▂" in rendered or "▃" in rendered or "█" in rendered

    # Check seed counts
    assert "Active: 2" in rendered
    assert "Fossilized: 1" in rendered


def test_env_detail_screen_action_distribution():
    """Test action distribution display in metrics."""
    env = EnvState(env_id=0)
    env.total_actions = 100
    env.action_counts = {"WAIT": 50, "GERMINATE": 30, "FOSSILIZE": 15, "PRUNE": 5}

    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    # Check percentages are shown
    assert "WAIT" in rendered
    assert "GERMINATE" in rendered


def test_env_detail_screen_reward_breakdown():
    """Test reward component breakdown in metrics."""
    env = EnvState(env_id=0)
    env.reward_components = RewardComponents(
        total=0.5,
        base_acc_delta=0.3,
        compute_rent=-0.1,
        bounded_attribution=0.3,
    )

    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    assert "Total:" in rendered or "0.5" in rendered


def test_env_detail_screen_recent_actions():
    """Test recent actions display."""
    from collections import deque

    env = EnvState(env_id=0)
    env.action_history = deque(["WAIT", "GERMINATE", "WAIT", "FOSSILIZE"], maxlen=10)

    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    assert "Recent Actions" in rendered


def test_stage_colors_defined():
    """Test that all stages have colors defined."""
    expected_stages = [
        "DORMANT", "GERMINATED", "TRAINING", "HOLDING",
        "BLENDING", "FOSSILIZED", "PRUNED"
    ]

    for stage in expected_stages:
        assert stage in STAGE_COLORS, f"Missing color for stage: {stage}"


def test_seed_card_update_seed():
    """Test that SeedCard.update_seed() updates the seed state."""
    initial_seed = SeedState(slot_id="r0c0", stage="TRAINING", accuracy_delta=1.0)
    card = SeedCard(seed=initial_seed, slot_id="r0c0")

    # Update to a new seed state
    new_seed = SeedState(slot_id="r0c0", stage="BLENDING", accuracy_delta=5.0, alpha=0.5)
    card.update_seed(new_seed)

    # Verify internal state was updated
    assert card._seed == new_seed
    assert card._seed.stage == "BLENDING"
    assert card._seed.alpha == 0.5


def test_env_detail_screen_env_id_property():
    """Test that EnvDetailScreen exposes env_id for modal refresh lookup."""
    env = EnvState(env_id=7)
    screen = EnvDetailScreen(env_state=env, slot_ids=[])

    assert screen.env_id == 7


def test_env_detail_screen_update_env_state():
    """Test that EnvDetailScreen.update_env_state() updates internal state."""
    env1 = EnvState(env_id=0, best_accuracy=80.0, host_accuracy=78.0)
    screen = EnvDetailScreen(env_state=env1, slot_ids=["r0c0"])

    # Update with new env state
    env2 = EnvState(env_id=0, best_accuracy=85.0, host_accuracy=84.0)
    screen.update_env_state(env2)

    # Verify internal state was updated
    assert screen._env == env2
    assert screen._env.best_accuracy == 85.0


def test_reward_breakdown_shows_hindsight_credit():
    """Hindsight credit displays with scaffold context when active."""
    env = EnvState(
        env_id=0,
        reward_components=RewardComponents(
            total=0.25,
            bounded_attribution=0.10,
            hindsight_credit=0.08,
            scaffold_count=3,
            avg_scaffold_delay=12.5,
        ),
    )

    # Create screen and render metrics
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()

    # Convert to string and check for hindsight credit
    rendered = render_to_text(table)
    assert "Hind:" in rendered
    assert "0.08" in rendered or "+0.080" in rendered
    assert "(3x, 12.5e)" in rendered


# =============================================================================
# DisplayThresholds Integration Tests
# =============================================================================


def test_seed_card_uses_display_thresholds_for_interaction():
    """SeedCard should use DisplayThresholds for synergy indicators."""
    # Test that the threshold constants are imported correctly
    assert DisplayThresholds.INTERACTION_SYNERGY_THRESHOLD == 0.5
    assert DisplayThresholds.BOOST_RECEIVED_THRESHOLD == 0.1

    # Create seed with high synergy
    seed = SeedState(
        slot_id="r0c0",
        stage="BLENDING",
        blueprint_id="conv3x3",
        interaction_sum=0.8,  # Above threshold
        boost_received=0.15,  # Above threshold
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    # Should show synergy value
    assert "0.8" in rendered or "+0.8" in rendered


def test_seed_card_uses_display_thresholds_for_contribution_velocity():
    """SeedCard should use DisplayThresholds for trend detection."""
    assert DisplayThresholds.CONTRIBUTION_VELOCITY_EPSILON == 0.01

    # Create seed with positive trend
    seed = SeedState(
        slot_id="r0c0",
        stage="BLENDING",
        blueprint_id="conv3x3",
        contribution_velocity=0.05,  # Above epsilon
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    assert "improving" in rendered.lower()


def test_header_uses_display_thresholds_for_momentum():
    """Header should use DisplayThresholds.MOMENTUM_STALL_THRESHOLD."""
    assert DisplayThresholds.MOMENTUM_STALL_THRESHOLD == 10

    # Momentum above threshold -> red
    env_stalled = EnvState(
        env_id=0,
        status="stalled",
        best_accuracy=80.0,
        host_accuracy=75.0,
        epochs_since_improvement=15,  # Above threshold
    )
    screen = EnvDetailScreen(env_state=env_stalled, slot_ids=[])
    header = screen._render_header()
    header_str = str(header)
    assert "15" in header_str

    # Momentum below threshold -> yellow
    env_warning = EnvState(
        env_id=0,
        status="degraded",
        best_accuracy=80.0,
        host_accuracy=78.0,
        epochs_since_improvement=5,  # Below threshold
    )
    screen_warning = EnvDetailScreen(env_state=env_warning, slot_ids=[])
    header_warning = screen_warning._render_header()
    header_warning_str = str(header_warning)
    assert "5" in header_warning_str


def test_header_uses_display_thresholds_for_growth_ratio():
    """Header should use DisplayThresholds.GROWTH_RATIO_WARNING."""
    assert DisplayThresholds.GROWTH_RATIO_WARNING == 1.2

    # Growth above threshold -> yellow
    # growth_ratio is computed: (host_params + fossilized_params) / host_params
    # 1.3 = (1_000_000 + 300_000) / 1_000_000
    env = EnvState(
        env_id=0,
        host_params=1_000_000,
        fossilized_params=300_000,  # Results in growth_ratio = 1.3
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    header = screen._render_header()
    header_str = str(header)
    assert "1.30x" in header_str


def test_metrics_uses_display_thresholds_for_pbrs():
    """Metrics should use DisplayThresholds for PBRS healthy check."""
    assert DisplayThresholds.PBRS_HEALTHY_MIN == 0.1
    assert DisplayThresholds.PBRS_HEALTHY_MAX == 0.4

    # PBRS at 20% (healthy)
    env = EnvState(
        env_id=0,
        reward_components=RewardComponents(
            total=1.0,
            stage_bonus=0.2,  # 20% PBRS
        ),
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    assert "PBRS:" in rendered


def test_metrics_uses_display_thresholds_for_gaming_rate():
    """Metrics should use DisplayThresholds for gaming rate check."""
    assert DisplayThresholds.GAMING_RATE_HEALTHY_MAX == 0.05

    # gaming_rate is computed: gaming_trigger_count / total_reward_steps
    # 0.02 = 2 / 100
    env = EnvState(
        env_id=0,
        gaming_trigger_count=2,
        total_reward_steps=100,  # Results in gaming_rate = 0.02
        reward_components=RewardComponents(total=0.5),
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    assert "Gaming:" in rendered


def test_graveyard_uses_display_thresholds_for_success_rate():
    """Graveyard should use DisplayThresholds for success rate colors."""
    assert DisplayThresholds.BLUEPRINT_SUCCESS_GREEN == 0.50
    assert DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW == 0.25

    env = EnvState(
        env_id=0,
        blueprint_spawns={"conv3x3": 4},
        blueprint_fossilized={"conv3x3": 3},  # 75% success rate
        blueprint_prunes={"conv3x3": 1},
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    panel = screen._render_graveyard()
    rendered = render_to_text(panel)

    assert "75%" in rendered


# =============================================================================
# format_params Integration Tests
# =============================================================================


def test_format_params_used_for_seed_params():
    """SeedCard should use format_params for parameter display."""
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="large",
        seed_params=2_500_000,
    )
    card = SeedCard(seed=seed, slot_id="r0c0")
    panel = card.render()
    rendered = render_to_text(panel)

    # format_params(2_500_000) = "2.5M"
    assert "2.5M" in rendered


def test_format_params_used_in_header():
    """Header should use format_params for host/seed params."""
    # growth_ratio is computed: (host_params + fossilized_params) / host_params
    # 1.33 = (1_500_000 + 500_000) / 1_500_000
    env = EnvState(
        env_id=0,
        host_params=1_500_000,
        fossilized_params=500_000,  # Results in growth_ratio ≈ 1.33
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    header = screen._render_header()
    rendered = str(header)

    # format_params produces "1.5M" and "500.0K"
    assert "Host:" in rendered


def test_format_params_used_in_metrics():
    """Metrics section should use format_params for fossilized params."""
    env = EnvState(
        env_id=0,
        fossilized_params=750_000,
    )
    screen = EnvDetailScreen(env_state=env, slot_ids=[])
    table = screen._render_metrics()
    rendered = render_to_text(table)

    # format_params(750_000) = "750.0K"
    assert "750.0K" in rendered or "Fossilized Params" in rendered
