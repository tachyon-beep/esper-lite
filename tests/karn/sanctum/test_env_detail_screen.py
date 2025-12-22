"""Tests for EnvDetailScreen modal - detailed environment and seed inspection."""
from io import StringIO

import pytest
from rich.console import Console

from esper.karn.sanctum.schema import (
    EnvState,
    RewardComponents,
    SeedState,
)
from esper.karn.sanctum.widgets.env_detail_screen import (
    EnvDetailScreen,
    SeedCard,
    STAGE_COLORS,
)


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
