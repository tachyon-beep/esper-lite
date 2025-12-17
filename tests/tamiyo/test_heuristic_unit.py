"""Unit tests for HeuristicTamiyo policy."""

import pytest

from esper.leyline import DEFAULT_BLUEPRINT_PENALTY_DECAY, DEFAULT_BLUEPRINT_PENALTY_THRESHOLD
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig


@pytest.mark.tamiyo
class TestConfiguration:
    """Tests for HeuristicPolicyConfig and initialization."""

    def test_config_defaults_applied_when_none(self):
        """Should use default config when None is passed."""
        policy = HeuristicTamiyo(config=None, topology="cnn")

        # Verify default values are applied
        assert policy.config.plateau_epochs_to_germinate == 3
        assert policy.config.min_epochs_before_germinate == 5
        assert policy.config.cull_after_epochs_without_improvement == 5
        assert policy.config.cull_if_accuracy_drops_by == 2.0
        assert policy.config.min_improvement_to_fossilize == 0.5
        assert policy.config.embargo_epochs_after_cull == 5

    def test_custom_config_respected(self):
        """Should use custom config when provided."""
        custom_config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=10,
            min_epochs_before_germinate=20,
            embargo_epochs_after_cull=15,
        )
        policy = HeuristicTamiyo(config=custom_config, topology="cnn")

        assert policy.config.plateau_epochs_to_germinate == 10
        assert policy.config.min_epochs_before_germinate == 20
        assert policy.config.embargo_epochs_after_cull == 15

    def test_topology_affects_actions(self):
        """Different topologies should have different action enums."""
        cnn_policy = HeuristicTamiyo(topology="cnn")
        # P1-B: Transformer needs transformer-compatible blueprints in config
        transformer_config = HeuristicPolicyConfig(
            blueprint_rotation=["attention", "mlp", "norm"]  # Transformer-compatible
        )
        transformer_policy = HeuristicTamiyo(
            topology="transformer", config=transformer_config
        )

        # Both should have action enums but they're different
        assert cnn_policy._action_enum is not None
        assert transformer_policy._action_enum is not None
        # CNN has GERMINATE_CONV_LIGHT, transformer might not
        assert hasattr(cnn_policy._action_enum, "GERMINATE_CONV_LIGHT")


@pytest.mark.tamiyo
class TestBlueprintValidation:
    """Tests for P1-B: Blueprint validation at initialization.

    The heuristic should fail fast if blueprint_rotation contains blueprints
    not available for the specified topology.
    """

    def test_default_config_with_transformer_topology_fails(self):
        """Default config has CNN blueprints which fail for transformer topology.

        Default blueprint_rotation includes conv_light, conv_heavy, depthwise
        which are CNN-only blueprints. Using default config with transformer
        should raise ValueError immediately at init, not during training.
        """
        with pytest.raises(ValueError, match="blueprint_rotation contains blueprints not available"):
            HeuristicTamiyo(topology="transformer")

    def test_invalid_blueprints_rejected_with_clear_error(self):
        """Invalid blueprint names should produce a helpful error message."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "nonexistent_blueprint"]
        )

        with pytest.raises(ValueError) as exc_info:
            HeuristicTamiyo(config=config, topology="cnn")

        error_msg = str(exc_info.value)
        assert "nonexistent_blueprint" in error_msg
        assert "Available:" in error_msg

    def test_valid_cnn_blueprints_accepted(self):
        """Valid CNN blueprints should be accepted without error."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy", "norm"]
        )
        # Should not raise
        policy = HeuristicTamiyo(config=config, topology="cnn")
        assert policy.config.blueprint_rotation == ["conv_light", "conv_heavy", "norm"]

    def test_valid_transformer_blueprints_accepted(self):
        """Valid transformer blueprints should be accepted without error."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["attention", "norm", "lora"]
        )
        # Should not raise
        policy = HeuristicTamiyo(config=config, topology="transformer")
        assert policy.config.blueprint_rotation == ["attention", "norm", "lora"]

    def test_error_lists_available_blueprints(self):
        """Error message should list available blueprints for the topology."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["invalid_blueprint"]
        )

        with pytest.raises(ValueError) as exc_info:
            HeuristicTamiyo(config=config, topology="cnn")

        error_msg = str(exc_info.value)
        # Should mention what IS available
        assert "conv_light" in error_msg or "Available:" in error_msg


@pytest.mark.tamiyo
class TestBlueprintPenaltySystem:
    """Tests for blueprint penalty tracking and application."""

    def test_blueprint_penalty_applied_on_cull(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Should apply penalty to blueprint when seed is culled."""
        from esper.leyline import SeedStage

        config = HeuristicPolicyConfig(
            blueprint_penalty_on_cull=2.0,
            cull_after_epochs_without_improvement=1,
            cull_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Create failing seed
        seed = mock_seed_factory(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-3.0,  # Failing
            blueprint_id="conv_light",
        )
        signals = mock_signals_factory(epoch=10)

        # Should be empty before cull
        assert policy._blueprint_penalties.get("conv_light", 0.0) == 0.0

        # Make decision (should cull)
        decision = policy.decide(signals, [seed])
        assert decision.action.name == "CULL"

        # Penalty should be applied
        assert policy._blueprint_penalties["conv_light"] == 2.0

    def test_blueprint_penalty_decay_per_epoch(self, mock_signals_factory):
        """Penalty decay should happen once per epoch, not per decision."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=DEFAULT_BLUEPRINT_PENALTY_DECAY,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Manually set penalty
        policy._blueprint_penalties["conv_light"] = 10.0
        policy._last_decay_epoch = -1  # Ensure first decay happens

        # First decision at epoch 5
        signals1 = mock_signals_factory(epoch=5, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals1, [])

        # Penalty should decay
        penalty_after_first = policy._blueprint_penalties["conv_light"]
        assert penalty_after_first == pytest.approx(5.0)

        # Second decision at same epoch 5
        signals2 = mock_signals_factory(epoch=5, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals2, [])

        # Penalty should NOT decay again (same epoch)
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(5.0)

        # Third decision at epoch 6
        signals3 = mock_signals_factory(epoch=6, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals3, [])

        # Penalty should decay again
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(2.5)

    def test_blueprint_penalty_threshold_skip(self, mock_signals_factory):
        """Should skip blueprints above penalty threshold."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy", "attention"],
            blueprint_penalty_threshold=DEFAULT_BLUEPRINT_PENALTY_THRESHOLD,
            blueprint_penalty_decay=1.0,  # No decay (intentionally different from default)
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Penalize first blueprint heavily
        policy._blueprint_penalties["conv_light"] = 5.0
        policy._last_decay_epoch = 10  # Prevent decay

        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )

        decision = policy.decide(signals, [])

        # Should skip conv_light and pick conv_heavy
        assert decision.action.name == "GERMINATE_CONV_HEAVY"

    def test_all_penalized_picks_lowest(self, mock_signals_factory):
        """When all blueprints penalized, should pick one with lowest penalty."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy", "attention"],
            blueprint_penalty_threshold=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Penalize all blueprints above threshold
        policy._blueprint_penalties["conv_light"] = 5.0
        policy._blueprint_penalties["conv_heavy"] = 3.0
        policy._blueprint_penalties["attention"] = 10.0
        policy._last_decay_epoch = 10  # Prevent decay

        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )

        decision = policy.decide(signals, [])

        # Should pick conv_heavy (lowest penalty = 3.0)
        assert decision.action.name == "GERMINATE_CONV_HEAVY"

    def test_blueprint_penalties_cleared_on_reset(self):
        """Reset should clear all blueprint penalties."""
        policy = HeuristicTamiyo(topology="cnn")
        policy._blueprint_penalties["conv_light"] = 5.0
        policy._blueprint_penalties["conv_heavy"] = 3.0

        policy.reset()

        assert len(policy._blueprint_penalties) == 0


@pytest.mark.tamiyo
class TestDecisionsProperty:
    """Tests for decisions property."""

    def test_decisions_property_returns_copy(self, mock_signals_factory):
        """Should return a copy, not expose internal list."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = mock_signals_factory(epoch=10, plateau_epochs=0, host_stabilized=0)

        # Make a decision
        policy.decide(signals, [])

        # Get decisions
        decisions1 = policy.decisions
        decisions2 = policy.decisions

        # Should be different objects (copies)
        assert decisions1 is not decisions2

        # But same content
        assert len(decisions1) == len(decisions2)

    def test_decisions_accumulate(self, mock_signals_factory):
        """Should accumulate decisions over multiple calls."""
        policy = HeuristicTamiyo(topology="cnn")

        # Make multiple decisions
        for i in range(5):
            signals = mock_signals_factory(epoch=i, plateau_epochs=0, host_stabilized=0)
            policy.decide(signals, [])

        decisions = policy.decisions
        assert len(decisions) == 5

    def test_decisions_cleared_on_reset(self, mock_signals_factory):
        """Reset should clear decision history."""
        policy = HeuristicTamiyo(topology="cnn")

        # Make decisions
        for i in range(3):
            signals = mock_signals_factory(epoch=i, plateau_epochs=0, host_stabilized=0)
            policy.decide(signals, [])

        assert len(policy.decisions) == 3

        policy.reset()

        assert len(policy.decisions) == 0


@pytest.mark.tamiyo
class TestGerminationCounting:
    """Tests for germination count tracking."""

    def test_germination_count_increments(self, mock_signals_factory):
        """Should increment germination count when germinating."""
        policy = HeuristicTamiyo(topology="cnn")

        assert policy._germination_count == 0

        # Trigger germination
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )
        decision = policy.decide(signals, [])

        # Should germinate
        assert decision.action.name.startswith("GERMINATE_")
        assert policy._germination_count == 1

    def test_germination_count_not_incremented_on_wait(self, mock_signals_factory):
        """Should not increment count when not germinating."""
        policy = HeuristicTamiyo(topology="cnn")

        # Trigger WAIT (not stabilized)
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=0
        )
        decision = policy.decide(signals, [])

        assert decision.action.name == "WAIT"
        assert policy._germination_count == 0

    def test_germination_count_cleared_on_reset(self, mock_signals_factory):
        """Reset should clear germination count."""
        policy = HeuristicTamiyo(topology="cnn")

        # Germinate once
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )
        policy.decide(signals, [])

        assert policy._germination_count == 1

        policy.reset()

        assert policy._germination_count == 0


@pytest.mark.tamiyo
class TestEmbargoMechanism:
    """Tests for embargo after cull."""

    def test_embargo_blocks_germination(self, mock_signals_factory):
        """Should block germination during embargo period."""
        config = HeuristicPolicyConfig(embargo_epochs_after_cull=5)
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._last_cull_epoch = 8

        # Try to germinate at epoch 10 (2 epochs after cull, within embargo)
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )
        decision = policy.decide(signals, [])

        assert decision.action.name == "WAIT"
        assert "Embargo" in decision.reason

    def test_embargo_expires_after_period(self, mock_signals_factory):
        """Should allow germination after embargo expires."""
        config = HeuristicPolicyConfig(embargo_epochs_after_cull=3)
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._last_cull_epoch = 5

        # Try to germinate at epoch 10 (5 epochs after cull, past embargo)
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )
        decision = policy.decide(signals, [])

        # Should germinate (not blocked by embargo)
        assert decision.action.name.startswith("GERMINATE_")

    def test_cull_updates_embargo_epoch(self, mock_signals_factory, mock_seed_factory):
        """Culling a seed should update last_cull_epoch."""
        from esper.leyline import SeedStage

        config = HeuristicPolicyConfig(
            cull_after_epochs_without_improvement=1,
            cull_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        initial_cull_epoch = policy._last_cull_epoch
        assert initial_cull_epoch == -100  # Default

        # Create failing seed
        seed = mock_seed_factory(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-3.0,
        )
        signals = mock_signals_factory(epoch=15)

        decision = policy.decide(signals, [seed])

        # Should cull
        assert decision.action.name == "CULL"
        # Should update last_cull_epoch
        assert policy._last_cull_epoch == 15


@pytest.mark.tamiyo
class TestTerminalSeedFiltering:
    """Tests for filtering terminal/failure seeds."""

    def test_terminal_seeds_ignored(self, mock_signals_factory, mock_seed_factory):
        """Should ignore seeds in terminal stages."""
        from esper.leyline import SeedStage

        policy = HeuristicTamiyo(topology="cnn")

        # Create fossilized seed (terminal)
        terminal_seed = mock_seed_factory(stage=SeedStage.FOSSILIZED)

        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )

        # Should treat as no active seeds and consider germination
        decision = policy.decide(signals, [terminal_seed])

        # Should try to germinate (sees no "live" seeds)
        assert decision.action.name.startswith("GERMINATE_")

    def test_culled_seeds_ignored(self, mock_signals_factory, mock_seed_factory):
        """Should ignore culled seeds."""
        from esper.leyline import SeedStage

        policy = HeuristicTamiyo(topology="cnn")

        # Create culled seed
        culled_seed = mock_seed_factory(stage=SeedStage.CULLED)

        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )

        decision = policy.decide(signals, [culled_seed])

        # Should try to germinate
        assert decision.action.name.startswith("GERMINATE_")


@pytest.mark.tamiyo
class TestBlueprintIndexRotation:
    """Tests for blueprint index management."""

    def test_blueprint_index_increments_on_germination(self, mock_signals_factory):
        """Blueprint index should increment each germination."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy", "attention"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        assert policy._blueprint_index == 0

        # First germination
        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )
        decision1 = policy.decide(signals, [])
        assert decision1.action.name == "GERMINATE_CONV_LIGHT"
        assert policy._blueprint_index == 1

        # Second germination (simulate conditions allow it)
        policy._last_cull_epoch = -100  # Reset embargo
        decision2 = policy.decide(signals, [])
        assert decision2.action.name == "GERMINATE_CONV_HEAVY"
        assert policy._blueprint_index == 2

    def test_blueprint_index_wraps_around(self, mock_signals_factory):
        """Blueprint index should wrap around rotation list."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy"],
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        signals = mock_signals_factory(
            epoch=10, plateau_epochs=5, host_stabilized=1
        )

        # Germinate through all blueprints
        blueprints = []
        for _ in range(4):  # More than rotation length
            policy._last_cull_epoch = -100  # Reset embargo
            decision = policy.decide(signals, [])
            if decision.action.name.startswith("GERMINATE_"):
                blueprint = decision.action.name.replace("GERMINATE_", "").lower()
                blueprints.append(blueprint)

        # Should see pattern: conv_light, conv_heavy, conv_light, conv_heavy
        assert blueprints == ["conv_light", "conv_heavy", "conv_light", "conv_heavy"]

    def test_blueprint_index_reset_to_zero(self):
        """Reset should clear blueprint index."""
        policy = HeuristicTamiyo(topology="cnn")
        policy._blueprint_index = 42

        policy.reset()

        assert policy._blueprint_index == 0


@pytest.mark.tamiyo
class TestResetCompleteness:
    """Tests verifying reset clears all state."""

    def test_reset_clears_all_state_fields(self):
        """Reset should clear all internal state to initial values."""
        policy = HeuristicTamiyo(topology="cnn")

        # Modify all state
        policy._blueprint_index = 10
        policy._germination_count = 5
        policy._last_cull_epoch = 50
        policy._blueprint_penalties = {"conv_light": 3.0, "conv_heavy": 2.0}
        policy._last_decay_epoch = 45
        # Add some fake decisions
        from esper.tamiyo.decisions import TamiyoDecision
        policy._decisions_made.append(
            TamiyoDecision(action=policy._action_enum.WAIT)
        )

        # Reset
        policy.reset()

        # Verify all fields reset
        assert policy._blueprint_index == 0
        assert policy._germination_count == 0
        assert policy._last_cull_epoch == -100
        assert len(policy._blueprint_penalties) == 0
        assert policy._last_decay_epoch == -1
        assert len(policy._decisions_made) == 0
