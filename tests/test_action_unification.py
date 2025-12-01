"""Integration tests for action unification and code review remediation.

Verifies that:
1. Tamiyo and Simic use the same Action enum from leyline
2. Feature dimensions are consistent across PPO and IQL
3. Telemetry enforcement works correctly
4. Blueprint mappings are centralized
"""

import pytest

from esper.leyline.actions import build_action_enum, get_blueprint_from_action, is_germinate_action
from esper.kasmina.blueprints import BlueprintRegistry

ACTION_ENUM = build_action_enum("cnn")


class TestActionUnification:
    """Tests for unified action space."""

    def test_action_importable_from_leyline(self):
        """Action enum builds from registry."""
        assert ACTION_ENUM is not None
        expected = len(BlueprintRegistry.list_for_topology("cnn")) + 3  # WAIT + ADVANCE + CULL
        assert len(ACTION_ENUM) == expected

    def test_simicaction_is_alias(self):
        """SimicAction should be an alias for Action."""
        from esper.leyline import Action, SimicAction
        assert SimicAction is Action

    def test_tamiyo_decision_uses_action(self):
        """TamiyoDecision.action should be an Action enum."""
        from esper.leyline import Action
        from esper.tamiyo import TamiyoDecision

        decision = TamiyoDecision(action=Action.WAIT)
        assert isinstance(decision.action, Action)

    def test_heuristic_tamiyo_returns_action(self):
        """HeuristicTamiyo.decide() should return decision with Action."""
        from esper.leyline.actions import build_action_enum
        from esper.leyline import TrainingSignals
        from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig

        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.plateau_epochs = 5

        decision = tamiyo.decide(signals, active_seeds=[])

        ActionEnum = build_action_enum("cnn")
        assert isinstance(decision.action, ActionEnum)

    def test_no_tamiyoaction_exists(self):
        """TamiyoAction should not exist anymore."""
        import esper.tamiyo as tamiyo_module
        assert not hasattr(tamiyo_module, 'TamiyoAction')

    def test_action_from_decision_returns_action(self):
        """action_from_decision should return ActionTaken with Action."""
        from esper.leyline.actions import build_action_enum
        from esper.tamiyo import TamiyoDecision
        from esper.simic.episodes import action_from_decision

        ActionEnum = build_action_enum("cnn")
        germinate = getattr(ActionEnum, "GERMINATE_CONV_ENHANCE", None) or getattr(ActionEnum, "GERMINATE_CONV", None)
        decision = TamiyoDecision(action=germinate)
        action_taken = action_from_decision(decision)

        assert action_taken.action == decision.action

    def test_stage_constants_match_leyline(self):
        """Stage constants in rewards should match leyline."""
        from esper.leyline import SeedStage
        from esper.simic.rewards import STAGE_TRAINING, STAGE_BLENDING, STAGE_FOSSILIZED

        assert STAGE_TRAINING == SeedStage.TRAINING.value
        assert STAGE_BLENDING == SeedStage.BLENDING.value
        assert STAGE_FOSSILIZED == SeedStage.FOSSILIZED.value


class TestBlueprintCentralization:
    """Tests for centralized blueprint mappings."""

    def test_blueprint_mappings_dynamic(self):
        """Dynamic action enum aligns with registry."""
        from esper.leyline.actions import build_action_enum, get_blueprint_from_action, is_germinate_action
        from esper.kasmina.blueprints import BlueprintRegistry

        ActionEnum = build_action_enum("cnn")
        specs = BlueprintRegistry.list_for_topology("cnn")
        for spec in specs:
            action = getattr(ActionEnum, f"GERMINATE_{spec.name.upper()}")
            assert is_germinate_action(action)
            assert get_blueprint_from_action(action) == spec.name


class TestFeatureDimensionConsistency:
    """Tests for consistent feature dimensions across modules."""

    def test_ppo_and_comparison_dimensions_match(self):
        """PPO and comparison should produce same dimensions."""
        # This is tested in test_simic_networks.py::test_ppo_features_match_comparison_dimensions
        # We just verify the dimensions are what we expect
        from esper.leyline import SeedTelemetry

        # Base features + seed telemetry = 37 dims
        expected_dim = 27 + SeedTelemetry.feature_dim()
        assert expected_dim == 37

    def test_seed_telemetry_dimension(self):
        """SeedTelemetry should be exactly 10 dims."""
        from esper.leyline import SeedTelemetry

        assert SeedTelemetry.feature_dim() == 10

        telem = SeedTelemetry(seed_id="test")
        features = telem.to_features()
        assert len(features) == 10

    def test_ppo_signals_to_features_with_telemetry(self):
        """PPO signals_to_features should return 37-dim with telemetry."""
        from esper.simic.ppo import signals_to_features
        from esper.tamiyo import SignalTracker
        from esper.tolaria import create_model

        model = create_model("cpu")
        tracker = SignalTracker()
        signals = tracker.update(
            epoch=1, global_step=100, train_loss=1.0, train_accuracy=50.0,
            val_loss=1.0, val_accuracy=50.0, active_seeds=[], available_slots=1
        )

        features = signals_to_features(signals, model, tracker, use_telemetry=True)
        assert len(features) == 37

    def test_ppo_signals_to_features_without_telemetry(self):
        """PPO signals_to_features should return 27-dim without telemetry."""
        from esper.simic.ppo import signals_to_features
        from esper.tamiyo import SignalTracker
        from esper.tolaria import create_model

        model = create_model("cpu")
        tracker = SignalTracker()
        signals = tracker.update(
            epoch=1, global_step=100, train_loss=1.0, train_accuracy=50.0,
            val_loss=1.0, val_accuracy=50.0, active_seeds=[], available_slots=1
        )

        features = signals_to_features(signals, model, tracker, use_telemetry=False)
        assert len(features) == 27


class TestTelemetryEnforcement:
    """Tests for telemetry requirement enforcement."""

    def test_snapshot_to_features_enforces_telemetry(self):
        """snapshot_to_features should require telemetry when seed is active."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
            has_active_seed=True,  # Seed active but no telemetry provided
            seed_stage=2, seed_epochs_in_stage=3, seed_alpha=0.0,
            seed_improvement=5.0, available_slots=0
        )

        # Should raise ValueError, not warn
        with pytest.raises(ValueError, match="seed_telemetry is required"):
            snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)

    def test_snapshot_to_features_allows_none_when_no_seed(self):
        """When use_telemetry=True but no seed is active, None telemetry is OK."""
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot

        snapshot = TrainingSnapshot(
            epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
            loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
            accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
            best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
            has_active_seed=False,  # No seed active
            available_slots=1
        )

        # Should NOT raise when no seed is active
        features = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)
        assert len(features) == 37  # 27 base + 10 zero telemetry


class TestBackwardsCompatibility:
    """Tests for backward compatibility aliases."""

    def test_simicaction_alias_works(self):
        """SimicAction alias should work for backward compatibility."""
        from esper.leyline import SimicAction

        # Should be able to use SimicAction
        action = SimicAction.WAIT
        assert action is not None

        # Should have all the same members
        assert hasattr(SimicAction, 'WAIT')
        assert hasattr(SimicAction, 'GERMINATE_CONV')
        assert hasattr(SimicAction, 'ADVANCE')
        assert hasattr(SimicAction, 'CULL')

    def test_action_exportable_from_simic(self):
        """Action should be exportable from simic module."""
        from esper.simic import Action

        assert Action is not None
        assert len(Action) == 7


class TestEndToEndIntegration:
    """End-to-end integration tests across all modules."""

    def test_tamiyo_decision_to_action_taken_pipeline(self):
        """Test complete pipeline from TamiyoDecision to ActionTaken."""
        from esper.tamiyo import TamiyoDecision
        from esper.simic.episodes import action_from_decision
        from esper.kasmina.blueprints import BlueprintRegistry

        blueprint_id = BlueprintRegistry.list_for_topology("cnn")[0].name
        action = getattr(ACTION_ENUM, f"GERMINATE_{blueprint_id.upper()}")

        decision = TamiyoDecision(
            action=action,
            target_seed_id="seed_123",
            reason="Plateau detected",
            confidence=0.85
        )

        # Convert to ActionTaken
        action_taken = action_from_decision(decision)

        assert action_taken.action == action
        assert action_taken.blueprint_id == blueprint_id
        assert action_taken.target_seed_id == "seed_123"
        assert action_taken.confidence == 0.85
        assert action_taken.reason == "Plateau detected"

    def test_blueprint_to_action_round_trip(self):
        """Test blueprint_id to action and back."""
        for spec in BlueprintRegistry.list_for_topology("cnn"):
            action = getattr(ACTION_ENUM, f"GERMINATE_{spec.name.upper()}")
            recovered_blueprint = get_blueprint_from_action(action)
            assert recovered_blueprint == spec.name

    def test_feature_dimension_consistency_across_all_modes(self):
        """Verify feature dimensions are consistent across PPO, IQL, and comparison."""
        from esper.simic.ppo import signals_to_features
        from esper.simic.comparison import snapshot_to_features
        from esper.simic.episodes import TrainingSnapshot
        from esper.tamiyo import SignalTracker
        from esper.tolaria import create_model
        from esper.leyline import SeedTelemetry

        # Setup common test data
        model = create_model("cpu")
        tracker = SignalTracker()
        signals = tracker.update(
            epoch=5, global_step=500, train_loss=0.5, train_accuracy=75.0,
            val_loss=0.6, val_accuracy=72.0, active_seeds=[], available_slots=1
        )

        snapshot = TrainingSnapshot(
            epoch=5, global_step=500, train_loss=0.5, val_loss=0.6,
            loss_delta=-0.1, train_accuracy=75.0, val_accuracy=72.0,
            accuracy_delta=2.0, plateau_epochs=0, best_val_accuracy=72.0,
            best_val_loss=0.6, loss_history_5=(0.9, 0.8, 0.7, 0.6, 0.6),
            accuracy_history_5=(60.0, 65.0, 70.0, 72.0, 72.0),
            has_active_seed=False, seed_stage=0, seed_epochs_in_stage=0,
            seed_alpha=0.0, seed_improvement=0.0, available_slots=1
        )

        zero_telemetry = SeedTelemetry(seed_id="test")

        # Get features from all modes with telemetry enabled
        ppo_features = signals_to_features(signals, model, tracker, use_telemetry=True)
        comparison_features = snapshot_to_features(
            snapshot, use_telemetry=True, seed_telemetry=zero_telemetry
        )

        # All should be 37-dim
        assert len(ppo_features) == 37, f"PPO: expected 37, got {len(ppo_features)}"
        assert len(comparison_features) == 37, f"Comparison: expected 37, got {len(comparison_features)}"

        # Without telemetry, all should be 27-dim
        ppo_features_no_telem = signals_to_features(signals, model, tracker, use_telemetry=False)
        comparison_features_no_telem = snapshot_to_features(snapshot, use_telemetry=False)

        assert len(ppo_features_no_telem) == 27
        assert len(comparison_features_no_telem) == 27

    def test_heuristic_tamiyo_uses_centralized_blueprints(self):
        """HeuristicTamiyo should use centralized blueprint mappings."""
        from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig
        from esper.leyline import TrainingSignals

        config = HeuristicPolicyConfig(
            min_epochs_before_germinate=2,
            plateau_epochs_to_germinate=3,
        )
        tamiyo = HeuristicTamiyo(config)

        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.plateau_epochs = 5  # Trigger germination

        decision = tamiyo.decide(signals, active_seeds=[])

        # Should be a germinate action
        assert is_germinate_action(decision.action)

        # Blueprint ID should match centralized mapping
        blueprint_id = decision.blueprint_id
        spec_names = {s.name for s in BlueprintRegistry.list_for_topology("cnn")}
        assert blueprint_id in spec_names
        assert get_blueprint_from_action(decision.action) == blueprint_id
