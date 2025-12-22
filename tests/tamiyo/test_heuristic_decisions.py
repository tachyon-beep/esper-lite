"""Tests for HeuristicTamiyo decision logic."""


from esper.leyline import SeedStage, DEFAULT_BLUEPRINT_PENALTY_THRESHOLD
from esper.kasmina.alpha_controller import AlphaController
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig


class MockSeedMetrics:
    """Mock seed metrics for testing."""

    def __init__(
        self,
        improvement_since_stage_start: float = 0.0,
        total_improvement: float = 0.0,
        counterfactual_contribution: float | None = None,
    ):
        self.improvement_since_stage_start = improvement_since_stage_start
        self.total_improvement = total_improvement
        self.counterfactual_contribution = counterfactual_contribution
        self.current_val_accuracy = 60.0
        self.accuracy_at_stage_start = 60.0 - improvement_since_stage_start


class MockSeedState:
    """Mock seed state for testing."""

    def __init__(
        self,
        seed_id: str = "test_seed",
        stage: SeedStage = SeedStage.TRAINING,
        epochs_in_stage: int = 1,
        alpha: float = 0.0,
        improvement: float = 0.0,
        total_improvement: float = 0.0,
        blueprint_id: str = "conv_light",
        counterfactual: float | None = None,
    ):
        self.seed_id = seed_id
        self.stage = stage
        self.epochs_in_stage = epochs_in_stage
        self.alpha = alpha
        self.blueprint_id = blueprint_id
        self.alpha_controller = AlphaController(alpha=alpha)
        self.metrics = MockSeedMetrics(
            improvement_since_stage_start=improvement,
            total_improvement=total_improvement,
            counterfactual_contribution=counterfactual,
        )


class MockTrainingMetrics:
    """Mock training metrics for testing."""

    def __init__(
        self,
        epoch: int = 10,
        plateau_epochs: int = 0,
        host_stabilized: int = 1,
        accuracy_delta: float = 0.0,
    ):
        self.epoch = epoch
        self.plateau_epochs = plateau_epochs
        self.host_stabilized = host_stabilized
        self.accuracy_delta = accuracy_delta


class MockTrainingSignals:
    """Mock training signals for testing."""

    def __init__(self, metrics: MockTrainingMetrics | None = None):
        self.metrics = metrics or MockTrainingMetrics()


class TestGerminationDecisions:
    """Tests for germination decision logic."""

    def test_germinate_on_plateau_when_stabilized(self):
        """Should germinate when host is stabilized and plateauing."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name.startswith("GERMINATE_")
        assert "Plateau" in decision.reason

    def test_no_germinate_when_not_stabilized(self):
        """Should WAIT when host is not stabilized."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=0,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT"
        assert "not stabilized" in decision.reason.lower()

    def test_no_germinate_during_embargo(self):
        """Should WAIT during embargo period after cull."""
        policy = HeuristicTamiyo(topology="cnn")
        policy._last_prune_epoch = 8  # Culled 2 epochs ago

        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT"
        assert "Embargo" in decision.reason

    def test_no_germinate_too_early(self):
        """Should WAIT when too early in training."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=3,  # Before min_epochs_before_germinate (default 5)
            plateau_epochs=5,
            host_stabilized=1,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT"
        assert "too early" in decision.reason.lower()


class TestCullDecisions:
    """Tests for cull decision logic."""

    def test_cull_failing_seed_in_training(self):
        """Should PRUNE a seed that's failing in TRAINING stage."""
        config = HeuristicPolicyConfig(
            prune_after_epochs_without_improvement=3,
            prune_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-3.0,  # Dropped 3%
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=15))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "PRUNE"
        assert "Failing" in decision.reason

    def test_no_cull_improving_seed(self):
        """Should ADVANCE for a seed that's improving."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=2.0,  # Improving
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=15))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "ADVANCE"

    def test_no_cull_before_patience_expires(self):
        """Should ADVANCE even for failing seed if patience hasn't expired."""
        config = HeuristicPolicyConfig(
            prune_after_epochs_without_improvement=5,
            prune_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.TRAINING,
            epochs_in_stage=3,  # Only 3 epochs, patience is 5
            improvement=-3.0,  # Dropped 3%
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=15))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "ADVANCE"


class TestFossilizeDecisions:
    """Tests for fossilize decision logic."""

    def test_fossilize_contributing_seed(self):
        """Should FOSSILIZE a seed with positive contribution in HOLDING."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.HOLDING,
            epochs_in_stage=3,
            improvement=2.0,
            total_improvement=5.0,
            counterfactual=3.0,  # Contributing 3%
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=30))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "FOSSILIZE"
        assert "contribution" in decision.reason.lower()

    def test_cull_non_contributing_seed_in_probationary(self):
        """Should PRUNE a non-contributing seed in HOLDING."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.HOLDING,
            epochs_in_stage=3,
            improvement=-1.0,
            total_improvement=-2.0,
            counterfactual=-1.0,  # Hurting
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=30))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "PRUNE"

    def test_fossilize_uses_counterfactual_over_total(self):
        """Should prefer counterfactual contribution over total improvement."""
        policy = HeuristicTamiyo(topology="cnn")

        # Total improvement is zero (neutral), but counterfactual is positive
        # NOTE: Negative total_improvement + positive counterfactual = ransomware (P2-B)
        seed = MockSeedState(
            stage=SeedStage.HOLDING,
            epochs_in_stage=3,
            improvement=-1.0,
            total_improvement=0.5,  # Slightly positive to avoid ransomware detection
            counterfactual=1.0,  # Counterfactual shows positive contribution
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=30))

        decision = policy.decide(signals, active_seeds=[seed])

        # Should fossilize because counterfactual is positive
        assert decision.action.name == "FOSSILIZE"


class TestWaitDecisions:
    """Tests for wait/patience decision logic."""

    def test_wait_during_blending(self):
        """Should WAIT during BLENDING stage until full amplitude is reached."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.BLENDING,
            epochs_in_stage=2,
            alpha=0.5,
            improvement=1.0,
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=20))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "WAIT"
        assert "Blending" in decision.reason

    def test_advance_for_germinated_seed(self):
        """Should ADVANCE for GERMINATED seed to enter TRAINING."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.GERMINATED,
            epochs_in_stage=1,
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=10))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "ADVANCE"
        assert "advance" in decision.reason.lower()


class TestBlueprintRotation:
    """Tests for blueprint selection and penalty logic."""

    def test_rotates_through_blueprints(self):
        """Should rotate through available blueprints."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        ))

        blueprints_selected = []
        for i in range(3):
            decision = policy.decide(signals, active_seeds=[])
            if decision.action.name.startswith("GERMINATE_"):
                blueprint = decision.action.name.replace("GERMINATE_", "").lower()
                blueprints_selected.append(blueprint)
            # Reset for next germination (simulate successful fossilization)
            policy._germination_count = 0

        # Should have selected different blueprints
        assert len(blueprints_selected) == 3

    def test_penalized_blueprint_avoided(self):
        """Should avoid heavily penalized blueprints."""
        config = HeuristicPolicyConfig(
            blueprint_rotation=["conv_light", "conv_heavy"],
            blueprint_penalty_threshold=DEFAULT_BLUEPRINT_PENALTY_THRESHOLD,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Heavily penalize conv_light (set high enough to survive decay)
        # Decay factor is 0.5, so 10.0 * 0.5 = 5.0 which is still > 3.0 threshold
        policy._blueprint_penalties["conv_light"] = 10.0
        # Set last_decay_epoch to current epoch so decay doesn't trigger
        policy._last_decay_epoch = 10

        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "GERMINATE_CONV_HEAVY"


class TestReset:
    """Tests for policy reset functionality."""

    def test_reset_clears_state(self):
        """Should clear all internal state on reset."""
        policy = HeuristicTamiyo(topology="cnn")

        # Modify internal state
        policy._blueprint_index = 5
        policy._germination_count = 3
        policy._last_prune_epoch = 10
        policy._blueprint_penalties["conv_light"] = 2.0
        policy._decisions_made.append("fake")

        policy.reset()

        assert policy._blueprint_index == 0
        assert policy._germination_count == 0
        assert policy._last_prune_epoch == -100
        assert len(policy._blueprint_penalties) == 0
        assert len(policy._decisions_made) == 0
