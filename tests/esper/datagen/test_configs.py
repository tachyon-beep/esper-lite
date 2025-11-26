"""Tests for data generation config dataclasses."""

import pytest
from esper.datagen.configs import (
    EnvironmentConfig,
    BehaviorPolicyConfig,
    ActionProbabilities,
    RewardComponents,
    StepMetadata,
    ENVIRONMENT_PRESETS,
    POLICY_PRESETS,
)


class TestEnvironmentConfig:
    def test_create_baseline(self):
        config = EnvironmentConfig(
            config_id="baseline",
            architecture="HostCNN",
            learning_rate=0.01,
            batch_size=128,
            optimizer="SGD",
        )
        assert config.config_id == "baseline"
        assert config.momentum == 0.9  # default

    def test_from_preset(self):
        config = EnvironmentConfig.from_preset("resnet34-adam")
        assert config.architecture == "ResNet-34"
        assert config.optimizer == "Adam"

    def test_to_dict_roundtrip(self):
        config = EnvironmentConfig.from_preset("baseline")
        data = config.to_dict()
        restored = EnvironmentConfig.from_dict(data)
        assert config == restored


class TestBehaviorPolicyConfig:
    def test_create_baseline(self):
        config = BehaviorPolicyConfig(policy_id="baseline")
        assert config.min_epochs_before_germinate == 5
        assert config.epsilon == 0.0

    def test_from_preset(self):
        config = BehaviorPolicyConfig.from_preset("aggressive")
        assert config.min_epochs_before_germinate == 3
        assert config.cull_after_epochs_without_improvement == 3

    def test_with_epsilon(self):
        config = BehaviorPolicyConfig.from_preset("baseline").with_epsilon(0.2)
        assert config.epsilon == 0.2
        assert config.policy_id == "baseline-eps0.2"


class TestActionProbabilities:
    def test_compute_behavior_prob_no_epsilon(self):
        greedy_probs = {"WAIT": 0.9, "GERMINATE": 0.05, "ADVANCE": 0.03, "CULL": 0.02}
        result = ActionProbabilities.compute_behavior_prob(greedy_probs, "WAIT", epsilon=0.0)
        assert result == 0.9

    def test_compute_behavior_prob_with_epsilon(self):
        greedy_probs = {"WAIT": 1.0, "GERMINATE": 0.0, "ADVANCE": 0.0, "CULL": 0.0}
        # μ(a|s) = (1-ε) * π_greedy + ε/|A|
        # For GERMINATE with ε=0.2: (1-0.2)*0.0 + 0.2/4 = 0.05
        result = ActionProbabilities.compute_behavior_prob(greedy_probs, "GERMINATE", epsilon=0.2)
        assert abs(result - 0.05) < 1e-6

    def test_create_from_decision(self):
        greedy_probs = {"WAIT": 0.8, "GERMINATE": 0.1, "ADVANCE": 0.05, "CULL": 0.05}
        ap = ActionProbabilities.from_decision(
            greedy_probs=greedy_probs,
            sampled_action="GERMINATE",
            epsilon=0.1,
        )
        assert ap.greedy_action == "WAIT"
        assert ap.sampled_action == "GERMINATE"
        assert ap.was_exploratory == True
        assert ap.behavior_prob > 0


class TestRewardComponents:
    def test_total_no_shaping(self):
        rc = RewardComponents(
            accuracy_delta=0.5,
            loss_delta=-0.1,
            potential_prev=70.0,
            potential_next=70.5,
            intervention_cost=0.0,
        )
        # Default: accuracy_delta * 10 + shaping + cost
        total = rc.total(shaping_weight=0.0)
        assert total == 5.0  # 0.5 * 10

    def test_total_with_shaping(self):
        rc = RewardComponents(
            accuracy_delta=0.5,
            loss_delta=-0.1,
            potential_prev=70.0,
            potential_next=70.5,
            intervention_cost=-0.02,
        )
        # shaping = 0.99 * 70.5 - 70.0 = 69.795 - 70.0 = -0.205
        total = rc.total(gamma=0.99, shaping_weight=1.0)
        expected = 0.5 * 10 + (0.99 * 70.5 - 70.0) - 0.02
        assert abs(total - expected) < 1e-6
