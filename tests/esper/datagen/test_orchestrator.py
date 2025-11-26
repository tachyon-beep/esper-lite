"""Tests for generation orchestrator."""

import pytest
from esper.datagen.orchestrator import (
    GenerationOrchestrator,
    GenerationPlan,
    create_generation_matrix,
)
from esper.datagen.configs import ENVIRONMENT_PRESETS, POLICY_PRESETS


class TestGenerationMatrix:
    def test_creates_all_combinations(self):
        matrix = create_generation_matrix(
            env_ids=["baseline", "fast-lr"],
            policy_ids=["baseline", "aggressive"],
            episodes_per_combo=5,
        )
        # 2 envs × 2 policies × 5 episodes = 20
        assert len(matrix) == 20

    def test_default_matrix_size(self):
        matrix = create_generation_matrix(episodes_per_combo=1)
        # 13 envs × 11 policies × 1 = 143
        assert len(matrix) == 143

    def test_adds_epsilon_variants(self):
        matrix = create_generation_matrix(
            env_ids=["baseline"],
            policy_ids=["baseline"],
            episodes_per_combo=1,
            epsilon_values=[0.0, 0.1, 0.2],
        )
        # 1 env × 1 policy × 3 epsilons = 3
        assert len(matrix) == 3


class TestGenerationPlan:
    def test_plan_creation(self):
        plan = GenerationPlan(
            env_config_id="baseline",
            policy_config_id="aggressive",
            epsilon=0.1,
            episode_idx=0,
        )
        assert plan.episode_id == "baseline_aggressive-eps0.1_0000"

    def test_plan_to_dict(self):
        plan = GenerationPlan(
            env_config_id="resnet34",
            policy_config_id="random",
            epsilon=1.0,
            episode_idx=5,
        )
        d = plan.to_dict()
        assert d["env_config_id"] == "resnet34"
        assert d["policy_config_id"] == "random"


class TestOrchestrator:
    def test_orchestrator_init(self, tmp_path):
        orch = GenerationOrchestrator(
            output_dir=tmp_path / "test_gen",
            episodes_per_combo=2,
        )
        assert orch.total_planned > 0

    def test_get_next_batch(self, tmp_path):
        orch = GenerationOrchestrator(
            output_dir=tmp_path / "test_gen",
            episodes_per_combo=1,
            env_ids=["baseline"],
            policy_ids=["baseline", "random"],
        )
        batch = orch.get_next_batch(batch_size=2)
        assert len(batch) == 2

    def test_marks_complete(self, tmp_path):
        orch = GenerationOrchestrator(
            output_dir=tmp_path / "test_gen",
            episodes_per_combo=1,
            env_ids=["baseline"],
            policy_ids=["baseline"],
        )
        batch = orch.get_next_batch(batch_size=1)
        assert len(batch) == 1

        orch.mark_complete(batch[0].episode_id)
        assert orch.completed_count == 1

        # Next batch should be empty
        batch = orch.get_next_batch(batch_size=1)
        assert len(batch) == 0
