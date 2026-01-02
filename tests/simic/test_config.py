"""Tests for TrainingConfig dataclass."""

import inspect

import pytest

from esper.simic.rewards import RewardFamily, RewardMode
from esper.simic.training import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized


class TestTrainingConfigDefaults:
    """Tests for TrainingConfig default values."""

    def test_default_config_creates_successfully(self):
        """Default TrainingConfig should initialize without errors."""
        config = TrainingConfig()
        assert config.lstm_hidden_dim == 512  # V3: DEFAULT_LSTM_HIDDEN_DIM updated to 512
        assert config.chunk_length == config.max_epochs  # Auto-matched
        assert config.reward_family.value == "contribution"
        assert config.reward_mode.value == "shaped"

    def test_default_gamma_gae_lambda(self):
        """Default gamma should be optimized for 25-epoch episodes."""
        config = TrainingConfig()
        assert config.gamma == 0.995

    def test_gae_lambda_configurable(self):
        """Verify GAE lambda can be configured for scaffolding."""
        config = TrainingConfig(gae_lambda=0.98)
        assert config.gae_lambda == 0.98

    def test_gae_lambda_default(self):
        """Verify default GAE lambda is suitable for scaffolding (0.98)."""
        config = TrainingConfig()
        # DRL expert recommendation: 0.98 for longer credit horizon
        assert config.gae_lambda == 0.98

    def test_chunk_length_auto_matches_max_epochs(self):
        """chunk_length=None should auto-match to max_epochs."""
        config = TrainingConfig(max_epochs=50)
        assert config.chunk_length == 50

    def test_chunk_length_explicit_value_preserved(self):
        """Explicit chunk_length should not be overridden."""
        config = TrainingConfig(max_epochs=25, chunk_length=25)
        assert config.chunk_length == 25


class TestTrainingConfigPresets:
    """Tests for TrainingConfig preset helpers."""

    def test_cifar10_stable_preset_is_conservative(self):
        """Stable preset should slow policy updates and anneal exploration."""
        config = TrainingConfig.for_cifar10_stable()
        assert config.n_episodes == 200
        assert config.lr == 1e-4
        assert config.clip_ratio == 0.1
        assert config.entropy_coef == 0.06
        assert config.entropy_coef_start == 0.06
        assert config.entropy_coef_end == 0.03
        assert config.entropy_anneal_episodes == 200


class TestTrainingConfigConversion:
    """Tests for TrainingConfig to kwargs conversion methods."""

    def test_lstm_config_to_ppo_kwargs(self):
        """LSTM params should flow to PPOAgent kwargs."""
        config = TrainingConfig(
            lstm_hidden_dim=256,
            max_epochs=25,  # chunk_length must match max_epochs
            chunk_length=25,
            entropy_anneal_episodes=8,
            n_envs=4,
            ppo_updates_per_batch=2,
        )
        kwargs = config.to_ppo_kwargs()
        assert kwargs["lstm_hidden_dim"] == 256
        assert kwargs["chunk_length"] == 25
        # ceil(episodes / n_envs) * updates_per_batch
        assert kwargs["entropy_anneal_steps"] == 4

    def test_to_ppo_kwargs_includes_vectorized_params(self):
        """to_ppo_kwargs should include vectorized training parameters."""
        config = TrainingConfig(entropy_anneal_episodes=8, n_envs=4, ppo_updates_per_batch=2)
        ppo_kwargs = config.to_ppo_kwargs()

        assert ppo_kwargs.get("num_envs") == config.n_envs
        assert ppo_kwargs.get("max_steps_per_env") == config.max_epochs
        assert ppo_kwargs.get("gamma") == 0.995
        assert ppo_kwargs.get("entropy_anneal_steps") == 4

    def test_lstm_config_to_train_kwargs(self):
        """LSTM params should flow to train_ppo_vectorized kwargs."""
        config = TrainingConfig()
        kwargs = config.to_train_kwargs()
        assert "lstm_hidden_dim" in kwargs
        assert "chunk_length" in kwargs
        assert kwargs["reward_family"] == "contribution"
        assert kwargs["reward_mode"] == "shaped"

    def test_to_train_kwargs_is_subset_of_vectorized_signature(self):
        """Config → train kwargs must stay in sync with train_ppo_vectorized signature."""
        signature = inspect.signature(train_ppo_vectorized)
        allowed = set(signature.parameters)

        config = TrainingConfig()
        kwargs = set(config.to_train_kwargs())

        assert kwargs <= allowed


class TestTrainingConfigSummary:
    """Tests for TrainingConfig summary display."""

    def test_summary_includes_lstm_info(self):
        """Summary should show LSTM info."""
        config = TrainingConfig(lstm_hidden_dim=256, max_epochs=25, chunk_length=25)
        summary = config.summary()
        assert "LSTM" in summary
        assert "hidden=256" in summary
        assert "chunk=25" in summary


class TestTrainingConfigSerialization:
    """Tests for TrainingConfig serialization."""

    def test_from_dict_rejects_unknown_keys(self):
        """Unknown keys should hard-fail to avoid drift."""
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({"lr": 1e-4, "unknown": 1})

    def test_to_dict_roundtrip_preserves_enums(self):
        """Enum fields should serialize to values and back."""
        original = TrainingConfig(
            reward_family=RewardFamily.LOSS,
            reward_mode=RewardMode.SHAPED,
            slots=["r0c0", "r0c2"],
        )
        loaded = TrainingConfig.from_dict(original.to_dict())
        assert loaded.reward_family.value == "loss"
        assert loaded.slots == ["r0c0", "r0c2"]

    def test_from_dict_coerces_strings_to_enums(self):
        """from_dict should coerce string values to enums for external data."""
        config = TrainingConfig.from_dict({
            "reward_family": "loss",
            "reward_mode": "shaped",
        })
        assert config.reward_family == RewardFamily.LOSS
        assert config.reward_mode == RewardMode.SHAPED

    def test_direct_construction_requires_enum_values(self):
        """Direct TrainingConfig construction should not silently coerce strings.

        This prevents bug-hiding: if you pass the wrong type, fail loudly.
        Use from_dict() for external/JSON data that needs coercion.
        """
        # This will fail validation because strings don't have .value attribute
        with pytest.raises((ValueError, AttributeError)):
            config = TrainingConfig(reward_family="loss")
            # _validate() checks reward_family.value, which fails on strings
            _ = config.to_dict()  # Force access to trigger the error


def test_entropy_anneal_rounds_alias():
    """entropy_anneal_rounds should be an alias for entropy_anneal_episodes."""
    # Old name still works
    config = TrainingConfig(entropy_anneal_episodes=50)
    assert config.entropy_anneal_episodes == 50

    # New name also works (via from_dict for JSON compat)
    config2 = TrainingConfig.from_dict({"entropy_anneal_rounds": 75})
    assert config2.entropy_anneal_episodes == 75


def test_entropy_anneal_alias_conflict_rejected():
    """Specifying both entropy_anneal_rounds and entropy_anneal_episodes should fail."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        TrainingConfig.from_dict({
            "entropy_anneal_rounds": 50,
            "entropy_anneal_episodes": 100,
        })


# =============================================================================
# Prevention tests for code review issues (2025-01-01)
# =============================================================================

class TestDocstringImportPath:
    """Tests to prevent docstring import path rot."""

    def test_documented_import_path_works(self):
        """Ensure the import path in the docstring is valid.

        The docstring in config.py says:
            from esper.simic.training import TrainingConfig

        This test verifies that path actually works.
        """
        # This import should succeed (same as what the test file uses)
        from esper.simic.training import TrainingConfig as TC
        assert TC is TrainingConfig


class TestTaskFieldInToTrainKwargs:
    """Tests for task field inclusion in to_train_kwargs()."""

    def test_task_included_when_set(self):
        """task field should be in to_train_kwargs() when explicitly set."""
        config = TrainingConfig(task="tinystories")
        kwargs = config.to_train_kwargs()
        assert "task" in kwargs
        assert kwargs["task"] == "tinystories"

    def test_task_excluded_when_none(self):
        """task should NOT be in kwargs when not set (use function default)."""
        config = TrainingConfig()  # task=None by default
        kwargs = config.to_train_kwargs()
        assert "task" not in kwargs

    def test_task_from_config_overrides_correctly(self):
        """Verify task value flows through correctly for all valid tasks."""
        for task_name in ["cifar_baseline", "cifar_scale", "cifar_impaired", "cifar_minimal", "tinystories"]:
            config = TrainingConfig(task=task_name)
            kwargs = config.to_train_kwargs()
            assert kwargs.get("task") == task_name


class TestSlotsStringRejection:
    """Tests for slots parameter type validation."""

    def test_slots_string_produces_clear_error(self):
        """Passing a string for slots should fail with helpful message.

        Before the fix, list("r0c1") would produce ['r','0','c','1'],
        leading to confusing "Invalid slot 'r'" errors.
        """
        with pytest.raises(TypeError, match="slots must be a list"):
            TrainingConfig(slots="r0c1")

    def test_slots_string_error_suggests_fix(self):
        """The error message should suggest the correct syntax."""
        with pytest.raises(TypeError, match=r"Did you mean slots=\['r0c1'\]"):
            TrainingConfig(slots="r0c1")

    def test_slots_tuple_still_works(self):
        """Tuple input should still be normalized to list (convenience feature)."""
        config = TrainingConfig(slots=("r0c0", "r0c1"))
        assert config.slots == ["r0c0", "r0c1"]
        assert isinstance(config.slots, list)
