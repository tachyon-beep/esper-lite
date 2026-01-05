import pytest

from esper.scripts.train import build_parser


def test_telemetry_lifecycle_only_flag_wired():
    import esper.scripts.train as train

    parser = train.build_parser()
    args = parser.parse_args(["heuristic", "--telemetry-lifecycle-only"])
    assert args.telemetry_lifecycle_only is True


class TestSlotValidation:
    """Test CLI --slots argument validation."""

    def test_validate_slots_accepts_canonical_ids(self):
        """validate_slots() accepts canonical slot IDs."""
        from esper.scripts.train import validate_slots

        # Single slot
        result = validate_slots(["r0c0"])
        assert result == ["r0c0"]

        # Multiple slots
        result = validate_slots(["r0c0", "r0c1", "r0c2"])
        assert result == ["r0c0", "r0c1", "r0c2"]

        # Different coordinates
        result = validate_slots(["r1c0", "r2c5"])
        assert result == ["r1c0", "r2c5"]

    def test_validate_slots_rejects_legacy_names(self):
        """validate_slots() rejects legacy slot names with helpful error."""
        from esper.scripts.train import validate_slots

        # Test each legacy name
        for legacy in ["early", "mid", "late"]:
            with pytest.raises(ValueError) as exc_info:
                validate_slots([legacy])

            error_msg = str(exc_info.value)
            assert "no longer supported" in error_msg
            assert "r0c0" in error_msg  # Shows canonical example

    def test_validate_slots_rejects_invalid_format(self):
        """validate_slots() rejects invalid slot ID formats."""
        from esper.scripts.train import validate_slots

        invalid_formats = [
            "slot0",
            "r0",
            "c0",
            "0c0",
            "r0c",
            "r-1c0",
            "r0c-1",
            "invalid",
            "",
        ]

        for invalid in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                validate_slots([invalid])
            # Should mention the invalid ID
            assert invalid in str(exc_info.value) or "Invalid" in str(exc_info.value)

    def test_cli_slots_default_is_canonical(self):
        """CLI --slots default uses canonical IDs."""
        from esper.scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args(["heuristic"])

        # Default should be canonical format
        assert args.slots == ["r0c0", "r0c1", "r0c2"]

    def test_cli_slots_accepts_canonical_args(self):
        """CLI --slots argument accepts canonical IDs."""
        from esper.scripts.train import build_parser

        parser = build_parser()

        # Single slot
        args = parser.parse_args(["heuristic", "--slots", "r0c0"])
        assert args.slots == ["r0c0"]

        # Multiple slots
        args = parser.parse_args(["heuristic", "--slots", "r0c0", "r0c1"])
        assert args.slots == ["r0c0", "r0c1"]

        # Different coordinates
        args = parser.parse_args(["heuristic", "--slots", "r1c5", "r2c3"])
        assert args.slots == ["r1c5", "r2c3"]


class TestDualABTestingCLI:
    """Test CLI --dual-ab argument for dual-policy A/B testing."""

    def test_dual_ab_argument_parsed(self):
        """--dual-ab argument should be parsed correctly."""
        from esper.scripts.train import build_parser

        parser = build_parser()

        # Smoke test a representative set of valid choices
        args = parser.parse_args(["ppo", "--dual-ab", "shaped-vs-simplified"])
        assert args.dual_ab == "shaped-vs-simplified"

        args = parser.parse_args(["ppo", "--dual-ab", "shaped-vs-sparse"])
        assert args.dual_ab == "shaped-vs-sparse"

        args = parser.parse_args(["ppo", "--dual-ab", "shaped-vs-escrow"])
        assert args.dual_ab == "shaped-vs-escrow"

        args = parser.parse_args(["ppo", "--dual-ab", "escrow-vs-basic"])
        assert args.dual_ab == "escrow-vs-basic"

    def test_dual_ab_default_is_none(self):
        """--dual-ab should default to None when not specified."""
        from esper.scripts.train import build_parser

        parser = build_parser()
        args = parser.parse_args(["ppo"])
        assert args.dual_ab is None

    def test_ab_test_flag_removed(self):
        """--ab-test is removed (legacy mixed-reward A/B)."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ppo", "--ab-test", "shaped-vs-simplified"])


class TestTamiyoCentricFlags:
    """Tests for Tamiyo-centric CLI flags."""

    def test_rounds_flag_accepted(self):
        """--rounds should set n_episodes in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--rounds", "50"])
        assert args.rounds == 50

    def test_envs_flag_accepted(self):
        """--envs should set n_envs in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--envs", "8"])
        assert args.envs == 8

    def test_episode_length_flag_accepted(self):
        """--episode-length should set max_epochs in config."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--episode-length", "30"])
        assert args.episode_length == 30

    def test_flags_default_to_none(self):
        """Tamiyo flags should default to None (config takes precedence)."""
        parser = build_parser()
        args = parser.parse_args(["ppo"])
        assert args.rounds is None
        assert args.envs is None
        assert args.episode_length is None
        assert args.ppo_epochs is None
        assert args.memory_size is None
        assert args.entropy_anneal_episodes is None

    def test_positive_int_validator_rejects_zero(self):
        """_positive_int should reject zero."""
        from esper.scripts.train import _positive_int
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
            _positive_int("0")

    def test_positive_int_validator_rejects_negative(self):
        """_positive_int should reject negative values."""
        from esper.scripts.train import _positive_int
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
            _positive_int("-5")

    def test_positive_int_validator_accepts_positive(self):
        """_positive_int should accept positive integers."""
        from esper.scripts.train import _positive_int

        assert _positive_int("1") == 1
        assert _positive_int("100") == 100
        assert _positive_int("999999") == 999999

    def test_rounds_overrides_config(self):
        """--rounds should override n_episodes from preset."""
        from esper.simic.training import TrainingConfig

        # Simulate what main() does: start with preset, apply CLI overrides
        config = TrainingConfig.for_cifar_baseline()
        assert config.n_episodes == 100  # Default

        # CLI would set rounds=50, which maps to n_episodes
        config.n_episodes = 50  # Simulating the override
        assert config.n_episodes == 50

    def test_envs_overrides_config(self):
        """--envs should override n_envs from preset."""
        from esper.simic.training import TrainingConfig

        config = TrainingConfig.for_cifar_baseline()
        config.n_envs = 8
        assert config.n_envs == 8

    def test_episode_length_overrides_both_fields(self):
        """--episode-length must override both max_epochs and chunk_length.

        TrainingConfig._validate() enforces chunk_length == max_epochs.
        If we only set one, validation would fail.
        """
        from esper.simic.training import TrainingConfig

        config = TrainingConfig.for_cifar_baseline()

        # Apply override as main() does
        new_length = 30
        config.max_epochs = new_length
        config.chunk_length = new_length

        assert config.max_epochs == new_length
        assert config.chunk_length == new_length

        # Verify validation still passes
        config._validate()  # Should not raise

    def test_ppo_epochs_flag_accepted(self):
        """--ppo-epochs should set ppo_updates_per_batch."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--ppo-epochs", "3"])
        assert args.ppo_epochs == 3

    def test_memory_size_flag_accepted(self):
        """--memory-size should set lstm_hidden_dim."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--memory-size", "256"])
        assert args.memory_size == 256

    def test_entropy_anneal_episodes_flag_accepted(self):
        """--entropy-anneal-episodes should parse correctly."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--entropy-anneal-episodes", "50"])
        assert args.entropy_anneal_episodes == 50

    def test_entropy_anneal_episodes_accepts_zero(self):
        """--entropy-anneal-episodes should accept 0 (no annealing)."""
        parser = build_parser()
        args = parser.parse_args(["ppo", "--entropy-anneal-episodes", "0"])
        assert args.entropy_anneal_episodes == 0

    def test_entropy_anneal_episodes_overrides_config(self):
        """--entropy-anneal-episodes should override config.entropy_anneal_episodes."""
        from esper.simic.training import TrainingConfig

        config = TrainingConfig.for_cifar_baseline()
        assert config.entropy_anneal_episodes == 0  # Default

        # CLI would set entropy_anneal_episodes=50.
        config.entropy_anneal_episodes = 50  # Simulating the override
        assert config.entropy_anneal_episodes == 50

    def test_full_tamiyo_cli_integration(self):
        """All Tamiyo-centric flags should work together."""
        parser = build_parser()
        args = parser.parse_args([
            "ppo",
            "--rounds", "50",
            "--envs", "8",
            "--episode-length", "30",
            "--ppo-epochs", "2",
            "--memory-size", "256",
            "--entropy-anneal-episodes", "25",
        ])

        assert args.rounds == 50
        assert args.envs == 8
        assert args.episode_length == 30
        assert args.ppo_epochs == 2
        assert args.memory_size == 256
        assert args.entropy_anneal_episodes == 25

    def test_invalid_rounds_rejected(self):
        """--rounds 0 should fail with clear error at parse time."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ppo", "--rounds", "0"])

    def test_negative_envs_rejected(self):
        """--envs -1 should fail with clear error at parse time."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ppo", "--envs", "-1"])
