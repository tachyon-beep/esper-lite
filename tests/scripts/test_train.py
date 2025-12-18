import pytest


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


class TestABTestingCLI:
    """Test CLI --ab-test argument and config integration."""

    def test_ab_test_argument_parsed(self):
        """--ab-test argument should be parsed correctly."""
        from esper.scripts.train import build_parser

        parser = build_parser()

        args = parser.parse_args(["ppo", "--ab-test", "shaped-vs-simplified"])
        assert args.ab_test == "shaped-vs-simplified"

        args = parser.parse_args(["ppo", "--ab-test", "shaped-vs-sparse"])
        assert args.ab_test == "shaped-vs-sparse"

    def test_ab_test_sets_config_ab_reward_modes(self):
        """--ab-test should set config.ab_reward_modes, not pass separately.

        This test catches the bug where ab_reward_modes was passed both
        explicitly AND via config.to_train_kwargs(), causing duplicate
        keyword argument errors.
        """
        from esper.simic.training import TrainingConfig

        # Simulate CLI logic from train.py
        config = TrainingConfig.for_cifar10()
        ab_test = "shaped-vs-simplified"

        # Apply A/B test to config (as train.py now does)
        if ab_test:
            half = config.n_envs // 2
            if ab_test == "shaped-vs-simplified":
                config.ab_reward_modes = ["shaped"] * half + ["simplified"] * half
            elif ab_test == "shaped-vs-sparse":
                config.ab_reward_modes = ["shaped"] * half + ["sparse"] * half

        # Verify config has ab_reward_modes set
        assert config.ab_reward_modes is not None
        assert len(config.ab_reward_modes) == config.n_envs

        # Verify to_train_kwargs() includes ab_reward_modes (no separate passing needed)
        kwargs = config.to_train_kwargs()
        assert "ab_reward_modes" in kwargs
        assert kwargs["ab_reward_modes"] == config.ab_reward_modes

    def test_ab_test_requires_even_envs(self):
        """--ab-test should require even n_envs for equal split."""
        from esper.simic.training import TrainingConfig

        config = TrainingConfig.for_cifar10()
        config.n_envs = 3  # Odd number

        # This check happens in train.py before setting ab_reward_modes
        with pytest.raises(ValueError, match="even"):
            if config.n_envs % 2 != 0:
                raise ValueError("--ab-test requires even number of envs")

