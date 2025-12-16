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

