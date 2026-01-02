"""Tests for esper.nissa.config - TelemetryConfig validation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from esper.leyline import OBS_V3_NON_BLUEPRINT_DIM
from esper.nissa.config import TelemetryConfig


class TestFromYamlValidation:
    """Test YAML validation in TelemetryConfig.from_yaml()."""

    def test_empty_yaml_raises_clear_error(self, tmp_path: Path) -> None:
        """Empty YAML file raises ValueError with clear message."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="must be a mapping, got NoneType"):
            TelemetryConfig.from_yaml(empty_file)

    def test_scalar_yaml_raises_clear_error(self, tmp_path: Path) -> None:
        """YAML with only a scalar raises ValueError."""
        scalar_file = tmp_path / "scalar.yaml"
        scalar_file.write_text("just a string")

        with pytest.raises(ValueError, match="must be a mapping, got str"):
            TelemetryConfig.from_yaml(scalar_file)

    def test_list_yaml_raises_clear_error(self, tmp_path: Path) -> None:
        """YAML with only a list raises ValueError."""
        list_file = tmp_path / "list.yaml"
        list_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="must be a mapping, got list"):
            TelemetryConfig.from_yaml(list_file)

    def test_valid_yaml_loads_successfully(self, tmp_path: Path) -> None:
        """Valid YAML mapping loads without error."""
        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text("history_length: 15\nprofile_name: custom")

        config = TelemetryConfig.from_yaml(valid_file)
        assert config.history_length == 15
        assert config.profile_name == "custom"

    def test_overrides_applied_correctly(self, tmp_path: Path) -> None:
        """Overrides are deep-merged into YAML config."""
        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text("history_length: 10\ngradients:\n  enabled: false")

        config = TelemetryConfig.from_yaml(
            valid_file,
            overrides={"gradients": {"enabled": True}},
        )
        assert config.history_length == 10
        assert config.gradients.enabled is True


class TestFromProfileValidation:
    """Test YAML validation in TelemetryConfig.from_profile()."""

    def test_empty_profiles_yaml_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty profiles.yaml raises ValueError."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("")

        # Patch __file__ in the config module to point to our tmp_path
        import esper.nissa.config as config_module

        monkeypatch.setattr(config_module, "__file__", str(tmp_path / "config.py"))

        with pytest.raises(ValueError, match="must be a mapping, got NoneType"):
            TelemetryConfig.from_profile("standard")

    def test_profiles_yaml_missing_profiles_key_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """profiles.yaml without 'profiles' key raises clear error."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("something_else:\n  key: value")

        # Patch __file__ in the config module to point to our tmp_path
        import esper.nissa.config as config_module

        monkeypatch.setattr(config_module, "__file__", str(tmp_path / "config.py"))

        with pytest.raises(ValueError, match="missing required 'profiles' key"):
            TelemetryConfig.from_profile("standard")

    def test_profiles_key_not_mapping_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'profiles' key that is not a mapping raises clear error."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text("profiles: [minimal, standard]")  # List, not dict

        # Patch __file__ in the config module to point to our tmp_path
        import esper.nissa.config as config_module

        monkeypatch.setattr(config_module, "__file__", str(tmp_path / "config.py"))

        with pytest.raises(ValueError, match="'profiles' key must be a mapping"):
            TelemetryConfig.from_profile("standard")

    def test_unknown_profile_lists_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown profile name shows available profiles in error."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text(
            "profiles:\n  minimal:\n    history_length: 5\n  standard:\n    history_length: 10"
        )

        # Patch __file__ in the config module to point to our tmp_path
        import esper.nissa.config as config_module

        monkeypatch.setattr(config_module, "__file__", str(tmp_path / "config.py"))

        with pytest.raises(ValueError, match="Unknown profile: nonexistent") as exc_info:
            TelemetryConfig.from_profile("nonexistent")

        # Check that available profiles are listed
        assert "minimal" in str(exc_info.value)
        assert "standard" in str(exc_info.value)


class TestBuiltInProfiles:
    """Test loading of built-in profiles."""

    def test_minimal_profile_loads(self) -> None:
        """Minimal profile loads successfully."""
        config = TelemetryConfig.minimal()
        assert config.profile_name == "minimal"
        assert config.gradients.enabled is False

    def test_standard_profile_loads(self) -> None:
        """Standard profile loads successfully."""
        config = TelemetryConfig.standard()
        assert config.profile_name == "standard"
        assert config.gradients.enabled is True

    def test_diagnostic_profile_loads(self) -> None:
        """Diagnostic profile loads successfully."""
        config = TelemetryConfig.diagnostic()
        assert config.profile_name == "diagnostic"
        assert config.per_class.enabled is True

    def test_research_profile_loads(self) -> None:
        """Research profile loads successfully."""
        config = TelemetryConfig.research()
        assert config.profile_name == "research"
        assert config.gradients.full_histogram is True


class TestFeatureCountEstimate:
    """Test feature_count_estimate() uses correct constants."""

    def test_base_count_uses_leyline_constant(self) -> None:
        """Base feature count comes from leyline OBS_V3_NON_BLUEPRINT_DIM."""
        config = TelemetryConfig.minimal()

        # With minimal config (no extra features), base count should match leyline
        # Plus the history adjustment: (5 - 5) * 2 = 0
        expected = OBS_V3_NON_BLUEPRINT_DIM
        actual = config.feature_count_estimate()

        assert actual == expected, (
            f"Expected base count {expected} (OBS_V3_NON_BLUEPRINT_DIM), got {actual}"
        )

    def test_per_class_uses_num_classes_parameter(self) -> None:
        """Per-class features scale with num_classes parameter."""
        config = TelemetryConfig.from_profile(
            "diagnostic",
            overrides={"per_class": {"track_loss": False, "track_confusion": False}},
        )

        # Get counts for different class counts
        count_10 = config.feature_count_estimate(num_classes=10)
        count_100 = config.feature_count_estimate(num_classes=100)

        # The difference should be exactly 90 (100 - 10) for per-class accuracy
        assert count_100 - count_10 == 90

    def test_confusion_matrix_scales_quadratically(self) -> None:
        """Confusion matrix features scale as num_classes^2."""
        config = TelemetryConfig.from_profile(
            "research",  # Has track_confusion: true
        )

        count_10 = config.feature_count_estimate(num_classes=10)
        count_5 = config.feature_count_estimate(num_classes=5)

        # With confusion matrix, per_class adds: N + N (if track_loss) + N*N
        # Research has track_accuracy=True, track_loss=True, track_confusion=True
        # Difference for N=10 vs N=5:
        # (10 + 10 + 100) - (5 + 5 + 25) = 120 - 35 = 85
        expected_diff = (10 + 10 + 100) - (5 + 5 + 25)
        actual_diff = count_10 - count_5

        assert actual_diff == expected_diff


class TestDeepMerge:
    """Test deep_merge utility function."""

    def test_nested_override(self) -> None:
        """Nested dictionaries are properly merged."""
        from esper.nissa.config import deep_merge

        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"nested": {"y": 99}}

        result = deep_merge(base, override)

        assert result["a"] == 1
        assert result["nested"]["x"] == 10  # Preserved from base
        assert result["nested"]["y"] == 99  # Overridden

    def test_new_keys_added(self) -> None:
        """New keys from override are added."""
        from esper.nissa.config import deep_merge

        base = {"a": 1}
        override = {"b": 2}

        result = deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"] == 2

    def test_base_not_mutated(self) -> None:
        """Original base dict is not mutated."""
        from esper.nissa.config import deep_merge

        base = {"a": 1, "nested": {"x": 10}}
        override = {"nested": {"y": 20}}

        deep_merge(base, override)

        # Base should be unchanged
        assert "y" not in base["nested"]
