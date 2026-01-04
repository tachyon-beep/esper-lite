"""Tests for telemetry ingestion helpers.

These functions coerce untyped payloads into expected types, with
proper validation and logging of malformed data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from esper.leyline import SeedStage
from esper.karn.ingest import (
    coerce_int,
    coerce_float,
    coerce_str,
    coerce_bool_or_none,
    coerce_float_or_none,
    coerce_int_or_none,
    coerce_str_or_none,
    coerce_float_dict,
    coerce_datetime,
    coerce_path,
    coerce_seed_stage,
    filter_dataclass_kwargs,
)


class TestCoerceInt:
    """Tests for coerce_int function."""

    def test_valid_int_passthrough(self) -> None:
        """Valid ints pass through unchanged."""
        assert coerce_int(42, field="x") == 42
        assert coerce_int(-10, field="x") == -10
        assert coerce_int(0, field="x") == 0

    def test_none_returns_default(self) -> None:
        """None returns the default value."""
        assert coerce_int(None, field="x", default=99) == 99
        assert coerce_int(None, field="x") == 0  # Default default

    def test_bool_rejected(self) -> None:
        """Booleans are rejected (they're ints in Python)."""
        assert coerce_int(True, field="x", default=5) == 5
        assert coerce_int(False, field="x", default=5) == 5

    def test_string_coerced(self) -> None:
        """Numeric strings are coerced to int."""
        assert coerce_int("42", field="x") == 42
        assert coerce_int("-10", field="x") == -10

    def test_invalid_string_returns_default(self) -> None:
        """Non-numeric strings return default."""
        assert coerce_int("hello", field="x", default=7) == 7
        assert coerce_int("", field="x", default=7) == 7

    def test_float_truncated(self) -> None:
        """Floats are truncated to int."""
        assert coerce_int(3.9, field="x") == 3
        assert coerce_int(-2.1, field="x") == -2

    def test_minimum_enforced(self) -> None:
        """Values below minimum return default."""
        assert coerce_int(5, field="x", minimum=10, default=10) == 10
        assert coerce_int(10, field="x", minimum=10, default=0) == 10

    def test_maximum_enforced(self) -> None:
        """Values above maximum return default."""
        assert coerce_int(15, field="x", maximum=10, default=10) == 10
        assert coerce_int(10, field="x", maximum=10, default=0) == 10


class TestCoerceFloat:
    """Tests for coerce_float function."""

    def test_valid_float_passthrough(self) -> None:
        """Valid floats pass through unchanged."""
        assert coerce_float(3.14, field="x") == 3.14
        assert coerce_float(-0.5, field="x") == -0.5

    def test_int_converted_to_float(self) -> None:
        """Ints are converted to floats."""
        assert coerce_float(42, field="x") == 42.0

    def test_none_returns_default(self) -> None:
        """None returns the default value."""
        assert coerce_float(None, field="x", default=1.5) == 1.5
        assert coerce_float(None, field="x") == 0.0

    def test_bool_rejected(self) -> None:
        """Booleans are rejected."""
        assert coerce_float(True, field="x", default=1.0) == 1.0
        assert coerce_float(False, field="x", default=2.0) == 2.0

    def test_nan_rejected(self) -> None:
        """NaN values are rejected."""
        assert coerce_float(float("nan"), field="x", default=0.0) == 0.0

    def test_inf_rejected(self) -> None:
        """Infinity values are rejected."""
        assert coerce_float(float("inf"), field="x", default=0.0) == 0.0
        assert coerce_float(float("-inf"), field="x", default=0.0) == 0.0

    def test_string_coerced(self) -> None:
        """Numeric strings are coerced to float."""
        assert coerce_float("3.14", field="x") == 3.14
        assert coerce_float("-0.5", field="x") == -0.5


class TestCoerceStr:
    """Tests for coerce_str function."""

    def test_valid_string_passthrough(self) -> None:
        """Valid strings pass through unchanged."""
        assert coerce_str("hello", field="x") == "hello"
        assert coerce_str("", field="x") == ""

    def test_none_returns_default(self) -> None:
        """None returns the default value."""
        assert coerce_str(None, field="x", default="fallback") == "fallback"
        assert coerce_str(None, field="x") == ""

    def test_non_string_returns_default(self) -> None:
        """Non-strings return default (no implicit str())."""
        assert coerce_str(42, field="x", default="nope") == "nope"
        assert coerce_str(3.14, field="x", default="nope") == "nope"


class TestCoerceBoolOrNone:
    """Tests for coerce_bool_or_none function."""

    def test_none_passthrough(self) -> None:
        """None passes through as None."""
        assert coerce_bool_or_none(None, field="x") is None

    def test_bool_passthrough(self) -> None:
        """Booleans pass through unchanged."""
        assert coerce_bool_or_none(True, field="x") is True
        assert coerce_bool_or_none(False, field="x") is False

    def test_int_01_coerced(self) -> None:
        """Integers 0 and 1 are coerced to bool."""
        assert coerce_bool_or_none(0, field="x") is False
        assert coerce_bool_or_none(1, field="x") is True

    def test_string_true_variants(self) -> None:
        """String true variants are recognized."""
        for val in ["true", "True", "TRUE", "t", "T", "yes", "YES", "y", "Y", "1"]:
            assert coerce_bool_or_none(val, field="x") is True

    def test_string_false_variants(self) -> None:
        """String false variants are recognized."""
        for val in ["false", "False", "FALSE", "f", "F", "no", "NO", "n", "N", "0"]:
            assert coerce_bool_or_none(val, field="x") is False

    def test_invalid_returns_none(self) -> None:
        """Invalid values return None."""
        assert coerce_bool_or_none(2, field="x") is None
        assert coerce_bool_or_none("maybe", field="x") is None


class TestCoerceFloatOrNone:
    """Tests for coerce_float_or_none function."""

    def test_none_passthrough(self) -> None:
        """None passes through as None."""
        assert coerce_float_or_none(None, field="x") is None

    def test_valid_float_passthrough(self) -> None:
        """Valid floats pass through unchanged."""
        assert coerce_float_or_none(3.14, field="x") == 3.14

    def test_nan_returns_none(self) -> None:
        """NaN returns None."""
        assert coerce_float_or_none(float("nan"), field="x") is None

    def test_inf_returns_none(self) -> None:
        """Infinity returns None."""
        assert coerce_float_or_none(float("inf"), field="x") is None

    def test_bool_returns_none(self) -> None:
        """Booleans return None."""
        assert coerce_float_or_none(True, field="x") is None


class TestCoerceIntOrNone:
    """Tests for coerce_int_or_none function."""

    def test_none_passthrough(self) -> None:
        """None passes through as None."""
        assert coerce_int_or_none(None, field="x") is None

    def test_valid_int_passthrough(self) -> None:
        """Valid ints pass through unchanged."""
        assert coerce_int_or_none(42, field="x") == 42

    def test_bool_returns_none(self) -> None:
        """Booleans return None."""
        assert coerce_int_or_none(True, field="x") is None

    def test_minimum_enforced(self) -> None:
        """Values below minimum return None."""
        assert coerce_int_or_none(5, field="x", minimum=10) is None
        assert coerce_int_or_none(10, field="x", minimum=10) == 10

    def test_maximum_enforced(self) -> None:
        """Values above maximum return None."""
        assert coerce_int_or_none(15, field="x", maximum=10) is None


class TestCoerceStrOrNone:
    """Tests for coerce_str_or_none function."""

    def test_none_passthrough(self) -> None:
        """None passes through as None."""
        assert coerce_str_or_none(None, field="x") is None

    def test_valid_string_passthrough(self) -> None:
        """Valid strings pass through unchanged."""
        assert coerce_str_or_none("hello", field="x") == "hello"

    def test_non_string_returns_none(self) -> None:
        """Non-strings return None."""
        assert coerce_str_or_none(42, field="x") is None


class TestCoerceFloatDict:
    """Tests for coerce_float_dict function."""

    def test_none_returns_empty_dict(self) -> None:
        """None returns empty dict."""
        assert coerce_float_dict(None, field="x") == {}

    def test_valid_dict_passthrough(self) -> None:
        """Valid dicts pass through with coerced values."""
        result = coerce_float_dict({"a": 1.5, "b": 2.0}, field="x")
        assert result == {"a": 1.5, "b": 2.0}

    def test_non_dict_returns_empty(self) -> None:
        """Non-dicts return empty dict."""
        assert coerce_float_dict("not a dict", field="x") == {}
        assert coerce_float_dict([1, 2, 3], field="x") == {}

    def test_non_string_keys_skipped(self) -> None:
        """Non-string keys are skipped."""
        result = coerce_float_dict({1: 1.0, "valid": 2.0}, field="x")
        assert result == {"valid": 2.0}

    def test_values_coerced_to_float(self) -> None:
        """Values are coerced to float."""
        result = coerce_float_dict({"a": "3.14", "b": 42}, field="x")
        assert result == {"a": 3.14, "b": 42.0}


class TestCoerceDatetime:
    """Tests for coerce_datetime function."""

    def test_none_returns_default(self) -> None:
        """None returns the default value."""
        default = datetime(2025, 1, 1)
        assert coerce_datetime(None, field="x", default=default) == default
        assert coerce_datetime(None, field="x") is None

    def test_datetime_passthrough(self) -> None:
        """Datetime objects pass through unchanged."""
        dt = datetime(2025, 6, 15, 12, 30, 0)
        assert coerce_datetime(dt, field="x") == dt

    def test_iso_string_parsed(self) -> None:
        """ISO format strings are parsed."""
        result = coerce_datetime("2025-06-15T12:30:00", field="x")
        assert result == datetime(2025, 6, 15, 12, 30, 0)

    def test_invalid_string_returns_default(self) -> None:
        """Invalid strings return default."""
        default = datetime(2025, 1, 1)
        assert coerce_datetime("not a date", field="x", default=default) == default


class TestCoercePath:
    """Tests for coerce_path function."""

    def test_none_returns_none(self) -> None:
        """None returns None."""
        assert coerce_path(None, field="x") is None

    def test_path_passthrough(self) -> None:
        """Path objects pass through unchanged."""
        p = Path("/tmp/test")
        assert coerce_path(p, field="x") == p

    def test_string_converted_to_path(self) -> None:
        """Strings are converted to Path."""
        result = coerce_path("/tmp/test", field="x")
        assert result == Path("/tmp/test")

    def test_non_string_returns_none(self) -> None:
        """Non-strings/non-paths return None."""
        assert coerce_path(42, field="x") is None


class TestCoerceSeedStage:
    """Tests for coerce_seed_stage function."""

    def test_none_returns_default(self) -> None:
        """None returns the default value."""
        assert coerce_seed_stage(None, field="x") == SeedStage.DORMANT
        assert coerce_seed_stage(None, field="x", default=SeedStage.TRAINING) == SeedStage.TRAINING

    def test_seed_stage_passthrough(self) -> None:
        """SeedStage objects pass through unchanged."""
        assert coerce_seed_stage(SeedStage.TRAINING, field="x") == SeedStage.TRAINING

    def test_int_converted(self) -> None:
        """Valid stage ints are converted."""
        assert coerce_seed_stage(SeedStage.TRAINING.value, field="x") == SeedStage.TRAINING

    def test_string_name_converted(self) -> None:
        """String stage names are converted."""
        assert coerce_seed_stage("TRAINING", field="x") == SeedStage.TRAINING
        assert coerce_seed_stage("DORMANT", field="x") == SeedStage.DORMANT

    def test_bool_rejected(self) -> None:
        """Booleans are rejected."""
        assert coerce_seed_stage(True, field="x", default=SeedStage.DORMANT) == SeedStage.DORMANT

    def test_invalid_returns_default(self) -> None:
        """Invalid values return default."""
        assert coerce_seed_stage("INVALID", field="x") == SeedStage.DORMANT
        assert coerce_seed_stage(999, field="x") == SeedStage.DORMANT


class TestFilterDataclassKwargs:
    """Tests for filter_dataclass_kwargs function."""

    @dataclass
    class SampleDataclass:
        name: str
        value: int

    def test_valid_fields_kept(self) -> None:
        """Fields matching dataclass are kept."""
        raw = {"name": "test", "value": 42}
        result = filter_dataclass_kwargs(self.SampleDataclass, raw, context="test")
        assert result == {"name": "test", "value": 42}

    def test_unknown_fields_filtered(self) -> None:
        """Unknown fields are filtered out."""
        raw = {"name": "test", "value": 42, "extra": "ignored"}
        result = filter_dataclass_kwargs(self.SampleDataclass, raw, context="test")
        assert result == {"name": "test", "value": 42}
        assert "extra" not in result

    def test_non_dataclass_returns_raw(self) -> None:
        """Non-dataclass input returns raw dict unchanged."""
        raw = {"a": 1, "b": 2}
        result = filter_dataclass_kwargs(str, raw, context="test")  # type: ignore
        assert result == {"a": 1, "b": 2}
