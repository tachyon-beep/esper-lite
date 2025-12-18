"""Tests for SIMPLIFIED reward mode."""
import pytest
from esper.simic.rewards import RewardMode


def test_simplified_mode_exists():
    """RewardMode.SIMPLIFIED should be a valid enum member."""
    assert hasattr(RewardMode, "SIMPLIFIED")
    assert RewardMode.SIMPLIFIED.value == "simplified"


def test_simplified_mode_string_conversion():
    """SIMPLIFIED mode should round-trip through string."""
    mode = RewardMode("simplified")
    assert mode == RewardMode.SIMPLIFIED
