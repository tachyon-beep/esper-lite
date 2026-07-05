"""Tests for the shared Sanctum health classifiers.

These are the single source of truth consulted by both the always-on
AnomalyStrip and the per-tab HealthStatusPanel, so the two surfaces render the
same verdict for a given condition.
"""
from esper.karn.sanctum.health import (
    classify_value_health,
    ev_is_critical,
    ratio_is_critical,
)
from esper.karn.sanctum.schema import TamiyoState


# --- classify_value_health ---------------------------------------------------


def test_value_health_empty_state_reads_as_collapse():
    """All-zero value stats trip the collapse branch — callers must gate on data."""
    assert classify_value_health(TamiyoState()) == "Critical"


def test_value_health_healthy_spread_is_ok():
    t = TamiyoState(value_min=-8.0, value_max=14.0, value_std=6.0, value_mean=3.0)
    assert classify_value_health(t) == "OK"


def test_value_health_collapse_constant_values():
    t = TamiyoState(value_min=2.5, value_max=2.5, value_std=0.0, value_mean=2.5)
    assert classify_value_health(t) == "Critical"


def test_value_health_coefficient_of_variation_critical():
    # |mean| > 0.1 and std/|mean| > 3.0 → Critical (range not collapsed).
    t = TamiyoState(value_min=-5.0, value_max=5.0, value_std=4.0, value_mean=1.0)
    assert classify_value_health(t) == "Critical"


def test_value_health_coefficient_of_variation_warning():
    # std/|mean| between 2.0 and 3.0 → Warning.
    t = TamiyoState(value_min=-3.0, value_max=3.0, value_std=2.5, value_mean=1.0)
    assert classify_value_health(t) == "Warning"


def test_value_health_explosion_absolute_fallback():
    t = TamiyoState(value_min=-2000.0, value_max=2000.0, value_std=100.0, value_mean=0.0)
    assert classify_value_health(t) == "Critical"


def test_value_health_relative_to_initial_spread():
    # ratio = range/initial = 20/1 = 20 > 10 → Critical.
    t = TamiyoState(
        value_min=-10.0, value_max=10.0, value_std=1.0, value_mean=0.0,
        initial_value_spread=1.0,
    )
    assert classify_value_health(t) == "Critical"


# --- ev_is_critical ----------------------------------------------------------


def test_ev_critical_when_non_positive():
    assert ev_is_critical(TamiyoState(explained_variance=-0.1)) is True
    assert ev_is_critical(TamiyoState(explained_variance=0.0)) is True


def test_ev_not_critical_when_positive():
    assert ev_is_critical(TamiyoState(explained_variance=0.42)) is False


def test_ev_suppressed_when_low_return_variance():
    """A floored EV denominator is an artifact, not a critic failure."""
    t = TamiyoState(explained_variance=-0.5, ev_low_return_variance=True)
    assert ev_is_critical(t) is False


# --- ratio_is_critical -------------------------------------------------------


def test_ratio_critical_above_explosion_threshold():
    assert ratio_is_critical(TamiyoState(joint_ratio_max=6.0)) is True


def test_ratio_not_critical_at_unity():
    assert ratio_is_critical(TamiyoState(joint_ratio_max=1.0)) is False
