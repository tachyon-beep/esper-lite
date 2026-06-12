"""Tests for leyline reports module - SeedMetrics, SeedStateReport, FieldReport."""

from esper.leyline.reports import SeedMetrics


def test_seed_metrics_has_interaction_fields():
    """SeedMetrics should have interaction tracking fields."""
    metrics = SeedMetrics()

    assert metrics.interaction_sum == 0.0

    assert metrics.boost_received == 0.0

    assert metrics.upstream_alpha_sum == 0.0

    assert metrics.downstream_alpha_sum == 0.0
