"""Tests for leyline reports module - SeedMetrics, SeedStateReport, FieldReport."""

from esper.leyline.reports import SeedMetrics


def test_seed_metrics_has_interaction_fields():
    """SeedMetrics should have interaction tracking fields."""
    metrics = SeedMetrics()

    # Interaction sum: total synergy received from other seeds
    assert hasattr(metrics, "interaction_sum")
    assert metrics.interaction_sum == 0.0

    # Boost received: max single interaction from any other seed
    assert hasattr(metrics, "boost_received")
    assert metrics.boost_received == 0.0

    # Upstream alpha sum: total alpha of seeds in earlier slots
    assert hasattr(metrics, "upstream_alpha_sum")
    assert metrics.upstream_alpha_sum == 0.0

    # Downstream alpha sum: total alpha of seeds in later slots
    assert hasattr(metrics, "downstream_alpha_sum")
    assert metrics.downstream_alpha_sum == 0.0
