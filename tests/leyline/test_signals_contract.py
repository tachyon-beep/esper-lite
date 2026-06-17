"""Contract tests for the Leyline TrainingMetrics signals dataclass.

Pins the absence of the dead `grad_norm_host` / `grad_norm_seed` fields:
those had ZERO writers and ZERO readers in src/ and were removed (P1-2).
The live per-slot gradient signal travels via DualGradientStats ->
SeedMetrics.seed_gradient_norm_ratio -> per-slot features, NOT via these
TrainingMetrics fields.
"""

from __future__ import annotations

from esper.leyline.signals import TrainingMetrics


def test_training_metrics_has_no_dead_grad_fields():
    """grad_norm_host / grad_norm_seed must be fully absent (no silently-zeroed remnant)."""
    metrics = TrainingMetrics()

    assert not hasattr(metrics, "grad_norm_host")
    assert not hasattr(metrics, "grad_norm_seed")

    # TrainingMetrics uses __slots__; a deleted field must leave no slot behind.
    slots = TrainingMetrics.__slots__
    assert "grad_norm_host" not in slots
    assert "grad_norm_seed" not in slots


def test_training_metrics_constructs_with_defaults():
    """Round-trip: the dataclass still constructs with defaults after field removal."""
    metrics = TrainingMetrics()

    assert metrics.epoch == 0
    assert metrics.global_step == 0
    assert metrics.best_val_loss == float("inf")
