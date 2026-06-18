"""Contract tests for the Leyline TrainingMetrics signals dataclass.

Pins the absence of the dead `grad_norm_host` / `grad_norm_seed` fields:
those had ZERO writers and ZERO readers in src/ and were removed (P1-2).
The live per-slot gradient signal travels via DualGradientStats ->
SeedMetrics.seed_gradient_norm_ratio -> per-slot features, NOT via these
TrainingMetrics fields.
"""

from __future__ import annotations

import pytest

from esper.leyline.signals import TrainingMetrics


def test_training_metrics_has_no_dead_grad_fields():
    """grad_norm_host / grad_norm_seed must be fully absent (no silently-zeroed remnant)."""
    metrics = TrainingMetrics()

    # TrainingMetrics is a slots dataclass: __slots__ is the authoritative field
    # contract. A deleted field must leave no slot behind, so the access must fail
    # loudly rather than return a silently-zeroed remnant.
    slots = TrainingMetrics.__slots__
    assert "grad_norm_host" not in slots
    assert "grad_norm_seed" not in slots

    with pytest.raises(AttributeError):
        metrics.grad_norm_host
    with pytest.raises(AttributeError):
        metrics.grad_norm_seed


def test_training_metrics_constructs_with_defaults():
    """Round-trip: the dataclass still constructs with defaults after field removal."""
    metrics = TrainingMetrics()

    assert metrics.epoch == 0
    assert metrics.global_step == 0
    assert metrics.best_val_loss == float("inf")
