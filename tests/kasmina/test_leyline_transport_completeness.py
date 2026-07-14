"""Transport-completeness: every obs-consumed metric survives kasmina -> leyline.

Defect class this closes (defect register A2/A3, PDR-0100): `SeedMetrics.to_leyline()`
is a manual field-by-field copy, so a field can be computed on the kasmina side,
serialized into checkpoints, and still silently never reach the leyline report the
observation encoder consumes — the policy then reads the leyline dataclass default
forever. Instance zero: `contribution_velocity`, dead (constant 0.0) in every run
from Obs V3's inception until 2026-07-14 (bug esper-lite-f0a82adccb).

The completeness test enumerates the metric fields `batch_obs_to_features` actually
reads from `report.metrics` and asserts a distinct non-default value survives the
transport for each. Adding an obs-consumed field without its `to_leyline` copy line
fails HERE, not in production three months later.
"""

from __future__ import annotations

from esper.kasmina.slot import SeedMetrics

# The report.metrics fields consumed by the observation encoder
# (src/esper/tamiyo/policy/features.py per-slot block; grep `report.metrics.`).
# Keep in lockstep with the encoder — the dim-liveness test in
# tests/tamiyo/test_obs_dim_liveness.py guards the other direction.
OBS_CONSUMED_METRIC_FIELDS: dict[str, float] = {
    "counterfactual_contribution": 7.25,
    "contribution_velocity": 3.75,
    "interaction_sum": 2.5,
    "current_alpha": 0.625,
    "gradient_norm_avg": 1.125,
    "seed_param_count": 12345,
    "epochs_in_current_stage": 9,
    "epochs_total": 17,
}


def _kasmina_metrics_with_sentinels() -> SeedMetrics:
    metrics = SeedMetrics()
    for field, sentinel in OBS_CONSUMED_METRIC_FIELDS.items():
        setattr(metrics, field, sentinel)
    return metrics


def test_every_obs_consumed_field_survives_to_leyline():
    metrics = _kasmina_metrics_with_sentinels()
    leyline = metrics.to_leyline()
    missing = {
        field: getattr(leyline, field)
        for field, sentinel in OBS_CONSUMED_METRIC_FIELDS.items()
        if getattr(leyline, field) != sentinel
    }
    assert not missing, (
        f"to_leyline() dropped obs-consumed fields (leyline value != sentinel): {missing}. "
        "A dropped field reads as the leyline dataclass default in the observation "
        "FOREVER — the contribution_velocity defect class (esper-lite-f0a82adccb). "
        "Add the copy line in SeedMetrics.to_leyline()."
    )


def test_contribution_velocity_transport_specifically():
    # The named instance-zero regression test (one-line fix at slot.py to_leyline).
    metrics = SeedMetrics()
    metrics.contribution_velocity = -1.5
    assert metrics.to_leyline().contribution_velocity == -1.5
