"""Phase 0: the SeedResidencyTelemetry leyline contract + its AnalyticsSnapshotPayload carry.

Pins the full serialization path the residency 5-link chain depends on: accumulator →
SeedResidencyTelemetry → AnalyticsSnapshotPayload(kind='seed_residency') → to_dict (the JSON the
Karn `seed_residency` view reads) → from_dict round-trip.
"""

from __future__ import annotations

import pytest

from esper.leyline.telemetry import AnalyticsSnapshotPayload
from esper.leyline.telemetry_contracts import SeedResidencyTelemetry
from esper.simic.rewards.residency import SeedResidencyAccumulator
from esper.simic.rewards.types import STAGE_BLENDING, STAGE_FOSSILIZED


def test_seed_residency_telemetry_round_trips() -> None:
    t = SeedResidencyTelemetry(
        env_id=1, episode_idx=3, seed_id="sA", params=2000,
        cf_weighted_integral=12.0, cf_weighted_integral_committed=8.0,
        cf_weighted_integral_uncommitted=4.0, raw_alpha_integral=1.75,
        n_on_path_steps=3, n_none_steps=1, j_per_param=6.0e-3,
    )
    rt = SeedResidencyTelemetry.from_dict(t.to_dict())
    assert rt == t


def test_to_telemetry_from_accumulator() -> None:
    """to_telemetry serializes the accumulator's computed j() + committed/uncommitted split."""
    acc = SeedResidencyAccumulator(params=2000)
    acc.add_step(stage=STAGE_BLENDING, alpha=1.0, seed_contribution=4.0)   # uncommitted
    acc.add_step(stage=STAGE_FOSSILIZED, alpha=1.0, seed_contribution=8.0)  # committed
    t = acc.to_telemetry(env_id=0, episode_idx=2, seed_id="sB")
    assert t.cf_weighted_integral == pytest.approx(12.0)
    assert t.cf_weighted_integral_committed == pytest.approx(8.0)
    assert t.cf_weighted_integral_uncommitted == pytest.approx(4.0)
    assert t.j_per_param == pytest.approx(12.0 / 2000)
    assert t.seed_id == "sB"


def test_analytics_payload_carries_seed_residency_through_to_dict() -> None:
    """The ANALYTICS_SNAPSHOT payload serializes the nested SeedResidencyTelemetry (the JSON the
    Karn seed_residency view projects from $.seed_residency.*) and round-trips."""
    t = SeedResidencyTelemetry(env_id=0, episode_idx=5, seed_id="sC", params=1000, j_per_param=0.5)
    payload = AnalyticsSnapshotPayload(
        kind="seed_residency", env_id=0, episode_idx=5, seed_id="sC", seed_residency=t
    )
    d = payload.to_dict()
    assert d["kind"] == "seed_residency"
    assert d["seed_residency"]["j_per_param"] == pytest.approx(0.5)
    assert d["seed_residency"]["seed_id"] == "sC"
    rt = AnalyticsSnapshotPayload.from_dict(d)
    assert rt.seed_residency == t
    assert rt.seed_id == "sC"


def test_analytics_payload_without_seed_residency_is_none() -> None:
    """A non-residency snapshot (e.g. last_action) carries seed_residency=None and round-trips."""
    payload = AnalyticsSnapshotPayload(kind="last_action", env_id=0)
    d = payload.to_dict()
    assert d["seed_residency"] is None
    assert AnalyticsSnapshotPayload.from_dict(d).seed_residency is None
