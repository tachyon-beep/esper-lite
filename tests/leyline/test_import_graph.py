"""Import-graph guard tests for the leyline → simic back-dependency resolution.

These tests enforce that esper.leyline has ZERO outbound imports to any domain
(simic, kasmina, tamiyo, tolaria, karn, nissa), and that the two telemetry
contracts are importable directly from esper.leyline.

Tests use subprocesses so that previously-imported domain modules in the test
session's sys.modules do not pollute the isolation check.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest


def _run_subprocess_check(code: str) -> tuple[bool, str]:
    """Run a Python snippet in a fresh subprocess and return (success, stderr)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr + result.stdout


# ---------------------------------------------------------------------------
# Import-graph guard tests (must fail BEFORE the move)
# ---------------------------------------------------------------------------

DOMAIN_PREFIXES = [
    "esper.simic",
    "esper.kasmina",
    "esper.tamiyo",
    "esper.tolaria",
    "esper.karn",
    "esper.nissa",
]


def test_leyline_has_no_simic_imports() -> None:
    """leyline must not pull in esper.simic at module load time."""
    code = """
import sys
# Ensure a clean slate - only stdlib preloaded
for key in list(sys.modules):
    if key.startswith("esper"):
        del sys.modules[key]

import esper.leyline  # noqa: E402

simic_keys = [k for k in sys.modules if k.startswith("esper.simic")]
if simic_keys:
    print("FAIL: simic modules found in sys.modules after importing esper.leyline:")
    for k in simic_keys:
        print(f"  {k}")
    raise SystemExit(1)
print("OK")
"""
    ok, output = _run_subprocess_check(code)
    assert ok, f"leyline imported esper.simic:\n{output}"


@pytest.mark.parametrize("domain_prefix", DOMAIN_PREFIXES)
def test_leyline_has_no_domain_imports(domain_prefix: str) -> None:
    """leyline must not pull in any domain module at load time."""
    code = f"""
import sys
for key in list(sys.modules):
    if key.startswith("esper"):
        del sys.modules[key]

import esper.leyline  # noqa: E402

domain_keys = [k for k in sys.modules if k.startswith({domain_prefix!r})]
if domain_keys:
    print(f"FAIL: domain modules found: {{domain_keys}}")
    raise SystemExit(1)
print("OK")
"""
    ok, output = _run_subprocess_check(code)
    assert ok, f"leyline imported {domain_prefix}:\n{output}"


# ---------------------------------------------------------------------------
# Contract importability tests (must fail BEFORE the move, pass after)
# ---------------------------------------------------------------------------


def test_reward_components_telemetry_importable_from_leyline() -> None:
    """RewardComponentsTelemetry must be importable directly from esper.leyline."""
    from esper.leyline import RewardComponentsTelemetry  # type: ignore[attr-defined]

    assert hasattr(RewardComponentsTelemetry, "total_reward"), (
        "RewardComponentsTelemetry must have total_reward field"
    )


def test_observation_stats_telemetry_importable_from_leyline() -> None:
    """ObservationStatsTelemetry must be importable directly from esper.leyline."""
    from esper.leyline import ObservationStatsTelemetry  # type: ignore[attr-defined]

    assert hasattr(ObservationStatsTelemetry, "nan_count"), (
        "ObservationStatsTelemetry must have nan_count field"
    )


# ---------------------------------------------------------------------------
# Round-trip tests (must fail BEFORE the move, pass after)
# ---------------------------------------------------------------------------


def test_reward_components_telemetry_round_trip_in_leyline() -> None:
    """RewardComponentsTelemetry.to_dict() → from_dict() must be identity."""
    from esper.leyline import RewardComponentsTelemetry  # type: ignore[attr-defined]

    original = RewardComponentsTelemetry(
        total_reward=1.5,
        base_acc_delta=0.3,
        compute_rent=-0.1,
        action_name="FOSSILIZE",
        epoch=42,
        val_acc=0.87,
    )
    data = original.to_dict()
    restored = RewardComponentsTelemetry.from_dict(data)
    assert restored.total_reward == original.total_reward
    assert restored.base_acc_delta == original.base_acc_delta
    assert restored.compute_rent == original.compute_rent
    assert restored.action_name == original.action_name
    assert restored.epoch == original.epoch
    assert restored.val_acc == original.val_acc


def test_observation_stats_telemetry_round_trip_in_leyline() -> None:
    """ObservationStatsTelemetry.to_dict() → from_dict() must be identity."""
    from esper.leyline import ObservationStatsTelemetry  # type: ignore[attr-defined]

    original = ObservationStatsTelemetry(
        nan_count=3,
        inf_count=0,
        nan_pct=0.01,
        slot_features_mean=0.5,
        slot_features_std=0.2,
        batch_size=64,
    )
    data = original.to_dict()
    restored = ObservationStatsTelemetry.from_dict(data)
    assert restored.nan_count == original.nan_count
    assert restored.inf_count == original.inf_count
    assert restored.nan_pct == original.nan_pct
    assert restored.slot_features_mean == original.slot_features_mean
    assert restored.batch_size == original.batch_size


# ---------------------------------------------------------------------------
# new_drip_state excision test (must pass both before and after)
# ---------------------------------------------------------------------------


def test_reward_components_telemetry_has_no_new_drip_state() -> None:
    """new_drip_state must NOT be a dataclass field on RewardComponentsTelemetry.

    It was a misused in-memory side channel holding a simic type.
    After excision, the drip state is returned separately by compute_reward().
    """
    from esper.leyline import RewardComponentsTelemetry  # type: ignore[attr-defined]

    field_names = {f.name for f in dataclasses.fields(RewardComponentsTelemetry)}
    assert "new_drip_state" not in field_names, (
        "new_drip_state must be excised from RewardComponentsTelemetry - "
        "it held a simic type and was never serialised"
    )
