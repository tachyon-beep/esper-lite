"""End-to-end acceptance tests for the oracle sandbox proof path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from esper.karn.mcp.views import scan_ingestion_integrity
from esper.leyline import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    TempoAction,
)
from esper.simic.training.oracle_sandbox import (
    OracleSchedule,
    OracleScheduleStep,
    build_default_oracle_schedule,
    build_oracle_ci_fixture,
    run_oracle_schedule,
)


_PROOF_PACKET_PATH = Path(__file__).parents[3] / "scripts" / "proof_packet.py"
_SPEC = importlib.util.spec_from_file_location("proof_packet", _PROOF_PACKET_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_PROOF_PACKET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROOF_PACKET)
build_proof_packet = _PROOF_PACKET.build_proof_packet


def _wait_action(*, op: LifecycleOp) -> FactoredAction:
    return FactoredAction(
        slot_idx=0,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.INSTANT,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=op,
    )


def _rewrite_events_without_operation(tmp_path: Path, operation: str) -> None:
    events_file = next(tmp_path.glob("*/events.jsonl"))
    retained: list[dict] = []
    for raw in events_file.read_text().splitlines():
        event = json.loads(raw)
        data = event["data"]
        if (
            event["event_type"] == "MORPHOLOGY_CAUSAL_LOG"
            and data["operation"] == operation
        ):
            continue
        retained.append(event)
    events_file.write_text("\n".join(json.dumps(event) for event in retained) + "\n")


def test_oracle_sandbox_acceptance_clean_packet_reaches_product_verdict(
    tmp_path: Path,
) -> None:
    schedule = build_default_oracle_schedule()
    result = run_oracle_schedule(
        schedule=schedule,
        fixture=build_oracle_ci_fixture(),
        telemetry_dir=tmp_path,
    )

    assert result.success is True
    assert tuple(step.stage_after for step in result.steps) == (
        "GERMINATED",
        "TRAINING",
        "BLENDING",
        "BLENDING",
        "HOLDING",
        "FOSSILIZED",
        "FOSSILIZED",
    )
    assert result.steps[-1].blocked_by == "mask"
    assert scan_ingestion_integrity(str(tmp_path)).is_clean

    packet = build_proof_packet(
        str(tmp_path),
        proof_profile="oracle-sandbox",
        min_mean_accuracy_roi=0.1,
    )

    assert "Proof profile: `oracle-sandbox`" in packet
    assert "Verdict: `CONTINUE`" in packet
    assert "Oracle sandbox lifecycle trace is proof-grade." in packet
    assert "Oracle policy evidence is present." in packet


def test_oracle_sandbox_acceptance_invalid_schedule_is_blocked_by_masks() -> None:
    invalid_schedule = OracleSchedule(
        name="invalid-prune-from-dormant",
        steps=(
            OracleScheduleStep(
                step_id="prune_dormant",
                action=_wait_action(op=LifecycleOp.PRUNE),
            ),
        ),
    )

    result = run_oracle_schedule(
        schedule=invalid_schedule,
        fixture=build_oracle_ci_fixture(),
    )

    assert result.success is False
    assert len(result.steps) == 1
    assert result.steps[0].blocked_by == "mask"
    assert result.steps[0].handler_called is False


def test_oracle_sandbox_acceptance_missing_oracle_evidence_blocks_packet(
    tmp_path: Path,
) -> None:
    result = run_oracle_schedule(
        schedule=build_default_oracle_schedule(),
        fixture=build_oracle_ci_fixture(),
        telemetry_dir=tmp_path,
    )
    assert result.success is True
    _rewrite_events_without_operation(tmp_path, "ORACLE_POLICY")

    packet = build_proof_packet(
        str(tmp_path),
        proof_profile="oracle-sandbox",
        min_mean_accuracy_roi=0.1,
    )

    assert "Verdict: `BLOCKED_INSTRUMENTATION`" in packet
    assert "missing oracle-policy evidence" in packet
