#!/usr/bin/env python3
"""Capture canonical Tamiyo risk-engine fixtures for regression testing."""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToJson

from esper.core import TelemetryEvent
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext
from esper.tamiyo import (
    FieldReportStoreConfig,
    RiskConfig,
    TamiyoPolicy,
    TamiyoPolicyConfig,
    TamiyoService,
)
from esper.urza import UrzaLibrary


@dataclass
class Scenario:
    name: str
    description: str
    command: leyline_pb2.AdaptationCommand
    state: leyline_pb2.SystemStatePacket
    loss_delta: float
    blueprint_info: dict[str, Any] | None
    blueprint_timeout: bool
    timed_out: bool
    training_metrics: dict[str, float]


RAND_SEED = 1337


def _baseline_command() -> leyline_pb2.AdaptationCommand:
    cmd = leyline_pb2.AdaptationCommand(version=1, command_type=leyline_pb2.COMMAND_SEED)
    cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = "bp-demo"
    cmd.target_seed_id = "seed-demo"
    cmd.annotations["policy_risk_score"] = "0.10"
    cmd.annotations["policy_risk_index"] = "0"
    cmd.annotations["blending_method"] = "slerp"
    return cmd


def _base_state() -> leyline_pb2.SystemStatePacket:
    return leyline_pb2.SystemStatePacket(
        version=1,
        training_run_id="run-fixture",
        packet_id="pkt-fixture",
        current_epoch=3,
    )


def build_scenarios(risk: RiskConfig) -> list[Scenario]:
    base_state = _base_state()
    blueprint_common = {
        "tier": "BLUEPRINT_TIER_EXPERIMENTAL",
        "tier_index": int(leyline_pb2.BlueprintTier.BLUEPRINT_TIER_EXPERIMENTAL),
        "risk": 0.1,
        "stage": 1,
        "quarantine_only": False,
        "approval_required": False,
        "description": "Fixture baseline",
        "parameter_count": 0,
        "allowed_parameters": {},
    }
    scenarios: list[Scenario] = []

    scenarios.append(
        Scenario(
            name="baseline",
            description="No risk triggers; command remains SEED",
            command=_baseline_command(),
            state=base_state,
            loss_delta=0.05,
            blueprint_info=dict(blueprint_common),
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={},
        )
    )

    cmd_policy = _baseline_command()
    cmd_policy.annotations["policy_risk_score"] = "0.99"
    scenarios.append(
        Scenario(
            name="policy_risk_critical",
            description="policy_risk_score >= 0.98 triggers COMMAND_PAUSE",
            command=cmd_policy,
            state=base_state,
            loss_delta=0.05,
            blueprint_info=dict(blueprint_common),
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={},
        )
    )

    scenarios.append(
        Scenario(
            name="blueprint_quarantine",
            description="Blueprint marked quarantine_only pauses and enters conservative mode",
            command=_baseline_command(),
            state=base_state,
            loss_delta=0.05,
            blueprint_info={**blueprint_common, "quarantine_only": True, "risk": 0.9},
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={},
        )
    )

    scenarios.append(
        Scenario(
            name="bsds_hazard_high",
            description="BSDS hazard HIGH downgrades SEED to OPTIMIZER",
            command=_baseline_command(),
            state=base_state,
            loss_delta=0.05,
            blueprint_info={
                **blueprint_common,
                "risk": 0.4,
                "bsds": {
                    "hazard_band": "HIGH",
                    "risk_score": 0.7,
                    "handling_class": "standard",
                    "provenance": "urabrask",
                },
            },
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={},
        )
    )

    scenarios.append(
        Scenario(
            name="loss_spike_pause",
            description="Loss spike above risk threshold pauses execution",
            command=_baseline_command(),
            state=base_state,
            loss_delta=risk.max_loss_spike * 1.2,
            blueprint_info=dict(blueprint_common),
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={},
        )
    )

    scenarios.append(
        Scenario(
            name="latency_hook_pause",
            description="Hook latency exceeds budget -> pause",
            command=_baseline_command(),
            state=base_state,
            loss_delta=0.05,
            blueprint_info=dict(blueprint_common),
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={"hook_latency_ms": float(risk.hook_budget_ms) * 1.5},
        )
    )

    scenarios.append(
        Scenario(
            name="isolation_violation",
            description="Kasmina isolation violations trigger pause",
            command=_baseline_command(),
            state=base_state,
            loss_delta=0.05,
            blueprint_info=dict(blueprint_common),
            blueprint_timeout=False,
            timed_out=False,
            training_metrics={"kasmina.isolation.violations": 2.0},
        )
    )

    return scenarios


def _prepare_service(tmpdir: Path) -> TamiyoService:
    random.seed(RAND_SEED)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    policy = TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))
    service = TamiyoService(
        policy=policy,
        store_config=FieldReportStoreConfig(path=tmpdir / "field_reports.log"),
        urza=UrzaLibrary(root=tmpdir / "urza"),
        signature_context=SignatureContext(secret=b"tamiyo-fixture-secret"),
        step_timeout_ms=1000.0,
    )
    return service


def _command_to_json(command: leyline_pb2.AdaptationCommand) -> str:
    # Older protobuf runtimes used by the test harness do not support
    # including_default_value_fields. Rely on default behaviour for readability.
    return MessageToJson(command, sort_keys=True)


def _events_to_json(events: list[TelemetryEvent]) -> list[dict[str, Any]]:
    return [
        {
            "description": event.description,
            "level": int(event.level),
            "attributes": dict(event.attributes),
        }
        for event in events
    ]


def _write_fixture(
    out_dir: Path,
    scenario: Scenario,
    input_command: leyline_pb2.AdaptationCommand,
    command: leyline_pb2.AdaptationCommand,
    events: list[TelemetryEvent],
    conservative_mode: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(f"{scenario.description}\n")
    (out_dir / "state_packet.pb").write_bytes(scenario.state.SerializeToString())
    (out_dir / "command_before.pb").write_bytes(input_command.SerializeToString())
    (out_dir / "command_before.json").write_text(_command_to_json(input_command) + "\n")
    (out_dir / "command_after.pb").write_bytes(command.SerializeToString())
    (out_dir / "command_after.json").write_text(_command_to_json(command) + "\n")
    (out_dir / "training_metrics.json").write_text(json.dumps(scenario.training_metrics, indent=2, sort_keys=True) + "\n")
    (out_dir / "loss_delta.json").write_text(json.dumps({"loss_delta": scenario.loss_delta}, indent=2) + "\n")
    (out_dir / "blueprint_info.json").write_text(json.dumps(scenario.blueprint_info or {}, indent=2, sort_keys=True) + "\n")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "blueprint_timeout": scenario.blueprint_timeout,
                "timed_out": scenario.timed_out,
                "conservative_mode_after": conservative_mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out_dir / "events.json").write_text(json.dumps(_events_to_json(events), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/tamiyo_risk_engine"),
        help="Directory to write scenario fixtures",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Specific scenario(s) to capture (default: all)",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        service = _prepare_service(tmp_path)
        scenarios = build_scenarios(service._risk)
        selected = set(args.scenario) if args.scenario else None
        for scenario in scenarios:
            if selected and scenario.name not in selected:
                continue
            service._risk.conservative_mode = False
            command = leyline_pb2.AdaptationCommand()
            command.CopyFrom(scenario.command)
            blueprint_info = None if scenario.blueprint_info is None else dict(scenario.blueprint_info)
            _, events = service._apply_risk_engine(
                command,
                state=scenario.state,
                loss_delta=scenario.loss_delta,
                blueprint_info=blueprint_info,
                blueprint_timeout=scenario.blueprint_timeout,
                timed_out=scenario.timed_out,
                training_metrics=dict(scenario.training_metrics),
            )
            scenario_dir = args.output / scenario.name
            _write_fixture(
                scenario_dir,
                scenario,
                scenario.command,
                command,
                events,
                service._risk.conservative_mode,
            )
            print(f"Captured {scenario.name} -> {scenario_dir}")


if __name__ == "__main__":
    main()
