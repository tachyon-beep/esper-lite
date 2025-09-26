from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
from esper.core import TelemetryEvent
from esper.leyline import leyline_pb2

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "tamiyo_risk_engine"


@dataclass(slots=True)
class RiskEngineFixture:
    name: str
    description: str
    command_before: leyline_pb2.AdaptationCommand
    command_after: leyline_pb2.AdaptationCommand
    state: leyline_pb2.SystemStatePacket
    loss_delta: float
    blueprint_info: dict[str, Any] | None
    blueprint_timeout: bool
    timed_out: bool
    training_metrics: dict[str, float]
    events: list[TelemetryEvent]
    conservative_mode_after: bool


def _load_proto(path: Path, message: Any) -> Any:
    data = path.read_bytes()
    message.ParseFromString(data)
    return message


def _load_events(path: Path) -> list[TelemetryEvent]:
    events_json: list[dict[str, Any]] = json.loads(path.read_text())
    return [
        TelemetryEvent(
            description=str(item.get("description", "")),
            level=int(item.get("level", 0)),
            attributes={str(k): str(v) for k, v in item.get("attributes", {}).items()},
        )
        for item in events_json
    ]


def load_risk_fixture(name: str) -> RiskEngineFixture:
    folder = _FIXTURE_ROOT / name
    if not folder.exists():
        raise FileNotFoundError(f"Risk engine fixture '{name}' not found at {folder}")
    command_before = _load_proto(folder / "command_before.pb", leyline_pb2.AdaptationCommand())
    command_after = _load_proto(folder / "command_after.pb", leyline_pb2.AdaptationCommand())
    state = _load_proto(folder / "state_packet.pb", leyline_pb2.SystemStatePacket())
    loss_delta = float(json.loads((folder / "loss_delta.json").read_text())["loss_delta"])
    training_metrics = json.loads((folder / "training_metrics.json").read_text())
    blueprint_info = json.loads((folder / "blueprint_info.json").read_text()) or None
    metadata = json.loads((folder / "metadata.json").read_text())
    events = _load_events(folder / "events.json")
    description = (folder / "README.md").read_text().strip()
    return RiskEngineFixture(
        name=name,
        description=description,
        command_before=command_before,
        command_after=command_after,
        state=state,
        loss_delta=loss_delta,
        blueprint_info=blueprint_info,
        blueprint_timeout=bool(metadata.get("blueprint_timeout", False)),
        timed_out=bool(metadata.get("timed_out", False)),
        training_metrics={str(k): float(v) for k, v in training_metrics.items()},
        events=events,
        conservative_mode_after=bool(metadata.get("conservative_mode_after", False)),
    )


@pytest.fixture
def risk_fixture_loader() -> Callable[[str], RiskEngineFixture]:
    return load_risk_fixture


@pytest.fixture(scope="session")
def risk_fixture_names() -> list[str]:
    return sorted(p.name for p in _FIXTURE_ROOT.iterdir() if p.is_dir())
