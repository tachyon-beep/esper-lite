#!/usr/bin/env python3
"""Tamiyo WAL soak harness.

Generates a stream of field reports, optionally injecting WAL corruption to
validate retry/backlog telemetry. Designed for quick CI smoke runs as well as
longer manual drills.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import struct
from dataclasses import dataclass
from pathlib import Path

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext
from esper.core.config import EsperSettings
from esper.tamiyo import FieldReportStoreConfig, TamiyoPolicy, TamiyoPolicyConfig, TamiyoService
from esper.tamiyo.persistence import FieldReportStore
from esper.urza import UrzaLibrary

DEFAULT_WORKDIR = Path("./var/tamiyo/soak")


@dataclass
class SoakResult:
    iterations: int
    injections: int
    load_errors_retry_index: int
    load_errors_windows: int
    backlog_total: int


class _CountingOona:
    def __init__(self) -> None:
        self.telemetry: list[leyline_pb2.TelemetryPacket] = []

    async def publish_field_report(self, report: leyline_pb2.FieldReport) -> bool:
        return True

    async def publish_telemetry(
        self,
        packet: leyline_pb2.TelemetryPacket,
        priority: int | None = None,
    ) -> bool:
        self.telemetry.append(packet)
        return True


def _make_policy() -> TamiyoPolicy:
    return TamiyoPolicy(TamiyoPolicyConfig(enable_compile=False))


def _make_service(workdir: Path, strict: bool) -> TamiyoService:
    wal_path = workdir / "field_reports.log"
    config = FieldReportStoreConfig(path=wal_path, strict_validation=strict)
    urza_root = workdir / "urza"
    urza = UrzaLibrary(root=urza_root)
    settings = EsperSettings(TAMIYO_WAL_STRICT_VALIDATION=strict)
    service = TamiyoService(
        policy=_make_policy(),
        store_config=config,
        settings=settings,
        urza=urza,
        signature_context=SignatureContext(secret=b"tamiyo-soak"),
    )
    return service


def _inject_truncated_entry(store: FieldReportStore) -> None:
    path = store.path
    with path.open("ab") as handle:
        handle.write(struct.pack("<I", 64))


def _make_state(run_id: str, idx: int) -> leyline_pb2.SystemStatePacket:
    packet = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=idx,
        training_run_id=run_id,
        packet_id=f"packet-{idx}",
    )
    return packet


async def run_soak(
    workdir: Path,
    *,
    iterations: int = 50,
    inject_every: int = 10,
    strict_reload: bool = False,
) -> SoakResult:
    workdir.mkdir(parents=True, exist_ok=True)
    service = _make_service(workdir, strict_reload)
    injections = 0
    fake = _CountingOona()

    for idx in range(1, iterations + 1):
        state = _make_state("run-soak", idx)
        report = service.generate_field_report(
            command=leyline_pb2.AdaptationCommand(
                version=1,
                command_type=leyline_pb2.COMMAND_PAUSE,
                command_id=f"cmd-{idx}",
            ),
            outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
            metrics_delta={"loss_delta": 0.01 * idx},
            training_run_id="run-soak",
            seed_id=f"seed-{idx}",
            blueprint_id="bp-soak",
        )
        _ = report
        if inject_every and idx % inject_every == 0:
            _inject_truncated_entry(service._field_report_store)  # type: ignore[attr-defined]
            service.close()
            service = _make_service(workdir, strict_reload)
            injections += 1

    await service.publish_history(fake)
    metrics = {}
    if fake.telemetry:
        summary = fake.telemetry[-1]
        metrics = {m.name: m.value for m in summary.metrics}

    result = SoakResult(
        iterations=iterations,
        injections=injections,
        load_errors_retry_index=len(getattr(service, "_retry_index_errors", [])),
        load_errors_windows=len(getattr(service, "_windows_errors", [])),
        backlog_total=int(metrics.get("tamiyo.field_reports.retry_backlog_total", 0)),
    )
    service.close()
    return result


async def _async_main(args: argparse.Namespace) -> int:
    result = await run_soak(
        workdir=Path(args.workdir).expanduser().resolve(),
        iterations=args.iterations,
        inject_every=args.inject_every,
        strict_reload=args.strict,
    )
    print(
        "Tamiyo WAL soak completed",
        {
            "iterations": result.iterations,
            "injections": result.injections,
            "retry_load_errors": result.load_errors_retry_index,
            "window_load_errors": result.load_errors_windows,
            "backlog_total": result.backlog_total,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tamiyo WAL soak harness")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, type=Path)
    parser.add_argument("--iterations", default=50, type=int)
    parser.add_argument("--inject-every", default=10, type=int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
