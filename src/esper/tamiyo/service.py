"""Tamiyo service wrapper combining policy inference and risk gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Tuple

import torch
import logging

from esper.core import EsperSettings, TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
try:
    from esper.urza import UrzaLibrary
except ImportError:  # pragma: no cover - optional import in certain test contexts
    UrzaLibrary = None  # type: ignore

from .policy import TamiyoPolicy, TamiyoPolicyConfig
from .persistence import FieldReportStore, FieldReportStoreConfig

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


@dataclass(slots=True)
class RiskConfig:
    """Configuration for Tamiyo risk thresholds."""

    max_loss_spike: float = 0.15
    conservative_mode: bool = False


_DEFAULT_REPORT_LOG = Path("var/tamiyo/field_reports.log")


class TamiyoService:
    """High-level Tamiyo orchestration component."""

    def __init__(
        self,
        policy: TamiyoPolicy | None = None,
        risk_config: RiskConfig | None = None,
        store: FieldReportStore | None = None,
        store_config: FieldReportStoreConfig | None = None,
        settings: EsperSettings | None = None,
        urza: UrzaLibrary | None = None,
        metadata_cache_ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        self._policy = policy or TamiyoPolicy(TamiyoPolicyConfig())
        self._risk = risk_config or RiskConfig()
        self._settings = settings or EsperSettings()
        self._urza = urza
        self._metadata_cache_ttl = metadata_cache_ttl
        self._blueprint_cache: Dict[str, Tuple[datetime, dict[str, float | str | bool | int]]] = {}
        if store and store_config:
            msg = "Provide either a FieldReportStore instance or a config, not both"
            raise ValueError(msg)
        if store is None:
            if store_config is None:
                store_config = FieldReportStoreConfig(
                    path=_DEFAULT_REPORT_LOG,
                    retention=timedelta(hours=self._settings.tamiyo_field_report_retention_hours),
                )
            store = FieldReportStore(store_config)
        self._field_report_store = store
        self._telemetry_packets: list[leyline_pb2.TelemetryPacket] = []
        self._field_reports: list[leyline_pb2.FieldReport] = store.reports()
        self._policy_updates: list[leyline_pb2.PolicyUpdate] = []
        self._last_validation_loss: float | None = None
        self._policy_version = "policy-stub"

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Evaluate epoch state and apply risk gating."""

        command = self._policy.select_action(state)
        last_action = self._policy.last_action
        command.annotations["policy_action"] = str(int(last_action.get("action", 0.0)))
        command.annotations["policy_param_delta"] = f"{last_action.get('param_delta', 0.0):.6f}"
        risk_event: list[TelemetryEvent] = []
        loss_delta = 0.0
        if self._last_validation_loss is not None:
            loss_delta = state.validation_loss - self._last_validation_loss

        blueprint_info = self._resolve_blueprint_info(command)
        if blueprint_info:
            command.annotations["blueprint_tier"] = blueprint_info["tier"]
            command.annotations["blueprint_stage"] = str(blueprint_info["stage"])
            command.annotations["blueprint_risk"] = f"{blueprint_info['risk']:.2f}"
            if blueprint_info["quarantine_only"] or blueprint_info["risk"] >= 0.8:
                command.command_type = leyline_pb2.COMMAND_PAUSE
                command.annotations.setdefault("risk_reason", "blueprint_high_risk")
                risk_event.append(
                    TelemetryEvent(
                        description="Tamiyo quarantined blueprint",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={
                            "tier": blueprint_info["tier"],
                            "risk_score": f"{blueprint_info['risk']:.2f}",
                        },
                    )
                )

        if self._risk.conservative_mode or (
            self._last_validation_loss is not None
            and loss_delta > self._risk.max_loss_spike
        ):
            command.command_type = leyline_pb2.COMMAND_PAUSE
            command.annotations["risk_reason"] = (
                "conservative_mode"
                if self._risk.conservative_mode
                else f"loss_spike:{loss_delta:.6f}"
            )
            risk_event.append(
                TelemetryEvent(
                    description="Tamiyo pause triggered",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "loss_delta": f"{loss_delta:.6f}",
                        "conservative": str(self._risk.conservative_mode).lower(),
                    },
                )
            )

        metrics = [
            TelemetryMetric("tamiyo.validation_loss", state.validation_loss, unit="loss"),
            TelemetryMetric("tamiyo.loss_delta", loss_delta, unit="loss"),
            TelemetryMetric("tamiyo.policy.action", last_action.get("action", 0.0), unit="index"),
            TelemetryMetric("tamiyo.policy.param_delta", last_action.get("param_delta", 0.0), unit="delta"),
        ]
        if blueprint_info:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blueprint.risk",
                    blueprint_info["risk"],
                    unit="score",
                )
            )
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blueprint.stage",
                    float(blueprint_info["stage"]),
                    unit="stage",
                )
            )
        telemetry = build_telemetry_packet(
            packet_id=state.packet_id or "tamiyo-telemetry",
            source="tamiyo",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=risk_event,
        )
        self._telemetry_packets.append(telemetry)
        self._last_validation_loss = state.validation_loss
        return command

    def generate_field_report(
        self,
        command: leyline_pb2.AdaptationCommand,
        outcome: leyline_pb2.FieldReportOutcome,
        metrics_delta: dict[str, float],
        *,
        training_run_id: str,
        seed_id: str,
        blueprint_id: str,
        notes: str | None = None,
    ) -> leyline_pb2.FieldReport:
        """Produce a field report entry for downstream ingestion (Simic)."""

        report = leyline_pb2.FieldReport(
            version=1,
            report_id=f"field-report-{len(self._field_reports)}",
            command_id=command.command_id,
            training_run_id=training_run_id,
            seed_id=seed_id,
            blueprint_id=blueprint_id,
            outcome=outcome,
            observation_window_epochs=1,
            tamiyo_policy_version=self._policy_version,
            notes=notes or "",
        )
        for key, value in metrics_delta.items():
            report.metrics[key] = value
        report.issued_at.CopyFrom(command.issued_at)
        self._field_report_store.append(report)
        self._field_reports = self._field_report_store.reports()
        return report

    def update_policy(self, new_policy: TamiyoPolicy) -> None:
        """Hot-swap the in-memory policy."""

        self._policy = new_policy

    def set_conservative_mode(self, enabled: bool) -> None:
        """Toggle conservative mode (breaker support)."""

        self._risk.conservative_mode = enabled

    @property
    def telemetry_packets(self) -> list[leyline_pb2.TelemetryPacket]:
        """Expose telemetry packets generated by Tamiyo."""

        return list(self._telemetry_packets)

    @property
    def field_reports(self) -> list[leyline_pb2.FieldReport]:
        """Return cached field reports for inspection/testing."""

        return list(self._field_reports)

    @property
    def policy_updates(self) -> list[leyline_pb2.PolicyUpdate]:
        """Return policy updates applied to the service."""

        return list(self._policy_updates)

    async def publish_history(self, oona: OonaClient) -> None:
        """Publish collected field reports and telemetry via Oona."""

        for report in self._field_reports:
            await oona.publish_field_report(report)
        for telemetry in self._telemetry_packets:
            await oona.publish_telemetry(telemetry)

    def ingest_policy_update(self, update: leyline_pb2.PolicyUpdate) -> None:
        """Apply a policy update produced by Simic."""

        if update.tamiyo_policy_version:
            self._policy_version = update.tamiyo_policy_version
        if update.payload:
            state_buffer = BytesIO(update.payload)
            state_dict = torch.load(state_buffer, map_location="cpu")
            try:
                self._policy.load_state_dict(state_dict, strict=False)
            except RuntimeError as exc:  # pragma: no cover - defensive
                logger.warning("Tamiyo policy update incompatible: %s", exc)
        self._policy_updates.append(update)

    async def consume_policy_updates(
        self,
        client: OonaClient,
        *,
        stream: str | None = None,
        count: int = 10,
        block_ms: int = 1000,
    ) -> None:
        """Consume policy updates from Oona and apply them."""

        async def handler(message: OonaMessage) -> None:
            update = leyline_pb2.PolicyUpdate()
            update.ParseFromString(message.payload)
            self.ingest_policy_update(update)

        await client.consume(
            handler,
            stream=stream or client.policy_stream,
            count=count,
            block_ms=block_ms,
        )

    def _resolve_blueprint_info(
        self, command: leyline_pb2.AdaptationCommand
    ) -> dict[str, float | str | bool | int] | None:
        if not self._urza:
            return None
        if command.command_type != leyline_pb2.COMMAND_SEED or not command.HasField("seed_operation"):
            return None
        blueprint_id = command.seed_operation.blueprint_id
        if not blueprint_id:
            return None

        cached = self._blueprint_cache.get(blueprint_id)
        now = datetime.now(tz=UTC)
        if cached:
            timestamp, data = cached
            if (now - timestamp) < self._metadata_cache_ttl:
                return data

        record = self._urza.get(blueprint_id)
        if record is None:
            return None
        data = {
            "tier": record.metadata.tier.value,
            "risk": float(record.metadata.risk),
            "stage": int(record.metadata.stage),
            "quarantine_only": bool(record.metadata.quarantine_only),
            "approval_required": bool(record.metadata.approval_required),
        }
        self._blueprint_cache[blueprint_id] = (now, data)
        return data


__all__ = ["TamiyoService", "RiskConfig"]
logger = logging.getLogger(__name__)
