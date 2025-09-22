"""Tamiyo service wrapper combining policy inference and risk gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Iterable
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from uuid import uuid4

import torch
import logging
import contextlib

from esper.core import EsperSettings, TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.leyline import leyline_pb2
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext, sign
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
        signature_context: SignatureContext | None = None,
        step_timeout_ms: float = 5.0,
        metadata_timeout_ms: float = 20.0,
        executor: ThreadPoolExecutor | None = None,
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
        self._signing_context = signature_context or SignatureContext.from_environment(DEFAULT_SECRET_ENV)
        self._executor = executor
        self._owns_executor = False
        if self._executor is None and (step_timeout_ms > 0 or metadata_timeout_ms > 0):
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tamiyo")
            self._owns_executor = True
        self._step_timeout_s = max(step_timeout_ms, 0.0) / 1000.0
        self._metadata_timeout_s = max(metadata_timeout_ms, 0.0) / 1000.0

    def evaluate_step(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Evaluate step state under tight deadlines (ADR-001 3A)."""

        return self._evaluate(state, enforce_timeouts=True)

    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        """Retained for backwards compatibility (no timeouts)."""

        return self._evaluate(state, enforce_timeouts=False)

    def _evaluate(
        self,
        state: leyline_pb2.SystemStatePacket,
        *,
        enforce_timeouts: bool,
    ) -> leyline_pb2.AdaptationCommand:
        events: list[TelemetryEvent] = []

        command, inference_ms, timed_out = self._run_policy(state, enforce_timeouts)
        if timed_out:
            events.append(
                TelemetryEvent(
                    description="timeout_inference",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"budget_ms": f"{self._step_timeout_s * 1000.0:.1f}"},
                )
            )

        loss_delta = 0.0
        if self._last_validation_loss is not None:
            loss_delta = state.validation_loss - self._last_validation_loss

        blueprint_info: Optional[dict[str, float | str | bool | int]] = None
        blueprint_timeout = False
        if not timed_out:
            blueprint_info, blueprint_timeout = self._resolve_blueprint_with_timeout(
                command,
                enforce_timeouts,
            )
        if blueprint_timeout:
            events.append(
                TelemetryEvent(
                    description="timeout_urza",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"budget_ms": f"{self._metadata_timeout_s * 1000.0:.1f}"},
                )
            )

        if blueprint_info:
            command.annotations["blueprint_tier"] = blueprint_info["tier"]
            command.annotations["blueprint_stage"] = str(blueprint_info["stage"])
            command.annotations["blueprint_risk"] = f"{blueprint_info['risk']:.2f}"
            if blueprint_info["quarantine_only"] or blueprint_info["risk"] >= 0.8:
                command.command_type = leyline_pb2.COMMAND_PAUSE
                command.annotations.setdefault("risk_reason", "blueprint_high_risk")
                events.append(
                    TelemetryEvent(
                        description="bp_quarantine",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"tier": blueprint_info["tier"]},
                    )
                )

        if not timed_out and (
            self._risk.conservative_mode
            or (
                self._last_validation_loss is not None
                and loss_delta > self._risk.max_loss_spike
            )
        ):
            command.command_type = leyline_pb2.COMMAND_PAUSE
            reason = "conservative_mode" if self._risk.conservative_mode else "loss_spike"
            command.annotations["risk_reason"] = reason
            events.append(
                TelemetryEvent(
                    description="pause_triggered",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "loss_delta": f"{loss_delta:.3f}",
                        "conservative": str(self._risk.conservative_mode).lower(),
                    },
                )
            )

        if timed_out:
            command.annotations.setdefault("risk_reason", "timeout_inference")

        metrics = [
            TelemetryMetric("tamiyo.validation_loss", state.validation_loss, unit="loss"),
            TelemetryMetric("tamiyo.loss_delta", loss_delta, unit="loss"),
            TelemetryMetric(
                "tamiyo.conservative_mode",
                1.0 if self._risk.conservative_mode else 0.0,
                unit="bool",
            ),
            TelemetryMetric("tamiyo.inference.latency_ms", inference_ms, unit="ms"),
        ]
        if blueprint_info:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blueprint.risk",
                    float(blueprint_info["risk"]),
                    unit="score",
                )
            )

        telemetry = build_telemetry_packet(
            packet_id=state.packet_id or "tamiyo-telemetry",
            source="tamiyo",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=self._derive_health_status(command, events),
            health_summary=self._derive_health_summary(command, events, loss_delta),
            health_indicators=self._build_health_indicators(state, loss_delta, blueprint_info),
        )
        priority_value = self._priority_from_events(events)
        telemetry.system_health.indicators["priority"] = leyline_pb2.MessagePriority.Name(priority_value)
        self._sign_command(command)
        self._telemetry_packets.append(telemetry)
        self._last_validation_loss = state.validation_loss
        return command

    def _run_policy(
        self,
        state: leyline_pb2.SystemStatePacket,
        enforce_timeouts: bool,
    ) -> tuple[leyline_pb2.AdaptationCommand, float, bool]:
        start = time.perf_counter()
        timed_out = False

        if enforce_timeouts and self._step_timeout_s > 0 and self._executor is not None:
            future = self._executor.submit(self._policy.select_action, state)
            try:
                command = future.result(timeout=self._step_timeout_s)
            except FuturesTimeout:
                future.cancel()
                timed_out = True
                command = self._build_timeout_command("timeout_inference")
        else:
            command = self._policy.select_action(state)

        inference_ms = (time.perf_counter() - start) * 1000.0

        if timed_out:
            command.annotations.setdefault("policy_action", "timeout")
            command.annotations.setdefault("policy_param_delta", "0.0")
        else:
            self._ensure_policy_annotations(command)

        if not command.issued_by:
            command.issued_by = "tamiyo"

        return command, inference_ms, timed_out

    def _ensure_policy_annotations(self, command: leyline_pb2.AdaptationCommand) -> None:
        last_action = self._policy.last_action
        command.annotations.setdefault("policy_action", str(int(last_action.get("action", 0.0))))
        command.annotations.setdefault(
            "policy_param_delta",
            f"{last_action.get('param_delta', 0.0):.6f}",
        )

    def _build_timeout_command(self, reason: str) -> leyline_pb2.AdaptationCommand:
        command = leyline_pb2.AdaptationCommand(
            version=1,
            command_type=leyline_pb2.COMMAND_PAUSE,
            issued_by="tamiyo",
        )
        command.annotations["policy_action"] = "timeout"
        command.annotations["policy_param_delta"] = "0.0"
        command.annotations["risk_reason"] = reason
        return command

    def _resolve_blueprint_with_timeout(
        self,
        command: leyline_pb2.AdaptationCommand,
        enforce_timeouts: bool,
    ) -> tuple[Optional[dict[str, float | str | bool | int]], bool]:
        if self._urza is None:
            return None, False
        if command.command_type != leyline_pb2.COMMAND_SEED or not command.HasField("seed_operation"):
            return None, False

        if enforce_timeouts and self._metadata_timeout_s > 0 and self._executor is not None:
            future = self._executor.submit(self._resolve_blueprint_info, command)
            try:
                return future.result(timeout=self._metadata_timeout_s), False
            except FuturesTimeout:
                future.cancel()
                return None, True
        return self._resolve_blueprint_info(command), False

    @staticmethod
    def _priority_from_events(events: Iterable[TelemetryEvent]) -> int:
        priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        for event in events:
            if event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL:
                return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
            if event.level in (
                leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
            ):
                priority = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
        return priority

    def _sign_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Assign identifiers and attach an HMAC signature."""

        if "signature" in command.annotations:
            del command.annotations["signature"]
        command.command_id = str(uuid4())
        command.issued_at.GetCurrentTime()
        payload = command.SerializeToString()
        command.annotations["signature"] = sign(payload, self._signing_context)

    def _derive_health_status(
        self,
        command: leyline_pb2.AdaptationCommand,
        events: list[TelemetryEvent],
    ) -> leyline_pb2.HealthStatus:
        if any(event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL for event in events):
            return leyline_pb2.HealthStatus.HEALTH_STATUS_CRITICAL
        if command.command_type == leyline_pb2.COMMAND_PAUSE:
            return leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
        if any(event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING for event in events):
            return leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
        return leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY

    def _derive_health_summary(
        self,
        command: leyline_pb2.AdaptationCommand,
        events: list[TelemetryEvent],
        loss_delta: float,
    ) -> str:
        if command.command_type == leyline_pb2.COMMAND_PAUSE:
            reason = command.annotations.get("risk_reason", "pause")
            if reason == "blueprint_high_risk":
                return "bp_high_risk"
            return reason
        if events:
            return events[-1].description
        if loss_delta > self._risk.max_loss_spike:
            return "loss_spike"
        return "stable"

    def _build_health_indicators(
        self,
        state: leyline_pb2.SystemStatePacket,
        loss_delta: float,
        blueprint_info: dict[str, float | str | bool | int] | None,
    ) -> dict[str, str]:
        indicators = {
            "policy": self._policy_version[:8],
            "mode": "1" if self._risk.conservative_mode else "0",
        }
        if blueprint_info:
            tier = blueprint_info.get("tier")
            if tier is not None:
                indicators["tier"] = str(tier)
        step_index = getattr(state, "global_step", 0)
        if step_index:
            indicators["step_index"] = str(step_index)
        return indicators

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

    def close(self) -> None:
        """Release internal executor resources if owned."""

        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._owns_executor = False
            self._executor = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        with contextlib.suppress(Exception):
            self.close()

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
        tier_name = leyline_pb2.BlueprintTier.Name(record.metadata.tier)
        data = {
            "tier": tier_name,
            "risk": float(record.metadata.risk),
            "stage": int(record.metadata.stage),
            "quarantine_only": bool(record.metadata.quarantine_only),
            "approval_required": bool(record.metadata.approval_required),
            "description": record.metadata.description,
        }
        self._blueprint_cache[blueprint_id] = (now, data)
        return data


__all__ = ["TamiyoService", "RiskConfig"]
logger = logging.getLogger(__name__)
