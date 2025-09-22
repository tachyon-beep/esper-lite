"""Tamiyo service wrapper combining policy inference and risk gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Iterable, Callable
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


class TamiyoCircuitBreaker:
    """Minimal circuit breaker for Tamiyo components."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int = 3,
        cooldown_ms: float = 100.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_s = max(cooldown_ms, 0.0) / 1000.0
        self._clock = clock
        self._state = leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
        self._failure_count = 0
        self._open_until: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> int:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def record_failure(self, reason: str) -> TelemetryEvent:
        self._failure_count += 1
        attributes = {
            "component": self._name,
            "reason": reason,
            "failures": str(self._failure_count),
        }
        if (
            self._state != leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_OPEN
            and self._failure_count >= self._failure_threshold
        ):
            self._state = leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_OPEN
            self._open_until = self._clock() + self._cooldown_s
            attributes["action"] = "open"
            level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
        else:
            attributes["action"] = "count"
            level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
        return TelemetryEvent(
            description="breaker_event",
            level=level,
            attributes=attributes,
        )

    def record_success(self) -> TelemetryEvent | None:
        if self._state == leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_OPEN:
            if self._open_until is not None and self._clock() < self._open_until:
                return None
            self._state = leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
            self._failure_count = 0
            self._open_until = None
            return TelemetryEvent(
                description="breaker_event",
                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                attributes={"component": self._name, "action": "close"},
            )
        self._failure_count = max(0, self._failure_count - 1)
        return None


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
        self._inference_breaker = TamiyoCircuitBreaker(name="inference")
        self._metadata_breaker = TamiyoCircuitBreaker(name="metadata")

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

        training_metrics = dict(state.training_metrics)
        blueprint_info, risk_events = self._apply_risk_engine(
            command,
            state=state,
            loss_delta=loss_delta,
            blueprint_info=blueprint_info,
            blueprint_timeout=blueprint_timeout,
            timed_out=timed_out,
            training_metrics=training_metrics,
        )
        events.extend(risk_events)

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
        metrics.append(
            TelemetryMetric(
                "tamiyo.breaker.inference_state",
                float(self._inference_breaker.state),
                unit="state",
            )
        )
        metrics.append(
            TelemetryMetric(
                "tamiyo.breaker.metadata_state",
                float(self._metadata_breaker.state),
                unit="state",
            )
        )
        metrics.append(
            TelemetryMetric(
                "tamiyo.breaker.inference_failures",
                float(self._inference_breaker.failure_count),
                unit="count",
            )
        )
        metrics.append(
            TelemetryMetric(
                "tamiyo.breaker.metadata_failures",
                float(self._metadata_breaker.failure_count),
                unit="count",
            )
        )
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
        if not timed_out:
            self._emit_field_report(command, state, loss_delta, events)
        self._last_validation_loss = state.validation_loss
        return command

    def _emit_field_report(
        self,
        command: leyline_pb2.AdaptationCommand,
        state: leyline_pb2.SystemStatePacket,
        loss_delta: float,
        events: Iterable[TelemetryEvent],
    ) -> None:
        events_list = list(events)
        metrics_delta = {"loss_delta": loss_delta}
        self.generate_field_report(
            command=command,
            outcome=leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS,
            metrics_delta=metrics_delta,
            training_run_id=state.training_run_id or "run-unknown",
            seed_id=command.target_seed_id,
            blueprint_id=command.seed_operation.blueprint_id if command.HasField("seed_operation") else "",
            observation_window_epochs=max(1, int(state.current_epoch) if state.current_epoch else 1),
            notes=events_list[-1].description if events_list else None,
        )

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

    def _set_conservative_mode(
        self,
        enabled: bool,
        reason: str,
        events: list[TelemetryEvent],
    ) -> None:
        if enabled and not self._risk.conservative_mode:
            self._risk.conservative_mode = True
            events.append(
                TelemetryEvent(
                    description="conservative_entered",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"reason": reason},
                )
            )
        elif not enabled and self._risk.conservative_mode:
            self._risk.conservative_mode = False
            events.append(
                TelemetryEvent(
                    description="conservative_exited",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                    attributes={"reason": reason},
                )
            )

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

    def _apply_risk_engine(
        self,
        command: leyline_pb2.AdaptationCommand,
        *,
        state: leyline_pb2.SystemStatePacket,
        loss_delta: float,
        blueprint_info: Optional[dict[str, float | str | bool | int]],
        blueprint_timeout: bool,
        timed_out: bool,
        training_metrics: dict[str, float],
    ) -> tuple[Optional[dict[str, float | str | bool | int]], list[TelemetryEvent]]:
        events: list[TelemetryEvent] = []
        reason = command.annotations.get("risk_reason")

        if self._risk.conservative_mode and not timed_out:
            reason = reason or "conservative_mode"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="pause_triggered",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"reason": "conservative_mode"},
                )
            )

        if timed_out:
            events.append(self._inference_breaker.record_failure("timeout_inference"))
            self._set_conservative_mode(True, "timeout_inference", events)
            reason = reason or "timeout_inference"
        else:
            success_event = self._inference_breaker.record_success()
            if success_event:
                events.append(success_event)

        if blueprint_timeout:
            events.append(self._metadata_breaker.record_failure("timeout_urza"))
            reason = reason or "timeout_urza"
        else:
            success_event = self._metadata_breaker.record_success()
            if success_event:
                events.append(success_event)

        if blueprint_info:
            command.annotations.setdefault("blueprint_tier", blueprint_info["tier"])
            command.annotations.setdefault("blueprint_stage", str(blueprint_info["stage"]))
            command.annotations.setdefault("blueprint_risk", f"{blueprint_info['risk']:.2f}")
            risk_score = float(blueprint_info["risk"])
            if blueprint_info.get("quarantine_only") or risk_score >= 0.8:
                events.append(
                    TelemetryEvent(
                        description="bp_quarantine",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                        attributes={"tier": str(blueprint_info["tier"])},
                    )
                )
                reason = "bp_quarantine"
                command.command_type = leyline_pb2.COMMAND_PAUSE
                self._set_conservative_mode(True, "bp_quarantine", events)
            elif risk_score >= 0.5 and command.command_type == leyline_pb2.COMMAND_SEED:
                reason = reason or "blueprint_risk"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"
                events.append(
                    TelemetryEvent(
                        description="blueprint_risk",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"risk": f"{risk_score:.2f}"},
                    )
                )

        if not timed_out and loss_delta > self._risk.max_loss_spike:
            reason = "loss_spike"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="loss_spike",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"delta": f"{loss_delta:.3f}"},
                )
            )
            self._set_conservative_mode(True, "loss_spike", events)
        elif (
            not timed_out
            and loss_delta > self._risk.max_loss_spike * 0.5
            and command.command_type == leyline_pb2.COMMAND_SEED
        ):
            reason = reason or "loss_warning"
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER
            command.optimizer_adjustment.optimizer_id = "sgd"
            events.append(
                TelemetryEvent(
                    description="loss_warning",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"delta": f"{loss_delta:.3f}"},
                )
            )

        hook_latency = training_metrics.get("hook_latency_ms")
        if hook_latency and hook_latency > 50.0:
            reason = "hook_budget"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="hook_budget",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency:.2f}"},
                )
            )

        isolation = training_metrics.get("kasmina.isolation.violations")
        if isolation and isolation > 0:
            reason = reason or "isolation_violation"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="isolation_violations",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"violations": str(int(isolation))},
                )
            )

        if not reason and command.command_type == leyline_pb2.COMMAND_PAUSE and self._risk.conservative_mode:
            reason = "conservative_mode"

        if (
            not timed_out
            and not blueprint_timeout
            and self._inference_breaker.state == leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
            and self._metadata_breaker.state == leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
        ):
            self._set_conservative_mode(False, "stabilised", events)

        if reason:
            command.annotations["risk_reason"] = reason
        elif "risk_reason" not in command.annotations:
            command.annotations["risk_reason"] = "policy"

        return blueprint_info, events

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
        observation_window_epochs: int = 1,
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
            observation_window_epochs=max(1, observation_window_epochs),
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
            priority_name = telemetry.system_health.indicators.get("priority")
            priority_enum = None
            if priority_name:
                with contextlib.suppress(ValueError):
                    priority_enum = leyline_pb2.MessagePriority.Value(priority_name)
            await oona.publish_telemetry(telemetry, priority=priority_enum)

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
