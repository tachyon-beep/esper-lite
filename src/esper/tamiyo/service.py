"""Tamiyo service wrapper combining policy inference and risk gating."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch

from esper.core import (
    AsyncTimeoutError,
    AsyncWorker,
    DependencyContext,
    DependencyViolationError,
    EsperSettings,
    TelemetryEvent,
    TelemetryMetric,
    build_telemetry_packet,
    ensure_present,
)
from esper.leyline import leyline_pb2
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext, sign
from esper.urza import UrzaLibrary

from .persistence import FieldReportStore, FieldReportStoreConfig, _atomic_write_json, _load_json
from .policy import TamiyoPolicy, TamiyoPolicyConfig

if TYPE_CHECKING:
    from esper.oona import OonaClient, OonaMessage


@dataclass(slots=True)
class RiskConfig:
    """Configuration for Tamiyo risk thresholds."""

    max_loss_spike: float = 0.15
    conservative_mode: bool = False
    # Latency/heuristic thresholds (ms) — tune per hardware profile
    step_latency_high_ms: float = 120.0
    kasmina_apply_slow_ms: float = 30.0
    kasmina_finalize_slow_ms: float = 30.0
    hook_budget_ms: float = 50.0


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
        step_timeout_ms: float = 15.0,
        metadata_timeout_ms: float = 25.0,
        async_worker: AsyncWorker | None = None,
    ) -> None:
        # Resolve settings early for device/compile detection
        self._settings = settings or EsperSettings()
        if policy is None:
            # Build policy config with device auto-detect and compile override
            p_cfg = TamiyoPolicyConfig()
            # Device selection: explicit via settings, else prefer CUDA if available
            try:
                prefer = getattr(self._settings, "tamiyo_device", None)
                if prefer:
                    p_cfg.device = str(prefer)
                else:
                    try:
                        if torch.cuda.is_available():
                            p_cfg.device = "cuda"
                    except Exception:
                        pass
            except Exception:  # pragma: no cover - defensive
                pass
            # Compile override: explicit via settings; else default ON when device is CUDA
            try:
                enable = getattr(self._settings, "tamiyo_enable_compile", None)
                if enable is not None:
                    p_cfg.enable_compile = bool(enable)
                else:
                    if str(p_cfg.device).lower().startswith("cuda"):
                        p_cfg.enable_compile = True
            except Exception:  # pragma: no cover - defensive
                pass
            self._policy = TamiyoPolicy(p_cfg)
        else:
            self._policy = policy
        self._risk = risk_config or RiskConfig()
        # Urza is required for prototype operation; do not mask its absence.
        if urza is None:
            raise RuntimeError(
                "TamiyoService requires an UrzaLibrary instance (urza=...) to operate"
            )
        self._urza = urza
        self._metadata_cache_ttl = metadata_cache_ttl
        self._blueprint_cache: dict[str, tuple[datetime, dict[str, float | str | bool | int]]] = {}
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
        # Retry hedging for field-report publishing (in-memory only)
        self._report_retry_count: dict[str, int] = {}
        try:
            self._max_report_retries = int(
                getattr(self._settings, "tamiyo_field_report_max_retries", 3)
            )
        except Exception:
            self._max_report_retries = 3
        self._last_validation_loss: float | None = None
        self._policy_version = getattr(self._policy, "architecture_version", "policy-stub")
        self._signing_context = signature_context or SignatureContext.from_environment(
            DEFAULT_SECRET_ENV
        )
        self._worker = async_worker
        self._owns_worker = False
        if self._worker is None and (step_timeout_ms > 0 or metadata_timeout_ms > 0):
            self._worker = AsyncWorker(max_concurrency=4, name="tamiyo-worker")
            self._owns_worker = True
        # Step timeout defaulting: if caller did not explicitly override the legacy
        # default (15.0 ms), prefer the settings-driven default to align with the
        # timeout matrix (2–5 ms). Explicit non-default values always take precedence.
        try:
            settings_default_ms = float(self._settings.tamiyo_step_timeout_ms)
        except Exception:
            settings_default_ms = 5.0
        if step_timeout_ms == 15.0:
            step_timeout_ms = settings_default_ms
        self._step_timeout_s = max(step_timeout_ms, 0.0) / 1000.0
        self._metadata_timeout_s = max(metadata_timeout_ms, 0.0) / 1000.0
        self._inference_breaker = TamiyoCircuitBreaker(name="inference")
        self._metadata_breaker = TamiyoCircuitBreaker(name="metadata")
        # P9 — Observation windows and durable retry index sidecars
        try:
            self._obs_window_epochs = max(
                1, int(getattr(self._settings, "tamiyo_fr_obs_window_epochs", 1))
            )
        except Exception:
            self._obs_window_epochs = 1
        sidecar_dir = self._field_report_store.path.parent
        self._retry_index_path = sidecar_dir / "field_reports.index.json"
        self._windows_path = sidecar_dir / "field_reports.windows.json"
        self._retry_index: dict[str, dict[str, object]] = _load_json(self._retry_index_path)
        self._windows: dict[str, dict[str, object]] = _load_json(self._windows_path)
        # Keep sidecars small: ensure shapes are dicts
        if not isinstance(self._retry_index, dict):
            self._retry_index = {}
        if not isinstance(self._windows, dict):
            self._windows = {}

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

        self._ensure_blueprint_metadata_for_packet(state)
        if hasattr(self._policy, "update_blueprint_metadata") and self._blueprint_cache:
            metadata_payload = {bp: info for bp, (_, info) in self._blueprint_cache.items()}
            with contextlib.suppress(Exception):
                self._policy.update_blueprint_metadata(metadata_payload)

        command, inference_ms, timed_out = self._run_policy(state, enforce_timeouts)
        if state.training_run_id:
            command.annotations["training_run_id"] = state.training_run_id
        if timed_out:
            events.append(
                TelemetryEvent(
                    description="timeout_inference",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"budget_ms": f"{self._step_timeout_s * 1000.0:.1f}"},
                )
            )
            # Ensure downstream consumers have typed coverage field present (even if empty)
            command.annotations.setdefault("coverage_types", "{}")
            command.annotations.setdefault("feature_coverage", "0.0")

        loss_delta = 0.0
        if self._last_validation_loss is not None:
            loss_delta = state.validation_loss - self._last_validation_loss

        blueprint_info: dict[str, float | str | bool | int] | None = None
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
        # If Tolaria supplied a precomputed loss_delta (WP8.5), prefer it when
        # we lack a prior validation snapshot. This keeps step-level risk gates
        # responsive even on the first invocation.
        if (self._last_validation_loss is None) and ("loss_delta" in training_metrics):
            try:
                loss_delta = float(training_metrics.get("loss_delta", loss_delta))
            except Exception:
                pass
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

        if not timed_out:
            self._validate_command_dependencies(command)

        # WP12: Degraded-input routing based on feature coverage thresholds
        try:
            cov = getattr(self._policy, "feature_coverage", {})
            if cov:
                vals = []
                for v in cov.values():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv > 0.0:
                        vals.append(fv)
                if vals:
                    avg_cov = float(sum(vals) / max(1, len(vals)))
                warn_th = (
                    getattr(self._risk, "degraded_inputs_warn", 0.30)
                    if hasattr(self._risk, "degraded_inputs_warn")
                    else 0.30
                )
                crit_th = (
                    getattr(self._risk, "degraded_inputs_crit", 0.10)
                    if hasattr(self._risk, "degraded_inputs_crit")
                    else 0.10
                )
                evt_level = None
                if avg_cov <= crit_th:
                    evt_level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
                elif avg_cov < warn_th:
                    evt_level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
                if vals and evt_level is not None:
                    events.append(
                        TelemetryEvent(
                            description="degraded_inputs",
                            level=evt_level,
                            attributes={
                                "coverage_avg": f"{avg_cov:.3f}",
                                "missing_features": str(sum(1 for v in cov.values() if not v)),
                            },
                        )
                    )
                    # If no prior reason set by risk engine, annotate degraded inputs
                    if (
                        evt_level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL
                        and "risk_reason" not in command.annotations
                    ):
                        command.annotations["risk_reason"] = "degraded_inputs"
        except Exception:
            pass

        compile_reason = getattr(self._policy, "compile_disabled_reason", None)
        if compile_reason:
            desc = (
                "compile_disabled_cpu"
                if compile_reason == "device_not_cuda"
                else "compile_disabled"
            )
            level = (
                leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
                if compile_reason in {"device_not_cuda", "cuda_unavailable"}
                else leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
            )
            device_str = "cpu"
            try:
                device_str = str(
                    getattr(self._policy, "device", getattr(self._policy, "_device", "unknown"))
                )
            except Exception:
                device_str = "unknown"
            events.append(
                TelemetryEvent(
                    description=desc,
                    level=level,
                    attributes={
                        "reason": compile_reason,
                        "device": device_str,
                    },
                )
            )
            command.annotations.setdefault("policy_compile_reason", compile_reason)

        last_action = self._policy.last_action
        metrics = [
            TelemetryMetric("tamiyo.validation_loss", state.validation_loss, unit="loss"),
            TelemetryMetric("tamiyo.loss_delta", loss_delta, unit="loss"),
            TelemetryMetric(
                "tamiyo.conservative_mode",
                1.0 if self._risk.conservative_mode else 0.0,
                unit="bool",
            ),
            TelemetryMetric("tamiyo.inference.latency_ms", inference_ms, unit="ms"),
            TelemetryMetric(
                "tamiyo.gnn.inference.latency_ms",
                inference_ms,
                unit="ms",
            ),
            TelemetryMetric(
                "tamiyo.gnn.compile_enabled",
                1.0 if getattr(self._policy, "compile_enabled", False) else 0.0,
                unit="bool",
            ),
        ]
        # Append warm-up latency metric if measured
        try:
            warm_ms = float(getattr(self._policy, "compile_warm_ms", 0.0))
            if warm_ms >= 0.0:
                metrics.append(
                    TelemetryMetric(
                        "tamiyo.gnn.compile_warm_ms",
                        warm_ms,
                        unit="ms",
                    )
                )
        except Exception:
            pass
        # Append compile fallback counter if any
        try:
            fallbacks = int(getattr(self._policy, "compile_fallbacks", 0))
        except Exception:
            fallbacks = 0
        if fallbacks > 0:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.gnn.compile_fallback_total",
                    float(fallbacks),
                    unit="count",
                )
            )
        metrics.extend(
            [
                TelemetryMetric(
                    "tamiyo.policy.value_estimate",
                    float(last_action.get("value_estimate", 0.0)),
                    unit="score",
                ),
                TelemetryMetric(
                    "tamiyo.policy.risk_index",
                    float(last_action.get("risk_index", 0.0)),
                    unit="index",
                ),
                TelemetryMetric(
                    "tamiyo.policy.risk_score",
                    float(last_action.get("risk_score", 0.0)),
                    unit="score",
                ),
            ]
        )
        coverage = getattr(self._policy, "feature_coverage", {})
        if coverage:
            average_coverage = float(sum(coverage.values()) / max(1, len(coverage)))
            metrics.append(
                TelemetryMetric(
                    "tamiyo.gnn.feature_coverage",
                    average_coverage,
                    unit="ratio",
                )
            )
            command.annotations.setdefault("feature_coverage", f"{average_coverage:.3f}")
            # WP15: emit granular coverage by type and attach full map
            try:
                # Group keys by type prefix
                groups: dict[str, list[float]] = {
                    "node.seed": [],
                    "node.layer": [],
                    "node.activation": [],
                    "node.parameter": [],
                    "node.blueprint": [],
                    "node.global": [],
                    "edges.layer_connects": [],
                    "edges.seed_monitors": [],
                    "edges.layer_feeds": [],
                }
                for key, val in coverage.items():
                    if key.startswith("seed."):
                        groups["node.seed"].append(float(val))
                    elif key.startswith("layer."):
                        groups["node.layer"].append(float(val))
                    elif key.startswith("activation."):
                        groups["node.activation"].append(float(val))
                    elif key.startswith("parameter."):
                        groups["node.parameter"].append(float(val))
                    elif key.startswith("blueprint."):
                        groups["node.blueprint"].append(float(val))
                    elif key.startswith("global."):
                        groups["node.global"].append(float(val))
                    elif key == "edges.layer_connects":
                        groups["edges.layer_connects"].append(float(val))
                    elif key == "edges.seed_monitors":
                        groups["edges.seed_monitors"].append(float(val))
                    elif key == "edges.layer_feeds":
                        groups["edges.layer_feeds"].append(float(val))
                # Prefer builder-provided typed coverage when available
                per_type: dict[str, float] = {}
                try:
                    per_type = dict(getattr(self._policy, "feature_coverage_types", {}) or {})
                except Exception:
                    per_type = {}
                if not per_type:
                    for gkey, arr in groups.items():
                        if arr:
                            per_type[gkey] = float(sum(arr) / max(1, len(arr)))
                # Ensure layer connectivity coverage key is present (defaults to 0.0)
                per_type.setdefault("edges.layer_connects", 0.0)
                # Emit telemetry metrics for each type
                for gkey, ratio in per_type.items():
                    metrics.append(
                        TelemetryMetric(
                            f"tamiyo.gnn.feature_coverage.{gkey}",
                            ratio,
                            unit="ratio",
                        )
                    )
                # Attach full map + types to command annotations (bounded)
                cov_json = json.dumps(coverage)
                types_json = json.dumps(per_type)
                # Size guard ~ 1KB per field
                if len(cov_json) <= 1024:
                    command.annotations.setdefault("coverage_map", cov_json)
                command.annotations.setdefault("coverage_types", types_json)
            except Exception:  # pragma: no cover - defensive
                pass
        # Ensure layer connectivity coverage metric exists for dashboards (defaults to 0 when unavailable)
        try:
            names = {m.name for m in metrics}
            target_name = "tamiyo.gnn.feature_coverage.edges.layer_connects"
            if target_name not in names:
                metrics.append(TelemetryMetric(target_name, 0.0, unit="ratio"))
        except Exception:
            pass
        if "blending_method" in last_action:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blending.method_index",
                    float(last_action.get("blending_index", 0.0)),
                    unit="enum",
                    attributes={"method": str(last_action.get("blending_method", ""))},
                )
            )
        if "blending_schedule_start" in last_action and "blending_schedule_end" in last_action:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blending.schedule_start",
                    float(last_action.get("blending_schedule_start", 0.0)),
                    unit="fraction",
                )
            )
            metrics.append(
                TelemetryMetric(
                    "tamiyo.blending.schedule_end",
                    float(last_action.get("blending_schedule_end", 0.0)),
                    unit="fraction",
                )
            )
        if "selected_seed_index" in last_action:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.selection.seed_index",
                    float(last_action.get("selected_seed_index", -1.0)),
                    unit="index",
                )
            )
        if "selected_seed_score" in last_action:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.selection.seed_score",
                    float(last_action.get("selected_seed_score", 0.0)),
                    unit="score",
                )
            )
        if "selected_blueprint_index" in last_action:
            metrics.append(
                TelemetryMetric(
                    "tamiyo.selection.blueprint_index",
                    float(last_action.get("selected_blueprint_index", -1.0)),
                    unit="index",
                )
            )
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
        telemetry.system_health.indicators["priority"] = leyline_pb2.MessagePriority.Name(
            priority_value
        )
        self._sign_command(command)
        self._telemetry_packets.append(telemetry)
        # Emit a field report for every decision, including timeouts (neutral outcome).
        self._emit_field_report(command, state, loss_delta, events, timed_out=timed_out)
        # Update/synthesise observation windows (non-blocking bookkeeping)
        try:
            self._update_observation_windows(state, command, events, loss_delta=loss_delta)
            self._synthesise_due_windows()
        except Exception:
            pass
        self._last_validation_loss = state.validation_loss
        return command

    def _emit_field_report(
        self,
        command: leyline_pb2.AdaptationCommand,
        state: leyline_pb2.SystemStatePacket,
        loss_delta: float,
        events: Iterable[TelemetryEvent],
        *,
        timed_out: bool,
    ) -> None:
        # Build a compact metrics delta. Always include loss_delta. Optionally include
        # step/timing metrics if present on the packet to aid downstream analysis.
        events_list = list(events)
        metrics_delta: dict[str, float] = {"loss_delta": float(loss_delta)}
        last_action = self._policy.last_action
        if "param_delta" in last_action:
            metrics_delta["param_delta"] = float(last_action.get("param_delta", 0.0))
        if "value_estimate" in last_action:
            metrics_delta["value_estimate"] = float(last_action.get("value_estimate", 0.0))
        if "risk_index" in last_action:
            metrics_delta["risk_index"] = float(last_action.get("risk_index", 0.0))
        if "blending_index" in last_action:
            metrics_delta["blending_index"] = float(last_action.get("blending_index", 0.0))
        if "risk_score" in last_action:
            metrics_delta["risk_score"] = float(last_action.get("risk_score", 0.0))
        if "selected_seed_score" in last_action:
            metrics_delta["selected_seed_score"] = float(
                last_action.get("selected_seed_score", 0.0)
            )
        if "blending_schedule_start" in last_action:
            metrics_delta["blending_schedule_start"] = float(
                last_action.get("blending_schedule_start", 0.0)
            )
        if "blending_schedule_end" in last_action:
            metrics_delta["blending_schedule_end"] = float(
                last_action.get("blending_schedule_end", 0.0)
            )
        # Pull auxiliary timings from the training metrics if available
        try:
            tm = dict(state.training_metrics)
            if "tamiyo_latency_ms" in tm:
                metrics_delta["tamiyo_latency_ms"] = float(tm["tamiyo_latency_ms"])
            if "hook_latency_ms" in tm:
                metrics_delta["hook_latency_ms"] = float(tm["hook_latency_ms"])
        except Exception:
            pass

        # Derive a coarse outcome from the risk reason and command type
        reason = command.annotations.get("risk_reason", "")
        outcome = leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS
        if timed_out or reason in {"timeout_inference", "timeout_urza"}:
            outcome = leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL
        elif reason in {"bp_quarantine", "loss_spike", "isolation_violation", "hook_budget"}:
            outcome = leyline_pb2.FIELD_REPORT_OUTCOME_REGRESSION
        elif command.command_type == leyline_pb2.COMMAND_PAUSE and reason in {
            "conservative_mode",
            "policy",
            "",
        }:
            outcome = leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL

        # If blueprint risk is annotated, surface it in the metrics delta for Simic
        try:
            if "blueprint_risk" in command.annotations:
                metrics_delta["blueprint_risk"] = float(command.annotations["blueprint_risk"])  # type: ignore[arg-type]
        except Exception:
            pass

        # Prefer a timeout-related note if present; otherwise use the last event description
        note_text = None
        for ev in events_list:
            try:
                if ev.description and ev.description.startswith("timeout"):
                    note_text = ev.description
                    break
            except Exception:
                pass
        if note_text is None and events_list:
            note_text = events_list[-1].description
        if note_text is None and "blending_method" in last_action:
            note_text = f"blending:{last_action['blending_method']}"
        self.generate_field_report(
            command=command,
            outcome=outcome,
            metrics_delta=metrics_delta,
            training_run_id=state.training_run_id or "run-unknown",
            seed_id=command.target_seed_id,
            blueprint_id=(
                command.seed_operation.blueprint_id if command.HasField("seed_operation") else ""
            ),
            observation_window_epochs=1,
            notes=note_text,
        )

    # -------------------------
    # P9: Observation windows
    # -------------------------

    def _update_observation_windows(
        self,
        state: leyline_pb2.SystemStatePacket,
        command: leyline_pb2.AdaptationCommand,
        events: Iterable[TelemetryEvent],
        *,
        loss_delta: float,
    ) -> None:
        # Initialise a window for the newly issued command
        cmd_id = command.command_id or ""
        if cmd_id and cmd_id not in self._windows and self._obs_window_epochs > 1:
            # Prepare immutable command metadata for synthesis
            bp_id = (
                command.seed_operation.blueprint_id if command.HasField("seed_operation") else ""
            )
            issued_iso = None
            try:
                issued_dt = command.issued_at.ToDatetime()  # type: ignore[attr-defined]
                if issued_dt:
                    issued_iso = issued_dt.isoformat()
            except Exception:
                issued_iso = None
            self._windows[cmd_id] = {
                "seed_id": command.target_seed_id,
                "blueprint_id": bp_id,
                "training_run_id": state.training_run_id or "run-unknown",
                "policy_version": self._policy_version,
                "start_epoch": int(getattr(state, "current_epoch", 0) or 0),
                "collected": 0,
                "target": int(self._obs_window_epochs),
                "sum_loss_delta": 0.0,
                "min_loss_delta": float("inf"),
                "max_loss_delta": float("-inf"),
                "sum_hook_latency_ms": 0.0,
                "count_hook_latency": 0,
                "last_reason": "",
                "has_critical": False,
                "issued_at_iso": issued_iso or "",
            }
        # Aggregate this step across all active windows
        if not self._windows:
            return
        # Gather per-step context
        try:
            tm = dict(state.training_metrics)
            hook_ms = float(tm.get("hook_latency_ms", 0.0))
        except Exception:
            hook_ms = 0.0
        last_reason = None
        has_critical = False
        for ev in events:
            try:
                last_reason = ev.description or last_reason
                if ev.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL:
                    has_critical = True
            except Exception:
                continue
        for _, win in list(self._windows.items()):
            try:
                win["collected"] = int(win.get("collected", 0)) + 1
                # Update aggregates
                s = float(win.get("sum_loss_delta", 0.0)) + float(loss_delta)
                win["sum_loss_delta"] = s
                mn = float(win.get("min_loss_delta", float("inf")))
                mx = float(win.get("max_loss_delta", float("-inf")))
                if loss_delta < mn:
                    win["min_loss_delta"] = float(loss_delta)
                if loss_delta > mx:
                    win["max_loss_delta"] = float(loss_delta)
                if hook_ms > 0.0:
                    win["sum_hook_latency_ms"] = (
                        float(win.get("sum_hook_latency_ms", 0.0)) + hook_ms
                    )
                    win["count_hook_latency"] = int(win.get("count_hook_latency", 0)) + 1
                if last_reason:
                    win["last_reason"] = str(last_reason)
                if has_critical:
                    win["has_critical"] = True
            except Exception:
                continue
        # Persist windows sidecar
        _atomic_write_json(self._windows_path, self._windows)

    def _synthesise_due_windows(self) -> None:
        if not self._windows:
            return
        done: list[str] = []
        synth_telemetry: list[leyline_pb2.TelemetryPacket] = []
        for cmd_id, win in self._windows.items():
            try:
                collected = int(win.get("collected", 0))
                target = max(1, int(win.get("target", self._obs_window_epochs)))
            except Exception:
                continue
            if collected < target:
                continue
            # Derive outcome
            try:
                sum_delta = float(win.get("sum_loss_delta", 0.0))
            except Exception:
                sum_delta = 0.0
            has_crit = bool(win.get("has_critical", False))
            if sum_delta < 0.0 and not has_crit:
                outcome = leyline_pb2.FIELD_REPORT_OUTCOME_SUCCESS
            elif sum_delta > 0.0 or has_crit:
                outcome = leyline_pb2.FIELD_REPORT_OUTCOME_REGRESSION
            else:
                outcome = leyline_pb2.FIELD_REPORT_OUTCOME_NEUTRAL
            # Build metrics
            mn = win.get("min_loss_delta", 0.0)
            mx = win.get("max_loss_delta", 0.0)
            s_hook = float(win.get("sum_hook_latency_ms", 0.0))
            c_hook = int(win.get("count_hook_latency", 0))
            metrics = {
                "loss_delta_total": float(sum_delta),
                "loss_delta_min": float(mn if mn != float("inf") else 0.0),
                "loss_delta_max": float(mx if mx != float("-inf") else 0.0),
            }
            if c_hook > 0:
                metrics["avg_hook_latency_ms"] = float(s_hook / max(1, c_hook))
            last_reason = str(win.get("last_reason", ""))
            # Compose a minimal command shell to carry IDs/timestamps
            cmd = leyline_pb2.AdaptationCommand(version=1)
            cmd.command_id = cmd_id
            seed_id = str(win.get("seed_id", ""))
            bp_id = str(win.get("blueprint_id", ""))
            tr_id = str(win.get("training_run_id", "run-unknown"))
            issued_iso = str(win.get("issued_at_iso", ""))
            if issued_iso:
                with contextlib.suppress(Exception):
                    # From isoformat to Timestamp
                    dt = datetime.fromisoformat(issued_iso)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    cmd.issued_at.FromDatetime(dt)
            # Report ID includes synthesis marker
            report = self.generate_field_report(
                command=cmd,
                outcome=outcome,
                metrics_delta=metrics,
                training_run_id=tr_id,
                seed_id=seed_id,
                blueprint_id=bp_id,
                observation_window_epochs=target,
                notes=last_reason or "",
            )
            # Stamp a stable synthesised report_id if needed
            try:
                start_epoch = int(win.get("start_epoch", 0))
                report.report_id = f"fr-synth-{cmd_id}-{start_epoch}+{target}"
            except Exception:
                report.report_id = f"fr-synth-{cmd_id}+{target}"
            # Telemetry event for synthesis
            try:
                synth_telemetry.append(
                    build_telemetry_packet(
                        packet_id=f"tamiyo-field-report-synth-{cmd_id}",
                        source="tamiyo",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                        metrics=[],
                        events=[
                            TelemetryEvent(
                                description="field_report_synthesised",
                                level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                                attributes={
                                    "report_id": report.report_id,
                                    "command_id": cmd_id,
                                    "target_epochs": str(target),
                                },
                            )
                        ],
                        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
                        health_summary="field_report_synthesised",
                        health_indicators={},
                    )
                )
            except Exception:
                pass
            done.append(cmd_id)
        if done:
            # Persist removal and store append has already updated memory via store.reports()
            for k in done:
                self._windows.pop(k, None)
            _atomic_write_json(self._windows_path, self._windows)
        # Queue telemetry for publication
        if synth_telemetry:
            self._telemetry_packets.extend(synth_telemetry)

    def _run_policy(
        self,
        state: leyline_pb2.SystemStatePacket,
        enforce_timeouts: bool,
    ) -> tuple[leyline_pb2.AdaptationCommand, float, bool]:
        start = time.perf_counter()
        timed_out = False

        if enforce_timeouts and self._step_timeout_s > 0 and self._worker is not None:
            handle = self._worker.submit(
                self._policy.select_action,
                state,
                timeout=self._step_timeout_s,
            )
            try:
                command = handle.result()
            except AsyncTimeoutError:
                timed_out = True
                command = self._build_timeout_command("timeout_inference")
        else:
            command = self._policy.select_action(state)

        inference_ms = (time.perf_counter() - start) * 1000.0

        if timed_out:
            command.annotations.setdefault("policy_action", "timeout")
            command.annotations.setdefault("policy_param_delta", "0.0")
            if "policy_version" not in command.annotations:
                command.annotations["policy_version"] = self._policy_version
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
        command.annotations.setdefault(
            "policy_value_estimate",
            f"{last_action.get('value_estimate', 0.0):.6f}",
        )
        command.annotations.setdefault(
            "policy_risk_score",
            f"{last_action.get('risk_score', 0.0):.6f}",
        )
        command.annotations.setdefault(
            "policy_risk_index",
            str(int(last_action.get("risk_index", 0.0))),
        )
        if "policy_version" not in command.annotations:
            command.annotations["policy_version"] = self._policy_version
        if "blending_method" not in command.annotations and "blending_method" in last_action:
            command.annotations["blending_method"] = str(last_action.get("blending_method", ""))

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
    ) -> tuple[dict[str, float | str | bool | int] | None, bool]:
        if command.command_type != leyline_pb2.COMMAND_SEED or not command.HasField(
            "seed_operation"
        ):
            return None, False

        if enforce_timeouts and self._metadata_timeout_s > 0 and self._worker is not None:
            handle = self._worker.submit(
                self._resolve_blueprint_info,
                command,
                timeout=self._metadata_timeout_s,
            )
            try:
                return handle.result(), False
            except AsyncTimeoutError:
                return None, True
        return self._resolve_blueprint_info(command), False

    def _ensure_blueprint_metadata_for_packet(self, state: leyline_pb2.SystemStatePacket) -> None:
        blueprint_id = state.packet_id or state.training_run_id
        if not blueprint_id or blueprint_id in self._blueprint_cache:
            return
        # Bound the pre-warm fetch so we never stall the step path
        record = None
        if self._worker is not None and self._metadata_timeout_s > 0:
            handle = self._worker.submit(
                self._urza.get,
                blueprint_id,
                timeout=self._metadata_timeout_s,
            )
            try:
                record = handle.result()
            except AsyncTimeoutError:
                return
        else:
            record = self._urza.get(blueprint_id)
        if record is None:
            return
        data = self._serialize_blueprint_record(record)
        self._blueprint_cache[blueprint_id] = (datetime.now(tz=UTC), data)

    def _serialize_blueprint_record(
        self, record: Any
    ) -> dict[str, float | str | bool | int | dict | list]:
        descriptor = record.metadata
        allowed: dict[str, dict[str, float]] = {}
        allowed_mapping = getattr(descriptor, "allowed_parameters", None)
        if isinstance(allowed_mapping, Mapping):
            for key, bounds in allowed_mapping.items():
                min_value = float(getattr(bounds, "min_value", 0.0))
                max_value = float(getattr(bounds, "max_value", 0.0))
                allowed[key] = {
                    "min": min_value,
                    "max": max_value,
                    "span": float(max_value - min_value),
                }
        tier_value = int(getattr(descriptor, "tier", 0) or 0)
        try:
            tier_name = leyline_pb2.BlueprintTier.Name(tier_value)
        except ValueError:
            tier_name = str(tier_value)
        data: dict[str, float | str | bool | int | dict | list] = {
            "tier": tier_name,
            "tier_index": tier_value,
            "risk": float(getattr(descriptor, "risk", 0.0) or 0.0),
            "stage": int(getattr(descriptor, "stage", 0) or 0),
            "quarantine_only": bool(getattr(descriptor, "quarantine_only", False)),
            "approval_required": bool(getattr(descriptor, "approval_required", False)),
            "description": getattr(descriptor, "description", ""),
            "parameter_count": int(len(allowed)),
            "allowed_parameters": allowed,
        }
        candidate_score = (
            (1.0 - float(data["risk"])) + 0.05 * float(data["stage"]) + 0.02 * len(allowed)
        )
        data["candidate_score"] = max(candidate_score, 0.0)
        compile_ms = getattr(record, "compile_ms", None)
        if compile_ms is not None:
            data["compile_ms"] = float(compile_ms)
        prewarm_ms = getattr(record, "prewarm_ms", None)
        if prewarm_ms is not None:
            data["prewarm_ms"] = float(prewarm_ms)
        samples = getattr(record, "prewarm_samples", ()) or ()
        if samples:
            data["prewarm_samples_ms"] = [float(sample) for sample in samples]
        guard_spec = getattr(record, "guard_spec", ()) or ()
        if guard_spec:
            data["guard_spec"] = list(guard_spec)
        guard_summary = getattr(record, "guard_spec_summary", ()) or ()
        if guard_summary:
            data["guard_spec_summary"] = list(guard_summary)
        guard_digest = getattr(record, "guard_digest", None)
        if guard_digest:
            data["guard_digest"] = guard_digest
        checksum = getattr(record, "checksum", None)
        if checksum:
            data["checksum"] = checksum
        tags = getattr(record, "tags", ()) or ()
        if tags:
            data["tags"] = list(tags)

        graph_metadata = self._extract_graph_metadata(record)
        if graph_metadata:
            data["graph"] = graph_metadata
        # Optional: attach BSDS-lite payload from Urabrask/Urza extras when present.
        # This follows the prototype-delta guidance to consume a minimal safety sheet
        # without introducing new Leyline contracts. When available, the BSDS risk
        # score supersedes descriptor risk for decision gating.
        bsds = self._extract_bsds(record)
        if bsds:
            try:
                risk_score = float(bsds.get("risk_score", data.get("risk", 0.0)))
                if risk_score >= 0.0:
                    data["risk"] = risk_score
                    data["risk_provenance"] = "bsds"
            except Exception:
                pass
            data["bsds"] = bsds
        return data

    def _extract_graph_metadata(self, record: Any) -> dict[str, Any] | None:
        extras: Mapping[str, Any] | None = getattr(record, "extras", None)
        graph_section: Any = None
        if extras:
            graph_section = extras.get("graph_metadata")
            if isinstance(graph_section, str):
                try:
                    graph_section = json.loads(graph_section)
                except json.JSONDecodeError:
                    graph_section = None
        if not graph_section:
            graph_section = self._graph_metadata_from_guard_spec(record)
        if not graph_section:
            return None
        return self._normalise_graph_metadata(graph_section, record)

    def _extract_bsds(self, record: Any) -> dict[str, str | float] | None:
        """Extract a minimal BSDS-lite block from Urza extras, if available.

        Expected schema (all optional):
          {
            "risk_score": 0.0..1.0,
            "hazard_band": "LOW|MEDIUM|HIGH|CRITICAL",
            "handling_class": "standard|restricted|quarantine",
            "resource_profile": "cpu|gpu|memory_heavy|io_heavy|mixed",
            "provenance": "URABRASK" | "external",
            "issued_at": "ISO8601"
          }
        """
        extras: Mapping[str, Any] | None = getattr(record, "extras", None)
        if not extras or "bsds" not in extras:
            return None
        raw = extras.get("bsds")
        block: dict[str, str | float] | None = None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                block = {
                    str(k): parsed[k]
                    for k in (
                        "hazard_band",
                        "handling_class",
                        "resource_profile",
                        "provenance",
                        "issued_at",
                    )
                    if k in parsed
                }
                with contextlib.suppress(Exception):
                    block["risk_score"] = float(parsed.get("risk_score", 0.0))  # type: ignore[index]
        elif isinstance(raw, Mapping):
            block = {
                str(k): raw[k]
                for k in (
                    "hazard_band",
                    "handling_class",
                    "resource_profile",
                    "provenance",
                    "issued_at",
                )
                if k in raw
            }
            with contextlib.suppress(Exception):
                block["risk_score"] = float(raw.get("risk_score", 0.0))  # type: ignore[index]
        return block or None

    def _graph_metadata_from_guard_spec(self, record: Any) -> dict[str, Any] | None:
        guard_spec: Sequence[Mapping[str, Any]] = tuple(getattr(record, "guard_spec", ()) or ())
        if not guard_spec:
            return None
        latency_ms = float(getattr(record, "prewarm_ms", 0.0) or 0.0)
        latency_per_entry = latency_ms / max(1, len(guard_spec))
        layers: list[dict[str, Any]] = []
        activations: list[dict[str, Any]] = []
        for idx, spec in enumerate(guard_spec):
            shape = spec.get("shape") or []
            dims: list[int] = [int(d) for d in shape if isinstance(d, (int, float))]
            param_count = 1
            for dim in dims:
                param_count *= max(1, dim)
            input_channels = dims[-2] if len(dims) >= 2 else dims[-1] if dims else 1
            output_channels = dims[-1] if dims else 1
            layer_entry = {
                "layer_id": f"{record.metadata.blueprint_id}-L{idx}",
                "type": "guard_tensor",
                "depth": idx,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "parameter_count": param_count,
                "latency_ms": latency_per_entry,
                "gradient_norm": 0.0,
                "weight_norm": 0.0,
            }
            layers.append(layer_entry)
            activations.append(
                {
                    "activation_id": f"{record.metadata.blueprint_id}-A{idx}",
                    "type": spec.get("dtype", "unknown"),
                    "saturation_rate": 0.0,
                    "gradient_flow": 1.0,
                    "computational_cost": float(param_count),
                }
            )
        parameters = []
        allowed_mapping = getattr(record.metadata, "allowed_parameters", {})
        for name, bounds in allowed_mapping.items():
            lower = float(getattr(bounds, "min_value", 0.0))
            upper = float(getattr(bounds, "max_value", 0.0))
            parameters.append(
                {
                    "name": name,
                    "min": lower,
                    "max": upper,
                    "default": (lower + upper) * 0.5,
                    "span": upper - lower,
                }
            )
        return {
            "layers": layers,
            "activations": activations,
            "parameters": parameters,
            "capabilities": {},
        }

    def _normalise_graph_metadata(
        self,
        graph_section: Mapping[str, Any],
        record: Any,
    ) -> dict[str, Any]:
        layers = self._coerce_descriptor_list(graph_section.get("layers"), key_field="layer_id")
        activations = self._coerce_descriptor_list(
            graph_section.get("activations"), key_field="activation_id"
        )
        parameters = self._coerce_descriptor_list(graph_section.get("parameters"), key_field="name")
        if not parameters:
            allowed_mapping = getattr(record.metadata, "allowed_parameters", {})
            for name, bounds in allowed_mapping.items():
                lower = float(getattr(bounds, "min_value", 0.0))
                upper = float(getattr(bounds, "max_value", 0.0))
                parameters.append(
                    {
                        "name": name,
                        "min": lower,
                        "max": upper,
                        "default": (lower + upper) * 0.5,
                        "span": upper - lower,
                    }
                )

        capabilities = {}
        raw_capabilities = graph_section.get("capabilities", {})
        if isinstance(raw_capabilities, Mapping):
            for key, value in raw_capabilities.items():
                if isinstance(value, (list, tuple)):
                    capabilities[str(key)] = [str(item) for item in value]
                elif isinstance(value, (int, float, str, bool)):
                    capabilities[str(key)] = value

        return {
            "layers": layers,
            "activations": activations,
            "parameters": parameters,
            "capabilities": capabilities,
            "source": graph_section.get("source", "urza"),
        }

    @staticmethod
    def _coerce_descriptor_list(value: Any, *, key_field: str) -> list[dict[str, Any]]:
        if not value:
            return []
        result: list[dict[str, Any]] = []
        for index, entry in enumerate(value):
            if isinstance(entry, Mapping):
                converted = {str(k): v for k, v in entry.items()}
            else:
                continue
            converted.setdefault(key_field, f"auto-{index}")
            result.append(converted)
        return result

    def _apply_risk_engine(
        self,
        command: leyline_pb2.AdaptationCommand,
        *,
        state: leyline_pb2.SystemStatePacket,
        loss_delta: float,
        blueprint_info: dict[str, float | str | bool | int] | None,
        blueprint_timeout: bool,
        timed_out: bool,
        training_metrics: dict[str, float],
    ) -> tuple[dict[str, float | str | bool | int] | None, list[TelemetryEvent]]:
        events: list[TelemetryEvent] = []
        reason = command.annotations.get("risk_reason")

        policy_risk_score: float | None = None
        risk_score_raw = command.annotations.get("policy_risk_score")
        if risk_score_raw:
            try:
                policy_risk_score = float(risk_score_raw)
                if not math.isfinite(policy_risk_score):
                    policy_risk_score = None
            except (TypeError, ValueError):
                policy_risk_score = None
        if policy_risk_score is not None:
            events.append(
                TelemetryEvent(
                    description="policy_risk_signal",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                    attributes={
                        "score": f"{policy_risk_score:.3f}",
                        "index": command.annotations.get("policy_risk_index", "0"),
                    },
                )
            )
            if policy_risk_score >= 0.98:
                reason = "policy_risk_critical"
                command.command_type = leyline_pb2.COMMAND_PAUSE
                events.append(
                    TelemetryEvent(
                        description="policy_risk_critical",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                        attributes={"score": f"{policy_risk_score:.3f}"},
                    )
                )
                self._set_conservative_mode(True, "policy_risk_critical", events)
            elif policy_risk_score >= 0.85 and command.command_type == leyline_pb2.COMMAND_SEED:
                reason = reason or "policy_risk_elevated"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"
                events.append(
                    TelemetryEvent(
                        description="policy_risk_elevated",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                        attributes={"score": f"{policy_risk_score:.3f}"},
                    )
                )

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
            # BSDS-lite (if present in blueprint_info)
            bsds_block = blueprint_info.get("bsds") if isinstance(blueprint_info, Mapping) else None
            if isinstance(bsds_block, Mapping):
                # Optional signature verification (observability only; fail-open)
                try:
                    settings = EsperSettings()
                    if getattr(settings, "urabrask_signing_enabled", False):
                        from esper.urabrask import metrics as _ura_metrics
                        from esper.urabrask.wal import verify_bsds_signature_in_extras

                        ctx = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
                        extras_map = blueprint_info if isinstance(blueprint_info, Mapping) else {}
                        ok = verify_bsds_signature_in_extras(extras_map, ctx=ctx)
                        if not ok:
                            _ura_metrics.inc_integrity_failures()
                except Exception:
                    pass
                # Surface annotations for downstream consumers
                for k_src, k_dst in (
                    ("hazard_band", "bsds_hazard_band"),
                    ("handling_class", "bsds_handling_class"),
                    ("resource_profile", "bsds_resource_profile"),
                    ("provenance", "bsds_provenance"),
                ):
                    val = bsds_block.get(k_src)
                    if isinstance(val, (str, int, float)):
                        command.annotations.setdefault(k_dst, str(val))
                if "risk_score" in bsds_block:
                    with contextlib.suppress(Exception):
                        command.annotations.setdefault(
                            "bsds_risk", f"{float(bsds_block['risk_score']):.2f}"
                        )
                prov = str(bsds_block.get("provenance", ""))
                events.append(
                    TelemetryEvent(
                        description="bsds_present",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                        attributes={
                            "hazard": str(bsds_block.get("hazard_band", "")),
                            "risk": str(bsds_block.get("risk_score", "")),
                            "provenance": prov,
                        },
                    )
                )
                hazard = str(bsds_block.get("hazard_band", "")).upper()
                handling = str(bsds_block.get("handling_class", "")).lower()
                # Handling override: quarantine treated as CRITICAL
                if handling == "quarantine":
                    reason = "bsds_handling_quarantine"
                    command.command_type = leyline_pb2.COMMAND_PAUSE
                    events.append(
                        TelemetryEvent(
                            description="bsds_handling_quarantine",
                            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                            attributes={"handling": handling, "provenance": prov},
                        )
                    )
                    self._set_conservative_mode(True, reason, events)
                # Escalate on high/critical hazards
                if hazard == "CRITICAL":
                    reason = "bsds_hazard_critical"
                    command.command_type = leyline_pb2.COMMAND_PAUSE
                    events.append(
                        TelemetryEvent(
                            description="bsds_hazard_critical",
                            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                            attributes={"hazard": hazard, "provenance": prov},
                        )
                    )
                    self._set_conservative_mode(True, reason, events)
                elif hazard == "HIGH":
                    # Always record the event; downgrade action only if still SEED
                    events.append(
                        TelemetryEvent(
                            description="bsds_hazard_high",
                            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                            attributes={"hazard": hazard, "provenance": prov},
                        )
                    )
                    if command.command_type == leyline_pb2.COMMAND_SEED:
                        reason = reason or "bsds_hazard_high"
                        command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                        command.optimizer_adjustment.optimizer_id = "sgd"

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

        # Latency guardrails
        hook_latency = training_metrics.get("hook_latency_ms")
        if hook_latency and hook_latency > float(self._risk.hook_budget_ms):
            reason = "hook_budget"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="hook_budget",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{hook_latency:.2f}"},
                )
            )
        step_latency = training_metrics.get("step_latency_ms")
        if step_latency and step_latency > float(self._risk.step_latency_high_ms):
            # Treat high step latency as HIGH priority; prefer PAUSE
            reason = "step_latency_high"
            command.command_type = leyline_pb2.COMMAND_PAUSE
            events.append(
                TelemetryEvent(
                    description="step_latency_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{step_latency:.2f}"},
                )
            )
        apply_ms = training_metrics.get("kasmina.apply_ms")
        if apply_ms and apply_ms > float(self._risk.kasmina_apply_slow_ms):
            desc = "kasmina_apply_slow"
            events.append(
                TelemetryEvent(
                    description=desc,
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{apply_ms:.2f}"},
                )
            )
            if command.command_type == leyline_pb2.COMMAND_SEED:
                reason = "kasmina_apply_slow"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"
        finalize_ms = training_metrics.get("kasmina.finalize_ms")
        if finalize_ms and finalize_ms > float(self._risk.kasmina_finalize_slow_ms):
            desc = "kasmina_finalize_slow"
            events.append(
                TelemetryEvent(
                    description=desc,
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"latency_ms": f"{finalize_ms:.2f}"},
                )
            )
            if command.command_type == leyline_pb2.COMMAND_SEED:
                reason = "kasmina_finalize_slow"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"

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

        # Drift/stability signals
        vol = training_metrics.get("loss_volatility")
        if vol and vol > (self._risk.max_loss_spike * 0.75):
            events.append(
                TelemetryEvent(
                    description="loss_volatility_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"volatility": f"{vol:.3f}"},
                )
            )
            if command.command_type == leyline_pb2.COMMAND_SEED:
                reason = reason or "loss_volatility_high"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"
        gvar = training_metrics.get("grad_var")
        if gvar and gvar > 1.0 and command.command_type == leyline_pb2.COMMAND_SEED:
            events.append(
                TelemetryEvent(
                    description="grad_variance_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"var": f"{gvar:.3f}"},
                )
            )
            reason = reason or "grad_variance_high"
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER
            command.optimizer_adjustment.optimizer_id = "sgd"
        conflict = training_metrics.get("grad_conflict_rate")
        if conflict and conflict > 0.5 and command.command_type == leyline_pb2.COMMAND_SEED:
            events.append(
                TelemetryEvent(
                    description="grad_conflict_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"rate": f"{conflict:.3f}"},
                )
            )
            reason = reason or "grad_conflict_high"
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER
            command.optimizer_adjustment.optimizer_id = "sgd"

        # Optimizer hints
        lr = training_metrics.get("optimizer_lr")
        if lr is not None and lr <= 0.0 and loss_delta > (self._risk.max_loss_spike * 0.5):
            events.append(
                TelemetryEvent(
                    description="optimizer_hint",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"lr": f"{lr:.6f}", "loss_delta": f"{loss_delta:.3f}"},
                )
            )
            if command.command_type == leyline_pb2.COMMAND_SEED:
                reason = "optimizer_hint"
                command.command_type = leyline_pb2.COMMAND_OPTIMIZER
                command.optimizer_adjustment.optimizer_id = "sgd"

        # Device pressure (best-effort)
        gpu_util = training_metrics.get("gpu_util_percent")
        gpu_free = training_metrics.get("gpu_mem_free_gb")
        cpu_util = training_metrics.get("cpu_util_percent")
        if gpu_util and gpu_util >= 95.0 and gpu_free is not None and gpu_free <= 0.2:
            events.append(
                TelemetryEvent(
                    description="device_pressure_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "gpu_util": f"{gpu_util:.1f}",
                        "gpu_mem_free_gb": f"{gpu_free:.2f}",
                    },
                )
            )
            if command.command_type == leyline_pb2.COMMAND_SEED:
                reason = reason or "device_pressure_high"
                command.command_type = leyline_pb2.COMMAND_PAUSE
        elif cpu_util and cpu_util >= 95.0 and command.command_type == leyline_pb2.COMMAND_SEED:
            events.append(
                TelemetryEvent(
                    description="cpu_pressure_high",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"cpu_util": f"{cpu_util:.1f}"},
                )
            )
            reason = reason or "cpu_pressure_high"
            command.command_type = leyline_pb2.COMMAND_OPTIMIZER

        if (
            not reason
            and command.command_type == leyline_pb2.COMMAND_PAUSE
            and self._risk.conservative_mode
        ):
            reason = "conservative_mode"

        if (
            not timed_out
            and not blueprint_timeout
            and self._inference_breaker.state
            == leyline_pb2.CircuitBreakerState.CIRCUIT_STATE_CLOSED
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
        # Promote to CRITICAL on any critical event. Promote to HIGH only for
        # specific HIGH-severity operational reasons; otherwise keep NORMAL.
        escalate_high = {
            "timeout_inference",
            "timeout_urza",
            "step_latency_high",
            "loss_spike",
            "isolation_violations",
            "hook_budget",
            "policy_risk_critical",
            "bp_quarantine",
            "device_pressure_high",
            "cpu_pressure_high",
        }
        saw_high = False
        for event in events:
            if event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL:
                return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL
            if (
                event.level
                in (
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
                )
                and (event.description or "") in escalate_high
            ):
                saw_high = True
        return (
            leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
            if saw_high
            else leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
        )

    def _sign_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        """Assign identifiers and attach an HMAC signature."""

        if "signature" in command.annotations:
            del command.annotations["signature"]
        command.command_id = str(uuid4())
        command.issued_at.GetCurrentTime()
        payload = command.SerializeToString(deterministic=True)
        command.annotations["signature"] = sign(payload, self._signing_context)

    def _derive_health_status(
        self,
        command: leyline_pb2.AdaptationCommand,
        events: list[TelemetryEvent],
    ) -> leyline_pb2.HealthStatus:
        if any(
            event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL for event in events
        ):
            return leyline_pb2.HealthStatus.HEALTH_STATUS_CRITICAL
        if command.command_type == leyline_pb2.COMMAND_PAUSE:
            return leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
        if any(
            event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING for event in events
        ):
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
        indicators["policy_compile"] = (
            "1" if getattr(self._policy, "compile_enabled", False) else "0"
        )
        indicators["policy_arch"] = self._policy_version
        # Timeout budgets (ms)
        try:
            indicators["timeout_budget_ms"] = f"{self._step_timeout_s * 1000.0:.1f}"
        except Exception:
            pass
        try:
            indicators["metadata_timeout_budget_ms"] = f"{self._metadata_timeout_s * 1000.0:.1f}"
        except Exception:
            pass
        last_action = self._policy.last_action
        blending = last_action.get("blending_method")
        if blending:
            indicators["blending_method"] = str(blending)
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
        version = getattr(new_policy, "architecture_version", None)
        expected_version = getattr(self._policy, "architecture_version", None)
        if expected_version and version != expected_version:
            raise ValueError(f"Unsupported Tamiyo policy version: {version}")
        if hasattr(new_policy, "update_blueprint_metadata"):
            with contextlib.suppress(Exception):
                metadata_payload = {bp: info for bp, (_, info) in self._blueprint_cache.items()}
                new_policy.update_blueprint_metadata(metadata_payload)
        self._policy = new_policy
        self._policy_version = getattr(new_policy, "architecture_version", self._policy_version)

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

        if self._owns_worker and self._worker is not None:
            self._worker.shutdown(cancel_pending=True)
            self._owns_worker = False
            self._worker = None

    def _validate_command_dependencies(self, command: leyline_pb2.AdaptationCommand) -> None:
        if command.command_type != leyline_pb2.COMMAND_SEED:
            return
        command_id = command.command_id or "<unset>"
        ensure_present(
            command.HasField("seed_operation"),
            DependencyContext(
                subsystem="tamiyo",
                dependency_type="seed_operation",
                identifier=command_id,
                details={"command_type": leyline_pb2.CommandType.Name(command.command_type)},
            ),
            reason="seed command missing seed_operation",
        )
        blueprint_id = (command.seed_operation.blueprint_id or "").strip()
        ensure_present(
            bool(blueprint_id),
            DependencyContext(
                subsystem="tamiyo",
                dependency_type="blueprint",
                identifier=blueprint_id or "<empty>",
                details={"command_id": command_id},
            ),
            reason="seed operation missing blueprint_id",
        )

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    def _field_report_key(self, report: leyline_pb2.FieldReport) -> str:
        """Compute a stable retry key for a field report."""
        rid = getattr(report, "report_id", "")
        if rid:
            return str(rid)
        try:
            import hashlib

            return hashlib.sha256(report.SerializeToString()).hexdigest()
        except Exception:
            return f"digest-{id(report)}"

    async def publish_history(self, oona: OonaClient) -> bool:
        """Publish collected field reports and telemetry via Oona with hedging.

        Returns True if all payloads were sent; False if any field reports were
        retained due to transient failures. WAL is not modified here.
        """

        all_sent = True
        failed_reports: list[leyline_pb2.FieldReport] = []
        now_ms = int(time.time() * 1000)
        retries_total = 0
        dropped_total = 0
        published_total = 0
        new_telemetry: list[leyline_pb2.TelemetryPacket] = []
        for report in list(self._field_reports):
            key = self._field_report_key(report)
            idx = self._retry_index.get(key) or {}
            published = bool(idx.get("published", False))
            if published:
                continue
            next_due = int(idx.get("next_attempt_ms", 0))
            if next_due and now_ms < next_due:
                failed_reports.append(report)
                all_sent = False
                continue
            try:
                await oona.publish_field_report(report)
                self._report_retry_count.pop(key, None)
                self._retry_index[key] = {
                    "published": True,
                    "retry_count": int(idx.get("retry_count", 0)),
                    "next_attempt_ms": 0,
                }
                published_total += 1
            except Exception as exc:  # pragma: no cover - asserted via tests
                count = int(self._report_retry_count.get(key, 0)) + 1
                self._report_retry_count[key] = count
                retries_total += 1
                # Compute backoff schedule
                try:
                    base = max(0, int(getattr(self._settings, "tamiyo_fr_retry_backoff_ms", 1000)))
                except Exception:
                    base = 1000
                try:
                    mult = float(getattr(self._settings, "tamiyo_fr_retry_backoff_mult", 2.0))
                except Exception:
                    mult = 2.0
                delay = int(base * (mult ** max(1, count))) if base > 0 else 0
                next_ms = now_ms + max(0, delay)
                self._retry_index[key] = {
                    "published": False,
                    "retry_count": count,
                    "next_attempt_ms": next_ms,
                    "last_error": str(exc),
                }
                if count <= self._max_report_retries:
                    failed_reports.append(report)
                else:
                    # Drop from memory after cap; WAL remains intact; mark dropped in index
                    self._report_retry_count.pop(key, None)
                    dropped_total += 1
                    self._retry_index[key]["dropped"] = True
                # Emit retry/drop telemetry
                try:
                    evt = (
                        "field_report_retry"
                        if count <= self._max_report_retries
                        else "field_report_drop"
                    )
                    lvl = (
                        leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                        if evt == "field_report_retry"
                        else leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
                    )
                    new_telemetry.append(
                        build_telemetry_packet(
                            packet_id=f"tamiyo-fr-retry-{key}",
                            source="tamiyo",
                            level=lvl,
                            metrics=[],
                            events=[
                                TelemetryEvent(
                                    description=evt,
                                    level=lvl,
                                    attributes={
                                        "report_id": key,
                                        "retry_count": str(count),
                                    },
                                )
                            ],
                            health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED,
                            health_summary=evt,
                            health_indicators={},
                        )
                    )
                except Exception:
                    pass
                all_sent = False

        # Publish pre-existing telemetry first
        for telemetry in self._telemetry_packets:
            priority_name = telemetry.system_health.indicators.get("priority")
            priority_enum = None
            if priority_name:
                with contextlib.suppress(ValueError):
                    priority_enum = leyline_pb2.MessagePriority.Value(priority_name)
            try:
                await oona.publish_telemetry(telemetry, priority=priority_enum)
            except Exception:  # pragma: no cover - best effort
                all_sent = False
        # Publish retry/drop + summary telemetry
        if retries_total or dropped_total or published_total:
            try:
                summary = build_telemetry_packet(
                    packet_id="tamiyo-field-report-summary",
                    source="tamiyo",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
                    metrics=[
                        TelemetryMetric(
                            "tamiyo.field_reports.published_total",
                            float(published_total),
                            unit="count",
                        ),
                        TelemetryMetric(
                            "tamiyo.field_reports.retries_total", float(retries_total), unit="count"
                        ),
                        TelemetryMetric(
                            "tamiyo.field_reports.dropped_total", float(dropped_total), unit="count"
                        ),
                        TelemetryMetric(
                            "tamiyo.field_reports.pending_total",
                            float(len(failed_reports)),
                            unit="count",
                        ),
                    ],
                    events=[],
                    health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY,
                    health_summary="field_report_publish_summary",
                    health_indicators={},
                )
                new_telemetry.append(summary)
            except Exception:
                pass
        for telemetry in new_telemetry:
            try:
                await oona.publish_telemetry(telemetry)
            except Exception:  # pragma: no cover - best effort
                all_sent = False
        # Clear published buffers to avoid duplicate exports when Weatherlight
        # drains Tamiyo on each flush cycle (prototype-delta Tamiyo §8).
        # Retain only unsent reports for next attempt
        self._field_reports = failed_reports
        # Persist retry index sidecar
        _atomic_write_json(self._retry_index_path, self._retry_index)
        if self._telemetry_packets:
            self._telemetry_packets = []
        return all_sent

    def ingest_policy_update(self, update: leyline_pb2.PolicyUpdate) -> None:
        """Apply a policy update produced by Simic with transactional validation."""

        def _emit_rejection(reason: str) -> None:
            self._telemetry_packets.append(
                build_telemetry_packet(
                    packet_id="tamiyo-policy-update-error",
                    source="tamiyo",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    metrics=[],
                    events=[
                        TelemetryEvent(
                            description="policy_update_rejected",
                            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                            attributes={"reason": reason},
                        )
                    ],
                )
            )

        # Strict version/freshness checks (configurable)
        if getattr(self._settings, "tamiyo_verify_updates", True):
            try:
                incoming_version = (update.tamiyo_policy_version or "").strip()
            except Exception:
                incoming_version = ""
            if incoming_version and incoming_version != getattr(
                self._policy, "architecture_version", ""
            ):
                _emit_rejection("version_mismatch")
                return
            freshness_sec = max(0, int(getattr(self._settings, "tamiyo_update_freshness_sec", 0)))
            if freshness_sec > 0:
                try:
                    issued = update.issued_at.ToDatetime()  # type: ignore[attr-defined]
                    # Treat returned time as UTC; compare against now UTC
                    now = datetime.now(tz=UTC)
                    # Some protobuf impls return naive dt; coerce to UTC
                    if issued.tzinfo is None:
                        from datetime import timezone as _tz

                        issued = issued.replace(tzinfo=_tz.utc)
                    age = (now - issued).total_seconds()
                    if age > float(freshness_sec):
                        _emit_rejection("stale_update")
                        return
                except Exception:
                    # If parsing failed, proceed (do not block updates solely on timestamp parsing)
                    pass

        # If no payload, treat as metadata-only update (record and return)
        if not update.payload:
            self._policy_updates.append(update)
            if update.tamiyo_policy_version:
                self._policy_version = update.tamiyo_policy_version
            return

        # Transactional load: validate and load into a fresh policy instance first
        state_buffer = BytesIO(update.payload)
        try:
            state_dict = torch.load(state_buffer, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - invalid payload
            logger.warning("Tamiyo policy update payload invalid: %s", exc)
            _emit_rejection("invalid_payload")
            return

        # Build a new policy using the current config if available
        cfg = getattr(self._policy, "_config", None)
        try:
            candidate = TamiyoPolicy(cfg if cfg is not None else TamiyoPolicyConfig())
            candidate.validate_state_dict(state_dict)
            candidate.load_state_dict(state_dict, strict=False)
        except ValueError as exc:
            logger.warning("Tamiyo policy update rejected: %s", exc)
            _emit_rejection(
                "registry_mismatch" if "registry" in str(exc).lower() else "shape_mismatch"
            )
            return
        except RuntimeError as exc:  # pragma: no cover - defensive
            logger.warning("Tamiyo policy update incompatible: %s", exc)
            _emit_rejection("incompatible_update")
            return

        # Swap the live policy atomically and record the update
        try:
            self.update_policy(candidate)
            if update.tamiyo_policy_version:
                self._policy_version = update.tamiyo_policy_version
            self._policy_updates.append(update)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tamiyo policy swap failed: %s", exc)
            _emit_rejection("swap_failed")

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
        if command.command_type != leyline_pb2.COMMAND_SEED or not command.HasField(
            "seed_operation"
        ):
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
        data = self._serialize_blueprint_record(record)
        self._blueprint_cache[blueprint_id] = (now, data)
        return data


__all__ = ["TamiyoService", "RiskConfig"]
logger = logging.getLogger(__name__)
