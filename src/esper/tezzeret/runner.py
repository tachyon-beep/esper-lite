"""Tezzeret compilation forge for Esper-Lite (TKT-202)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import logging

from esper.core import TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.karn import BlueprintDescriptor, KarnCatalog
from esper.leyline import leyline_pb2
from esper.urza import UrzaLibrary

from .compiler import TezzeretCompiler

LOGGER = logging.getLogger("esper.tezzeret")


def _midpoint(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


@dataclass(slots=True)
class ForgeMetrics:
    breaker_state: int = 0
    breaker_open_total: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    jobs_started: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    conservative_mode: int = 0
    last_strategy: str = "standard"


@dataclass(slots=True)
class CompilationJob:
    blueprint_id: str


class TezzeretForge:
    """Coordinates blueprint compilation and persistence with WAL recovery."""

    def __init__(
        self,
        catalog: KarnCatalog,
        library: UrzaLibrary,
        compiler: TezzeretCompiler,
        *,
        wal_path: Path,
        compile_timeout_s: float = 60.0,
        breaker_threshold: int = 3,
    ) -> None:
        self._catalog = catalog
        self._library = library
        self._compiler = compiler
        self._wal_path = wal_path
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)
        self._compile_timeout_s = max(compile_timeout_s, 1.0)
        self._breaker_threshold = max(breaker_threshold, 1)
        self._consecutive_failures = 0
        self._breaker_open_until: float | None = None
        self._conservative_mode = False
        self._metrics = ForgeMetrics()
        self._telemetry_events: list[TelemetryEvent] = []

    def run(self) -> None:
        pending = self._load_pending_jobs()
        if not pending:
            pending = [job.blueprint_id for job in self._discover_jobs()]
            self._persist_pending(pending)

        for blueprint_id in list(pending):
            if self._breaker_open():
                LOGGER.warning(
                    "Tezzeret breaker open; skipping compilation for %s",
                    blueprint_id,
                )
                self._emit_event(
                    "breaker_open_skip",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={"blueprint_id": blueprint_id},
                )
                break
            metadata = self._catalog.get(blueprint_id)
            if metadata is None:
                pending.remove(blueprint_id)
                self._persist_pending(pending)
                continue

            if self._library.get(blueprint_id):
                pending.remove(blueprint_id)
                self._persist_pending(pending)
                continue

            parameters = {
                key: _midpoint((bounds.min_value, bounds.max_value))
                for key, bounds in metadata.allowed_parameters.items()
            }

            strategy = "conservative" if self._conservative_mode else "standard"
            self._metrics.conservative_mode = 1 if self._conservative_mode else 0
            self._metrics.jobs_started += 1
            self._emit_event(
                "compile_started",
                attributes={
                    "blueprint_id": blueprint_id,
                    "strategy": strategy,
                },
            )
            try:
                artifact_path = self._compile_with_breaker(
                    metadata,
                    parameters,
                    strategy=strategy,
                )
            except Exception:
                self._metrics.jobs_failed += 1
                self._metrics.last_strategy = strategy
                raise
            update = self._compiler.latest_catalog_update()
            result = self._compiler.latest_result()
            extras = None
            if result is not None:
                extras = {
                    "guard_spec": result.guard_spec,
                    "guard_digest": result.guard_digest,
                    "guard_spec_summary": list(result.guard_summary),
                    "compile_ms": result.compile_ms,
                    "prewarm_ms": result.prewarm_ms,
                    "compile_strategy": result.compile_strategy,
                    "eager_fallback": result.eager_fallback,
                    "inductor_cache_dir": result.inductor_cache_dir,
                }
                # Build minimal graph metadata for Tamiyo consumers
                try:
                    extras["graph_metadata"] = _build_graph_metadata(metadata, result)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to build graph metadata: %s", exc)
                strategy_used = result.compile_strategy
                compile_ms = f"{result.compile_ms:.2f}"
                prewarm_ms = f"{result.prewarm_ms:.2f}"
            else:
                strategy_used = strategy
                compile_ms = "0.00"
                prewarm_ms = "0.00"
            self._library.save(
                metadata,
                artifact_path,
                catalog_update=update,
                extras=extras,
            )
            self._consecutive_failures = 0
            self._metrics.consecutive_failures = 0
            self._metrics.breaker_state = 0
            self._metrics.jobs_completed += 1
            self._metrics.last_error = ""
            self._conservative_mode = False
            self._metrics.conservative_mode = 0
            self._metrics.last_strategy = strategy_used
            self._emit_event(
                "compile_succeeded",
                attributes={
                    "blueprint_id": blueprint_id,
                    "strategy": strategy_used,
                    "compile_ms": compile_ms,
                    "prewarm_ms": prewarm_ms,
                },
            )
            pending.remove(blueprint_id)
            self._persist_pending(pending)

        if not pending and self._wal_path.exists():
            self._wal_path.unlink()

    def _discover_jobs(self) -> Iterable[CompilationJob]:
        for metadata in self._catalog.all():
            yield CompilationJob(blueprint_id=metadata.blueprint_id)

    def _load_pending_jobs(self) -> list[str]:
        if not self._wal_path.exists():
            return []
        try:
            payload = json.loads(self._wal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return list(payload.get("pending", []))

    def _persist_pending(self, pending: Iterable[str]) -> None:
        data = {"pending": list(pending)}
        self._wal_path.write_text(json.dumps(data), encoding="utf-8")

    def _breaker_open(self) -> bool:
        if self._breaker_open_until is None:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._breaker_open_until = None
            self._consecutive_failures = 0
            self._metrics.breaker_state = 0
            return False
        return True

    def _open_breaker(self) -> None:
        backoff = min(60.0, self._compile_timeout_s * 2)
        self._breaker_open_until = time.monotonic() + backoff
        self._metrics.breaker_state = 2
        self._metrics.breaker_open_total += 1
        self._enter_conservative_mode()
        self._emit_event(
            "breaker_open",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
            attributes={
                "backoff": f"{backoff:.1f}",
                "failures": str(self._consecutive_failures),
            },
        )
        LOGGER.error("Tezzeret breaker opened for %.1f seconds", backoff)

    def _compile_with_breaker(
        self,
        metadata: BlueprintDescriptor,
        parameters: dict[str, float],
        *,
        strategy: Literal["standard", "conservative"],
    ) -> Path:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._compiler.compile,
                metadata,
                parameters,
                strategy=strategy,
            )
            try:
                artifact_path = future.result(timeout=self._compile_timeout_s)
                return artifact_path
            except FuturesTimeout as exc:
                future.cancel()
                self._consecutive_failures += 1
                self._metrics.consecutive_failures = self._consecutive_failures
                self._metrics.last_error = "timeout"
                if self._consecutive_failures >= self._breaker_threshold:
                    self._open_breaker()
                self._metrics.last_strategy = strategy
                self._emit_event(
                    "compile_failed",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "blueprint_id": metadata.blueprint_id,
                        "strategy": strategy,
                        "reason": "timeout",
                    },
                )
                LOGGER.error(
                    "Tezzeret compile timed out for %s after %.1fs",
                    metadata.blueprint_id,
                    self._compile_timeout_s,
                )
                raise RuntimeError("compile timeout") from exc
            except Exception as exc:  # pragma: no cover - defensive path
                self._consecutive_failures += 1
                self._metrics.consecutive_failures = self._consecutive_failures
                self._metrics.last_error = type(exc).__name__
                if self._consecutive_failures >= self._breaker_threshold:
                    self._open_breaker()
                self._metrics.last_strategy = strategy
                self._emit_event(
                    "compile_failed",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "blueprint_id": metadata.blueprint_id,
                        "strategy": strategy,
                        "reason": type(exc).__name__,
                    },
                )
                LOGGER.error("Tezzeret compile failed for %s: %s", metadata.blueprint_id, exc)
                raise

    def metrics_snapshot(self) -> dict[str, float]:
        snapshot = self._compiler.metrics_snapshot()
        snapshot.update(
            {
                "tezzeret.breaker.state": float(self._metrics.breaker_state),
                "tezzeret.breaker.open_total": float(self._metrics.breaker_open_total),
                "tezzeret.compile.consecutive_failures": float(self._metrics.consecutive_failures),
                "tezzeret.breaker.consecutive_failures": float(self._metrics.consecutive_failures),
                "tezzeret.mode.conservative": float(self._metrics.conservative_mode),
                "tezzeret.jobs.started": float(self._metrics.jobs_started),
                "tezzeret.jobs.completed": float(self._metrics.jobs_completed),
                "tezzeret.jobs.failed": float(self._metrics.jobs_failed),
            }
        )
        return snapshot

    def build_telemetry_packet(
        self,
        *,
        packet_id: str | None = None,
        level_override: leyline_pb2.TelemetryLevel | None = None,
    ) -> leyline_pb2.TelemetryPacket:
        snapshot = self.metrics_snapshot()
        metrics = [
            TelemetryMetric(name, float(value))
            for name, value in snapshot.items()
        ]
        events = self.drain_telemetry_events()
        level = level_override
        if level is None:
            if self._metrics.breaker_state >= 2 or self._conservative_mode:
                level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING
            elif any(event.level == leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR for event in events):
                level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR
            else:
                level = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO
        return build_telemetry_packet(
            packet_id=packet_id or f"tezzeret-{int(time.time() * 1000)}",
            source="tezzeret",
            level=level,
            metrics=metrics,
            events=events,
        )

    def drain_telemetry_events(self) -> list[TelemetryEvent]:
        events = list(self._telemetry_events)
        self._telemetry_events.clear()
        return events

    def _emit_event(
        self,
        description: str,
        *,
        level: leyline_pb2.TelemetryLevel = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        attributes: dict[str, str] | None = None,
    ) -> None:
        payload = {k: str(v) for k, v in (attributes or {}).items()}
        self._telemetry_events.append(
            TelemetryEvent(
                description=f"tezzeret.{description}",
                level=level,
                attributes=payload,
            )
        )

    def _enter_conservative_mode(self) -> None:
        self._conservative_mode = True
        self._metrics.conservative_mode = 1


__all__ = ["TezzeretForge", "CompilationJob"]


def _build_graph_metadata(
    descriptor: BlueprintDescriptor,
    result: "TezzeretCompiler.CompilationResult" | object,  # type: ignore[name-defined]
) -> dict:
    """Construct a compact graph_metadata block from compile artifacts.

    Heuristic mapping (prototype):
    - One layer per guard_spec entry (input signature), with parameter_count derived from shape product
    - Activation list mirrors layers, using dtype as type
    - Latency distributed evenly from prewarm_ms across layers
    - Parameters mirrored from allowed_parameters bounds in the descriptor
    - Adjacency chain (L0->L1->...)
    """

    try:
        guard_spec = list(getattr(result, "guard_spec", []) or [])  # type: ignore[attr-defined]
    except Exception:
        guard_spec = []
    try:
        prewarm_ms = float(getattr(result, "prewarm_ms", 0.0))  # type: ignore[attr-defined]
    except Exception:
        prewarm_ms = 0.0

    layer_count = max(1, len(guard_spec))
    latency_per = prewarm_ms / float(layer_count)
    layers: list[dict] = []
    activations: list[dict] = []

    for idx in range(layer_count):
        spec = guard_spec[idx] if idx < len(guard_spec) else {}
        try:
            shape = list(spec.get("shape", [])) if isinstance(spec, dict) else []
        except Exception:
            shape = []
        dims = []
        for d in shape:
            try:
                dims.append(int(d))
            except Exception:
                continue
        param_count = 1
        for d in dims:
            param_count *= max(1, d)
        input_channels = dims[-2] if len(dims) >= 2 else (dims[-1] if dims else 1)
        output_channels = dims[-1] if dims else 1
        act_type = str(spec.get("dtype", "unknown")) if isinstance(spec, dict) else "unknown"

        layers.append(
            {
                "layer_id": f"{descriptor.blueprint_id}-L{idx}",
                "type": "guard_tensor",
                "depth": idx,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "parameter_count": param_count,
                "latency_ms": latency_per,
                "gradient_norm": 0.0,
                "weight_norm": 0.0,
                "activation": act_type,
            }
        )
        activations.append(
            {
                "activation_id": f"{descriptor.blueprint_id}-A{idx}",
                "type": act_type,
                "saturation_rate": 0.0,
                "gradient_flow": 1.0,
                "computational_cost": float(param_count),
                "nonlinearity_strength": 0.0,
            }
        )

    # Parameters from descriptor bounds
    parameters: list[dict] = []
    try:
        for name, bounds in descriptor.allowed_parameters.items():
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
    except Exception:
        pass

    # Adjacency chain
    adj_pairs = [[i, i + 1] for i in range(max(0, layer_count - 1))]

    return {
        "layers": layers,
        "activations": activations,
        "parameters": parameters,
        "adjacency": {"layer": adj_pairs},
        "capabilities": {},
        "source": "tezzeret",
    }
