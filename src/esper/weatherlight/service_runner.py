"""Weatherlight service runner.

Implements the prototype-delta Weatherlight supervisor described in
`docs/prototype-delta/weatherlight/implementation-roadmap.md` by composing the
existing Esper subsystems without mutating their internal behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Dict

from esper.core import EsperSettings, TelemetryEvent, TelemetryMetric, build_telemetry_packet
from esper.kasmina import KasminaPrefetchCoordinator, KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.oona import OonaClient, StreamConfig
from esper.tolaria.rollback import SharedDeadlineSignal
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext
from esper.tamiyo import TamiyoService
from esper.urza import UrzaLibrary, UrzaRuntime
from esper.urza.prefetch import UrzaPrefetchWorker

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from esper.tezzeret import TezzeretForge

LOGGER = logging.getLogger("esper.weatherlight")


@dataclass(slots=True)
class WorkerState:
    """Tracks Weatherlight worker lifecycle metrics."""

    name: str
    coroutine_factory: Callable[[], Awaitable[None]]
    task: asyncio.Task | None = None
    running: bool = False
    restart_count: int = 0
    backoff_seconds: float = 0.0
    last_error: str | None = None
    last_error_at: float | None = None


class WeatherlightService:
    """Coordinates Esper subsystems under a single async supervisor."""

    TELEMETRY_INTERVAL_SECONDS = 10
    HOUSEKEEPING_INTERVAL_SECONDS = 60

    def __init__(self, settings: EsperSettings | None = None) -> None:
        self._settings = settings or EsperSettings()
        self._shutdown_requested = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        self._shutdown_started = False
        self._workers: Dict[str, WorkerState] = {}
        self._telemetry_task: asyncio.Task | None = None
        self._housekeeping_task: asyncio.Task | None = None
        self._kasmina_telemetry_task: asyncio.Task | None = None
        self._start_time = time.monotonic()
        self._last_error_ts: float | None = None
        self._oona: OonaClient | None = None
        self._urza_library: UrzaLibrary | None = None
        self._urza_runtime: UrzaRuntime | None = None
        self._urza_worker: UrzaPrefetchWorker | None = None
        self._kasmina_manager: KasminaSeedManager | None = None
        self._kasmina_coordinator: KasminaPrefetchCoordinator | None = None
        self._tamiyo_service: TamiyoService | None = None
        self._tezzeret_metrics_provider: Callable[[], dict[str, float]] | None = None
        self._tezzeret_telemetry_provider: Callable[[], leyline_pb2.TelemetryPacket | None] | None = None
        self._rollback_signal_name: str | None = self._settings.tolaria_rollback_signal_name
        self._rollback_signal: SharedDeadlineSignal | None = None
        self._rollback_last_detect_s: float | None = None
        self._rollback_last_detect_ts_ms: int | None = None
        self._rollback_detections_total: int = 0
        self._rollback_skipped_total: int = 0
        self._rollback_monitor_task: asyncio.Task | None = None
        self._kasmina_packet_queue: asyncio.Queue[leyline_pb2.TelemetryPacket] | None = None
        self._kasmina_packet_drops: int = 0

    async def start(self) -> None:
        """Initialise subsystems and spawn background workers (Slice 1 & 2)."""

        self._configure_logging()
        self._ensure_secret_present()
        LOGGER.info("Starting Weatherlight supervisor")

        self._oona = await self._build_oona_client()
        self._urza_library, self._urza_runtime = self._build_urza_components()
        self._urza_worker = UrzaPrefetchWorker(self._oona, self._urza_library)
        self._kasmina_packet_queue = asyncio.Queue(maxsize=256)

        def _on_kasmina_packet(packet: leyline_pb2.TelemetryPacket) -> None:
            if self._kasmina_packet_queue is None:
                return
            try:
                self._kasmina_packet_queue.put_nowait(packet)
            except asyncio.QueueFull:
                self._kasmina_packet_drops += 1

        self._kasmina_manager = KasminaSeedManager(
            self._urza_runtime,
            packet_callback=_on_kasmina_packet,
        )
        self._kasmina_coordinator = KasminaPrefetchCoordinator(self._kasmina_manager, self._oona)
        self._kasmina_manager.set_prefetch(self._kasmina_coordinator)
        tamiyo_signature = SignatureContext.from_environment(DEFAULT_SECRET_ENV)
        self._tamiyo_service = TamiyoService(
            settings=self._settings,
            urza=self._urza_library,
            signature_context=tamiyo_signature,
        )

        self._register_worker(
            WorkerState(
                name="urza_prefetch",
                coroutine_factory=lambda: self._urza_worker.run_forever(interval_ms=200),
            )
        )
        self._register_worker(
            WorkerState(
                name="kasmina_prefetch",
                coroutine_factory=self._kasmina_supervisor_loop,
            )
        )
        self._register_worker(
            WorkerState(
                name="tamiyo_policy",
                coroutine_factory=self._tamiyo_policy_loop,
            )
        )

        for state in self._workers.values():
            state.task = asyncio.create_task(self._worker_wrapper(state), name=f"weatherlight.{state.name}")

        self._telemetry_task = asyncio.create_task(self._telemetry_loop(), name="weatherlight.telemetry")
        self._housekeeping_task = asyncio.create_task(self._housekeeping_loop(), name="weatherlight.housekeeping")
        self._kasmina_telemetry_task = asyncio.create_task(
            self._kasmina_telemetry_loop(),
            name="weatherlight.kasmina_telemetry",
        )
        # Rollback signal monitor (slice 2)
        if self._rollback_signal_name:
            # Always start the monitor; it will attempt to attach when available
            self._rollback_monitor_task = asyncio.create_task(self._rollback_signal_loop(), name="weatherlight.rollback_monitor")

    def initiate_shutdown(self) -> None:
        """Trigger graceful shutdown (idempotent)."""

        if not self._shutdown_requested.is_set():
            LOGGER.info("Shutdown requested")
            self._shutdown_requested.set()

    async def run(self) -> None:
        """Block until shutdown is requested, then drain workers."""

        await self._shutdown_requested.wait()
        await self._shutdown()
        await self._shutdown_complete.wait()

    async def shutdown(self) -> None:
        """External helper for tests to wait until shutdown completes."""

        self.initiate_shutdown()
        await self._shutdown()
        await self._shutdown_complete.wait()

    def set_tezzeret_metrics_provider(
        self, provider: Callable[[], dict[str, float]] | None
    ) -> None:
        """Register a callable that returns Tezzeret metrics for telemetry."""

        self._tezzeret_metrics_provider = provider

    def set_tezzeret_telemetry_provider(
        self, provider: Callable[[], leyline_pb2.TelemetryPacket | None] | None
    ) -> None:
        """Register a callable that returns Tezzeret telemetry packets."""

        self._tezzeret_telemetry_provider = provider

    def connect_tezzeret_forge(self, forge: "TezzeretForge") -> None:
        """Attach TezzeretForge telemetry streams to Weatherlight."""

        self.set_tezzeret_metrics_provider(forge.metrics_snapshot)
        self.set_tezzeret_telemetry_provider(forge.build_telemetry_packet)

    async def _rollback_signal_loop(self) -> None:
        """Monitor a shared rollback signal and publish emergency telemetry when triggered."""
        assert self._oona is not None
        while not self._shutdown_requested.is_set():
            try:
                # Lazy attach if signal not yet available
                if self._rollback_signal is None and self._rollback_signal_name:
                    try:
                        self._rollback_signal = SharedDeadlineSignal.attach(self._rollback_signal_name)
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue
                if self._rollback_signal.is_set():
                    now_s = time.monotonic()
                    # Cooldown: avoid floods (500ms)
                    if self._rollback_last_detect_s is not None and (now_s - self._rollback_last_detect_s) < 0.5:
                        self._rollback_skipped_total += 1
                        self._rollback_signal.clear()
                        await asyncio.sleep(0.05)
                        continue
                    self._rollback_last_detect_s = now_s
                    # Compute detection latency if timestamp present
                    latency_ms = 0.0
                    try:
                        ts_ms = self._rollback_signal.read_timestamp_ms()
                        if ts_ms is not None:
                            if self._rollback_last_detect_ts_ms is not None and ts_ms == self._rollback_last_detect_ts_ms:
                                self._rollback_signal.clear()
                                await asyncio.sleep(0.05)
                                continue
                            self._rollback_last_detect_ts_ms = ts_ms
                            now_ms = int(time.monotonic() * 1000)
                            latency_ms = max(0.0, float(now_ms - ts_ms))
                    except Exception:
                        latency_ms = 0.0
                    pkt = build_telemetry_packet(
                        packet_id=f"weatherlight-rollback-signal",
                        source="weatherlight",
                        level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
                        metrics=[TelemetryMetric("weatherlight.rollback.deadline_latency_ms", latency_ms, unit="ms")],
                        events=[TelemetryEvent(description="rollback_deadline_triggered", level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL, attributes={})],
                        health_status=leyline_pb2.HealthStatus.HEALTH_STATUS_UNHEALTHY,
                        health_summary="rollback_deadline_triggered",
                        health_indicators={"priority": "MESSAGE_PRIORITY_HIGH"},
                    )
                    await self._oona.publish_telemetry(pkt, priority=leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH)
                    self._rollback_detections_total += 1
                    # Clear to avoid repeated floods; rely on trainer to set on each deadline
                    self._rollback_signal.clear()
            except Exception:
                pass
            await asyncio.sleep(0.1)

    async def _build_oona_client(self) -> OonaClient:
        hostname = socket.gethostname().replace(" ", "-")
        stream_config = StreamConfig(
            normal_stream=self._settings.oona_normal_stream,
            emergency_stream=self._settings.oona_emergency_stream,
            telemetry_stream=self._settings.oona_telemetry_stream,
            policy_stream=self._settings.oona_policy_stream,
            group="weatherlight",
            consumer=f"weatherlight-{hostname}",
            dead_letter_stream="oona.deadletter",
            max_stream_length=10_000,
            message_ttl_ms=self._settings.oona_message_ttl_ms,
            kernel_freshness_window_ms=self._settings.kernel_freshness_window_ms,
            kernel_nonce_cache_size=self._settings.kernel_nonce_cache_size,
        )
        client = OonaClient(redis_url=self._settings.redis_url, config=stream_config)
        await client.ensure_consumer_group()
        return client

    def _build_urza_components(self) -> tuple[UrzaLibrary, UrzaRuntime]:
        artifact_dir = Path(self._settings.urza_artifact_dir)
        library_root = artifact_dir.parent if artifact_dir.parent != artifact_dir else artifact_dir
        library_root.mkdir(parents=True, exist_ok=True)
        library = UrzaLibrary(
            root=library_root,
            database_url=self._settings.urza_database_url,
            cache_ttl_seconds=self._settings.urza_cache_ttl_seconds,
        )
        runtime = UrzaRuntime(library)
        return library, runtime

    def _register_worker(self, state: WorkerState) -> None:
        if state.name in self._workers:
            msg = f"Worker '{state.name}' already registered"
            raise ValueError(msg)
        self._workers[state.name] = state

    async def _worker_wrapper(self, state: WorkerState) -> None:
        backoff_seconds = 1.0
        max_backoff = 30.0
        jitter_factor = 0.25
        while not self._shutdown_requested.is_set():
            state.running = True
            state.backoff_seconds = 0.0
            try:
                await state.coroutine_factory()
                return
            except asyncio.CancelledError:
                state.running = False
                raise
            except Exception as exc:  # pragma: no cover - exercised in integration tests
                state.running = False
                state.restart_count += 1
                state.last_error = f"{type(exc).__name__}: {exc}"
                state.last_error_at = time.time()
                self._last_error_ts = state.last_error_at
                LOGGER.exception("Worker %s crashed (restart #%d)", state.name, state.restart_count)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)
                sleep_for = backoff_seconds + random.uniform(0, backoff_seconds * jitter_factor)
                state.backoff_seconds = sleep_for
                try:
                    await asyncio.wait_for(self._shutdown_requested.wait(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    continue
        state.running = False

    async def _kasmina_supervisor_loop(self) -> None:
        assert self._kasmina_coordinator is not None
        self._kasmina_coordinator.start()
        try:
            while not self._shutdown_requested.is_set():
                # Monitor child tasks for unexpected failures via coordinator accessor.
                issue = self._kasmina_coordinator.poll_task_issue()
                if issue is not None:
                    raise issue
                await asyncio.sleep(1.0)
        finally:
            with contextlib.suppress(Exception):
                await self._kasmina_coordinator.close()

    async def _tamiyo_policy_loop(self) -> None:
        assert self._tamiyo_service is not None and self._oona is not None
        while not self._shutdown_requested.is_set():
            await self._tamiyo_service.consume_policy_updates(self._oona, block_ms=500)

    async def _telemetry_loop(self) -> None:
        assert self._oona is not None
        while not self._shutdown_requested.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_requested.wait(),
                    timeout=self.TELEMETRY_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass
            if self._shutdown_requested.is_set():
                break
            try:
                await self._flush_telemetry_once()
            except asyncio.CancelledError:  # pragma: no cover - cancellation path
                raise
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.warning("Failed to publish telemetry: %s", exc)

    async def _kasmina_telemetry_loop(self) -> None:
        assert self._kasmina_manager is not None and self._oona is not None
        queue = self._kasmina_packet_queue
        while not self._shutdown_requested.is_set():
            packets: list[leyline_pb2.TelemetryPacket] = []
            if queue is not None:
                try:
                    packet = await asyncio.wait_for(queue.get(), timeout=0.1)
                    packets.append(packet)
                    queue.task_done()
                    while True:
                        try:
                            pkt = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        packets.append(pkt)
                        queue.task_done()
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0.1)
            packets.extend(self._kasmina_manager.drain_telemetry_packets())
            if not packets:
                continue
            for packet in packets:
                priority_name = packet.system_health.indicators.get(
                    "priority",
                    "MESSAGE_PRIORITY_NORMAL",
                )
                try:
                    priority_enum = leyline_pb2.MessagePriority.Value(priority_name)
                except ValueError:
                    priority_enum = leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL
                try:
                    await self._oona.publish_telemetry(packet, priority=priority_enum)
                except asyncio.CancelledError:  # pragma: no cover
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Failed to forward Kasmina telemetry: %s", exc)

    async def _housekeeping_loop(self) -> None:
        assert self._oona is not None
        while not self._shutdown_requested.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_requested.wait(),
                    timeout=self.HOUSEKEEPING_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass
            if self._shutdown_requested.is_set():
                break
            try:
                await self._oona.housekeeping()
                if self._urza_library is not None:
                    self._urza_library.maintenance()
            except asyncio.CancelledError:  # pragma: no cover - cancellation path
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.debug("Housekeeping failed: %s", exc)

    async def _build_telemetry_packet(self) -> leyline_pb2.TelemetryPacket:
        assert self._oona is not None and self._urza_worker is not None
        try:
            if self._rollback_signal_name:
                # Probe a fresh attachment to handle cases where the segment was recreated
                try:
                    probe = SharedDeadlineSignal.attach(self._rollback_signal_name)
                except Exception:
                    probe = None
                if probe is not None:
                    try:
                        # Fast path: flag set
                        if probe.is_set():
                            self._rollback_detections_total += 1
                            self._rollback_last_detect_s = time.monotonic()
                            with contextlib.suppress(Exception):
                                probe.clear()
                        else:
                            # Fallback: detect via timestamp changes
                            ts_ms = probe.read_timestamp_ms()
                            if ts_ms is not None and (self._rollback_last_detect_ts_ms is None or ts_ms != self._rollback_last_detect_ts_ms):
                                self._rollback_last_detect_ts_ms = ts_ms
                                self._rollback_last_detect_s = time.monotonic()
                                self._rollback_detections_total += 1
                        # Cache the attachment for the monitor loop if not set already
                        if self._rollback_signal is None:
                            self._rollback_signal = probe
                        else:
                            with contextlib.suppress(Exception):
                                probe.close()
                    except Exception:
                        with contextlib.suppress(Exception):
                            probe.close()
        except Exception:
            pass
        metrics: list[TelemetryMetric] = []
        workers_running = sum(1 for state in self._workers.values() if state.running)
        workers_backing_off = sum(1 for state in self._workers.values() if state.backoff_seconds > 0)
        uptime = time.monotonic() - self._start_time
        metrics.append(TelemetryMetric("weatherlight.tasks.running", float(workers_running), unit="count"))
        metrics.append(
            TelemetryMetric("weatherlight.tasks.backing_off", float(workers_backing_off), unit="count")
        )
        metrics.append(TelemetryMetric("weatherlight.uptime_s", uptime, unit="seconds"))
        metrics.append(
            TelemetryMetric(
                "weatherlight.restarts_total",
                float(sum(state.restart_count for state in self._workers.values())),
                unit="count",
            )
        )
        if self._last_error_ts is not None:
            metrics.append(
                TelemetryMetric("weatherlight.last_error_ts", self._last_error_ts, unit="unix_s")
            )
        for state in self._workers.values():
            metrics.append(
                TelemetryMetric(
                    f"weatherlight.worker.{state.name}.running",
                    1.0 if state.running else 0.0,
                    unit="bool",
                )
            )
            metrics.append(
                TelemetryMetric(
                    f"weatherlight.worker.{state.name}.restarts",
                    float(state.restart_count),
                    unit="count",
                )
            )
            if state.backoff_seconds > 0:
                metrics.append(
                    TelemetryMetric(
                        f"weatherlight.worker.{state.name}.backoff_seconds",
                        state.backoff_seconds,
                        unit="seconds",
                    )
                )
        urza_metrics = self._urza_worker.metrics
        metrics.extend(
            (
                TelemetryMetric("urza.prefetch.hits", float(urza_metrics.hits), unit="count"),
                TelemetryMetric("urza.prefetch.misses", float(urza_metrics.misses), unit="count"),
                TelemetryMetric("urza.prefetch.errors", float(urza_metrics.errors), unit="count"),
                TelemetryMetric("urza.prefetch.latency_ms", float(urza_metrics.latency_ms), unit="ms"),
            )
        )
        if self._urza_library is not None:
            for name, value in self._urza_library.metrics_snapshot().items():
                metrics.append(TelemetryMetric(f"urza.library.{name}", float(value)))
        oona_snapshot = await self._oona.metrics_snapshot()
        for name, value in oona_snapshot.items():
            metrics.append(TelemetryMetric(f"oona.{name}", float(value)))
        if self._kasmina_packet_drops:
            metrics.append(
                TelemetryMetric(
                    "kasmina.telemetry.dropped_total",
                    float(self._kasmina_packet_drops),
                    unit="count",
                )
            )
        # Append rollback monitor counters
        det_total = self._rollback_detections_total
        if det_total:
            metrics.append(TelemetryMetric("weatherlight.rollback.detections_total", float(det_total), unit="count"))
        if self._rollback_skipped_total:
            metrics.append(TelemetryMetric("weatherlight.rollback.skipped_total", float(self._rollback_skipped_total), unit="count"))
        if self._rollback_last_detect_s is not None:
            metrics.append(TelemetryMetric("weatherlight.rollback.last_detect_ms_ago", float((time.monotonic() - self._rollback_last_detect_s) * 1000.0), unit="ms"))
        if self._tezzeret_metrics_provider is not None:
            try:
                tezzeret_metrics = self._tezzeret_metrics_provider()
            except Exception:  # pragma: no cover - defensive
                tezzeret_metrics = {}
            for name, value in tezzeret_metrics.items():
                metrics.append(TelemetryMetric(name, float(value)))
        events: list[TelemetryEvent] = []
        health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_HEALTHY
        health_summary = "steady"
        if workers_backing_off or any(state.last_error for state in self._workers.values()):
            health_status = leyline_pb2.HealthStatus.HEALTH_STATUS_DEGRADED
            health_summary = "backoff" if workers_backing_off else "recent_error"
        if self._last_error_ts is not None:
            events.append(
                TelemetryEvent(
                    description="worker_error",
                    level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
                    attributes={
                        "last_error_ts": f"{self._last_error_ts:.3f}",
                    },
                )
            )
        packet = build_telemetry_packet(
            packet_id=f"weatherlight-{uuid.uuid4()}",
            source="weatherlight",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
            metrics=metrics,
            events=events,
            health_status=health_status,
            health_summary=health_summary,
            health_indicators={
                "workers_running": str(workers_running),
                "workers_backing_off": str(workers_backing_off),
            },
        )
        return packet


    async def probe_rollback_signal_for_test(self) -> bool:
        """Test helper: synchronously detect and clear rollback signal if set."""
        try:
            if not self._rollback_signal_name:
                return False
            try:
                probe = SharedDeadlineSignal.attach(self._rollback_signal_name)
            except Exception:
                return False
            detected = False
            try:
                if probe.is_set():
                    self._rollback_detections_total += 1
                    self._rollback_last_detect_s = time.monotonic()
                    detected = True
                    with contextlib.suppress(Exception):
                        probe.clear()
                else:
                    ts_ms = probe.read_timestamp_ms()
                    if ts_ms is not None and (self._rollback_last_detect_ts_ms is None or ts_ms != self._rollback_last_detect_ts_ms):
                        self._rollback_last_detect_ts_ms = ts_ms
                        self._rollback_last_detect_s = time.monotonic()
                        self._rollback_detections_total += 1
                        detected = True
            finally:
                if self._rollback_signal is None:
                    self._rollback_signal = probe
                else:
                    with contextlib.suppress(Exception):
                        probe.close()
            return detected
        except Exception:
            return False

    def get_rollback_detection_count(self) -> int:
        """Expose the rollback detection counter for verification in tests."""

        return self._rollback_detections_total

    async def _flush_telemetry_once(self) -> None:
        assert self._oona is not None
        packets: list[leyline_pb2.TelemetryPacket] = []
        packets.append(await self._build_telemetry_packet())
        tezzeret_packet = self._build_tezzeret_packet()
        if tezzeret_packet is not None:
            packets.append(tezzeret_packet)
        for packet in packets:
            priority = self._telemetry_priority(packet)
            await self._oona.publish_telemetry(packet, priority=priority)
        await self._oona.emit_metrics_telemetry(
            source_subsystem="weatherlight.oona",
            level=leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
        )

    def _build_tezzeret_packet(self) -> leyline_pb2.TelemetryPacket | None:
        if self._tezzeret_telemetry_provider is None:
            return None
        try:
            packet = self._tezzeret_telemetry_provider()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("Tezzeret telemetry provider failed: %s", exc)
            return None
        if packet is None:
            return None
        return packet

    @staticmethod
    def _telemetry_priority(packet: leyline_pb2.TelemetryPacket) -> leyline_pb2.MessagePriority:
        level = packet.level
        if level in (
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_WARNING,
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_ERROR,
            leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_CRITICAL,
        ):
            return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH
        return leyline_pb2.MessagePriority.MESSAGE_PRIORITY_NORMAL

    def _configure_logging(self) -> None:
        if logging.getLogger().handlers:
            return
        level = getattr(logging, str(self._settings.log_level).upper(), logging.INFO)
        logging.basicConfig(level=level)

    def _ensure_secret_present(self) -> None:
        secret = os.getenv(DEFAULT_SECRET_ENV, "").strip()
        if not secret:
            raise RuntimeError(
                "Weatherlight requires ESPER_LEYLINE_SECRET for HMAC signing"
            )

    async def _shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        LOGGER.info("Stopping Weatherlight supervisor")
        tasks = [state.task for state in self._workers.values() if state.task is not None]
        for task in tasks:
            task.cancel()
        if self._telemetry_task is not None:
            self._telemetry_task.cancel()
        if self._housekeeping_task is not None:
            self._housekeeping_task.cancel()
        if self._kasmina_telemetry_task is not None:
            self._kasmina_telemetry_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(
                *[task for task in tasks if task is not None],
                *[
                    task
                    for task in (
                        self._telemetry_task,
                        self._housekeeping_task,
                        self._kasmina_telemetry_task,
                    )
                    if task is not None
                ],
                return_exceptions=True,
            )
        with contextlib.suppress(Exception):
            if self._kasmina_coordinator is not None:
                await self._kasmina_coordinator.close()
        with contextlib.suppress(Exception):
            if self._oona is not None:
                await self._oona.close()
        self._shutdown_complete.set()


async def run_service() -> int:
    """Entry point used by the CLI shim and tests."""

    service = WeatherlightService()
    await service.start()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        LOGGER.info("Received %s", sig.name)
        service.initiate_shutdown()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, _handle_signal, signum)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(signum, lambda *_args, s=signum: _handle_signal(s))

    await service.run()
    return 0


def main() -> None:
    """CLI entrypoint."""

    exit_code = asyncio.run(run_service())
    raise SystemExit(exit_code)


__all__ = ["run_service", "main", "WeatherlightService"]
