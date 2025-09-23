"""Redis Streams client scaffold for Oona with proto-delta features."""

from __future__ import annotations

import inspect
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import redis.asyncio as aioredis

from esper.leyline import leyline_pb2
from esper.security.signing import DEFAULT_SECRET_ENV, SignatureContext, sign, verify


@dataclass(slots=True)
class StreamConfig:
    normal_stream: str
    emergency_stream: str
    group: str
    consumer: str = "oona-client"
    max_stream_length: int | None = None
    emergency_threshold: int | None = None
    telemetry_stream: str | None = None
    policy_stream: str | None = None
    backpressure_drop_threshold: int | None = None
    kernel_request_stream: str | None = None
    kernel_ready_stream: str | None = None
    kernel_error_stream: str | None = None
    retry_max_attempts: int = 5
    retry_idle_ms: int = 60_000
    retry_batch_size: int = 50
    dead_letter_stream: str | None = None
    message_ttl_ms: int | None = None
    kernel_nonce_cache_size: int = 2048
    kernel_freshness_window_ms: int = 120_000
    # Emergency rate limiting (token bucket: max per minute)
    emergency_max_per_min: int | None = 120


@dataclass(slots=True)
class OonaMessage:
    """Container passed to message handlers."""

    stream: str
    message_id: str
    message_type: leyline_pb2.BusMessageType.ValueType
    payload: bytes
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BreakerSnapshot:
    state: int
    failure_count: int
    success_count: int


class CircuitBreaker:
    """Simple circuit breaker used by the Oona client."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        success_threshold: int = 1,
        timeout_ms: float = 30_000.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._failure_threshold = max(failure_threshold, 1)
        self._success_threshold = max(success_threshold, 1)
        self._timeout_ms = max(timeout_ms, 0.0)
        self._clock = clock or time.monotonic
        self._state = leyline_pb2.CIRCUIT_STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._open_until: float | None = None

    def allow(self) -> tuple[bool, BreakerSnapshot | None]:
        if self._state == leyline_pb2.CIRCUIT_STATE_OPEN:
            if self._open_until is not None and self._clock() < self._open_until:
                return False, self.snapshot()
            self._state = leyline_pb2.CIRCUIT_STATE_HALF_OPEN
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
        return True, None

    def record_failure(self) -> BreakerSnapshot:
        self._failure_count += 1
        self._success_count = 0
        if self._failure_count >= self._failure_threshold:
            self._state = leyline_pb2.CIRCUIT_STATE_OPEN
            self._open_until = self._clock() + (self._timeout_ms / 1000.0)
        return self.snapshot()

    def record_success(self) -> BreakerSnapshot:
        if self._state == leyline_pb2.CIRCUIT_STATE_OPEN:
            return self.snapshot()
        if self._state == leyline_pb2.CIRCUIT_STATE_CLOSED:
            self._failure_count = 0
            return self.snapshot()
        self._success_count += 1
        if self._success_count >= self._success_threshold:
            self._state = leyline_pb2.CIRCUIT_STATE_CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._open_until = None
        return self.snapshot()

    def force_open(self) -> None:
        self._state = leyline_pb2.CIRCUIT_STATE_OPEN
        self._open_until = self._clock() + (self._timeout_ms / 1000.0)

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
        )


class OonaClient:
    """High-level Oona client for interacting with Redis Streams."""

    def __init__(
        self,
        redis_url: str,
        config: StreamConfig,
        *,
        redis_client: aioredis.Redis | None = None,
        signing_context: SignatureContext | None = None,
    ) -> None:
        self._config = config
        self._redis = redis_client or aioredis.from_url(redis_url)
        self._telemetry_stream = config.telemetry_stream or config.normal_stream
        self._policy_stream = config.policy_stream or config.normal_stream
        self._kernel_request_stream = (
            config.kernel_request_stream or "oona.kernels.requests"
        )
        self._kernel_ready_stream = (
            config.kernel_ready_stream or "oona.kernels.ready"
        )
        self._kernel_error_stream = (
            config.kernel_error_stream or "oona.kernels.errors"
        )
        self._metrics: dict[str, float] = {
            "publish_total": 0.0,
            "publish_rerouted": 0.0,
            "publish_dropped": 0.0,
            "queue_depth_max": 0.0,
            "retry_total": 0.0,
            "dead_letter_total": 0.0,
            "breaker_publish_open": 0.0,
            "breaker_consume_open": 0.0,
            "publish_latency_ms": 0.0,
            "consume_latency_ms": 0.0,
            "kernel_stale_dropped": 0.0,
            "kernel_replay_dropped": 0.0,
            "emergency_published": 0.0,
            "emergency_rate_dropped": 0.0,
        }
        self._signing_context = signing_context or self._load_signing_context()
        self._publish_breaker = CircuitBreaker()
        self._consume_breaker = CircuitBreaker()
        self._conservative_mode = False
        self._kernel_requests_seen: OrderedDict[str, float] = OrderedDict()
        self._kernel_responses_seen: OrderedDict[str, float] = OrderedDict()
        # Emergency rate limiter
        self._em_tokens: float = float(self._config.emergency_max_per_min or 0)
        self._em_refill_rate_per_s: float = (
            float(self._config.emergency_max_per_min) / 60.0
            if self._config.emergency_max_per_min and self._config.emergency_max_per_min > 0
            else 0.0
        )
        self._em_last_refill: float = time.monotonic()

    async def close(self) -> None:
        await self._redis.close()

    async def ensure_consumer_group(self) -> None:
        """Create consumer groups for configured streams if needed."""

        streams = {
            self._config.normal_stream,
            self._config.emergency_stream,
            self._telemetry_stream,
            self._policy_stream,
            self._kernel_request_stream,
            self._kernel_ready_stream,
            self._kernel_error_stream,
        }
        if self._config.dead_letter_stream:
            streams.add(self._config.dead_letter_stream)
        for stream in streams:
            try:
                await self._redis.xgroup_create(stream, self._config.group, id="$", mkstream=True)
            except aioredis.ResponseError as exc:  # group exists
                if "BUSYGROUP" in str(exc):
                    continue
                if "NOGROUP" in str(exc) or "No such key" in str(exc):
                    await self._redis.xadd(stream, {"bootstrap": b""})
                    await self._redis.xgroup_create(stream, self._config.group, id="0-0")
                else:
                    raise

    async def publish_state(
        self,
        packet: leyline_pb2.SystemStatePacket,
        *,
        emergency: bool = False,
    ) -> bool:
        return await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=emergency,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_SYSTEM_STATE,
            payload=packet.SerializeToString(),
        )

    async def publish_command(self, command: leyline_pb2.AdaptationCommand) -> bool:
        return await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_ADAPTATION_COMMAND,
            payload=command.SerializeToString(),
        )

    async def publish_field_report(self, report: leyline_pb2.FieldReport) -> bool:
        return await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_FIELD_REPORT,
            payload=report.SerializeToString(),
        )

    async def publish_telemetry(
        self,
        packet: leyline_pb2.TelemetryPacket,
        *,
        priority: leyline_pb2.MessagePriority.ValueType | None = None,
    ) -> bool:
        emergency = False
        if priority is not None:
            emergency = priority in {
                leyline_pb2.MessagePriority.MESSAGE_PRIORITY_HIGH,
                leyline_pb2.MessagePriority.MESSAGE_PRIORITY_CRITICAL,
            }
        return await self._publish_proto(
            preferred_stream=self._telemetry_stream,
            emergency_flag=emergency,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_TELEMETRY,
            payload=packet.SerializeToString(),
        )

    async def publish_policy_update(self, update: leyline_pb2.PolicyUpdate) -> bool:
        return await self._publish_proto(
            preferred_stream=self._policy_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_POLICY_UPDATE,
            payload=update.SerializeToString(),
        )

    async def publish_bsds_issued(self, report: leyline_pb2.BSDSIssued) -> bool:
        """Publish a BSDSIssued event to the normal stream."""
        return await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_BSDS_ISSUED,
            payload=report.SerializeToString(),
        )

    async def publish_bsds_failed(self, report: leyline_pb2.BSDSFailed) -> bool:
        """Publish a BSDSFailed event to the normal stream."""
        return await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_BSDS_FAILED,
            payload=report.SerializeToString(),
        )

    async def publish_kernel_prefetch_request(
        self, request: leyline_pb2.KernelPrefetchRequest
    ) -> bool:
        serialized = request.SerializeToString()
        return await self._publish_proto(
            preferred_stream=self._kernel_request_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_REQUEST,
            payload=serialized,
            drop_threshold=self._config.backpressure_drop_threshold,
        )

    async def publish_kernel_artifact_ready(
        self, ready: leyline_pb2.KernelArtifactReady
    ) -> bool:
        serialized = ready.SerializeToString()
        return await self._publish_proto(
            preferred_stream=self._kernel_ready_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY,
            payload=serialized,
        )

    async def publish_kernel_artifact_error(
        self, error: leyline_pb2.KernelArtifactError
    ) -> bool:
        serialized = error.SerializeToString()
        return await self._publish_proto(
            preferred_stream=self._kernel_error_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR,
            payload=serialized,
        )

    async def publish_kernel_catalog_update(
        self, update: leyline_pb2.KernelCatalogUpdate
    ) -> bool:
        serialized = update.SerializeToString()
        return await self._publish_proto(
            preferred_stream=self._kernel_ready_stream,
            emergency_flag=False,
            message_type=leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_CATALOG_UPDATE,
            payload=serialized,
        )

    async def consume_kernel_requests(
        self,
        handler: Callable[[OonaMessage], Awaitable[None] | None],
        *,
        count: int = 1,
        block_ms: int = 1000,
    ) -> None:
        await self.consume(handler, stream=self._kernel_request_stream, count=count, block_ms=block_ms)

    async def consume_kernel_ready(
        self,
        handler: Callable[[OonaMessage], Awaitable[None] | None],
        *,
        count: int = 1,
        block_ms: int = 1000,
    ) -> None:
        await self.consume(handler, stream=self._kernel_ready_stream, count=count, block_ms=block_ms)

    async def consume_kernel_errors(
        self,
        handler: Callable[[OonaMessage], Awaitable[None] | None],
        *,
        count: int = 1,
        block_ms: int = 1000,
    ) -> None:
        await self.consume(handler, stream=self._kernel_error_stream, count=count, block_ms=block_ms)

    async def consume(
        self,
        handler: Callable[[OonaMessage], Awaitable[None] | None],
        *,
        stream: str | None = None,
        count: int = 1,
        block_ms: int = 1000,
    ) -> None:
        target_stream = stream or self._config.normal_stream
        allow, _ = self._consume_breaker.allow()
        if not allow:
            self._metrics["breaker_consume_open"] += 1.0
            self._conservative_mode = True
            return
        await self._claim_stale_messages(target_stream)
        response = await self._redis.xreadgroup(
            groupname=self._config.group,
            consumername=self._config.consumer,
            streams={target_stream: ">"},
            count=count,
            block=block_ms,
        )
        if not response:
            self._consume_breaker.record_success()
            return

        for stream_name, messages in response:
            for message_id, payload in messages:
                envelope_bytes = bytes(payload.get(b"payload", b""))
                sig = payload.get(b"signature", b"").decode("utf-8")
                if not self._verify_payload(envelope_bytes, sig):
                    await self._redis.xack(stream_name, self._config.group, message_id)
                    continue
                envelope = leyline_pb2.BusEnvelope()
                envelope.ParseFromString(envelope_bytes)
                if not self._enforce_kernel_freshness(envelope):
                    await self._redis.xack(stream_name, self._config.group, message_id)
                    continue
                message = OonaMessage(
                    stream=stream_name,
                    message_id=message_id,
                    message_type=envelope.message_type,
                    payload=envelope.payload,
                    attributes=dict(envelope.attributes),
                )
                start = time.perf_counter()
                try:
                    result = handler(message)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    await self._handle_handler_error(stream_name, message_id, envelope)
                    self._publish_breaker.record_failure()
                    continue
                await self._redis.xack(stream_name, self._config.group, message_id)
                self._metrics["consume_latency_ms"] = (time.perf_counter() - start) * 1000.0
                self._consume_breaker.record_success()
                self._conservative_mode = False

    async def backlog(self, stream: str | None = None) -> int:
        target = stream or self._config.normal_stream
        try:
            summary = await self._redis.xpending(target, self._config.group)
        except aioredis.ResponseError:
            return 0
        return int(summary["pending"])

    async def stream_length(self, stream: str | None = None) -> int:
        target = stream or self._config.normal_stream
        return int(await self._redis.xlen(target))

    async def metrics_snapshot(self) -> dict[str, float]:
        snapshot = dict(self._metrics)
        snapshot["queue_depth_normal"] = float(await self._redis.xlen(self._config.normal_stream))
        snapshot["queue_depth_emergency"] = float(await self._redis.xlen(self._config.emergency_stream))
        snapshot["queue_depth_kernel_requests"] = float(
            await self._redis.xlen(self._kernel_request_stream)
        )
        snapshot["queue_depth_kernel_ready"] = float(
            await self._redis.xlen(self._kernel_ready_stream)
        )
        snapshot["queue_depth_kernel_errors"] = float(
            await self._redis.xlen(self._kernel_error_stream)
        )
        snapshot["publish_breaker_state"] = float(self._publish_breaker.snapshot().state)
        snapshot["consume_breaker_state"] = float(self._consume_breaker.snapshot().state)
        snapshot["conservative_mode"] = 1.0 if self._conservative_mode else 0.0
        # Flatten per-source emergency counters
        def _san(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s) or "unknown"
        for src, val in getattr(self, "_em_src_published", {}).items():
            snapshot[f"emergency_published.src.{_san(src)}"] = float(val)
        for src, val in getattr(self, "_em_src_dropped", {}).items():
            snapshot[f"emergency_rate_dropped.src.{_san(src)}"] = float(val)
        return snapshot

    async def health_snapshot(self) -> dict[str, float]:
        publish = self._publish_breaker.snapshot()
        consume = self._consume_breaker.snapshot()
        return {
            "publish_state": float(publish.state),
            "publish_failures": float(publish.failure_count),
            "consume_state": float(consume.state),
            "consume_failures": float(consume.failure_count),
            "conservative_mode": 1.0 if self._conservative_mode else 0.0,
        }

    @property
    def telemetry_stream(self) -> str:
        return self._telemetry_stream

    @property
    def policy_stream(self) -> str:
        return self._policy_stream

    @property
    def normal_stream(self) -> str:
        return self._config.normal_stream

    @property
    def kernel_request_stream(self) -> str:
        return self._kernel_request_stream

    @property
    def kernel_ready_stream(self) -> str:
        return self._kernel_ready_stream

    @property
    def kernel_error_stream(self) -> str:
        return self._kernel_error_stream

    async def housekeeping(self) -> dict[str, float]:
        """Run TTL-based housekeeping across managed streams."""

        ttl_ms = self._config.message_ttl_ms
        if not ttl_ms:
            return {}
        min_id = f"{max(0, int(time.time() * 1000) - ttl_ms)}-0"
        trimmed: dict[str, float] = {}
        for stream in {
            self._config.normal_stream,
            self._config.emergency_stream,
            self._telemetry_stream,
            self._policy_stream,
            self._kernel_request_stream,
            self._kernel_ready_stream,
            self._kernel_error_stream,
        }:
            trimmed[stream] = float(
                await self._redis.xtrim(stream, minid=min_id, approximate=True)
            )
        return trimmed

    async def _publish_proto(
        self,
        *,
        preferred_stream: str,
        emergency_flag: bool,
        message_type: leyline_pb2.BusMessageType.ValueType,
        payload: bytes,
        drop_threshold: int | None = None,
    ) -> bool:
        allow, _ = self._publish_breaker.allow()
        if not allow:
            self._metrics["breaker_publish_open"] += 1.0
            self._conservative_mode = True
            return False
        stream, rerouted, dropped, backlog = await self._resolve_stream(
            preferred_stream, emergency_flag, drop_threshold=drop_threshold
        )
        self._metrics["queue_depth_max"] = max(self._metrics["queue_depth_max"], float(backlog))
        if dropped:
            self._metrics["publish_dropped"] += 1.0
            self._publish_breaker.record_failure()
            return False
        # Emergency rate limiter
        if stream == self._config.emergency_stream and self._config.emergency_max_per_min:
            now = time.monotonic()
            # Refill tokens
            elapsed = max(0.0, now - self._em_last_refill)
            if self._em_refill_rate_per_s > 0:
                self._em_tokens = min(
                    float(self._config.emergency_max_per_min),
                    self._em_tokens + elapsed * self._em_refill_rate_per_s,
                )
            self._em_last_refill = now
            if self._em_tokens < 1.0:
                # Drop due to rate limit
                self._metrics["emergency_rate_dropped"] += 1.0
                return False
            self._em_tokens -= 1.0
        if rerouted:
            self._metrics["publish_rerouted"] += 1.0
        envelope = leyline_pb2.BusEnvelope(message_type=message_type, payload=payload)
        envelope_bytes = envelope.SerializeToString()
        fields = {"payload": envelope_bytes}
        signature = self._generate_signature(envelope_bytes)
        if signature:
            fields["signature"] = signature.encode("utf-8")
        kwargs = {}
        if self._config.max_stream_length and stream in {
            self._config.normal_stream,
            self._kernel_request_stream,
        }:
            kwargs["maxlen"] = self._config.max_stream_length
            kwargs["approximate"] = True
        start = time.perf_counter()
        try:
            await self._redis.xadd(stream, fields, **kwargs)
            length_after = await self._redis.xlen(stream)
        except Exception:  # pragma: no cover - surfaced via tests
            self._metrics["breaker_publish_open"] += 1.0
            self._publish_breaker.record_failure()
            self._conservative_mode = True
            return False
        self._metrics["publish_latency_ms"] = (time.perf_counter() - start) * 1000.0
        self._metrics["queue_depth_max"] = max(self._metrics["queue_depth_max"], float(length_after))
        self._metrics["publish_total"] += 1.0
        self._publish_breaker.record_success()
        if rerouted or stream == self._config.emergency_stream:
            if stream == self._config.emergency_stream:
                self._metrics["emergency_published"] += 1.0
            self._conservative_mode = True
        else:
            self._conservative_mode = False
        return True

    async def _resolve_stream(
        self,
        preferred: str,
        emergency_flag: bool,
        *,
        drop_threshold: int | None = None,
    ) -> tuple[str, bool, bool, int]:
        if emergency_flag or self._conservative_mode:
            return self._config.emergency_stream, emergency_flag, False, 0

        backlog = await self._redis.xlen(preferred)
        pending = 0
        try:
            summary = await self._redis.xpending(preferred, self._config.group)
            if isinstance(summary, (list, tuple)):
                pending = int(summary[0]) if summary else 0
            elif isinstance(summary, dict) and "pending" in summary:
                pending = int(summary["pending"])
        except aioredis.ResponseError:
            pending = backlog

        effective_drop = drop_threshold
        if effective_drop is None and preferred == self._config.normal_stream:
            effective_drop = self._config.backpressure_drop_threshold

        active_depth = max(backlog, pending)

        if (
            effective_drop is not None
            and preferred == self._kernel_request_stream
            and active_depth >= effective_drop
        ):
            return preferred, False, True, int(backlog)

        if (
            effective_drop is not None
            and preferred == self._config.normal_stream
            and active_depth >= effective_drop
        ):
            return preferred, False, True, int(backlog)

        threshold = self._config.emergency_threshold
        if threshold is not None and preferred == self._config.normal_stream and backlog >= threshold:
            return self._config.emergency_stream, True, False, int(backlog)

        return preferred, False, False, int(backlog)

    def _load_signing_context(self) -> SignatureContext | None:
        env_var = DEFAULT_SECRET_ENV
        if env_var in os.environ:
            return SignatureContext.from_environment(env_var)
        return None

    def _generate_signature(self, payload: bytes) -> str | None:
        if not self._signing_context:
            return None
        return sign(payload, self._signing_context)

    def _verify_payload(self, payload: bytes, signature: str) -> bool:
        if not self._signing_context:
            return True
        if not signature:
            return False
        return verify(payload, signature, self._signing_context)

    def _enforce_kernel_freshness(self, envelope: leyline_pb2.BusEnvelope) -> bool:
        message_type = envelope.message_type
        if message_type == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_REQUEST:
            request = leyline_pb2.KernelPrefetchRequest()
            request.ParseFromString(envelope.payload)
            issued_at_ms = _timestamp_to_ms(request.issued_at)
            if issued_at_ms is None:
                self._metrics["kernel_stale_dropped"] += 1.0
                return False
            now_ms = time.time() * 1000.0
            if self._config.kernel_freshness_window_ms > 0:
                self._purge_kernel_requests(now_ms)
                if issued_at_ms < now_ms - self._config.kernel_freshness_window_ms:
                    self._metrics["kernel_stale_dropped"] += 1.0
                    return False
            if request.request_id in self._kernel_requests_seen:
                self._metrics["kernel_replay_dropped"] += 1.0
                return False
            self._register_kernel_request(request.request_id, issued_at_ms)
            return True
        if message_type in {
            leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY,
            leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR,
        }:
            request_id = _extract_request_id(envelope)
            if not request_id:
                return True
            if request_id in self._kernel_responses_seen:
                self._metrics["kernel_replay_dropped"] += 1.0
                return False
            self._kernel_responses_seen[request_id] = time.time() * 1000.0
            self._trim_kernel_cache(self._kernel_responses_seen)
            return True
        return True

    def _register_kernel_request(self, request_id: str, issued_at_ms: float) -> None:
        if request_id in self._kernel_requests_seen:
            self._kernel_requests_seen.pop(request_id)
        self._kernel_requests_seen[request_id] = issued_at_ms
        self._trim_kernel_cache(self._kernel_requests_seen)

    def _purge_kernel_requests(self, now_ms: float) -> None:
        if not self._kernel_requests_seen or self._config.kernel_freshness_window_ms <= 0:
            return
        threshold = now_ms - self._config.kernel_freshness_window_ms
        while self._kernel_requests_seen:
            oldest_id, oldest_ms = next(iter(self._kernel_requests_seen.items()))
            if oldest_ms >= threshold:
                break
            self._kernel_requests_seen.pop(oldest_id)

    def _trim_kernel_cache(self, cache: OrderedDict[str, float]) -> None:
        limit = max(self._config.kernel_nonce_cache_size, 1)
        while len(cache) > limit:
            cache.popitem(last=False)

    async def _claim_stale_messages(self, stream: str) -> None:
        if self._config.retry_idle_ms <= 0:
            return
        try:
            cursor = "0-0"
            while True:
                result = await self._redis.xautoclaim(
                    stream,
                    self._config.group,
                    self._config.consumer,
                    min_idle_time=self._config.retry_idle_ms,
                    start_id=cursor,
                    count=self._config.retry_batch_size,
                )
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    next_cursor = result[0]
                    messages = result[1]
                else:  # pragma: no cover - unexpected shapes
                    break
                if not messages or next_cursor == cursor:
                    break
                cursor = next_cursor
                if cursor == "0-0":
                    break
        except (aioredis.ResponseError, AttributeError):  # pragma: no cover
            return

    async def _handle_handler_error(
        self,
        stream_name: str,
        message_id: str,
        envelope: leyline_pb2.BusEnvelope,
    ) -> None:
        attempt = int(envelope.attributes.get("attempt", "0")) + 1
        envelope.attributes["attempt"] = str(attempt)
        await self._redis.xack(stream_name, self._config.group, message_id)
        if attempt >= self._config.retry_max_attempts:
            if self._config.dead_letter_stream:
                await self._requeue(envelope, self._config.dead_letter_stream)
                self._metrics["dead_letter_total"] += 1.0
            return
        await self._requeue(envelope, stream_name)
        self._metrics["retry_total"] += 1.0
        self._conservative_mode = True

    async def _requeue(self, envelope: leyline_pb2.BusEnvelope, stream: str) -> None:
        cloned = leyline_pb2.BusEnvelope()
        cloned.CopyFrom(envelope)
        envelope_bytes = cloned.SerializeToString()
        fields = {"payload": envelope_bytes}
        signature = self._generate_signature(envelope_bytes)
        if signature:
            fields["signature"] = signature.encode("utf-8")
        kwargs = {}
        if self._config.max_stream_length and stream == self._config.normal_stream:
            kwargs["maxlen"] = self._config.max_stream_length
            kwargs["approximate"] = True
        await self._redis.xadd(stream, fields, **kwargs)

    async def emit_metrics_telemetry(
        self,
        *,
        source_subsystem: str = "oona",
        level: leyline_pb2.TelemetryLevel.ValueType = leyline_pb2.TelemetryLevel.TELEMETRY_LEVEL_INFO,
    ) -> bool:
        packet = await self._build_metrics_packet(
            source_subsystem=source_subsystem,
            level=level,
        )
        return await self.publish_telemetry(packet)

    async def _build_metrics_packet(
        self,
        *,
        source_subsystem: str,
        level: leyline_pb2.TelemetryLevel.ValueType,
    ) -> leyline_pb2.TelemetryPacket:
        metrics = await self.metrics_snapshot()
        packet = leyline_pb2.TelemetryPacket(
            packet_id=f"oona-{uuid.uuid4()}",
            source_subsystem=source_subsystem,
            level=level,
        )
        packet.timestamp.GetCurrentTime()
        for name, value in metrics.items():
            metric = packet.metrics.add()
            metric.name = f"oona.{name}"
            metric.value = float(value)
        return packet


__all__ = ["OonaClient", "StreamConfig", "OonaMessage"]


def _timestamp_to_ms(stamp) -> float | None:
    if stamp is None:
        return None
    seconds = getattr(stamp, "seconds", 0)
    nanos = getattr(stamp, "nanos", 0)
    if not seconds and not nanos:
        return None
    return (seconds * 1000.0) + (nanos / 1_000_000.0)


def _extract_request_id(envelope: leyline_pb2.BusEnvelope) -> str | None:
    message_type = envelope.message_type
    if message_type == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_READY:
        payload = leyline_pb2.KernelArtifactReady()
        payload.ParseFromString(envelope.payload)
        return payload.request_id
    if message_type == leyline_pb2.BusMessageType.BUS_MESSAGE_TYPE_KERNEL_PREFETCH_ERROR:
        payload = leyline_pb2.KernelArtifactError()
        payload.ParseFromString(envelope.payload)
        return payload.request_id
    return None
