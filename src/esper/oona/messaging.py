"""Redis Streams client scaffold for Oona."""

from __future__ import annotations

import inspect
import os
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


@dataclass(slots=True)
class OonaMessage:
    """Container passed to message handlers."""

    stream: str
    message_id: str
    message_type: leyline_pb2.BusMessageType.ValueType
    payload: bytes
    attributes: dict[str, str] = field(default_factory=dict)


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
        self._metrics: dict[str, float] = {
            "publish_total": 0.0,
            "publish_rerouted": 0.0,
            "publish_dropped": 0.0,
            "queue_depth_max": 0.0,
        }
        self._signing_context = signing_context or self._load_signing_context()

    async def close(self) -> None:
        await self._redis.close()

    async def ensure_consumer_group(self) -> None:
        """Create consumer groups for both streams if they do not already exist."""

        streams = {
            self._config.normal_stream,
            self._config.emergency_stream,
            self._telemetry_stream,
            self._policy_stream,
        }
        for stream in streams:
            try:
                await self._redis.xgroup_create(stream, self._config.group, id="$", mkstream=True)
            except aioredis.ResponseError as exc:  # group exists
                message = str(exc)
                if "BUSYGROUP" in message:
                    continue
                if "NOGROUP" in message or "No such key" in message:
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

    async def publish_telemetry(self, packet: leyline_pb2.TelemetryPacket) -> bool:
        return await self._publish_proto(
            preferred_stream=self._telemetry_stream,
            emergency_flag=False,
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

    async def consume(
        self,
        handler: Callable[[OonaMessage], Awaitable[None] | None],
        *,
        stream: str | None = None,
        count: int = 1,
        block_ms: int = 1000,
    ) -> None:
        target_stream = stream or self._config.normal_stream
        response = await self._redis.xreadgroup(
            groupname=self._config.group,
            consumername=self._config.consumer,
            streams={target_stream: ">"},
            count=count,
            block=block_ms,
        )
        if not response:
            return

        for stream_name, messages in response:
            for message_id, payload in messages:
                envelope_bytes = bytes(payload.get(b"payload", b""))
                sig = payload.get(b"signature", b"").decode("utf-8")
                if not self._verify_payload(envelope_bytes, sig):
                    continue
                envelope = leyline_pb2.BusEnvelope()
                envelope.ParseFromString(envelope_bytes)
                message = OonaMessage(
                    stream=stream_name,
                    message_id=message_id,
                    message_type=envelope.message_type,
                    payload=envelope.payload,
                    attributes=dict(envelope.attributes),
                )
                result = handler(message)
                if inspect.isawaitable(result):
                    await result
                await self._redis.xack(stream_name, self._config.group, message_id)

    async def backlog(self, stream: str | None = None) -> int:
        """Return the number of pending entries for a stream."""

        target = stream or self._config.normal_stream
        try:
            summary = await self._redis.xpending(target, self._config.group)
        except aioredis.ResponseError:
            return 0
        return int(summary["pending"])

    async def stream_length(self, stream: str | None = None) -> int:
        """Return the underlying Redis stream length."""

        target = stream or self._config.normal_stream
        return int(await self._redis.xlen(target))

    async def metrics_snapshot(self) -> dict[str, float]:
        """Return a snapshot of publish/backpressure metrics and queue depth."""

        snapshot = dict(self._metrics)
        snapshot["queue_depth_normal"] = float(await self._redis.xlen(self._config.normal_stream))
        snapshot["queue_depth_emergency"] = float(await self._redis.xlen(self._config.emergency_stream))
        return snapshot

    @property
    def telemetry_stream(self) -> str:
        """Return the configured telemetry stream name."""

        return self._telemetry_stream

    @property
    def policy_stream(self) -> str:
        """Return the configured policy update stream name."""

        return self._policy_stream

    @property
    def normal_stream(self) -> str:
        """Return the configured normal stream name."""

        return self._config.normal_stream

    async def _publish_proto(
        self,
        *,
        preferred_stream: str,
        emergency_flag: bool,
        message_type: leyline_pb2.BusMessageType.ValueType,
        payload: bytes,
    ) -> bool:
        stream, rerouted, dropped, backlog = await self._resolve_stream(preferred_stream, emergency_flag)
        self._metrics["queue_depth_max"] = max(self._metrics["queue_depth_max"], float(backlog))
        if dropped:
            self._metrics["publish_dropped"] += 1.0
            return False
        if rerouted:
            self._metrics["publish_rerouted"] += 1.0
        envelope = leyline_pb2.BusEnvelope(message_type=message_type, payload=payload)
        envelope_bytes = envelope.SerializeToString()
        fields = {"payload": envelope_bytes}
        signature = self._generate_signature(envelope_bytes)
        if signature:
            fields["signature"] = signature.encode("utf-8")
        if self._config.max_stream_length:
            await self._redis.xadd(
                stream,
                fields,
                maxlen=self._config.max_stream_length,
                approximate=True,
            )
        else:
            await self._redis.xadd(stream, fields)
        length_after = await self._redis.xlen(stream)
        self._metrics["queue_depth_max"] = max(self._metrics["queue_depth_max"], float(length_after))
        self._metrics["publish_total"] += 1.0
        return True

    async def _resolve_stream(
        self,
        preferred: str,
        emergency_flag: bool,
    ) -> tuple[str, bool, bool, int]:
        if emergency_flag:
            return self._config.emergency_stream, False, False, 0

        backlog = await self._redis.xlen(preferred)

        drop_threshold = self._config.backpressure_drop_threshold
        if (
            drop_threshold is not None
            and preferred == self._config.normal_stream
            and backlog >= drop_threshold
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


__all__ = ["OonaClient", "StreamConfig", "OonaMessage"]
