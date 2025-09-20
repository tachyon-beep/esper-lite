"""Redis Streams client scaffold for Oona."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import redis.asyncio as aioredis

from esper.leyline import leyline_pb2


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


@dataclass(slots=True)
class OonaMessage:
    """Container passed to message handlers."""

    stream: str
    message_id: str
    message_type: str
    payload: bytes


class OonaClient:
    """High-level Oona client for interacting with Redis Streams."""

    def __init__(
        self,
        redis_url: str,
        config: StreamConfig,
        *,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        self._config = config
        self._redis = redis_client or aioredis.from_url(redis_url)
        self._telemetry_stream = config.telemetry_stream or config.normal_stream
        self._policy_stream = config.policy_stream or config.normal_stream

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
    ) -> None:
        await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=emergency,
            message_type="system_state",
            payload=packet.SerializeToString(),
        )

    async def publish_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type="adaptation_command",
            payload=command.SerializeToString(),
        )

    async def publish_field_report(self, report: leyline_pb2.FieldReport) -> None:
        await self._publish_proto(
            preferred_stream=self._config.normal_stream,
            emergency_flag=False,
            message_type="field_report",
            payload=report.SerializeToString(),
        )

    async def publish_telemetry(self, packet: leyline_pb2.TelemetryPacket) -> None:
        await self._publish_proto(
            preferred_stream=self._telemetry_stream,
            emergency_flag=False,
            message_type="telemetry",
            payload=packet.SerializeToString(),
        )

    async def publish_policy_update(self, update: leyline_pb2.PolicyUpdate) -> None:
        await self._publish_proto(
            preferred_stream=self._policy_stream,
            emergency_flag=False,
            message_type="policy_update",
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
                message = OonaMessage(
                    stream=stream_name,
                    message_id=message_id,
                    message_type=payload.get(b"type", b"").decode("utf-8"),
                    payload=bytes(payload.get(b"payload", b"")),
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
        message_type: str,
        payload: bytes,
    ) -> None:
        stream = await self._resolve_stream(preferred_stream, emergency_flag)
        fields = {"type": message_type, "payload": payload}
        if self._config.max_stream_length:
            await self._redis.xadd(
                stream,
                fields,
                maxlen=self._config.max_stream_length,
                approximate=True,
            )
        else:
            await self._redis.xadd(stream, fields)

    async def _resolve_stream(self, preferred: str, emergency_flag: bool) -> str:
        if emergency_flag:
            return self._config.emergency_stream
        threshold = self._config.emergency_threshold
        if threshold is None:
            return preferred
        backlog = await self._redis.xlen(preferred)
        if backlog >= threshold:
            return self._config.emergency_stream
        return preferred


__all__ = ["OonaClient", "StreamConfig", "OonaMessage"]
