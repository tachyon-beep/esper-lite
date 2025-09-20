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

    async def close(self) -> None:
        await self._redis.close()

    async def ensure_consumer_group(self) -> None:
        """Create consumer groups for both streams if they do not already exist."""

        for stream in (self._config.normal_stream, self._config.emergency_stream):
            try:
                await self._redis.xgroup_create(stream, self._config.group, id="$", mkstream=True)
            except aioredis.ResponseError as exc:  # group exists
                if "BUSYGROUP" not in str(exc):
                    raise

    async def publish_state(
        self,
        packet: leyline_pb2.SystemStatePacket,
        *,
        emergency: bool = False,
    ) -> None:
        await self._publish_proto(
            stream=self._config.emergency_stream if emergency else self._config.normal_stream,
            message_type="system_state",
            payload=packet.SerializeToString(),
        )

    async def publish_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        await self._publish_proto(
            stream=self._config.normal_stream,
            message_type="adaptation_command",
            payload=command.SerializeToString(),
        )

    async def publish_field_report(self, report: leyline_pb2.FieldReport) -> None:
        await self._publish_proto(
            stream=self._config.normal_stream,
            message_type="field_report",
            payload=report.SerializeToString(),
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

    async def _publish_proto(self, *, stream: str, message_type: str, payload: bytes) -> None:
        await self._redis.xadd(
            stream,
            {
                "type": message_type,
                "payload": payload,
            },
        )


__all__ = ["OonaClient", "StreamConfig", "OonaMessage"]
