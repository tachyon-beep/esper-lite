"""Tests for Karn WebSocket telemetry output."""

from __future__ import annotations

import json

import pytest

from esper.karn import websocket_output
from esper.karn.websocket_output import WebSocketOutput


class FakeEvent:
    pass


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        if self.fail:
            raise RuntimeError("client disconnected")
        self.messages.append(message)


class EmptyAsyncWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self) -> "EmptyAsyncWebSocket":
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


class ExplodingAsyncWebSocket:
    async def send(self, message: str) -> None:
        raise RuntimeError(f"send failed: {message}")


class FakeLoop:
    def __init__(self) -> None:
        self.called = False

    def call_soon_threadsafe(self, callback) -> None:
        self.called = True
        callback()


class ClosedLoop:
    def call_soon_threadsafe(self, callback) -> None:
        raise RuntimeError("loop closed")


class FakeShutdownEvent:
    def __init__(self) -> None:
        self.set_called = False

    def set(self) -> None:
        self.set_called = True


def test_emit_queues_serialized_event(monkeypatch: pytest.MonkeyPatch) -> None:
    output = WebSocketOutput(start_server=False)
    event = FakeEvent()

    def serialize(received: FakeEvent) -> str:
        assert received is event
        return '{"kind":"epoch"}'

    monkeypatch.setattr(websocket_output, "_serialize_event", serialize)

    output.emit(event)

    assert output.queue_size == 1
    assert output._queue.get_nowait() == '{"kind":"epoch"}'


def test_emit_drops_when_queue_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    output = WebSocketOutput(start_server=False, max_queue_size=1)
    monkeypatch.setattr(websocket_output, "_serialize_event", lambda event: "event")

    output.emit(FakeEvent())
    output.emit(FakeEvent())

    assert output.queue_size == 1


def test_emit_swallows_serializer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    output = WebSocketOutput(start_server=False)

    def fail_serialize(event: FakeEvent) -> str:
        raise RuntimeError("bad telemetry event")

    monkeypatch.setattr(websocket_output, "_serialize_event", fail_serialize)

    output.emit(FakeEvent())

    assert output.queue_size == 0


def test_close_is_idempotent_before_start() -> None:
    output = WebSocketOutput(start_server=False)

    output.close()
    output.close()

    assert output.client_count == 0
    assert output.queue_size == 0


def test_close_signals_running_loop() -> None:
    output = WebSocketOutput(start_server=False)
    loop = FakeLoop()
    shutdown_event = FakeShutdownEvent()
    output._running = True
    output._loop = loop
    output._shutdown_event = shutdown_event

    output.close()

    assert loop.called is True
    assert shutdown_event.set_called is True
    assert output._loop is None
    assert output._shutdown_event is None


def test_close_tolerates_closed_loop() -> None:
    output = WebSocketOutput(start_server=False)
    output._running = True
    output._loop = ClosedLoop()
    output._shutdown_event = FakeShutdownEvent()

    output.close()

    assert output._loop is None
    assert output._shutdown_event is None


@pytest.mark.asyncio
async def test_handle_client_sends_welcome_and_discards_on_disconnect() -> None:
    output = WebSocketOutput(start_server=False)
    websocket = EmptyAsyncWebSocket()

    await output._handle_client(websocket)

    assert output.client_count == 0
    payload = json.loads(websocket.sent[0])
    assert payload["type"] == "connected"
    assert payload["message"] == "Connected to Karn telemetry stream"


@pytest.mark.asyncio
async def test_handle_client_discards_after_send_failure() -> None:
    output = WebSocketOutput(start_server=False)
    websocket = ExplodingAsyncWebSocket()

    await output._handle_client(websocket)

    assert output.client_count == 0


@pytest.mark.asyncio
async def test_broadcast_drains_queue_without_clients() -> None:
    output = WebSocketOutput(start_server=False)
    output._queue.put_nowait("one")
    output._queue.put_nowait("two")

    await output._broadcast_queued_events()

    assert output.queue_size == 0


@pytest.mark.asyncio
async def test_broadcast_sends_batch_and_removes_disconnected_clients() -> None:
    output = WebSocketOutput(start_server=False)
    healthy_client = FakeClient()
    disconnected_client = FakeClient(fail=True)
    output._clients.add(healthy_client)
    output._clients.add(disconnected_client)
    output._queue.put_nowait("one")
    output._queue.put_nowait("two")

    await output._broadcast_queued_events()

    assert healthy_client.messages == ["one", "two"]
    assert disconnected_client not in output._clients
    assert output.queue_size == 0
