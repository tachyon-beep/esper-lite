"""WebSocket output backend for real-time browser dashboard.

This backend bridges the synchronous training loop to async WebSocket
clients using a thread-safe queue and background asyncio event loop.

Architecture:
    Training Thread              WebSocket Thread
    +-------------+             +-----------------+
    | emit(event) |-- queue --->| broadcast(msg)  |
    +-------------+             +-----------------+

Usage:
    from esper.karn import WebSocketOutput

    output = WebSocketOutput(port=8765)
    hub.add_backend(output)

    # Dashboard available at ws://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from queue import Queue, Empty
from typing import Any, TYPE_CHECKING

from esper.karn.serialization import serialize_event as _serialize_event

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike

_logger = logging.getLogger(__name__)


class WebSocketOutput:
    """Output backend that broadcasts telemetry via WebSocket.

    This backend is designed to work with the synchronous training loop.
    Events are queued and broadcast asynchronously to connected clients.

    Args:
        host: WebSocket server host (default: localhost)
        port: WebSocket server port (default: 8765)
        start_server: If True, start WebSocket server immediately (default: True)
        max_queue_size: Maximum events to queue before dropping (default: 10000)

    Usage:
        output = WebSocketOutput(port=8765)

        # In training loop (sync):
        output.emit(event)

        # When done:
        output.close()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        start_server: bool = True,
        max_queue_size: int = 10000,
    ):
        self.host = host
        self.port = port
        self.max_queue_size = max_queue_size

        self._queue: Queue[str] = Queue(maxsize=max_queue_size)
        self._clients: set[Any] = set()  # WebSocket connections
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._server_ready = threading.Event()

        if start_server:
            self.start()

    def emit(self, event: "TelemetryEventLike") -> None:
        """Queue event for async broadcast to WebSocket clients.

        This method is called from the synchronous training loop.
        Events are serialized and queued for the async broadcast loop.

        If the queue is full, the event is dropped (training should not block).
        """
        try:
            message = _serialize_event(event)
            self._queue.put_nowait(message)
        except Exception as e:
            # Don't let telemetry errors break training
            _logger.debug(f"Failed to queue telemetry event: {e}")

    def start(self) -> None:
        """Start WebSocket server in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="karn-websocket",
        )
        self._thread.start()

        # Wait for server to be ready (with timeout)
        if not self._server_ready.wait(timeout=5.0):
            _logger.warning("WebSocket server startup timed out")

    def close(self) -> None:
        """Stop WebSocket server and cleanup."""
        self._running = False

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        _logger.info("WebSocket server stopped")

    def _run_event_loop(self) -> None:
        """Run asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            _logger.error(f"WebSocket server error: {e}")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Main async server loop."""
        try:
            from websockets.server import serve  # type: ignore[import-not-found]  # no stubs
        except ImportError:
            _logger.error(
                "websockets not installed. Install with: pip install esper-lite[dashboard]"
            )
            self._server_ready.set()
            return

        try:
            async with serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
            ):
                _logger.info(
                    f"Karn WebSocket server listening on ws://{self.host}:{self.port}"
                )
                self._server_ready.set()

                # Broadcast loop - poll queue and send to clients
                while self._running:
                    await self._broadcast_queued_events()
                    await asyncio.sleep(0.05)  # 50ms poll interval (20 Hz)

        except OSError as e:
            _logger.error(f"Failed to start WebSocket server: {e}")
            self._server_ready.set()

    async def _handle_client(self, websocket: Any) -> None:
        """Handle new WebSocket client connection."""
        self._clients.add(websocket)
        _logger.info(f"Dashboard client connected. Total: {len(self._clients)}")

        try:
            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "message": "Connected to Karn telemetry stream",
                    }
                )
            )

            # Keep connection alive until client disconnects
            async for message in websocket:
                # Handle client messages if needed (e.g., subscription filters)
                # For now, just acknowledge
                pass

        except Exception as e:
            _logger.debug(f"Client disconnected: {e}")
        finally:
            self._clients.discard(websocket)
            _logger.info(f"Dashboard client disconnected. Total: {len(self._clients)}")

    async def _broadcast_queued_events(self) -> None:
        """Broadcast all queued events to connected clients."""
        if not self._clients:
            # Drain queue if no clients (prevent memory buildup)
            drained = 0
            while not self._queue.empty() and drained < 100:
                try:
                    self._queue.get_nowait()
                    drained += 1
                except Empty:
                    break
            return

        # Batch events for efficiency (up to 100 per cycle)
        messages: list[str] = []
        while not self._queue.empty() and len(messages) < 100:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break

        if not messages:
            return

        # Broadcast to all clients
        disconnected: set[Any] = set()
        for client in self._clients:
            try:
                for msg in messages:
                    await client.send(msg)
            except Exception:
                disconnected.add(client)

        # Remove disconnected clients
        self._clients -= disconnected

    @property
    def client_count(self) -> int:
        """Number of connected WebSocket clients."""
        return len(self._clients)

    @property
    def queue_size(self) -> int:
        """Current number of events in queue."""
        return self._queue.qsize()
