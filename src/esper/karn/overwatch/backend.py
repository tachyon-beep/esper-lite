"""OverwatchBackend - WebSocket server for Overwatch dashboard.

This module provides the Python backend that:
1. Receives telemetry events from the Nissa hub via emit()
2. Aggregates them using SanctumAggregator (same as Sanctum TUI)
3. Broadcasts snapshots to web clients at configurable Hz via WebSocket
4. Serves the Vue frontend from web/dist/
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import SanctumSnapshot

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent

_logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for special types.

    Handles:
    - Enums: serialize as .name (string)
    - datetime: serialize as ISO string
    - Path: serialize as string
    - deque: serialize as list
    - Dataclasses: serialize via asdict()
    """
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, deque):
        return list(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class OverwatchBackend:
    """WebSocket backend broadcasting SanctumSnapshot to web clients.

    Wraps SanctumAggregator and broadcasts snapshots at configurable Hz
    over WebSocket. Serves the Vue frontend from web/dist/.

    Thread-safe: emit() can be called from training threads while the
    WebSocket server runs in a background thread.

    Usage:
        backend = OverwatchBackend(port=8080)
        hub.add_backend(backend)
        backend.start()

        # ... training emits events ...

        backend.stop()
    """

    def __init__(
        self,
        port: int = 8080,
        snapshot_rate_hz: float = 10.0,
        num_envs: int = 16,
    ) -> None:
        """Initialize the backend.

        Args:
            port: WebSocket server port (default 8080)
            snapshot_rate_hz: Maximum snapshot broadcast rate (default 10 Hz)
            num_envs: Number of training environments (default 16)
        """
        self.port = port
        self.snapshot_rate_hz = snapshot_rate_hz
        self._min_broadcast_interval = 1.0 / snapshot_rate_hz

        # Aggregator for event processing (same as Sanctum TUI)
        self.aggregator = SanctumAggregator(num_envs=num_envs)

        # Rate limiting state
        self._last_broadcast_time: float = 0.0
        self._lock = threading.Lock()

        # Server lifecycle
        self._running = False
        self._server_thread: threading.Thread | None = None
        self._broadcast_queue: Queue[str] = Queue()

        # Connected WebSocket clients
        self._clients: list[Any] = []
        self._clients_lock = threading.Lock()

        # FastAPI app (created on start if available)
        self._app: Any = None
        self._server: Any = None

        # Static files path
        self._static_path = Path(__file__).parent / "web" / "dist"

    def emit(self, event: "TelemetryEvent") -> None:
        """Process a telemetry event.

        Thread-safe: can be called from training threads.

        Args:
            event: The telemetry event to process.
        """
        self.aggregator.process_event(event)
        self.maybe_broadcast()

    def get_snapshot(self) -> SanctumSnapshot:
        """Get the current SanctumSnapshot.

        Returns:
            Current aggregated snapshot state.
        """
        return self.aggregator.get_snapshot()

    def snapshot_to_json(self, snapshot: SanctumSnapshot) -> str:
        """Serialize a SanctumSnapshot to JSON.

        Args:
            snapshot: The snapshot to serialize.

        Returns:
            JSON string representation.
        """
        return json.dumps(asdict(snapshot), default=_json_serializer)

    def maybe_broadcast(self) -> None:
        """Trigger a rate-limited broadcast.

        Only broadcasts if enough time has passed since the last broadcast.
        Thread-safe.
        """
        now = time.time()
        with self._lock:
            if now - self._last_broadcast_time < self._min_broadcast_interval:
                return
            self._last_broadcast_time = now

        # Queue the broadcast (will be picked up by server thread)
        self._broadcast()

    def _broadcast(self) -> None:
        """Queue a snapshot broadcast to all connected clients."""
        if not self._running:
            return

        snapshot = self.get_snapshot()
        json_str = self.snapshot_to_json(snapshot)
        message = json.dumps({"type": "snapshot", "data": json.loads(json_str)})

        # Queue for async broadcast
        self._broadcast_queue.put(message)

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            return

        # Try to import FastAPI/uvicorn (optional dependency)
        try:
            import uvicorn
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import FileResponse
            from fastapi.staticfiles import StaticFiles

            # Mark as running only after successful import
            self._running = True

            # Create FastAPI app
            app = FastAPI(title="Overwatch Dashboard")

            @app.websocket("/ws")  # type: ignore[untyped-decorator]
            async def websocket_endpoint(websocket: WebSocket) -> None:
                await websocket.accept()

                # Send connected message
                await websocket.send_json({
                    "type": "connected",
                    "message": "Connected to Overwatch backend",
                })

                # Track client
                with self._clients_lock:
                    self._clients.append(websocket)

                try:
                    while self._running:
                        # Check for broadcast messages
                        try:
                            message = self._broadcast_queue.get_nowait()
                            with self._clients_lock:
                                for client in self._clients:
                                    try:
                                        await client.send_text(message)
                                    except Exception:
                                        pass
                        except Empty:
                            pass

                        # Wait for client messages (with timeout for broadcast check)
                        try:
                            await websocket.receive_text()
                        except WebSocketDisconnect:
                            break
                        except Exception:
                            break
                finally:
                    with self._clients_lock:
                        if websocket in self._clients:
                            self._clients.remove(websocket)

            # Serve index.html at root
            @app.get("/")  # type: ignore[untyped-decorator]
            async def serve_index() -> FileResponse:
                return FileResponse(self._static_path / "index.html")

            # Serve static assets
            if self._static_path.exists():
                app.mount(
                    "/assets",
                    StaticFiles(directory=self._static_path / "assets"),
                    name="assets",
                )

            self._app = app

            # Run server in background thread
            def run_server() -> None:
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=self.port,
                    log_level="warning",
                )
                self._server = uvicorn.Server(config)
                self._server.run()

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()

            _logger.info("Overwatch backend started on port %d", self.port)

        except ImportError:
            _logger.warning(
                "FastAPI/uvicorn not installed. Install with: pip install esper[dashboard]"
            )
            # _running stays False - emit() will skip broadcasts to avoid queue leak
            _logger.info(
                "Overwatch backend unavailable (no web server)"
            )

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        # Signal server to stop
        if self._server is not None:
            self._server.should_exit = True

        # Wait for thread to finish
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        # Clear clients
        with self._clients_lock:
            self._clients.clear()

        _logger.info("Overwatch backend stopped")

    def close(self) -> None:
        """Close the backend (called by Nissa hub on shutdown).

        Delegates to stop() and clears any remaining broadcast queue items.
        """
        self.stop()

        # Clear any remaining queued messages
        while not self._broadcast_queue.empty():
            try:
                self._broadcast_queue.get_nowait()
            except Empty:
                break
