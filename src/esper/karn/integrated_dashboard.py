"""Integrated dashboard that serves HTTP + WebSocket on same port.

This module provides a complete dashboard solution that:
1. Serves the dashboard HTML at http://localhost:PORT/
2. Accepts WebSocket connections at ws://localhost:PORT/ws
3. Bridges sync training events to async WebSocket broadcast

Usage:
    from esper.karn.integrated_dashboard import DashboardServer

    # Start dashboard (non-blocking)
    dashboard = DashboardServer(port=8000)
    dashboard.start()

    # Send events from training loop
    dashboard.emit(event)

    # Stop when done
    dashboard.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike

_logger = logging.getLogger(__name__)

# Dashboard HTML file path
_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


def _serialize_event(event: "TelemetryEventLike") -> str:
    """Serialize TelemetryEvent-like object to JSON string.

    Uses explicit field access instead of asdict() since TelemetryEventLike
    is a Protocol, not a dataclass. This allows any object implementing
    the protocol to be serialized correctly.
    """
    # Extract fields explicitly - Protocol doesn't guarantee dataclass
    # hasattr AUTHORIZED by John on 2025-12-17 10:00:00 UTC
    # Justification: Serialization - handle both enum and string event_type values
    event_type = event.event_type
    if hasattr(event_type, "name"):
        event_type = event_type.name

    # hasattr AUTHORIZED by John on 2025-12-17 10:00:00 UTC
    # Justification: Serialization - safely handle datetime objects
    timestamp = event.timestamp
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()

    data = {
        "event_type": event_type,
        "timestamp": timestamp,
        "data": event.data,
        "epoch": event.epoch,
        "seed_id": event.seed_id,
        "slot_id": event.slot_id,
        "severity": event.severity,
        "message": event.message,
    }

    return json.dumps(data, default=str)


class DashboardServer:
    """Integrated HTTP + WebSocket dashboard server.

    Serves dashboard HTML and broadcasts telemetry via WebSocket,
    all on a single port. Designed to be started once and forgotten.

    Args:
        port: Server port (default: 8000)
        host: Server host (default: 0.0.0.0)

    Example:
        dashboard = DashboardServer(port=8000)
        dashboard.start()

        # In training loop:
        hub.add_backend(dashboard)  # DashboardServer is an OutputBackend

        # Or emit directly:
        dashboard.emit(event)
    """

    def __init__(self, port: int = 8000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._queue: Queue[str] = Queue(maxsize=10000)
        self._thread: threading.Thread | None = None
        self._running = False
        self._ready = threading.Event()

    def emit(self, event: "TelemetryEventLike") -> None:
        """Queue event for WebSocket broadcast (OutputBackend interface)."""
        try:
            message = _serialize_event(event)
            self._queue.put_nowait(message)
        except Exception as e:
            # Log at debug level to help diagnose nonconforming events
            # without spamming logs during normal operation
            _logger.debug(
                f"Failed to serialize event {type(event).__name__}: {e}"
            )

    def close(self) -> None:
        """Stop the dashboard server (OutputBackend interface)."""
        self.stop()

    def start(self) -> None:
        """Start dashboard server in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="karn-dashboard",
        )
        self._thread.start()

        # Wait for server to be ready
        if not self._ready.wait(timeout=5.0):
            _logger.warning("Dashboard server startup timed out")

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_server(self) -> None:
        """Run the FastAPI server."""
        try:
            import uvicorn
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse
        except ImportError:
            _logger.error(
                "Dashboard dependencies not installed. "
                "Install with: pip install esper-lite[dashboard]"
            )
            self._ready.set()
            return

        app = FastAPI(title="Karn Dashboard")
        clients: set[WebSocket] = set()

        @app.get("/", response_class=HTMLResponse)
        async def get_dashboard() -> HTMLResponse:
            if _DASHBOARD_PATH.exists():
                content = _DASHBOARD_PATH.read_text()
            else:
                _logger.warning(f"Dashboard HTML not found at {_DASHBOARD_PATH}, serving fallback.")
                content = "<html><body><h1>Dashboard not found</h1><p>Please ensure 'dashboard.html' is present in the karn package directory.</p></body></html>"

            return HTMLResponse(
                content=content,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            clients.add(websocket)
            _logger.info(f"Dashboard client connected. Total: {len(clients)}")

            try:
                await websocket.send_json({
                    "type": "connected",
                    "message": "Connected to Karn telemetry stream",
                })

                # Keep alive
                while True:
                    try:
                        await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=60.0,
                        )
                    except asyncio.TimeoutError:
                        await websocket.send_json({"type": "ping"})

            except WebSocketDisconnect:
                pass
            except Exception as e:
                _logger.debug(f"WebSocket error: {e}")
            finally:
                clients.discard(websocket)

        @app.get("/api/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "clients": len(clients)}

        @app.on_event("startup")
        async def start_broadcast() -> None:
            asyncio.create_task(self._broadcast_loop(clients))
            self._ready.set()

        # Run uvicorn
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.run()

    async def _broadcast_loop(self, clients: set[Any]) -> None:
        """Broadcast queued events to WebSocket clients."""
        while self._running:
            try:
                # Batch events
                messages: list[str] = []
                while len(messages) < 50:
                    try:
                        messages.append(self._queue.get_nowait())
                    except Empty:
                        break

                if messages and clients:
                    disconnected: set[Any] = set()
                    for client in list(clients):
                        try:
                            for msg in messages:
                                await client.send_text(msg)
                        except Exception:
                            disconnected.add(client)
                    clients -= disconnected

                await asyncio.sleep(0.05)  # 50ms

            except Exception as e:
                _logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1.0)

    @property
    def url(self) -> str:
        """Dashboard URL."""
        return f"http://localhost:{self.port}"
