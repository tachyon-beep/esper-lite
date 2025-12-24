"""FastAPI server for Karn dashboard.

Serves:
- Static HTML dashboard at /
- WebSocket endpoint at /ws for telemetry stream
- REST API at /api/* for health checks

Usage:
    # Standalone mode (for testing)
    python -m esper.karn.dashboard_server

    # With training (via --dashboard flag)
    python -m esper.scripts.train ppo --dashboard --dashboard-port 8000
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from queue import Queue, Empty
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

_logger = logging.getLogger(__name__)

# Dashboard HTML file path
_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


def _load_dashboard_html() -> str:
    """Load dashboard HTML from file."""
    if _DASHBOARD_PATH.exists():
        return _DASHBOARD_PATH.read_text()
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Karn Dashboard</title></head>
    <body style="background:#0a1020;color:#fff;font-family:monospace;padding:2em;">
        <h1>Karn Dashboard</h1>
        <p style="color:#f85;">Dashboard HTML not found at: {}</p>
        <p>Expected location: src/esper/karn/dashboard.html</p>
    </body>
    </html>
    """.format(_DASHBOARD_PATH)


# TODO: [DEAD CODE] - create_app() and run_dashboard_server() are exported but never called.
# Production uses integrated_dashboard.DashboardServer instead (train.py line 309).
# This appears to be superseded code. Either consolidate with integrated_dashboard.py
# or delete these functions. See: architectural risk assessment 2024-12-24.
def create_app(telemetry_queue: Queue[str] | None = None) -> FastAPI:
    """Create FastAPI app for Karn dashboard.

    Args:
        telemetry_queue: Thread-safe queue for telemetry events from training.
                        If None, server runs in standalone/demo mode.

    Returns:
        FastAPI application instance.
    """
    app = FastAPI(
        title="Karn Dashboard",
        description="Real-time morphogenetic training monitor",
        version="0.1.0",
    )

    # Connected WebSocket clients
    clients: set[WebSocket] = set()

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard() -> str:
        """Serve the dashboard HTML."""
        return _load_dashboard_html()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for telemetry stream."""
        await websocket.accept()
        clients.add(websocket)
        _logger.info(f"Dashboard client connected. Total: {len(clients)}")

        try:
            # Send welcome message
            await websocket.send_json(
                {
                    "type": "connected",
                    "message": "Connected to Karn telemetry stream",
                }
            )

            # Keep connection alive
            while True:
                try:
                    # Wait for client messages (ping/pong handled by FastAPI)
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0,
                    )
                    # Handle client commands if needed
                    _logger.debug(f"Received from client: {data}")
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            _logger.debug(f"WebSocket error: {e}")
        finally:
            clients.discard(websocket)
            _logger.info(f"Dashboard client disconnected. Total: {len(clients)}")

    @app.get("/api/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "clients": len(clients),
            "queue_size": telemetry_queue.qsize() if telemetry_queue else 0,
        }

    # Background task to broadcast telemetry from queue
    @app.on_event("startup")
    async def start_broadcast_task() -> None:
        if telemetry_queue:
            asyncio.create_task(_broadcast_loop(clients, telemetry_queue))

    return app


async def _broadcast_loop(clients: set[WebSocket], queue: Queue[str]) -> None:
    """Broadcast telemetry events from queue to all connected clients."""
    while True:
        try:
            # Non-blocking check with small sleep
            messages: list[str] = []
            while len(messages) < 50:
                try:
                    messages.append(queue.get_nowait())
                except Empty:
                    break

            if messages and clients:
                disconnected: set[WebSocket] = set()
                for client in clients:
                    try:
                        for msg in messages:
                            await client.send_text(msg)
                    except Exception:
                        disconnected.add(client)

                # Remove disconnected clients
                clients -= disconnected

            await asyncio.sleep(0.05)  # 50ms poll interval

        except Exception as e:
            _logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(1.0)


def run_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    telemetry_queue: Queue[str] | None = None,
) -> None:
    """Run dashboard server (blocking).

    Args:
        host: Server host (default: 0.0.0.0 for all interfaces)
        port: Server port (default: 8000)
        telemetry_queue: Queue for telemetry events from training
    """
    import uvicorn

    app = create_app(telemetry_queue)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Standalone mode for testing dashboard UI
    import argparse

    parser = argparse.ArgumentParser(description="Karn Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    print(f"Starting Karn dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    run_dashboard_server(host=args.host, port=args.port)
