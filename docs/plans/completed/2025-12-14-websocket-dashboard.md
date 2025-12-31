# WebSocket Dashboard Implementation Plan

> **For Claude:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Real-time browser dashboard for Karn telemetry via WebSocket.

**Architecture:** Add `WebSocketOutput` backend that bridges sync training loop to async WebSocket server. Serve static HTML dashboard that renders the Karn monitor layout.

**Tech Stack:** Python 3.11+, FastAPI, websockets, vanilla JS (no framework)

**Dependency on:** Karn implementation (can be developed in parallel with stubs)

---

## Design Decisions

### D1: Sync/Async Bridge Pattern

**Problem:** Training loop is synchronous, WebSocket requires async.

**Solution:** Background thread running asyncio event loop with thread-safe queue.

```
Training Thread                    WebSocket Thread
┌─────────────┐                   ┌─────────────────┐
│ train_step  │                   │ asyncio loop    │
│     │       │                   │     │           │
│     ▼       │   thread-safe    │     ▼           │
│ emit(event) │──── queue ──────▶│ broadcast(msg)  │
│             │                   │     │           │
└─────────────┘                   │     ▼           │
                                  │ WebSocket send  │
                                  └─────────────────┘
```

**Why not:**
- Async training loop: Major refactor, breaks existing patterns
- Subprocess: Adds IPC complexity
- Polling: Defeats real-time purpose

### D2: No Frontend Framework

**Problem:** Could use React/Vue/Svelte, but adds build complexity.

**Solution:** Vanilla JS with template literals. Dashboard is simple enough.

**Why:**
- No npm/webpack/vite setup
- Single HTML file, easy to embed
- Sufficient for tabular data + charts

### D3: FastAPI over Raw websockets

**Problem:** Could use raw `websockets` library or full FastAPI.

**Solution:** FastAPI with WebSocket support.

**Why:**
- Single dependency handles HTTP (dashboard serve) + WebSocket
- Future: REST API for historical queries
- Better error handling and lifecycle management

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add FastAPI and uvicorn**

In `pyproject.toml`, add to dependencies:

```toml
"fastapi>=0.115.0",
"uvicorn[standard]>=0.32.0",
"websockets>=14.0",
```

**Step 2: Verify installation**

Run: `pip install -e ".[dev]"` or `uv sync`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add fastapi, uvicorn, websockets for dashboard"
```

---

## Task 2: Create WebSocket Output Backend

**Files:**
- Create: `src/esper/karn/websocket_output.py`
- Modify: `src/esper/karn/__init__.py`

**Step 1: Write the failing test**

```python
# tests/karn/test_websocket_output.py
"""Tests for WebSocket telemetry output."""

import pytest
from unittest.mock import MagicMock, patch
from queue import Queue

def test_websocket_output_queues_events():
    """WebSocketOutput queues events for async broadcast."""
    from esper.karn.websocket_output import WebSocketOutput
    from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType

    output = WebSocketOutput(start_server=False)  # Don't start server in test

    event = TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        data={"val_accuracy": 75.0},
    )

    output.emit(event)

    # Event should be in queue
    assert not output._queue.empty()
    queued = output._queue.get_nowait()
    assert "EPOCH_COMPLETED" in queued
    assert "75.0" in queued
```

**Step 2: Write implementation**

```python
# src/esper/karn/websocket_output.py
"""WebSocket output backend for real-time browser dashboard.

This backend bridges the synchronous training loop to async WebSocket
clients using a thread-safe queue and background asyncio event loop.

Architecture:
    Training Thread              WebSocket Thread
    ┌─────────────┐             ┌─────────────────┐
    │ emit(event) │── queue ───▶│ broadcast(msg)  │
    └─────────────┘             └─────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import asdict
from queue import Queue, Empty
from typing import Any

from esper.leyline.telemetry import TelemetryEvent

_logger = logging.getLogger(__name__)


def _serialize_event(event: TelemetryEvent) -> str:
    """Serialize TelemetryEvent to JSON string."""
    data = asdict(event)
    # Convert enum to string
    data["event_type"] = event.event_type.name
    # Convert datetime to ISO string
    if data.get("timestamp"):
        data["timestamp"] = data["timestamp"].isoformat()
    return json.dumps(data)


class WebSocketOutput:
    """Output backend that broadcasts telemetry via WebSocket.

    This backend is designed to work with the synchronous training loop.
    Events are queued and broadcast asynchronously to connected clients.

    Args:
        host: WebSocket server host (default: localhost)
        port: WebSocket server port (default: 8765)
        start_server: If True, start WebSocket server immediately (default: True)

    Usage:
        output = WebSocketOutput(port=8765)
        output.start()  # Start server in background thread

        # In training loop:
        output.emit(event)

        # When done:
        output.close()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        start_server: bool = True,
    ):
        self.host = host
        self.port = port
        self._queue: Queue[str] = Queue()
        self._clients: set[Any] = set()  # WebSocket connections
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        if start_server:
            self.start()

    def emit(self, event: TelemetryEvent) -> None:
        """Queue event for async broadcast to WebSocket clients.

        This method is called from the synchronous training loop.
        Events are serialized and queued for the async broadcast loop.
        """
        try:
            message = _serialize_event(event)
            self._queue.put_nowait(message)
        except Exception as e:
            _logger.warning(f"Failed to queue telemetry event: {e}")

    def start(self) -> None:
        """Start WebSocket server in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        _logger.info(f"WebSocket server starting on ws://{self.host}:{self.port}")

    def close(self) -> None:
        """Stop WebSocket server and cleanup."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
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
            import websockets
            from websockets.server import serve
        except ImportError:
            _logger.error("websockets not installed. Run: pip install websockets")
            return

        async with serve(self._handle_client, self.host, self.port):
            _logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")

            # Broadcast loop
            while self._running:
                await self._broadcast_queued_events()
                await asyncio.sleep(0.05)  # 50ms poll interval

    async def _handle_client(self, websocket: Any) -> None:
        """Handle new WebSocket client connection."""
        self._clients.add(websocket)
        _logger.info(f"Client connected. Total clients: {len(self._clients)}")

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Connected to Karn telemetry stream",
            }))

            # Keep connection alive until client disconnects
            async for message in websocket:
                # Handle client messages if needed (e.g., subscription filters)
                pass
        except Exception as e:
            _logger.debug(f"Client disconnected: {e}")
        finally:
            self._clients.discard(websocket)
            _logger.info(f"Client disconnected. Total clients: {len(self._clients)}")

    async def _broadcast_queued_events(self) -> None:
        """Broadcast all queued events to connected clients."""
        if not self._clients:
            # Drain queue if no clients (prevent memory buildup)
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            return

        # Batch events for efficiency
        messages = []
        while not self._queue.empty() and len(messages) < 100:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break

        if not messages:
            return

        # Broadcast to all clients
        disconnected = set()
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
```

**Step 3: Run test**

Run: `PYTHONPATH=src pytest tests/karn/test_websocket_output.py -v`

**Step 4: Commit**

```bash
git add src/esper/karn/websocket_output.py tests/karn/test_websocket_output.py
git commit -m "feat(karn): add WebSocketOutput backend for real-time dashboard"
```

---

## Task 3: Create Dashboard Server

**Files:**
- Create: `src/esper/karn/dashboard_server.py`

**Step 1: Write implementation**

**Note:** The dashboard HTML is in `src/esper/karn/dashboard.html` (not embedded inline). The server reads this file at startup.

```python
# src/esper/karn/dashboard_server.py
"""FastAPI server for Karn dashboard.

Serves:
- Static HTML dashboard at /
- WebSocket endpoint at /ws for telemetry stream
- REST API at /api/* for historical queries (future)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_logger = logging.getLogger(__name__)

# Load dashboard HTML from file
_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


def _load_dashboard_html() -> str:
    """Load dashboard HTML from file, with fallback for missing file."""
    if _DASHBOARD_PATH.exists():
        return _DASHBOARD_PATH.read_text()
    return "<html><body><h1>Dashboard not found</h1><p>Missing: dashboard.html</p></body></html>"


# NOTE: The full dashboard HTML is in src/esper/karn/dashboard.html
# This inline version is a minimal fallback only.
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Karn · Morphogenetic Training Monitor</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'SF Mono', 'Consolas', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        .header {
            border: 1px solid var(--border);
            border-radius: 6px 6px 0 0;
            padding: 12px 16px;
            background: var(--bg-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 14px;
            font-weight: 600;
        }

        .header .epoch {
            color: var(--text-secondary);
        }

        .mode-badge {
            background: var(--accent-blue);
            color: var(--bg-primary);
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
        }

        .panel {
            border: 1px solid var(--border);
            border-top: none;
            padding: 16px;
            background: var(--bg-secondary);
        }

        .panel:last-child {
            border-radius: 0 0 6px 6px;
        }

        .panel-title {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: var(--text-secondary);
        }

        .metric-value {
            font-weight: 600;
        }

        .metric-value.win {
            color: var(--accent-green);
        }

        .metric-value.loss {
            color: var(--accent-red);
        }

        .seed-row {
            display: grid;
            grid-template-columns: 80px 1fr 100px;
            gap: 12px;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
            align-items: center;
        }

        .seed-row:last-child {
            border-bottom: none;
        }

        .seed-name {
            color: var(--accent-blue);
            font-weight: 600;
        }

        .seed-details {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .seed-efficiency {
            text-align: right;
        }

        .seed-efficiency.star {
            color: var(--accent-yellow);
        }

        .policy-actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .action-chip {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
        }

        .action-chip.germinate {
            border-color: var(--accent-green);
            color: var(--accent-green);
        }

        .action-chip.fossilize {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }

        .alerts {
            color: var(--text-secondary);
        }

        .alerts.has-alert {
            color: var(--accent-red);
        }

        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
        }

        .connection-status.connected {
            border-color: var(--accent-green);
            color: var(--accent-green);
        }

        .connection-status.disconnected {
            border-color: var(--accent-red);
            color: var(--accent-red);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>KARN · MORPHOGENETIC TRAINING MONITOR</h1>
            <span class="epoch" id="epoch">Epoch --/--</span>
        </div>

        <div class="header" style="border-top: none; border-radius: 0; padding: 8px 16px;">
            <span>Mode: <span class="mode-badge" id="mode">--</span></span>
        </div>

        <div class="panel">
            <div class="panel-title">Performance</div>
            <div class="metric-row">
                <span class="metric-label">ACCURACY</span>
                <span class="metric-value" id="accuracy">--% (-- vs host-only)</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">PARAM RATIO</span>
                <span class="metric-value" id="param-ratio">--x host params</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">EFFICIENCY</span>
                <span class="metric-value" id="efficiency">--x acc/param</span>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Seed Contributions (Shapley)</div>
            <div id="seeds-container">
                <div class="seed-row">
                    <span class="seed-name">--</span>
                    <span class="seed-details">No active seeds</span>
                    <span class="seed-efficiency">--</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Policy Health</div>
            <div class="metric-row">
                <span class="metric-label">Value Variance</span>
                <span class="metric-value" id="value-variance">-- (--)</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Entropy</span>
                <span class="metric-value" id="entropy">-- (--)</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Recent Actions</span>
                <div class="policy-actions" id="recent-actions">
                    <span class="action-chip">--</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Alerts</div>
            <div class="alerts" id="alerts">None</div>
        </div>
    </div>

    <div class="connection-status disconnected" id="connection-status">
        ● Disconnected
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        // State
        let state = {
            epoch: 0,
            maxEpochs: 100,
            mode: 'SHAPED',
            accuracy: 0,
            hostAccuracy: 0,
            paramRatio: 1.0,
            efficiency: 1.0,
            seeds: [],
            valueVariance: 0,
            entropy: 0,
            recentActions: [],
            alerts: [],
        };

        function connect() {
            const wsUrl = `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('Connected to Karn telemetry');
                reconnectAttempts = 0;
                updateConnectionStatus(true);
            };

            ws.onclose = () => {
                console.log('Disconnected from Karn telemetry');
                updateConnectionStatus(false);

                // Reconnect with backoff
                if (reconnectAttempts < maxReconnectAttempts) {
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                    reconnectAttempts++;
                    setTimeout(connect, delay);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleEvent(data);
                } catch (e) {
                    console.error('Failed to parse event:', e);
                }
            };
        }

        function handleEvent(event) {
            // Handle different event types
            switch (event.event_type) {
                case 'EPOCH_COMPLETED':
                    state.epoch = event.epoch || state.epoch;
                    state.accuracy = event.data?.val_accuracy || state.accuracy;
                    break;

                case 'SEED_GERMINATED':
                    state.recentActions.push('GERMINATE');
                    if (state.recentActions.length > 5) state.recentActions.shift();
                    break;

                case 'SEED_FOSSILIZED':
                    state.recentActions.push('FOSSILIZE');
                    if (state.recentActions.length > 5) state.recentActions.shift();
                    break;

                case 'SEED_CULLED':
                    state.recentActions.push('CULL');
                    if (state.recentActions.length > 5) state.recentActions.shift();
                    break;

                case 'PPO_UPDATE_COMPLETED':
                    state.valueVariance = event.data?.value_variance || state.valueVariance;
                    state.entropy = event.data?.entropy || state.entropy;
                    break;

                case 'connected':
                    console.log('Server welcome:', event.message);
                    break;

                default:
                    // Log unknown events for debugging
                    console.debug('Unknown event:', event.event_type);
            }

            render();
        }

        function render() {
            // Epoch
            document.getElementById('epoch').textContent =
                `Epoch ${state.epoch}/${state.maxEpochs}`;

            // Mode
            document.getElementById('mode').textContent = state.mode;

            // Accuracy
            const accDelta = state.accuracy - state.hostAccuracy;
            const accEl = document.getElementById('accuracy');
            accEl.textContent = `${state.accuracy.toFixed(1)}% (${accDelta >= 0 ? '+' : ''}${accDelta.toFixed(1)}% vs host-only)`;
            accEl.className = `metric-value ${accDelta > 0 ? 'win' : accDelta < 0 ? 'loss' : ''}`;

            // Param ratio
            document.getElementById('param-ratio').textContent =
                `${state.paramRatio.toFixed(2)}x host params`;

            // Efficiency
            const effEl = document.getElementById('efficiency');
            effEl.textContent = `${state.efficiency.toFixed(2)}x acc/param`;
            effEl.className = `metric-value ${state.efficiency > 1 ? 'win' : state.efficiency < 1 ? 'loss' : ''}`;

            // Policy health
            const vvEl = document.getElementById('value-variance');
            vvEl.textContent = `${state.valueVariance.toFixed(2)} (${state.valueVariance > 0.1 ? 'healthy' : 'low'})`;

            const entEl = document.getElementById('entropy');
            entEl.textContent = `${state.entropy.toFixed(2)} (${state.entropy > 0.5 ? 'exploring' : 'exploiting'})`;

            // Recent actions
            const actionsEl = document.getElementById('recent-actions');
            if (state.recentActions.length > 0) {
                actionsEl.innerHTML = state.recentActions.map(a => {
                    const cls = a.toLowerCase().includes('germinate') ? 'germinate' :
                               a.toLowerCase().includes('fossilize') ? 'fossilize' : '';
                    return `<span class="action-chip ${cls}">${a}</span>`;
                }).join('');
            } else {
                actionsEl.innerHTML = '<span class="action-chip">WAIT</span>';
            }

            // Alerts
            const alertsEl = document.getElementById('alerts');
            if (state.alerts.length > 0) {
                alertsEl.textContent = state.alerts.join(', ');
                alertsEl.className = 'alerts has-alert';
            } else {
                alertsEl.textContent = 'None';
                alertsEl.className = 'alerts';
            }
        }

        function updateConnectionStatus(connected) {
            const el = document.getElementById('connection-status');
            if (connected) {
                el.textContent = '● Connected';
                el.className = 'connection-status connected';
            } else {
                el.textContent = '● Disconnected';
                el.className = 'connection-status disconnected';
            }
        }

        // Initial render and connect
        render();
        connect();
    </script>
</body>
</html>
"""


def create_app(telemetry_queue: asyncio.Queue | None = None) -> FastAPI:
    """Create FastAPI app for Karn dashboard.

    Args:
        telemetry_queue: Optional async queue for telemetry events.
                        If None, server runs in standalone mode.

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
    async def get_dashboard():
        """Serve the dashboard HTML."""
        return _load_dashboard_html()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for telemetry stream."""
        await websocket.accept()
        clients.add(websocket)
        _logger.info(f"Client connected. Total: {len(clients)}")

        try:
            # Send welcome
            await websocket.send_json({
                "type": "connected",
                "message": "Connected to Karn telemetry stream",
            })

            # Keep alive and handle client messages
            while True:
                try:
                    # Wait for client message (ping/pong handled automatically)
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0,
                    )
                    # Handle client commands if needed
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            _logger.debug(f"WebSocket error: {e}")
        finally:
            clients.discard(websocket)
            _logger.info(f"Client disconnected. Total: {len(clients)}")

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "clients": len(clients),
        }

    # Background task to broadcast telemetry
    @app.on_event("startup")
    async def start_broadcast_task():
        if telemetry_queue:
            asyncio.create_task(broadcast_loop(clients, telemetry_queue))

    return app


async def broadcast_loop(clients: set[WebSocket], queue: asyncio.Queue) -> None:
    """Broadcast telemetry events to all connected clients."""
    while True:
        try:
            event = await queue.get()

            if not clients:
                continue

            disconnected = set()
            for client in clients:
                try:
                    await client.send_text(event)
                except Exception:
                    disconnected.add(client)

            clients -= disconnected

        except Exception as e:
            _logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(1.0)


def run_dashboard_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run dashboard server (blocking).

    Args:
        host: Server host (default: 0.0.0.0 for all interfaces)
        port: Server port (default: 8000)
    """
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Standalone mode for testing
    run_dashboard_server()
```

**Step 2: Test server**

Run: `PYTHONPATH=src python -m esper.karn.dashboard_server`

Open browser to `http://localhost:8000`

**Step 3: Commit**

```bash
git add src/esper/karn/dashboard_server.py
git commit -m "feat(karn): add FastAPI dashboard server with embedded HTML"
```

---

## Task 4: Integrate WebSocket with Karn Collector

**Files:**
- Modify: `src/esper/karn/collector.py` (when it exists)
- Create: `src/esper/karn/output.py`

**Step 1: Create unified output module**

```python
# src/esper/karn/output.py
"""Karn output backends for telemetry distribution.

Available backends:
- ConsoleOutput: Print to stdout (inherited from Nissa)
- StructuredLogOutput: JSON lines to file
- WebSocketOutput: Real-time browser dashboard
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.leyline.telemetry import TelemetryEvent


class OutputBackend(ABC):
    """Base class for Karn output backends."""

    @abstractmethod
    def emit(self, event: "TelemetryEvent") -> None:
        """Emit a telemetry event."""
        pass

    def close(self) -> None:
        """Close backend and release resources."""
        pass


# Re-export existing backends
from esper.nissa.output import ConsoleOutput, FileOutput

# New backends
from esper.karn.websocket_output import WebSocketOutput

__all__ = [
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "WebSocketOutput",
]
```

**Step 2: Commit**

```bash
git add src/esper/karn/output.py
git commit -m "feat(karn): add unified output module with WebSocket support"
```

---

## Task 5: Add CLI Command for Dashboard

**Files:**
- Create: `src/esper/scripts/dashboard.py`
- Modify: `src/esper/scripts/__init__.py`

**Step 1: Create dashboard CLI**

```python
# src/esper/scripts/dashboard.py
"""CLI for Karn dashboard server.

Usage:
    python -m esper.scripts.dashboard [--port 8000] [--host 0.0.0.0]
"""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Start Karn telemetry dashboard server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print(f"Starting Karn dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    from esper.karn.dashboard_server import run_dashboard_server
    run_dashboard_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/esper/scripts/dashboard.py
git commit -m "feat(scripts): add dashboard CLI command"
```

---

## Task 6: Add --dashboard Flag to Training Script

**Files:**
- Modify: `src/esper/scripts/train.py`

**Step 1: Add dashboard argument**

In `ppo_parser`, add:

```python
    ppo_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable real-time WebSocket dashboard",
    )
    ppo_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Dashboard server port (default: 8000)",
    )
```

**Step 2: Wire up WebSocket output**

In the training setup, add:

```python
    # Start dashboard if requested
    websocket_output = None
    if args.dashboard:
        from esper.karn.websocket_output import WebSocketOutput
        websocket_output = WebSocketOutput(port=args.dashboard_port)
        hub.add_backend(websocket_output)
        print(f"Dashboard available at http://localhost:{args.dashboard_port}")
```

**Step 3: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(train): add --dashboard flag for real-time monitoring"
```

---

## Task 7: Integration Test

**Files:**
- Create: `tests/integration/test_dashboard.py`

**Step 1: Write integration test**

```python
# tests/integration/test_dashboard.py
"""Integration tests for dashboard WebSocket communication."""

import asyncio
import pytest
import json


@pytest.mark.asyncio
async def test_dashboard_websocket_connection():
    """Dashboard accepts WebSocket connections."""
    pytest.importorskip("websockets")
    pytest.importorskip("fastapi")

    import websockets
    from esper.karn.dashboard_server import create_app
    import uvicorn

    # Start server
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="error")
    server = uvicorn.Server(config)

    # Run server in background
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)  # Wait for startup

    try:
        # Connect WebSocket
        async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
            # Should receive welcome message
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)

            assert data["type"] == "connected"
            assert "Karn" in data["message"]
    finally:
        server.should_exit = True
        await server_task


@pytest.mark.asyncio
async def test_websocket_output_broadcasts():
    """WebSocketOutput broadcasts events to connected clients."""
    pytest.importorskip("websockets")

    from esper.karn.websocket_output import WebSocketOutput
    from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType
    import websockets

    # Start output backend
    output = WebSocketOutput(port=8766)
    await asyncio.sleep(0.5)  # Wait for startup

    try:
        # Connect client
        async with websockets.connect("ws://127.0.0.1:8766") as ws:
            # Skip welcome message
            await ws.recv()

            # Emit event
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=10,
                data={"val_accuracy": 75.5},
            )
            output.emit(event)

            # Should receive broadcast
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)

            assert data["event_type"] == "EPOCH_COMPLETED"
            assert data["epoch"] == 10
            assert data["data"]["val_accuracy"] == 75.5
    finally:
        output.close()
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/integration/test_dashboard.py -v`

**Step 3: Commit**

```bash
git add tests/integration/test_dashboard.py
git commit -m "test(dashboard): add WebSocket integration tests"
```

---

## Summary

| Task | Description | Effort |
|------|-------------|--------|
| 1 | Add dependencies | 10 min |
| 2 | WebSocketOutput backend | 1 hr |
| 3 | Dashboard server + HTML | 2 hr |
| 4 | Output module integration | 30 min |
| 5 | Dashboard CLI | 20 min |
| 6 | --dashboard training flag | 30 min |
| 7 | Integration tests | 30 min |
| **Total** | | **~5 hours** |

---

## Usage After Implementation

```bash
# Start dashboard standalone (for debugging)
python -m esper.scripts.dashboard --port 8000

# Train with live dashboard
python -m esper.scripts.train ppo --episodes 1000 --dashboard --dashboard-port 8000

# Open browser to http://localhost:8000
```

---

## Future Enhancements (Not in Scope)

- **Historical playback**: Replay training runs from saved telemetry
- **Multi-run comparison**: Side-by-side dashboards
- **Charts**: Loss/accuracy curves with Chart.js
- **Alerts**: Browser notifications for anomalies
- **Mobile responsive**: Touch-friendly layout
