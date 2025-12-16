# BUG-018: Karn dashboard telemetry queue (not an active bug)

- **Title:** Karn dashboard telemetry queue handling (not an active bug)
- **Context:** Bug report claimed `dashboard_server.py` ignores telemetry_queue
- **Impact:** P3 – Code review observation / no production impact
- **Environment:** Main branch
- **Status:** Deferred (downgraded from P1)

## Analysis (2025-12-17)

**Not an active bug.** Investigation found the code is correct.

### 1. Training Uses `integrated_dashboard.py` (NOT `dashboard_server.py`)

From `train.py`:
```python
from esper.karn import DashboardServer
dashboard_backend = DashboardServer(port=args.dashboard_port)
dashboard_backend.start()
hub.add_backend(dashboard_backend)
```

`DashboardServer` is imported from `integrated_dashboard.py`, which:
- Has `emit()` method that queues events (line 90-96)
- Has `_broadcast_loop()` that consumes the queue (line 197-223)
- Properly started via `@app.on_event("startup")` (line 181-184)

### 2. dashboard_server.py Also Works Correctly

The standalone `dashboard_server.py` DOES consume the queue:
- `_broadcast_loop()` defined at lines 130-158
- Started via `@app.on_event("startup")` at lines 121-126
- Reads from queue with `queue.get_nowait()` and broadcasts to clients

### Bug Report Claims vs Reality

| Claim | Status | Evidence |
|-------|--------|----------|
| "telemetry_queue is unused" | FALSE | `_broadcast_loop` consumes it |
| "Server only sends pings/welcome" | FALSE | Broadcasts all queued events |
| "No producer/consumer wiring" | FALSE | Queue consumed in broadcast loop |

## Why The Bug Report Was Wrong

The bug report author likely:
1. Looked at `dashboard_server.py` instead of `integrated_dashboard.py`
2. Saw the WebSocket endpoint only handles keep-alive (correct - broadcast is in separate task)
3. Missed the `@app.on_event("startup")` that starts `_broadcast_loop`

## Links

- `src/esper/karn/integrated_dashboard.py` (actual production implementation)
- `src/esper/karn/dashboard_server.py` (standalone mode, also works)
- `src/esper/scripts/train.py:248` (DashboardServer integration)
