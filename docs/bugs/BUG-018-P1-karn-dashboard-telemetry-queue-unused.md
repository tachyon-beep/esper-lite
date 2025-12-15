# BUG Template

- **Title:** Karn dashboard server ignores telemetry_queue (no producer/consumer wiring)
- **Context:** `dashboard_server.py::create_app` accepts `telemetry_queue` but never reads from it to broadcast to WebSocket clients; the `/ws` endpoint only sends pings/welcome messages.
- **Impact:** P1 – `--dashboard` flag wires Karn as an output backend, but dashboard never receives telemetry, making it effectively dead.
- **Environment:** Main branch; using dashboard server via CLI.
- **Reproduction Steps:** Run `python -m esper.scripts.train ... --dashboard`; connect to WebSocket; observe only pings/welcome, no telemetry frames.
- **Expected Behavior:** Server should consume telemetry_queue and broadcast events to connected clients.
- **Observed Behavior:** telemetry_queue is unused; no data sent.
- **Hypotheses:** Telemetry streaming was planned but not implemented; placeholder WebSocket loop has no producer.
- **Fix Plan:** Implement a queue consumer task pushing telemetry to clients (with backpressure handling); ensure queue is populated by Karn backend.
- **Validation Plan:** Start training with dashboard enabled; connect via WebSocket and verify telemetry frames arrive; add test feeding mock queue data and assert clients receive it.
- **Status:** Open
- **Links:** `src/esper/karn/dashboard_server.py` (`telemetry_queue` arg unused)
