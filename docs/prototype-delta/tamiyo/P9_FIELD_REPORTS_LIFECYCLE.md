# P9 — Field Report Lifecycle: Observation Windows + Ack/Retry (Tamiyo)

Objective
- Complete the Tamiyo prototype’s field‑report lifecycle by adding:
  - Observation windows (aggregate N subsequent epochs/steps for each decision) before emitting a synthesised FieldReport.
  - Bounded ack/retry semantics with a durable retry/index to avoid duplicates across restarts and to back off on transient failures.

Design Principles
- Contracts first: continue using `leyline_pb2.FieldReport` without schema changes.
- Idempotency & durability: unique `report_id`, WAL append, explicit retry index, safe on restarts.
- Non‑blocking: synthesis and publish steps must not stall Tamiyo decisions.

Scope (Prototype)
- Implement per‑decision observation windows; default N=3 epochs (configurable).
- Add a durable retry/index sidecar for field reports; bounded retries with backoff.
- Keep report content minimal (as per docs): outcome, observation_window_epochs, metrics delta.

Configuration (Env → EsperSettings)
- `TAMIYO_FR_OBS_WINDOW_EPOCHS` (int, default 3) — number of epochs/steps to aggregate per decision before synthesis. Use 1 to retain current per‑decision emit.
- `TAMIYO_FIELD_REPORT_MAX_RETRIES` (int, default existing value 3) — reuse current knob.
- `TAMIYO_FR_RETRY_BACKOFF_MS` (int, default 1000) — initial backoff for retries.
- `TAMIYO_FR_RETRY_BACKOFF_MULT` (float, default 2.0) — multiplier for exponential backoff.

Data Structures
- WAL (existing): `var/tamiyo/field_reports.log` — append‑only JSONL of serialised `FieldReport`s.
- Retry Index (new): `var/tamiyo/field_reports.index.json`
  - Map: `report_id` → { `retry_count`, `next_attempt_ms`, `last_error`?, `published` }
- Observation Window State (new): `var/tamiyo/field_reports.windows.json`
  - Map: `command_id` → {
    `seed_id`, `blueprint_id`, `policy_version`, `start_epoch`, `collected`, `target` (N),
    aggregate metrics (e.g., `sum_loss_delta`, `min_loss_delta`, `max_loss_delta`, `sum_hook_latency_ms`),
    last event reason/severity,
  }

Synthesis Logic
1) On each `evaluate_step` (or `evaluate_epoch`), update observation state:
   - If a new decision was issued this step, initialise state `{ collected=0, target=N }` keyed by `command_id`.
   - For all active windows, increment `collected` and update aggregates using current step’s metrics and/or events.
2) When `collected >= target` (or if an expiry window elapses), synthesise a `FieldReport`:
   - `report_id = f"fr-{command_id}-{start_epoch}+{target}"`
   - `observation_window_epochs = target`
   - `outcome` heuristic (prototype):
     - Success if `sum_loss_delta < 0` (net improvement) and no CRITICAL events
     - Regression if `sum_loss_delta > 0` or any CRITICAL (e.g., quarantine)
     - Neutral otherwise
   - `metrics`: include `loss_delta_total`, `loss_delta_min/max`, `avg_hook_latency_ms`, `reasons` (string code if any)
3) Append synthesised report to WAL and remove the window state for that `command_id`.

Ack/Retry Logic
- In `publish_history`:
  - Attempt to publish field reports not marked `published` in the retry index.
  - On success: mark `published=true`, remove from pending list, clear retry metadata.
  - On failure: increment `retry_count`, set `next_attempt_ms = now + backoff_0 * mult**retry_count`, store `last_error` (string); if `retry_count > max_retries`, emit a `field_report_drop` warning telemetry and leave the report in WAL (not in memory) for operator inspection.
- Use per‑call cap to avoid flooding; keep the existing batch behaviour.

API/Code Changes
- `src/esper/tamiyo/service.py`:
  - Add `_obs_config = ObservationConfig(N)` from settings.
  - Add `_windows: dict[str, WindowState]` with load/save to `windows.json`.
  - On each evaluation:
    - `_update_observation_windows(state, command, events)`
    - `_synthesise_due_windows()` before/after building telemetry
  - Extend `publish_history` to consult and update `field_reports.index.json` for retry metadata and backoff.
- `src/esper/tamiyo/persistence.py`:
  - Small helpers to load/save sidecar JSON indices atomically (write to temp, fsync, rename).

Telemetry
- Metrics:
  - `tamiyo.field_reports.pending_total`
  - `tamiyo.field_reports.published_total`
  - `tamiyo.field_reports.retries_total`
  - `tamiyo.field_reports.dropped_total`
- Events:
  - `field_report_synthesised` (INFO)
  - `field_report_retry` (WARNING; attributes: `report_id`, `retry_count`)
  - `field_report_drop` (WARNING; attributes: `report_id`, `retries`)

Testing Strategy
- Unit tests (Tamiyo):
  - `test_observation_window_synthesises_after_n_epochs`: produce N sequential steps after a command; assert a single synthesised report with expected metrics fields and `observation_window_epochs=N`.
  - `test_publish_history_ack_retry_success`: simulate failure then success; verify retry index updates and report marked published.
  - `test_retry_cap_drops_from_memory_preserves_wal`: set max retries to 1, ensure report remains in WAL but no longer in memory, and drop event is emitted.
  - `test_restart_restores_window_and_retry_index`: load from `windows.json` and index JSON after a simulated restart; continue synthesis and retries.
- Integration (optional):
  - End‑to‑end with FakeRedis: publish synthesised report, verify stream length increments and retry behaviour on temporary errors.

Acceptance Criteria
- Reports synthesised only after the configured observation window completes; metrics aggregate as specified.
- Bounded retries with backoff; durable index prevents duplicate publishes after restart.
- No Tamiyo stalls; evaluation latency budgets unaffected.

Risks & Mitigations
- Complexity in persistence: keep JSON sidecars small and atomic; avoid corrupting WAL on crashes.
- Outcome heuristic oversimplified: acceptable for prototype; document and evolve later.
- Potential memory growth in windows map: guard with max active windows and expiry TTL.

Timeline & Estimate
- Implementation: 1–1.5 days including unit tests.
- Optional integration tests: +0.5 day.

