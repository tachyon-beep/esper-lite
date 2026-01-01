---
id: "karn"
title: "Karn - Research Telemetry & Memory"
aliases: ["memory", "archivist", "telemetry", "dashboard", "tui"]
biological_role: "Memory"
layer: "Observation"
criticality: "Tier-2"
tech_stack:
  - "Python 3.11+"
  - "dataclasses"
  - "FastAPI + uvicorn (optional dashboard)"
  - "websockets (optional WS server)"
  - "rich (optional TUI)"
primary_device: "cpu"
thread_safety: "unsafe"
owners: "core-team"
compliance_tags:
  - "Best-Effort Telemetry"
  - "Bounded-Memory"
  - "Non-Blocking by Default"
schema_version: "1.0"
last_updated: "2025-12-14"
last_reviewed_commit: "40226ca"
---

# Karn Bible

# 1. Prime Directive

**Role:** Karn is Esper’s memory system: it ingests `TelemetryEvent` streams, materializes them into a bounded, typed `TelemetryStore`, triggers higher-fidelity diagnostics on anomalies, and exposes live + offline observability surfaces (TUI, web dashboard, JSONL export).

**Anti-Scope:** Karn does NOT make training decisions, does NOT mutate model weights/slots, and does NOT guarantee lossless telemetry capture—its backends are explicitly best-effort and may drop events under load.

---

# 2. Interface Contract

## 2.1 Entry Points (Public API)

### `get_collector() -> KarnCollector`
> Returns the global collector instance (singleton).

- **Invariants:**
  - Caller treats the returned collector as process-global state.
- **Notes:** Used as a Nissa backend in `src/esper/scripts/train.py` (see `hub.add_backend(get_collector())`).

### `configure(config: KarnConfig) -> KarnCollector`
> Replaces the global collector with a new configured instance.

- **Invariants:**
  - Reconfiguration is expected to happen before training begins (avoid swapping mid-run).

### `emit(event: TelemetryEvent) -> None`
> Convenience wrapper for `get_collector().emit(event)`.

### `class KarnCollector`
> Central telemetry collector + router; implements the `OutputBackend` protocol used by Nissa.

- **Source:** `src/esper/karn/collector.py:81`

**Constructor:**
- `KarnCollector(config: KarnConfig | None = None)`

**Key Methods:**
- `emit(event: TelemetryEvent) -> None` (store update + backend fanout) — `src/esper/karn/collector.py:128`
- `add_backend(backend: OutputBackend) -> None` — `src/esper/karn/collector.py:104`
- `start_episode(...) -> EpisodeContext` — `src/esper/karn/collector.py:384`
- `end_episode() -> None` — `src/esper/karn/collector.py:429`

**Key Properties:**
- `store: TelemetryStore` (typed, bounded in-memory history)

### `class TelemetryStore`
> Bounded in-memory store for typed telemetry tiers (context, epoch snapshots, dense traces).

- **Source:** `src/esper/karn/store.py:323`

**Key Methods:**
- `start_episode(context: EpisodeContext) -> None`
- `start_epoch(epoch: int) -> EpochSnapshot`
- `commit_epoch() -> None`
- `export_jsonl(path: Path | str) -> int` — `src/esper/karn/store.py:378`
- `import_jsonl(path: Path | str) -> TelemetryStore` — `src/esper/karn/store.py:436`
- `import_from_nissa_dir(dir_path: Path | str) -> TelemetryStore` — `src/esper/karn/store.py:483`

### `class TUIOutput`
> Rich-based terminal dashboard backend (auto-starts on first event).

- **Source:** `src/esper/karn/tui.py:431`
- **Implements:** `emit(event)`, `close()`

### `class DashboardServer`
> Integrated HTTP + WebSocket dashboard backend (FastAPI + uvicorn in a background thread).

- **Source:** `src/esper/karn/integrated_dashboard.py:61`
- **Implements:** `emit(event)`, `close()`, plus `start()` / `stop()` and `url`.

### `class WebSocketOutput`
> Lightweight WS-only backend (uses `websockets` library + background asyncio loop).

- **Source:** `src/esper/karn/websocket_output.py:59`

### `CounterfactualEngine` / `CounterfactualHelper`
> Research attribution utilities for “removal cost” / (approx) Shapley estimates.

- **Engine:** `src/esper/karn/counterfactual.py:198`
- **Helper:** `src/esper/karn/counterfactual_helper.py:40`

## 2.2 Configuration Schema

```python
# src/esper/karn/collector.py:69
@dataclass
class KarnConfig:
    capture_dense_traces: bool = True
    dense_trigger: DenseTraceTrigger = DenseTraceTrigger()
    on_emission_error: str = "warn"  # "ignore" | "warn" | "halt"

# src/esper/karn/store.py:243
@dataclass
class DenseTraceTrigger:
    stage_transition: bool = True
    loss_spike_threshold: float = 2.0
    accuracy_drop_threshold: float = 5.0
    gradient_explosion: float = 100.0
    gate_failure: bool = True
    value_collapse: bool = True
    entropy_collapse: bool = True
    force_dense: bool = False

# src/esper/karn/tui.py:55
@dataclass
class ThresholdConfig:
    entropy_critical: float = 0.3
    entropy_warning: float = 0.5
    clip_critical: float = 0.3
    clip_warning: float = 0.25
    explained_var_critical: float = 0.5
    explained_var_warning: float = 0.7
    grad_norm_critical: float = 10.0
    grad_norm_warning: float = 5.0
    kl_warning: float = 0.05
    wait_warning: float = 0.7
```

## 2.3 Events (Pub/Sub via Nissa)

### Subscribes (as a backend)

| Event | Handler | Action |
|------|---------|--------|
| `TRAINING_STARTED` | `KarnCollector._handle_training_started()` | Starts episode + opens epoch 1 (`src/esper/karn/collector.py:183`) |
| `EPOCH_COMPLETED` | `KarnCollector._handle_epoch_completed()` | Commits host snapshot + anomaly check + opens next epoch (`src/esper/karn/collector.py:198`) |
| `SEED_*` | `KarnCollector._handle_seed_event()` | Updates namespaced `SlotSnapshot` (`src/esper/karn/collector.py:249`) |
| `PPO_UPDATE_COMPLETED` | `KarnCollector._handle_ppo_update()` | Stores PPO diagnostics (`src/esper/karn/collector.py:297`) |
| `REWARD_COMPUTED` | `KarnCollector._handle_reward_computed()` | Stores reward/action summary (`src/esper/karn/collector.py:313`) |
| anomaly strings (e.g. `RATIO_EXPLOSION_DETECTED`) | `KarnCollector._handle_anomaly_event()` | Starts dense trace capture window (`src/esper/karn/collector.py:341`) |

### Emits

Karn itself does not originate events; it is a sink/router.

---

# 3. Tensor Contracts

Karn is **tensor-agnostic**: it stores scalar metrics and structured dataclasses, not tensors.

| Contract | Requirement |
|---------|-------------|
| Telemetry payloads | Prefer JSON-safe primitives; avoid raw tensors/large objects in `TelemetryEvent.data` to keep dashboards/export sane (`src/esper/karn/websocket_output.py:34`) |
| GPU stats | Read-only CUDA memory queries (no sync required by design; still best-effort) (`src/esper/karn/tui.py:1617`) |

---

# 4. Operational Physics

## 4.1 State Machines

### 4.1.1 Collector Episode Lifecycle

```
[Idle]
  --(TRAINING_STARTED OR start_episode())--> [EpisodeActive + EpochOpen]
  --(EPOCH_COMPLETED)--> commit(EpochSnapshot) ; open(next_epoch)
  --(end_episode())--> [Idle]
```

Key semantics:
- `EPOCH_COMPLETED` is treated as a **commit barrier** for an “outer epoch” (Simic uses `episodes_completed`) (`src/esper/simic/vectorized.py:1694`).

### 4.1.2 DenseTrace Capture

```
[NoTrace] --(trigger_reason != None)--> [Capturing window]
[Capturing window] --(epoch >= window_end)--> [NoTrace] + append(DenseTrace)
```

Implementation: `AnomalyDetector.start_trace()` / `finalize_trace()` (`src/esper/karn/triggers.py:171`).

## 4.2 Data Governance

### Authoritative (Source of Truth)
- `TelemetryStore.context` (`EpisodeContext`) — run identity + reproducibility (`src/esper/karn/store.py:34`)
- `TelemetryStore.epoch_snapshots` (`deque[EpochSnapshot]`, bounded) — analysis tier (`src/esper/karn/store.py:337`)
- `TelemetryStore.dense_traces` (`deque[DenseTrace]`, bounded) — anomaly-driven diagnostics (`src/esper/karn/store.py:343`)

### Ephemeral (Cached/Temporary)
- Dashboard / WS queues (`Queue[str]`, bounded) — may drop events under pressure (`src/esper/karn/websocket_output.py:101`)
- TUI ring buffers (`deque`, bounded) — rendering convenience only (`src/esper/karn/tui.py:148`)

### Read-Only (Consumed)
- `TelemetryEvent` stream from Nissa (`src/esper/leyline/telemetry.py:96`)

## 4.3 Concurrency Model

- **Training thread:** produces telemetry via `hub.emit(event)`; KarnCollector runs synchronously inside hub fanout.
- **Dashboard threads:** both `DashboardServer` and `WebSocketOutput` bridge sync emitters to async sockets using thread-safe queues.
- **Thread Safety:** Best-effort; individual backends usually self-contain locking/queues, but `KarnCollector` itself is not internally synchronized (`src/esper/karn/collector.py:128`).

## 4.4 Memory Lifecycle

- Store retention is bounded: `epoch_snapshots.maxlen = 500`, `dense_traces.maxlen = 20` (`src/esper/karn/store.py:337`, `src/esper/karn/store.py:342`).
- Dashboard queues are bounded (`maxsize=10000`) and intentionally drop when full (`src/esper/karn/integrated_dashboard.py:84`).

---

# 5. Dependencies

## 5.1 Upstream (Modules that call this module)

| Module | Interaction | Failure Impact |
|--------|-------------|----------------|
| `nissa` | Adds `KarnCollector`/`TUIOutput` as backends (`src/esper/scripts/train.py:270`) | **Tier-2**: observability loss only |
| `simic` | Emits PPO + epoch events (`src/esper/simic/vectorized.py:1633`) | **Tier-2**: missing analytics/history |
| `kasmina` | Emits lifecycle events (`SEED_*`) | **Tier-2**: seed timeline becomes sparse |

## 5.2 Downstream (Modules this module depends on)

| Module | Interaction | Failure Handling |
|--------|-------------|------------------|
| `leyline` | Uses `TelemetryEvent` + `SeedStage` contracts | **Fatal** for typed semantics |
| `rich` | TUI rendering (`TUIOutput`) | **Graceful** if TUI not used |
| `fastapi`/`uvicorn` | Integrated dashboard (`DashboardServer`) | **Graceful** (prints install hint) |
| `websockets` | WS-only backend (`WebSocketOutput`) | **Graceful** (prints install hint) |
| `torch`/`psutil`/`pynvml` | Optional system stats | **Graceful** (silent fallback) |

---

# 6. Esper Integration

## 6.1 Commandment Compliance

| # | Commandment | Compliance | Notes |
|---|-------------|------------|-------|
| 1 | Sensors match capabilities | ✅ | Karn is the canonical telemetry sink; enables “no blind growth”. |
| 2 | Complexity pays rent | ⚠️ | Surfaces compute rent/reward components, but not yet a single canonical efficiency model. |
| 3 | GPU-first iteration | ✅ | Uses non-blocking queues; telemetry failures should not block training by default. |
| 4 | Progressive curriculum | N/A | Not a curriculum implementer. |
| 5 | Train Anything protocol | ✅ | Consumes contract events only; no host-specific assumptions. |
| 6 | Morphogenetic plane | N/A | Observes plane, does not manage it. |
| 7 | Governor prevents catastrophe | ⚠️ | Observes governor events; does not enforce safety itself. |
| 8 | Hierarchical scaling | N/A | Observability layer only. |
| 9 | Frozen Core economy | ⚪ | Provides retention/export primitives but no long-term artifact store yet. |

## 6.2 Biological Role

**Analogy:** Karn is the organism’s memory/archivist: a place where training history becomes recallable structure, not ephemeral printouts.

**Responsibilities in the organism:**
- Record training history as typed snapshots (context → epochs → dense traces).
- Provide research-facing attribution tools (counterfactual/Shapley estimates).
- Surface health/anomaly signals via UI backends.

## 6.3 CLI Integration

| Command | Flags | Effect on Module |
|---------|-------|------------------|
| `python -m esper.scripts.train ppo` | `--no-tui` | Disables `TUIOutput` backend (`src/esper/scripts/train.py:200`) |
| `python -m esper.scripts.train ppo` | `--dashboard --dashboard-port PORT` | Starts `DashboardServer` backend (`src/esper/scripts/train.py:234`) |
| `python -m esper.scripts.train ppo` | `--export-karn PATH` | Exports `TelemetryStore` to JSONL in `finally:` (`src/esper/scripts/train.py:343`) |

---

# 7. Cross-References

## 7.1 Related Bibles

| Bible | Relationship | Integration Point |
|-------|--------------|-------------------|
| [simic](simic.md) | **Produces** | Emits PPO/epoch events that Karn stores and visualizes |
| [kasmina](kasmina.md) | **Produces** | Emits seed lifecycle events (`SEED_*`) |
| [tamiyo](tamiyo.md) | **Produces** | Emits strategic decisions (telemetry sink for decision traces) |
| [index](index.md) | **Roadmap** | Karn’s role in the overall organ map |

## 7.2 Key Source Files

| File | Purpose |
|------|---------|
| `src/esper/karn/collector.py` | Central collector + backend fanout + store update logic |
| `src/esper/karn/store.py` | Typed telemetry contracts + bounded in-memory store + JSONL export/import |
| `src/esper/karn/triggers.py` | Dense-trace triggers (EMA-based anomaly detection) |
| `src/esper/karn/analytics.py` | Read-only research queries over `TelemetryStore` |
| `src/esper/karn/counterfactual.py` | Counterfactual matrix + Shapley estimate primitives |
| `src/esper/karn/counterfactual_helper.py` | Training-loop-friendly wrapper that emits attribution events |
| `src/esper/karn/health.py` | Health/vitals monitors (optional) |
| `src/esper/karn/tui.py` | Rich TUI backend (multi-env aware) |
| `src/esper/karn/integrated_dashboard.py` | FastAPI dashboard backend (HTTP + WS) |
| `src/esper/karn/websocket_output.py` | Standalone WS backend (websockets library) |
| `src/esper/karn/dashboard.html` | Dashboard UI frontend |

## 7.3 Test Coverage

| Test File | Coverage | Critical Tests |
|-----------|----------|----------------|
| `tests/karn/test_store_export.py` | JSONL export/import basics | `export_jsonl()` round-trip constraints |
| `tests/karn/test_collector_multienv.py` | Slot namespacing | env-key uniqueness + event routing |
| `tests/karn/test_tui_state.py` | Multi-env aggregation | action counts, reward history, env status |
| `tests/karn/test_tui_rendering.py` | Render smoke tests | panel creation + event log formatting |

---

# 8. Tribal Knowledge

## 8.1 Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| `TelemetryStore.import_jsonl()` does not rehydrate enums/datetimes (they remain strings) (`src/esper/karn/store.py:436`) | Offline analysis code may see wrong types | Treat imported store as “untyped replay”; add a stricter loader if needed |
| `DashboardServer.stop()` does not actually shut down uvicorn (`src/esper/karn/integrated_dashboard.py:119`) | Dashboard thread may keep running; ports stay bound | Treat as “start once per process”; stop by terminating the process |
| Dashboard seed rendering expects a per-seed list shape (`src/esper/karn/dashboard.html:872`) but Simic emits aggregate counts (`src/esper/simic/vectorized.py:1636`) | Seeds panel may be empty/broken | Use the TUI for seed state; align payload schema before relying on web UI |

## 8.2 Performance Cliffs

| Operation | Trigger | Symptom | Mitigation |
|-----------|---------|---------|------------|
| WS/dashboard backends under heavy telemetry | High event rate + slow clients | Queue fills → events dropped (`src/esper/karn/websocket_output.py:101`) | Reduce telemetry verbosity; batch/aggregate events; prefer JSONL export for full-fidelity history |
| Shapley computation | Slot count grows (factorial perms) | `compute_shapley_values()` enumerates permutations (`src/esper/karn/counterfactual.py:336`) | Keep `n_active_seeds` small; implement true random permutation sampling without materializing `permutations()` |

## 8.3 Common Pitfalls

| Pitfall | Why It Happens | Correct Approach |
|---------|----------------|------------------|
| Counterfactual contributions not appearing in `TelemetryStore` | Slots are namespaced as `env{env_id}:{slot_id}` (`src/esper/karn/collector.py:263`), but counterfactual events may emit raw `slot_id` (`src/esper/karn/counterfactual_helper.py:143`) | Include `env_id` in counterfactual events and namespace slot IDs consistently |
| Misinterpreting “removal cost” as causal contribution | Ablating a seed late is confounded by adaptation (`src/esper/karn/counterfactual.py:1`) | Use removal cost as a debug heuristic; use parallel control runs for causal claims |
| Trusting `VitalSignsMonitor` seed failure stats | Stage numeric mapping is stale (`src/esper/karn/health.py:379`) vs authoritative `SeedStage` (`src/esper/leyline/stages.py:44`) | Treat vitals seed failure rate as experimental until stage mapping is updated |
| False “reward hacking” alarms in TUI | Heuristic: `base_acc_delta < 0 && total_reward > 0` (`src/esper/karn/tui.py:799`) | Calibrate reward components per reward mode; treat as a “suspicious pattern” flag |

## 8.4 Debugging Tips

- If the web dashboard connects but shows no updates, check `/api/health` and `queue_size` (`src/esper/karn/dashboard_server.py:114`).
- If export fails with a serialization error, scrub `TelemetryEvent.data` to primitives and avoid tensors/large objects (`src/esper/karn/store.py:398`).

---

# 9. Changelog

| Date | Change | Commit | Impact |
|------|--------|--------|--------|
| 2025-12-14 | Initial Karn bible (no specialist subagent review in this Codex run) | `40226ca` | Establishes contracts, boundaries, and known gaps |
