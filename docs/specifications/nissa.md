---
id: "nissa"
title: "Nissa - Telemetry Hub"
aliases: ["telemetry", "senses", "nissa-hub"]
biological_role: "Sensory Organs"
layer: "Observation"
criticality: "Tier-2"
tech_stack:
  - "Python 3.11+"
  - "PyTorch 2.x"
  - "NumPy"
  - "Pydantic 2.x"
  - "PyYAML"
primary_device: "mixed"
thread_safety: "unsafe"
owners: "core-team"
compliance_tags:
  - "Best-Effort"
  - "Non-Fatal"
  - "I/O"
schema_version: "1.0"
last_updated: "2026-01-01"
last_reviewed_commit: "40226ca"
---

# Nissa Bible

# 1. Prime Directive

**Role:** Route Esper’s telemetry events (Leyline `TelemetryEvent`) to one-or-more output backends, providing a single “sensory bus” that surfaces lifecycle/training/health signals for humans and tools. (`src/esper/nissa/output.py:393`, `src/esper/leyline/telemetry.py:91`)

**Anti-Scope:** Nissa does NOT make training decisions, compute rewards, or enforce lifecycle semantics; it does NOT provide durable storage guarantees or async backpressure—telemetry is best-effort and must never be a training dependency. (`src/esper/nissa/output.py:428`, `src/esper/nissa/output.py:458`)

---

# 2. Interface Contract

## 2.1 Entry Points (Public API)

### `get_hub() -> NissaHub`
> Return the process-global telemetry hub singleton.

- **Invariants:**
  - Calling `get_hub()` repeatedly returns the same instance for the process lifetime unless tests monkeypatch `_global_hub`.
  - Hub creation is lazy; if no backends are added, events are effectively dropped.
- **Thread Safety:** Unsafe (no locks around `_global_hub` or backend list).
- **Reference:** `src/esper/nissa/output.py:454`, `src/esper/nissa/output.py:458`

### `emit(event: TelemetryEvent) -> None`
> Convenience wrapper that emits to the global hub.

- **Failure Semantics:** Backend exceptions are caught inside `NissaHub.emit()`; failures are logged to stderr but do not stop other backends. (`src/esper/nissa/output.py:428`)
- **Reference:** `src/esper/nissa/output.py:470`

### `class NissaHub`
> Synchronous in-process router for `TelemetryEvent` fan-out.

**Key Methods:**
- `add_backend(backend: OutputBackend) -> None` (`src/esper/nissa/output.py:411`)
- `remove_backend(backend: OutputBackend) -> None` (`src/esper/nissa/output.py:419`)
- `emit(event: TelemetryEvent) -> None` (`src/esper/nissa/output.py:428`)
- `close() -> None` (`src/esper/nissa/output.py:441`)

### `class OutputBackend(ABC)`
> Minimal backend protocol: “accept a `TelemetryEvent`”.

**Required:**
- `emit(event: TelemetryEvent) -> None` (`src/esper/nissa/output.py:33`)

**Optional:**
- `close() -> None` (`src/esper/nissa/output.py:42`)

### `class ConsoleOutput(OutputBackend)`
> Human-facing formatter for telemetry, with severity filtering and special-case formatting for key event types.

**Constructor:**
- `ConsoleOutput(verbose: bool = False, use_color: bool = True, min_severity: str = "info")` (`src/esper/nissa/output.py:59`)

**Behavioral Notes:**
- Severity filter is string-ordered via `_SEVERITY_ORDER`. (`src/esper/nissa/output.py:56`)
- Event formatting is *string driven* (`event.event_type.name` or `str(event.event_type)`) and must be updated when new event types are added for best UX. (`src/esper/nissa/output.py:75`)

### `class FileOutput(OutputBackend)`
> Append-only JSONL telemetry log with in-memory buffering.

**Constructor:**
- `FileOutput(path: str | Path, buffer_size: int = 10)` (`src/esper/nissa/output.py:287`)

**Flush/Close:**
- Buffered writes flush when buffer reaches `buffer_size` or on `close()`. (`src/esper/nissa/output.py:304`, `src/esper/nissa/output.py:315`)

### `class DirectoryOutput(OutputBackend)`
> Creates a timestamped folder and writes `events.jsonl` inside it (via an internal `FileOutput`).

**Constructor:**
- `DirectoryOutput(base_path: str | Path, buffer_size: int = 10)` (`src/esper/nissa/output.py:363`)

**Key Property:**
- `output_dir: Path` returns the created timestamped folder. (`src/esper/nissa/output.py:375`)

### `class TelemetryConfig(BaseModel)`
> Pydantic configuration for *diagnostic collection* (not for routing). Loaded from `profiles.yaml` with optional overrides.

**Key Methods:**
- `from_profile(name: str, overrides: dict[str, Any] | None = None) -> TelemetryConfig` (`src/esper/nissa/config.py:122`)
- `from_yaml(path: Path | str, overrides: dict[str, Any] | None = None) -> TelemetryConfig` (`src/esper/nissa/config.py:101`)
- `feature_count_estimate() -> int` (rough dimensionality estimate; advisory only). (`src/esper/nissa/config.py:179`)

**Profiles Source:**
- Built-ins live in `src/esper/nissa/profiles.yaml:12` (minimal/standard/diagnostic/research).

### `class DiagnosticTracker`
> Optional “rich telemetry” collector (gradient stats, loss noise, sharpness probes) intended for debugging and narratives.

**Constructor:**
- `DiagnosticTracker(model: nn.Module, config: TelemetryConfig, device: str = "cuda")` (`src/esper/nissa/tracker.py:138`)

**Key Methods:**
- `on_batch_loss(loss: float) -> None` (`src/esper/nissa/tracker.py:229`)
- `end_epoch(...) -> EpochSnapshot` (`src/esper/nissa/tracker.py:344`)
- `generate_narrative(snapshot: EpochSnapshot) -> str` (`src/esper/nissa/tracker.py:399`)
- `get_decision_brief(action: str, reason: str = "") -> dict` (`src/esper/nissa/tracker.py:503`)
- `cleanup() -> None` (must be called to reliably remove hooks). (`src/esper/nissa/tracker.py:526`)

### `class BlueprintAnalytics(OutputBackend)`
> Aggregates seed lifecycle events into per-blueprint and per-env rollups (“scoreboard”).

**Key Methods:**
- `emit(event: TelemetryEvent) -> None` (processes lifecycle events only). (`src/esper/nissa/analytics.py:145`)
- `summary_table() -> str` (`src/esper/nissa/analytics.py:224`)
- `scoreboard_table(env_id: int = 0) -> str` (`src/esper/nissa/analytics.py:244`)
- `snapshot() -> dict` (`src/esper/nissa/analytics.py:264`)

## 2.2 Configuration Schema

```python
# src/esper/nissa/config.py:78
class TelemetryConfig(BaseModel):
    profile_name: str = "custom"
    history_length: int  # [5, 100]

    gradients: GradientConfig
    loss_landscape: LossLandscapeConfig
    per_class: PerClassConfig

    track_weight_norms: bool = False
    track_activation_stats: bool = False
```

## 2.3 Events (Pub/Sub via Nissa)

**Event Contract:** Nissa routes Leyline `TelemetryEvent` objects. (`src/esper/leyline/telemetry.py:91`)

### Emits
*None by default.* Nissa is primarily a router/formatter; modules emit events, Nissa forwards them. Backend failures are printed to stderr rather than emitted as telemetry. (`src/esper/nissa/output.py:437`)

### Subscribes
| Event | Handler | Action |
|-------|---------|--------|
| `TelemetryEventType.*` | `NissaHub.emit()` | Fan-out to `OutputBackend.emit()` |

Notable consumers:
- `BlueprintAnalytics` consumes `SEED_GERMINATED`, `SEED_FOSSILIZED`, `SEED_PRUNED`. (`src/esper/nissa/analytics.py:147`)
- `ConsoleOutput` formats many event families by name (EPOCH/COMMAND/SEED/etc). (`src/esper/nissa/output.py:83`)

---

# 3. Tensor Contracts

Nissa’s hub/backends are **tensor-agnostic**; they operate on Python dataclasses and JSON-serializable payloads.

## 3.1 DiagnosticTracker Inputs

| Name | Shape | Dtype | Device | Description |
|------|-------|-------|--------|-------------|
| `grad` | `[*param.shape]` | float | cpu/cuda | Gradient tensor observed in a parameter hook. (`src/esper/nissa/tracker.py:173`) |
| `inputs` | task-defined | float/int | moved to `device` | Validation inputs from `val_loader` for sharpness/loss probes. (`src/esper/nissa/tracker.py:312`) |
| `targets` | task-defined | int | moved to `device` | Validation targets. (`src/esper/nissa/tracker.py:314`) |

## 3.2 DiagnosticTracker Outputs

| Output | Type | Description |
|--------|------|-------------|
| `GradientStats` | dataclass | Per-layer scalar stats (norm/std/mean + optional percentiles). (`src/esper/nissa/tracker.py:31`) |
| `GradientHealth` | dataclass | Aggregated health score and counts. (`src/esper/nissa/tracker.py:54`) |
| `EpochSnapshot` | dataclass | End-of-epoch bundle with narrative + flags. (`src/esper/nissa/tracker.py:73`) |

## 3.3 Gradient Flow

```
grad (hook input)
  └─ detach() (read-only)
      └─ scalar stats computed on-device
          └─ stack(...).tolist() -> CPU sync (one per tracked param)
```

Nissa does not modify gradients or parameters; it only *observes*. (`src/esper/nissa/tracker.py:182`, `src/esper/nissa/tracker.py:208`)

---

# 4. Operational Physics

## 4.1 State Machine

### Event Routing (Core Loop)

```
[Any module] --emit(TelemetryEvent)--> [NissaHub]
[NissaHub] --for backend in backends--> backend.emit(event)
[backend failure] --caught--> stderr log, continue other backends
```

References: `src/esper/nissa/output.py:428`, `src/esper/nissa/output.py:437`

### FileOutput Buffering

```
BUFFER < buffer_size  --emit--> append
BUFFER == buffer_size --emit--> flush() -> write JSONL -> clear
close()              --close--> flush() -> close file
```

References: `src/esper/nissa/output.py:298`, `src/esper/nissa/output.py:307`, `src/esper/nissa/output.py:315`

### DirectoryOutput Session Folder

```
init(base_path) -> mkdir base_path
              -> mkdir telemetry_YYYY-MM-DD_HHMMSS/
              -> create FileOutput(events.jsonl)
```

Reference: `src/esper/nissa/output.py:363`

### DiagnosticTracker Lifecycle

```
init(config.gradients.enabled) -> register param hooks
backward() -> hooks call _record_grad(...) per tracked param
end_epoch(...) -> assemble EpochSnapshot + narrative + flags
cleanup() -> remove all hooks
```

References: `src/esper/nissa/tracker.py:150`, `src/esper/nissa/tracker.py:164`, `src/esper/nissa/tracker.py:344`, `src/esper/nissa/tracker.py:526`

### BlueprintAnalytics Aggregation

```
SEED_GERMINATED -> germinated++ ; scoreboard.live_blueprint=bp
SEED_FOSSILIZED -> fossilized++ ; stats.acc_deltas+=Δacc ; scoreboard.fossilized_by_blueprint[bp]++
SEED_PRUNED     -> pruned++     ; stats.churns+=Δacc
```

References: `src/esper/nissa/analytics.py:145`, `src/esper/nissa/analytics.py:161`, `src/esper/nissa/analytics.py:190`

## 4.2 Data Governance

### Authoritative (Source of Truth)
- *None.* Nissa does not own training state; it only forwards observations.

### Ephemeral (Cached/Temporary)
- `NissaHub._backends`: in-memory list of attached sinks. (`src/esper/nissa/output.py:409`)
- `FileOutput._buffer`: buffered JSON records pending flush. (`src/esper/nissa/output.py:290`)
- `DiagnosticTracker._grad_stats`, `.history`, `._batch_losses`: per-run diagnostic caches. (`src/esper/nissa/tracker.py:144`, `src/esper/nissa/tracker.py:147`, `src/esper/nissa/tracker.py:155`)

### Durable (If Enabled)
- `FileOutput`/`DirectoryOutput` JSONL logs are the durable artifact used by post-hoc tools (e.g., Karn import). (`src/esper/nissa/output.py:279`, `src/esper/karn/store.py:478`)

## 4.3 Concurrency Model

- **Thread Safety:** Unsafe. No locking around backend mutation or emit loop. (`src/esper/nissa/output.py:409`)
- **Async Pattern:** Blocking/synchronous; emit cost is on caller’s thread.
- **Backpressure:** None. If a backend is slow (e.g., console spam, disk flush), training slows.

## 4.4 Memory Lifecycle

- **Allocation:** `DirectoryOutput` creates a new folder per run; `FileOutput` keeps a file handle open. (`src/esper/nissa/output.py:363`, `src/esper/nissa/output.py:296`)
- **Retention:** Global hub holds strong references to backends; ensure `hub.close()` for deterministic teardown. (`src/esper/nissa/output.py:441`)
- **Cleanup:** `DiagnosticTracker.cleanup()` must be called; `__del__` is a best-effort fallback only. (`src/esper/nissa/tracker.py:526`, `src/esper/nissa/tracker.py:532`)

---

# 5. Dependencies

## 5.1 Upstream (Modules that call this module)

| Module | Interaction | Failure Impact |
|--------|-------------|----------------|
| `simic.vectorized` | Adds `BlueprintAnalytics` backend; emits training events | Mostly **Tier-2** (loss of observability). (`src/esper/simic/vectorized.py:484`) |
| `scripts.train` | Attaches Console/File/Directory/TUI/Dashboard backends | **Tier-2** unless file/dash setup exceptions abort run. (`src/esper/scripts/train.py:188`) |
| `tamiyo.tracker` | Emits `TAMIYO_INITIATED` on stabilization | Lost “germination gate opened” signal only. (`src/esper/tamiyo/tracker.py:121`) |
| `tolaria.governor` | Emits rollback telemetry | Lost catastrophe visibility only. (`src/esper/tolaria/governor.py:187`) |
| `kasmina.slot` | Emits seed lifecycle events via callback/hub | Lost lifecycle observability only. (`src/esper/kasmina/slot.py:848`) |

## 5.2 Downstream (Modules this module depends on)

| Module | Interaction | Failure Handling |
|--------|-------------|------------------|
| `leyline.telemetry` | Defines `TelemetryEvent`, `TelemetryEventType` | **Fatal** to type-level contract. (`src/esper/leyline/telemetry.py:34`) |
| `karn.store` | Can import Nissa directory logs | Optional; post-hoc tooling. (`src/esper/karn/store.py:478`) |

## 5.3 External Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `torch` | DiagnosticTracker hooks + probes | If absent, NissaHub/backends still usable (but tracker unusable). (`src/esper/nissa/tracker.py:25`) |
| `numpy` | Aggregate stats in DiagnosticTracker | Optional (could be replaced with pure Python). (`src/esper/nissa/tracker.py:24`) |
| `pydantic` | Validated TelemetryConfig | Required for config loading. (`src/esper/nissa/config.py:23`) |
| `yaml` | Profile/config parsing | Required for profiles. (`src/esper/nissa/config.py:22`) |

---

# 6. Esper Integration

## 6.1 Commandment Compliance

| # | Commandment | Compliance | Notes |
|---|-------------|------------|-------|
| 1 | Sensors match capabilities | ✅ | Nissa is the “sensory bus”; routes all events to output sinks. (`src/esper/nissa/output.py:393`) |
| 2 | Complexity pays rent | ✅ | BlueprintAnalytics estimates compute cost multipliers and tracks params added. (`src/esper/nissa/analytics.py:21`, `src/esper/nissa/analytics.py:86`) |
| 3 | GPU-first iteration | ⚠️ | DiagnosticTracker minimizes syncs per stat, but still syncs once per tracked param; keep profiles conservative. (`src/esper/nissa/tracker.py:176`, `src/esper/nissa/tracker.py:208`) |
| 4 | Progressive curriculum | N/A | Not owned by telemetry. |
| 5 | Train Anything protocol | ✅ | Telemetry contract is host-agnostic: `TelemetryEvent(data: dict)`. (`src/esper/leyline/telemetry.py:91`) |
| 6 | Morphogenetic plane | ✅ | Observes slot/seed events without embedding slot logic. (`src/esper/nissa/analytics.py:145`) |
| 7 | Governor prevents catastrophe | ✅ | Surfaces `GOVERNOR_*` events and formats them for humans. (`src/esper/nissa/output.py:163`) |
| 8 | Hierarchical scaling | ⚠️ | Nissa is flat routing; hierarchy and dashboards increasingly live in Karn. (`src/esper/karn/__init__.py:1`) |
| 9 | Frozen Core economy | ✅ | Telemetry logs fossilization/prune decisions and outcomes for later reuse. (`src/esper/nissa/output.py:92`) |

## 6.2 Biological Role

**Analogy:** Sensory organs + afferent nerves—convert internal training activity into observable signals without changing the organism’s behavior.

**Responsibilities in the organism:**
- Forward events to human-visible channels (console/TUI) and machine-readable logs (JSONL).
- Provide lightweight aggregation (blueprint scoreboard) for strategy evaluation.

**Interaction with other organs:**
- Receives signals from: Kasmina (seed lifecycle), Tamiyo (stabilization), Simic (training), Tolaria (governor). (`src/esper/leyline/telemetry.py:6`)
- Sends signals to: Humans/tools (Console/File/Directory/TUI/Dashboard backends). (`src/esper/scripts/train.py:199`)

## 6.3 CLI Integration

Training CLI wires Nissa backends into the global hub:

| Command | Flags | Effect on Nissa |
|---------|-------|-----------------|
| `python -m esper.scripts.train ...` | `--telemetry-level {off,minimal,normal,debug}` | Controls console severity filter + event volume. (`src/esper/scripts/train.py:170`) |
| `python -m esper.scripts.train ...` | `--telemetry-file PATH` | Adds `FileOutput(PATH)` backend. (`src/esper/scripts/train.py:211`) |
| `python -m esper.scripts.train ...` | `--telemetry-dir DIR` | Adds `DirectoryOutput(DIR)` backend (timestamped run folder). (`src/esper/scripts/train.py:218`) |
| `python -m esper.scripts.train ...` | `--no-tui` / `--tui-layout ...` | Switches between TUI backend and console backend. (`src/esper/scripts/train.py:199`) |
| `python -m esper.scripts.train ...` | `--dashboard --dashboard-port PORT` | Adds WebSocket dashboard backend (Karn). (`src/esper/scripts/train.py:232`) |

---

# 7. Cross-References

## 7.1 Related Bibles

| Bible | Relationship | Integration Point |
|------|--------------|-------------------|
| [leyline](leyline.md) | **Defines contracts for** | `TelemetryEvent`, `TelemetryEventType`. (`src/esper/leyline/telemetry.py:91`) |
| [kasmina](kasmina.md) | **Observed by** | Emits `SEED_*` lifecycle events to hub. (`src/esper/kasmina/slot.py:848`) |
| [tamiyo](tamiyo.md) | **Observed by** | Emits `TAMIYO_INITIATED`. (`src/esper/tamiyo/tracker.py:121`) |
| [simic](simic.md) | **Observed by / configures** | Wires `BlueprintAnalytics` backend and emits training events. (`src/esper/simic/vectorized.py:484`) |
| [tolaria](tolaria.md) | **Observed by** | Emits governor panic/rollback. (`src/esper/tolaria/governor.py:187`) |
| [karn](karn.md) | **Successor / consumer** | Can import Nissa logs; also provides Nissa-compatible backends (TUI/Dashboard). (`src/esper/karn/store.py:478`, `src/esper/karn/tui.py:1`) |

## 7.2 Key Source Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/esper/nissa/output.py` | Hub + output backends | `NissaHub`, `get_hub`, `ConsoleOutput`, `FileOutput`, `DirectoryOutput` |
| `src/esper/nissa/config.py` | Diagnostic configuration | `TelemetryConfig`, `GradientConfig`, `deep_merge` |
| `src/esper/nissa/tracker.py` | Rich diagnostics collector | `DiagnosticTracker`, `EpochSnapshot`, `GradientHealth` |
| `src/esper/nissa/analytics.py` | Blueprint aggregation | `BlueprintAnalytics`, `SeedScoreboard`, `compute_cost_for_blueprint` |
| `src/esper/nissa/profiles.yaml` | Built-in diagnostic profiles | `profiles.{minimal,standard,diagnostic,research}` |

## 7.3 Test Coverage

| Test File | Coverage | Critical Tests |
|-----------|----------|----------------|
| `tests/nissa/test_output.py` | DirectoryOutput + hub routing | Timestamped dir creation, JSONL write, hub→dir integration |
| `tests/integration/test_telemetry_event_formatters.py` | Console formatting + hub fanout | New event type formatting, backend routing semantics |
| `tests/test_nissa_analytics.py` | BlueprintAnalytics correctness | Germinate/fossilize/prune tracking, snapshot serializability |
| `tests/integration/test_blueprint_analytics.py` | End-to-end lifecycle integration | Real model lifecycle emits events and updates analytics |

---

# 8. Tribal Knowledge

## 8.1 Known Limitations

| Limitation | Impact | Workaround | Reference |
|------------|--------|------------|-----------|
| Synchronous emit path (no queue/backpressure) | Slow backends (console spam, disk flush) directly slow training | Use TUI, raise `FileOutput.buffer_size`, avoid verbose/debug except when needed | `src/esper/nissa/output.py:428` |
| Global singleton hub is mutable | Tests and long-running processes can accidentally leak backends across runs | In tests, monkeypatch `_global_hub` and restore; in apps, call `hub.close()` and/or reset global explicitly | `src/esper/nissa/output.py:454`, `tests/tamiyo/test_tracker.py:24` |
| DiagnosticTracker is not integrated into hub | Rich stats don’t automatically appear in Nissa outputs | Call `DiagnosticTracker.end_epoch()` and then emit a custom `TelemetryEvent` yourself if desired | `src/esper/nissa/tracker.py:344`, `src/esper/leyline/telemetry.py:91` |

## 8.2 Performance Cliffs

| Operation | Trigger | Symptom | Mitigation | Reference |
|-----------|---------|---------|------------|-----------|
| Gradient hook stats | Tracking many params + percentiles | Large backward slowdown (sync per tracked param) | Prefer `profiles.standard` with explicit layer list; minimize percentiles | `src/esper/nissa/tracker.py:164`, `src/esper/nissa/tracker.py:208`, `src/esper/nissa/profiles.yaml:28` |
| Loss sharpness probe | `loss_landscape.enabled=True` on large GPU models | GPU OOM / major slowdown (state_dict cloning + reload) | Disable for large models; run on CPU or reduce samples | `src/esper/nissa/tracker.py:288`, `src/esper/nissa/profiles.yaml:69` |
| Console formatting churn | High-frequency events in debug/verbose | Training appears “stuck” due to stdout bottleneck | Use TUI or `min_severity="info"` and avoid `verbose=True` | `src/esper/nissa/output.py:59`, `src/esper/scripts/train.py:181` |

## 8.3 Common Pitfalls

| Pitfall | Why It Happens | Correct Approach | Reference |
|---------|----------------|------------------|-----------|
| Confusing `TelemetryConfig` types | There is a Nissa `TelemetryConfig` (diagnostics) and a Simic `TelemetryConfig` (telemetry level) | In CLI/training, use `esper.simic.telemetry_config.TelemetryConfig`; in diagnostics, use `esper.nissa.config.TelemetryConfig` | `src/esper/nissa/config.py:78`, `src/esper/scripts/train.py:171` |
| BlueprintAnalytics mis-attributes envs | `env_id` defaults to 0 if not present; multi-env callers must inject it | Wrap emission with a callback that sets `event.data["env_id"]` | `src/esper/simic/vectorized.py:505`, `src/esper/nissa/analytics.py:149` |
| Forgetting to cleanup DiagnosticTracker hooks | `__del__` is not deterministic; hooks can accumulate across experiments | Always call `tracker.cleanup()` when done | `src/esper/nissa/tracker.py:526` |

## 8.4 Historical Context / Technical Debt

| Item | Reason It Exists | Future Plan | Reference |
|------|------------------|-------------|-----------|
| `ConsoleOutput.use_color` currently unused | Placeholder for richer formatting, but output is plain prints | Either implement ANSI coloring or remove param | `src/esper/nissa/output.py:59` |
| Nissa vs Karn split-brain | Karn is a richer telemetry system but Nissa remains the hub used by training loops | Decide whether Nissa becomes thin compatibility layer over Karn | `src/esper/karn/__init__.py:1`, `src/esper/nissa/output.py:393` |

## 8.5 Debugging Tips

- **Symptom:** No telemetry appears
  - **Likely Cause:** No backends attached to global hub
  - **Diagnostic:** Verify `get_hub()._backends` length > 0
  - **Fix:** Add `ConsoleOutput()` or `DirectoryOutput()` before training starts (`src/esper/scripts/train.py:188`)

- **Symptom:** Training slows dramatically when diagnostics enabled
  - **Likely Cause:** DiagnosticTracker tracking too many parameters/percentiles
  - **Diagnostic:** Switch to `profiles.standard` and compare runtime
  - **Fix:** Reduce tracked layers and percentiles (`src/esper/nissa/profiles.yaml:28`)

- **Symptom:** Blueprint scoreboard always shows env0
  - **Likely Cause:** `env_id` not injected into events
  - **Diagnostic:** Log `event.data.get("env_id")` at emission site
  - **Fix:** Use the `make_telemetry_callback()` pattern (`src/esper/simic/vectorized.py:505`)

---

# 9. Changelog

| Date | Change | Commit | Impact |
|------|--------|--------|--------|
| 2025-12-14 | Added initial Nissa UMB bible | `40226ca` | Documentation only |
