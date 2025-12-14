# Esper‑Overwatch UI

## Design philosophy and thinking (UI-agnostic, reuse for Web GUI)

### 1) From scoreboard to Air Traffic Control

At small scale, a flat table is fine. At real scale (GPUs × envs × slots), a flat table becomes a **liar**: it hides structure, masks bottlenecks, and overwhelms the operator.

**Esper‑Overwatch must behave like ATC:**

* **Sectorise first** (group by GPU / node / worker lane).
* **Highlight distress signals** (outliers-first sorting by anomaly).
* **Make drill-down cheap** (one keypress from overview → env detail → seed/slot detail).
* **Keep the safety rails visible** (policy health, staleness, resource bounds, divergence flags).

### 2) Encode the actual topology as the UI’s primary organising principle

Esper’s runtime shape is hierarchical:

**Run → Devices (GPU lanes) → Environments → Slots → Seeds**

The UI must represent that hierarchy explicitly. Any design that hardcodes `early/mid/late` as “columns” is already obsolete; slots are a **set**, not a schema.

**Core rule:** the renderer iterates *slot dictionaries*, not slot names.

### 3) “Outliers-first” is not a nice-to-have — it’s how humans scale

Humans don’t scan 64 rows. They scan **what’s on fire**.

So the system must compute an **Anomaly Score** for each env (and optionally each GPU lane) and sort by it by default. “Normal” envs should be deemphasised (collapsed, paged, or hidden behind a filter), while outliers are surfaced.

Anomaly scoring is a diagnostic tool, not a truth oracle:

* It must be **explainable** (show which factors contributed).
* It must be **pluggable** (weights/toggles configurable).
* It must be **stable** (avoid jittery resorting every frame).

### 4) The UI is a view over telemetry, not a second analytics engine

A world-class UI fails if it computes heavy metrics inside the render loop.

**Non-negotiable separation:**

* **Telemetry producers** emit raw counters/stats/events.
* **Aggregator** derives higher-level signals (bounds, anomaly, rollups).
* **UI** renders a **snapshot** produced by the aggregator.

This keeps both TUI and future Web GUI honest and consistent, and it prevents “UI-induced stalls”.

### 5) Interpret numbers into diagnoses (“Bound detector”)

Don’t show “GPU util 40%” and force the user to infer the bottleneck.

Esper‑Overwatch must interpret infrastructure signals into a small set of **actionable bound states**, e.g.:

* **Compute-bound** (good / expected during heavy training)
* **Memory-bound / OOM-risk**
* **I/O-bound** (dataloader starvation)
* **Sync-bound** (sawtooth util, barrier waits, stragglers)
* **Thermal/power throttled**

This becomes the operator’s “what do I fix?” prompt.

### 6) Always-visible “Policy Pulse” (brain) + “Body Vitals” (infrastructure)

Learning can fail in a healthy system (bad PPO health) and systems can fail in healthy learning (dataloader starvation). Both must be visible **together** to correlate cause/effect.

**Always visible:**

* PPO health: KL, entropy, clip frac, explained variance, grad norm, update cadence
* System health: per-GPU util/mem/temp/power + CPU/RAM/I/O + telemetry staleness + render latency

### 7) A stable visual grammar (chip strip) that works everywhere

Slots/seeds need a compact representation that scales from TUI to Web.

Define a **slot “chip” grammar**:

* `glyph + slot_label + blueprint + alpha(optional) + stage(optional)`
* glyph conveys stage/status (with colour where available)
* blueprint is short and consistent (registry ID, not verbose name)

This is crucial because it becomes the *shared language* across TUI, logs, and web dashboard.

### 8) Telemetry parity: everything visible must be serialisable

If the UI can show it, it must be representable as structured telemetry:

* snapshots (low-frequency)
* streams (high-frequency, decimated)
* events (structured log)

This unlocks:

* offline replay (“post-mortem TUI”)
* side-by-side compare runs later
* web GUI reusing the same schema

### 9) Performance and truthfulness over prettiness

The UI must never:

* introduce extra synchronisation points,
* force GPU sync for metrics,
* starve the trainer thread,
* or block on log I/O.

It should **show staleness** and degrade gracefully rather than pretend.

### 10) Operator workflow: glance → diagnose → confirm → drill

The UI should support a simple loop:

1. glance at global pulse + infra bounds
2. see top anomalies (envs / GPUs)
3. drill into one env
4. inspect per-slot/seed details + reasons (gates, regression, grad health)
5. confirm via raw telemetry feed

That’s “active diagnostics” rather than “passive monitoring”.

---

## Confidence (WEP)

It’s **highly likely** that this hierarchy + outliers-first + telemetry-parity approach will scale cleanly to many envs/GPUs without turning into a scroll festival, and it’s **likely** it will reduce time-to-diagnosis when runs go sideways (stalls, cull storms, policy collapse).

---

# Master tasking statement for a coding agent (TUI build)

## A) Deliverable

Build **Esper‑Overwatch**, an interactive terminal UI for Esper/esper-lite PPO training that supports:

* **1..X GPUs**
* **1..Y envs per GPU**
* **0..K slots per env (dynamic, not hardcoded)**
* **seed lifecycle telemetry** per slot and per env
* **policy health + infrastructure health** always visible
* **raw structured telemetry feed** always visible
* **drill-down per environment** (detail drawer / overlay)

Target runtime: SSH-friendly, low overhead, resilient to missing metrics.

---

## B) Scope and non-scope

### In-scope (v1)

* Overview screen with four core panels:

  1. Global Context + Policy Pulse (always visible header)
  2. System Performance & Stats (GPU/CPU/IO + bound detector)
  3. Per‑Env “Flight Board” (grouped by GPU; outliers-first; chip strip)
  4. Raw Telemetry Feed (structured events; filter/search)
* Keyboard navigation (select env, open details, toggle sort/filter, help overlay)
* A **Telemetry Aggregator** producing UI snapshots outside render loop
* Snapshot logging (JSONL) so the UI can be replayed offline

### Explicit non-scope (v1)

* Sending control commands back into training (pause env, “checkpoint now”, etc.)
* Multi-node UI federation (can be designed for, not implemented)
* Full histogram visualisations that require heavy sampling (keep light)

---

## C) Architecture requirements

### 1) Data model: Snapshot-first

Implement a `TuiSnapshot` (or equivalent) as the sole input to rendering.

**Requirements:**

* Snapshot is **pre-aggregated** (already sorted, already summarised)
* Snapshot includes staleness timing (now - last_update)
* Snapshot includes render timing metrics (last_render_ms, avg_render_ms)

Suggested shape (names flexible, semantics fixed):

* `RunContext`

  * run_id, task, algo, reward_mode, git_sha, dirty_flag
  * devices list, n_envs, slots_enabled (informational), compile/amp/ddp modes
  * episode/batch/update counters, elapsed, ETA, checkpoint age/path

* `PolicyHealth`

  * KL, entropy, clip_frac, explained_variance, grad_norm, lr, update_time_ms
  * each metric has: value, status (OK/WARN/CRIT), threshold used

* `SystemStats`

  * per_gpu: util%, mem_used/total, tempC, powerW, clocks, throttle flags
  * cpu%, ram_used/total, disk_io, net_io, dataloader queue depth (if available)
  * `bound_state` per GPU and global (compute/memory/io/sync/throttle)

* `EnvSummary[]` (pre-sorted)

  * env_id, gpu_id, fps/step_time, acc (or task metric), reward, rent
  * last_action (decoded), status (OK/STALLED/DEGRADED/DIVERGING/CRASHED)
  * anomaly_score + anomaly_reasons (top 2–3 factors)
  * slots: `Dict[slot_name, SlotChipState]` (dynamic keys)

* `SlotChipState`

  * stage, stage_glyph, blueprint_id, alpha (optional), age, gate_status summary

* `Aggregates`

  * seed_census by stage (total, per GPU optional)
  * action distribution (global + per GPU optional)
  * top env ids (best acc, worst acc, best reward, worst reward)

* `EventFeed`

  * last N structured events with severity, component, env_id, gpu_id, slot, seed_id, message, fields

### 2) Aggregator: compute derived signals outside the UI thread

Implement a `TelemetryAggregator` (or similarly named component) that:

* subscribes to existing telemetry/events (policy stats, env stats, seed events, system stats)
* maintains ring buffers / sliding windows
* emits snapshots at a configured cadence

**Cadence requirements:**

* Fast lane (e.g. 5–10 Hz): env summaries, actions, rewards, slot chips
* Slow lane (e.g. 1 Hz): GPU temps/power, IO stats, percentile latency

**Performance:**

* aggregator must avoid GPU synchronisation
* heavy stats must be decimated / sampled

### 3) Anomaly Score (pluggable + explainable)

Implement a scoring function `score_env(env)` returning:

* `score: float`
* `reasons: List[str]` (human readable, short)

Minimum factors (v1):

* stalled fps / high step_time
* reward collapse (rolling mean drop)
* high rent with low reward/Δacc
* excessive cull rate / gate failures
* NaN/Inf flags (hard CRIT)

Sorting behaviour:

* default = anomaly desc
* stable sort with hysteresis (don’t reorder wildly every frame)

### 4) Bound detector (interpreting infrastructure)

Implement a `BoundDetector` producing per-GPU and global bound states:

* compute-bound
* memory-bound / OOM-risk
* I/O-bound (dataloader starvation)
* sync-bound (util sawtooth / barrier waits if available)
* thermal/power throttled

Each bound state should include a one-line “hint” string for operator action.

### 5) Slot chip strip grammar (stage glyph mapping)

Define a canonical mapping from seed stage → glyph (and status label):

* includes at minimum: dormant/empty, germinated, training, blending, probationary, fossilised, culled, embargoed/resetting
* stage must be derived from source-of-truth stage enum, not ad hoc strings
* blueprint should render as short ID (registry ID), not verbose class names

### 6) Telemetry feed must be structured first, pretty second

Events in the feed must include key=value fields:

* timestamp
* severity
* subsystem (simic/tolaria/kasmina/nissa)
* env_id, gpu_id, slot, seed_id (when applicable)
* action + gate reason + key stats (grad health, Δacc, rent, etc.)

### 7) Offline replay mode (v1 if possible, v1.5 acceptable)

The same TUI should be able to:

* read a telemetry JSONL (or msgpack) log
* drive snapshots deterministically
* allow pause/step through time

---

## D) UI layout requirements (Rich/Textual)

### Global Context + Policy Pulse (always visible)

Must show:

* run identity: run_id, task, reward mode, amp/ddp/compile status, git sha dirty flag
* progress: episode/batch/update, elapsed, ETA, checkpoint age
* policy pulse with threshold colour/status: KL, entropy, clip frac, explained variance, grad norm, lr
* telemetry staleness + render latency indicators

### System Performance & Stats panel

Must show:

* per-GPU line items: util/mem/temp/power + bound label (and warning icon if unhealthy)
* CPU/RAM and at least one IO indicator (disk/net/dataloader depth)
* top alerts as a short “alarm rail”

### Per‑Env “Flight Board” panel

Must support:

* grouping by GPU (collapsible headers)
* default sort: anomaly desc
* scroll/paging for many envs
* per-row: env_id, gpu_id, fps, acc/metric, reward, rent, last action, slot chip strip, status badge
* a “top-N outliers + summary” mode for compact terminals

### Per‑Env child panel (detail drawer / overlay)

When an env is selected:

* list all slots (dynamic) with stage, blueprint, alpha, age, gate status
* recent action history for that env
* last few seed lifecycle events for that env
* key trends (sparklines): reward, acc/metric, rent (short window)

### Raw Telemetry Feed panel

Must show:

* last N events (scrollback)
* quick filter toggles (errors only, seed events only, policy updates only)
* search by env_id/seed_id keyword

### Help overlay

Must show:

* hotkeys
* legend for glyphs/status colours
* current sort/filter mode

---

## E) Interaction model (keyboard-first)

Minimum hotkeys (v1):

* `↑/↓` or `j/k`: move selection in env list
* `Enter`: open/close env detail drawer
* `g`: jump to global/overview (reset focus)
* `s`: cycle sort (anomaly, env_id, reward, fps, acc)
* `/`: filter prompt (env=, gpu=, status=, blueprint=)
* `l`: focus log panel; `e`: focus env panel
* `?`: help overlay
* `q`: quit

---

## F) Engineering tasks (step-by-step)

### Phase 0 — Discovery (short)

1. Locate existing TUI implementation and current telemetry sources (policy stats, env stats, seed events, system stats).
2. Identify what is already available without adding sync points (e.g., GPU stats provider, existing event bus).

### Phase 1 — Telemetry snapshot pipeline (foundation)

3. Implement `TelemetryAggregator`:

   * ring buffers for per-env rolling windows
   * event ingestion + indexing by env_id/seed_id
   * snapshot emission timer (fast/slow lanes)
4. Implement `AnomalyScore` and `BoundDetector` with unit tests.
5. Implement snapshot serialisation (JSONL) + replay reader.

### Phase 2 — TUI rendering (overview)

6. Build the Overview screen layout with four panels.
7. Render slot chip strips from dynamic slot dicts (no hardcoded early/mid/late).
8. Implement grouping by GPU and stable anomaly sorting (with hysteresis).

### Phase 3 — Interaction + drill-down

9. Add selection model, env detail drawer, and log focus.
10. Add filtering + searching.
11. Add help/legend overlay.

### Phase 4 — Performance hardening

12. Add staleness and render latency instrumentation.
13. Ensure no blocking calls in UI thread; rate-limit heavy metrics.
14. Add “compact mode” that reduces columns when terminal width is small.

### Phase 5 — Acceptance + regression testing

15. Simulate:

* 1 GPU × 4 envs × 3 slots
* 2 GPUs × 32 envs × 6 slots
* env crash storms, cull storms, policy collapse events

16. Confirm UI remains responsive and correctly represents per-env per-slot status.

---

## G) Acceptance criteria (definition of done)

### Correctness

* Works with arbitrary numbers of GPUs/envs/slots.
* Slot display is driven by dynamic slot keys; no schema assumptions.
* Outliers-first default view surfaces stalls, crashes, cull storms, and policy collapse quickly.
* Raw event feed includes env_id/gpu_id/slot/seed_id for lifecycle events.

### Performance

* UI updates do not measurably slow training (no extra GPU sync).
* Rendering remains responsive over SSH (target: <50ms render time typical).
* Slow-lane metrics do not block fast-lane updates.

### Operability

* Users can drill into any env, inspect its slot/seed states, and correlate with events.
* Snapshot logs can be replayed offline to reproduce the same view.

### Extensibility

* Snapshot schema is versioned and reusable for web GUI later.
* New metrics can be added in the aggregator without changing UI architecture.
