# Esper Scry Design

> **WebSocket-based browser dashboard for remote Esper training monitoring.**
>
> **Status:** Design Complete
> **Branch:** `feat/overwatch-textual-ui` (shared foundation with Overwatch)
> **Supersedes:** Existing `--dashboard` feature

---

## Overview

**Esper Scry** is a Vue 3 browser dashboard that connects to a running Esper training process via WebSocket. It provides the same monitoring capabilities as the Textual-based Overwatch TUI, but optimized for browser-based remote access.

### Why "Scry"?

Following Esper's MTG naming convention (Kasmina, Tamiyo, Nissa, Karn), "Scry" means viewing at a distance — fitting for remote monitoring.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Name** | Esper Scry | MTG-flavored, "seeing at a distance" |
| **Location** | `src/esper/karn/scry/` | Sibling to Overwatch under Karn (Memory) |
| **Frontend** | Vue 3 + Composition API | Already used in existing dashboard, reactive primitives |
| **Server** | In-process (aiohttp) | Simple deployment, matches `--tui` pattern |
| **Protocol** | Periodic `TuiSnapshot` @ 1-2Hz | Sufficient for human consumption, simple client |
| **Auth** | Simple token on startup | Balances security with zero-config |
| **Layout** | Same mental model, browser-native | Consistent with Overwatch, leverages browser capabilities |
| **Aesthetic** | Dark ops / monitoring | Professional, familiar to ML engineers |
| **Replay** | Server-side | Single client implementation for live and replay |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PROCESS                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │   Simic     │───▶│    Nissa     │───▶│   ScryServer            │ │
│  │  (PPO RL)   │    │ (Telemetry)  │    │  (WebSocket + HTTP)     │ │
│  └─────────────┘    └──────────────┘    │                         │ │
│                                          │  - Aggregates to        │ │
│                                          │    TuiSnapshot @ 1-2Hz  │ │
│                                          │  - Serves Vue SPA       │ │
│                                          │  - Token auth           │ │
│                                          └───────────┬─────────────┘ │
└──────────────────────────────────────────────────────┼───────────────┘
                                                       │ WebSocket
                    ┌──────────────────────────────────┼──────────────┐
                    │              BROWSER(S)          ▼              │
                    │  ┌─────────────────────────────────────────┐   │
                    │  │           Vue 3 SPA                     │   │
                    │  │  - Receives TuiSnapshot                 │   │
                    │  │  - Renders 5 logical regions            │   │
                    │  │  - Sends playback commands (replay)     │   │
                    │  └─────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────┘
```

### Key Points

- **ScryServer** lives inside the training process (spawned by `--scry` flag)
- Aggregates raw telemetry into `TuiSnapshot` (same schema as Overwatch)
- Serves both the Vue SPA (static files) and WebSocket connections
- Multiple browsers can connect simultaneously
- Replay mode: `esper scry replay file.jsonl` — same server, different data source

---

## Server Components

```
src/esper/karn/scry/
├── __init__.py
├── server.py          # ScryServer: HTTP + WebSocket
├── aggregator.py      # TelemetryAggregator → TuiSnapshot
├── protocol.py        # WebSocket message types
├── auth.py            # Token generation and validation
└── static/            # Built Vue SPA (or served from package)
```

### ScryServer (`server.py`)

- Built on `aiohttp` (async HTTP + WebSocket)
- Single class managing both HTTP (serves SPA) and WebSocket (streams snapshots)
- Lifecycle: started by `--scry` flag, runs in background asyncio task
- Binds to `0.0.0.0:{port}` with configurable port (default 8765)

### TelemetryAggregator (`aggregator.py`)

- Subscribes to Nissa telemetry hub
- Maintains rolling window of metrics for trend calculation
- Builds `TuiSnapshot` on demand (or on timer)
- Computes anomaly scores per environment
- **Shared with Overwatch** — same aggregation logic, both produce `TuiSnapshot`

### Protocol (`protocol.py`)

- `ServerMessage`: snapshot, replay_status, error
- `ClientMessage`: replay_control (play, pause, step, speed), subscribe filters
- JSON serialization (simple, debuggable, sufficient for 1-2Hz)

### Auth (`auth.py`)

- `generate_token()` → 32-char random hex
- `validate_token(request)` → bool
- Token passed via query param: `ws://host:8765/ws?token=abc123`

---

## Vue Client Architecture

```
src/esper/karn/scry/frontend/
├── index.html
├── main.ts                    # App entry, WebSocket setup
├── App.vue                    # Root layout
├── composables/
│   ├── useScrySocket.ts       # WebSocket connection + reconnect
│   ├── useSnapshot.ts         # Reactive snapshot state
│   ├── usePinnedEnvs.ts       # Pinned environments (localStorage)
│   ├── useBestRuns.ts         # Leaderboard with dismiss
│   └── useReplayControls.ts   # Playback state machine
├── components/
│   ├── layout/
│   │   ├── ScryHeader.vue     # Connection, resources, run identity
│   │   ├── TamiyoStrip.vue    # PPO vitals, action mix, confidence
│   │   ├── FlightBoard.vue    # Env grid with slot chips
│   │   ├── DetailPanel.vue    # Context or Tamiyo detail (tabbed)
│   │   └── EventFeed.vue      # Scrolling event log
│   ├── widgets/
│   │   ├── EnvCard.vue        # Single environment row
│   │   ├── SlotChip.vue       # Seed stage indicator
│   │   ├── HealthBadge.vue    # OK/WARN/CRIT status
│   │   ├── TrendArrow.vue     # ↑↑ ↑ → ↓ ↓↓ indicator
│   │   ├── Sparkline.vue      # SVG mini chart
│   │   ├── ProgressBar.vue    # Alpha / utilization bars
│   │   └── BestRunsPanel.vue  # Leaderboard with dismiss
│   └── replay/
│       └── ReplayBar.vue      # Play/pause, scrubber, speed
├── types/
│   └── snapshot.ts            # TypeScript types matching TuiSnapshot
└── styles/
    └── theme.css              # Dark ops color palette
```

### Key Composables

- **`useScrySocket`** — Manages WebSocket lifecycle: connect, reconnect with backoff, token auth. Returns reactive `connectionStatus`.

- **`useSnapshot`** — Receives snapshots, exposes as `ref<TuiSnapshot>`. All components read from this single source of truth.

- **`usePinnedEnvs`** — Manages Set of pinned env IDs, syncs to localStorage.

- **`useBestRuns`** — Tracks best runs, handles dismissals, syncs to localStorage.

- **`useReplayControls`** — State machine for replay (playing/paused/stepping). Sends control messages to server.

### Data Flow

```
WebSocket → useSnapshot.snapshot → App.vue → child components (reactive)
```

No aggregation in the client — server sends complete snapshots, Vue just renders.

---

## UI Layout

Same mental model as Overwatch (5 logical regions), but browser-native layout:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ESPER SCRY                                          GPU0 94% ▓▓▓▓▓▓▓░░ 11/12G│
│ cifar10 · Episode 47 · 2h 14m           ● Connected  GPU1 91% ▓▓▓▓▓▓░░░ 10/12G│
├────────────────────────────────────────────┬────────────────────────────────┤
│ TAMIYO                                     │ FLIGHT BOARD                   │
│ ┌─────────────────────────────────────┐    │                                │
│ │ KL 0.019 ✓   Entropy 1.24 ↓         │    │ ┌─ Pinned ───────────────────┐ │
│ │ Clip 4.8% ✓  Expl.Var 0.42 ↓↓       │    │ │ 📌 Env 2  gpu:1  OK        │ │
│ │ Grad 0.8     LR 3e-4                │    │ │    [r0c1] TRAINING ▓▓▓░ .45│ │
│ └─────────────────────────────────────┘    │ └────────────────────────────┘ │
│ Actions: G 34% B 28% C 12% W 26%           │ ┌────────────────────────────┐ │
│ Recent: [G][B][B][W][G][G][B]              │ │ ⚠ Env 3  gpu:1  WARN   [📌]│ │
│ Confidence: 73% ▓▓▓▓▓▓▓░░░                 │ │   [r0c1] BLENDING ▓▓▓▓░ .78│ │
│                                            │ │   102 fps · grad 3.2x      │ │
│ ┌─ Entropy Trend ────────────────────┐     │ └────────────────────────────┘ │
│ │    ╭──╮    ╭─                      │     │ ┌────────────────────────────┐ │
│ │ ──╯  ╰────╯                        │     │ │ ✓ Env 0  gpu:0  OK     [📌]│ │
│ └────────────────────────────────────┘     │ │   [r0c1] FOSSILIZED ▓▓▓▓▓▓ │ │
│                                            │ │   98 fps                   │ │
│ ┌─ BEST RUNS ─────────────────────────┐    │ └────────────────────────────┘ │
│ │      Env   Acc     Reward   Params  │    │ ┌────────────────────────────┐ │
│ │ 🥇   0    82.1%    +47.2   +1.2M [×]│    │ │ ✓ Env 1  gpu:0  OK     [📌]│ │
│ │ 🥈   2    81.3%    +52.1   +0.8M [×]│    │ │   [r0c0] TRAINING ▓▓░░░ .32│ │
│ │ 🥉   1    79.8%    +38.9   +1.1M [×]│    │ └────────────────────────────┘ │
│ └─────────────────────────────────────┘    │                                │
├────────────────────────────────────────────┴────────────────────────────────┤
│ DETAIL                                                                       │
│ ┌─ Why Flagged: Env 3 ─────────────────────────────────────────────────────┐ │
│ │ • High gradient ratio (3.2× mean)                                        │ │
│ │ • Memory pressure (94% utilized)                                         │ │
│ │ • Slot r0c1 alpha plateau (5 epochs)                                     │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────────────┤
│ EVENT FEED                                                          [Filter] │
│ 12:04:23  [GATE]  Env 3 r0c1 gate opened (grad health 0.82)                  │
│ 12:04:18  [PPO]   Policy update: KL=0.019, clip=4.8%                         │
│ 12:03:55  [STAGE] Env 0 r0c1 BLENDING → FOSSILIZED                           │
│ 12:03:41  [WARN]  Env 3 anomaly score exceeded threshold (0.72)              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layout Differences from Overwatch TUI

| Aspect | Overwatch (TUI) | Scry (Web) |
|--------|-----------------|------------|
| **Tamiyo + Detail** | Horizontal strip + side panel | Left column (stacked) |
| **Flight Board** | Center with side detail | Right column (scrollable) |
| **Sparklines** | ASCII approximation | SVG charts |
| **Env cards** | Compact rows | Cards with more whitespace |
| **Responsiveness** | Fixed terminal size | Adapts to viewport |

### Unique Features

| Feature | Description |
|---------|-------------|
| **Pinned envs** | 📌 icon pins env to top of Flight Board, persists in localStorage |
| **Best Runs leaderboard** | Top 3 by accuracy, shows reward for Goodhart detection |
| **Dismissable entries** | `×` removes investigated items from leaderboard |
| **SVG sparklines** | Real charts for trends |
| **Sortable leaderboard** | Click headers to sort by Accuracy/Reward/Efficiency |

---

## CLI Integration

### Training with Scry

```bash
# Start training with Scry server
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 \
    --scry                    # Enable Scry server
    --scry-port 8765          # Optional, default 8765

# Output on startup:
# Scry server started at:
#   Local:   http://localhost:8765?token=a3f9b2...
#   Network: http://192.168.1.50:8765?token=a3f9b2...
```

### Replay Mode

```bash
# Replay a saved session
PYTHONPATH=src uv run python -m esper.scripts.scry replay \
    --file telemetry/run_2025-01-15.jsonl \
    --port 8765

# Same URLs displayed, browser connects normally
```

### Flags Summary

| Flag | Default | Description |
|------|---------|-------------|
| `--scry` | off | Enable Scry server during training |
| `--scry-port` | 8765 | Port for HTTP + WebSocket |
| `--no-scry-auth` | off | Disable token auth (LAN-trust mode) |

---

## Implementation Stages

**Shared Foundation:** Scry and Overwatch share the same `TuiSnapshot` schema and replay infrastructure.

| Stage | Name | Deliverable | Dependency |
|-------|------|-------------|------------|
| **0** | **Shared Schema** | `TuiSnapshot`, `EnvSummary`, `SlotChipState`, etc. + `SnapshotWriter`/`SnapshotReader` | *Same as Overwatch Stage 0* |
| **0.5** | Schema Extensions | `BestRunEntry` for leaderboard, add `best_runs` field to `TuiSnapshot` | Stage 0 |
| **1** | Server Skeleton | `ScryServer` (aiohttp), token auth, serves placeholder HTML | Stage 0 |
| **2** | Vue Scaffold | Vite project, `useScrySocket`, displays raw JSON | Stage 1 |
| **3** | Layout Shell | 5 regions with placeholders, dark ops CSS | Stage 2 |
| **4** | Flight Board | EnvCard, SlotChip, pinning, anomaly sort | Stage 3 |
| **5** | Tamiyo + Best Runs | TamiyoStrip, sparklines, leaderboard with reward | Stage 3 |
| **6** | Header + Detail | Connection status, resources, "why flagged" | Stage 3 |
| **7** | Event Feed | Scrolling log, filtering, event badges | Stage 3 |
| **8** | Replay Controls | Play/pause/step/speed, scrubber | Stage 2 |
| **9** | Live Integration | Wire aggregator to Nissa, `--scry` flag | Stage 0.5 |

### Stage 0.5: Schema Extensions

Scry needs one addition to the shared schema — the **Best Runs leaderboard**:

```python
# src/esper/karn/overwatch/schema.py (addition)

@dataclass
class BestRunEntry:
    """Entry in the Best Runs leaderboard."""

    env_id: int
    best_accuracy: float
    cumulative_reward: float
    param_delta: int  # Parameter count delta from baseline
    slot_configs: dict[str, str]  # slot_id -> final stage
    achieved_at_episode: int
    dismissed: bool = False  # User dismissed from leaderboard

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BestRunEntry: ...
```

Then extend `TuiSnapshot`:

```python
@dataclass
class TuiSnapshot:
    # ... existing fields ...

    # Best runs leaderboard (Scry extension)
    best_runs: list[BestRunEntry] = field(default_factory=list)
```

This is backwards-compatible — Overwatch ignores `best_runs`, Scry uses it.

---

## File Structure

```
src/esper/karn/
├── overwatch/                    # SHARED (Stage 0)
│   ├── __init__.py
│   ├── schema.py                 # TuiSnapshot, EnvSummary, BestRunEntry, etc.
│   └── replay.py                 # SnapshotWriter, SnapshotReader
│
└── scry/                         # SCRY-SPECIFIC (Stage 1+)
    ├── __init__.py
    ├── server.py                 # ScryServer (HTTP + WebSocket)
    ├── aggregator.py             # TelemetryAggregator
    ├── protocol.py               # WebSocket message types
    ├── auth.py                   # Token generation/validation
    └── frontend/                 # Vue SPA
        ├── index.html
        ├── main.ts
        ├── vite.config.ts
        ├── App.vue
        ├── composables/
        │   ├── useScrySocket.ts
        │   ├── useSnapshot.ts
        │   ├── usePinnedEnvs.ts
        │   ├── useBestRuns.ts
        │   └── useReplayControls.ts
        ├── components/
        │   ├── layout/
        │   │   ├── ScryHeader.vue
        │   │   ├── TamiyoStrip.vue
        │   │   ├── FlightBoard.vue
        │   │   ├── DetailPanel.vue
        │   │   └── EventFeed.vue
        │   ├── widgets/
        │   │   ├── EnvCard.vue
        │   │   ├── SlotChip.vue
        │   │   ├── HealthBadge.vue
        │   │   ├── TrendArrow.vue
        │   │   ├── Sparkline.vue
        │   │   ├── ProgressBar.vue
        │   │   └── BestRunsPanel.vue
        │   └── replay/
        │       └── ReplayBar.vue
        ├── types/
        │   └── snapshot.ts
        └── styles/
            └── theme.css
```

---

## Testing Strategy

| Layer | Approach | Tools |
|-------|----------|-------|
| **Schema** | Unit tests for serialization round-trips | pytest (shared with Overwatch) |
| **Server** | Integration tests with mock WebSocket clients | pytest-aiohttp |
| **Aggregator** | Unit tests with mock telemetry events | pytest |
| **Vue Components** | Component tests with mock snapshots | Vitest + Vue Test Utils |
| **E2E** | Playwright tests against running server | Playwright |

**Key test scenarios:**

1. WebSocket connection with valid/invalid token
2. Snapshot streaming at correct cadence
3. Replay playback controls (pause/step/speed)
4. Vue reactivity on snapshot updates
5. Pinning/dismissing persists to localStorage
6. Graceful handling of connection loss + reconnect

**Shared fixtures:** Scry reuses the JSONL fixtures from Overwatch Stage 0:
- `healthy_run.jsonl`
- `anomaly_detected.jsonl`
- `tamiyo_active.jsonl`

---

## Comparison: Scry vs Overwatch

| Aspect | Overwatch (TUI) | Scry (Web) |
|--------|-----------------|------------|
| **Runtime** | Textual in terminal | Vue in browser |
| **Access** | Local terminal only | Remote via browser |
| **Multi-viewer** | No | Yes (multiple clients) |
| **Aesthetic** | Terminal ASCII | Dark ops monitoring |
| **Charts** | ASCII sparklines | SVG sparklines |
| **Pinning** | No | Yes |
| **Leaderboard** | No | Yes (Best Runs) |
| **Dependencies** | textual | aiohttp, Vue 3, Vite |
| **Replay** | Local file | Server-streamed |

**Shared:** Schema (`TuiSnapshot`), replay infrastructure, aggregation logic.

---

## Dependencies

### Python (add to pyproject.toml)

```toml
[project.optional-dependencies]
scry = [
    "aiohttp>=3.9.0",
]
```

### Frontend (package.json in frontend/)

```json
{
  "dependencies": {
    "vue": "^3.4.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@vitejs/plugin-vue": "^5.0.0",
    "typescript": "^5.3.0",
    "vitest": "^1.0.0",
    "@vue/test-utils": "^2.4.0"
  }
}
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| WebSocket latency | <100ms snapshot delivery |
| Reconnect time | <3s on connection loss |
| Vue render time | <50ms for 16 envs |
| Bundle size | <200KB gzipped |
| Test coverage | >80% for new code |

---

## Next Steps

1. **If Overwatch Stage 0 not yet complete:** Execute Overwatch Stage 0 first (shared schema)
2. **Stage 0.5:** Add `BestRunEntry` schema extension
3. **Stage 1:** Implement `ScryServer` skeleton
4. **Continue through stages...**

Each stage will be expanded into a detailed implementation plan before execution.
