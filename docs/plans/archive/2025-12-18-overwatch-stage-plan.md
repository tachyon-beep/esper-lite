# Overwatch Textual UI - Stage Plan

> **Goal:** Ship the Textual "Esper-Overwatch" TUI in incremental stages, each mergeable to main independently.
>
> **Branch:** `feat/overwatch-textual-ui`
>
> **Primary Concerns:** Deployment risk (ship working increments) + Context management (each stage fits one Claude session)

---

## Refined Layout (from UX Review)

Three-region visual separation for clear telemetry domains:

| Region | Contains | Domain |
|--------|----------|--------|
| **Header** | Connection, resources, run identity, dual health indicators | Infrastructure |
| **Tamiyo Strip** | PPO vitals, action summary, confidence, exploration | Agent Brain |
| **Flight Board** | Env rows, slot chips, anomaly scores, throughput | Seeds/Hosts |
| **Detail Panel** | Context ("why flagged") or Tamiyo detail (toggleable) | Drill-down |
| **Event Feed** | Timestamped lifecycle events, filterable | History |

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ESPER OVERWATCH  [✓ ENVS OK] [↓ EV LOW]  Ep: 47  Best: 82.1%  2h 14m   │
│ [● Live 0.3s]  GPU0: 94% 11.2/12GB  GPU1: 91% 10.8/12GB                │
├─────────────────────────────────────────────────────────────────────────┤
│ TAMIYO  KL 0.019✓  Ent 1.24↓  Clip 4.8%✓  EV 0.42↓↓                    │
│ Mix: G34% B28% C12% W26%  Recent: [G][B][B][W][G]  Conf: 73%           │
├─────────────────────────────────────────────────────────────────────────┤
│ FLIGHT BOARD                            │ DETAIL PANEL                  │
│                                         │                               │
│ [!] Env 3  gpu:1  WARN   102 fps        │ WHY FLAGGED:                  │
│     [r0c1] BLENDING ████░░ 0.78α G2✓    │ • High grad ratio (3.2x)      │
│                                         │ • Memory pressure (94%)       │
│ [ ] Env 0  gpu:0  OK     98 fps         │                               │
│     [r0c1] FOSSILIZED ██████ 1.0α       │ [t] Tamiyo [c] Context        │
├─────────────────────────────────────────┴───────────────────────────────┤
│ EVENT FEED [f]                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Slot positions | Numerical tuples `[r0c1]` | Replaced early/mid/late with row,col coordinates |
| Development order | Replay-first | Deterministic test data, easier debugging |
| Tamiyo visibility | Strip always visible | Operators need continuous PPO health awareness |
| Health indicators | Dual `[✓ ENVS]` `[↓ TAMIYO]` | Instant triage of both telemetry domains |
| Sort stability | Hysteresis 3↑ / 5↓ | Prevent visual jitter at threshold boundaries |
| Panel switching | Toggle detail content | `t`/`c` switch detail panel, don't hide regions |
| Refresh cadence | 1Hz slow, 2Hz fast | Balance responsiveness with resource usage |

---

## Stage Overview

| Stage | Name | Deliverable | Est. Size |
|-------|------|-------------|-----------|
| 0 | Foundation + Replay | Snapshot schema, JSONL read/write, test fixtures | Small |
| 1 | App Shell | Textual app launches, basic layout, quit works | Small |
| 2 | Flight Board | Env rows, slot chips, navigation, anomaly sort | Medium |
| 3 | Header + Tamiyo Strip | Dual indicators, PPO vitals, action summary | Medium |
| 4 | Detail Panels | Context panel, Tamiyo detail, toggles, empty states | Medium |
| 5 | Event Feed + Replay | Feed widget, replay controls, filtering | Small |
| 6 | Live Telemetry | Listener wiring, aggregator, `--overwatch` flag | Medium |

---

## Stage 0: Foundation + Replay Infrastructure

### Goal
Establish the data layer: snapshot schema, serialization, and test fixtures that all subsequent stages build upon.

### Scope

**In Scope:**
- `TuiSnapshot` dataclass with all fields
- `EnvSummary` dataclass (env_id, device_id, throughput, metrics, slots, anomaly, status)
- `SlotChipState` dataclass (slot_id as tuple, stage, blueprint_id, alpha, gate status)
- `ConnectionStatus` dataclass with `status_text` property
- `TamiyoState` dataclass (action_distribution, recent_actions, confidence, exploration)
- `to_dict()` / `from_dict()` for JSON serialization
- `SnapshotWriter` — append JSONL with flushing
- `SnapshotReader` — yield snapshots with optional filter
- Test fixtures: 3-5 sample JSONL files covering various states

**Out of Scope:**
- Aggregator logic (Stage 6)
- Any UI code (Stage 1+)
- Telemetry event handling (Stage 6)

### Files to Create
```
src/esper/karn/overwatch/__init__.py
src/esper/karn/overwatch/schema.py          # TuiSnapshot, EnvSummary, SlotChipState, etc.
src/esper/karn/overwatch/replay.py          # SnapshotWriter, SnapshotReader
tests/karn/overwatch/__init__.py
tests/karn/overwatch/test_schema.py
tests/karn/overwatch/test_replay.py
tests/karn/overwatch/fixtures/              # Sample JSONL files
```

### Acceptance Criteria
- [ ] `TuiSnapshot` round-trips through JSON without data loss
- [ ] `SnapshotWriter` creates valid JSONL files
- [ ] `SnapshotReader` yields `TuiSnapshot` objects
- [ ] Filter function works on reader
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 1: Minimal Textual App Shell

### Goal
Get a Textual app that launches, shows placeholder layout, and exits cleanly. Proves the Textual integration works.

### Scope

**In Scope:**
- `OverwatchApp(App)` class with `compose()` method
- Basic CSS file with layout regions
- Placeholder widgets for each region (Static with region name)
- Keyboard bindings: `q` (quit), `?` (help overlay)
- Help overlay with placeholder content
- CLI entry point: `esper overwatch --replay FILE`
- Loads first snapshot from JSONL file
- Displays snapshot timestamp to prove data flows

**Out of Scope:**
- Real widget rendering (Stage 2-5)
- Navigation within regions (Stage 2)
- Live telemetry (Stage 6)

### Files to Create
```
src/esper/karn/overwatch/app.py             # OverwatchApp
src/esper/karn/overwatch/styles.tcss        # Textual CSS
src/esper/karn/overwatch/widgets/__init__.py
src/esper/karn/overwatch/widgets/help.py    # HelpOverlay
src/esper/scripts/overwatch.py              # CLI entry point
tests/karn/overwatch/test_app_shell.py
```

### Acceptance Criteria
- [ ] `uv run python -m esper.scripts.overwatch --replay fixtures/sample.jsonl` launches TUI
- [ ] Layout shows 5 regions (header, tamiyo strip, flight board, detail panel, feed)
- [ ] `q` exits cleanly
- [ ] `?` shows help overlay, `Esc` dismisses
- [ ] Snapshot timestamp displayed somewhere
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 2: Flight Board (Seeds/Hosts Telemetry)

### Goal
Render the main Flight Board with environment rows, slot chips, and navigation. This is the primary UI surface.

### Scope

**In Scope:**
- `FlightBoard` widget (scrollable container)
- `EnvRow` widget (single environment with slots)
- `SlotChip` rendering (stage, alpha bar, gate status)
- Slot position display as `[r0c1]` tuples
- Anomaly sorting with hysteresis (3↑ / 5↓)
- Status colors (OK=green, WARN=yellow, CRIT=red)
- Progress bars for alpha values
- Navigation: `j`/`k` or arrows to select env
- `Enter` to expand env (shows all slots)
- `Esc` to collapse
- Focus indicator (reverse video on selected row)

**Out of Scope:**
- Header content (Stage 3)
- Tamiyo Strip (Stage 3)
- Detail Panel content (Stage 4)
- Event Feed (Stage 5)

### Files to Create/Modify
```
src/esper/karn/overwatch/widgets/flight_board.py
src/esper/karn/overwatch/widgets/env_row.py
src/esper/karn/overwatch/widgets/slot_chip.py
src/esper/karn/overwatch/display_state.py   # Hysteresis logic
src/esper/karn/overwatch/app.py             # Wire FlightBoard into compose()
src/esper/karn/overwatch/styles.tcss        # Stage colors, status colors
tests/karn/overwatch/test_flight_board.py
tests/karn/overwatch/test_hysteresis.py
```

### Acceptance Criteria
- [ ] Flight Board renders all envs from snapshot
- [ ] Envs sorted by anomaly score (highest first)
- [ ] Slot chips show `[r0c1]` position, stage name, alpha bar
- [ ] `j`/`k` navigation works with visible focus
- [ ] `Enter` expands to show all slots, `Esc` collapses
- [ ] Hysteresis prevents sort jitter (test with edge cases)
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 3: Header + Tamiyo Strip

### Goal
Add the Header (infrastructure telemetry) and Tamiyo Strip (agent brain telemetry), establishing the clear domain separation.

### Scope

**In Scope:**
- `Header` widget with:
  - Run identity (task name, episode, batch, best metric, runtime)
  - Connection indicator (`● Live` / `○ Stale` / `✕ Disconnected`)
  - Dual health indicators: `[✓ ENVS OK]` and `[↓ EV LOW]`
  - Resource vitals (GPU util, memory, temp)
- `TamiyoStrip` widget (2 lines, always visible):
  - Line 1: PPO vitals with trend arrows (KL, Entropy, Clip, EV, Grad norm, LR)
  - Line 2: Action mix, recent actions `[G][B][W]`, confidence %, exploration %
- Trend arrow logic: `↓↓` for >2σ decline, `↓` for >1σ, etc.
- Color coding: healthy=green, warning=yellow, critical=red, Tamiyo=magenta

**Out of Scope:**
- Detail Panel content (Stage 4)
- Tamiyo Detail Panel (Stage 4)
- Event Feed (Stage 5)

### Files to Create/Modify
```
src/esper/karn/overwatch/widgets/header.py
src/esper/karn/overwatch/widgets/tamiyo_strip.py
src/esper/karn/overwatch/widgets/health_indicator.py
src/esper/karn/overwatch/trends.py          # Trend calculation logic
src/esper/karn/overwatch/app.py             # Wire Header + TamiyoStrip
src/esper/karn/overwatch/styles.tcss        # Tamiyo magenta theme
tests/karn/overwatch/test_header.py
tests/karn/overwatch/test_tamiyo_strip.py
tests/karn/overwatch/test_trends.py
```

### Acceptance Criteria
- [ ] Header shows run identity and connection status
- [ ] Dual health indicators visible and update from snapshot
- [ ] Resource vitals display GPU/CPU/RAM
- [ ] Tamiyo Strip shows PPO vitals with trend arrows
- [ ] Action mix and recent actions render correctly
- [ ] Clear visual separation between Header and Tamiyo Strip
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 4: Detail Panels (Context + Tamiyo)

### Goal
Implement the toggleable detail panels: Context Panel ("why flagged") and Tamiyo Detail Panel (full agent diagnostics).

### Scope

**In Scope:**
- `DetailPanel` container (switches content based on mode)
- `ContextPanel` widget:
  - "Why flagged" bullet list from anomaly reasons
  - Selected slot details (stage, blueprint, alpha, gate history)
  - Reward breakdown for last action
- `TamiyoDetailPanel` widget:
  - Full action distribution bars
  - Recent actions grid (last 10-20)
  - Confidence sparkline with min/max
  - Exploration bar (entropy as % of max)
  - Learning signals (KL, EV, Clip with status)
- Panel toggles: `t` for Tamiyo, `c` for Context
- Empty states for each panel (warmup, no selection, etc.)
- Progressive disclosure: compact by default, expand sections

**Out of Scope:**
- Event Feed (Stage 5)
- PyTorch/RL deep diagnostics (future enhancement)

### Files to Create/Modify
```
src/esper/karn/overwatch/widgets/detail_panel.py
src/esper/karn/overwatch/widgets/context_panel.py
src/esper/karn/overwatch/widgets/tamiyo_detail.py
src/esper/karn/overwatch/widgets/empty_states.py
src/esper/karn/overwatch/app.py             # Wire panel toggles
src/esper/karn/overwatch/styles.tcss        # Panel styling
tests/karn/overwatch/test_context_panel.py
tests/karn/overwatch/test_tamiyo_detail.py
tests/karn/overwatch/test_empty_states.py
```

### Acceptance Criteria
- [ ] `c` key shows Context Panel with "why flagged"
- [ ] `t` key shows Tamiyo Detail Panel
- [ ] Panels toggle (pressing same key hides)
- [ ] Context shows selected env's anomaly reasons
- [ ] Tamiyo shows full action distribution and confidence
- [ ] Empty states display when no data available
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 5: Event Feed + Replay Polish

### Goal
Add the Event Feed widget and polish the replay experience with playback controls.

### Scope

**In Scope:**
- `EventFeed` widget (scrollable log)
- Event type badges: `[GATE]`, `[STAGE]`, `[PPO]`, `[GERM]`, `[CULL]`, etc.
- Event filtering by type
- `f` key to toggle feed size (compact/expanded)
- `Shift+j`/`Shift+k` to scroll feed when focused
- Replay controls:
  - `Space` to play/pause
  - `.` to step forward one snapshot
  - `,` to step backward
  - `<`/`>` to change playback speed
- Replay header: `[▶ REPLAY 2x] 12:00:03 / 12:15:47 [████░░░░░░] 40%`
- Filter bar: `/` to open, type to filter, `Esc` to clear

**Out of Scope:**
- Live telemetry (Stage 6)
- Export functionality (future enhancement)

### Files to Create/Modify
```
src/esper/karn/overwatch/widgets/event_feed.py
src/esper/karn/overwatch/widgets/filter_bar.py
src/esper/karn/overwatch/replay_controller.py   # Playback state machine
src/esper/karn/overwatch/app.py                 # Wire replay controls
src/esper/karn/overwatch/styles.tcss            # Feed styling
tests/karn/overwatch/test_event_feed.py
tests/karn/overwatch/test_replay_controller.py
```

### Acceptance Criteria
- [ ] Event Feed renders events from snapshot
- [ ] Event type badges are color-coded
- [ ] `f` toggles feed between compact (4 lines) and expanded (8 lines)
- [ ] `Space` pauses/resumes replay
- [ ] `.` and `,` step through snapshots
- [ ] `<`/`>` change playback speed (0.5x, 1x, 2x, 4x)
- [ ] Replay progress bar shows position
- [ ] Filter bar filters events by type/env
- [ ] All tests pass
- [ ] Can be merged to main independently

---

## Stage 6: Live Telemetry Integration

### Goal
Wire up live telemetry from training to the UI, completing the full monitoring experience.

### Scope

**In Scope:**
- `TelemetryAggregator` class:
  - Accepts telemetry events from hub
  - Maintains rolling state for all metrics
  - Builds `TuiSnapshot` on demand
  - Throttles: 1Hz for slow path, 2Hz for fast path
- `TelemetryListener` — registers with telemetry hub, forwards to aggregator
- Anomaly scoring function with weighted factors
- Update loop in app: `set_interval()` to poll aggregator
- Reactive data flow: snapshot changes trigger widget updates
- CLI: `--overwatch` flag on `esper ppo` command
- CLI: `esper overwatch` for standalone viewer (live mode default)
- Connection status based on time since last event
- Graceful handling of telemetry gaps

**Out of Scope:**
- Advanced PyTorch diagnostics (future telemetry work)
- Distributed training support (future)

### Files to Create/Modify
```
src/esper/karn/overwatch/aggregator.py
src/esper/karn/overwatch/listener.py
src/esper/karn/overwatch/anomaly.py         # score_anomaly() function
src/esper/karn/overwatch/app.py             # Update loop, reactive binding
src/esper/scripts/overwatch.py              # Live mode default
src/esper/scripts/train.py                  # Add --overwatch flag
tests/karn/overwatch/test_aggregator.py
tests/karn/overwatch/test_listener.py
tests/karn/overwatch/test_anomaly.py
tests/karn/overwatch/test_integration.py    # End-to-end with mock telemetry
```

### Acceptance Criteria
- [ ] `esper overwatch` launches in live mode (waiting for telemetry)
- [ ] `esper ppo --overwatch` launches training with TUI
- [ ] Telemetry events flow to aggregator
- [ ] Aggregator builds valid snapshots
- [ ] UI updates at appropriate cadence (no flicker, no lag)
- [ ] Connection status reflects actual telemetry flow
- [ ] Anomaly scores computed correctly
- [ ] All tests pass
- [ ] Full integration test with mock training
- [ ] Can be merged to main independently

---

## Implementation Notes

### Textual Patterns to Use

```python
# Reactive properties for data binding
class FlightBoard(Container):
    snapshot = reactive(None, recompose=True)

    def compose(self) -> ComposeResult:
        if not self.snapshot:
            yield EmptyState("waiting")
            return
        for env in self.snapshot.flight_board:
            yield EnvRow(env)

# Update loop for live data
def on_mount(self) -> None:
    self.set_interval(0.5, self._refresh)  # 2Hz

async def _refresh(self) -> None:
    self.query_one(FlightBoard).snapshot = self.aggregator.build_snapshot()
```

### Test Strategy

Each stage should have:
1. **Unit tests** for individual components
2. **Snapshot tests** for widget rendering (using Textual's snapshot testing)
3. **Integration tests** for data flow

### Fixture Files Needed (Stage 0)

| File | Contents |
|------|----------|
| `healthy_run.jsonl` | 4 envs, all OK, varied stages |
| `anomaly_detected.jsonl` | 1 env WARN, 1 CRIT, anomaly reasons |
| `tamiyo_active.jsonl` | Rich action distribution, confidence varies |
| `warmup_period.jsonl` | Empty Tamiyo state, warmup progress |
| `replay_sequence.jsonl` | 50+ snapshots for replay testing |

---

## Dependencies

### Python Packages (add to pyproject.toml)
```toml
[project.optional-dependencies]
tui = [
    "textual>=0.47.0",
]
```

### Existing Esper Modules Used
- `esper.leyline.telemetry` — TelemetryEvent, TelemetryEventType
- `esper.nissa.tracker` — Existing telemetry infrastructure
- `esper.karn.health` — MemoryStats, existing health checks

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to first useful UI | Stage 2 complete (~2-3 sessions) |
| Full replay experience | Stage 5 complete (~5-6 sessions) |
| Live monitoring | Stage 6 complete (~6-7 sessions) |
| Test coverage | >80% for new code |
| Render performance | <50ms for 16 envs |

---

## Next Steps

1. Expand Stage 0 into detailed implementation plan
2. Execute Stage 0, merge to main
3. Repeat for each subsequent stage

Each stage plan will be written to `docs/plans/overwatch-stage-N-*.md` before implementation.
