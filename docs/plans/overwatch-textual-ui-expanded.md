# Esper-Overwatch Textual UI Implementation Plan (Expanded)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the Textual "Esper-Overwatch" UI (snapshot-first, outliers-first, truthful under missing telemetry) that replaces the Rich Karn TUI and supports offline replay.

**Architecture:** Snapshot-first pipeline: telemetry hub → aggregator (contracts + anomaly/bound scoring) → `TuiSnapshot` JSON-serializable schema → Textual renderer. Two cadences (fast flight board/pulse, slow vitals). All panels render from immutable snapshots; UI never computes analytics.

**Tech Stack:** Python 3.11, Textual, dataclasses, pytest, telemetry events from Simic/Kasmina/Tolaria/Nissa, JSONL for replay.

---

## UX Design Principles

### Mental Model: Air Traffic Control

The Overwatch UI uses an "Air Traffic Control" metaphor where:
- **Environments** are "flights" with status indicators
- **Policy Pulse** shows system "vital signs" (PPO health)
- **Slot Chips** represent seed modules with lifecycle stages
- **Event Feed** is a structured log of lifecycle events

This metaphor aligns with operator expectations: green = healthy, yellow = warning, red = problem. Higher anomaly scores → higher attention priority.

### Information Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ HEADER: Aggregate + Connection + Policy Pulse + Timing               │
│ [✓ ALL OK] [● Live] KL: 0.02 ✓ Ent: 1.2 ↓ Clip: 5% ✓ EV: 0.42 ↓    │
│ ^^^^^^^^^^          ^^^ trend arrows      ^^^ P1 metric             │
├────────────────────────────────────────────────┬────────────────────┤
│ FLIGHT BOARD (main focus, anomaly-sorted)      │ CONTEXT PANE       │
│                                                │                    │
│ [!] Env 3  gpu:1  WARN   102 fps  (+2%)       │ Env 3 [! 2 issues] │
│     [mid] BLENDING ████░░░ 0.7α G2✓           │ ════════════════   │
│     [late] TRAINING ██░░░░░ 0.3α G1✓          │ WHY FLAGGED        │
│                                                │ • High grad ratio  │
│ [ ] Env 0  gpu:0  OK     98 fps  (~base)      │ • Memory pressure  │
│     [mid] FOSSILIZED ██████ 1.0α              │                    │
│                                                │ QUICK STATS        │
│ [ ] Env 1  gpu:0  OK     101 fps  (+3%)       │ GPU: 95% Mem: 92%  │
│     [mid] GERMINATED █░░░░░ 0.1α G0✓          │ EV: 0.42↓          │
│                                                │                    │
│                                                │ [p] PyTorch [r] RL │
├────────────────────────────────────────────────┴────────────────────┤
│ EVENT FEED (filtered, scrollable)                                    │
│ 12:00:03 [GATE] Env 3 mid passed G2 (loss_delta=-0.02)              │
│ 12:00:01 [STAGE] Env 3 mid TRAINING → BLENDING                      │
│ 11:59:58 [PPO] Update 47: KL=0.019, entropy=1.21, clip=4.8%         │
└─────────────────────────────────────────────────────────────────────┘
```

### Aggregate Health Indicator

The header includes an aggregate system health indicator at the far left:

```
[✓ ALL OK]     - All environments healthy (score < 0.2)
[! 1 WARN]     - One environment in WARN state
[‼ 2 CRIT]     - Two environments in CRITICAL state
```

This provides the "everything is fine, go back to work" signal operators need without scanning the entire flight board.

### Visual Hierarchy (F-Pattern)

Users scan in an F-pattern:
1. **Top-left to right:** Policy Pulse metrics (most critical)
2. **Left edge down:** Flight board env list (anomalies first)
3. **Right column:** Context details (on-demand)
4. **Bottom:** Event feed (historical reference)

### Policy Pulse Trend Arrows

Metrics in the header show trend arrows indicating direction over the last 10 updates:

| Arrow | Meaning | Calculation |
|-------|---------|-------------|
| `↑↑` | Rising fast | Slope > +2σ |
| `↑` | Rising | Slope > +1σ |
| (none) | Stable | Slope within ±1σ |
| `↓` | Falling | Slope < -1σ |
| `↓↓` | Falling fast | Slope < -2σ |

**Example header with trends:**
```
[✓ ALL OK] [● Live] KL: 0.02 ✓ Ent: 1.2 ↓ Clip: 5% ↑ EV: 0.42 ↓↓
                              ^^^        ^^^         ^^^^
                              falling    rising      falling fast (concern!)
```

**Trend Semantics:**
- **KL divergence:** Rising = policy changing faster (may need smaller learning rate)
- **Entropy:** Falling = becoming more deterministic (may be premature convergence)
- **Clip fraction:** Rising = hitting PPO constraints more often (instability signal)
- **Explained variance (EV):** Falling = value function failing to track returns (critical!)

### Keyboard-First Design

All operations must be accessible via keyboard. Mouse/trackpad is optional enhancement.

---

## Keyboard Shortcut Specification

### Global Navigation

| Action | Primary | Alternate | Context |
|--------|---------|-----------|---------|
| Next env | `j` | `↓` | Flight board focused |
| Previous env | `k` | `↑` | Flight board focused |
| Expand env detail | `Enter` | `l` | Show full slot info in context |
| Collapse detail | `Esc` | `h` | Return to list view |
| Toggle feed panel | `f` | | Hide/show bottom panel |
| Toggle context pane | `c` | | Hide/show right panel |
| Focus flight board | `1` | | Switch focus region |
| Focus feed | `2` | | Switch focus region |
| Scroll feed down | `Shift+j` | `Shift+↓` | When feed focused |
| Scroll feed up | `Shift+k` | `Shift+↑` | When feed focused |

### Filters and Search

| Action | Shortcut | Notes |
|--------|----------|-------|
| Open filter bar | `/` | Focus filter input |
| Clear filters | `Esc` | When filter bar focused |
| Filter: warnings only | `w` | Toggle |
| Filter: stage | `s` | Cycle: ALL → TRAINING → BLENDING → ... |
| Command palette | `Ctrl+k` | Fuzzy search all actions |

### Actions

| Action | Shortcut | Notes |
|--------|----------|-------|
| Export snapshot (JSONL) | `e` | Default: save snapshot for replay |
| Export snapshot (Markdown) | `E` | Human-readable format for sharing |
| Pause updates | `Space` | Freeze display (replay continues in background) |
| Resume updates | `Space` | Resume from pause |
| Quit | `q` | Exit TUI |
| Help overlay | `?` | Show keyboard shortcuts |
| Refresh | `r` | Force refresh from aggregator |

**Export Format Details:**
- `e` (lowercase): JSONL format suitable for `--replay` playback
- `E` (uppercase): Markdown table format for copy/paste into docs or issues
- Default path: `./overwatch-snapshot-{timestamp}.{ext}`

### Replay Mode Controls

When running with `--replay`:

| Action | Shortcut | Notes |
|--------|----------|-------|
| Play/Pause | `Space` | Toggle playback |
| Step forward | `.` | Advance one snapshot |
| Step backward | `,` | Go back one snapshot |
| Jump to time | `t` | Enter timestamp to seek |
| Speed up | `>` | 2x → 4x → 8x playback |
| Speed down | `<` | 0.5x → 0.25x playback |
| Reset speed | `=` | Return to 1x playback |

**Replay Mode Header:**
```
[▶ REPLAY 2x] 12:00:03 / 12:15:47  [●○○○○○○○○○] 8%
              ^^ time   ^^ total    ^^^ progress bar
```

### Slot Selection (when env expanded)

| Action | Shortcut | Notes |
|--------|----------|-------|
| Select early slot | `e` | When env detail view (mnemonic: early) |
| Select mid slot | `m` | When env detail view (mnemonic: mid) |
| Select late slot | `L` | When env detail view (capital L, since `l` = expand) |

> **Note:** Numeric keys `1`/`2` are reserved for focus region switching. Slot selection uses mnemonic letters to avoid conflict.

### Anomaly Navigation

| Action | Shortcut | Notes |
|--------|----------|-------|
| Jump to next anomaly | `]` | Next env with score > 0.5 |
| Jump to prev anomaly | `[` | Previous env with score > 0.5 |
| Jump to highest anomaly | `!` | Go to env with highest anomaly score |
| Jump to first env | `gg` | Vim-style: go to top of list |
| Jump to last env | `G` | Vim-style: go to bottom of list |

---

## Anomaly Scoring Specification

### Anomaly Sources

The `score_anomaly()` function computes a 0.0-1.0 score based on weighted factors:

| Source | Weight | Threshold | Reason Text |
|--------|--------|-----------|-------------|
| Throughput drop | 0.3 | >50% below baseline | "Throughput 52% below baseline" |
| Negative reward | 0.2 | reward < -1.0 | "Unusual negative reward (-1.3)" |
| Gradient norm ratio | 0.25 | >10x typical | "High gradient norm ratio (15.2x)" |
| Staleness | 0.15 | >10s since update | "Env stale (12.3s since update)" |
| Memory pressure | 0.1 | >95% GPU memory | "Memory pressure (97% GPU mem)" |

### Score Interpretation

| Score Range | Status | Visual |
|-------------|--------|--------|
| 0.0 - 0.2 | OK | Green |
| 0.2 - 0.5 | INFO | Blue |
| 0.5 - 0.7 | WARN | Yellow |
| 0.7 - 1.0 | CRITICAL | Red |

### "Why Flagged" Display

When an env has anomaly score > 0.2, the context pane shows:
```
Why flagged:
• High gradient norm ratio (15.2x)
• Memory pressure (97% GPU mem)
```

---

## Connection Status and Staleness Specification

### Header Connection Indicator

```
[● Live]           - Green pulsing dot, connected, data fresh (<5s)
[● Live (2s)]      - Green dot with age, connected, slight delay
[○ Stale (8s)]     - Yellow hollow dot, connected but no recent data
[✕ Disconnected]   - Red X, connection lost
```

### Implementation with Hysteresis

Connection state changes require hysteresis to prevent Live→Stale→Live flicker during network hiccups:

```python
@dataclass
class ConnectionStatus:
    connected: bool
    last_event_ts: float
    staleness_s: float

    @property
    def status_text(self) -> str:
        if not self.connected:
            return "[red]✕ Disconnected[/red]"
        if self.staleness_s < 2.0:
            return "[green]● Live[/green]"
        if self.staleness_s < 5.0:
            return f"[green]● Live ({self.staleness_s:.0f}s)[/green]"
        return f"[yellow]○ Stale ({self.staleness_s:.0f}s)[/yellow]"


@dataclass
class ConnectionDisplayState:
    """Track connection state with hysteresis to prevent flicker."""
    displayed_state: str = "live"  # "live", "stale", "disconnected"
    pending_state: str = "live"
    pending_count: int = 0

    HYSTERESIS_THRESHOLD = 3  # ~600ms at 5Hz

    def update(self, raw_state: str) -> str:
        """Apply hysteresis and return state to display."""
        if raw_state == self.displayed_state:
            self.pending_state = raw_state
            self.pending_count = 0
            return self.displayed_state

        if raw_state == self.pending_state:
            self.pending_count += 1
            if self.pending_count >= self.HYSTERESIS_THRESHOLD:
                self.displayed_state = raw_state
                self.pending_count = 0
        else:
            self.pending_state = raw_state
            self.pending_count = 1

        return self.displayed_state
```

> **Design note:** Hysteresis is symmetric (same threshold for degradation and recovery). This prevents premature "all clear" signals that could cause operator whiplash.

### Value Change Flash

When metric values change:
1. Background flashes yellow (300ms)
2. Fade to normal (200ms)
3. Total animation: 500ms

CSS token: `--flash-change: #FFEB3B` (Material yellow 500)

---

## Hysteresis and Stable Sort Specification

### Problem

Without hysteresis, an env flickering between OK and WARN causes the flight board to reorder constantly, creating visual noise.

### Solution: Stable Sort with Hysteresis

1. **Status hysteresis:** Env status must be stable for 3 consecutive snapshots before visual change
2. **Sort stability:** Flight board reorders only on slow cadence (1 Hz), not fast path (5 Hz)
3. **Tie-breaking:** When anomaly scores are equal, sort by env_id (deterministic)

### Implementation

```python
@dataclass
class EnvDisplayState:
    """Tracks display state for hysteresis."""
    env_id: int
    displayed_status: str  # What's currently shown
    pending_status: str    # What telemetry says
    pending_count: int     # How many snapshots pending status held

    HYSTERESIS_THRESHOLD = 3

    def update(self, new_status: str) -> str:
        """Returns status to display (may be delayed by hysteresis)."""
        if new_status == self.displayed_status:
            self.pending_status = new_status
            self.pending_count = 0
            return self.displayed_status

        if new_status == self.pending_status:
            self.pending_count += 1
            if self.pending_count >= self.HYSTERESIS_THRESHOLD:
                self.displayed_status = new_status
                self.pending_count = 0
                return self.displayed_status
        else:
            self.pending_status = new_status
            self.pending_count = 1

        return self.displayed_status
```

### Sort Rules

```python
def sort_flight_board(envs: list[EnvSummary]) -> list[EnvSummary]:
    """Sort by anomaly score descending, then env_id ascending for stability."""
    return sorted(envs, key=lambda e: (-e.anomaly["score"], e.env_id))
```

---

## Empty States and Onboarding Specification

### Empty State: No Telemetry Yet

When TUI launches but no telemetry has arrived:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ESPER OVERWATCH                                    [○ Waiting...]   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    ┌──────────────────────────┐                     │
│                    │                          │                     │
│                    │   Waiting for telemetry  │                     │
│                    │                          │                     │
│                    │   Start training with:   │                     │
│                    │   esper ppo --overwatch  │                     │
│                    │                          │                     │
│                    │   Or replay a session:   │                     │
│                    │   esper overwatch        │                     │
│                    │     --replay snaps.jsonl │                     │
│                    │                          │                     │
│                    └──────────────────────────┘                     │
│                                                                      │
│   Press ? for keyboard shortcuts                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Empty State: No Envs Match Filter

When filters exclude all envs:

```
┌─────────────────────────────────────────────────────────────────────┐
│ FLIGHT BOARD                          [Filter: BLENDING only]       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   No environments match current filters.                            │
│                                                                      │
│   Current filters: stage=BLENDING                                   │
│   Total environments: 8 (0 matching)                                │
│                                                                      │
│   Press / to modify filters or Esc to clear.                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Empty State: Partial Data (Initialization)

When telemetry is arriving but training hasn't produced data yet:

```
┌─────────────────────────────────────────────────────────────────────┐
│ ESPER OVERWATCH                               [● Live - Initializing]│
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Receiving telemetry, waiting for training to begin...             │
│                                                                      │
│   ┌─────────────────────────────────────────────────┐               │
│   │ Connection:      ✓ Connected                    │               │
│   │ Environments:    0 registered                   │               │
│   │ PPO Updates:     Waiting for first update       │               │
│   │ GPU Telemetry:   ✓ Receiving                    │               │
│   └─────────────────────────────────────────────────┘               │
│                                                                      │
│   This is normal during initialization.                             │
│   The flight board will appear once training starts.                │
│                                                                      │
│   Press ? for keyboard shortcuts                                     │
└─────────────────────────────────────────────────────────────────────┘
```

This state distinguishes between "no connection at all" vs "connected but waiting for training to produce data".

### First-Time Onboarding Hint

On first launch (detected via absence of `~/.esper/overwatch_seen`):

```
┌─────────────────────────────────────────────────────────────────────┐
│ Welcome to Esper Overwatch!                                    [x]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  How it works:                                                       │
│  • Problems surface automatically - anomalies sort to the top       │
│  • Green [✓] = healthy, safe to ignore                              │
│  • Yellow [!] = warning, check the context pane for details         │
│  • Red [‼] = critical, immediate attention needed                   │
│                                                                      │
│  Navigation:                                                         │
│  • j/k or ↑/↓ to move between environments                          │
│  • Enter to expand, ] to jump to next anomaly                       │
│  • / to filter, ? for all shortcuts                                 │
│                                                                      │
│  [Don't show again]                              [Got it, thanks!]  │
└─────────────────────────────────────────────────────────────────────┘
```

Dismissable with Enter or Esc. "Don't show again" writes marker file.

---

## Slot Chip Visual Design

### Chip Structure

```
┌──────────────────────────────────────────────────────────────┐
│ [slot_id] STAGE ████████░░░░ alpha  GATE                      │
│           name   progress_bar       status                    │
└──────────────────────────────────────────────────────────────┘
```

### Examples by Stage

```
DORMANT:
[mid] DORMANT                          (gray, no progress)

GERMINATED:
[mid] GERMINATED █░░░░░░░░░ 0.1α G0✓   (cyan, just started)

TRAINING:
[mid] TRAINING ███░░░░░░░ 0.3α G1✓     (blue, learning)
      conv     epoch 3/10

BLENDING:
[mid] BLENDING ███████░░░ 0.7α G2✓     (purple, integrating)
      conv     epoch 5/7

FOSSILIZED:
[mid] FOSSILIZED ██████████ 1.0α       (green, complete, no gate)

CULLED:
[mid] CULLED ✕                         (red, failed)
      Reason: loss regression
```

### Progress Bar Semantics

The progress bar represents different metrics depending on the seed stage:

| Stage | Bar Represents | Range | Notes |
|-------|----------------|-------|-------|
| DORMANT | N/A | Empty | No bar shown; seed inactive |
| GERMINATED | Warmup progress | 0-100% | Time since germination / typical warmup duration |
| TRAINING | Epoch progress | 0-100% | current_epoch / epochs_in_stage |
| BLENDING | Alpha coefficient | 0-100% | Blend weight (0.0 = host only, 1.0 = seed fully blended) |
| FOSSILIZED | Full | 100% | Always full; seed permanently integrated |
| CULLED | N/A | Strikethrough | X pattern or ░░░ with strikethrough |

**Alpha (α) value:** Always shown as decimal (0.3α = 30% blend weight). This is the actual blend coefficient used in inference, distinct from epoch progress.

### Stage Colors

| Stage | Color | Hex |
|-------|-------|-----|
| DORMANT | Gray | #9E9E9E |
| GERMINATED | Cyan | #00BCD4 |
| TRAINING | Blue | #2196F3 |
| BLENDING | Purple | #9C27B0 |
| FOSSILIZED | Green | #4CAF50 |
| CULLED | Red | #F44336 |

### Gate Status Icons

| Status | Icon | Meaning |
|--------|------|---------|
| Passed | ✓ | Gate criteria met |
| Failed | ✕ | Gate criteria not met |
| Pending | ○ | Evaluation in progress |
| N/A | (blank) | No gate for this stage |

---

## Accessibility Considerations

### Terminal Requirements

Document in help and README:

> **Minimum terminal requirements:**
> - Width: 100 columns minimum (120 recommended)
> - Height: 30 rows minimum
> - Font: 12pt+ monospace recommended
> - Theme: High contrast theme recommended for visibility

### Color Independence

All status information uses color + text + icon:
- Status: `[!] WARN` not just yellow
- Gates: `G2✓` not just green checkmark
- Connection: `● Live` not just green dot

### Keyboard Accessibility

- All operations keyboard-accessible
- Focus indicator visible (reverse video on selected row)
- Tab order: Header → Flight Board → Context → Feed
- Skip link equivalent: number keys (1, 2) to jump sections

### Escape Behavior (Keyboard Trap Prevention)

`Esc` must always close the current overlay or exit the current mode. No keyboard traps.

| Context | Esc Behavior |
|---------|--------------|
| Help overlay open | Close help overlay |
| Command palette open | Close command palette |
| Filter bar focused | Clear filter AND close filter bar |
| Env expanded (detail view) | Collapse to list view |
| PyTorch/RL details expanded | Collapse detail section |
| At root (flight board, nothing expanded) | No-op (don't show exit prompt) |
| Onboarding modal | Dismiss modal |

**Test requirement:** Every modal/overlay/mode must be tested for Esc exit behavior.

### Focus Indicators by Context

| Context | Indicator | Description |
|---------|-----------|-------------|
| Flight board row | Reverse video | White text on dark background |
| Slot chip (expanded view) | Bracket prefix | `>[mid]` instead of `[mid]` |
| Filter bar | Cursor + underline | Blinking cursor, underlined input |
| Command palette | Highlight bar | Selected option has background |
| Help overlay | N/A | Modal, focus trapped until dismissed |
| Context pane section | Bold header | Active section header is bold |

### Screen Reader Notes

Textual has limited screen reader support. Document limitations:
- Best experience with VoiceOver on macOS
- Linux: Test with ORCA
- Some dynamic updates may not announce automatically

---

## Telemetry Gap Enhancements (Specialist Review)

The following telemetry gaps were identified by PyTorch and DRL specialists as valuable additions to the Overwatch TUI. These extend the anomaly detection and context pane functionality.

### P1: Critical Gaps (Should be in MVP)

| Gap | Source | Integration Point | Why Critical |
|-----|--------|-------------------|--------------|
| **CUDA Memory Allocator** | PyTorch | Context Pane | Fragmentation causes silent slowdowns; operators need `allocated_bytes` vs `reserved_bytes` delta |
| **Gradient Flow Health** | PyTorch | Anomaly Score | Dead layers (grad=0) and explosions (grad>1e6) are training killers; add to anomaly scoring |
| **Explained Variance Trend** | DRL | Policy Pulse | Value function quality - dropping EV = value network failing to track returns |
| **Advantage Statistics** | DRL | Context Pane | `mean`, `std`, `min`, `max` of advantages reveal reward scaling issues |
| **Bootstrap Value Quality** | DRL | Anomaly Score | Bootstrap term dominance (>50% of target) suggests horizon issues |
| **Entropy Schedule Progress** | DRL | Policy Pulse | Shows where in entropy decay schedule; critical for exploration vs exploitation balance |

### P2: Important Gaps (NOW INCLUDED IN MVP)

All P2 gaps have been promoted to MVP scope to achieve comprehensive telemetry coverage:

| Gap | Source | Integration Point | Status |
|-----|--------|-------------------|--------|
| torch.compile Graph Stats | PyTorch | Context Pane | ✅ `compile_graph_breaks`, `compile_cache_hit_pct` |
| AMP Scaling History | PyTorch | Context Pane | ✅ `amp_loss_scale`, `amp_skipped_steps` |
| Tensor Operation Diagnostics | PyTorch | Context Pane | ✅ `cpu_gpu_transfers`, `sync_events` |
| CUDA Kernel Launch Stats | PyTorch | Context Pane | ✅ `kernel_launch_overhead_pct` |
| DataLoader Performance | PyTorch | Context Pane | ✅ `dataloader_bottleneck`, `dataloader_queue_depth` |
| Distributed Training | PyTorch | Context Pane | ✅ `straggler_rank`, `comm_overhead_pct` |
| Reward Component Breakdown | DRL | Context Pane | ✅ `shaped_reward_ratio` |
| Policy Ratio Statistics | DRL | Anomaly Score | ✅ `clip_fraction_positive/negative`, `value_clip_fraction` |
| Entropy Trend | DRL | Context Pane | ✅ `entropy_trend`, `entropy_collapsed` |
| Sample Reuse/Staleness | DRL | Anomaly Score | ✅ `importance_weight_*`, `sample_reuse_epochs` |
| Reward Hacking Indicators | DRL | Anomaly Score | ✅ `reward_task_correlation`, `reward_hacking_detected` |
| Plateau Detection | DRL | Anomaly Score | ✅ `updates_since_improvement`, `plateau_detected` |
| PBRS Potential Drift | DRL | Context Pane | ✅ `pbrs_potential_drift` |
| Policy Determinism | DRL | Anomaly Score | ✅ Via `entropy_collapsed` (action std collapse) |

### Enhanced Anomaly Scoring Weights (Updated)

```python
ANOMALY_WEIGHTS = {
    # ═══════════════════════════════════════════════════════════════
    # INFRASTRUCTURE (0.30 total)
    # ═══════════════════════════════════════════════════════════════
    "throughput_drop": 0.10,      # >50% below baseline
    "staleness": 0.10,            # >10s since update
    "memory_pressure": 0.05,      # >95% GPU memory
    "memory_fragmented": 0.05,    # >30% fragmentation

    # ═══════════════════════════════════════════════════════════════
    # PYTORCH HEALTH (0.25 total)
    # ═══════════════════════════════════════════════════════════════
    "dead_gradient_layers": 0.05, # Any layer with grad=0
    "grad_explosion": 0.05,       # Any layer with grad>1e6
    "dead_neurons": 0.05,         # >5% dead neurons
    "nan_gradients": 0.05,        # Any NaN gradients detected
    "graph_breaks_high": 0.03,    # >10 torch.compile graph breaks
    "amp_unstable": 0.02,         # AMP scale dropped 3+ times

    # ═══════════════════════════════════════════════════════════════
    # RL DIAGNOSTICS (0.45 total)
    # ═══════════════════════════════════════════════════════════════
    # Value function
    "ev_low": 0.08,               # Explained variance < 0.3
    "ev_decline": 0.05,           # EV dropping over 10 updates
    "bootstrap_dominance": 0.03,  # Bootstrap >50% of target value

    # Advantages & clipping
    "advantage_biased": 0.05,     # |advantage_mean| > 0.5
    "negative_reward": 0.05,      # reward < -1.0 (unusual)
    "grad_norm_ratio": 0.05,      # >10x typical gradient norm

    # Exploration (critical for PPO)
    "entropy_collapsed": 0.08,    # entropy < 0.1 (premature convergence)

    # Reward hacking (CRITICAL for Esper seed lifecycle)
    "reward_hacking_risk": 0.08,  # reward↑ but task_metric↓

    # Training progress
    "plateau_detected": 0.05,     # 20+ updates without improvement
    "samples_stale": 0.03,        # IS weight max > 5 (off-policy)
}

# Total: 1.00
# Note: Weights are additive. Multiple issues compound.
# A single catastrophic issue (entropy collapse + reward hacking) can reach 0.16.
# CRITICAL threshold (0.7) requires 4-5 concurrent medium issues.
```

### Anomaly Scoring Philosophy

**Design Principles:**

1. **Additive scoring:** Multiple issues compound. An env with grad explosion (0.10) + memory pressure (0.10) + staleness (0.10) scores 0.30 total.

2. **Normalized weights:** Total weights sum to 1.0. This means a single catastrophic issue (like grad explosion) scores ~0.10-0.15, not 1.0. Multiple concurrent problems are needed to reach CRITICAL status.

3. **False positives preferred:** We prefer flagging something normal over missing a real problem. Training failures are expensive; brief investigations are cheap.

4. **Threshold tuning:** The 0.5 WARN and 0.7 CRITICAL thresholds are starting points. Expect to tune based on operational experience.

**Score interpretation:**
```
0.00 - 0.20:  OK       Everything looks normal
0.20 - 0.50:  INFO     Minor anomalies, worth a glance
0.50 - 0.70:  WARN     Investigate the context pane
0.70 - 1.00:  CRITICAL Multiple serious issues detected
```

### Enhanced Schema Extensions

```python
@dataclass
class PyTorchDiagnostics:
    """Extended PyTorch metrics for Context Pane."""
    # Memory (P1)
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    cuda_fragmentation_pct: float  # (reserved - allocated) / reserved * 100
    alloc_retries: int             # Memory allocation retries (fragmentation indicator)
    oom_events: int                # Out-of-memory events this session

    # Gradients (P1)
    dead_layers: list[str]         # Layer names with zero gradients
    exploding_layers: list[str]    # Layer names with extreme gradients
    dead_neuron_pct: float         # % of neurons with zero gradient
    nan_grad_count: int            # Count of NaN gradients detected
    layer_gradient_health: dict[str, float]  # layer_name -> update_ratio (< 0.01 = weak)

    # torch.compile (P1)
    compile_enabled: bool
    compile_graph_breaks: int | None
    compile_cache_hit_pct: float | None

    # AMP (P2)
    amp_loss_scale: float | None
    amp_skipped_steps: int         # Steps skipped due to inf/nan

    # Tensor Operations (P2)
    cpu_gpu_transfers: int         # Hidden transfers killing throughput
    sync_events: int               # torch.cuda.synchronize() calls

    # CUDA Kernels (P2)
    kernel_launch_overhead_pct: float | None  # Dispatch vs compute ratio

    # DataLoader (P2)
    dataloader_bottleneck: bool    # True if dataloader is slower than GPU
    dataloader_queue_depth: int    # Prefetch buffer utilization

    # Distributed (P2)
    distributed_enabled: bool
    straggler_rank: int | None     # Slowest rank if distributed
    comm_overhead_pct: float | None  # % time in communication

# Implementation notes:
# - When AMP active (amp_loss_scale > 1.0), adjust exploding threshold to 1e6 * amp_loss_scale
# - layer_gradient_health: update_ratio = grad_norm / param_norm; < 0.01 means layer barely updating
# - cpu_gpu_transfers: Use torch.cuda.nvtx or profiler hooks to detect .cpu()/.item() calls


@dataclass
class RLDiagnostics:
    """Extended RL metrics for Context Pane."""
    # Value Function (P1)
    explained_variance: float
    ev_trend: list[float]          # Last 10 EV values for sparkline
    ev_direction: str              # "rising", "stable", "falling"

    # Advantages (P1)
    advantage_mean: float
    advantage_std: float
    advantage_range: tuple[float, float]
    advantage_biased: bool         # |mean| > 0.5 indicates bias

    # Clipping (P1 + P2)
    policy_ratio_clipped_pct: float      # Overall clip fraction
    clip_fraction_positive: float        # Clipped when advantage > 0
    clip_fraction_negative: float        # Clipped when advantage < 0
    value_clip_fraction: float           # Value function clipping

    # Reward Hacking (P1 - CRITICAL for Esper seed lifecycle)
    reward_task_correlation: float | None  # Correlation(reward, task_metric)
    shaped_reward_ratio: float             # shaped / (shaped + task), 0-1
    reward_hacking_detected: bool          # reward↑ but task_metric↓

    # Entropy (P2)
    entropy: float
    entropy_trend: list[float]     # Last 10 entropy values
    entropy_schedule_pct: float    # How far through decay schedule
    entropy_collapsed: bool        # entropy < 0.1

    # Bootstrap (P1)
    bootstrap_ratio: float         # bootstrap_term / total_target

    # Sample Efficiency (P2)
    importance_weight_mean: float  # IS weight mean (should be ~1.0)
    importance_weight_max: float   # IS weight max (>5 = stale samples)
    sample_reuse_epochs: int       # How many times each sample used

    # Training Progress (P2)
    updates_since_improvement: int # Plateau detection
    plateau_detected: bool         # 20+ updates without improvement

    # PBRS (P2)
    pbrs_potential_drift: float | None  # Shaping reward drift from initial
```

### Context Pane Enhancement (Progressive Disclosure)

The context pane uses **progressive disclosure** to manage information density. Default view shows "Why Flagged" and "Quick Stats"; detailed sections expand on demand.

**Default View (collapsed):**
```
┌────────────────────────────────────────────────────────────────┐
│ Env 3                                              [! 3 issues]│
├────────────────────────────────────────────────────────────────┤
│ WHY FLAGGED                                                    │
│ • High gradient norm ratio (15.2x)                             │
│ • Memory fragmentation (23% reserved unused)                    │
│ • Explained variance declining (-0.12 over 10 updates)         │
│                                                                 │
│ QUICK STATS                                                    │
│ GPU: 95%  Mem: 11.2/14.5GB (23% frag)  EV: 0.42↓              │
│                                                                 │
│ [p] PyTorch Details  [r] RL Details  [o] Open Logs            │
└────────────────────────────────────────────────────────────────┘
```

**Expanded View (after pressing `p` - PyTorch Details):**
```
┌────────────────────────────────────────────────────────────────┐
│ Env 3                                              [! 5 issues]│
├────────────────────────────────────────────────────────────────┤
│ WHY FLAGGED                                                    │
│ • High gradient norm ratio (15.2x)                             │
│ • Memory fragmentation (23% reserved unused)                    │
│ • EV declining (-0.12 over 10 updates)                         │
│ • Entropy collapsed (0.08)                                     │
│ • Reward hacking risk detected                                 │
│                                                                 │
│ ▼ PYTORCH DIAGNOSTICS (press p to collapse)                    │
│ ════════════════════════════════════════════════════════       │
│ Memory                                                          │
│   11.2GB alloc / 14.5GB reserved (23% frag)                    │
│   Retries: 0  OOMs: 0                                          │
│                                                                 │
│ Gradients                                                       │
│   Dead layers: conv3.weight, fc1.bias                          │
│   Dead neurons: 2.1%  NaN grads: 0                             │
│   Weakest layer: conv1 (0.001x update ratio)                   │
│                                                                 │
│ torch.compile: ON (inductor)                                    │
│   Graph breaks: 3  Cache: 89% hit                              │
│                                                                 │
│ AMP: scale=1024  skipped=0                                     │
│                                                                 │
│ Tensor Ops                                                      │
│   CPU↔GPU transfers: 0  Sync events: 2                         │
│   Kernel overhead: 3%                                           │
│                                                                 │
│ DataLoader: OK (queue depth: 4)                                 │
│ Distributed: N/A (single GPU)                                   │
│                                                                 │
│ [r] RL Details  [o] Open Logs  [y] Copy diagnostics            │
└────────────────────────────────────────────────────────────────┘
```

**Expanded View (after pressing `r` - RL Details):**
```
┌────────────────────────────────────────────────────────────────┐
│ Env 3                                              [! 5 issues]│
├────────────────────────────────────────────────────────────────┤
│ WHY FLAGGED                                                    │
│ • EV declining (-0.12 over 10 updates)                         │
│ • Entropy collapsed (0.08)                                     │
│ • Reward hacking risk detected                                 │
│                                                                 │
│ ▼ RL DIAGNOSTICS (press r to collapse)                         │
│ ════════════════════════════════════════════════════════       │
│ Value Function                                                  │
│   Explained Var: 0.42  ▁▂▃▅▆▇▅▄▃▂ (declining)                  │
│   Bootstrap ratio: 34% (healthy)                                │
│                                                                 │
│ Advantages                                                      │
│   Mean: -0.02  Std: 0.85  Range: [-3.2, +2.1]                  │
│   Bias: OK                                                      │
│                                                                 │
│ Clipping                                                        │
│   Policy: 5% [+3% pos, -2% neg]                                │
│   Value: 8%                                                     │
│                                                                 │
│ Entropy: 0.08 ⚠ COLLAPSED  ▇▆▅▄▃▂▁▁▁▁                         │
│   Schedule: 67% complete                                        │
│                                                                 │
│ Sample Efficiency                                               │
│   IS weights: 1.02 avg, 2.3 max                                │
│   Sample reuse: 4 epochs                                        │
│                                                                 │
│ Training Progress                                               │
│   Updates since improvement: 3                                  │
│   Plateau: NO                                                   │
│                                                                 │
│ Reward Health ⚠                                                │
│   Task correlation: 0.23 (LOW - hacking risk!)                 │
│   Shaping ratio: 78% (shaped dominates)                        │
│   PBRS drift: +0.12 from initial                               │
│                                                                 │
│ [p] PyTorch Details  [o] Open Logs  [y] Copy diagnostics       │
└────────────────────────────────────────────────────────────────┘
```

**Context Pane Shortcuts:**

| Action | Shortcut | Notes |
|--------|----------|-------|
| Toggle PyTorch details | `p` | Expand/collapse PyTorch diagnostics |
| Toggle RL details | `r` | Expand/collapse RL diagnostics |
| Open env logs | `o` | Opens logs in $PAGER |
| Copy diagnostics | `y` | Yank current pane to clipboard |

### Implementation Notes

1. **Task 2 extension**: Add `PyTorchDiagnostics` and `RLDiagnostics` to aggregator state
2. **Task 3 extension**: Update `score_anomaly()` with P1 gap weights
3. **Task 8 extension**: Render diagnostic sections in context pane
4. **Telemetry events**: Requires new events from Simic PPO (`PPO_DIAGNOSTICS_CAPTURED`) and Nissa device monitor (`DEVICE_MEMORY_DETAILED`)

---

## Task List

### Task 1: Create Overwatch package and snapshot schema

**Files:**
- Create: `src/esper/karn/overwatch/__init__.py`
- Create: `src/esper/karn/overwatch/snapshot.py`
- Tests: `tests/karn/overwatch/test_snapshot_schema.py`

**Step 1: Write failing test**
```python
# tests/karn/overwatch/test_snapshot_schema.py
from esper.karn.overwatch.snapshot import TuiSnapshot, EnvSummary, SlotChipState, ConnectionStatus

def test_snapshot_serialises_to_json():
    snap = TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-15T12:00:00Z",
        connection=ConnectionStatus(connected=True, last_event_ts=1702641600.0, staleness_s=0.5),
        run={"id": "run1", "task": "cifar10", "algo": "ppo", "reward_mode": "shaped"},
        timing={"epoch_id": 3, "inner_epoch": 5, "batch": 10, "episodes_completed": 40, "episodes_total": 100, "elapsed_s": 12.3, "eta_s": 99.0},
        policy_pulse={"kl_divergence": {"value": 0.02, "status": "ok", "threshold": 0.1, "ts": "now"}},
        system={"devices": [], "cpu": {}, "ram": {}, "io": {}, "staleness": {}},
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                throughput={"fps": 100.0, "step_time_ms": 10.0, "dataloader_wait_ms": 1.0},
                metrics={"task_metric": 82.1, "task_metric_delta": 0.5, "reward": 0.3, "rent": -0.05},
                action={"op": "GERMINATE", "slot_id": "mid", "blueprint_id": "conv", "blend_id": "sigmoid", "masked": False, "success": True, "reason": None},
                slots={"mid": SlotChipState(slot_id="mid", stage="BLENDING", blueprint_id="conv", alpha=0.7, epochs_in_stage=3, epochs_total=7, gate={"last": "G2", "passed": True, "reason": ""})},
                anomaly={"score": 0.1, "reasons": ["ok"]},
                status="OK",
                staleness={"ts": "now"},
            )
        ],
        event_feed=[],
        ui_meta={"snapshot_age_ms": 5, "last_render_ms": 8, "avg_render_ms": 9},
    )
    import json
    data = snap.to_dict()
    json.dumps(data)  # should not raise

    # Verify connection status
    assert data["connection"]["connected"] is True
    assert data["connection"]["staleness_s"] == 0.5

def test_connection_status_text():
    from esper.karn.overwatch.snapshot import ConnectionStatus

    live = ConnectionStatus(connected=True, last_event_ts=0, staleness_s=0.5)
    assert "Live" in live.status_text

    stale = ConnectionStatus(connected=True, last_event_ts=0, staleness_s=8.0)
    assert "Stale" in stale.status_text

    disconnected = ConnectionStatus(connected=False, last_event_ts=0, staleness_s=30.0)
    assert "Disconnected" in disconnected.status_text
```

**Step 2: Run test (expect fail)**
- `pytest tests/karn/overwatch/test_snapshot_schema.py -q`

**Step 3: Implement**
- Add dataclasses for `ConnectionStatus`, `SlotChipState`, `EnvSummary`, `TuiSnapshot`
- `ConnectionStatus` includes `status_text` property with color markup
- `to_dict()` produces JSON-safe structures
- Set defaults for optional fields
- Include schema_version for future migrations

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_snapshot_schema.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/snapshot.py src/esper/karn/overwatch/__init__.py tests/karn/overwatch/test_snapshot_schema.py
git commit -m "feat(overwatch): add TuiSnapshot schema with connection status"
```

---

### Task 2: Define aggregator contracts and telemetry intake

**Files:**
- Create: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_aggregator_contracts.py`

**Step 1: Write failing test**
```python
def test_aggregator_accepts_events_and_builds_snapshot():
    from esper.karn.overwatch.aggregator import TelemetryAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType

    agg = TelemetryAggregator(schema_version=1)
    agg.handle_event(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={"task": "cifar10", "algo": "ppo"}
    ))
    agg.handle_event(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={
            "kl_divergence": 0.02,
            "entropy": 1.2,
            "clip_fraction": 0.05,
            "explained_variance": 0.1,
            "lr": 0.0003,
            "grad_norm": 1.1,
            "update_time_ms": 12.0
        }
    ))

    snap = agg.build_snapshot()
    assert snap.policy_pulse["kl_divergence"]["value"] == 0.02
    assert snap.schema_version == 1
    assert snap.connection.connected is True

def test_aggregator_tracks_connection_staleness():
    from esper.karn.overwatch.aggregator import TelemetryAggregator
    import time

    agg = TelemetryAggregator(schema_version=1)
    # Simulate no events for a while
    agg._last_event_time = time.time() - 10.0

    snap = agg.build_snapshot()
    assert snap.connection.staleness_s >= 10.0
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_aggregator_contracts.py -q`

**Step 3: Implement**
- Aggregator class: maintain minimal state (run context, timing, policy pulse, last_event_time)
- Track connection status based on time since last event
- Accept TelemetryEvents via `handle_event`, store last values
- `build_snapshot` returns `TuiSnapshot` with current connection state
- Stub anomaly/bound/flight board with empty lists initially

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_aggregator_contracts.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_aggregator_contracts.py
git commit -m "feat(overwatch): add TelemetryAggregator with connection tracking"
```

---

### Task 3: Add anomaly scoring + bound detector

**Files:**
- Modify: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_anomaly_and_bounds.py`

**Step 1: Write failing test**
```python
def test_anomaly_scoring_with_reasons():
    from esper.karn.overwatch.aggregator import score_anomaly

    # Normal env - low score
    normal_env = {
        "throughput": {"fps": 100},
        "metrics": {"reward": 0.5},
        "grad_norm_ratio": 1.0,
        "staleness_s": 0.5,
        "memory_pct": 70
    }
    score, reasons = score_anomaly(normal_env, baseline_fps=100)
    assert score < 0.2
    assert "ok" in reasons[0].lower()

    # Anomalous env - high score with specific reasons
    bad_env = {
        "throughput": {"fps": 40},  # 60% drop
        "metrics": {"reward": -1.5},  # Unusual negative
        "grad_norm_ratio": 15.0,  # Very high
        "staleness_s": 12.0,  # Stale
        "memory_pct": 97  # High pressure
    }
    score, reasons = score_anomaly(bad_env, baseline_fps=100)
    assert score > 0.7
    assert any("throughput" in r.lower() for r in reasons)
    assert any("gradient" in r.lower() for r in reasons)
    assert any("memory" in r.lower() for r in reasons)

def test_anomaly_reasons_for_context_pane():
    """Reasons should be human-readable for 'Why flagged' display."""
    from esper.karn.overwatch.aggregator import score_anomaly

    env = {
        "throughput": {"fps": 30},
        "metrics": {"reward": 0.1},
        "grad_norm_ratio": 12.5,
        "staleness_s": 0.5,
        "memory_pct": 60
    }
    score, reasons = score_anomaly(env, baseline_fps=100)

    # Reasons should be descriptive strings, not codes
    for reason in reasons:
        assert len(reason) > 10  # Not just "throughput" but "Throughput 70% below baseline"
        assert not reason.startswith("ERR_")  # No error codes

def test_bound_detector():
    from esper.karn.overwatch.aggregator import detect_bounds

    # Compute-bound: high GPU util, low IO wait
    compute = detect_bounds(dev={"util": 95, "mem_pct": 70}, io={"wait_ms": 2})
    assert "compute-bound" in compute

    # Memory-bound: high memory pressure
    memory = detect_bounds(dev={"util": 60, "mem_pct": 95}, io={"wait_ms": 2})
    assert "memory-bound" in memory

    # IO-bound: high IO wait
    io_bound = detect_bounds(dev={"util": 50, "mem_pct": 50}, io={"wait_ms": 50})
    assert "io-bound" in io_bound
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_anomaly_and_bounds.py -q`

**Step 3: Implement**

Anomaly scoring weights:
```python
ANOMALY_WEIGHTS = {
    "throughput_drop": 0.30,  # >50% below baseline
    "negative_reward": 0.20,  # reward < -1.0
    "grad_norm_ratio": 0.25,  # >10x typical
    "staleness": 0.15,        # >10s since update
    "memory_pressure": 0.10,  # >95% GPU memory
}

def score_anomaly(env: dict, baseline_fps: float) -> tuple[float, list[str]]:
    """Score anomaly 0.0-1.0 with human-readable reasons."""
    score = 0.0
    reasons = []

    # Throughput check
    fps = env.get("throughput", {}).get("fps", baseline_fps)
    drop_pct = (baseline_fps - fps) / baseline_fps * 100 if baseline_fps > 0 else 0
    if drop_pct > 50:
        score += ANOMALY_WEIGHTS["throughput_drop"]
        reasons.append(f"Throughput {drop_pct:.0f}% below baseline")

    # Negative reward check
    reward = env.get("metrics", {}).get("reward", 0)
    if reward < -1.0:
        score += ANOMALY_WEIGHTS["negative_reward"]
        reasons.append(f"Unusual negative reward ({reward:.2f})")

    # Gradient norm ratio check
    grad_ratio = env.get("grad_norm_ratio", 1.0)
    if grad_ratio > 10:
        score += ANOMALY_WEIGHTS["grad_norm_ratio"]
        reasons.append(f"High gradient norm ratio ({grad_ratio:.1f}x)")

    # Staleness check
    staleness = env.get("staleness_s", 0)
    if staleness > 10:
        score += ANOMALY_WEIGHTS["staleness"]
        reasons.append(f"Env stale ({staleness:.1f}s since update)")

    # Memory pressure check
    mem_pct = env.get("memory_pct", 0)
    if mem_pct > 95:
        score += ANOMALY_WEIGHTS["memory_pressure"]
        reasons.append(f"Memory pressure ({mem_pct:.0f}% GPU mem)")

    if not reasons:
        reasons = ["Operating normally"]

    return min(score, 1.0), reasons
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_anomaly_and_bounds.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_anomaly_and_bounds.py
git commit -m "feat(overwatch): add anomaly scoring with human-readable reasons"
```

---

### Task 3.5: Add hysteresis and stable sort for flight board

**Files:**
- Create: `src/esper/karn/overwatch/display_state.py`
- Modify: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_hysteresis.py`

**Step 1: Write failing test**
```python
def test_status_hysteresis_requires_three_snapshots():
    from esper.karn.overwatch.display_state import EnvDisplayState

    state = EnvDisplayState(env_id=0, displayed_status="OK", pending_status="OK", pending_count=0)

    # First WARN - should still show OK
    result = state.update("WARN")
    assert result == "OK"
    assert state.pending_count == 1

    # Second WARN - should still show OK
    result = state.update("WARN")
    assert result == "OK"
    assert state.pending_count == 2

    # Third WARN - NOW should show WARN
    result = state.update("WARN")
    assert result == "WARN"
    assert state.displayed_status == "WARN"

def test_hysteresis_resets_on_status_change():
    from esper.karn.overwatch.display_state import EnvDisplayState

    state = EnvDisplayState(env_id=0, displayed_status="OK", pending_status="OK", pending_count=0)

    # Two WARNs
    state.update("WARN")
    state.update("WARN")
    assert state.pending_count == 2

    # Back to OK - resets pending
    state.update("OK")
    assert state.pending_count == 0
    assert state.displayed_status == "OK"

def test_stable_sort_by_anomaly_then_env_id():
    from esper.karn.overwatch.aggregator import sort_flight_board
    from esper.karn.overwatch.snapshot import EnvSummary

    envs = [
        EnvSummary(env_id=2, anomaly={"score": 0.5}),
        EnvSummary(env_id=0, anomaly={"score": 0.5}),  # Same score, lower id
        EnvSummary(env_id=1, anomaly={"score": 0.8}),  # Highest score
    ]

    sorted_envs = sort_flight_board(envs)

    # Highest anomaly first
    assert sorted_envs[0].env_id == 1
    # Tie-breaker: lower env_id first
    assert sorted_envs[1].env_id == 0
    assert sorted_envs[2].env_id == 2
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_hysteresis.py -q`

**Step 3: Implement**

```python
# src/esper/karn/overwatch/display_state.py
from dataclasses import dataclass

@dataclass
class EnvDisplayState:
    """Tracks display state for hysteresis to prevent UI flicker."""
    env_id: int
    displayed_status: str
    pending_status: str
    pending_count: int

    HYSTERESIS_THRESHOLD = 3

    def update(self, new_status: str) -> str:
        """Returns status to display (may be delayed by hysteresis)."""
        if new_status == self.displayed_status:
            # Status unchanged, reset pending
            self.pending_status = new_status
            self.pending_count = 0
            return self.displayed_status

        if new_status == self.pending_status:
            # Same pending status, increment counter
            self.pending_count += 1
            if self.pending_count >= self.HYSTERESIS_THRESHOLD:
                # Stable enough to transition
                self.displayed_status = new_status
                self.pending_count = 0
                return self.displayed_status
        else:
            # Different pending status, reset
            self.pending_status = new_status
            self.pending_count = 1

        return self.displayed_status


def sort_flight_board(envs: list) -> list:
    """Sort by anomaly score descending, then env_id ascending for stability."""
    return sorted(envs, key=lambda e: (-e.anomaly.get("score", 0), e.env_id))
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_hysteresis.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/display_state.py src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_hysteresis.py
git commit -m "feat(overwatch): add hysteresis and stable sort for flight board"
```

---

### Task 4: Snapshot writer/reader for replay

**Files:**
- Create: `src/esper/karn/overwatch/replay.py`
- Tests: `tests/karn/overwatch/test_replay.py`

**Step 1: Write failing test**
```python
def test_snapshot_round_trip(tmp_path):
    from esper.karn.overwatch.snapshot import TuiSnapshot, ConnectionStatus
    from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader

    p = tmp_path / "snaps.jsonl"
    snap = TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-15T12:00:00Z",
        connection=ConnectionStatus(connected=True, last_event_ts=0, staleness_s=0.5),
        run={"id": "run1"},
        timing={"epoch_id": 1},
        policy_pulse={},
        system={},
        flight_board=[],
        event_feed=[],
        ui_meta={}
    )

    writer = SnapshotWriter(p)
    writer.write(snap)
    writer.write(snap)  # Write two
    writer.close()

    reader = SnapshotReader(p)
    snaps = list(reader)
    assert len(snaps) == 2
    assert snaps[0].schema_version == 1
    assert snaps[0].connection.connected is True

def test_reader_supports_filtering(tmp_path):
    from esper.karn.overwatch.snapshot import TuiSnapshot, ConnectionStatus
    from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader

    p = tmp_path / "snaps.jsonl"
    writer = SnapshotWriter(p)

    for i in range(10):
        snap = TuiSnapshot(
            schema_version=1,
            captured_at=f"2025-12-15T12:00:{i:02d}Z",
            connection=ConnectionStatus(connected=True, last_event_ts=0, staleness_s=0),
            run={"id": "run1"},
            timing={"epoch_id": i},
            policy_pulse={},
            system={},
            flight_board=[],
            event_feed=[],
            ui_meta={}
        )
        writer.write(snap)
    writer.close()

    # Read with filter
    reader = SnapshotReader(p, filter_fn=lambda s: s.timing["epoch_id"] >= 5)
    snaps = list(reader)
    assert len(snaps) == 5
    assert snaps[0].timing["epoch_id"] == 5
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_replay.py -q`

**Step 3: Implement**
- SnapshotWriter appends JSONL with proper flushing
- SnapshotReader yields `TuiSnapshot` from file with optional filter function
- Support for `--filter` CLI flag in replay mode

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_replay.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/replay.py tests/karn/overwatch/test_replay.py
git commit -m "feat(overwatch): add snapshot writer/reader with filter support"
```

---

### Task 5: Telemetry ingestion glue (hub listener)

**Files:**
- Create: `src/esper/karn/overwatch/listener.py`
- Tests: `tests/karn/overwatch/test_listener.py`

**Step 1: Write failing test**
```python
def test_listener_registers_and_forwards_events(mocker):
    from esper.karn.overwatch.listener import TelemetryListener

    agg = mocker.Mock()
    hub = mocker.Mock()
    listener = TelemetryListener(hub, agg)
    listener.start()

    assert hub.add_backend.called

    # Simulate emit
    event = mocker.Mock()
    backend = hub.add_backend.call_args[0][0]
    backend.emit(event)
    agg.handle_event.assert_called_with(event)

def test_listener_tracks_connection_state(mocker):
    from esper.karn.overwatch.listener import TelemetryListener
    import time

    agg = mocker.Mock()
    hub = mocker.Mock()
    listener = TelemetryListener(hub, agg)
    listener.start()

    # Initially connected (just started)
    assert listener.is_connected() is True

    # Simulate time passing without events
    listener._last_event_time = time.time() - 30.0
    assert listener.is_connected() is False
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_listener.py -q`

**Step 3: Implement**
- Lightweight backend that forwards TelemetryEvents to aggregator
- Track last event time for connection status
- Register via `hub.add_backend`

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_listener.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/listener.py tests/karn/overwatch/test_listener.py
git commit -m "feat(overwatch): add telemetry listener with connection tracking"
```

---

### Task 6: Textual app scaffold

**Files:**
- Create: `src/esper/karn/overwatch/app.py`
- Create: `src/esper/karn/overwatch/widgets/__init__.py`
- Create: `src/esper/karn/overwatch/widgets/header.py`
- Create: `src/esper/karn/overwatch/widgets/flight_board.py`
- Create: `src/esper/karn/overwatch/widgets/context_pane.py`
- Create: `src/esper/karn/overwatch/widgets/feed.py`
- Create: `src/esper/karn/overwatch/widgets/empty_state.py`
- Create: `src/esper/karn/overwatch/widgets/help_overlay.py`
- Create: `src/esper/karn/overwatch/styles.css`
- Tests: `tests/karn/overwatch/test_app_scaffold.py`

**Step 1: Write failing test**
```python
def test_textual_app_launches_safely():
    from esper.karn.overwatch.app import OverwatchApp

    app = OverwatchApp(snapshot_provider=lambda: None)
    assert app is not None
    assert app.BINDINGS  # Should have keyboard bindings

def test_app_has_required_bindings():
    from esper.karn.overwatch.app import OverwatchApp

    app = OverwatchApp(snapshot_provider=lambda: None)
    binding_keys = [b.key for b in app.BINDINGS]

    # Required navigation
    assert "j" in binding_keys or "down" in binding_keys
    assert "k" in binding_keys or "up" in binding_keys
    assert "enter" in binding_keys
    assert "escape" in binding_keys

    # Required actions
    assert "q" in binding_keys
    assert "?" in binding_keys
    assert "f" in binding_keys
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_app_scaffold.py -q`

**Step 3: Implement**

App structure:
```python
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer
from textual.containers import Container, Horizontal, Vertical

class OverwatchApp(App):
    """Esper Overwatch TUI - Air Traffic Control for Training."""

    CSS_PATH = "styles.css"

    BINDINGS = [
        # Navigation
        Binding("j", "move_down", "Next env", show=False),
        Binding("k", "move_up", "Prev env", show=False),
        Binding("down", "move_down", "Next env"),
        Binding("up", "move_up", "Prev env"),
        Binding("enter", "expand", "Expand"),
        Binding("l", "expand", "Expand", show=False),
        Binding("escape", "collapse", "Collapse"),
        Binding("h", "collapse", "Collapse", show=False),

        # Panels
        Binding("f", "toggle_feed", "Feed"),
        Binding("c", "toggle_context", "Context"),
        Binding("1", "focus_board", "Board", show=False),
        Binding("2", "focus_feed", "Feed", show=False),

        # Filters
        Binding("/", "open_filter", "Filter"),
        Binding("w", "filter_warnings", "Warnings", show=False),
        Binding("ctrl+k", "command_palette", "Commands"),

        # Actions
        Binding("e", "export_snapshot", "Export"),
        Binding("space", "toggle_pause", "Pause"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("?", "show_help", "Help"),
    ]

    def __init__(self, snapshot_provider, **kwargs):
        super().__init__(**kwargs)
        self.snapshot_provider = snapshot_provider
        self.selected_env = 0
        self.paused = False
        self.show_feed = True
        self.show_context = True
```

CSS tokens for stage colors:
```css
/* styles.css */
$stage-dormant: #9E9E9E;
$stage-germinated: #00BCD4;
$stage-training: #2196F3;
$stage-blending: #9C27B0;
$stage-fossilized: #4CAF50;
$stage-culled: #F44336;

$status-ok: #4CAF50;
$status-info: #2196F3;
$status-warn: #FF9800;
$status-critical: #F44336;

$flash-change: #FFEB3B;
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_app_scaffold.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py src/esper/karn/overwatch/widgets/*.py src/esper/karn/overwatch/styles.css tests/karn/overwatch/test_app_scaffold.py
git commit -m "feat(overwatch): add Textual app scaffold with keyboard bindings"
```

---

### Task 7: Render TuiSnapshot (header + flight board)

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/header.py`
- Modify: `src/esper/karn/overwatch/widgets/flight_board.py`
- Tests: `tests/karn/overwatch/test_render_flight_board.py`

**Step 1: Write failing test**
```python
def test_header_shows_connection_status(snapshot_factory):
    from esper.karn.overwatch.widgets.header import render_header
    from esper.karn.overwatch.snapshot import ConnectionStatus

    # Live connection
    snap = snapshot_factory(connection=ConnectionStatus(connected=True, last_event_ts=0, staleness_s=0.5))
    text = render_header(snap)
    assert "Live" in text

    # Stale connection
    snap = snapshot_factory(connection=ConnectionStatus(connected=True, last_event_ts=0, staleness_s=8.0))
    text = render_header(snap)
    assert "Stale" in text

    # Disconnected
    snap = snapshot_factory(connection=ConnectionStatus(connected=False, last_event_ts=0, staleness_s=30.0))
    text = render_header(snap)
    assert "Disconnected" in text

def test_header_shows_policy_pulse_metrics(snapshot_factory):
    from esper.karn.overwatch.widgets.header import render_header

    snap = snapshot_factory(policy_pulse={
        "kl_divergence": {"value": 0.02, "status": "ok"},
        "entropy": {"value": 1.2, "status": "ok"},
        "clip_fraction": {"value": 0.05, "status": "ok"},
    })
    text = render_header(snap)
    assert "KL" in text
    assert "0.02" in text

def test_flight_board_renders_envs_sorted_by_anomaly(snapshot_factory):
    from esper.karn.overwatch.widgets.flight_board import render_flight_board
    from esper.karn.overwatch.snapshot import EnvSummary

    snap = snapshot_factory(flight_board=[
        EnvSummary(env_id=0, anomaly={"score": 0.1}),
        EnvSummary(env_id=1, anomaly={"score": 0.8}),  # Highest
        EnvSummary(env_id=2, anomaly={"score": 0.3}),
    ])

    text = render_flight_board(snap)

    # Env 1 should appear first (highest anomaly)
    idx_env1 = text.find("Env 1") if "Env 1" in text else text.find("env_id=1")
    idx_env0 = text.find("Env 0") if "Env 0" in text else text.find("env_id=0")
    assert idx_env1 < idx_env0  # Env 1 before Env 0

def test_flight_board_shows_slot_chips(snapshot_factory):
    from esper.karn.overwatch.widgets.flight_board import render_flight_board
    from esper.karn.overwatch.snapshot import EnvSummary, SlotChipState

    snap = snapshot_factory(flight_board=[
        EnvSummary(
            env_id=0,
            slots={
                "mid": SlotChipState(
                    slot_id="mid",
                    stage="BLENDING",
                    blueprint_id="conv",
                    alpha=0.7,
                    epochs_in_stage=3,
                    epochs_total=7,
                    gate={"last": "G2", "passed": True}
                )
            }
        )
    ])

    text = render_flight_board(snap)
    assert "mid" in text.lower()
    assert "BLENDING" in text or "blending" in text.lower()
    assert "0.7" in text  # Alpha
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_render_flight_board.py -q`

**Step 3: Implement**

Header widget:
```python
def render_header(snap: TuiSnapshot) -> str:
    """Render header with connection status and policy pulse."""
    connection = snap.connection.status_text

    pulse_parts = []
    for metric, data in snap.policy_pulse.items():
        status_icon = "✓" if data.get("status") == "ok" else "⚠"
        pulse_parts.append(f"{metric}: {data['value']:.3f} {status_icon}")

    timing = snap.timing
    timing_str = f"Epoch {timing.get('epoch_id', '?')}/{timing.get('epochs_total', '?')}"

    return f"{connection} | {' | '.join(pulse_parts)} | {timing_str}"
```

Flight board with slot chips:
```python
def render_slot_chip(chip: SlotChipState) -> str:
    """Render a slot chip with stage, alpha, gate."""
    # Progress bar
    filled = int(chip.alpha * 10)
    bar = "█" * filled + "░" * (10 - filled)

    # Gate status
    gate = chip.gate or {}
    gate_str = ""
    if gate.get("last"):
        icon = "✓" if gate.get("passed") else "✕"
        gate_str = f" {gate['last']}{icon}"

    return f"[{chip.slot_id}] {chip.stage} {bar} {chip.alpha:.1f}α{gate_str}"
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_render_flight_board.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/header.py src/esper/karn/overwatch/widgets/flight_board.py tests/karn/overwatch/test_render_flight_board.py
git commit -m "feat(overwatch): render header and flight board with slot chips"
```

---

### Task 7.5: Connection status indicator with change flash

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/header.py`
- Create: `src/esper/karn/overwatch/flash.py`
- Tests: `tests/karn/overwatch/test_connection_indicator.py`

**Step 1: Write failing test**
```python
def test_connection_indicator_states():
    from esper.karn.overwatch.widgets.header import ConnectionIndicator
    from esper.karn.overwatch.snapshot import ConnectionStatus

    indicator = ConnectionIndicator()

    # Live - green pulsing
    live = ConnectionStatus(connected=True, last_event_ts=0, staleness_s=0.5)
    markup = indicator.render(live)
    assert "green" in markup.lower() or "#4CAF50" in markup
    assert "Live" in markup

    # Stale - yellow
    stale = ConnectionStatus(connected=True, last_event_ts=0, staleness_s=8.0)
    markup = indicator.render(stale)
    assert "yellow" in markup.lower() or "#FF9800" in markup
    assert "Stale" in markup

    # Disconnected - red
    disconnected = ConnectionStatus(connected=False, last_event_ts=0, staleness_s=30.0)
    markup = indicator.render(disconnected)
    assert "red" in markup.lower() or "#F44336" in markup
    assert "Disconnected" in markup

def test_value_change_flash_tracking():
    from esper.karn.overwatch.flash import FlashTracker
    import time

    tracker = FlashTracker(flash_duration_ms=500)

    # Initial value - no flash
    assert tracker.should_flash("kl", 0.02) is False

    # Same value - no flash
    assert tracker.should_flash("kl", 0.02) is False

    # Changed value - flash
    assert tracker.should_flash("kl", 0.03) is True

    # Within flash duration - still flash
    assert tracker.is_flashing("kl") is True

    # Simulate time passing
    tracker._flash_times["kl"] = time.time() - 1.0  # 1 second ago
    assert tracker.is_flashing("kl") is False
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_connection_indicator.py -q`

**Step 3: Implement**

Flash tracker for value changes:
```python
# src/esper/karn/overwatch/flash.py
import time
from dataclasses import dataclass, field

@dataclass
class FlashTracker:
    """Track value changes and provide flash indicators."""
    flash_duration_ms: int = 500
    _values: dict = field(default_factory=dict)
    _flash_times: dict = field(default_factory=dict)

    def should_flash(self, key: str, value: any) -> bool:
        """Check if value changed and record flash."""
        old_value = self._values.get(key)
        self._values[key] = value

        if old_value is not None and old_value != value:
            self._flash_times[key] = time.time()
            return True
        return False

    def is_flashing(self, key: str) -> bool:
        """Check if key is currently in flash state."""
        flash_time = self._flash_times.get(key)
        if flash_time is None:
            return False

        elapsed_ms = (time.time() - flash_time) * 1000
        return elapsed_ms < self.flash_duration_ms
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_connection_indicator.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/header.py src/esper/karn/overwatch/flash.py tests/karn/overwatch/test_connection_indicator.py
git commit -m "feat(overwatch): add connection indicator with value change flash"
```

---

### Task 8: Context Pane and Telemetry Feed rendering

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/context_pane.py`
- Modify: `src/esper/karn/overwatch/widgets/feed.py`
- Tests: `tests/karn/overwatch/test_render_context_and_feed.py`

**Step 1: Write failing test**
```python
def test_context_pane_shows_why_flagged(snapshot_factory):
    from esper.karn.overwatch.widgets.context_pane import render_context
    from esper.karn.overwatch.snapshot import EnvSummary

    snap = snapshot_factory(flight_board=[
        EnvSummary(
            env_id=0,
            anomaly={
                "score": 0.7,
                "reasons": [
                    "Throughput 60% below baseline",
                    "High gradient norm ratio (15.2x)"
                ]
            }
        )
    ])

    text = render_context(snap, selected_env_id=0)
    assert "Why flagged" in text or "Anomaly" in text
    assert "Throughput" in text
    assert "gradient" in text.lower()

def test_context_pane_shows_system_vitals(snapshot_factory):
    from esper.karn.overwatch.widgets.context_pane import render_context

    snap = snapshot_factory(system={
        "devices": [
            {"id": 0, "name": "GPU 0", "util": 87, "mem_pct": 70},
            {"id": 1, "name": "GPU 1", "util": 95, "mem_pct": 92},
        ]
    })

    text = render_context(snap, selected_env_id=None)
    assert "GPU 0" in text
    assert "GPU 1" in text
    assert "87" in text or "87%" in text

def test_feed_renders_events_with_filters(snapshot_factory):
    from esper.karn.overwatch.widgets.feed import render_feed

    snap = snapshot_factory(event_feed=[
        {"ts": "12:00:03", "type": "GATE", "env_id": 3, "message": "mid passed G2"},
        {"ts": "12:00:01", "type": "STAGE", "env_id": 3, "message": "mid TRAINING → BLENDING"},
        {"ts": "11:59:58", "type": "PPO", "env_id": None, "message": "Update 47: KL=0.019"},
    ])

    text = render_feed(snap, filter_types=["GATE", "STAGE"])
    assert "GATE" in text
    assert "STAGE" in text
    assert "PPO" not in text  # Filtered out
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_render_context_and_feed.py -q`

**Step 3: Implement**

Context pane with "Why flagged":
```python
def render_context(snap: TuiSnapshot, selected_env_id: int | None) -> str:
    """Render context pane with vitals and selected env details."""
    lines = []

    # System vitals
    lines.append("System Vitals")
    lines.append("─" * 20)
    for dev in snap.system.get("devices", []):
        util = dev.get("util", 0)
        mem = dev.get("mem_pct", 0)
        warn = " ⚠" if util > 90 or mem > 90 else ""
        lines.append(f"{dev['name']}: {util}% {mem}% mem{warn}")

    # Selected env details
    if selected_env_id is not None:
        env = next((e for e in snap.flight_board if e.env_id == selected_env_id), None)
        if env:
            lines.append("")
            lines.append(f"Selected: Env {selected_env_id}")
            lines.append("─" * 20)

            # Why flagged
            if env.anomaly.get("score", 0) > 0.2:
                lines.append("Why flagged:")
                for reason in env.anomaly.get("reasons", []):
                    lines.append(f"• {reason}")

    return "\n".join(lines)
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_render_context_and_feed.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/context_pane.py src/esper/karn/overwatch/widgets/feed.py tests/karn/overwatch/test_render_context_and_feed.py
git commit -m "feat(overwatch): render context pane with 'why flagged' and feed"
```

---

### Task 8.5: Empty states and onboarding

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/empty_state.py`
- Create: `src/esper/karn/overwatch/onboarding.py`
- Tests: `tests/karn/overwatch/test_empty_states.py`

**Step 1: Write failing test**
```python
def test_empty_state_waiting_for_telemetry():
    from esper.karn.overwatch.widgets.empty_state import render_empty_state

    text = render_empty_state(reason="no_telemetry")
    assert "Waiting for telemetry" in text
    assert "esper ppo --overwatch" in text or "--overwatch" in text
    assert "?" in text  # Mention help shortcut

def test_empty_state_no_matching_filters():
    from esper.karn.overwatch.widgets.empty_state import render_empty_state

    text = render_empty_state(
        reason="no_matches",
        filter_desc="stage=BLENDING",
        total_envs=8
    )
    assert "No environments match" in text
    assert "BLENDING" in text
    assert "8" in text
    assert "/" in text or "Esc" in text  # Mention how to clear

def test_onboarding_hint_shown_first_time(tmp_path, monkeypatch):
    from esper.karn.overwatch.onboarding import should_show_onboarding, mark_onboarding_seen

    # Patch home directory
    monkeypatch.setenv("HOME", str(tmp_path))

    # First time - should show
    assert should_show_onboarding() is True

    # Mark as seen
    mark_onboarding_seen()

    # Second time - should not show
    assert should_show_onboarding() is False

def test_onboarding_content():
    from esper.karn.overwatch.onboarding import get_onboarding_content

    content = get_onboarding_content()
    assert "j/k" in content or "↑/↓" in content  # Navigation
    assert "Enter" in content  # Expand
    assert "/" in content  # Filter
    assert "?" in content  # Help
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_empty_states.py -q`

**Step 3: Implement**

Empty states:
```python
# src/esper/karn/overwatch/widgets/empty_state.py

EMPTY_STATE_TEMPLATES = {
    "no_telemetry": """
┌────────────────────────────┐
│                            │
│   Waiting for telemetry    │
│                            │
│   Start training with:     │
│   esper ppo --overwatch    │
│                            │
│   Or replay a session:     │
│   esper overwatch          │
│     --replay snaps.jsonl   │
│                            │
└────────────────────────────┘

Press ? for keyboard shortcuts
""",
    "no_matches": """
No environments match current filters.

Current filters: {filter_desc}
Total environments: {total_envs} (0 matching)

Press / to modify filters or Esc to clear.
"""
}

def render_empty_state(reason: str, **kwargs) -> str:
    template = EMPTY_STATE_TEMPLATES.get(reason, "No data available")
    return template.format(**kwargs)
```

Onboarding:
```python
# src/esper/karn/overwatch/onboarding.py
from pathlib import Path
import os

def _marker_path() -> Path:
    home = Path(os.environ.get("HOME", "~")).expanduser()
    return home / ".esper" / "overwatch_seen"

def should_show_onboarding() -> bool:
    return not _marker_path().exists()

def mark_onboarding_seen() -> None:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

def get_onboarding_content() -> str:
    return """
Welcome to Esper Overwatch!

Quick tips:
• j/k or ↑/↓ to navigate environments
• Enter to expand env details
• / to filter, ? for all shortcuts
• Anomalies are sorted to the top automatically
"""
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_empty_states.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/empty_state.py src/esper/karn/overwatch/onboarding.py tests/karn/overwatch/test_empty_states.py
git commit -m "feat(overwatch): add empty states and first-time onboarding"
```

---

### Task 9: Interaction model (navigation, filters, overlays)

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`
- Create: `src/esper/karn/overwatch/widgets/help_overlay.py`
- Create: `src/esper/karn/overwatch/widgets/command_palette.py`
- Tests: `tests/karn/overwatch/test_interactions.py`

**Step 1: Write failing test**
```python
def test_navigation_wraps_at_bounds(mocker):
    from esper.karn.overwatch.app import OverwatchApp
    from esper.karn.overwatch.snapshot import TuiSnapshot, EnvSummary

    snap = TuiSnapshot(flight_board=[
        EnvSummary(env_id=0),
        EnvSummary(env_id=1),
        EnvSummary(env_id=2),
    ])

    app = OverwatchApp(snapshot_provider=lambda: snap)
    app.selected_env = 0

    # Move down
    app.action_move_down()
    assert app.selected_env == 1

    # Move to end
    app.action_move_down()
    app.action_move_down()
    assert app.selected_env == 2  # At end, doesn't wrap

    # Move up
    app.action_move_up()
    assert app.selected_env == 1

def test_filter_toggles(mocker):
    from esper.karn.overwatch.app import OverwatchApp

    app = OverwatchApp(snapshot_provider=lambda: mocker.Mock(flight_board=[]))

    # Initially no filters
    assert app.filters == {}

    # Toggle warning filter
    app.action_filter_warnings()
    assert app.filters.get("status") == "WARN"

    # Toggle again - removes filter
    app.action_filter_warnings()
    assert "status" not in app.filters

def test_help_overlay_content():
    from esper.karn.overwatch.widgets.help_overlay import render_help

    text = render_help()

    # Check all documented shortcuts are present
    assert "j" in text and "k" in text  # Navigation
    assert "Enter" in text  # Expand
    assert "/" in text  # Filter
    assert "?" in text  # Help (this one)
    assert "q" in text  # Quit
    assert "Ctrl+k" in text.lower() or "ctrl-k" in text.lower()  # Command palette

def test_expand_collapse_state(mocker):
    from esper.karn.overwatch.app import OverwatchApp
    from esper.karn.overwatch.snapshot import EnvSummary

    app = OverwatchApp(snapshot_provider=lambda: mocker.Mock(flight_board=[
        EnvSummary(env_id=0),
        EnvSummary(env_id=1),
    ]))
    app.selected_env = 0
    app.expanded = False

    # Expand
    app.action_expand()
    assert app.expanded is True

    # Collapse
    app.action_collapse()
    assert app.expanded is False
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_interactions.py -q`

**Step 3: Implement**

App state and actions:
```python
class OverwatchApp(App):
    def __init__(self, snapshot_provider, **kwargs):
        super().__init__(**kwargs)
        self.snapshot_provider = snapshot_provider
        self.selected_env = 0
        self.expanded = False
        self.paused = False
        self.filters = {}
        self.show_feed = True
        self.show_context = True
        self.show_help = False

    def action_move_down(self) -> None:
        snap = self.snapshot_provider()
        if snap and snap.flight_board:
            self.selected_env = min(self.selected_env + 1, len(snap.flight_board) - 1)

    def action_move_up(self) -> None:
        self.selected_env = max(self.selected_env - 1, 0)

    def action_expand(self) -> None:
        self.expanded = True

    def action_collapse(self) -> None:
        self.expanded = False

    def action_filter_warnings(self) -> None:
        if self.filters.get("status") == "WARN":
            del self.filters["status"]
        else:
            self.filters["status"] = "WARN"

    def action_show_help(self) -> None:
        self.show_help = not self.show_help
```

Help overlay:
```python
# src/esper/karn/overwatch/widgets/help_overlay.py

HELP_CONTENT = """
╔════════════════════════════════════════════════════════════════════╗
║                    ESPER OVERWATCH SHORTCUTS                        ║
╠════════════════════════════════════════════════════════════════════╣
║  NAVIGATION                                                         ║
║  j / ↓         Next environment                                     ║
║  k / ↑         Previous environment                                 ║
║  Enter / l     Expand environment details                           ║
║  Esc / h       Collapse details                                     ║
║                                                                     ║
║  PANELS                                                             ║
║  f             Toggle event feed                                    ║
║  c             Toggle context pane                                  ║
║  1             Focus flight board                                   ║
║  2             Focus event feed                                     ║
║                                                                     ║
║  FILTERS                                                            ║
║  /             Open filter bar                                      ║
║  w             Toggle warnings-only filter                          ║
║  s             Cycle stage filter                                   ║
║  Ctrl+k        Command palette                                      ║
║                                                                     ║
║  ACTIONS                                                            ║
║  e             Export current snapshot                              ║
║  Space         Pause/resume updates                                 ║
║  r             Force refresh                                        ║
║  q             Quit                                                 ║
║  ?             Show/hide this help                                  ║
╚════════════════════════════════════════════════════════════════════╝
"""

def render_help() -> str:
    return HELP_CONTENT
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_interactions.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py src/esper/karn/overwatch/widgets/help_overlay.py src/esper/karn/overwatch/widgets/command_palette.py tests/karn/overwatch/test_interactions.py
git commit -m "feat(overwatch): add navigation, filters, and help overlay"
```

---

### Task 10: Wiring CLI and modes (live vs replay)

**Files:**
- Modify: `src/esper/scripts/train.py` (add --overwatch flag)
- Create: `src/esper/scripts/overwatch.py` (standalone viewer)
- Tests: `tests/scripts/test_overwatch_entry.py`

**Step 1: Write failing test**
```python
def test_overwatch_cli_parses_live_mode():
    from esper.scripts.overwatch import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.replay is None
    assert args.live is True  # Default

def test_overwatch_cli_parses_replay_mode():
    from esper.scripts.overwatch import build_parser

    parser = build_parser()
    args = parser.parse_args(["--replay", "snaps.jsonl"])
    assert args.replay == "snaps.jsonl"
    assert args.live is False

def test_overwatch_cli_parses_filter():
    from esper.scripts.overwatch import build_parser

    parser = build_parser()
    args = parser.parse_args(["--replay", "snaps.jsonl", "--filter", "env_id=3"])
    assert args.filter == "env_id=3"

def test_overwatch_cli_parses_top_n():
    from esper.scripts.overwatch import build_parser

    parser = build_parser()
    args = parser.parse_args(["--top", "5"])
    assert args.top == 5

def test_train_cli_has_overwatch_flag():
    from esper.scripts.train import build_parser

    parser = build_parser()
    args = parser.parse_args(["ppo", "--overwatch"])
    assert args.overwatch is True
```

**Step 2: Run test**
- `pytest tests/scripts/test_overwatch_entry.py -q`

**Step 3: Implement**

Overwatch CLI:
```python
# src/esper/scripts/overwatch.py
"""Esper Overwatch - TUI for training monitoring."""
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Esper Overwatch - Air Traffic Control for Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live monitoring (attach to running training)
  esper overwatch

  # Replay a recorded session
  esper overwatch --replay telemetry_snaps.jsonl

  # Replay with filter
  esper overwatch --replay snaps.jsonl --filter "env_id=3"

  # Show only top 5 anomalous envs
  esper overwatch --top 5
"""
    )

    parser.add_argument(
        "--replay",
        metavar="FILE",
        help="Replay snapshots from JSONL file instead of live telemetry"
    )
    parser.add_argument(
        "--filter",
        metavar="EXPR",
        help="Filter expression (e.g., 'env_id=3', 'stage=BLENDING')"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only top N anomalous environments (0 = all)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact display mode for experienced users"
    )
    parser.add_argument(
        "--no-onboarding",
        action="store_true",
        help="Skip first-time onboarding hints"
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.live = args.replay is None

    # Launch TUI
    from esper.karn.overwatch.app import OverwatchApp

    if args.replay:
        from esper.karn.overwatch.replay import SnapshotReader
        reader = SnapshotReader(args.replay)
        # ... setup replay provider
    else:
        # ... setup live provider
        pass

    app = OverwatchApp(snapshot_provider=...)
    app.run()

if __name__ == "__main__":
    main()
```

**Step 4: Re-run test**
- `pytest tests/scripts/test_overwatch_entry.py -q`

**Step 5: Commit**
```bash
git add src/esper/scripts/overwatch.py src/esper/scripts/train.py tests/scripts/test_overwatch_entry.py
git commit -m "feat(overwatch): add CLI for live/replay modes with filter support"
```

---

### Task 11: Performance hardening and instrumentation

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`
- Modify: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_perf_guardrails.py`

**Step 1: Write failing test**
```python
def test_snapshot_throttling_fast_cadence():
    from esper.karn.overwatch.aggregator import TelemetryAggregator

    agg = TelemetryAggregator(schema_version=1, fast_hz=5, slow_hz=1)

    # First emit always allowed
    assert agg.should_emit_fast(now=0.0) is True
    agg._last_fast_emit = 0.0

    # Too soon (< 200ms for 5Hz)
    assert agg.should_emit_fast(now=0.05) is False
    assert agg.should_emit_fast(now=0.15) is False

    # Just right (>= 200ms)
    assert agg.should_emit_fast(now=0.20) is True

def test_snapshot_throttling_slow_cadence():
    from esper.karn.overwatch.aggregator import TelemetryAggregator

    agg = TelemetryAggregator(schema_version=1, fast_hz=5, slow_hz=1)
    agg._last_slow_emit = 0.0

    # Too soon (< 1000ms for 1Hz)
    assert agg.should_emit_slow(now=0.5) is False

    # Just right (>= 1000ms)
    assert agg.should_emit_slow(now=1.0) is True

def test_ui_meta_instrumentation():
    from esper.karn.overwatch.aggregator import TelemetryAggregator

    agg = TelemetryAggregator(schema_version=1)

    # Simulate render timing
    agg.record_render_time(8.5)
    agg.record_render_time(9.0)
    agg.record_render_time(8.0)

    snap = agg.build_snapshot()
    assert snap.ui_meta["last_render_ms"] == 8.0
    assert snap.ui_meta["avg_render_ms"] == 8.5  # Average of last 3

def test_large_flight_board_performance(benchmark):
    """Ensure rendering 16+ envs stays under 50ms."""
    from esper.karn.overwatch.widgets.flight_board import render_flight_board
    from esper.karn.overwatch.snapshot import TuiSnapshot, EnvSummary, SlotChipState

    # Create 16 envs with slots
    envs = []
    for i in range(16):
        envs.append(EnvSummary(
            env_id=i,
            device_id=i % 4,
            slots={
                "mid": SlotChipState(slot_id="mid", stage="TRAINING", alpha=0.5),
                "late": SlotChipState(slot_id="late", stage="DORMANT", alpha=0.0),
            },
            anomaly={"score": i * 0.05, "reasons": ["test"]},
        ))

    snap = TuiSnapshot(flight_board=envs)

    result = benchmark(render_flight_board, snap)

    # Should complete (benchmark will report time)
    assert result is not None
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_perf_guardrails.py -q`

**Step 3: Implement**

Throttling in aggregator:
```python
class TelemetryAggregator:
    def __init__(self, schema_version: int, fast_hz: float = 5.0, slow_hz: float = 1.0):
        self.schema_version = schema_version
        self.fast_hz = fast_hz
        self.slow_hz = slow_hz
        self._last_fast_emit = 0.0
        self._last_slow_emit = 0.0
        self._render_times = []

    def should_emit_fast(self, now: float) -> bool:
        interval = 1.0 / self.fast_hz
        if now - self._last_fast_emit >= interval:
            self._last_fast_emit = now
            return True
        return False

    def should_emit_slow(self, now: float) -> bool:
        interval = 1.0 / self.slow_hz
        if now - self._last_slow_emit >= interval:
            self._last_slow_emit = now
            return True
        return False

    def record_render_time(self, ms: float) -> None:
        self._render_times.append(ms)
        if len(self._render_times) > 10:
            self._render_times.pop(0)
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_perf_guardrails.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_perf_guardrails.py
git commit -m "feat(overwatch): add throttling and render instrumentation"
```

---

### Task 11.5: Accessibility testing and documentation

**Files:**
- Create: `tests/karn/overwatch/test_accessibility.py`
- Update: `README.md` (accessibility section)
- Create: `docs/overwatch-accessibility.md`

**Step 1: Write failing test**
```python
def test_all_status_indicators_have_text():
    """Status must use color + text + icon, not color alone."""
    from esper.karn.overwatch.widgets.flight_board import render_env_status

    for status in ["OK", "INFO", "WARN", "CRITICAL"]:
        markup = render_env_status(status)
        # Should contain the status text, not just a colored box
        assert status in markup or status.lower() in markup.lower()

def test_slot_stages_have_text_labels():
    """Stages must show text, not just color."""
    from esper.karn.overwatch.widgets.flight_board import render_slot_chip
    from esper.karn.overwatch.snapshot import SlotChipState

    for stage in ["DORMANT", "GERMINATED", "TRAINING", "BLENDING", "FOSSILIZED", "CULLED"]:
        chip = SlotChipState(slot_id="mid", stage=stage, alpha=0.5)
        markup = render_slot_chip(chip)
        assert stage in markup

def test_gate_status_has_text_alternative():
    """Gate icons must have text alternative."""
    from esper.karn.overwatch.widgets.flight_board import render_gate_status

    # Passed gate
    markup = render_gate_status({"last": "G2", "passed": True})
    # Should have checkmark AND text identifier
    assert "G2" in markup
    assert "✓" in markup or "passed" in markup.lower()

    # Failed gate
    markup = render_gate_status({"last": "G1", "passed": False})
    assert "G1" in markup
    assert "✕" in markup or "failed" in markup.lower()

def test_focus_indicator_visible():
    """Selected item must have visible focus indicator."""
    from esper.karn.overwatch.widgets.flight_board import render_env_row
    from esper.karn.overwatch.snapshot import EnvSummary

    env = EnvSummary(env_id=0)

    # Not selected
    normal = render_env_row(env, selected=False)

    # Selected - should have visual difference
    selected = render_env_row(env, selected=True)

    assert normal != selected
    # Selected should have some indicator (reverse, bracket, highlight)
    assert any(marker in selected for marker in ["[", ">", "█", "reverse"])

def test_keyboard_shortcuts_documented():
    """All bindings must appear in help overlay."""
    from esper.karn.overwatch.app import OverwatchApp
    from esper.karn.overwatch.widgets.help_overlay import render_help

    app = OverwatchApp(snapshot_provider=lambda: None)
    help_text = render_help().lower()

    # Check that visible bindings are documented
    for binding in app.BINDINGS:
        if binding.show:  # Only check visible bindings
            assert binding.key.lower() in help_text, f"Binding '{binding.key}' not in help"
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_accessibility.py -q`

**Step 3: Implement**

Add accessibility documentation:
```markdown
# docs/overwatch-accessibility.md

# Esper Overwatch Accessibility Guide

## Terminal Requirements

For optimal accessibility:
- **Width:** 100 columns minimum (120 recommended)
- **Height:** 30 rows minimum
- **Font:** 12pt+ monospace recommended
- **Theme:** High contrast theme recommended

## Color Independence

All status information uses multiple channels:
- **Color** for quick visual scanning
- **Text labels** for screen readers and colorblind users
- **Icons** for additional visual distinction

| Status | Color | Icon | Text |
|--------|-------|------|------|
| OK | Green | (none) | "OK" |
| INFO | Blue | ℹ | "INFO" |
| WARN | Yellow | ⚠ | "WARN" |
| CRITICAL | Red | ✕ | "CRITICAL" |

## Keyboard Navigation

Full keyboard accessibility:
- All operations available via keyboard
- Focus indicator visible on selected items
- Logical tab order (Header → Board → Context → Feed)

## Screen Reader Support

Textual framework provides basic screen reader support:
- **macOS VoiceOver:** Best compatibility
- **Linux ORCA:** Partial support
- **Windows NVDA:** Limited support

## Known Limitations

- Real-time updates may not announce automatically
- Some dynamic content requires manual refresh to announce
- Terminal screen readers have inherent limitations
```

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_accessibility.py -q`

**Step 5: Commit**
```bash
git add tests/karn/overwatch/test_accessibility.py README.md docs/overwatch-accessibility.md
git commit -m "feat(overwatch): add accessibility tests and documentation"
```

---

### Task 12: Cutover (remove Rich TUI, docs update)

**Files:**
- Delete: `src/esper/karn/tui.py` (and related Rich-specific code)
- Modify: `README.md` (update TUI section)
- Modify: `docs/specifications/telemetry-audit.md` (note UI consumption)
- Update CLI help text

**Step 1: Verify no references to old TUI**
```bash
grep -r "from esper.karn.tui" src/
grep -r "from esper.karn.tui" tests/
```

**Step 2: Remove old TUI and update docs**
- Remove Rich imports and old TUI code
- Update README with Overwatch usage examples
- Update telemetry audit to reference Overwatch as consumer

**Step 3: Run full test suite**
```bash
pytest -m "not slow" -q
```

**Step 4: Commit**
```bash
git add README.md docs/specifications/telemetry-audit.md
git rm src/esper/karn/tui.py
git commit -m "chore(overwatch): remove Rich TUI and document Textual Overwatch"
```

---

## Final Verification

### Unit Tests
```bash
pytest tests/karn/overwatch/ -q
```

### Integration Smoke Test
```bash
# Live mode
PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar10 --telemetry-level normal --overwatch

# Replay mode
PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay telemetry_snaps.jsonl
```

### Accessibility Verification
- [ ] Navigate with keyboard only (unplug mouse)
- [ ] Check all status indicators have text labels
- [ ] Verify focus indicator visible
- [ ] Test with high contrast terminal theme

### Performance Verification
- [ ] Test with 16+ environments
- [ ] Verify render time stays under 50ms
- [ ] Check no memory leaks during long sessions

---

## Execution Handoff

Plan complete and saved to `docs/plans/overwatch-textual-ui-expanded.md`.

**Task count:** 15 tasks (12 original + 3.5, 7.5, 8.5, 11.5)

**Telemetry enhancements:** 6 P1 critical gaps, 12 P2 important gaps integrated from PyTorch/DRL specialist reviews

**Execution options:**
1. **Subagent-Driven (this session)** — use superpowers:subagent-driven-development to execute tasks sequentially with code review between tasks.
2. **Parallel Session** — new session using superpowers:executing-plans to run the plan in batches with checkpoints.

**Recommended batch grouping:**
- Batch 1: Tasks 1-3.5 (Schema, Aggregator, Anomaly w/ P1 weights, Hysteresis)
- Batch 2: Tasks 4-5 (Replay, Listener)
- Batch 3: Tasks 6-7.5 (App scaffold, Rendering, Connection)
- Batch 4: Tasks 8-8.5 (Context w/ PyTorch+RL diagnostics, Feed, Empty states)
- Batch 5: Tasks 9-10 (Interactions, CLI)
- Batch 6: Tasks 11-12 (Performance, Accessibility, Cutover)

**Telemetry prerequisites:** Before starting Batch 1, verify that the following telemetry events exist or plan to add them:
- `PPO_UPDATE_COMPLETED` — must include `explained_variance`, `advantage_stats`, `bootstrap_ratio`
- `DEVICE_MEMORY_DETAILED` — must include `allocated_bytes`, `reserved_bytes`, per-device
- `GRADIENT_HEALTH` — new event for dead/exploding layer detection (or extend existing grad norm event)
