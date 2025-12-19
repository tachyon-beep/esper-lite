# Sanctum Design Document

> **Sanctum** - The inner chamber for deep diagnostic inspection of Esper training.

## The Karn TUI Family

Karn provides three monitoring interfaces:

| Interface | Framework | Purpose | User |
|-----------|-----------|---------|------|
| **Overwatch** | Textual | Operator monitoring | "How good are we doing?" |
| **Sanctum** | Textual | Developer debugging | "How bad are we doing?" |
| **Scry** | Vue/Web | Remote monitoring | Overwatch in a browser |

## Overview

Sanctum is the developer-focused diagnostic TUI for debugging misbehaving training runs. It complements Overwatch (operator monitoring) by providing deep inspection capabilities when something goes wrong.

| | **Overwatch** | **Sanctum** |
|---|---|---|
| **Question** | "How good are we doing?" | "How bad are we doing?" |
| **User** | Operator checking in | Developer debugging |
| **Understanding** | 80% at a glance | 95% after 2 minutes of study |
| **When to use** | Everything's working | Tamiyo's being weird |

### Example Use Case

Tamiyo has decided "no seeds is the best way to go" - it keeps choosing WAIT instead of GERMINATE. You need to understand:
- What action probabilities is Tamiyo seeing?
- What's the policy entropy/value estimate?
- Are any seeds performing well that should be fossilized?
- What do the reward components look like?

Sanctum answers these questions.

## Architecture

Sanctum lives as a peer to Overwatch within Karn:

```
src/esper/karn/
├── overwatch/          # Operator monitoring TUI (Textual)
│   ├── app.py
│   ├── schema.py
│   └── widgets/
├── sanctum/            # Developer diagnostic TUI (Textual) [NEW]
│   ├── app.py
│   ├── schema.py
│   └── widgets/
└── tui.py              # Legacy Rich TUI [TO BE DELETED after Phase 1]
```

## Core Panels (Phase 1)

Phase 1 ports the existing Rich TUI to Textual with the same functionality:

### 1. Environment Grid (16 cards)
- Compact cards showing per-env status
- Accuracy, loss, active seeds, reward
- Color-coded health indicators

### 2. Tamiyo Brain Panel
- Policy health: entropy, clip fraction, KL divergence
- Loss components: policy loss, value loss, entropy loss
- Vitals: gradient norm, learning rate, explained variance
- Action distribution: WAIT/GERMINATE/CULL/FOSSILIZE percentages

### 3. Reward Components Panel
- Breakdown of all reward components
- Per-env reward details when env selected

### 4. Seed Leaderboard
- Top seeds by contribution/improvement
- Stage, alpha, blueprint for each

### 5. Event Log
- Recent telemetry events
- Filtered by severity/type

### 6. System Vitals
- GPU utilization, memory
- Throughput (steps/sec)

## Existing Features to Preserve

The current Rich TUI has these features that must be ported 1:1:

### BEST RUNS Scoreboard (Already Exists!)
- Global best accuracy, Mean best
- Fossilized/Culled counts
- Top 10 envs by best accuracy with medals (🥇🥈🥉)
- Seeds at best accuracy for each env
- This IS the "Hall of Fame" - no new feature needed

### Flight Board
- 16 environment cards with per-env metrics
- Color-coded health status

### Tamiyo Brain
- Policy health, losses, vitals, action distribution

## Future Phases (Post-Port)

### Phase 2: Focus Mode
- Select any env card to expand into detail panel
- Full reward component breakdown for selected env
- Detailed seed telemetry for selected env

### Phase 3: Time Travel
- Scrub through historical decision data
- "What was Tamiyo thinking 50 steps ago?"
- Decision replay for debugging

## Layout (Phase 1)

```
┌─────────────────────────────────────────────────────────────────┐
│ HEADER: Run ID, Task, Episode, Runtime                          │
├─────────────────────────────────────────────────────────────────┤
│ ENV GRID (16 cards, 4x4)        │ TAMIYO BRAIN                  │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │ ┌─────────────────────────┐   │
│ │ 0 │ │ 1 │ │ 2 │ │ 3 │        │ │ Health │ Losses │Vitals│   │
│ └───┘ └───┘ └───┘ └───┘        │ └─────────────────────────┘   │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │ ┌─────────────────────────┐   │
│ │ 4 │ │ 5 │ │ 6 │ │ 7 │        │ │ Action Distribution     │   │
│ └───┘ └───┘ └───┘ └───┘        │ └─────────────────────────┘   │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐        ├─────────────────────────────────┤
│ │ 8 │ │ 9 │ │10 │ │11 │        │ SEED LEADERBOARD              │
│ └───┘ └───┘ └───┘ └───┘        │ Top 10 by contribution        │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │                               │
│ │12 │ │13 │ │14 │ │15 │        │                               │
│ └───┘ └───┘ └───┘ └───┘        │                               │
├─────────────────────────────────┼───────────────────────────────┤
│ REWARD COMPONENTS               │ EVENT LOG                     │
│ (per-env breakdown)             │ (recent telemetry)            │
├─────────────────────────────────┴───────────────────────────────┤
│ SYSTEM VITALS: GPU 85% | VRAM 12.4GB | 142 steps/sec            │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: Port to Textual

**Goal:** Same panels, same data, new framework. Delete Rich version when done.

**Approach:**
1. Create `karn/sanctum/` directory structure mirroring `karn/overwatch/`
2. Define `SanctumSnapshot` schema (similar to `TuiSnapshot`)
3. Create `SanctumApp` Textual application
4. Port each panel as a Textual widget:
   - `EnvGrid` - 16 environment cards
   - `TamiyoBrain` - Policy health/losses/vitals
   - `RewardPanel` - Reward component breakdown
   - `SeedLeaderboard` - Top seeds list
   - `EventLog` - Telemetry events (reuse from Overwatch?)
   - `SystemVitals` - GPU/memory/throughput
5. Wire to existing telemetry (same backend as old TUI)
6. Add CLI flag: `--sanctum` to launch instead of `--overwatch`
7. Test thoroughly, then delete `karn/tui.py`

**Reuse from Overwatch:**
- `styles.tcss` (base styles, can extend)
- `EventFeed` widget (if compatible)
- `ReplayController` (for future phases)
- Schema patterns (dataclasses with serialization)

**Key difference from Overwatch:**
- Denser information display (developer not operator)
- More numeric detail, less visual polish
- Focus on "explain why" not "show status"

## CLI Integration

```bash
# Operator monitoring (existing)
python -m esper.scripts.train ppo --overwatch

# Developer debugging (new)
python -m esper.scripts.train ppo --sanctum

# Both are mutually exclusive - pick one
```

## Success Criteria

Phase 1 is complete when:
1. All existing Rich TUI functionality works in Textual
2. `--sanctum` flag launches the new TUI
3. `karn/tui.py` is deleted
4. No regression in diagnostic capability

## Specialist Recommendations (Minor Enhancements for Phase 1)

### UX Specialist Suggestions

| # | Issue | Fix | Effort |
|---|-------|-----|--------|
| 1 | CPU indicator broken (collected but never displayed) | Add CPU% to ESPER STATUS panel | Trivial |
| 2 | No keyboard navigation in Rich | Add Textual focus states (Tab cycles panels) | Low |
| 3 | No data staleness indicator | Show "(Ns ago)" in panel headers when data >5s old | Low |
| 4 | Empty state unhelpful | Improve "Waiting for PPO data" message with guidance | Trivial |
| 5 | Row flicker on updates | Use Textual DataTable with stable row keys | Low |

### DRL Expert Suggestions

| # | Issue | Fix | Effort |
|---|-------|-----|--------|
| 1 | Entropy lacks context | Show as "Ent(/1.39)" or percentage of max | Trivial |
| 2 | ExplVar confusing scale | Add hint: "harm"/"weak"/"ok" based on value | Trivial |
| 3 | Clip thresholds permissive | Lower WARNING to 0.20, CRITICAL to 0.25 | Trivial |
| 4 | "GradHP" cryptic | Rename to "Grad Health" or "GradOK%" | Trivial |
| 5 | "Ratio↑/↓" unclear | Prefix with "π" to clarify policy ratio | Trivial |

All suggestions preserve existing layout. No new panels or major changes.

## Open Questions

1. Should Sanctum share telemetry backend with Overwatch, or have its own aggregator?
2. How much CSS/styling to share between Overwatch and Sanctum?
3. Should there be a way to switch between Overwatch and Sanctum at runtime?
