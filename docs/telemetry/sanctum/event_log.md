# Telemetry Audit: EventLog Widget

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/event_log.py`
**Purpose:** Append-only scrolling log displaying training events with rich metadata, smart drip-feed scrolling, and aggregation of high-frequency events.

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `event_log` | `list[EventLogEntry]` | `[]` | Primary data source - list of events to display |
| `total_events_received` | `int` | `0` | Used to compute events/sec rate for border title and drip-feed pacing |

### Source: EventLogEntry (from `snapshot.event_log`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `timestamp` | `str` | - | Display timestamp (HH:MM:SS format); used for sorting and display abbreviation |
| `event_type` | `str` | - | Determines color, display format, and aggregation strategy |
| `env_id` | `int \| None` | `None` | Right-justified env ID display; None for global events |
| `message` | `str` | - | Generic message (not currently displayed; metadata used instead) |
| `metadata` | `dict[str, str \| int \| float]` | `{}` | Rich inline metadata for individual events |

### Metadata Fields by Event Type

#### SEED_GERMINATED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `slot_id` | `str` | Displayed with cyan style |
| `blueprint` | `str` | Truncated to 10 chars, displayed after "GERM" |

#### SEED_STAGE_CHANGED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `slot_id` | `str` | Displayed with cyan style |
| `from` | `str` | Source stage (mapped to short form via `_STAGE_SHORT`) |
| `to` | `str` | Target stage (mapped to short form via `_STAGE_SHORT`) |

#### SEED_GATE_EVALUATED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `slot_id` | `str` | Displayed with cyan style |
| `gate` | `str` | Gate name (dim style) |
| `target_stage` | `str` | Target stage (mapped via `_STAGE_SHORT`) |
| `result` | `str` | "PASS" (green checkmark), "FAIL" (red X), or value (yellow) |
| `failed_checks` | `int` | Count of failed checks (displayed if > 0) |

#### SEED_FOSSILIZED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `slot_id` | `str` | Displayed with cyan style |
| `improvement` | `float` | Displayed as "+X.X%" in green |

#### SEED_PRUNED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `slot_id` | `str` | Displayed with cyan style |
| `reason` | `str` | Prune reason, truncated to 8 chars (dim style) |

#### PPO_UPDATE_COMPLETED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `entropy` | `float` | Displayed as "ent:X.XX" (cyan) |
| `clip_fraction` | `float` | Displayed as "clip:XX%" (yellow if > 0.2, green otherwise) |

#### BATCH_EPOCH_COMPLETED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| `batch` | `int \| str` | Batch number displayed as "#X" (cyan) |
| `episodes` | `int` | Episode count displayed as "+Xeps" (dim) |

#### TRAINING_STARTED
| Field | Type | Usage |
|-------|------|-------|
| `event_id` | `str` | Deduplication key |
| (no other fields) | - | Displays static "TRAINING STARTED" text |

---

## Thresholds and Color Coding

### Event Type Colors (`_EVENT_COLORS`)

| Event Type | Color | Meaning |
|------------|-------|---------|
| `SEED_GERMINATED` | `bright_yellow` | Seed activation - noteworthy lifecycle event |
| `SEED_STAGE_CHANGED` | `bright_white` | Stage transition - informational |
| `SEED_GATE_EVALUATED` | `bright_white` | Gate check - informational |
| `SEED_FOSSILIZED` | `bright_green` | Success - permanent integration |
| `SEED_PRUNED` | `bright_red` | Failure - seed removed |
| `PPO_UPDATE_COMPLETED` | `bright_magenta` | Training step - distinct from seed events |
| `REWARD_COMPUTED` | `dim` | High-frequency, aggregated |
| `TRAINING_STARTED` | `bright_green` | Session start marker |
| `EPOCH_COMPLETED` | `bright_blue` | High-frequency, aggregated |
| `BATCH_EPOCH_COMPLETED` | `bright_blue` | Batch milestone |

### Metric Thresholds

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| PPO clip_fraction | > 0.2 | `yellow` | High clipping rate warning |
| PPO clip_fraction | <= 0.2 | `green` | Normal clipping rate |

### Display Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_MAX_ENVS_SHOWN` | 3 | Max env IDs before truncating with "+" |
| `_DRIP_TICK_SECONDS` | 0.1 | Drip-feed tick interval |
| Default `max_lines` | 41 | Maximum visible log lines |
| Default `buffer_seconds` | 1.0 | Delay before displaying individual events |
| Default `max_delay_seconds` | 2.0 | Maximum wall-clock delay for events |
| `_max_processed_ids` | 10,000+ | Bounded dedup set size |

### Stage Abbreviations (`_STAGE_SHORT`)

| Full Stage | Short Form |
|------------|------------|
| DORMANT | DORM |
| GERMINATED | GERM |
| TRAINING | TRAIN |
| HOLDING | HOLD |
| BLENDING | BLEND |
| FOSSILIZED | FOSS |
| PRUNED | PRUNE |
| EMBARGOED | EMBG |
| RESETTING | RESET |

---

## Rendering Logic

### Event Classification

Events are classified into two categories:

1. **Individual Events** (`_INDIVIDUAL_EVENTS`): Displayed with full metadata, drip-fed for smooth scrolling
   - `SEED_GERMINATED`, `SEED_STAGE_CHANGED`, `SEED_GATE_EVALUATED`
   - `SEED_FOSSILIZED`, `SEED_PRUNED`
   - `PPO_UPDATE_COMPLETED`, `BATCH_EPOCH_COMPLETED`, `TRAINING_STARTED`

2. **Aggregated Events**: Grouped by timestamp + event_type, show count and env IDs
   - `REWARD_COMPUTED`, `EPOCH_COMPLETED`
   - Any event not in `_INDIVIDUAL_EVENTS`

### Drip-Feed Mechanism

The widget implements a smooth scrolling system:

1. **Queueing**: Individual events are queued with `ready_ts = now + buffer_seconds`
2. **Pacing**: Events are released at the observed events/sec rate
3. **Catch-up**: If oldest event exceeds `max_delay_seconds`, batch release occurs
4. **Budget**: Emission rate is capped at 3 events per tick with max budget of 15

### Timestamp Display

- **First line**: Always shows full `HH:MM:SS`
- **New minute**: Shows full `HH:MM:SS`
- **Same minute**: Shows abbreviated `:SS` with padding for alignment

### Border Title

Dynamic title showing:
- Total events received (formatted with commas)
- Events per second rate (5-second rolling window)
- Scroll indicator if lines were trimmed (e.g., "up-arrow 42")

Example: `EVENTS 1,234 (45.2/s) [click for detail] up-arrow 42`

### Click Handling

Clicking anywhere on the widget posts a `DetailRequested` message containing the full `event_log` list for modal display.

---

## Data Flow

```
SanctumSnapshot
    |
    +-- total_events_received --> _update_event_rate() --> events/sec calculation
    |                                                          |
    |                                                          v
    |                                                   _update_border_title()
    |
    +-- event_log: list[EventLogEntry]
            |
            +-- Individual events --> _pending_individual queue --> _drip_tick()
            |                                                           |
            |                                                           v
            |                                              _format_individual_event()
            |                                                           |
            +-- Aggregated events --> _aggregate_line_index dict        |
                        |                                               |
                        v                                               v
                _format_aggregate_event()                          _line_data: list[_LineData]
                        |                                               |
                        +-----------------------------------------------+
                                                |
                                                v
                                    _sort_and_trim_lines()
                                                |
                                                v
                                           render()
                                                |
                                                v
                                    Group of Text lines with:
                                    - Timestamp (full or abbreviated)
                                    - Event content (colored)
                                    - Right-justified env IDs
```

---

## Internal State

| Field | Type | Purpose |
|-------|------|---------|
| `_line_data` | `list[_LineData]` | Append-only line buffer |
| `_processed_ids` | `set[str]` | Event ID deduplication |
| `_processed_id_order` | `deque[str]` | LRU order for bounded dedup |
| `_aggregate_line_index` | `dict[tuple[str, str], int]` | (timestamp, event_type) to line index |
| `_snapshot` | `SanctumSnapshot \| None` | Retained for click handler |
| `_trimmed_count` | `int` | Cumulative trimmed lines count |
| `_pending_individual` | `deque[_QueuedEvent]` | Drip-feed queue |
| `_drip_budget` | `float` | Token bucket for emission pacing |
| `_eps_samples` | `deque[tuple[float, int]]` | Throughput samples for rate calculation |
| `_events_per_second` | `float` | Computed events/sec rate |
