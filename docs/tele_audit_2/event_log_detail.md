# Telemetry Audit: EventLogDetail

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/event_log_detail.py`
**Purpose:** Modal screen that displays raw, unformatted event log entries in a table format for debugging and detailed inspection.

---

## Telemetry Fields Consumed

### Source: EventLogEntry (via list passed to constructor)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `timestamp` | `str` | N/A (required) | Displayed in "Time" column (formatted as HH:MM:SS) |
| `event_type` | `str` | N/A (required) | Displayed in "Type" column |
| `env_id` | `int \| None` | `None` | Displayed in "Env" column; formatted as 2-digit number or "--" for global events |
| `episode` | `int` | `0` | Displayed in "Ep" column |
| `message` | `str` | N/A (required) | Displayed in "Message" column |
| `metadata` | `dict[str, str \| int \| float]` | `{}` | Displayed in "Details" column as space-separated key=value pairs |

**Note:** The `relative_time` field from `EventLogEntry` is NOT consumed by this widget.

---

## Thresholds and Color Coding

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| N/A | N/A | N/A | No threshold-based color coding |

### Static Column Styling

| Column | Style | Notes |
|--------|-------|-------|
| Time | `dim` | Subdued timestamp display |
| Type | `bright_white` | Event type stands out |
| Env | `bright_blue` | Environment ID in blue |
| Ep | `dim` | Episode number subdued |
| Message | `white` | Standard text |
| Details | `cyan` | Metadata key=value pairs |

---

## Rendering Logic

### Event Display Order
- Events are displayed in **reverse chronological order** (newest first)
- Uses `reversed(self._events)` to iterate from most recent

### Environment ID Formatting
- If `env_id` is not `None`: formatted as 2-digit zero-padded number (e.g., `"00"`, `"01"`)
- If `env_id` is `None`: displayed as `"--"` (using `GLOBAL_EVENT_ENV_DISPLAY` constant)
- Global events (PPO updates, batch events) have `None` env_id

### Metadata Formatting
- Metadata dict is converted to space-separated `key=value` pairs
- Empty metadata renders as empty string
- Example: `slot=r0c0 reward=0.5 blueprint=conv_light`

### Modal Behavior
- Dismissed via ESC key, Q key, or clicking anywhere
- Header shows event count: "RAW EVENT LOG  (N events)"

---

## Data Flow

```
SanctumSnapshot.event_log (list[EventLogEntry])
        |
        v
EventLogDetail.__init__(events)
        |
        v
_render_events() iterates reversed(events)
        |
        v
Rich Table with 6 columns
```

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `GLOBAL_EVENT_ENV_DISPLAY` | `"--"` | Display value for events without a specific env_id |
