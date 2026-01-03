# Telemetry Audit: RunHeader

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/run_header.py`
**Purpose:** Single-line status bar showing essential training state at a glance, including connection status, thread health, run progress, runtime, throughput, and system alarms.

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `connected` | `bool` | `False` | Connection status indicator (LIVE/SLOW/STALE/Disconnected) |
| `staleness_seconds` | `float` | `inf` | Staleness threshold checks for connection status styling |
| `training_thread_alive` | `bool \| None` | `None` | Thread health indicator (checkmark/X/question mark) |
| `task_name` | `str` | `""` | Run name display (truncated to 14 chars) |
| `current_episode` | `int` | `0` | Episode number display ("Ep 47") |
| `current_epoch` | `int` | `0` | Epoch progress bar numerator |
| `max_epochs` | `int` | `0` | Epoch progress bar denominator (0 = unbounded) |
| `current_batch` | `int` | `0` | Batch progress bar numerator |
| `max_batches` | `int` | `0` | Batch progress bar denominator |
| `runtime_seconds` | `float` | `0.0` | Elapsed time display via `format_runtime()` |
| `vitals` | `SystemVitals` | `SystemVitals()` | Nested vitals for throughput and alarm metrics |

### Source: SystemVitals (via `snapshot.vitals`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `epochs_per_second` | `float` | `0.0` | Throughput display ("0.8e/s") |
| `batches_per_hour` | `float` | `0.0` | Throughput display, converted to per-minute ("2.1b/m") |
| `cpu_percent` | `float \| None` | `0.0` | CPU alarm check (>90% triggers alarm) |
| `ram_used_gb` | `float \| None` | `0.0` | RAM alarm percentage calculation |
| `ram_total_gb` | `float \| None` | `0.0` | RAM alarm percentage calculation |
| `gpu_stats` | `dict[int, GPUStats]` | `{}` | Per-GPU memory alarm checks |
| `gpu_memory_used_gb` | `float` | `0.0` | Fallback single-GPU memory alarm |
| `gpu_memory_total_gb` | `float` | `0.0` | Fallback single-GPU memory alarm |
| `has_memory_alarm` | `bool` (property) | `False` | Quick check for any device >90% memory |
| `memory_alarm_devices` | `list[str]` (property) | `[]` | List of devices exceeding 90% memory threshold |

### Source: GPUStats (via `vitals.gpu_stats[device_id]`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `memory_used_gb` | `float` | `0.0` | GPU memory alarm percentage calculation |
| `memory_total_gb` | `float` | `0.0` | GPU memory alarm percentage calculation |

---

## Thresholds and Color Coding

### Connection Status

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `staleness_seconds` | < 2.0s | `green` | LIVE - receiving fresh data |
| `staleness_seconds` | 2.0s - 5.0s | `yellow` | SLOW - data arriving slowly |
| `staleness_seconds` | >= 5.0s | `red` | STALE - data is stale |
| `connected` | `False` | `red` | Disconnected from training process |
| No snapshot | n/a | `dim` | Waiting for initial data |

### Thread Health

| Metric | Value | Style | Meaning |
|--------|-------|-------|---------|
| `training_thread_alive` | `True` | `green` | Thread running normally |
| `training_thread_alive` | `False` | `bold red` | Thread dead - critical failure |
| `training_thread_alive` | `None` | `dim` | Unknown thread status |

### System Alarms

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `cpu_percent` | > 90% | `bold red` | CPU utilization alarm |
| RAM usage | > 90% | `bold red` | RAM memory alarm |
| GPU memory | > 90% | `bold red` | CUDA memory alarm |
| No alarms | n/a | `green dim` | System healthy |

---

## Rendering Logic

### Layout Structure
The header renders as a single line with fixed-width segments separated by vertical bars:

```
[Connection] [Thread] | [Run Name] | [Episode] [Epoch Progress] [Batch Progress] | [Runtime] | [Throughput] | [System] [UI Hz]
```

Example:
```
● LIVE ✓ | my_experiment  | Ep  47 ████████░░░░░░░░ 150/500 B:████░░░░ 25/100 |  1h 23m | 0.8e/s  2.1b/m | ✓ System @10.0Hz
```

### Segment Details

1. **Connection Status (6 chars)**: Icon + 4-char status word
   - Icons: `o` (filled = good), `o` (half = warning), `o` (empty = bad)
   - Status: LIVE, SLOW, STAL, DISC, WAIT

2. **Thread Health (1 char)**: Checkmark/X/question mark

3. **Run Name (14 chars fixed)**: Truncated with ellipsis if > 13 chars

4. **Episode**: "Ep " prefix + right-aligned 3-digit episode number

5. **Epoch Progress**: Visual bar (16 chars) + fraction
   - Uses `█` for filled and `░` for empty portions
   - Shows "(unbounded)" if `max_epochs <= 0`

6. **Batch Progress**: "B:" prefix + bar (8 chars) + fraction
   - Styled in magenta
   - Shows just "B:{current}" if `max_batches <= 0`

7. **Runtime (7 chars right-aligned)**: Formatted via `format_runtime()`
   - Examples: "1h 23m", "5m 30s", "45s", "--"

8. **Throughput (15 chars)**: "{eps}e/s {bpm}b/m"
   - `epochs_per_second` formatted as "X.Xe/s"
   - `batches_per_hour / 60` formatted as "X.Xb/m"

9. **System Alarm Indicator**:
   - If alarm active: Warning icon + device percentages
   - If no alarms: Checkmark + "System"

10. **UI Tick Rate**: "@{hz}Hz" for debugging refresh rate

### Data Flow

1. `update_snapshot()` receives new `SanctumSnapshot` from aggregator
2. Tracks UI tick times in deque for Hz calculation
3. Calls `refresh()` to trigger re-render
4. `render()` builds `Rich.Text` object by:
   - Calling `_get_connection_status()` for connection segment
   - Calling `_render_progress_bar()` for epoch progress
   - Calling `_render_batch_progress()` for batch progress
   - Calling `format_runtime()` for runtime display
   - Calling `_get_system_alarm_indicator()` for alarm segment
   - Checking `has_system_alarm` property for alarm styling

### Null Handling

- If `_snapshot` is `None`: Returns "o WAIT   | Waiting for training data..." in dim style
- All other fields have defaults in schema that prevent None checks

### Internal State

- `_snapshot`: Cached `SanctumSnapshot` for rendering
- `_ui_tick_times`: `deque[float]` (maxlen=32) for Hz calculation
- `_ui_hz`: Computed UI refresh rate for diagnostics
