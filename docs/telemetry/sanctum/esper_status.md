# Telemetry Audit: EsperStatus

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/esper_status.py`
**Purpose:** Displays system vitals and performance metrics including seed stage counts, host network params, throughput, runtime, GPU/RAM usage, and CPU percentage.

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `runtime_seconds` | `float` | `0.0` | Displayed as formatted runtime (Xh Ym Zs) |
| `envs` | `dict[int, EnvState]` | `{}` | Iterated to aggregate seed stage counts across all environments |

### Source: EnvState (via `snapshot.envs[env_id]`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `seeds` | `dict[str, SeedState]` | `{}` | Iterated to count non-DORMANT seeds by stage |

### Source: SeedState (via `env.seeds[slot_id]`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `stage` | `str` | `"DORMANT"` | Aggregated into stage counts (excludes DORMANT from display) |

### Source: SystemVitals (via `snapshot.vitals`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `host_params` | `int` | `0` | Displayed as M/K/raw format (e.g., "1.5M", "500K") |
| `epochs_per_second` | `float` | `0.0` | Displayed with 2 decimal places |
| `batches_per_hour` | `float` | `0.0` | Displayed with 0 decimal places |
| `gpu_stats` | `dict[int, GPUStats]` | `{}` | Multi-GPU memory and utilization display |
| `gpu_memory_used_gb` | `float` | `0.0` | Legacy single-GPU fallback |
| `gpu_memory_total_gb` | `float` | `0.0` | Legacy single-GPU fallback |
| `gpu_utilization` | `float` | `0.0` | Legacy single-GPU fallback |
| `ram_used_gb` | `float \| None` | `0.0` | RAM usage in GB |
| `ram_total_gb` | `float \| None` | `0.0` | Total RAM in GB |
| `cpu_percent` | `float \| None` | `0.0` | CPU utilization percentage |

### Source: GPUStats (via `vitals.gpu_stats[device_id]`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `memory_used_gb` | `float` | `0.0` | GPU memory usage in GB |
| `memory_total_gb` | `float` | `0.0` | Total GPU memory in GB |
| `utilization` | `float` | `0.0` | GPU compute utilization percentage |

### Source: leyline constants

| Constant | Type | Description |
|----------|------|-------------|
| `STAGE_ABBREVIATIONS` | `dict[str, str]` | Maps stage names to 4-5 char abbreviations (e.g., "TRAINING" -> "Train") |
| `STAGE_COLORS` | `dict[str, str]` | Maps stage names to Rich styles (e.g., "TRAINING" -> "cyan") |

---

## Thresholds and Color Coding

### GPU/RAM Memory Usage

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| GPU Memory | < 75% | `green` | Healthy memory usage |
| GPU Memory | 75-90% | `yellow` | Memory pressure, monitor closely |
| GPU Memory | > 90% | `red` | Critical memory usage |
| RAM | < 75% | `dim` | Normal memory usage |
| RAM | 75-90% | `yellow` | Memory pressure |
| RAM | > 90% | `red` | Critical memory usage |

### GPU Utilization

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| GPU Utilization | < 80% | `green` | Normal utilization |
| GPU Utilization | 80-95% | `yellow` | High utilization |
| GPU Utilization | > 95% | `red` | Saturated GPU |

### Seed Stage Colors

| Stage | Abbreviation | Style |
|-------|--------------|-------|
| DORMANT | Dorm | `dim` |
| GERMINATED | Germ | `bright_blue` |
| TRAINING | Train | `cyan` |
| HOLDING | Hold | `magenta` |
| BLENDING | Blend | `yellow` |
| FOSSILIZED | Foss | `green` |
| PRUNED | Prune | `red` |
| EMBARGOED | Embar | `bright_red` |
| RESETTING | Reset | `dim` |

---

## Rendering Logic

### Panel Structure
The widget renders a Rich Panel titled "ESPER STATUS" with `cyan` border. Content is a two-column table (Metric, Value) with no headers.

### Seed Stage Counts
1. Iterates all environments and their seeds
2. Counts seeds by stage (excluding DORMANT)
3. Displays each non-zero stage with abbreviated name and count
4. Colors each count using `STAGE_COLORS` lookup

### Host Network Params
- Values >= 1,000,000: formatted as "X.XM"
- Values >= 1,000: formatted as "XK"
- Values < 1,000: displayed raw
- Zero values: displayed as "--" (dim)

### Throughput
- `epochs_per_second`: formatted to 2 decimal places
- `batches_per_hour`: formatted to 0 decimal places (integer)

### Runtime
- Computed from `runtime_seconds`
- Formatted as "Xh Ym Zs"
- Zero values: displayed as "-"

### GPU Stats (Multi-GPU Support)
1. If `gpu_stats` dict is populated:
   - Iterates sorted by device_id
   - For each GPU with `memory_total_gb > 0`:
     - Calculates memory percentage
     - Displays memory as "X.X/X.XGB" with color
     - If `utilization > 0`, displays "X%" with color
   - Label is "GPU:" for single GPU, "GPUX:" for multi-GPU
2. Fallback to legacy single-GPU fields if `gpu_stats` empty but `gpu_memory_total_gb > 0`
3. If no GPU data: displays "GPU: -"

### RAM Display
- If `ram_total_gb` and `ram_used_gb` are valid and positive:
  - Displays as "X.X/X.XGB" with color coding
- Otherwise: displays "--" (dim)

### CPU Display
- If `cpu_percent` is valid and > 0:
  - Displays as "X.X%"
- Otherwise: displays "--" (dim)

---

## Data Flow

```
SanctumSnapshot
    |
    +-- envs: dict[int, EnvState]
    |       |
    |       +-- seeds: dict[str, SeedState]
    |               |
    |               +-- stage: str  --> Aggregated into stage_counts
    |
    +-- runtime_seconds: float  --> Formatted as "Xh Ym Zs"
    |
    +-- vitals: SystemVitals
            |
            +-- host_params: int  --> Formatted as M/K/raw
            +-- epochs_per_second: float  --> "X.XX"
            +-- batches_per_hour: float  --> "X"
            +-- gpu_stats: dict[int, GPUStats]
            |       |
            |       +-- memory_used_gb, memory_total_gb  --> Color-coded
            |       +-- utilization  --> Color-coded
            |
            +-- gpu_memory_used_gb, gpu_memory_total_gb  --> Fallback
            +-- gpu_utilization  --> Fallback
            +-- ram_used_gb, ram_total_gb  --> Color-coded
            +-- cpu_percent  --> Plain display
```

---

## Notes

1. **DORMANT seeds excluded**: Stage counts only display non-DORMANT stages, reducing visual noise.

2. **Multi-GPU support**: Primary path uses `gpu_stats` dict with per-device metrics. Legacy single-GPU fields serve as fallback for backwards compatibility.

3. **CPU was previously collected but not displayed**: This was identified as a bug in the original TUI (line 33 comment).

4. **No data placeholders**: Empty/zero values consistently display as "--" (dim) or "-" rather than showing zeros.
