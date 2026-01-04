# Telemetry Audit: AnomalyStrip

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/anomaly_strip.py`
**Purpose:** Single-line widget that surfaces system problems automatically, displaying either "ALL CLEAR" when healthy or a count of anomalies with color-coded severity when issues are detected.

---

## Telemetry Fields Consumed

### Source: EnvState (via snapshot.envs)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `status` | `str` | `"initializing"` | Checked for `"stalled"` and `"degraded"` values to count problematic environments |
| `seeds` | `dict[str, SeedState]` | `{}` | Iterated to access per-seed gradient health flags |

### Source: SeedState (via env.seeds.values())

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `has_exploding` | `bool` | `False` | Counted as gradient issue when True |
| `has_vanishing` | `bool` | `False` | Counted as gradient issue when True |

### Source: TamiyoState (via snapshot.tamiyo)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `entropy_collapsed` | `bool` | `False` | Sets `ppo_issues=True` when entropy has collapsed |
| `kl_divergence` | `float` | `0.0` | Sets `ppo_issues=True` when value exceeds 0.05 threshold |
| `dead_layers` | `int` | `0` | Displayed directly as network-level gradient issue count |
| `exploding_layers` | `int` | `0` | Displayed directly as network-level gradient issue count |
| `nan_grad_count` | `int` | `0` | Displayed directly as NaN gradient count |

### Source: SystemVitals (via snapshot.vitals)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `has_memory_alarm` | `property -> bool` | Computed | Sets `memory_alarm=True` when any device exceeds 90% memory usage |

---

## Thresholds and Color Coding

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Environment stalled | `status == "stalled"` | `yellow` | Environment has not improved for >10 epochs |
| Environment degraded | `status == "degraded"` | `red` | Environment accuracy dropped >1% |
| Seed gradient issues | `has_exploding or has_vanishing` | `red` | Per-seed gradient health problems |
| NaN gradients | `nan_grad_count > 0` | `red` | Network has NaN gradients (critical) |
| Exploding layers | `exploding_layers > 0` | `red` | Network layers with exploding gradients |
| Dead layers | `dead_layers > 0` | `yellow` | Network layers with vanishing gradients |
| PPO entropy collapsed | `entropy_collapsed == True` | `yellow` | Policy entropy has collapsed |
| PPO KL divergence | `kl_divergence > 0.05` | `yellow` | KL divergence exceeds safe threshold |
| Memory alarm | `has_memory_alarm == True` | `red` | Any device (RAM or GPU) exceeds 90% memory usage |

---

## Rendering Logic

The widget computes anomalies in `_compute_anomalies()` by scanning all environments and the global Tamiyo/vitals state:

1. **Environment scan**: Iterates `snapshot.envs.values()` counting stalled/degraded status and per-seed gradient issues
2. **PPO health check**: Examines `tamiyo.entropy_collapsed` and `tamiyo.kl_divergence > 0.05`
3. **Network gradient health**: Reads `tamiyo.dead_layers`, `tamiyo.exploding_layers`, and `tamiyo.nan_grad_count`
4. **Memory check**: Uses `vitals.has_memory_alarm` property

**Display behavior**:
- When `has_anomalies` is `False`: Displays `"ALL CLEAR"` in bold green
- When `has_anomalies` is `True`: Displays `"ANOMALIES: "` prefix in bold red, followed by pipe-separated issue summaries

**Issue display format** (in order of severity):
- Stalled envs: `"{count} stalled"` in yellow
- Degraded envs: `"{count} degraded"` in red
- Seed gradient issues: `"{count} seed grad(s)"` in red
- NaN gradients: `"{count} NaN grads"` in red
- Exploding layers: `"{count} exploding"` in red
- Dead layers: `"{count} dead layers"` in yellow
- PPO issues: `"PPO"` with warning icon in yellow
- Memory alarm: `"MEM"` with warning icon in red

The widget also applies a `has-anomalies` CSS class for visual styling when problems are detected.
