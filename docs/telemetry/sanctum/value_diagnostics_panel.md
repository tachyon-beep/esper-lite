# Telemetry Audit: ValueDiagnosticsPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/value_diagnostics_panel.py`

**Purpose:** Display value function quality diagnostics for DRL training health. Per DRL expert review, value function quality is THE primary diagnostic for RL training failures - low V-Return correlation means advantage estimates are garbage regardless of how healthy policy metrics look.

---

## Telemetry Fields Consumed

### Source: `SanctumSnapshot.tamiyo.value_function` (ValueFunctionMetrics)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `v_return_correlation` | `float` | `0.0` | Pearson correlation between V(s) predictions and actual returns |
| `td_error_mean` | `float` | `0.0` | Mean TD error: r + gamma*V(s') - V(s) |
| `td_error_std` | `float` | `0.0` | Standard deviation of TD errors |
| `bellman_error` | `float` | `0.0` | Squared Bellman error: \|V(s) - (r + gamma*V(s'))\|^2 |
| `return_p10` | `float` | `0.0` | 10th percentile of episode returns |
| `return_p50` | `float` | `0.0` | 50th percentile (median) of episode returns |
| `return_p90` | `float` | `0.0` | 90th percentile of episode returns |
| `return_variance` | `float` | `0.0` | Variance of return distribution |
| `return_skewness` | `float` | `0.0` | Skewness of return distribution |

### Source: `SanctumSnapshot.tamiyo.infrastructure` (InfrastructureMetrics)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compile_enabled` | `bool` | `False` | Whether torch.compile is active |
| `compile_backend` | `str` | `""` | Compile backend (e.g., "inductor") |
| `compile_mode` | `str` | `""` | Compile mode (e.g., "reduce-overhead", "max-autotune") |

---

## Field Paths (Full Access Pattern)

```python
# Value function metrics
snapshot.tamiyo.value_function.v_return_correlation
snapshot.tamiyo.value_function.td_error_mean
snapshot.tamiyo.value_function.td_error_std
snapshot.tamiyo.value_function.bellman_error
snapshot.tamiyo.value_function.return_p10
snapshot.tamiyo.value_function.return_p50
snapshot.tamiyo.value_function.return_p90
snapshot.tamiyo.value_function.return_variance
snapshot.tamiyo.value_function.return_skewness

# Infrastructure metrics
snapshot.tamiyo.infrastructure.compile_enabled
snapshot.tamiyo.infrastructure.compile_backend
snapshot.tamiyo.infrastructure.compile_mode
```

---

## Rendering Logic

### Panel Layout (5 lines, two sub-columns)

```
V-Corr     0.87^     Bellman      12.3
TD Mean    +2.4      TD Std        8.1
Returns    p10:-12   p50:+34   p90:+78 !
Ret sigma  14.2      Skew        +0.3
Compile    inductor:reduce-overhead [check]
```

### Line-by-Line Rendering

| Line | Left Column | Right Column |
|------|-------------|--------------|
| 1 | V-Return Correlation with trend icon | Bellman Error |
| 2 | TD Error Mean (signed) | TD Error Std |
| 3 | Return Percentiles (p10/p50/p90 with spread warning) | - |
| 4 | Return Std Dev (sqrt of variance) | Return Skewness |
| 5 | Compile status (full width) | - |

---

## Thresholds and Color Coding

### V-Return Correlation (`_get_correlation_style`)

| Threshold | Style | Icon | Meaning |
|-----------|-------|------|---------|
| `>= 0.8` | `green bold` | `^` (up arrow) | Excellent - value network well-calibrated |
| `>= 0.5` | `green` | `->` (right arrow) | Good - learning effectively |
| `>= 0.3` | `yellow` | `->` (right arrow) | Warning - advantage estimates may be noisy |
| `< 0.3` | `red bold` | `v` (down arrow) | Critical - value network not learning |

**Rationale:** Low correlation means advantage estimates are garbage, regardless of how policy metrics look.

### TD Error Mean (`_get_td_error_style`)

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `abs(mean) < 5` | `green` | Healthy - unbiased value estimates |
| `abs(mean) < 15` | `yellow` | Warning - possible target network staleness |
| `abs(mean) >= 15` | `red bold` | Critical - severely biased value estimates |

**Rationale:** High mean TD error indicates biased value estimates, often from stale target networks.

### Bellman Error (`_get_bellman_style`)

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `< 20` | `green` | Healthy |
| `< 50` | `yellow` | Warning - monitor for NaN |
| `>= 50` | `red bold` | Critical - often precedes NaN losses |

**Rationale:** Bellman error spikes are an early warning signal that often precede NaN losses.

### Return Percentiles (Inline styling)

| Condition | Style |
|-----------|-------|
| `p10 < 0` | `red` |
| `p10 >= 0` | `green` |
| `p50 < 0` | `red` |
| `p50 >= 0` | `green` |
| `p90 < 0` | `red` |
| `p90 >= 0` | `green` |

**Spread Warning:**
| Condition | Indicator |
|-----------|-----------|
| `(p90 - p10) > 50` | Yellow warning icon displayed |

**Rationale:** Large p90-p10 spread indicates bimodal policy (some episodes succeed, others fail catastrophically).

### Return Variance (displayed as std dev)

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `variance > 100` | `yellow` | High - inconsistent policy |
| `variance <= 100` | `cyan` | Normal |

**Note:** Displayed as `sqrt(variance)` for intuitive interpretation.

### Return Skewness (`_get_skewness_style`)

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `abs(skew) < 1.0` | `cyan` | Healthy - symmetric distribution |
| `abs(skew) < 2.0` | `yellow` | Warning - moderately asymmetric |
| `abs(skew) >= 2.0` | `red bold` | Critical - severely skewed, inconsistent policy |

**Rationale:**
- Positive skew: Few big wins (right tail)
- Negative skew: Few big losses (left tail)
- High absolute skew indicates policy inconsistency.

### Compile Status (Line 5)

| Condition | Display | Style |
|-----------|---------|-------|
| `compile_enabled=True` | `{backend}:{mode} [check]` | `green` / `green bold` |
| `compile_enabled=False` | `EAGER (3-5x slower)` | `red bold reverse` / `dim` |

**Rationale:** Per PyTorch expert, fallback to eager mode causes 3-5x performance degradation and should be immediately visible.

---

## Column Layout Constants

```python
LABEL_W = 10  # Label width (characters)
VAL_W = 8     # Value width (characters)
GAP = 2       # Gap between sub-columns
```

---

## Data Flow

1. `SanctumSnapshot` passed to `update_snapshot()`
2. Panel stores reference in `self._snapshot`
3. `render()` called on refresh
4. Accesses `snapshot.tamiyo.value_function` for value metrics
5. Accesses `snapshot.tamiyo.infrastructure` for compile status
6. Applies threshold-based styling via helper methods
7. Returns Rich `Text` object for display

---

## Missing Data Handling

When `self._snapshot is None`:
- Returns `Text("[no data]", style="dim")`

---

## Schema Definitions

### ValueFunctionMetrics (from schema.py lines 721-762)

```python
@dataclass
class ValueFunctionMetrics:
    v_return_correlation: float = 0.0
    td_error_mean: float = 0.0
    td_error_std: float = 0.0
    bellman_error: float = 0.0
    return_p10: float = 0.0
    return_p50: float = 0.0
    return_p90: float = 0.0
    return_skewness: float = 0.0
    return_variance: float = 0.0
    # Historical tracking (not used by this panel)
    value_predictions: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    actual_returns: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    td_errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))
```

### InfrastructureMetrics (from schema.py lines 796-820)

```python
@dataclass
class InfrastructureMetrics:
    cuda_memory_allocated_gb: float = 0.0
    cuda_memory_reserved_gb: float = 0.0
    cuda_memory_peak_gb: float = 0.0
    cuda_memory_fragmentation: float = 0.0
    compile_enabled: bool = False
    compile_backend: str = ""
    compile_mode: str = ""
```

---

## Expert Review Notes

From docstring:
- **DRL Expert:** Value function quality is THE primary diagnostic for RL training failures. Low V-Return correlation means advantage estimates are garbage, regardless of how healthy policy metrics look.
- **PyTorch Expert:** Compile status should be visible because fallback to eager mode results in 3-5x slower execution.
