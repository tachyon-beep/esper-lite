# Simic Telemetry Subsystem: Deep RL Review

**Reviewer**: Claude (DRL Specialist)
**Date**: 2025-12-17
**Scope**: `/src/esper/simic/telemetry/` (9 files, ~2083 lines)
**Focus**: RL debugging utility, PPO pathology detection, actionability

---

## Executive Summary

The simic telemetry subsystem provides a solid foundation for PPO training diagnostics with well-designed gradient collection, anomaly detection, and LSTM health monitoring. The architecture follows good practices: async-safe GPU operations, phase-dependent thresholds, and automatic telemetry escalation on anomalies.

**Strengths:**
- Excellent GPU optimization: single sync points via `torch._foreach_norm` and tensor stacking
- Phase-dependent explained variance thresholds (warmup/early/mid/late) prevent false positives
- LSTM health monitoring catches hidden state explosion/vanishing
- Gradient EMA drift detection catches slow training degradation

**Key Gaps (addressed in detail below):**
1. **Missing entropy collapse detection** in `AnomalyDetector` despite Karn having it
2. **No KL divergence anomaly detection** - only emitted to telemetry, never checked
3. **Hard-coded gradient thresholds** (1e-7 vanishing, 100.0 exploding) lack PPO-specific tuning
4. **Missing advantage statistics** - critical for debugging policy gradient issues
5. **No ratio diagnostic integration** with anomaly escalation

---

## File-by-File Analysis

### 1. `gradient_collector.py` (538 lines)

**Purpose**: Async-safe gradient statistics for seed and host networks.

#### Positive Findings

**Lines 128-148: Excellent vectorization**
```python
per_param_norms = torch._foreach_norm(grads, ord=2)
all_norms = torch.stack(per_param_norms)
```
Uses PyTorch's fused kernel for batch norm computation, avoiding O(n) kernel launches.

**Lines 356-393: `DualGradientStats.normalized_ratio`**
Correctly normalizes by sqrt(param_count) to remove scale bias between host/seed networks. This is mathematically sound for comparing gradient "intensity" across different-sized networks.

**Lines 272-283: Single sync point**
```python
stats_tensor = torch.stack([...])
stats = stats_tensor.tolist()
```
All scalar metrics collected into one tensor for a single CPU-GPU sync - excellent for throughput.

#### Issues

**MEDIUM: Hard-coded thresholds not tuned for PPO**
**Lines 76-77, 199-200, 329-330**
```python
vanishing_threshold: float = 1e-7,
exploding_threshold: float = 100.0,
```

For PPO specifically:
- **Vanishing**: 1e-7 is aggressive. PPO with gradient clipping at 0.5 (default) often produces per-param norms of 1e-5 to 1e-3, which could trigger false "vanishing" detections on individual parameters.
- **Exploding**: 100.0 may be too lenient. With `max_grad_norm=0.5`, total clipped norm is bounded; individual param norms of 100 indicate severe imbalance that clipping can't fix.

**Recommendation**: Make thresholds relative to `max_grad_norm` or use percentile-based detection.

**LOW: Missing gradient histogram bins**
**Lines 197-324**: `collect_seed_gradients`

For debugging gradient flow issues, knowing the distribution shape is valuable. Currently only reports `min_layer_norm`, `max_layer_norm`, `zero_grad_fraction`. Consider adding:
- Gradient magnitude histogram bins (e.g., [<1e-6, 1e-6 to 1e-4, 1e-4 to 1e-2, >1e-2])
- Per-layer gradient direction cosine similarity (detects conflicting gradients between heads)

---

### 2. `anomaly_detector.py` (256 lines)

**Purpose**: Phase-dependent training anomaly detection for telemetry escalation.

#### Positive Findings

**Lines 45-84: Phase-dependent EV thresholds**
```python
ev_threshold_warmup: float = -0.5   # Allow anti-correlated (random init)
ev_threshold_early: float = -0.2    # Expect some learning
ev_threshold_mid: float = 0.0       # Expect positive correlation
ev_threshold_late: float = 0.1      # Expect useful predictions
```
This is exactly right for PPO. Early explained variance is legitimately low because the critic hasn't learned the value function yet. The progressive thresholds prevent false alarms during warmup while still catching late-stage critic failures.

**Lines 86-114: Ratio explosion/collapse detection**
PPO ratio bounds of 5.0 max and 0.1 min are reasonable. These correspond to roughly log-prob differences of +/-1.6 nats, which indicates severe policy drift.

#### Critical Issues

**CRITICAL: Missing entropy collapse detection**
**Lines 27-255 (entire file)**

The `AnomalyDetector` class checks:
- Ratio explosion/collapse (lines 86-114)
- Value function collapse (lines 116-151)
- Numerical stability (lines 153-180)
- Gradient drift (lines 182-215)

But it does NOT check entropy collapse, despite this being one of the most common PPO failure modes. The Karn subsystem has `PolicyAnomalyDetector.check_entropy_collapse()` in `triggers.py:256-265`, but the simic telemetry module's `AnomalyDetector` - which is used for automatic escalation in `vectorized.py` - lacks it entirely.

**Impact**: Entropy collapse (policy becoming deterministic) won't trigger debug telemetry escalation. The operator sees `entropy_collapsed: true` in PPO_UPDATE_COMPLETED events but no anomaly escalation occurs.

**Recommendation**: Add entropy threshold check to `AnomalyDetector.check_all()`:
```python
def check_entropy_collapse(
    self,
    entropy: float,
    entropy_threshold: float = 0.1,
) -> AnomalyReport:
    report = AnomalyReport()
    if entropy < entropy_threshold:
        report.add_anomaly(
            "entropy_collapse",
            f"entropy={entropy:.4f} < {entropy_threshold}",
        )
    return report
```

**HIGH: Missing KL divergence anomaly detection**
**Lines 217-253: `check_all` method**

The `check_all` aggregator doesn't include KL divergence, even though:
1. `emit_ppo_update_event` (emitters.py:202) captures `kl_divergence`
2. PPO's target_kl early stopping (ppo.py:554) uses 1.5x threshold
3. Large KL indicates the policy changed too much in one update

KL spikes are a leading indicator of training instability that precedes ratio explosion. Detecting them early could prevent catastrophic policy updates.

**Recommendation**: Add KL check with threshold around 0.05-0.1 (before early stopping kicks in at 0.015 * 1.5 = 0.0225):
```python
def check_kl_divergence(
    self,
    approx_kl: float,
    kl_warning_threshold: float = 0.05,
) -> AnomalyReport:
    report = AnomalyReport()
    if approx_kl > kl_warning_threshold:
        report.add_anomaly(
            "kl_spike",
            f"approx_kl={approx_kl:.4f} > {kl_warning_threshold}",
        )
    return report
```

---

### 3. `emitters.py` (450 lines)

**Purpose**: Pure functions for formatting and emitting telemetry events.

#### Positive Findings

**Lines 158-238: `emit_ppo_update_event`**
Comprehensive PPO health metrics:
- `kl_divergence`, `clip_fraction` - policy update quality
- `ratio_max`, `ratio_min`, `ratio_std` - early warning for collapse
- `explained_variance` - critic health
- `entropy_collapsed` flag - policy determinism check
- Per-head entropy and gradient norms (P3-1, P4-6)

This is excellent coverage for the PPO diagnostics that matter.

**Lines 356-407: `check_performance_degradation`**
Correctly skips warmup phase (first 10%) to avoid false positives from normal exploration variance.

#### Issues

**MEDIUM: `entropy_collapsed` threshold is hard-coded**
**Line 227**
```python
"entropy_collapsed": metrics.get("entropy", 1.0) < 0.1,
```

The 0.1 threshold is reasonable for normalized entropy but:
1. Should be configurable via `TelemetryConfig`
2. Should match the threshold used in anomaly detection (once added)
3. Different action spaces may warrant different thresholds

**MEDIUM: Missing advantage statistics**
**Lines 158-238: `emit_ppo_update_event`**

For debugging policy gradient issues, advantage statistics are critical:
- `advantage_mean`, `advantage_std` - detect advantage normalization issues
- `advantage_max`, `advantage_min` - detect outliers that dominate updates
- `returns_mean`, `returns_std` - GAE computation health

Currently only `explained_variance` captures value function quality, but advantage distribution shape affects policy gradient variance significantly.

**LOW: `ratio_diagnostic` not integrated with anomaly emission**
**Line 655-656 in ppo.py** (integration point):
```python
metrics.setdefault("ratio_diagnostic", []).append(diag.to_dict())
```

The `RatioExplosionDiagnostic` captures problematic transition indices, but:
1. It's only logged to metrics, not emitted as a separate telemetry event
2. `_emit_anomaly_diagnostics` in vectorized.py receives it but doesn't always include it

This diagnostic data is highly valuable for post-hoc analysis of what caused ratio explosion.

---

### 4. `debug_telemetry.py` (310 lines)

**Purpose**: Expensive per-layer diagnostics for "Oh Shit" mode.

#### Positive Findings

**Lines 42-123: `collect_per_layer_gradients`**
- Batches all per-layer stats into single GPU->CPU transfer
- Uses `correction=0` for std to handle single-element tensors
- Comprehensive health indicators: zero_fraction, small_fraction, large_fraction, nan/inf counts

**Lines 223-310: `RatioExplosionDiagnostic`**
Captures exactly the right information for debugging ratio explosion:
- `worst_ratio_indices` - which transitions caused problems
- `worst_ratio_values` - how bad they were
- `worst_ratio_actions` - what actions were affected
- `logit_diff_mean/max` - log-prob divergence magnitude

#### Issues

**LOW: Missing per-head breakdown in `RatioExplosionDiagnostic`**
**Lines 240-298: `from_batch`**

For factored action spaces, ratio explosion often affects specific heads (e.g., blueprint head is less frequently updated). Currently captures only the "op" head's ratios. Consider adding per-head ratio diagnostics.

**LOW: `NumericalStabilityReport` missing gradient statistics**
**Lines 126-220**

The report captures:
- NaN/Inf locations in weights and gradients
- Max weight and gradient magnitudes
- Loss finiteness

Missing: Gradient clipping statistics. When `max_grad_norm` clips gradients, knowing the pre-clip total norm helps diagnose whether clipping is actively limiting learning.

---

### 5. `lstm_health.py` (107 lines)

**Purpose**: LSTM hidden state health monitoring.

#### Positive Findings

**Lines 19-68: `LSTMHealthMetrics`**
Comprehensive LSTM diagnostics:
- `h_norm`, `c_norm` - overall magnitude
- `h_max`, `c_max` - worst-case spikes
- `has_nan`, `has_inf` - numerical stability

**Lines 36-57: `is_healthy` method**
Appropriate thresholds:
- `max_norm=100.0` for explosion detection
- `min_norm=1e-6` for vanishing detection

For LSTM policy networks in RL, these are reasonable defaults.

#### Issues

**LOW: Missing cell state statistics**
**Lines 71-104: `compute_lstm_health`**

LSTM cell state often shows issues before hidden state does (cell state has direct additive connections through forget gate). Consider adding:
- `c_mean` - detect systematic drift in cell state
- `gate_saturation` - fraction of forget/input gates near 0 or 1 (indicates training difficulty)

This would require access to the LSTM's gate activations, which may not be available without hooks.

---

### 6. `gradient_ema.py` (118 lines)

**Purpose**: Exponential moving average tracking for gradient drift detection.

#### Positive Findings

**Lines 19-77: `GradientEMATracker`**
Sound implementation:
- Drift indicator `|current - ema| / (ema + epsilon)` is scale-invariant
- Momentum of 0.99 is appropriate for slow drift detection
- State dict support for checkpointing

**Lines 78-99: `check_drift`**
Drift threshold of 0.5 (50% deviation from EMA) is reasonable for detecting training instability without being overly sensitive.

#### Issues

**LOW: No warm-up handling**
**Lines 39-61: `update` method**

The first update initializes EMA to current values and returns zero drift. This is correct, but for the first few updates, the EMA is unreliable because it hasn't accumulated enough history. Consider:
- Tracking `_update_count` (already exists, line 37)
- Suppressing drift warnings until `_update_count > warmup_steps` (e.g., 10)

---

### 7. `telemetry_config.py` (93 lines)

**Purpose**: Configuration for telemetry levels and auto-escalation.

#### Positive Findings

**Lines 22-55: `TelemetryConfig`**
Clean design:
- `TelemetryLevel` enum (OFF, MINIMAL, NORMAL, DEBUG)
- Auto-escalation on anomaly with configurable duration
- `effective_level` property that accounts for temporary escalation

**Lines 77-90: `escalate_temporarily`**
Simple epoch-countdown mechanism for temporary debug mode.

#### Issues

**LOW: No per-anomaly-type escalation duration**
**Lines 43-44**
```python
auto_escalate_on_anomaly: bool = True
anomaly_escalation_epochs: int = 5
```

All anomalies trigger the same escalation duration. Severe anomalies (numerical_instability) might warrant longer debug capture than mild ones (gradient_drift).

---

### 8. `profiler.py` (91 lines)

**Purpose**: torch.profiler integration for GPU bottleneck analysis.

#### Positive Findings

**Lines 29-91: `training_profiler`**
Standard torch.profiler setup with:
- Configurable wait/warmup/active schedule
- TensorBoard trace output
- Auto-detection of CUDA availability

#### Issues

**None significant.** This is a thin wrapper around torch.profiler with sensible defaults.

---

### 9. `__init__.py` (120 lines)

**Purpose**: Public API exports.

#### Positive Findings

Well-organized exports grouped by category:
- Telemetry config
- Debug telemetry
- Gradient collection
- Anomaly detection
- LSTM health
- Gradient EMA
- Profiler
- Emitters

No issues identified.

---

## Cross-Cutting Concerns

### Integration with Training Loop (`vectorized.py`)

**Lines 789-791** (vectorized.py):
```python
anomaly_detector = AnomalyDetector()
```

The anomaly detector is instantiated but:
1. `check_gradient_drift` is available but gradient EMA results aren't passed to it
2. `check_all` is called in `_emit_anomaly_diagnostics` but without entropy/KL data

**Observation**: The telemetry modules are well-designed but not fully wired into the training loop's anomaly detection flow.

### Missing RL-Specific Diagnostics

Based on common PPO debugging needs, the telemetry subsystem is missing:

1. **Advantage distribution statistics** - mean, std, min, max, skewness
2. **Return distribution statistics** - for detecting reward scale issues
3. **Per-head learning metrics** - which heads are learning vs. stuck
4. **Clip fraction per head** - for factored action spaces
5. **Value prediction error distribution** - not just explained variance

### Threshold Configuration

Many thresholds are hard-coded:
- `vanishing_threshold=1e-7` (gradient_collector.py)
- `exploding_threshold=100.0` (gradient_collector.py)
- `entropy_collapsed < 0.1` (emitters.py)
- `ev_threshold_*` (anomaly_detector.py)
- `drift_threshold=0.5` (gradient_ema.py)

Consider consolidating these into `TelemetryConfig` or a dedicated `PPOThresholds` dataclass for easier tuning.

---

## Summary of Issues

### Critical (Missing Core Diagnostics)

| ID | File | Line(s) | Issue |
|----|------|---------|-------|
| C1 | anomaly_detector.py | 27-255 | Missing entropy collapse detection in `AnomalyDetector` |
| C2 | anomaly_detector.py | 217-253 | Missing KL divergence anomaly check in `check_all` |

### High Priority (Detection Gaps)

| ID | File | Line(s) | Issue |
|----|------|---------|-------|
| H1 | emitters.py | 158-238 | Missing advantage statistics in PPO update telemetry |
| H2 | anomaly_detector.py | 86-114 | Ratio thresholds not integrated with entropy/KL |

### Medium Priority (Best Practices)

| ID | File | Line(s) | Issue |
|----|------|---------|-------|
| M1 | gradient_collector.py | 76-77 | Hard-coded vanishing/exploding thresholds not PPO-tuned |
| M2 | emitters.py | 227 | Hard-coded entropy collapse threshold |
| M3 | debug_telemetry.py | 126-220 | `NumericalStabilityReport` missing pre-clip gradient norms |

### Low Priority (Enhancements)

| ID | File | Line(s) | Issue |
|----|------|---------|-------|
| L1 | gradient_collector.py | 197-324 | Missing gradient histogram bins |
| L2 | debug_telemetry.py | 240-298 | `RatioExplosionDiagnostic` lacks per-head breakdown |
| L3 | lstm_health.py | 71-104 | Missing cell state mean/gate saturation metrics |
| L4 | gradient_ema.py | 39-61 | No warm-up suppression for drift warnings |
| L5 | telemetry_config.py | 43-44 | No per-anomaly-type escalation duration |

---

## Recommendations

### Immediate Actions

1. **Add entropy collapse to `AnomalyDetector`** - This is the most impactful missing diagnostic. PPO policies frequently collapse to determinism without proper entropy incentives.

2. **Add KL divergence anomaly check** - KL spikes precede ratio explosion and are easier to catch early.

3. **Wire gradient EMA drift to anomaly detection** - The `GradientEMATracker` exists but its results aren't passed to `AnomalyDetector.check_gradient_drift`.

### Short-Term Improvements

4. **Add advantage statistics to `emit_ppo_update_event`** - Mean, std, min, max of advantages helps diagnose policy gradient variance issues.

5. **Make thresholds configurable** - Consolidate hard-coded values into `TelemetryConfig` or a separate `PPODiagnosticThresholds` dataclass.

6. **Add per-head clip fractions** - For factored action spaces, knowing which heads are clipping helps diagnose learning imbalances.

### Long-Term Enhancements

7. **Gradient histogram bins** - Distribution shape information for advanced debugging.

8. **Pre-clip gradient norm tracking** - Understand how much clipping is affecting learning.

9. **Cell state diagnostics for LSTM** - Catch LSTM issues earlier in training.

---

## Appendix: PPO Health Metrics Taxonomy

For reference, here's what a complete PPO telemetry suite should cover:

| Category | Metric | Current Status |
|----------|--------|----------------|
| **Policy** | entropy | Emitted, not anomaly-checked |
| | entropy_collapsed flag | Emitted |
| | kl_divergence | Emitted, not anomaly-checked |
| | clip_fraction | Emitted |
| | ratio_max/min/std | Emitted + anomaly-checked |
| **Value** | explained_variance | Emitted + anomaly-checked |
| | value_loss | Emitted |
| | value_std | Not emitted |
| **Advantage** | advantage_mean | **NOT EMITTED** |
| | advantage_std | **NOT EMITTED** |
| | advantage_min/max | **NOT EMITTED** |
| **Gradient** | grad_norm (clipped) | Emitted |
| | grad_norm (pre-clip) | **NOT EMITTED** |
| | per-layer stats | Debug mode only |
| | EMA drift | Tracked, not wired to anomaly |
| **LSTM** | h_norm, c_norm | Emitted |
| | has_nan/inf | Emitted |
| | gate saturation | **NOT EMITTED** |
