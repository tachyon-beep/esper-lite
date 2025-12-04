# SME Report: esper.tolaria

**Package:** Model Training Infrastructure
**Location:** `src/esper/tolaria/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert (Merged)

---

## 1. Executive Summary

The tolaria package provides a sophisticated training infrastructure with a standout TolariaGovernor watchdog that implements multi-threshold anomaly detection (absolute + statistical + multiplier), NaN/Inf immediate panic, and innovative "lobotomy detection" for models collapsing to random-guess output. Training loops follow modern PyTorch best practices with GPU-native accumulation and proper device handling.

---

## 2. Key Features & Responsibilities

| Feature | Description |
|---------|-------------|
| **create_model()** | Model factory via TaskSpec |
| **train_epoch_normal()** | Standard training without seeds |
| **train_epoch_incubator_mode()** | STE-based gradient isolation |
| **train_epoch_blended()** | Joint host+seed training |
| **TolariaGovernor** | Catastrophic failure watchdog with rollback |
| **validate_and_get_metrics()** | Comprehensive evaluation |

---

## 3. Notable Innovations

### Multi-Threshold Anomaly Detection
```python
def check_vital_signs(self, current_loss):
    # Three thresholds must ALL trigger:
    # 1. Absolute threshold (e.g., > 10.0)
    # 2. Statistical threshold (mean + 6σ)
    # 3. Multiplier threshold (3× average)
    panic = (loss > self.absolute_threshold and
             loss > statistical_threshold and
             loss > multiplier_threshold)
```
**Benefit:** Prevents false positives from normal training variance

### Lobotomy Detection
```python
random_guess_loss = math.log(self.num_classes)  # e.g., 2.3 for 10 classes
if current_loss >= random_guess_loss * 0.95:
    if previous_loss < random_guess_loss * 0.6:
        # Model collapsed to uniform output
        return self._panic("Lobotomy detected")
```
**Benefit:** Catches silent failures where model outputs uniform probabilities

### Consecutive Panic Requirement
```python
self._consecutive_panics += 1
if self._consecutive_panics >= self.min_panics_before_rollback:
    return self.execute_rollback()
```
**Benefit:** Requires 2+ consecutive anomalies before action

---

## 4. Complexity Analysis

| Aspect | Rating | Notes |
|--------|--------|-------|
| Overall | LOW | Clean separation of concerns |
| Training Loops | LOW | Standard PyTorch patterns |
| Governor | MEDIUM | Multi-threshold logic |

---

## 5. DRL Specialist Assessment

### Governor as RL Safety Mechanism

| Feature | RL Integration |
|---------|----------------|
| Punishment Reward | -10.0 injected into PPO buffer |
| Rollback | Preserves training continuity |
| Consecutive Requirement | Prevents false punishment |

### Training Stability for Policy Learning

| Aspect | Rating | Notes |
|--------|--------|-------|
| Incubator Mode | GOOD | Safe seed exploration |
| Blending | GOOD | Smooth alpha ramp |
| Rollback | GOOD | Quick recovery |

---

## 6. PyTorch Specialist Assessment

### Training Loop Best Practices

| Practice | Implemented | Notes |
|----------|-------------|-------|
| `non_blocking=True` | YES | Async data transfer |
| GPU accumulation | YES | Avoid CPU sync |
| `set_to_none=True` | YES | Faster grad clearing |
| `torch.inference_mode()` | YES | Validation efficiency |

### Optimizer Management

| Aspect | Rating | Notes |
|--------|--------|-------|
| Dual Optimizers | GOOD | Separate host/seed |
| LR Handling | GOOD | Configurable |
| State Preservation | NEEDS WORK | Not in Governor snapshot |

### Checkpoint/Rollback

| Aspect | Rating | Notes |
|--------|--------|-------|
| Model State | GOOD | Full state_dict clone |
| Extra State | GOOD | Deepcopy for non-tensors |
| Optimizer State | MISSING | Momentum not preserved |

---

## 7. Risks & Technical Debt

| Risk | Severity | Description |
|------|----------|-------------|
| Optimizer State | MEDIUM | Not included in Governor snapshot |
| Fixed Sample Size | LOW | 10-batch training metric sample |
| Device Validation | LOW | No explicit device string check |

---

## 8. Opportunities for Improvement

### High Value
1. **Add optimizer state to Governor** - Full rollback fidelity
2. **Configurable sample size** - For training metrics

### Medium Value
3. **Device validation** - Explicit torch.device checks
4. **Governor telemetry** - Emit events on anomaly detection

### Low Value
5. **Training loop unification** - Reduce code duplication

---

## 9. Critical Issues

### Missing Optimizer State in Rollback (MEDIUM)
```python
# governor.py
def snapshot(self):
    self._lkg_state = {
        k: v.clone() for k, v in self.model.state_dict().items()
    }
    # Missing: optimizer.state_dict()
```
**Issue:** After rollback, optimizers have stale momentum
**Impact:** Training may be unstable immediately after rollback
**Fix:** Snapshot optimizer state alongside model state

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort |
|----------|----------------|--------|
| P1 | Add optimizer state to Governor | 2 hours |
| P1 | Add Governor telemetry events | 1 hour |
| P2 | Configurable sample size | 30 min |
| P3 | Device string validation | 30 min |
| P3 | Training loop unification | 4 hours |

---

**Quality Score:** 8.5/10 - Solid implementation with excellent safety features
**Confidence:** HIGH
