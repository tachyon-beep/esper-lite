# SME Report: esper.tolaria Package

**Package:** `/home/john/esper-lite/src/esper/tolaria/`
**Report Date:** 2025-12-02
**Status:** Production-ready with minor improvements recommended

---

## 1. Executive Summary

The `esper.tolaria` package provides model training infrastructure including epoch trainers, a model factory, and the TolariaGovernor catastrophic failure watchdog. The Governor implements a sophisticated safety mechanism that monitors training stability and can rollback to "Last Known Good" state while signaling punishment rewards to the RL agent. The training loops are well-structured with proper PyTorch idioms for different seed lifecycle stages (normal, womb mode, blended).

---

## 2. Key Features & Responsibilities

### Module Structure

| Module | Responsibility | Lines |
|--------|----------------|-------|
| `environment.py` | Model factory via TaskSpec delegation | 28 |
| `trainer.py` | Epoch training functions (4 variants) | 253 |
| `governor.py` | Fail-safe watchdog with rollback capability | 199 |
| `__init__.py` | Public API exports | 41 |

### Core Capabilities

1. **Model Factory** (`create_model`)
   - Delegates to TaskSpec for model instantiation
   - CUDA availability validation with clear error messages
   - Supports any task type (CIFAR-10, language modeling, etc.)

2. **Epoch Training Functions**
   - `train_epoch_normal`: Standard training (host only)
   - `train_epoch_womb_mode`: STE training with isolated seed output
   - `train_epoch_blended`: Joint host+seed training
   - `validate_and_get_metrics`: Validation with optional per-class accuracy

3. **TolariaGovernor** (Fail-safe Watchdog)
   - Statistical anomaly detection (6-sigma, 3x multiplier)
   - NaN/Inf immediate panic
   - Lobotomy detection (jump to random-guess loss)
   - In-memory checkpoint/rollback
   - RL punishment signaling

---

## 3. Notable Innovations

### TolariaGovernor Design

The Governor implements a "Super-Ego" pattern for training safety:

```python
class TolariaGovernor:
    """The Super-Ego of the training system.

    Capabilities:
    1. Anomaly Detection - NaN/Inf and statistical outliers
    2. State Reversion - RAM checkpoint for instant rollback
    3. RL Punishment - Returns negative reward for PPO buffer injection
    """
```

**Multi-threshold Anomaly Detection:**
```python
# ALL conditions must be met for panic:
# 1. Loss exceeds absolute threshold (e.g., > 10.0)
# 2. Loss exceeds statistical threshold (6 sigma)
# 3. Loss exceeds multiplier threshold (3x average)
is_anomaly = (
    current_loss > self.absolute_threshold and
    current_loss > statistical_threshold and
    current_loss > multiplier_threshold
)
```

This triple-gate approach prevents false positives from transient loss spikes while still catching genuine catastrophic failures.

**Lobotomy Detection:**
```python
# Catches "silent failures" where model outputs uniform probabilities
# If we were doing well (loss < 60% of random guess) and suddenly
# hit exactly the random guess loss, that's a lobotomy
if (avg < self.random_guess_loss * 0.6 and
    abs(current_loss - self.random_guess_loss) < 0.15):
```

This catches the subtle failure mode where a neural network collapses to outputting uniform distributions (ln(num_classes) loss).

### Womb Mode Training

The womb mode concept isolates seed output while still training both host and seed:

```python
def train_epoch_womb_mode(...):
    """Train one epoch with seed in isolation (seed output doesn't affect forward pass).

    During TRAINING stage (womb mode), the seed uses a Straight-Through Estimator:
    - Forward: output = host_features (seed contribution is zero)
    - Backward: seed receives gradients as if fully blended

    CRITICAL: Both host AND seed train. The "isolation" refers to the seed's
    output not affecting the loss computation (alpha=0), NOT to freezing the host.
    """
```

This enables safe seed exploration without risking catastrophic interference with the host model.

---

## 4. Complexity Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Lines of Code | ~520 | Low |
| Cyclomatic Complexity | Low | Simple control flow |
| Module Dependencies | 4 internal | Minimal coupling |
| Public API Surface | 6 functions/classes | Clean interface |

**Overall Complexity Rating: LOW**

The package is well-factored with clear separation of concerns:
- Factory pattern in `environment.py`
- Pure functions in `trainer.py`
- Stateful watchdog in `governor.py`

---

## 5. DRL Specialist Assessment

### Governor as Safety Mechanism for RL

The TolariaGovernor serves as a critical safety layer for reinforcement learning:

**Strengths:**
1. **Multi-gate Panic Threshold**: Requiring absolute + statistical + multiplier thresholds prevents the agent from being punished for normal training variance
2. **Consecutive Panic Requirement**: `min_panics_before_rollback=2` adds temporal filtering to avoid false positives
3. **Immediate NaN/Inf Handling**: Bypasses consecutive requirement for truly catastrophic failures
4. **Clean RL Integration**: `get_punishment_reward()` returns -10.0 (configurable) for direct injection into PPO buffer

**Integration Pattern (from `simic/vectorized.py`):**
```python
# Governor watchdog: check vital signs after validation
is_panic = env_state.governor.check_vital_signs(val_loss)
if is_panic:
    governor_panic_envs.append(env_idx)

# Later: inject punishment
if env_idx in governor_panic_envs:
    punishment = env_state.governor.get_punishment_reward()
    reward += punishment  # Negative reward injected into PPO buffer
```

### Punishment Reward Integration

The death penalty reward (-10.0 default) is:
- **Magnitude Appropriate**: Comparable to episode-length negative experiences
- **Additive**: Combined with regular reward, not replacing it
- **Configurable**: Can be tuned per-experiment

**Potential Concern**: The punishment is applied after rollback, meaning the agent observes:
1. A state that caused catastrophic failure
2. Punishment reward
3. Model state reverted to pre-catastrophe

This could create confusing credit assignment if the state doesn't clearly indicate imminent failure.

### Training Stability for Policy Learning

The training functions maintain stable gradients:
1. `optimizer.zero_grad(set_to_none=True)`: Memory-efficient gradient clearing
2. Separate host/seed optimizers prevent gradient interference
3. `non_blocking=True` data transfers maintain training throughput

---

## 6. PyTorch Specialist Assessment

### Training Loop Patterns

**Positive Patterns:**

1. **GPU-native accumulation** (validated metrics):
```python
# Accumulate on GPU to avoid CPU-GPU sync per batch
val_loss_tensor = torch.tensor(0.0, device=device)
val_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
# ... accumulation loop ...
# Single CPU sync point at end of validation
val_loss = val_loss_tensor.item() / max(len(testloader), 1)
```

2. **Non-blocking data transfer**:
```python
inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
```

3. **Efficient gradient clearing**:
```python
optimizer.zero_grad(set_to_none=True)  # Better than set_to_none=False
```

4. **Vectorized per-class accuracy**:
```python
# Vectorized per-class counting using bincount
class_total_tensor += torch.bincount(labels, minlength=num_classes)
correct_mask = predicted == labels
class_correct_tensor += torch.bincount(labels[correct_mask], minlength=num_classes)
```

### Optimizer Management

**Pattern:**
```python
# Womb mode: separate optimizers, simultaneous stepping
host_optimizer.zero_grad(set_to_none=True)
seed_optimizer.zero_grad(set_to_none=True)
outputs = model(inputs)
loss.backward()
host_optimizer.step()
seed_optimizer.step()
```

This is correct for dual-optimizer scenarios. Both optimizers see gradients from the same backward pass.

### Checkpoint/Rollback Implementation

**Snapshot Implementation:**
```python
def snapshot(self) -> None:
    """Save Last Known Good state (GPU-native clone for tensors, deepcopy for extra_state)."""
    self.last_good_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in self.model.state_dict().items()
    }
```

**Assessment:**
- GPU-native tensor cloning avoids CPU roundtrip
- `copy.deepcopy()` handles non-tensor state (buffers, etc.)
- State dict approach is standard PyTorch pattern

**Rollback Implementation:**
```python
def execute_rollback(self) -> GovernorReport:
    # Clear live seeds FIRST (removes parameters from state_dict)
    if hasattr(self.model, 'seed_slot'):
        self.model.seed_slot.cull("governor_rollback")

    # Restore with strict=True ensures complete restoration
    self.model.load_state_dict(self.last_good_state, strict=True)
```

**Critical Observation:** The `cull()` call before `load_state_dict()` is essential - it removes dynamic seed parameters so the state dict keys match the snapshot. This is a sophisticated pattern for models with dynamic architecture.

### Device Handling

**Factory Validation:**
```python
if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        f"CUDA device '{device}' requested but CUDA is not available. "
        f"Use device='cpu' or check your CUDA installation."
    )
```

**Assessment:** Correct pattern with clear error message. Could additionally validate specific device index (e.g., `cuda:1` when only `cuda:0` exists).

---

## 7. Risks & Technical Debt

### Low Risk

1. **Hardcoded Random Guess Calculation**
   ```python
   self.random_guess_loss = math.log(num_classes)
   ```
   Assumes cross-entropy loss. Would need adjustment for other loss functions.

2. **Training Sample Size**
   ```python
   for inputs, labels in itertools.islice(trainloader, 10):  # Quick sample
   ```
   Fixed 10-batch sample for training metrics may not be representative for all batch sizes.

### Medium Risk

3. **Optimizer State Not Preserved in Rollback**
   The Governor only snapshots model weights, not optimizer state (momentum buffers, Adam statistics). After rollback, optimizers continue with stale momentum.

   **Impact:** May cause training instability immediately after rollback.

   **Location:** `governor.py:67-72`

4. **hasattr for Feature Detection**
   ```python
   # hasattr AUTHORIZED by John on 2025-12-01 16:30:00 UTC
   # Justification: Feature detection - MorphogeneticModel has seed_slot, base models may not
   if hasattr(self.model, 'seed_slot'):
   ```
   Properly authorized per CLAUDE.md policy, but indicates potential type contract gap.

### Mitigated Risks

5. **Auto-snapshot on Init**
   ```python
   # Capture an initial snapshot so rollback is always possible, even on first panic
   self.snapshot()
   ```
   This mitigates the risk of panic before first explicit snapshot.

---

## 8. Opportunities for Improvement

### Short-term (Low effort, High value)

1. **Add Optimizer State to Snapshots**
   ```python
   def snapshot(self) -> None:
       self.last_good_state = {
           'model': {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                     for k, v in self.model.state_dict().items()},
           # Consider adding optimizer state for complete rollback fidelity
       }
   ```

2. **Configurable Training Sample Size**
   ```python
   def validate_and_get_metrics(..., train_sample_batches: int = 10):
   ```

### Medium-term (Moderate effort)

3. **Protocol for Seed Slot Interface**
   Define a Protocol for models with seed slots to replace hasattr:
   ```python
   class SeedSlotModel(Protocol):
       seed_slot: SeedSlot
   ```

4. **torch.compile Support**
   The training loops are compile-friendly. Consider adding:
   ```python
   @torch.compile(mode="reduce-overhead")
   def _compiled_train_step(...):
   ```

### Long-term (Architecture)

5. **Gradient Checkpointing Integration**
   For larger models, integrate `torch.utils.checkpoint` for memory efficiency during womb mode training.

---

## 9. Critical Issues

**None identified.**

The package is well-implemented with no critical bugs or security issues. The hasattr usage is properly authorized and justified per project policy.

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort | Impact |
|----------|----------------|--------|--------|
| Medium | Add optimizer state to Governor snapshots | Low | High |
| Low | Make training sample size configurable | Low | Low |
| Low | Add CUDA device index validation | Low | Low |
| Medium | Define Protocol for seed slot interface | Medium | Medium |
| Low | Add torch.compile decorators for hot paths | Medium | Medium |

### Key Takeaways

1. **TolariaGovernor is production-ready** - The multi-threshold anomaly detection and lobotomy detection are sophisticated safety mechanisms appropriate for RL training.

2. **Training loops follow PyTorch best practices** - GPU-native accumulation, non-blocking transfers, and efficient gradient handling.

3. **Womb mode training is correctly implemented** - The dual-optimizer pattern with STE semantics enables safe seed exploration.

4. **Minor technical debt exists** - Optimizer state not included in snapshots is the most significant gap.

---

## Appendix: Test Coverage

| Test File | Tests | Coverage Focus |
|-----------|-------|----------------|
| `tests/test_tolaria_governor.py` | 26 | Governor panic detection, rollback, punishment |
| `tests/esper/test_tolaria.py` | 8 | Trainer smoke tests, metrics validation |

Test coverage is comprehensive for the Governor and adequate for training functions. Integration testing with actual seed lifecycles is covered in `test_tolaria_governor.py:test_execute_rollback_clears_live_seeds`.
