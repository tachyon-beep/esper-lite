# Simic PPO Module Audit Report

**Date:** 2025-12-16
**File:** `/home/john/esper-lite/src/esper/simic/ppo.py`
**Auditor:** Claude Code (PyTorch Expert)
**PyTorch Version Target:** 2.9+

---

## Executive Summary

The `ppo.py` module implements a PPO agent for Tamiyo seed lifecycle control with factored recurrent actor-critic architecture. The code demonstrates solid PyTorch 2.9 practices including proper `torch.compile` usage, fused optimizer selection, and correct device placement patterns. However, several issues warrant attention ranging from potential integration bugs to memory management concerns.

**Overall Assessment:** Good quality with 2 High, 4 Medium, and 3 Low severity issues identified.

---

## 1. torch.compile Compatibility Issues

### 1.1 Compiled Module Access Pattern - **MEDIUM**

**Location:** Lines 252-253, 294-306

```python
if compile_network:
    self.network = torch.compile(self.network, mode="default")
```

**Issue:** The `_base_network` property correctly handles compiled modules via `_orig_mod`, but this relies on undocumented PyTorch internals.

```python
@property
def _base_network(self):
    # hasattr AUTHORIZED by John on 2025-12-10 21:30:00 UTC
    if hasattr(self.network, '_orig_mod'):
        return self.network._orig_mod
    return self.network
```

**Risk:** The `_orig_mod` attribute is an implementation detail of `OptimizedModule` and may change in future PyTorch versions. While currently stable through PyTorch 2.9, this pattern is not part of the public API.

**Recommendation:** Consider using `torch._dynamo.OptimizedModule` type checking for explicit handling:
```python
from torch._dynamo import OptimizedModule
if isinstance(self.network, OptimizedModule):
    return self.network._orig_mod
return self.network
```

### 1.2 Graph Break in Entropy Floor Computation - **LOW**

**Location:** Lines 331-368 (`get_entropy_floor`)

```python
num_valid = int(action_mask.sum().item())  # Forces CPU sync
```

**Issue:** The `.item()` call forces a CUDA synchronization, creating a graph break if this method is called within a compiled region. Currently this is called from `get_entropy_coef()` which is used in the `update()` method.

**Impact:** Minor - this occurs once per update, not per step. The synchronization is unavoidable for the adaptive floor computation.

**Recommendation:** Document that `get_entropy_coef()` should not be called within `@torch.compile` regions. Consider caching the floor value per-batch.

### 1.3 Mode Selection for Compile - **LOW**

**Location:** Line 253

```python
self.network = torch.compile(self.network, mode="default")
```

**Observation:** Using `mode="default"` is appropriate given `MaskedCategorical` has `@torch.compiler.disable` on its validation. However, the code comment references `MaskedCategorical._validate_action_mask` which is isolated correctly.

**Recommendation:** The mode selection is correct. Consider adding `fullgraph=False` explicitly to document that graph breaks are expected and acceptable.

---

## 2. Device Placement Issues

### 2.1 Mixed Device String and torch.device - **MEDIUM**

**Location:** Lines 231, 243-244, 256

```python
self.device = device  # String: "cuda:0"
# ...
self.buffer = TamiyoRolloutBuffer(
    device=torch.device(device),  # Converted to torch.device
)
```

**Issue:** The agent stores `device` as a string (`"cuda:0"`) but the buffer receives `torch.device(device)`. This inconsistency could cause subtle bugs if code assumes one type or the other.

**Evidence of potential issue in update():**
```python
data = self.buffer.get_batched_sequences(device=self.device)  # String passed
```

The buffer's `get_batched_sequences` handles both types (line 344 in tamiyo_buffer.py), but this defensive code shouldn't be necessary.

**Recommendation:** Standardize on `torch.device` throughout:
```python
self.device = torch.device(device) if isinstance(device, str) else device
```

### 2.2 Optimizer fused/foreach Selection - **LOW**

**Location:** Lines 256-261

```python
use_cuda = device.startswith("cuda")
optimizer_kwargs = {'lr': lr, 'eps': 1e-5}
if use_cuda:
    optimizer_kwargs['fused'] = True
else:
    optimizer_kwargs['foreach'] = True
```

**Observation:** This is correct PyTorch 2.9 practice. Fused kernels require CUDA, foreach is optimal for CPU.

**Minor issue:** If `device` is a `torch.device` object (after fixing 2.1), `.startswith()` will fail.

**Recommendation:** After standardizing device type:
```python
use_cuda = self.device.type == "cuda"
```

---

## 3. Gradient Flow Concerns

### 3.1 Per-Head Advantage Masking Could Starve Gradients - **HIGH**

**Location:** Lines 444-447, integration with `advantages.py`

```python
per_head_advantages = compute_per_head_advantages(
    valid_advantages, valid_op_actions
)
```

**Issue:** The causal masking in `compute_per_head_advantages` zeros out advantages for heads that weren't "causally relevant." For example, `blueprint_advantages` is zero for all non-GERMINATE actions.

From `advantages.py`:
```python
blueprint_mask = is_germinate
blueprint_advantages = base_advantages * blueprint_mask.float()  # Zero for non-GERMINATE
```

**Risk:** If GERMINATE actions are rare (which they often are in later training), the blueprint and blend heads receive near-zero gradients. This could cause:
1. Gradient starvation for these heads
2. Weight decay (if enabled) pushing weights toward zero without counterbalancing gradients
3. Policy collapse where GERMINATE becomes increasingly unlikely

**Evidence:** The PPO loss computation:
```python
for key in ["slot", "blueprint", "blend", "op"]:
    ratio = per_head_ratios[key]
    adv = per_head_advantages[key]  # Often zero for blueprint/blend
    surr1 = ratio * adv
    # ...
    head_loss = -torch.min(surr1, surr2).mean()  # Mean of zeros = 0
```

**Recommendation:**
1. Monitor gradient norms per-head in telemetry
2. Consider adding a small exploration bonus for underutilized heads
3. Ensure weight_decay=0.0 for actor heads (currently correct per line 270-275)

### 3.2 Entropy Sum Across Heads May Over-Regularize - **MEDIUM**

**Location:** Lines 487-489

```python
entropy_loss = 0.0
for key, ent in entropy.items():
    entropy_loss = entropy_loss - ent.mean()
```

**Issue:** Entropy is summed across all 4 heads (slot, blueprint, blend, op). With 4 heads, the effective entropy coefficient is 4x the configured value.

**Impact:** If `entropy_coef=0.05`, the effective coefficient is 0.20, which may be too aggressive and impede convergence.

**Recommendation:** Either:
1. Divide the summed entropy by number of heads: `entropy_loss / 4`
2. Apply entropy coefficient per-head: `entropy_coef * ent.mean()` inside the loop
3. Document that `entropy_coef` should be set 4x lower than typical PPO

---

## 4. Memory Management

### 4.1 No Explicit CUDA Memory Management - **MEDIUM**

**Location:** Throughout `update()` method (lines 370-564)

**Observation:** The update method processes full batches without explicit memory management. For large state dimensions or many environments, this could cause OOM.

**Missing patterns:**
- No gradient accumulation for large batches
- No `torch.cuda.empty_cache()` after update
- No memory profiling hooks

**Recommendation:** Add optional gradient accumulation:
```python
def update(self, clear_buffer: bool = True, accumulation_steps: int = 1):
    # Split data into accumulation_steps chunks
    # Call optimizer.step() only after accumulation_steps
```

### 4.2 Buffer Reset Doesn't Free Memory - **LOW**

**Location:** `TamiyoRolloutBuffer.reset()` (line 384-389 in tamiyo_buffer.py)

```python
def reset(self) -> None:
    self.step_counts = [0] * self.num_envs
    # Tensors don't need zeroing - step_counts controls valid range
```

**Observation:** This is actually correct and efficient - pre-allocated tensors are reused. The comment accurately describes the design decision.

**No action required.**

### 4.3 Potential Memory Leak in metrics Dict - **LOW**

**Location:** Lines 407, 520

```python
metrics = defaultdict(list)
# ...
metrics.setdefault("ratio_diagnostic", []).append(diag.to_dict())
```

**Issue:** If `ratio_diagnostic` is triggered frequently, the list grows unboundedly within a single update. While unlikely to be problematic (update() is called infrequently), it's worth noting.

**Recommendation:** Consider limiting diagnostic history or aggregating instead of appending.

---

## 5. Integration Risks with Other Simic Files

### 5.1 TamiyoRolloutBuffer Hidden State Shape Mismatch Risk - **HIGH**

**Location:** Lines 238-244

```python
self.buffer = TamiyoRolloutBuffer(
    num_envs=num_envs,
    max_steps_per_env=max_steps_per_env,
    state_dim=state_dim,
    lstm_hidden_dim=lstm_hidden_dim,
    device=torch.device(device),
)
```

**Issue:** The buffer is created with `lstm_hidden_dim` but the network's actual hidden dimension depends on `FactoredRecurrentActorCritic` defaults. If the network is created with different `lstm_hidden_dim`, the hidden states won't match.

**Evidence in tamiyo_buffer.py:**
```python
self.hidden_h = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)
```

**Risk:** Silent dimension mismatch if network and buffer are created with different hidden dims.

**Recommendation:** Add validation in PPOAgent.__init__:
```python
assert self.buffer.lstm_hidden_dim == self.network.lstm_hidden_dim, \
    f"Buffer hidden_dim {self.buffer.lstm_hidden_dim} != network {self.network.lstm_hidden_dim}"
```

### 5.2 signals_to_features Import Coupling - **MEDIUM**

**Location:** Lines 85-87

```python
from esper.simic.features import obs_to_multislot_features
from esper.simic.slots import CANONICAL_SLOTS, ordered_slots
```

**Issue:** `signals_to_features` has runtime imports inside the function. This works but:
1. Makes the function slower (import lookup on every call)
2. Hides dependencies
3. Can cause circular import issues if the dependency graph changes

**Recommendation:** Move imports to module level or use `TYPE_CHECKING` guard properly.

### 5.3 AnomalyDetector Threshold Alignment - **LOW**

**Location:** Lines 246-247

```python
self.ratio_explosion_threshold = 5.0
self.ratio_collapse_threshold = 0.1
```

**Observation:** These match `AnomalyDetector` defaults (anomaly_detector.py lines 46-47). Good alignment.

**However:** The thresholds are duplicated. If `AnomalyDetector` thresholds change, `PPOAgent` thresholds won't update automatically.

**Recommendation:** Import thresholds from a single source (leyline constants or AnomalyDetector):
```python
from esper.simic.anomaly_detector import AnomalyDetector
_detector = AnomalyDetector()
self.ratio_explosion_threshold = _detector.max_ratio_threshold
self.ratio_collapse_threshold = _detector.min_ratio_threshold
```

### 5.4 vectorized.py Integration - **LOW**

**Location:** `vectorized.py` imports `PPOAgent, signals_to_features` from ppo.py

**Observation:** The vectorized training loop correctly uses PPOAgent. The integration appears sound.

**Minor concern:** `vectorized.py` is 31K+ tokens (very large file). Consider splitting telemetry helpers and training loop for maintainability.

---

## 6. Code Quality Issues

### 6.1 Magic Numbers Without Constants - **MEDIUM**

**Location:** Multiple locations

```python
optimizer_kwargs = {'lr': lr, 'eps': 1e-5}  # Line 257 - why 1e-5?
scale_factor = min(scale_factor, 3.0)  # Line 366 - why 3.0?
if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:  # Line 541 - why 1.5?
```

**Recommendation:** Define constants in leyline:
```python
# In leyline/constants.py
DEFAULT_ADAM_EPSILON = 1e-5  # Numerical stability for Adam
MAX_ENTROPY_SCALE_FACTOR = 3.0  # Cap for adaptive entropy floor
KL_EARLY_STOP_MULTIPLIER = 1.5  # Standard from OpenAI baselines
```

### 6.2 Inconsistent Return Type Documentation - **LOW**

**Location:** Line 373 `update()` method

```python
def update(self, clear_buffer: bool = True) -> dict:
    """...
    Returns:
        Dict of training metrics
    """
```

**Issue:** Return type is `dict` but should be `dict[str, float | dict]` given the heterogeneous values (floats and diagnostic dicts).

### 6.3 save/load Asymmetry - **LOW**

**Location:** Lines 566-630

**Issue:** `save()` stores `hidden_dim` but `load()` infers `state_dim` from weights:
```python
# save() stores:
'lstm_hidden_dim': self.lstm_hidden_dim,

# load() infers state_dim but uses config for lstm_hidden_dim:
state_dim = state_dict['feature_net.0.weight'].shape[1]
agent = cls(state_dim=state_dim, **checkpoint.get('config', {}))
```

**Risk:** If `config` is missing `lstm_hidden_dim`, the default will be used, potentially causing shape mismatches.

**Recommendation:** Also store `state_dim` in architecture dict and validate on load.

---

## 7. Summary Table

| ID | Issue | Severity | Category | Effort to Fix |
|----|-------|----------|----------|---------------|
| 3.1 | Per-head advantage masking gradient starvation | High | Gradient Flow | Medium |
| 5.1 | Buffer/network hidden dim mismatch risk | High | Integration | Low |
| 2.1 | Mixed device string/torch.device | Medium | Device | Low |
| 3.2 | Entropy sum over-regularization | Medium | Gradient Flow | Low |
| 4.1 | No gradient accumulation for large batches | Medium | Memory | Medium |
| 5.2 | Runtime imports in signals_to_features | Medium | Integration | Low |
| 6.1 | Magic numbers without constants | Medium | Quality | Low |
| 1.1 | _orig_mod reliance on internals | Medium | Compile | Low |
| 1.2 | Graph break in entropy floor | Low | Compile | N/A |
| 2.2 | Optimizer fused/foreach with string device | Low | Device | Low |
| 5.3 | Duplicated anomaly thresholds | Low | Integration | Low |
| 6.2 | Inconsistent return type docs | Low | Quality | Low |
| 6.3 | save/load asymmetry | Low | Quality | Low |

---

## 8. Positive Observations

1. **Correct torch.compile usage:** Mode selection, @torch.compiler.disable placement, and compilation of the whole network (not individual methods) follows best practices.

2. **Fused optimizer selection:** Proper detection of CUDA vs CPU for fused/foreach optimizer variants.

3. **Weight decay actor isolation:** Correctly applies weight_decay only to critic, preventing exploration collapse.

4. **Gradient clipping:** Uses `clip_grad_norm_` with configurable max_grad_norm.

5. **KL early stopping:** Implements standard 1.5x target_kl early stopping from OpenAI baselines.

6. **hasattr authorization:** The single hasattr usage is properly documented per project guidelines.

7. **Buffer pre-allocation:** TamiyoRolloutBuffer uses pre-allocated tensors with step_counts control, avoiding GC pressure.

8. **Per-environment GAE:** Correctly computes GAE per-environment to prevent cross-contamination (documented as "P0 bug fix").

---

## 9. Recommendations Priority

### Immediate (Before Next Training Run)
1. Add hidden dim validation between buffer and network (5.1)
2. Document entropy coefficient scaling behavior (3.2)

### Short-term (This Sprint)
3. Standardize device type handling (2.1, 2.2)
4. Move magic numbers to leyline constants (6.1)
5. Add per-head gradient norm monitoring for gradient starvation detection (3.1)

### Medium-term (Backlog)
6. Consider gradient accumulation option (4.1)
7. Refactor signals_to_features imports (5.2)
8. Consolidate threshold constants (5.3)

---

*Report generated by Claude Code PyTorch Expert analysis*
