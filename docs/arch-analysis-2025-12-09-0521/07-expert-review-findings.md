# Expert Review Findings: Architect Handover Analysis

**Review Date:** 2025-12-09
**Reviewers:** DRL Expert, PyTorch Expert, Code Review Expert
**Target Document:** 06-architect-handover.md
**Environment:** Python 3.13, PyTorch 2.9

---

## Executive Summary

Three specialist agents reviewed the architect handover document and identified **3 critical issues**, **4 missing items**, and **8 enhancement opportunities**. This document catalogs each finding with problem description, impact analysis, recommended fix, and resolution rationale.

---

## Critical Issues

### CRIT-1: Incorrect Function Signature in IMP-1

**Problem:**
The proposed `_train_one_epoch()` helper function has an incorrect return type signature.

```python
# PROPOSED (WRONG)
def _train_one_epoch(...) -> tuple[float, float]:

# ACTUAL REQUIREMENT (based on existing _loss_and_correct signature)
def _train_one_epoch(...) -> tuple[float, float, int]:
```

**Why It's a Problem:**
- The existing `_loss_and_correct()` function returns `tuple[torch.Tensor, float, int]`
- Training loops accumulate: `running_loss` (float from `loss.item()`), `correct` (float), and `total` (int)
- A developer implementing IMP-1 would encounter type errors immediately
- The accuracy calculation `correct / total` requires the float and int types

**Evidence:**
```python
# From simic/training.py lines 90-95 (actual function signature)
def _loss_and_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, float, int]:  # Returns (loss_tensor, correct_count, total_count)

# From simic/training.py lines 185-199 (training loop accumulator pattern)
running_loss = 0.0  # float accumulator
correct = 0         # receives float from _loss_and_correct
total = 0           # int accumulator
for inputs, targets in trainloader:
    ...
    loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
    running_loss += loss.item()  # Tensor.item() -> float
    correct += correct_batch     # float
    total += batch_total         # int
```

**The Fix:**
```python
def _train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str,
    collect_gradients: bool = False,
) -> tuple[float, float, int]:
    """Unified training loop for all seed stages.

    Returns:
        Tuple of (running_loss, correct_count, total_count)
        - running_loss: Sum of loss.item() across batches (float)
        - correct_count: Sum of correct predictions (float)
        - total_count: Total samples processed (int)
    """
```

**How This Fixes It:**
- Matches the actual return semantics of the existing code
- Preserves caller's ability to compute accuracy: `accuracy = correct / total`
- Maintains backward compatibility with all call sites
- Type checker will validate correct usage

**Verification Checklist:**
- [ ] Type signature matches `tuple[float, float, int]`
- [ ] All tests pass: `pytest tests/simic/ -v`
- [ ] Type checker validates: `mypy src/esper/simic/training.py`
- [ ] Training produces identical loss/accuracy values (compare before/after)

---

### CRIT-2: Non-Existent Directory Structure in IMP-2

**Problem:**
IMP-2 proposes creating `tests/tamiyo/test_decision_integration.py`, but the `tests/tamiyo/` directory does not exist.

**Why It's a Problem:**
- Developer will encounter immediate friction when attempting to create the file
- Unclear whether to create new directory or follow existing convention
- Existing Tamiyo test is at `tests/test_tamiyo_tracker.py` (root level, not subdirectory)
- Inconsistent test organization causes confusion

**Evidence:**
```
tests/
├── esper/
├── fixtures/
├── integration/
├── kasmina/          # Subdirectory exists
├── leyline/          # Subdirectory exists
├── nissa/            # Subdirectory exists
├── properties/
├── simic/            # Subdirectory exists
├── test_tamiyo_tracker.py   # ROOT level, not in subdirectory!
└── conftest.py
```

**The Fix:**
Create `tests/tamiyo/` directory to match the pattern used by other subsystems, then move existing test and add new tests:

```
tests/tamiyo/
├── __init__.py
├── test_tracker.py              # Moved from tests/test_tamiyo_tracker.py
├── test_heuristic_decisions.py  # NEW: HeuristicTamiyo decision tests
└── test_decision_integration.py # NEW: Integration tests
```

**How This Fixes It:**
- Establishes consistent directory structure matching kasmina/, simic/, etc.
- Provides clear location for all Tamiyo-related tests
- Developer knows exactly where to create new test files
- Consolidates scattered Tamiyo tests into single location

**Verification Checklist:**
- [ ] Directory `tests/tamiyo/` exists with `__init__.py`
- [ ] `test_tamiyo_tracker.py` moved to `tests/tamiyo/test_tracker.py`
- [ ] All existing Tamiyo tests still pass: `pytest tests/tamiyo/ -v`
- [ ] No broken imports in other test files

---

### CRIT-3: Incompatible Function Signatures in IMP-3

**Problem:**
IMP-3 proposes consolidating two loss computation functions that have different return signatures and serve different purposes.

```python
# simic/training.py lines 90-95 - returns 3 values
def _loss_and_correct(outputs, targets, criterion, task_type) -> tuple[torch.Tensor, float, int]:
    return loss, correct, total  # (Tensor, float, int)

# tolaria/trainer.py lines 39-44 - returns 1 value
def _compute_loss(outputs, targets, criterion, task_type) -> torch.Tensor:
    return loss  # Tensor only
```

**Why It's a Problem:**
- These functions are NOT duplicates - they serve different purposes and have different arities
- `_loss_and_correct()` returns 3 values: `(loss: Tensor, correct: float, total: int)`
- `_compute_loss()` returns 1 value: `loss: Tensor`
- Direct consolidation would break existing callers
- `_compute_loss()` callers don't need accuracy metrics
- Forcing all callers to handle 3 return values adds unnecessary complexity
- Adding `_ = ` ignores clutters code and wastes computation

**Evidence:**
The functions have fundamentally different use cases:
- `_loss_and_correct()`: Used in training loops that track accuracy, returns `float` for correct count (from `.item()` conversion)
- `_compute_loss()`: Used in validation where only loss matters for backprop

**The Fix:**
Instead of consolidating into one function, create two related functions with clear purposes:

```python
# In esper/leyline/loss.py (new file, keeps leyline as contracts)
# OR in esper/utils/loss.py (if leyline should stay PyTorch-free)

def compute_task_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> torch.Tensor:
    """Compute loss for classification or language modeling task.

    Use when you only need the loss value (e.g., validation, inference).
    """
    if task_type == "lm":
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
    return criterion(outputs, targets)


def compute_task_loss_with_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, float, int]:
    """Compute loss and accuracy metrics for classification or LM task.

    Use when you need both loss and accuracy (e.g., training loops).

    Returns:
        Tuple of (loss, correct_count, total_count)
        - loss: CrossEntropyLoss tensor (for backprop)
        - correct_count: Number of correct predictions (float from .item())
        - total_count: Total samples in batch (int)
    """
    loss = compute_task_loss(outputs, targets, criterion, task_type)

    if task_type == "lm":
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        predicted = outputs_flat.argmax(dim=1)
        correct = float((predicted == targets_flat).sum().item())
        total = targets_flat.size(0)
    else:
        predicted = outputs.argmax(dim=1)
        correct = float((predicted == targets).sum().item())
        total = targets.size(0)

    return loss, correct, total
```

**How This Fixes It:**
- Provides single source of truth for loss computation logic
- Maintains separate functions for different use cases
- `compute_task_loss_with_metrics` internally calls `compute_task_loss` (DRY)
- Preserves exact type signature: `tuple[torch.Tensor, float, int]`
- Callers choose the appropriate function for their needs
- No breaking changes to existing call sites

**Verification Checklist:**
- [ ] New file created at chosen location (leyline/loss.py or utils/loss.py)
- [ ] `compute_task_loss` returns `torch.Tensor`
- [ ] `compute_task_loss_with_metrics` returns `tuple[torch.Tensor, float, int]`
- [ ] All callers of `_loss_and_correct` updated to use new function
- [ ] All callers of `_compute_loss` updated to use new function
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Type checker validates: `mypy src/esper/`

---

## Missing Items

### MISS-1: Validation Pass Consolidation Not in Backlog

**Problem:**
The quality assessment (05-quality-assessment.md) identifies that `validate_and_get_metrics()` and `validate_with_attribution()` in tolaria/trainer.py share ~60% of their logic, but this improvement is completely absent from the handover backlog.

**Why It's a Problem:**
- Inconsistency between quality assessment findings and improvement backlog
- Developer reviewing both documents will be confused about priorities
- Duplicated validation logic is a maintenance burden
- Changes to validation must be made in two places

**The Fix:**
Add IMP-6 to the improvement backlog:

```markdown
#### IMP-6: Extract Common Validation Loop

**Status:** Ready for implementation
**Effort:** 2-3 hours
**Risk:** Low
**Priority:** P2

**Current State:**
`tolaria/trainer.py` contains two validation functions sharing ~60% logic:
- `validate_and_get_metrics()` (lines 159-269)
- `validate_with_attribution()` (lines 272-352)

**Target State:**
Extract common validation loop into `_run_validation_pass()` helper.

**Implementation Steps:**

1. Create helper function:
```python
def _run_validation_pass(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str,
    force_alpha: float | None = None,
) -> tuple[float, float]:
    """Run validation pass, optionally forcing seed alpha.

    Returns:
        Tuple of (average_loss, accuracy)
    """
```

2. Refactor both functions to use the helper
3. `validate_with_attribution()` calls helper twice (normal + alpha=0)
4. Run existing tests to verify behavior preserved

**Verification:**
- All tests in `tests/tolaria/` pass
- Validation produces identical results
```

**How This Fixes It:**
- Brings backlog into alignment with quality assessment findings
- Reduces code duplication in critical validation path
- Single place to update validation logic
- Easier to add new validation modes in future

---

### MISS-2: TYPE_CHECKING Pattern Not Documented

**Problem:**
The gotchas section doesn't mention the `TYPE_CHECKING` import pattern used throughout the codebase to avoid circular dependencies.

**Why It's a Problem:**
- New developers won't understand why imports are structured oddly
- May introduce circular dependencies by "fixing" the imports
- Pattern is used in kasmina/slot.py, tamiyo/heuristic.py, simic/features.py
- Critical for understanding the dependency architecture

**Evidence:**
```python
# From kasmina/slot.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.simic.features import TaskConfig  # Only for type hints
```

**The Fix:**
Add to Section 4.2 (Gotchas and Non-Obvious Behaviors):

```markdown
5. **TYPE_CHECKING Import Pattern** - Several modules use conditional imports
   to avoid circular dependencies:
   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from esper.simic.features import TaskConfig
   ```
   This pattern allows type hints without runtime imports. Used in:
   - `kasmina/slot.py` (imports TaskConfig)
   - `tamiyo/heuristic.py` (imports SeedState)
   - `simic/features.py` (imports various types)

   **Do not "fix" these by making them regular imports** - this will create
   circular import errors.
```

**How This Fixes It:**
- Documents a critical but non-obvious pattern
- Prevents well-intentioned "cleanup" that would break imports
- Helps developers understand the dependency architecture
- Reduces onboarding friction

---

### MISS-3: No Debugging Guidance

**Problem:**
The handover document lacks guidance on debugging tools, environment variables, and telemetry inspection.

**Why It's a Problem:**
- Developers will struggle to debug training issues
- Environment variables like `ESPER_DEBUG_STE` exist but aren't documented
- Telemetry system is powerful but usage isn't explained
- PPO training metrics interpretation not covered

**Evidence:**
```python
# From slot.py - undocumented debug flag
if os.environ.get("ESPER_DEBUG_STE"):
    print(f"STE forward: host_grad={host_features.requires_grad}, ...")
```

**The Fix:**
Add new Section 4.4 (Debugging Guide):

```markdown
### 4.4 Debugging Guide

**Environment Variables:**
| Variable | Purpose |
|----------|---------|
| `ESPER_DEBUG_STE` | Enable verbose STE gradient flow logging |
| `CUDA_LAUNCH_BLOCKING=1` | Synchronous CUDA for debugging |

**Telemetry Inspection:**
```python
from esper.nissa import get_hub, ConsoleOutput

# Enable verbose console output during development
hub = get_hub()
hub.register_backend(ConsoleOutput(min_severity="DEBUG"))
```

**PPO Training Metrics:**

| Metric | Healthy Range | Problem Indicator |
|--------|---------------|-------------------|
| `policy_loss` | -0.1 to 0.1 | Diverging = learning rate too high |
| `value_loss` | Decreasing | Increasing = value function unstable |
| `entropy` | 0.3 to 0.8 (normalized) | < 0.1 = policy collapsed, > 0.95 = too uniform |
| `approx_kl` | < 0.02 | > 0.05 = policy changing too fast |
| `clip_fraction` | 0.1 to 0.3 | > 0.5 = clipping too aggressive |
| `ratio` | 0.8 to 1.2 (normal) | 0.5-2.0 = warning, <0.1 or >10 = critical |

**Note:** Entropy is normalized to [0, 1] in this codebase (entropy / max_entropy).

**Common Issues:**

1. **Ratio explosion** (>100): Stale log_probs or lr too high
2. **Ratio collapse** (<0.01): Policy changed too drastically
3. **Advantage std < 0.1**: Normalization amplifying noise
```

**How This Fixes It:**
- Provides actionable debugging information
- Documents hidden environment variables
- Explains how to interpret telemetry
- Reduces time spent diagnosing common issues

---

### MISS-4: Reward Documentation Priority Incorrect

**Problem:**
IMP-5 (Document Reward Shaping) is marked as P3 (low priority), but reward design is the most critical decision in RL systems.

**Why It's a Problem:**
- Reward shaping bugs are notoriously hard to detect
- PBRS violations can cause policy divergence
- The `STAGE_POTENTIALS` values affect all training runs
- Undocumented reward design makes debugging nearly impossible
- DRL expert assessment: "reward design is the most critical RL decision"

**The Fix:**
Elevate IMP-5 to P1 and expand scope:

```markdown
#### IMP-5: Document and Validate Reward Shaping (ELEVATED TO P1)

**Status:** Ready for implementation
**Effort:** 2-4 hours
**Risk:** Low (documentation) to Medium (validation tests)
**Priority:** P1 (elevated from P3)

**Rationale for Elevation:**
Reward shaping is the most critical design decision in RL. Undocumented
reward functions are a maintenance nightmare and debugging blind spot.

**Current State:**
- `STAGE_POTENTIALS` has minimal inline comments
- No explanation of why BLENDING has largest delta (+1.5)
- No validation that PBRS telescoping property holds
- No documentation of empirical tuning process

**Target State:**
1. Comprehensive documentation of reward design rationale
2. Property-based tests validating PBRS guarantees
3. Per-component reward logging for debugging

**Implementation Steps:**

1. Add documentation block to rewards.py:
```python
# POTENTIAL-BASED REWARD SHAPING (PBRS)
# =====================================
#
# These values implement Ng et al. (1999) potential-based shaping:
#   F(s, s') = gamma * phi(s') - phi(s)
#
# Key properties maintained:
# 1. Telescoping: Sum of shaping rewards = gamma^T * phi(s_T) - phi(s_0)
# 2. Policy invariance: Optimal policy unchanged by shaping
#
# Value rationale (actual values from rewards.py):
# - UNKNOWN (0.0): Fallback state
# - DORMANT (0.0): Baseline state
# - GERMINATED (1.0): +1.0 for initiating growth
# - TRAINING (2.0): +1.0 for successful germination
# - BLENDING (3.5): +1.5 (largest delta) - critical transition point
# - SHADOWING (4.5): +1.0 for surviving blending
# - PROBATIONARY (5.5): +1.0 for stability validation
# - FOSSILIZED (6.0): +0.5 terminal bonus (small to prevent fossilization farming)
#
# Tuning history:
# - v1: Linear progression (1.0 increments) - insufficient BLENDING incentive
# - v2: Current values - emphasizes BLENDING transition
```

2. Add property-based test:
```python
# tests/properties/test_pbrs_telescoping.py
@given(trajectories=trajectory_strategy())
def test_pbrs_telescoping_property(trajectories):
    """Verify PBRS telescoping: sum(F) = gamma^T * phi(s_T) - phi(s_0)"""
    gamma = 0.99
    total_shaping = sum(compute_shaping_reward(s, s_next)
                        for s, s_next in pairwise(trajectories))
    # T = number of transitions (N states -> N-1 transitions)
    T = len(trajectories) - 1
    expected = (gamma ** T) * phi(trajectories[-1]) - phi(trajectories[0])
    assert abs(total_shaping - expected) < 1e-6
```

3. Add reward component telemetry (already exists, ensure documented)
```

**How This Fixes It:**
- Elevates critical RL concern to appropriate priority
- Documents the "why" behind reward values
- Adds mathematical validation of PBRS guarantees
- Creates audit trail for future reward modifications
- Enables proper debugging of reward-related training issues

---

## Enhancement Opportunities

### ENH-1: Add torch.compile Validation Tests

**Problem:**
The codebase strategically disables torch.compile for certain functions but has no tests validating that the compile-compatible functions remain so.

**Why It's a Problem:**
- Future changes could accidentally introduce graph breaks
- No regression testing for compilation compatibility
- Silent performance degradation if compiled functions break

**The Fix:**
```python
# tests/integration/test_compile_compatibility.py
import torch._dynamo

def test_blend_with_isolation_compile_compatible():
    """Verify gradient isolation functions remain compile-compatible."""
    with torch._dynamo.error_on_graph_break():
        # Use reduce-overhead mode to match production config
        compiled_fn = torch.compile(blend_with_isolation, mode="reduce-overhead")
        host = torch.randn(4, 64, requires_grad=True)
        seed = torch.randn(4, 64, requires_grad=True)
        result = compiled_fn(host, seed, 0.5)
        assert result.shape == host.shape

        # CRITICAL: Test backward pass to verify gradient flow compilation
        result.sum().backward()
        assert host.grad is None  # Detached path - no gradients to host
        assert seed.grad is not None  # Gradient flows to seed

def test_ste_forward_compile_compatible():
    """Verify STE forward remains compile-compatible."""
    with torch._dynamo.error_on_graph_break():
        compiled_fn = torch.compile(ste_forward, mode="reduce-overhead")
        host = torch.randn(4, 64, requires_grad=True)
        seed = torch.randn(4, 64, requires_grad=True)
        result = compiled_fn(host, seed)
        result.sum().backward()
        assert host.grad is not None  # STE passes gradients through
        assert seed.grad is not None
```

**How This Fixes It:**
- Catches graph breaks before they reach production
- Documents which functions must remain compile-compatible
- Tests both forward and backward passes for compilation
- Uses production compile mode (`reduce-overhead`) for realistic testing
- Enables confident refactoring of isolation code

---

### ENH-2: Consider Mixed Precision Training (AMP)

**Problem:**
No automatic mixed precision (AMP) usage found in the codebase despite using modern GPUs.

**Why It's a Problem:**
- Missing ~2x training speedup on Ampere/Hopper GPUs
- Higher memory usage than necessary
- Not leveraging Tensor Core capabilities

**The Fix:**
```python
# In training loops
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# Initialize based on device
if use_amp and device.startswith("cuda"):
    scaler = GradScaler('cuda')
    autocast_ctx = autocast('cuda', dtype=torch.float16)
else:
    scaler = None
    autocast_ctx = nullcontext()

with autocast_ctx:
    outputs = model(inputs)
    loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)

if scaler:
    scaler.scale(loss).backward()
    # IMPORTANT: Unscale before gradient collection for accurate stats
    if use_telemetry:
        scaler.unscale_(optimizer)
        grad_stats = collect_seed_gradients(model.get_seed_parameters())
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    if use_telemetry:
        grad_stats = collect_seed_gradients(model.get_seed_parameters())
    optimizer.step()
```

**AMP Gotchas for Gradient Isolation:**

1. **Threshold adjustment for FP16 precision:**
   ```python
   # When using AMP, adjust gradient isolation monitor threshold
   monitor = GradientIsolationMonitor(threshold=1e-4)  # Not 1e-6 for FP16
   ```

2. **STE pattern is AMP-safe:** The STE operations (subtraction, addition, `.detach()`) preserve the autocast dtype correctly.

**How This Fixes It:**
- Enables Tensor Core utilization
- Reduces memory footprint by ~50%
- Speeds up training significantly on modern GPUs
- GradScaler handles loss scaling automatically
- `scaler.unscale_()` ensures accurate gradient statistics

---

### ENH-3: Standardize PyTorch Best Practices

**Problem:**
Inconsistent usage of PyTorch 2.x optimization patterns across the codebase.

**Issues Found:**

1. **Gradient Zeroing:** `tolaria/trainer.py` uses `zero_grad(set_to_none=True)` but `simic/training.py` does not.

2. **Device Transfers:** `tolaria/trainer.py` uses `non_blocking=True` but `simic/training.py` does not:
   ```python
   # tolaria/trainer.py (correct)
   inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

   # simic/training.py (missing non_blocking)
   inputs, targets = inputs.to(device), targets.to(device)
   ```

3. **Validation Context:** Using `torch.no_grad()` when `torch.inference_mode()` would be faster (5-10% in some cases).

**Why It's a Problem:**
- `zero_grad()` without `set_to_none=True` writes zeros (memory write overhead)
- `to(device)` without `non_blocking=True` synchronizes unnecessarily
- `torch.no_grad()` maintains autograd overhead that `inference_mode()` eliminates

**The Fix:**
Standardize across all training and validation code:

```python
# Gradient zeroing
optimizer.zero_grad(set_to_none=True)

# Device transfers
inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

# Validation (when gradients never needed)
with torch.inference_mode():
    for inputs, targets in testloader:
        # ...
```

**How This Fixes It:**
- Consistent memory behavior across all training paths
- Reduces memory footprint during training
- Enables async CPU-GPU transfers
- Follows PyTorch 2.x best practices

---

### ENH-4: Add Offline-to-Online RL Pipeline

**Problem:**
The RL system is purely on-policy (PPO), which is sample-inefficient for expensive environments.

**Why It's a Problem:**
- Each episode runs actual neural network training (expensive)
- Episodes are short (25 epochs = 25 timesteps)
- Expect 1000+ episodes before convergence
- `QNetwork` and `VNetwork` classes exist but no offline RL algorithm is implemented

**Infrastructure Clarification:**
- `networks.py` contains `QNetwork` and `VNetwork` classes for value function approximation
- No IQL or CQL algorithm implementation exists - this would need to be built
- For discrete action spaces (like seed lifecycle), CQL (Conservative Q-Learning) may be more appropriate than IQL

**The Fix:**
Implement offline pre-training pipeline using CQL (recommended for discrete actions):

1. Collect trajectories from heuristic policy (already implemented)
2. Implement CQL algorithm using existing `QNetwork`/`VNetwork` infrastructure
3. Pre-train value functions on heuristic data with conservative penalties
4. Initialize PPO from pre-trained networks
5. Fine-tune online with PPO

**Alternative Approaches:**
- **Decision Transformer:** Treats RL as sequence modeling. Works well with short episodes (25 timesteps).
- **Filtered BC:** If heuristic is high-quality, behavioral cloning with return filtering may be simpler.

**How This Fixes It:**
- Bootstraps policy from known-good heuristic behavior
- Reduces online sample requirements significantly
- CQL's conservative penalties prevent overestimation of OOD actions
- Maintains theoretical guarantees of PPO for fine-tuning

---

### ENH-5: Add GAE Lambda Configuration

**Problem:**
GAE lambda is fixed at 0.95, but the seed lifecycle has sparse rewards that may benefit from different values.

**Why It's a Problem:**
- Fossilization bonus is sparse (only at episode end)
- High lambda (0.95) propagates more variance
- Lower lambda (0.8-0.9) may reduce variance for sparse rewards
- No ability to tune without code changes

**The Fix:**
Add `gae_lambda` to TrainingConfig:

```python
@dataclass
class TrainingConfig:
    # ... existing fields
    gae_lambda: float = 0.95  # Default matches current behavior
```

Document tuning guidance:
- Dense rewards: lambda=0.95-0.99
- Sparse rewards (like fossilization): lambda=0.8-0.9
- Very sparse: Consider lambda=0.5-0.7

**How This Fixes It:**
- Enables experimentation without code changes
- Documents the trade-off for future tuning
- Maintains backward compatibility with default

---

### ENH-6: Document Entropy Normalization Trade-offs

**Problem:**
The PPO implementation uses normalized entropy (entropy / max_entropy), which is non-standard and has implications for exploration.

**Why It's a Problem:**
- Standard PPO uses raw entropy
- Normalized entropy changes exploration gradient landscape
- With 2 valid actions, uniform entropy=log(2), but normalized=1.0
- entropy_coef=0.05 normalized ≈ 0.098 nats raw
- May over-incentivize exploration in restricted states

**The Fix:**
Add documentation to ppo.py explaining the design decision:

```python
# ENTROPY NORMALIZATION
# ====================
# This implementation normalizes entropy by the maximum entropy of valid actions:
#   normalized_entropy = entropy / log(num_valid_actions)
#
# Trade-offs:
# - PRO: entropy_coef comparable across states with different action counts
# - PRO: Exploration bonus scales with action space size
# - CON: Non-standard, complicates comparison with other PPO implementations
# - CON: May over-incentivize exploration when few actions valid
#
# Equivalence: entropy_coef=0.05 normalized ≈ entropy_coef=0.098 raw (7 actions)
#
# If experiencing over-exploration in restricted states (e.g., only FOSSILIZE
# valid from PROBATIONARY), consider reducing entropy_coef or switching to
# raw entropy.
```

**How This Fixes It:**
- Documents non-obvious design decision
- Explains implications for tuning
- Provides guidance for troubleshooting exploration issues
- Enables informed decisions about entropy handling

---

### ENH-7: Add "Before You Start" Section

**Problem:**
The handover document assumes developers can immediately start implementing improvements without setup verification.

**Why It's a Problem:**
- No guidance on running test suite
- No baseline verification instructions
- Development environment assumptions unclear
- Developer may start changes without confirming tests pass

**The Fix:**
Add Section 0 (Before You Start):

```markdown
## 0. Before You Start

### Verify Development Environment
```bash
# Confirm Python and PyTorch versions
python --version  # Should be 3.13.x
python -c "import torch; print(torch.__version__)"  # Should be 2.9.x
```

### Run Baseline Tests
```bash
# All tests should pass before making changes
pytest tests/ -v

# Run specific subsystem tests
pytest tests/kasmina/ -v
pytest tests/simic/ -v
```

### Verify GPU Access (if applicable)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Recommended First Steps
1. Read CLAUDE.md for project coding guidelines
2. Run test suite to establish baseline
3. Pick one P1 improvement to start with
4. Create a branch: `git checkout -b improvement/IMP-X-description`
```

**How This Fixes It:**
- Ensures environment is correctly configured
- Establishes baseline before changes
- Prevents "it was already broken" confusion
- Provides clear starting workflow

---

### ENH-8: Expand Critical Code Paths

**Problem:**
The critical code paths section is incomplete, missing several important flows.

**Why It's a Problem:**
- Developers won't know where to look for certain behaviors
- Missing paths include reward computation, quality gates, action resolution
- Incomplete documentation slows onboarding

**The Fix:**
Expand Section 4.1 with additional paths:

```markdown
| Path | Entry | Key Files |
|------|-------|-----------|
| PPO Training | scripts/train.py | simic/training.py, simic/ppo.py |
| Seed Lifecycle | kasmina/slot.py | QualityGates, SeedState |
| Decision Making | tamiyo/heuristic.py | HeuristicTamiyo.decide() |
| Telemetry | nissa/output.py | NissaHub.emit() |
| **Reward Computation** | simic/rewards.py | compute_shaped_reward(), STAGE_POTENTIALS |
| **Quality Gates** | kasmina/slot.py | QualityGates.check_gate(), GateResult |
| **Action Resolution** | leyline/actions.py | Action enum → Blueprint → Germination |
| **Feature Extraction** | simic/features.py | obs_to_base_features() (HOT PATH) |
| **Gradient Isolation** | kasmina/isolation.py | ste_forward(), blend_with_isolation() |
```

**How This Fixes It:**
- Complete coverage of important code paths
- Faster navigation for developers
- Clear entry points for each concern

---

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Critical Issues | 3 | CRIT-1 (signature), CRIT-2 (directory), CRIT-3 (consolidation) |
| Missing Items | 4 | MISS-1 (validation), MISS-2 (TYPE_CHECKING), MISS-3 (debugging), MISS-4 (priority) |
| Enhancements | 8 | ENH-1 through ENH-8 |

**Recommended Priority for Fixes:**

1. **Immediate (before using handover):**
   - CRIT-1, CRIT-2, CRIT-3
   - MISS-4 (elevate IMP-5)

2. **Before developer starts:**
   - MISS-2 (TYPE_CHECKING docs)
   - MISS-3 (debugging guide)
   - ENH-7 (before you start)

3. **As time permits:**
   - MISS-1 (validation consolidation)
   - ENH-1 through ENH-8
