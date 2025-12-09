# Architect Handover Report: esper-lite

## Purpose

This document provides the handover materials for an architect to assess and plan improvements to the esper-lite codebase. It synthesizes findings from the architecture analysis into actionable improvement recommendations.

---

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

### Project Structure Note

All Python modules are under `src/esper/`. When this document references paths like `simic/training.py`, the full path is `src/esper/simic/training.py`.

### Recommended First Steps

1. Read CLAUDE.md for project coding guidelines
2. Run test suite to establish baseline
3. Pick one P1 improvement to start with
4. Create a branch: `git checkout -b improvement/IMP-X-description`

---

## 1. Architecture Health Summary

### 1.1 Overall Assessment

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Modularity** | A | Clean 9-subsystem domain-driven design |
| **Coupling** | A- | Mostly loose, some structural coupling |
| **Cohesion** | A | Each subsystem has clear single responsibility |
| **Testability** | B+ | Good coverage, some gaps in Tamiyo |
| **Documentation** | A | Excellent docstrings and type hints |
| **Technical Debt** | A- | Minimal, 5 feature TODOs only |

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Scripts                           │   │
│  │              (train.py, evaluate.py)                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Core Subsystems                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Kasmina  │  │  Tamiyo  │  │ Tolaria  │  │  Simic   │    │
│  │  (Body)  │  │  (Brain) │  │ (Hands)  │  │  (Gym)   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Leyline  │  │  Nissa   │  │ Runtime  │  │  Utils   │    │
│  │(Contracts)│ │(Telemetry)│ │ (Config) │  │(Helpers) │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Improvement Backlog

### 2.1 Priority Matrix

| ID | Improvement | Impact | Effort | Priority |
|----|-------------|--------|--------|----------|
| IMP-1 | Extract training loop helper | High | Low | P1 |
| IMP-2 | Add Tamiyo integration tests | High | Medium | P1 |
| IMP-5 | Document and validate reward shaping | High | Medium | P1 |
| IMP-3 | Consolidate loss computation | Medium | Low | P2 |
| IMP-4 | Parameterize learning rates | Medium | Low | P2 |
| IMP-6 | Extract common validation loop | Medium | Low | P2 |

### 2.2 Detailed Improvement Plans

---

#### IMP-1: Extract Training Loop Helper

**Status:** Ready for implementation
**Effort:** 2-4 hours
**Risk:** Low

**Current State:**
`simic/training.py` contains the same inline training loop pattern repeated 5 times (lines 189-304) across different seed stages. Each loop:
1. Iterates over trainloader
2. Calls `_loss_and_correct()` which returns `tuple[torch.Tensor, float, int]`
3. Accumulates `running_loss`, `correct`, and `total` as local variables
4. Uses these to compute `train_loss` and `train_acc` (lines 305-306)

The loops are inline within `run_ppo_episode()` and do NOT currently return values.

**Target State:**
Extract the repeated inline loop into a `_train_one_epoch()` helper that returns the accumulated values.

**Implementation Steps:**

1. Create helper function that encapsulates the inline loop:
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

    This function extracts the repeated inline loop pattern. Callers use
    returned values to compute metrics:
        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

    Returns:
        Tuple of (running_loss, correct_count, total_count)
        - running_loss: Sum of loss.item() across batches (float)
        - correct_count: Sum of correct predictions (float, from _loss_and_correct)
        - total_count: Total samples processed (int)
    """
    running_loss = 0.0
    correct = 0.0
    total = 0
    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss, correct_batch, batch_total = _loss_and_correct(outputs, targets, criterion, task_type)
        loss.backward()
        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()
        running_loss += loss.item()
        correct += correct_batch
        total += batch_total
    return running_loss, correct, total
```

2. Replace 5 inline loops with calls to helper:
```python
# Before (inline, repeated 5x):
running_loss = 0.0
correct = 0
total = 0
for inputs, targets in trainloader:
    ...
train_loss = running_loss / len(trainloader)
train_acc = 100.0 * correct / total

# After (single call):
running_loss, correct, total = _train_one_epoch(model, trainloader, ...)
train_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
train_acc = 100.0 * correct / total if total > 0 else 0.0
```

3. Add `collect_gradients` parameter for TRAINING stage gradient telemetry

4. Run existing tests to verify behavior preserved

**Verification:**
- All tests in `tests/simic/` pass
- Training produces identical loss/accuracy values
- Type checker validates: `mypy src/esper/simic/training.py`

---

#### IMP-2: Add Tamiyo Integration Tests

**Status:** Ready for implementation
**Effort:** 4-8 hours
**Risk:** Low

**Current State:**
Limited test coverage for Tamiyo decision logic and heuristic policy.
Existing Tamiyo test is at root level: `tests/test_tamiyo_tracker.py`

**Target State:**
Comprehensive integration tests covering decision correctness across stages.
Consistent directory structure matching other subsystems.

**Implementation Steps:**

1. Create `tests/tamiyo/` directory structure:
```
tests/tamiyo/
├── __init__.py
├── test_tracker.py              # Move from tests/test_tamiyo_tracker.py
├── test_heuristic_decisions.py  # NEW: HeuristicTamiyo decision tests
└── test_decision_integration.py # NEW: Integration tests
```

2. Test scenarios:
```python
class TestHeuristicDecisions:
    def test_germination_when_stable(self): ...
    def test_no_germination_when_unstable(self): ...
    def test_cull_on_performance_drop(self): ...
    def test_fossilize_on_improvement(self): ...
    def test_wait_during_cooldown(self): ...
```

3. Test SignalTracker:
```python
class TestSignalTracker:
    def test_plateau_detection(self): ...
    def test_stabilization_latch(self): ...
    def test_accuracy_history(self): ...
```

4. Add property-based tests using hypothesis

**Verification:**
- All new tests pass
- Coverage report shows >80% for tamiyo module

---

#### IMP-3: Consolidate Loss Computation

**Status:** Ready for implementation
**Effort:** 1-2 hours
**Risk:** Low

**Current State:**
Two loss computation functions with different purposes:
- `simic/training.py:_loss_and_correct()` → returns `tuple[torch.Tensor, float, int]` (loss, correct, total)
- `tolaria/trainer.py:_compute_loss()` → returns `torch.Tensor` (loss only)

**Target State:**
Two related functions with clear purposes (NOT a single function - they have different return arities).

**Implementation Steps:**

1. Create new file `leyline/loss.py` (or `utils/loss.py` if leyline should stay PyTorch-free):
```python
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
    """
    loss = compute_task_loss(outputs, targets, criterion, task_type)
    # ... accuracy computation
    return loss, correct, total
```

2. Update imports:
   - `simic/training.py` uses `compute_task_loss_with_metrics`
   - `tolaria/trainer.py` uses `compute_task_loss`

3. Remove duplicate functions

4. Run tests to verify

**Verification:**
- Tests pass
- Single source of truth for loss computation logic
- Callers choose appropriate function for their needs

---

#### IMP-4: Parameterize Learning Rates

**Status:** Ready for implementation
**Effort:** 1-2 hours
**Risk:** Low

**Current State:**
Learning rates hardcoded as `lr=0.01` in simic/training.py.

**Target State:**
Learning rates configurable via TaskConfig.

**Implementation Steps:**

1. Add to `simic/features.py` TaskConfig:
```python
@dataclass
class TaskConfig:
    # existing fields...
    host_lr: float = 0.01
    seed_lr: float = 0.01
```

2. Update optimizer creation in training.py to use config values

3. Update task presets in runtime/tasks.py

**Verification:**
- Tests pass
- Different LRs can be specified per task

---

#### IMP-5: Document and Validate Reward Shaping (ELEVATED TO P1)

**Status:** Ready for implementation
**Effort:** 2-4 hours
**Risk:** Low (documentation) to Medium (validation tests)
**Priority:** P1 (elevated from P3)

**Rationale for Elevation:**
Reward shaping is the most critical design decision in RL. Undocumented reward functions are a maintenance nightmare and debugging blind spot.

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
# Value rationale (actual values):
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
    """Verify PBRS telescoping: sum(F) = gamma * phi(s_T) - phi(s_0).

    Note: When summing undiscounted shaping rewards F(s,s') = gamma*phi(s') - phi(s),
    intermediate terms cancel, leaving: gamma*phi(final) - phi(initial).
    """
    gamma = 0.99
    total_shaping = sum(compute_pbrs_bonus(phi(s), phi(s_next), gamma)
                        for s, s_next in pairwise(trajectories))
    # Telescoping: intermediate phi terms cancel out
    expected = gamma * phi(trajectories[-1]) - phi(trajectories[0])
    assert abs(total_shaping - expected) < 1e-6
```

3. Add reward component telemetry (already exists, ensure documented)

**Verification:**
- Property-based tests pass
- Documentation review complete
- Telemetry shows individual reward components

---

#### IMP-6: Extract Common Validation Loop

**Status:** Ready for implementation
**Effort:** 2-3 hours
**Risk:** Low

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

---

## 3. Architectural Decisions Log

### 3.1 Existing ADRs (Implied)

| Decision | Rationale | Status |
|----------|-----------|--------|
| Domain-driven 9-subsystem design | Clear separation of concerns | Active |
| Leyline as foundation layer | Centralized contracts prevent coupling | Active |
| TYPE_CHECKING imports | Avoid circular dependencies | Active |
| Plugin architecture for blueprints | Extensibility without core changes | Active |
| Telemetry-first design | Observability for RL training | Active |

### 3.2 Recommended Future ADRs

1. **ADR: Utils Module Growth Strategy**
   - When to split utils
   - Module organization if expanding

2. **ADR: Simic Expansion Strategy**
   - Guidelines for adding RL algorithms
   - When to consider subsystem split

---

## 4. Knowledge Transfer

### 4.1 Critical Code Paths

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

### 4.2 Gotchas and Non-Obvious Behaviors

1. **SeedSlot.forward() is not torch.compile compatible** due to stage-dependent control flow

2. **Lazy imports in leyline.actions** - BlueprintRegistry imported inside function to avoid circular deps

3. **Counterfactual validation** uses alpha=0 baseline, not simply disabling seed

4. **CUDA streams in vectorized training** - Environments run async, careful with synchronization

5. **TYPE_CHECKING Import Pattern** - Several modules use conditional imports to avoid circular dependencies:
   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from esper.simic.features import TaskConfig
   ```
   This pattern allows type hints without runtime imports. Used in:
   - `kasmina/slot.py` (imports TaskConfig)
   - `tamiyo/heuristic.py` (imports SeedState)
   - `simic/features.py` (imports various types)

   **Do not "fix" these by making them regular imports** - this will create circular import errors.

### 4.3 Testing Guidance

| Module | Test Type | Location |
|--------|-----------|----------|
| Kasmina | Unit + Integration | tests/kasmina/ |
| Simic | Unit + Property | tests/simic/, tests/properties/ |
| Leyline | Unit | tests/leyline/ |
| Integration | E2E | tests/integration/ |

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
hub.add_backend(ConsoleOutput(min_severity="DEBUG"))
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

---

## 5. Handover Checklist

### For Receiving Architect:

- [ ] Review 01-discovery-findings.md for system overview
- [ ] Review 02-subsystem-catalog.md for detailed module docs
- [ ] Review 03-diagrams.md for visual architecture
- [ ] Review 05-quality-assessment.md for code quality details
- [ ] Prioritize improvements from Section 2.1
- [ ] Decide on ADRs from Section 3.2

### Recommended First Actions:

1. Run test suite to verify baseline
2. Implement IMP-1 (training loop extraction) as quick win
3. Add Tamiyo tests before any decision logic changes

---

## 6. Contact and Resources

### Documentation Index

| Document | Purpose |
|----------|---------|
| CLAUDE.md | Project coding guidelines |
| AGENTS.md | Agent configuration |
| README.md | Project overview and quick start |
| ROADMAP.md | Feature roadmap |

### Analysis Workspace

All analysis artifacts in: `docs/arch-analysis-2025-12-09-0521/`

---

*Generated: 2025-12-09*
*Analysis Type: Architect-Ready (Option C)*
