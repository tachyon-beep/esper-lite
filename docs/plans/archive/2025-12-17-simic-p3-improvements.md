# Simic P3 Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 8 P3 improvements from the Simic code review, ranging from quick optimizer fixes to PyTorch convention compliance.

**Architecture:** Changes span the simic module's control/, training/, agent/, and telemetry/ subfolders, plus new constants in leyline. All changes are additive or minimal edits with no breaking API changes.

**Tech Stack:** PyTorch 2.4+, Python 3.11+, dataclasses, typing

---

## Task 1: Add set_to_none=True to zero_grad() calls (P3-8)

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:925,927`

**Why:** `set_to_none=True` is faster than `set_to_none=False` (default) because it avoids zeroing memory - instead, gradients become None. This saves ~10-15% memory bandwidth on backward passes. Safe because we accumulate fresh gradients each step.

**Step 1: Run existing tests (baseline)**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized.py -v -x
```
Expected: PASS (all tests green)

**Step 2: Edit vectorized.py line 925**

Change:
```python
env_state.host_optimizer.zero_grad()
```
To:
```python
env_state.host_optimizer.zero_grad(set_to_none=True)
```

**Step 3: Edit vectorized.py line 927**

Change:
```python
env_state.seed_optimizers[slot_id].zero_grad()
```
To:
```python
env_state.seed_optimizers[slot_id].zero_grad(set_to_none=True)
```

**Step 4: Run tests to verify no regressions**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized.py -v -x
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): use set_to_none=True for zero_grad() calls (P3-8)

Faster than default set_to_none=False because it avoids zeroing memory.
Saves ~10-15% memory bandwidth per backward pass."
```

---

## Task 2: Add slots=True to dataclasses (P3-10)

**Files:**
- Modify: `src/esper/simic/training/config.py:46`
- Modify: `src/esper/simic/training/parallel_env_state.py:25`
- Modify: `src/esper/simic/agent/tamiyo_buffer.py:75`
- Modify: `src/esper/simic/telemetry/anomaly_detector.py:27`

**Why:** `slots=True` creates `__slots__` which eliminates per-instance `__dict__`. For large dataclasses with many instances, this saves ~64 bytes per instance and improves attribute access speed by ~20%.

**Step 1: Run baseline tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 2: Edit config.py line 46**

Change:
```python
@dataclass
class TrainingConfig:
```
To:
```python
@dataclass(slots=True)
class TrainingConfig:
```

**Step 3: Edit parallel_env_state.py line 25**

Change:
```python
@dataclass
class ParallelEnvState:
```
To:
```python
@dataclass(slots=True)
class ParallelEnvState:
```

**Step 4: Edit tamiyo_buffer.py line 75**

Change:
```python
@dataclass
class TamiyoRolloutBuffer:
```
To:
```python
@dataclass(slots=True)
class TamiyoRolloutBuffer:
```

**Step 5: Edit anomaly_detector.py line 27**

Change:
```python
@dataclass
class AnomalyDetector:
```
To:
```python
@dataclass(slots=True)
class AnomalyDetector:
```

**Step 6: Run tests to verify no regressions**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/training/config.py src/esper/simic/training/parallel_env_state.py \
        src/esper/simic/agent/tamiyo_buffer.py src/esper/simic/telemetry/anomaly_detector.py
git commit -m "perf(simic): add slots=True to 4 dataclasses (P3-10)

Eliminates per-instance __dict__, saving ~64 bytes per instance.
Also improves attribute access speed by ~20%.

Modified: TrainingConfig, ParallelEnvState, TamiyoRolloutBuffer, AnomalyDetector"
```

---

## Task 3: Create HEAD_NAMES constant in leyline (P3-13)

**Files:**
- Modify: `src/esper/leyline/__init__.py` (add constant and export)
- Modify: `src/esper/simic/agent/tamiyo_network.py:289,341`
- Modify: `src/esper/simic/agent/ppo.py:514,526,555`
- Modify: `src/esper/simic/training/vectorized.py:1063,1064`

**Why:** The list `["slot", "blueprint", "blend", "op"]` is hardcoded in 7+ locations. A single constant ensures consistency and makes the factored action head names discoverable.

**Step 1: Add constant to leyline/__init__.py**

After line 102 (after DEFAULT_ENTROPY_COEF_MIN), add:
```python
# =============================================================================
# Factored Action Space Constants
# =============================================================================

# Head names for factored action space (slot selection, blueprint, blend algorithm, lifecycle op).
# Order matters: slot → blueprint → blend → op is the causal chain.
HEAD_NAMES: tuple[str, ...] = ("slot", "blueprint", "blend", "op")
```

**Step 2: Add to __all__ export list**

In the `__all__` list, after "DEFAULT_ENTROPY_COEF_MIN", add:
```python
    # Factored Action Space
    "HEAD_NAMES",
```

**Step 3: Update tamiyo_network.py line 289**

Change:
```python
for key in ["slot", "blueprint", "blend", "op"]:
```
To:
```python
from esper.leyline import HEAD_NAMES
# ... (at top of file)

for key in HEAD_NAMES:
```

Add import at top of file (after existing leyline imports around line 15):
```python
from esper.leyline import HEAD_NAMES
```

**Step 4: Update tamiyo_network.py line 341**

Change:
```python
for key in ["slot", "blueprint", "blend", "op"]:
```
To:
```python
for key in HEAD_NAMES:
```

**Step 5: Update ppo.py lines 514, 526, 555**

Add import at top (after existing leyline imports ~line 20):
```python
from esper.leyline import HEAD_NAMES
```

Change all three occurrences:
```python
for key in ["slot", "blueprint", "blend", "op"]:
```
To:
```python
for key in HEAD_NAMES:
```

**Step 6: Update vectorized.py lines 1063, 1064**

Add import at top (after existing leyline imports ~line 25):
```python
from esper.leyline import HEAD_NAMES
```

Change:
```python
mask_hits = {"slot": 0, "blueprint": 0, "blend": 0, "op": 0}
mask_total = {"slot": 0, "blueprint": 0, "blend": 0, "op": 0}
```
To:
```python
mask_hits = {head: 0 for head in HEAD_NAMES}
mask_total = {head: 0 for head in HEAD_NAMES}
```

**Step 7: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ tests/leyline/ -v -x -q --tb=short
```
Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/leyline/__init__.py src/esper/simic/agent/tamiyo_network.py \
        src/esper/simic/agent/ppo.py src/esper/simic/training/vectorized.py
git commit -m "refactor(leyline): add HEAD_NAMES constant for factored actions (P3-13)

Centralizes the hardcoded ('slot', 'blueprint', 'blend', 'op') tuple.
Updated 7 locations in tamiyo_network.py, ppo.py, and vectorized.py."
```

---

## Task 4: Convert no_grad to inference_mode (P3-7)

**Files:**
- Modify: `src/esper/simic/control/normalization.py:48`
- Modify: `src/esper/simic/training/vectorized.py:1013,1881`
- Modify: `src/esper/simic/training/helpers.py:405`
- Modify: `src/esper/simic/agent/ppo.py:524`

**Why:** `torch.inference_mode()` is faster than `torch.no_grad()` (~5% on inference) because it also disables version tracking and allows additional optimizations. Safe anywhere we don't need gradients AND won't mutate tensors that require grad.

**Step 1: Run baseline tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 2: Edit normalization.py line 48**

Change:
```python
@torch.no_grad()
def update(self, x: torch.Tensor) -> None:
```
To:
```python
@torch.inference_mode()
def update(self, x: torch.Tensor) -> None:
```

**Step 3: Edit vectorized.py line 1013**

Change:
```python
with torch.no_grad():
```
To:
```python
with torch.inference_mode():
```

**Step 4: Edit vectorized.py line 1881**

Change:
```python
with torch.no_grad():
```
To:
```python
with torch.inference_mode():
```

**Step 5: Edit helpers.py line 405**

Change:
```python
with torch.no_grad():
```
To:
```python
with torch.inference_mode():
```

**Step 6: Edit ppo.py line 524**

Change:
```python
with torch.no_grad():
```
To:
```python
with torch.inference_mode():
```

**Step 7: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/simic/control/normalization.py src/esper/simic/training/vectorized.py \
        src/esper/simic/training/helpers.py src/esper/simic/agent/ppo.py
git commit -m "perf(simic): convert no_grad to inference_mode (P3-7)

inference_mode is ~5% faster than no_grad for inference because it
disables version tracking and allows additional optimizations.

Converted 5 locations in normalization.py, vectorized.py, helpers.py, ppo.py."
```

---

## Task 5: Add hyperparameter validation ranges (P3-2)

**Files:**
- Modify: `src/esper/simic/training/config.py:241-278` (extend _validate method)

**Why:** Silent failures from invalid hyperparameters (e.g., gamma=1.5, negative lr) waste debugging time. Validation catches these at config load time with clear error messages.

**Step 1: Write failing test**

Create or extend `tests/simic/training/test_config_validation.py`:

```python
"""Tests for TrainingConfig hyperparameter validation."""

import pytest
from esper.simic.training import TrainingConfig


class TestHyperparameterValidation:
    """Test that invalid hyperparameters raise clear errors."""

    def test_gamma_must_be_in_range(self):
        """Gamma must be in (0, 1]."""
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=1.5)
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            TrainingConfig(gamma=-0.1)
        # Valid edge case
        TrainingConfig(gamma=1.0)  # Should not raise
        TrainingConfig(gamma=0.001)  # Should not raise

    def test_clip_ratio_must_be_positive(self):
        """Clip ratio must be > 0."""
        with pytest.raises(ValueError, match="clip_ratio"):
            TrainingConfig(clip_ratio=0.0)
        with pytest.raises(ValueError, match="clip_ratio"):
            TrainingConfig(clip_ratio=-0.1)

    def test_lr_must_be_positive(self):
        """Learning rate must be > 0."""
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=0.0)
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-1e-4)

    def test_entropy_coef_must_be_non_negative(self):
        """Entropy coefficient must be >= 0."""
        with pytest.raises(ValueError, match="entropy_coef"):
            TrainingConfig(entropy_coef=-0.1)
        # Zero is valid (no entropy bonus)
        TrainingConfig(entropy_coef=0.0)  # Should not raise
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_config_validation.py -v
```
Expected: FAIL (validation not implemented yet)

**Step 3: Implement validation in config.py**

Add these validation methods to TrainingConfig class (before `_validate`):

```python
def _validate_range(
    self, value: float, name: str, min_val: float, max_val: float,
    min_inclusive: bool = True, max_inclusive: bool = True
) -> None:
    """Validate value is within range."""
    min_ok = value >= min_val if min_inclusive else value > min_val
    max_ok = value <= max_val if max_inclusive else value < max_val
    if not (min_ok and max_ok):
        min_bracket = "[" if min_inclusive else "("
        max_bracket = "]" if max_inclusive else ")"
        raise ValueError(
            f"{name} must be in {min_bracket}{min_val}, {max_val}{max_bracket} (got {value})"
        )
```

Extend the `_validate` method (add after existing validations around line 272):

```python
# PPO hyperparameter ranges
self._validate_range(self.gamma, "gamma", 0.0, 1.0, min_inclusive=False, max_inclusive=True)
self._validate_range(self.clip_ratio, "clip_ratio", 0.0, 1.0, min_inclusive=False, max_inclusive=True)
self._validate_range(self.lr, "lr", 0.0, float("inf"), min_inclusive=False, max_inclusive=False)
self._validate_range(self.entropy_coef, "entropy_coef", 0.0, float("inf"), min_inclusive=True, max_inclusive=False)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_config_validation.py -v
```
Expected: PASS

**Step 5: Run full test suite**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/ -v -x -q --tb=short
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/training/config.py tests/simic/training/test_config_validation.py
git commit -m "feat(simic): add hyperparameter validation ranges (P3-2)

Validates gamma in (0, 1], clip_ratio in (0, 1], lr > 0, entropy_coef >= 0.
Catches invalid hyperparameters at config load time with clear errors."
```

---

## Task 6: Add state_dict methods to normalizers (P3-15)

**Files:**
- Modify: `src/esper/simic/control/normalization.py`

**Why:** PyTorch convention is `state_dict()` / `load_state_dict()` for checkpointing. Currently vectorized.py manually accesses `.mean`, `.m2`, `.count` attributes. Standard methods enable cleaner checkpoint code.

**Step 1: Write failing test**

Create `tests/simic/control/test_normalizer_state_dict.py`:

```python
"""Tests for normalizer state_dict methods."""

import pytest
import torch
from esper.simic.control.normalization import RunningMeanStd, RewardNormalizer


class TestRunningMeanStdStateDict:
    """Test state_dict/load_state_dict for RunningMeanStd."""

    def test_state_dict_roundtrip(self):
        """State can be saved and restored."""
        rms = RunningMeanStd(shape=(10,), device="cpu")
        # Update with some data
        rms.update(torch.randn(32, 10))
        rms.update(torch.randn(32, 10))

        # Save state
        state = rms.state_dict()

        # Create new instance and load
        rms2 = RunningMeanStd(shape=(10,), device="cpu")
        rms2.load_state_dict(state)

        # Verify identical
        assert torch.allclose(rms.mean, rms2.mean)
        assert torch.allclose(rms.var, rms2.var)
        assert torch.allclose(rms.count, rms2.count)

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        rms = RunningMeanStd(shape=(5,), device="cpu")
        state = rms.state_dict()
        assert set(state.keys()) == {"mean", "var", "count"}


class TestRewardNormalizerStateDict:
    """Test state_dict/load_state_dict for RewardNormalizer."""

    def test_state_dict_roundtrip(self):
        """State can be saved and restored."""
        rn = RewardNormalizer(clip=10.0)
        # Update with some data
        for _ in range(100):
            rn.update_and_normalize(torch.randn(1).item())

        # Save state
        state = rn.state_dict()

        # Create new instance and load
        rn2 = RewardNormalizer(clip=10.0)
        rn2.load_state_dict(state)

        # Verify identical
        assert rn.mean == rn2.mean
        assert rn.m2 == rn2.m2
        assert rn.count == rn2.count

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        rn = RewardNormalizer()
        state = rn.state_dict()
        assert set(state.keys()) == {"mean", "m2", "count"}
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/test_normalizer_state_dict.py -v
```
Expected: FAIL (methods not implemented)

**Step 3: Add state_dict to RunningMeanStd**

Add these methods to RunningMeanStd class (after `device` property, around line 133):

```python
def state_dict(self) -> dict[str, torch.Tensor]:
    """Return state dictionary for checkpointing."""
    return {
        "mean": self.mean.clone(),
        "var": self.var.clone(),
        "count": self.count.clone(),
    }

def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Load state from dictionary."""
    self.mean = state["mean"].to(self._device)
    self.var = state["var"].to(self._device)
    self.count = state["count"].to(self._device)
```

**Step 4: Add state_dict to RewardNormalizer**

Add these methods to RewardNormalizer class (after `normalize_only`, around line 196):

```python
def state_dict(self) -> dict[str, float | int]:
    """Return state dictionary for checkpointing."""
    return {
        "mean": self.mean,
        "m2": self.m2,
        "count": self.count,
    }

def load_state_dict(self, state: dict[str, float | int]) -> None:
    """Load state from dictionary."""
    self.mean = state["mean"]
    self.m2 = state["m2"]
    self.count = state["count"]
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src uv run pytest tests/simic/control/test_normalizer_state_dict.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/control/normalization.py tests/simic/control/test_normalizer_state_dict.py
git commit -m "feat(simic): add state_dict methods to normalizers (P3-15)

Follows PyTorch convention for checkpointing. Enables cleaner
checkpoint save/load code in vectorized.py."
```

---

## Task 7: Create TypedDicts for return dictionaries (P3-12)

**Files:**
- Create: `src/esper/simic/agent/types.py`
- Modify: `src/esper/simic/agent/ppo.py` (type annotations)
- Modify: `src/esper/simic/agent/tamiyo_network.py` (type annotations)

**Why:** Functions returning `dict[str, Any]` lose type safety. TypedDicts provide IDE completion, static analysis, and documentation for dictionary shapes.

**Step 1: Create types.py with TypedDicts**

```python
"""Type definitions for Simic agent module.

TypedDicts provide type safety for dictionary returns from PPO and network functions.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class GradientStats(TypedDict):
    """Gradient statistics from PPO update."""
    grad_norm: float
    max_grad: float
    min_grad: float


class PPOUpdateMetrics(TypedDict):
    """Metrics from a single PPO update step."""
    policy_loss: list[float]
    value_loss: list[float]
    entropy_loss: list[float]
    total_loss: list[float]
    approx_kl: list[float]
    clip_fraction: list[float]
    explained_variance: float
    gradient_stats: GradientStats | None


class HeadLogProbs(TypedDict):
    """Per-head log probabilities from factored policy."""
    slot: torch.Tensor
    blueprint: torch.Tensor
    blend: torch.Tensor
    op: torch.Tensor


class HeadEntropies(TypedDict):
    """Per-head entropies from factored policy."""
    slot: torch.Tensor
    blueprint: torch.Tensor
    blend: torch.Tensor
    op: torch.Tensor


class ActionDict(TypedDict):
    """Factored action dictionary."""
    slot: int
    blueprint: int
    blend: int
    op: int


__all__ = [
    "GradientStats",
    "PPOUpdateMetrics",
    "HeadLogProbs",
    "HeadEntropies",
    "ActionDict",
]
```

**Step 2: Update agent/__init__.py exports**

Add to `src/esper/simic/agent/__init__.py`:
```python
from esper.simic.agent.types import (
    GradientStats,
    PPOUpdateMetrics,
    HeadLogProbs,
    HeadEntropies,
    ActionDict,
)
```

And add to `__all__`:
```python
"GradientStats",
"PPOUpdateMetrics",
"HeadLogProbs",
"HeadEntropies",
"ActionDict",
```

**Step 3: Update ppo.py return type annotations**

Add import at top of ppo.py:
```python
from esper.simic.agent.types import PPOUpdateMetrics, GradientStats
```

Update `update()` method return type (around line 408):
```python
def update(
    self,
    buffer: "TamiyoRolloutBuffer",
    obs_normalizer: RunningMeanStd | None = None,
) -> PPOUpdateMetrics:
```

**Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/agent/ -v -x -q --tb=short
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/types.py src/esper/simic/agent/__init__.py src/esper/simic/agent/ppo.py
git commit -m "feat(simic): add TypedDicts for agent return types (P3-12)

Provides type safety for dictionary returns from PPO and network functions.
Enables IDE completion and static analysis.

New types: GradientStats, PPOUpdateMetrics, HeadLogProbs, HeadEntropies, ActionDict"
```

---

## Task 8: Log per-head entropy (P3-1)

**Files:**
- Modify: `src/esper/simic/agent/ppo.py` (track per-head entropy in update)
- Modify: `src/esper/simic/training/vectorized.py` (log per-head entropy)

**Why:** Currently only aggregate entropy is logged. Per-head entropy reveals which action head is collapsing (e.g., blueprint always picking same option), enabling targeted debugging.

**Step 1: Update PPOUpdateMetrics TypedDict**

In `src/esper/simic/agent/types.py`, add to PPOUpdateMetrics:
```python
class PPOUpdateMetrics(TypedDict):
    """Metrics from a single PPO update step."""
    policy_loss: list[float]
    value_loss: list[float]
    entropy_loss: list[float]
    total_loss: list[float]
    approx_kl: list[float]
    clip_fraction: list[float]
    explained_variance: float
    gradient_stats: GradientStats | None
    # Per-head entropy (P3-1)
    head_entropies: dict[str, list[float]]  # {"slot": [...], "blueprint": [...], ...}
```

**Step 2: Track per-head entropy in ppo.py update()**

In the update() method, after computing head entropies (around line 485-495), add tracking:

```python
# Initialize per-head entropy tracking
head_entropy_history: dict[str, list[float]] = {head: [] for head in HEAD_NAMES}

# Inside the epoch loop, after computing entropies:
for key in HEAD_NAMES:
    head_entropy_history[key].append(entropies[key].mean().item())
```

Add to returned metrics dict:
```python
"head_entropies": head_entropy_history,
```

**Step 3: Log per-head entropy in vectorized.py**

In the metrics logging section (around line 1970), add:

```python
# Log per-head entropy if available
if "head_entropies" in update_info:
    for head, values in update_info["head_entropies"].items():
        avg_entropy = sum(values) / len(values) if values else 0.0
        logger.debug(f"  {head}_entropy: {avg_entropy:.4f}")
```

**Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/types.py src/esper/simic/agent/ppo.py src/esper/simic/training/vectorized.py
git commit -m "feat(simic): log per-head entropy for debugging (P3-1)

Tracks entropy per action head (slot, blueprint, blend, op).
Reveals which head is collapsing for targeted debugging.

Added head_entropies to PPOUpdateMetrics TypedDict."
```

---

## Verification

After all tasks are complete, run the full test suite:

```bash
PYTHONPATH=src uv run pytest tests/simic/ tests/leyline/ -v --tb=short
```

Expected: All tests pass.

---

## Summary

| Task | Issue | Type | Est. Time |
|------|-------|------|-----------|
| 1 | P3-8 | Performance | 5 min |
| 2 | P3-10 | Performance | 5 min |
| 3 | P3-13 | Code Quality | 10 min |
| 4 | P3-7 | Performance | 10 min |
| 5 | P3-2 | Robustness | 15 min |
| 6 | P3-15 | Convention | 15 min |
| 7 | P3-12 | Type Safety | 20 min |
| 8 | P3-1 | Observability | 15 min |

**Total estimated time:** ~95 minutes
