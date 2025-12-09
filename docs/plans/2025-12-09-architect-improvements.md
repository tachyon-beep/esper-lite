# Architect Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the 6 code improvements (IMP-1 through IMP-6) identified in the architecture analysis.

**Architecture:** Extract repeated patterns into shared helpers, consolidate loss computation, add comprehensive Tamiyo tests, parameterize learning rates, document reward shaping with PBRS property tests, and extract validation loop helper.

**Tech Stack:** Python 3.13, PyTorch 2.9, pytest, hypothesis

---

## Task 1: Extract Training Loop Helper (IMP-1)

**Priority:** P1
**Files:**
- Modify: `src/esper/simic/training.py`
- Test: `tests/simic/test_training_helper.py` (new)

**Step 1: Write failing test for _train_one_epoch helper**

```python
# tests/simic/test_training_helper.py
"""Tests for extracted training loop helper."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestTrainOneEpoch:
    """Tests for _train_one_epoch helper function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 2)

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_returns_correct_tuple_types(self, simple_model, simple_dataloader):
        """Should return (float, float, int) tuple."""
        from esper.simic.training import _train_one_epoch

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        result = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="classification",
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        running_loss, correct, total = result
        assert isinstance(running_loss, float)
        assert isinstance(correct, float)
        assert isinstance(total, int)

    def test_accumulates_correctly(self, simple_model, simple_dataloader):
        """Should accumulate loss, correct, and total across batches."""
        from esper.simic.training import _train_one_epoch

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        running_loss, correct, total = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=optimizer,
            seed_optimizer=None,
            device="cpu",
            task_type="classification",
        )

        # Should process all 32 samples
        assert total == 32
        # Loss should be positive
        assert running_loss > 0
        # Correct should be between 0 and total
        assert 0 <= correct <= total

    def test_with_seed_optimizer(self, simple_model, simple_dataloader):
        """Should work with both host and seed optimizers."""
        from esper.simic.training import _train_one_epoch

        # Create a second "seed" module
        seed_module = nn.Linear(10, 10)
        criterion = nn.CrossEntropyLoss()
        host_optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        seed_optimizer = torch.optim.SGD(seed_module.parameters(), lr=0.01)

        result = _train_one_epoch(
            model=simple_model,
            trainloader=simple_dataloader,
            criterion=criterion,
            host_optimizer=host_optimizer,
            seed_optimizer=seed_optimizer,
            device="cpu",
            task_type="classification",
        )

        assert len(result) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_training_helper.py -v`
Expected: FAIL with "cannot import name '_train_one_epoch'"

**Step 3: Implement _train_one_epoch helper**

Add to `src/esper/simic/training.py` after line 108 (after `_loss_and_correct`):

```python
def _train_one_epoch(
    model: nn.Module,
    trainloader,
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

    Args:
        model: The model to train
        trainloader: Training data loader
        criterion: Loss function
        host_optimizer: Optimizer for host parameters
        seed_optimizer: Optimizer for seed parameters (optional)
        device: Device to train on
        task_type: "classification" or "lm"
        collect_gradients: If True, collect gradient stats (for telemetry)

    Returns:
        Tuple of (running_loss, correct_count, total_count)
        - running_loss: Sum of loss.item() across batches (float)
        - correct_count: Sum of correct predictions (float, from _loss_and_correct)
        - total_count: Total samples processed (int)
    """
    running_loss = 0.0
    correct = 0.0
    total = 0
    grad_stats = None

    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss, correct_batch, batch_total = _loss_and_correct(
            outputs, targets, criterion, task_type
        )
        loss.backward()

        if collect_gradients:
            grad_stats = collect_seed_gradients(model.get_seed_parameters())

        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()

        running_loss += loss.item()
        correct += correct_batch
        total += batch_total

    return running_loss, correct, total
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/simic/test_training_helper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/simic/test_training_helper.py src/esper/simic/training.py
git commit -m "feat(simic): extract _train_one_epoch helper

Extracts the repeated inline training loop pattern into a reusable
helper function. Returns (running_loss, correct, total) tuple for
callers to compute metrics.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Tamiyo Test Directory Structure (IMP-2)

**Priority:** P1
**Files:**
- Create: `tests/tamiyo/__init__.py`
- Move: `tests/test_tamiyo_tracker.py` → `tests/tamiyo/test_tracker.py`
- Create: `tests/tamiyo/test_heuristic_decisions.py`

**Step 1: Create directory and __init__.py**

```bash
mkdir -p tests/tamiyo
touch tests/tamiyo/__init__.py
```

**Step 2: Move existing test**

```bash
git mv tests/test_tamiyo_tracker.py tests/tamiyo/test_tracker.py
```

**Step 3: Verify moved test still passes**

Run: `pytest tests/tamiyo/test_tracker.py -v`
Expected: PASS

**Step 4: Write failing test for heuristic decisions**

```python
# tests/tamiyo/test_heuristic_decisions.py
"""Tests for HeuristicTamiyo decision logic."""

import pytest

from esper.leyline import SeedStage, TrainingSignals
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig


class MockSeedMetrics:
    """Mock seed metrics for testing."""

    def __init__(
        self,
        improvement_since_stage_start: float = 0.0,
        total_improvement: float = 0.0,
        counterfactual_contribution: float | None = None,
    ):
        self.improvement_since_stage_start = improvement_since_stage_start
        self.total_improvement = total_improvement
        self.counterfactual_contribution = counterfactual_contribution
        self.current_val_accuracy = 60.0
        self.accuracy_at_stage_start = 60.0 - improvement_since_stage_start


class MockSeedState:
    """Mock seed state for testing."""

    def __init__(
        self,
        seed_id: str = "test_seed",
        stage: SeedStage = SeedStage.TRAINING,
        epochs_in_stage: int = 1,
        alpha: float = 0.0,
        improvement: float = 0.0,
        total_improvement: float = 0.0,
        blueprint_id: str = "conv_light",
        counterfactual: float | None = None,
    ):
        self.seed_id = seed_id
        self.stage = stage
        self.epochs_in_stage = epochs_in_stage
        self.alpha = alpha
        self.blueprint_id = blueprint_id
        self.metrics = MockSeedMetrics(
            improvement_since_stage_start=improvement,
            total_improvement=total_improvement,
            counterfactual_contribution=counterfactual,
        )


class MockTrainingMetrics:
    """Mock training metrics for testing."""

    def __init__(
        self,
        epoch: int = 10,
        plateau_epochs: int = 0,
        host_stabilized: bool = True,
        accuracy_delta: float = 0.0,
    ):
        self.epoch = epoch
        self.plateau_epochs = plateau_epochs
        self.host_stabilized = host_stabilized
        self.accuracy_delta = accuracy_delta


class MockTrainingSignals:
    """Mock training signals for testing."""

    def __init__(self, metrics: MockTrainingMetrics | None = None):
        self.metrics = metrics or MockTrainingMetrics()


class TestGerminationDecisions:
    """Tests for germination decision logic."""

    def test_germinate_on_plateau_when_stabilized(self):
        """Should germinate when host is stabilized and plateauing."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=True,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name.startswith("GERMINATE_")
        assert "Plateau" in decision.reason

    def test_no_germinate_when_not_stabilized(self):
        """Should WAIT when host is not stabilized."""
        policy = HeuristicTamiyo(topology="cnn")
        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=False,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT"
        assert "not stabilized" in decision.reason.lower()

    def test_no_germinate_during_embargo(self):
        """Should WAIT during embargo period after cull."""
        policy = HeuristicTamiyo(topology="cnn")
        policy._last_cull_epoch = 8  # Culled 2 epochs ago

        signals = MockTrainingSignals(MockTrainingMetrics(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=True,
        ))

        decision = policy.decide(signals, active_seeds=[])

        assert decision.action.name == "WAIT"
        assert "Embargo" in decision.reason


class TestCullDecisions:
    """Tests for cull decision logic."""

    def test_cull_failing_seed_in_training(self):
        """Should CULL a seed that's failing in TRAINING stage."""
        config = HeuristicPolicyConfig(
            cull_after_epochs_without_improvement=3,
            cull_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-3.0,  # Dropped 3%
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=15))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "CULL"
        assert "Failing" in decision.reason

    def test_no_cull_improving_seed(self):
        """Should WAIT for a seed that's improving."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=2.0,  # Improving
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=15))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "WAIT"


class TestFossilizeDecisions:
    """Tests for fossilize decision logic."""

    def test_fossilize_contributing_seed(self):
        """Should FOSSILIZE a seed with positive contribution in PROBATIONARY."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.PROBATIONARY,
            epochs_in_stage=3,
            improvement=2.0,
            total_improvement=5.0,
            counterfactual=3.0,  # Contributing 3%
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=30))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "FOSSILIZE"
        assert "contribution" in decision.reason.lower()

    def test_cull_non_contributing_seed_in_probationary(self):
        """Should CULL a non-contributing seed in PROBATIONARY."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.PROBATIONARY,
            epochs_in_stage=3,
            improvement=-1.0,
            total_improvement=-2.0,
            counterfactual=-1.0,  # Hurting
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=30))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "CULL"


class TestWaitDecisions:
    """Tests for wait/patience decision logic."""

    def test_wait_during_blending(self):
        """Should WAIT during BLENDING stage (auto-advance handles it)."""
        policy = HeuristicTamiyo(topology="cnn")

        seed = MockSeedState(
            stage=SeedStage.BLENDING,
            epochs_in_stage=2,
            alpha=0.5,
            improvement=1.0,
        )
        signals = MockTrainingSignals(MockTrainingMetrics(epoch=20))

        decision = policy.decide(signals, active_seeds=[seed])

        assert decision.action.name == "WAIT"
        assert "Blending" in decision.reason
```

**Step 5: Run test to verify it passes (tests existing behavior)**

Run: `pytest tests/tamiyo/test_heuristic_decisions.py -v`
Expected: PASS (testing existing implementation)

**Step 6: Commit**

```bash
git add tests/tamiyo/
git commit -m "test(tamiyo): add comprehensive heuristic decision tests

- Creates tests/tamiyo/ directory structure matching other subsystems
- Moves test_tamiyo_tracker.py to tests/tamiyo/test_tracker.py
- Adds test_heuristic_decisions.py covering:
  - Germination: plateau detection, stabilization gate, embargo
  - Culling: failing seeds, protecting improving seeds
  - Fossilization: contribution-based decisions
  - Wait: patience during mechanical stages

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Consolidate Loss Computation (IMP-3)

**Priority:** P2
**Files:**
- Create: `src/esper/utils/loss.py`
- Modify: `src/esper/simic/training.py`
- Modify: `src/esper/tolaria/trainer.py`
- Test: `tests/test_utils_loss.py`

**Step 1: Write failing test for loss utilities**

```python
# tests/test_utils_loss.py
"""Tests for consolidated loss computation utilities."""

import pytest
import torch
import torch.nn as nn


class TestComputeTaskLoss:
    """Tests for compute_task_loss function."""

    def test_classification_loss(self):
        """Should compute CrossEntropyLoss for classification."""
        from esper.utils.loss import compute_task_loss

        outputs = torch.randn(8, 10)  # batch=8, classes=10
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "classification")

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_lm_loss(self):
        """Should reshape and compute loss for language modeling."""
        from esper.utils.loss import compute_task_loss

        outputs = torch.randn(4, 16, 1000)  # batch=4, seq=16, vocab=1000
        targets = torch.randint(0, 1000, (4, 16))
        criterion = nn.CrossEntropyLoss()

        loss = compute_task_loss(outputs, targets, criterion, "lm")

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestComputeTaskLossWithMetrics:
    """Tests for compute_task_loss_with_metrics function."""

    def test_returns_correct_types(self):
        """Should return (Tensor, float, int) tuple."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert isinstance(loss, torch.Tensor)
        assert isinstance(correct, float)
        assert isinstance(total, int)

    def test_correct_count_accuracy(self):
        """Should count correct predictions accurately."""
        from esper.utils.loss import compute_task_loss_with_metrics

        # Create outputs where we know the predictions
        outputs = torch.tensor([
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
            [10.0, 0.0],  # Predicts 0
            [0.0, 10.0],  # Predicts 1
        ])
        targets = torch.tensor([0, 1, 1, 0])  # 2 correct, 2 wrong
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "classification"
        )

        assert total == 4
        assert correct == 2.0

    def test_lm_correct_count(self):
        """Should count token-level correct predictions for LM."""
        from esper.utils.loss import compute_task_loss_with_metrics

        outputs = torch.randn(2, 4, 100)  # batch=2, seq=4, vocab=100
        targets = torch.randint(0, 100, (2, 4))
        criterion = nn.CrossEntropyLoss()

        loss, correct, total = compute_task_loss_with_metrics(
            outputs, targets, criterion, "lm"
        )

        assert total == 8  # 2 * 4 tokens
        assert 0 <= correct <= total
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils_loss.py -v`
Expected: FAIL with "No module named 'esper.utils.loss'"

**Step 3: Implement loss utilities**

```python
# src/esper/utils/loss.py
"""Loss computation utilities.

Consolidates loss computation patterns used across simic and tolaria.
Provides two functions:
- compute_task_loss: Returns loss tensor only (for validation)
- compute_task_loss_with_metrics: Returns loss + accuracy metrics (for training)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_task_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> torch.Tensor:
    """Compute loss for classification or language modeling task.

    Use when you only need the loss value (e.g., validation, inference).

    Args:
        outputs: Model outputs (batch, classes) or (batch, seq, vocab)
        targets: Ground truth labels
        criterion: Loss function (e.g., CrossEntropyLoss)
        task_type: "classification" or "lm"

    Returns:
        Loss tensor (scalar)
    """
    if task_type == "lm":
        vocab = outputs.size(-1)
        return criterion(outputs.view(-1, vocab), targets.view(-1))
    return criterion(outputs, targets)


def compute_task_loss_with_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, float, int]:
    """Compute loss and accuracy metrics for classification or LM task.

    Use when you need both loss and accuracy (e.g., training loops).

    Args:
        outputs: Model outputs (batch, classes) or (batch, seq, vocab)
        targets: Ground truth labels
        criterion: Loss function (e.g., CrossEntropyLoss)
        task_type: "classification" or "lm"

    Returns:
        Tuple of (loss, correct_count, total_count)
        - loss: CrossEntropyLoss tensor (for backprop)
        - correct_count: Number of correct predictions (float from .item())
        - total_count: Total samples in batch (int)
    """
    loss = compute_task_loss(outputs, targets, criterion, task_type)

    if task_type == "lm":
        predicted = outputs.argmax(dim=-1)
        correct = float((predicted == targets).sum().item())
        total = targets.numel()
    else:
        predicted = outputs.argmax(dim=1)
        correct = float((predicted == targets).sum().item())
        total = targets.size(0)

    return loss, correct, total


__all__ = [
    "compute_task_loss",
    "compute_task_loss_with_metrics",
]
```

**Step 4: Update utils __init__.py**

Add to `src/esper/utils/__init__.py`:

```python
from esper.utils.loss import compute_task_loss, compute_task_loss_with_metrics
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_utils_loss.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/utils/loss.py src/esper/utils/__init__.py tests/test_utils_loss.py
git commit -m "feat(utils): add consolidated loss computation utilities

Creates two functions with clear separation of concerns:
- compute_task_loss: Returns loss only (for validation)
- compute_task_loss_with_metrics: Returns (loss, correct, total) tuple

Both support classification and language modeling task types.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Parameterize Learning Rates (IMP-4)

**Priority:** P2
**Files:**
- Modify: `src/esper/simic/features.py` (add TaskConfig if not exists)
- Modify: `src/esper/simic/training.py` (use config LRs)
- Test: `tests/test_learning_rate_config.py`

**Step 1: Check if TaskConfig exists and add learning rate fields**

First, verify TaskConfig location. Add to existing TaskConfig or create if needed.

**Step 2: Write failing test**

```python
# tests/test_learning_rate_config.py
"""Tests for configurable learning rates."""

import pytest


class TestLearningRateConfig:
    """Tests for learning rate configuration in TaskConfig."""

    def test_taskconfig_has_lr_fields(self):
        """TaskConfig should have host_lr and seed_lr fields."""
        from esper.runtime.tasks import get_task_spec

        spec = get_task_spec("cifar10")

        # Should have learning rate attributes
        assert hasattr(spec, 'host_lr')
        assert hasattr(spec, 'seed_lr')
        # Default values
        assert spec.host_lr == 0.01
        assert spec.seed_lr == 0.01

    def test_custom_learning_rates(self):
        """Should be able to create spec with custom LRs."""
        from esper.runtime.tasks import TaskSpec

        spec = TaskSpec(
            name="custom",
            topology="cnn",
            task_type="classification",
            host_lr=0.001,
            seed_lr=0.005,
        )

        assert spec.host_lr == 0.001
        assert spec.seed_lr == 0.005
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_learning_rate_config.py -v`
Expected: FAIL with AttributeError

**Step 4: Add learning rate fields to TaskSpec**

Modify `src/esper/runtime/tasks.py` TaskSpec dataclass:

```python
@dataclass
class TaskSpec:
    """Task specification for training."""
    name: str
    topology: str  # "cnn" or "transformer"
    task_type: str  # "classification" or "lm"
    # ... existing fields ...

    # Learning rates (NEW)
    host_lr: float = 0.01
    seed_lr: float = 0.01
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_learning_rate_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/runtime/tasks.py tests/test_learning_rate_config.py
git commit -m "feat(runtime): add configurable learning rates to TaskSpec

Adds host_lr and seed_lr fields to TaskSpec with defaults of 0.01.
Enables task-specific learning rate configuration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Document and Validate Reward Shaping (IMP-5)

**Priority:** P1
**Files:**
- Modify: `src/esper/simic/rewards.py` (add documentation)
- Create: `tests/properties/test_pbrs_telescoping.py`

**Step 1: Write PBRS property test**

```python
# tests/properties/test_pbrs_telescoping.py
"""Property-based tests for PBRS telescoping guarantee.

PBRS (Ng et al., 1999) guarantees: F(s,a,s') = gamma * phi(s') - phi(s)
Over a trajectory, intermediate potentials cancel (telescoping):
    sum(F) = gamma^T * phi(s_T) - phi(s_0)
"""

import pytest
from hypothesis import given, strategies as st, assume

from esper.simic.rewards import (
    compute_seed_potential,
    compute_pbrs_bonus,
    STAGE_POTENTIALS,
)


class TestPBRSTelescopingProperty:
    """Tests that PBRS telescoping property holds."""

    @given(
        stages=st.lists(
            st.sampled_from([2, 3, 4, 5, 6, 7]),  # GERMINATED through FOSSILIZED
            min_size=2,
            max_size=10,
        )
    )
    def test_telescoping_for_stage_sequence(self, stages):
        """Sum of shaping rewards should telescope to final - initial potential."""
        gamma = 0.99

        # Build observations for each stage
        observations = [
            {
                "has_active_seed": 1,
                "seed_stage": stage,
                "seed_epochs_in_stage": 0,  # Simplify: just transitioned
            }
            for stage in stages
        ]

        # Compute potentials
        potentials = [compute_seed_potential(obs) for obs in observations]

        # Sum of PBRS bonuses
        total_shaping = sum(
            compute_pbrs_bonus(potentials[i], potentials[i + 1], gamma)
            for i in range(len(potentials) - 1)
        )

        # Telescoping property: sum = gamma * phi(final) - phi(initial)
        # Note: For N states, we have N-1 transitions
        expected = gamma * potentials[-1] - potentials[0]

        # Allow small numerical error
        assert abs(total_shaping - expected) < 1e-6, (
            f"Telescoping violated: got {total_shaping}, expected {expected}"
        )

    def test_stage_potentials_monotonic(self):
        """Stage potentials should be monotonically increasing toward FOSSILIZED."""
        stages = [2, 3, 4, 5, 6, 7]  # GERMINATED through FOSSILIZED
        for i in range(len(stages) - 1):
            current = STAGE_POTENTIALS[stages[i]]
            next_stage = STAGE_POTENTIALS[stages[i + 1]]
            assert next_stage >= current, (
                f"Potential should not decrease: stage {stages[i]}={current} "
                f"> stage {stages[i + 1]}={next_stage}"
            )

    def test_dormant_has_zero_potential(self):
        """DORMANT and UNKNOWN should have zero potential."""
        assert STAGE_POTENTIALS[0] == 0.0  # UNKNOWN
        assert STAGE_POTENTIALS[1] == 0.0  # DORMANT

    def test_fossilized_has_highest_potential(self):
        """FOSSILIZED should have the highest potential."""
        fossilized_potential = STAGE_POTENTIALS[7]  # FOSSILIZED
        for stage, potential in STAGE_POTENTIALS.items():
            assert fossilized_potential >= potential, (
                f"FOSSILIZED ({fossilized_potential}) should be >= stage {stage} ({potential})"
            )
```

**Step 2: Run test to verify it passes (testing existing implementation)**

Run: `pytest tests/properties/test_pbrs_telescoping.py -v`
Expected: PASS

**Step 3: Add comprehensive documentation to rewards.py**

Add after line 42 in `src/esper/simic/rewards.py`:

```python
# =============================================================================
# POTENTIAL-BASED REWARD SHAPING (PBRS) - DESIGN RATIONALE
# =============================================================================
#
# These values implement Ng et al. (1999) potential-based shaping:
#   F(s, s') = gamma * phi(s') - phi(s)
#
# KEY PROPERTIES MAINTAINED:
# 1. Telescoping: Sum of shaping rewards = gamma^T * phi(s_T) - phi(s_0)
#    Over a trajectory, intermediate potentials cancel, leaving only
#    terminal minus initial potential (discounted).
#
# 2. Policy Invariance: Optimal policy unchanged by shaping.
#    Adding PBRS to any reward function preserves the optimal policy
#    because the shaping is purely potential-based.
#
# VALUE RATIONALE (actual values):
# - UNKNOWN (0.0): Fallback/error state - no reward
# - DORMANT (0.0): Baseline state before germination - no reward
# - GERMINATED (1.0): +1.0 for initiating growth
# - TRAINING (2.0): +1.0 for successful G1 gate passage
# - BLENDING (3.5): +1.5 (LARGEST delta) - critical integration phase
#   This is where value is actually created; alpha ramp merges seed contribution
# - SHADOWING (4.5): +1.0 for surviving blending without regression
# - PROBATIONARY (5.5): +1.0 for stability validation
# - FOSSILIZED (6.0): +0.5 (SMALLEST delta) - terminal bonus
#   Small to prevent "fossilization farming" (rushing to completion)
#
# TUNING HISTORY:
# - v1: Linear progression (1.0 increments each stage)
#       Problem: Insufficient BLENDING incentive; seeds stalled at TRAINING
# - v2: Current values with BLENDING emphasis (+1.5)
#       Result: Improved seed integration success rate
#
# VALIDATION:
# Property-based tests in tests/properties/test_pbrs_telescoping.py verify:
# - Telescoping property holds for arbitrary stage sequences
# - Potentials are monotonically increasing toward FOSSILIZED
# - DORMANT/UNKNOWN have zero potential
# - FOSSILIZED has highest potential
```

**Step 4: Commit**

```bash
git add src/esper/simic/rewards.py tests/properties/test_pbrs_telescoping.py
git commit -m "docs(simic): comprehensive PBRS reward shaping documentation

Adds detailed documentation explaining:
- PBRS theory (Ng et al., 1999)
- Telescoping property
- Policy invariance guarantee
- Value rationale for each stage potential
- Tuning history

Adds property-based tests validating PBRS guarantees.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Extract Common Validation Loop (IMP-6)

**Priority:** P2
**Files:**
- Modify: `src/esper/tolaria/trainer.py`
- Test: `tests/test_tolaria_validation.py`

**Step 1: Write failing test for validation helper**

```python
# tests/test_tolaria_validation.py
"""Tests for extracted validation loop helper."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestRunValidationPass:
    """Tests for _run_validation_pass helper."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        return nn.Linear(10, 2)

    @pytest.fixture
    def testloader(self):
        """Create a simple test dataloader."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_returns_loss_and_accuracy(self, simple_model, testloader):
        """Should return (average_loss, accuracy) tuple."""
        from esper.tolaria.trainer import _run_validation_pass

        criterion = nn.CrossEntropyLoss()

        avg_loss, accuracy = _run_validation_pass(
            model=simple_model,
            testloader=testloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        assert isinstance(avg_loss, float)
        assert isinstance(accuracy, float)
        assert avg_loss > 0
        assert 0 <= accuracy <= 100

    def test_force_alpha_parameter(self, testloader):
        """Should support force_alpha for counterfactual validation."""
        from esper.tolaria.trainer import _run_validation_pass

        # Need a model with seed_slot for this test
        # This is an integration test that requires the full model
        pytest.skip("Requires full Tolaria model with seed_slot")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolaria_validation.py -v`
Expected: FAIL with "cannot import name '_run_validation_pass'"

**Step 3: Extract _run_validation_pass as module-level function**

In `src/esper/tolaria/trainer.py`, add after line 49 (after `_compute_loss`):

```python
def _run_validation_pass(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str,
) -> tuple[float, float]:
    """Run a single validation pass.

    Extracted from validate_with_attribution to enable reuse.

    Args:
        model: The model to evaluate
        testloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        task_type: "classification" or "lm"

    Returns:
        Tuple of (average_loss, accuracy)
    """
    loss_tensor = torch.tensor(0.0, device=device)
    correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = _compute_loss(outputs, labels, criterion, task_type)
            loss_tensor += loss

            if task_type == "lm":
                predicted = outputs.argmax(dim=-1)
                total += labels.numel()
                correct_tensor += (predicted == labels).sum()
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_tensor += predicted.eq(labels).sum()

    avg_loss = loss_tensor.item() / max(len(testloader), 1)
    accuracy = 100.0 * correct_tensor.item() / total if total > 0 else 0.0
    return avg_loss, accuracy
```

**Step 4: Update validate_with_attribution to use the helper**

In `src/esper/tolaria/trainer.py`, modify `validate_with_attribution` (around line 305):

```python
def validate_with_attribution(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str = "classification",
) -> AttributionResult:
    """Counterfactual validation for true seed contribution measurement."""
    was_training = model.training
    model.eval()

    try:
        # Pass 1: Real validation with current alpha
        real_loss, real_accuracy = _run_validation_pass(
            model, testloader, criterion, device, task_type
        )

        # Pass 2: Baseline validation with alpha=0 (host-only)
        seed_slot = model.seed_slot
        with seed_slot.force_alpha(0.0):
            baseline_loss, baseline_accuracy = _run_validation_pass(
                model, testloader, criterion, device, task_type
            )

        return AttributionResult(
            real_accuracy=real_accuracy,
            baseline_accuracy=baseline_accuracy,
            seed_contribution=real_accuracy - baseline_accuracy,
            real_loss=real_loss,
            baseline_loss=baseline_loss,
        )
    finally:
        model.train(was_training)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_tolaria_validation.py -v`
Expected: PASS

**Step 6: Run full tolaria test suite**

Run: `pytest tests/test_tolaria_trainer.py tests/esper/test_tolaria.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/tolaria/trainer.py tests/test_tolaria_validation.py
git commit -m "refactor(tolaria): extract _run_validation_pass helper

Extracts the common validation loop from validate_with_attribution
into a reusable helper function. Reduces code duplication and
enables easier testing of validation logic.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Final Verification

**Run full test suite:**

```bash
pytest tests/ -v --tb=short
```

**Verify type checking:**

```bash
mypy src/esper/simic/training.py src/esper/tolaria/trainer.py src/esper/utils/loss.py
```

---

## Summary

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 1 | Extract training loop helper | P1 | Ready |
| 2 | Create Tamiyo test directory | P1 | Ready |
| 3 | Consolidate loss computation | P2 | Ready |
| 4 | Parameterize learning rates | P2 | Ready |
| 5 | Document reward shaping + PBRS tests | P1 | Ready |
| 6 | Extract validation loop helper | P2 | Ready |
