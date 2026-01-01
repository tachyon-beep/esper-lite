# Tolaria Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract Model Alpha training infrastructure from simic_overnight.py into Tolaria subsystem and Utils, then delete the legacy offline training file.

**Architecture:**
- Tolaria owns generic Model Alpha training (epoch loops, validation)
- Utils provides dataset loading (CIFAR-10 today, others later)
- simic_overnight.py is deleted (legacy offline workflow replaced by online PPO)

**Tech Stack:** PyTorch, torchvision, CIFAR-10

**Review Status:**
- DRL Expert: APPROVED - Clean environment/agent separation
- Code Reviewer: APPROVED with fixes incorporated below

---

## Background

`simic_overnight.py` was the old offline training workflow. Now that we have online PPO (`simic/ppo.py`), it's legacy code. However, it contains reusable training infrastructure that should be extracted before deletion:

**Extract to Tolaria:** `create_model`, `_train_epoch_*`, `_validate_and_get_metrics`
**Extract to Utils:** `load_cifar10`
**Delete:** Everything else (offline episode generation, policy training, evaluation, comparison)

---

## Task 1: Create Utils Package with Data Loading

**Files:**
- Create: `src/esper/utils/__init__.py`
- Create: `src/esper/utils/data.py`

**Step 1: Create utils directory**

```bash
mkdir -p src/esper/utils
```

**Step 2: Create utils/__init__.py**

Create `src/esper/utils/__init__.py`:

```python
"""Esper Utils - Shared utilities (the bit bucket).

This package contains shared utilities that don't belong to any specific
domain subsystem:

- data: Dataset loading utilities (CIFAR-10, future datasets)
"""

from esper.utils.data import load_cifar10

__all__ = [
    "load_cifar10",
]
```

**Step 3: Create utils/data.py**

Create `src/esper/utils/data.py`:

```python
"""Data loading utilities.

Dataset loaders for training Model Alpha. Currently supports CIFAR-10,
with room to grow for ImageNet, synthetic datasets, etc.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def load_cifar10(
    batch_size: int = 128,
    generator: torch.Generator | None = None,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size for DataLoaders.
        generator: Optional torch.Generator for reproducible shuffling.
            Use different generators per environment to avoid GIL contention
            when multiple CUDA streams iterate shared DataLoaders.
        data_root: Root directory for dataset storage.

    Returns:
        Tuple of (trainloader, testloader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=generator,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader
```

**Step 4: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.utils import load_cifar10; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/esper/utils/
git commit -m "feat(utils): create utils package with CIFAR-10 loader

New shared utilities package (the bit bucket) for code that doesn't
belong to domain subsystems. Starts with load_cifar10() for dataset
loading, with room to grow for other datasets."
```

---

## Task 2: Create Tolaria Package Structure

**Files:**
- Create: `src/esper/tolaria/__init__.py`
- Create: `src/esper/tolaria/environment.py`
- Create: `src/esper/tolaria/trainer.py`

**Step 1: Create tolaria directory**

```bash
mkdir -p src/esper/tolaria
```

**Step 2: Create tolaria/__init__.py**

Create `src/esper/tolaria/__init__.py`:

```python
"""Tolaria - Model Alpha Training Infrastructure

This package owns the training loop for Model Alpha (the neural network being
enhanced with morphogenetic seeds). It provides:

- environment: Model factory
- trainer: Epoch training functions for different seed states

Tolaria is a generic trainer - dataset loading is handled by esper.utils.
Tolaria is tightly coupled with Kasmina (seed mechanics) and Tamiyo (decisions).
Simic uses Tolaria to create the RL environment for training Tamiyo.

Public API:
    from esper.tolaria import create_model
    from esper.tolaria import train_epoch_normal, train_epoch_seed_isolated
"""

from esper.tolaria.environment import create_model
from esper.tolaria.trainer import (
    train_epoch_normal,
    train_epoch_seed_isolated,
    train_epoch_blended,
    validate_and_get_metrics,
)

__all__ = [
    # Environment
    "create_model",
    # Trainer
    "train_epoch_normal",
    "train_epoch_seed_isolated",
    "train_epoch_blended",
    "validate_and_get_metrics",
]
```

**Step 3: Commit package structure**

```bash
git add src/esper/tolaria/__init__.py
git commit -m "feat(tolaria): create package structure

New subsystem for Model Alpha training infrastructure.
Tolaria owns the training loop, Simic imports from Tolaria for RL environment."
```

---

## Task 3: Create tolaria/environment.py (Model Factory)

**Files:**
- Create: `src/esper/tolaria/environment.py`

**Step 1: Create environment.py**

Create `src/esper/tolaria/environment.py`:

```python
"""Tolaria Environment - Model factory for Model Alpha.

This module provides the model factory for creating Model Alpha instances.
Dataset loading is handled separately by esper.utils.data.
"""

from __future__ import annotations

import torch

from esper.kasmina import HostCNN, MorphogeneticModel


def create_model(device: str = "cuda") -> MorphogeneticModel:
    """Create a MorphogeneticModel with HostCNN.

    Args:
        device: Target device (cuda/cpu).

    Returns:
        Initialized MorphogeneticModel on the specified device.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    host = HostCNN(num_classes=10)
    model = MorphogeneticModel(host, device=device)
    return model.to(device)
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.tolaria.environment import create_model; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/tolaria/environment.py
git commit -m "feat(tolaria): add environment module with model factory

create_model() creates MorphogeneticModel with HostCNN.
Includes CUDA availability validation."
```

---

## Task 4: Create tolaria/trainer.py (Epoch Training Functions)

**Files:**
- Create: `src/esper/tolaria/trainer.py`

**Step 1: Create trainer.py**

Create `src/esper/tolaria/trainer.py`:

```python
"""Tolaria Trainer - Epoch training functions for Model Alpha.

This module provides the core training loop functions for different seed states:
- train_epoch_normal: Standard training without seed
- train_epoch_seed_isolated: Seed-only training (host frozen)
- train_epoch_blended: Joint host+seed training
- validate_and_get_metrics: Validation and metric computation

These functions are generic and work with any DataLoader, not tied to CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    """Train one epoch without seed.

    Standard training loop - all model parameters updated.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        device: Device to train on.
    """
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def train_epoch_seed_isolated(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    """Train one epoch with seed in isolation (only seed weights updated).

    RL Context: During TRAINING stage, only seed parameters update.
    The RL agent observes seed_improvement and decides when to ADVANCE.

    Note: We don't freeze host params because that breaks gradient flow.
    Instead, we just don't step the host optimizer - gradients flow through
    but host weights stay fixed.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        seed_optimizer: Optimizer for seed parameters only.
        device: Device to train on.
    """
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero ALL grads to prevent accumulation
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Only step seed optimizer - host grads computed but not applied
        seed_optimizer.step()


def train_epoch_blended(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
) -> None:
    """Train one epoch with blended host+seed.

    Used during BLENDING and FOSSILIZED stages when both host and seed
    parameters are updated together.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        host_optimizer: Optimizer for host parameters.
        seed_optimizer: Optimizer for seed parameters (optional).
        device: Device to train on.
    """
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        host_optimizer.zero_grad()
        if seed_optimizer:
            seed_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()


def validate_and_get_metrics(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    compute_per_class: bool = False,
    num_classes: int = 10,
) -> tuple[float, float, float, float, dict[int, float] | None]:
    """Get validation and training metrics.

    Args:
        model: The model to evaluate.
        trainloader: Training data loader (for quick train metrics).
        testloader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        compute_per_class: If True, compute per-class accuracy (for telemetry).
        num_classes: Number of classes in the dataset.

    Returns:
        Tuple of (val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc)
        per_class_acc is None if compute_per_class is False.
    """
    model.eval()

    # Validation
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Per-class tracking
    if compute_per_class:
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            if compute_per_class:
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

    val_loss /= len(testloader) if len(testloader) > 0 else 1
    val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

    per_class_acc = None
    if compute_per_class:
        per_class_acc = {
            i: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            for i in range(num_classes)
        }

    # Training metrics (quick sample)
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 10:  # Sample first 10 batches
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

    train_loss /= min(10, len(trainloader)) if len(trainloader) > 0 else 1
    train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

    return val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.tolaria.trainer import train_epoch_normal, validate_and_get_metrics; print('OK')"
```

Expected: `OK`

**Step 3: Verify package-level import works**

```bash
PYTHONPATH=src python -c "from esper.tolaria import create_model, train_epoch_normal; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/tolaria/trainer.py
git commit -m "feat(tolaria): add trainer module with epoch functions

Generic epoch training functions for different seed states:
- train_epoch_normal: Standard training without seed
- train_epoch_seed_isolated: Seed-only training (host frozen)
- train_epoch_blended: Joint host+seed training
- validate_and_get_metrics: Validation and metric computation

Functions are dataset-agnostic, work with any DataLoader."
```

---

## Task 5: Update simic/ppo.py Imports

**Files:**
- Modify: `src/esper/simic/ppo.py`

**Step 1: Replace all simic_overnight imports**

Use sed to replace all occurrences:

```bash
sed -i 's/from esper\.simic_overnight import create_model/from esper.tolaria import create_model/g' src/esper/simic/ppo.py
sed -i 's/from esper\.simic_overnight import load_cifar10/from esper.utils import load_cifar10/g' src/esper/simic/ppo.py
sed -i 's/from esper\.simic_overnight import load_cifar10, create_model/from esper.tolaria import create_model\n    from esper.utils import load_cifar10/g' src/esper/simic/ppo.py
```

**Step 2: Manually verify and fix the combined import**

The last sed may not handle the combined import cleanly. Check and fix manually if needed:

Find:
```python
from esper.simic_overnight import load_cifar10, create_model
```

Replace with:
```python
from esper.tolaria import create_model
from esper.utils import load_cifar10
```

**Step 3: Verify no simic_overnight imports remain**

```bash
grep "simic_overnight" src/esper/simic/ppo.py
```

Expected: No output

**Step 4: Verify ppo imports work**

```bash
PYTHONPATH=src python -c "from esper.simic.ppo import PPOAgent; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): update ppo.py to import from tolaria/utils

Replace simic_overnight imports:
- create_model -> esper.tolaria
- load_cifar10 -> esper.utils"
```

---

## Task 6: Update simic/iql.py Imports

**Files:**
- Modify: `src/esper/simic/iql.py`

**Step 1: Replace all simic_overnight imports**

```bash
sed -i 's/from esper\.simic_overnight import create_model/from esper.tolaria import create_model/g' src/esper/simic/iql.py
```

**Step 2: Verify no simic_overnight imports remain**

```bash
grep "simic_overnight" src/esper/simic/iql.py
```

Expected: No output

**Step 3: Verify iql imports work**

```bash
PYTHONPATH=src python -c "from esper.simic.iql import IQLAgent; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/iql.py
git commit -m "refactor(simic): update iql.py to import from tolaria

Replace simic_overnight imports:
- create_model -> esper.tolaria"
```

---

## Task 7: Delete simic_overnight.py

**Files:**
- Delete: `src/esper/simic_overnight.py`

**Step 1: Archive to _archive (optional, for reference)**

```bash
mkdir -p _archive/legacy
mv src/esper/simic_overnight.py _archive/legacy/simic_overnight.py
```

**Step 2: Verify no imports of simic_overnight anywhere**

```bash
grep -r "simic_overnight" src/esper/
```

Expected: No output

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: archive legacy simic_overnight.py

The offline training workflow has been replaced by online PPO.
Reusable code extracted to:
- esper.tolaria (model factory, training loops)
- esper.utils (dataset loading)

Archived to _archive/legacy/ for reference."
```

---

## Task 8: Run Full Verification

**Step 1: Verify all package imports**

```bash
PYTHONPATH=src python -c "
from esper.utils import load_cifar10
print('utils OK')

from esper.tolaria import create_model
from esper.tolaria import train_epoch_normal, train_epoch_seed_isolated
from esper.tolaria import train_epoch_blended, validate_and_get_metrics
print('tolaria OK')

from esper.simic.ppo import PPOAgent
print('ppo OK')

from esper.simic.iql import IQLAgent
print('iql OK')

print('All imports successful!')
"
```

Expected:
```
utils OK
tolaria OK
ppo OK
iql OK
All imports successful!
```

**Step 2: Verify simic_overnight is gone from src**

```bash
ls src/esper/simic_overnight.py 2>&1
```

Expected: `ls: cannot access 'src/esper/simic_overnight.py': No such file or directory`

**Step 3: Run existing tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v --tb=short
```

Expected: All tests pass

---

## Task 9: Create Unit Tests

**Files:**
- Create: `tests/esper/test_utils.py`
- Create: `tests/esper/test_tolaria.py`

**Step 1: Create test_utils.py**

Create `tests/esper/test_utils.py`:

```python
"""Tests for esper.utils package."""

import pytest
from esper.utils import load_cifar10


class TestData:
    """Tests for utils.data module."""

    def test_load_cifar10_returns_loaders(self):
        """Test CIFAR-10 loading returns DataLoaders."""
        trainloader, testloader = load_cifar10(batch_size=32)
        assert len(trainloader) > 0
        assert len(testloader) > 0

    def test_load_cifar10_batch_size(self):
        """Test CIFAR-10 respects batch size."""
        trainloader, _ = load_cifar10(batch_size=64)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape[0] == 64
        assert labels.shape[0] == 64

    def test_load_cifar10_data_shape(self):
        """Test CIFAR-10 data has correct shape."""
        trainloader, _ = load_cifar10(batch_size=32)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape == (32, 3, 32, 32)  # CIFAR-10 is 32x32 RGB
```

**Step 2: Create test_tolaria.py**

Create `tests/esper/test_tolaria.py`:

```python
"""Tests for Tolaria subsystem."""

import itertools

import pytest
import torch

from esper.tolaria import (
    create_model,
    train_epoch_normal,
    validate_and_get_metrics,
)
from esper.utils import load_cifar10


class TestEnvironment:
    """Tests for tolaria.environment module."""

    def test_create_model_cpu(self):
        """Test model creation on CPU."""
        model = create_model(device="cpu")
        assert not next(model.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_cuda(self):
        """Test model creation on CUDA."""
        model = create_model(device="cuda")
        assert next(model.parameters()).is_cuda

    def test_create_model_invalid_cuda_raises(self):
        """Test error handling for invalid CUDA device."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")

        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            create_model(device="cuda")


class TestTrainer:
    """Tests for tolaria.trainer module."""

    @pytest.fixture
    def model_and_loader(self):
        """Create model and minimal data loader for testing."""
        model = create_model(device="cpu")
        trainloader, testloader = load_cifar10(batch_size=32)
        # Just use first 2 batches for speed
        mini_train = list(itertools.islice(trainloader, 2))
        mini_test = list(itertools.islice(testloader, 2))
        return model, mini_train, mini_test

    def test_train_epoch_normal_runs(self, model_and_loader):
        """Smoke test for train_epoch_normal."""
        model, mini_train, _ = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        train_epoch_normal(model, mini_train, criterion, optimizer, "cpu")

    def test_validate_and_get_metrics_returns_tuple(self, model_and_loader):
        """Test validate_and_get_metrics returns expected structure."""
        model, mini_train, mini_test = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()

        result = validate_and_get_metrics(
            model, mini_train, mini_test, criterion, "cpu"
        )

        assert len(result) == 5
        val_loss, val_acc, train_loss, train_acc, per_class = result
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 100
        assert per_class is None  # compute_per_class=False by default

    def test_validate_and_get_metrics_per_class(self, model_and_loader):
        """Test per-class accuracy computation."""
        model, mini_train, mini_test = model_and_loader
        criterion = torch.nn.CrossEntropyLoss()

        result = validate_and_get_metrics(
            model, mini_train, mini_test, criterion, "cpu",
            compute_per_class=True
        )

        _, _, _, _, per_class = result
        assert per_class is not None
        assert len(per_class) == 10  # CIFAR-10 has 10 classes
```

**Step 3: Run the new tests**

```bash
PYTHONPATH=src python -m pytest tests/esper/test_utils.py tests/esper/test_tolaria.py -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/esper/test_utils.py tests/esper/test_tolaria.py
git commit -m "test: add unit tests for utils and tolaria

Tests cover:
- load_cifar10 returns valid DataLoaders
- create_model works on CPU and CUDA
- create_model raises on missing CUDA
- train_epoch_normal smoke test
- validate_and_get_metrics return structure"
```

---

## Task 10: Update Architecture Documentation

**Files:**
- Modify: `docs/arch-analysis-2025-11-28-2223/02-subsystem-catalog.md`

**Step 1: Add Tolaria section**

Add new section documenting:
- Purpose: Generic Model Alpha training infrastructure
- Key components: environment.py (model factory), trainer.py (epoch functions)
- Public API: create_model, train_epoch_*, validate_and_get_metrics
- Dependencies: Kasmina (HostCNN, MorphogeneticModel)

**Step 2: Add Utils section**

Add section documenting:
- Purpose: Shared utilities (the bit bucket)
- Key components: data.py (dataset loaders)
- Public API: load_cifar10
- Future: ImageNet, synthetic datasets

**Step 3: Update Simic section**

Note that Simic now imports from Tolaria and Utils for environment setup.

**Step 4: Note simic_overnight.py deletion**

Document that legacy offline workflow was removed.

**Step 5: Commit**

```bash
git add docs/arch-analysis-2025-11-28-2223/02-subsystem-catalog.md
git commit -m "docs: add Tolaria and Utils to subsystem catalog

- Document Tolaria subsystem for Model Alpha training
- Document Utils package for shared utilities
- Update Simic to reflect new dependencies
- Note simic_overnight.py deletion"
```

---

## Task 11: Final Tag

**Step 1: Review git log**

```bash
git log --oneline -15
```

**Step 2: Tag the milestone**

```bash
git tag -a v1.0.1-tolaria -m "Extract Tolaria subsystem, add Utils, delete legacy simic_overnight"
```

---

## Summary

| Task | Purpose |
|------|---------|
| 1 | Create Utils package with load_cifar10 |
| 2 | Create Tolaria package structure |
| 3 | Create tolaria/environment.py (model factory) |
| 4 | Create tolaria/trainer.py (epoch functions) |
| 5 | Update simic/ppo.py imports |
| 6 | Update simic/iql.py imports |
| 7 | Delete simic_overnight.py (archive to _archive/) |
| 8 | Run full verification |
| 9 | Create unit tests |
| 10 | Update architecture documentation |
| 11 | Tag milestone |

**New file structure:**
```
src/esper/
├── utils/             # Bit bucket (shared utilities)
│   ├── __init__.py
│   └── data.py        # load_cifar10()
│
├── tolaria/           # Model Alpha trainer (generic)
│   ├── __init__.py
│   ├── environment.py # create_model()
│   └── trainer.py     # train_epoch_*, validate_and_get_metrics
│
└── simic_overnight.py # DELETED (archived to _archive/legacy/)
```
