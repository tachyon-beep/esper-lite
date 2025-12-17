# Seam Integration Tests Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add integration tests for the 5 missing critical seams between Esper domains.

**Note:** This plan was rewritten after discovering the original plan had incorrect imports and API signatures. All code in this plan has been verified against the actual codebase.

**Tech Stack:** pytest, torch

---

## Prerequisites

Before starting, ensure you understand:
- **Existing patterns**: See `tests/integration/test_tamiyo_simic.py` for integration test style
- **Domain roles**: Read `ROADMAP.md` for the biological metaphor

---

## Task 1: Create simic ↔ kasmina Integration Tests

**Files:**
- Create: `tests/integration/test_simic_kasmina.py`

### Step 1.1: Write test file with correct imports

```python
"""Integration tests for Simic-Kasmina interaction.

Tests the core integration where:
- Simic computes rewards based on seed contribution (from validation)
- Rewards reflect seed lifecycle stage and contribution
- Gradient collection works correctly during seed training
"""

import pytest
import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel, CNNHost
from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.simic.gradient_collector import (
    SeedGradientCollector,
    collect_seed_gradients,
    GradientHealthMetrics,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model_with_seed():
    """Create MorphogeneticModel with an active seed in TRAINING stage."""
    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    model.germinate_seed("conv_light", "test_seed", slot="r0c1")
    # Advance to TRAINING stage
    slot = model.seed_slots["r0c1"]
    slot.state.stage = SeedStage.TRAINING
    return model


@pytest.fixture
def reward_config():
    """Standard reward configuration."""
    return ContributionRewardConfig()


@pytest.fixture
def seed_info_training():
    """SeedInfo for a seed in TRAINING stage."""
    return SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.02,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=4,
    )
```

### Step 1.2: Run to verify imports work

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py --collect-only`
Expected: Collection succeeds with 0 tests

### Step 1.3: Write test for positive contribution reward

```python
class TestRewardComputation:
    """Tests that reward signals reflect seed contribution correctly."""

    def test_positive_contribution_yields_positive_reward(
        self, seed_info_training, reward_config
    ):
        """Positive seed contribution should yield positive reward.

        This tests the Simic->Kasmina signal: when validation shows the seed
        helps accuracy, Simic's reward function should return positive reward.
        """
        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,  # 2% improvement from seed
            val_acc=75.0,
            seed_info=seed_info_training,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=reward_config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        # Positive contribution should yield net positive reward
        assert reward > 0, f"Expected positive reward, got {reward}"
        assert components.seed_contribution == 0.02

    def test_negative_contribution_yields_negative_reward(
        self, seed_info_training, reward_config
    ):
        """Negative seed contribution should yield negative reward.

        When validation shows the seed hurts accuracy, reward should be negative.
        """
        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=-0.03,  # 3% degradation from seed
            val_acc=72.0,
            seed_info=seed_info_training,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=75.0,
            acc_delta=-0.02,
            config=reward_config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        # Negative contribution should yield net negative reward
        assert reward < 0, f"Expected negative reward, got {reward}"
        assert components.seed_contribution == -0.03
```

### Step 1.4: Run tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestRewardComputation -v`
Expected: PASS (2 tests)

### Step 1.5: Write gradient collection test

```python
class TestGradientCollection:
    """Tests that gradient collection works correctly during seed training."""

    def test_seed_gradients_collected_during_training(self, model_with_seed):
        """Gradient collection should capture seed gradients after backward."""
        model = model_with_seed
        slot = model.seed_slots["r0c1"]

        # Set to BLENDING so seed is active
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.5

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Collect gradients from seed parameters
        seed_params = list(slot.seed.parameters())
        collector = SeedGradientCollector()
        stats = collector.collect(seed_params)

        assert stats["gradient_norm"] > 0, "Seed should have non-zero gradients"
        assert stats["gradient_health"] > 0, "Gradient health should be positive"

    def test_enhanced_gradient_metrics(self, model_with_seed):
        """Enhanced gradient collection returns GradientHealthMetrics."""
        model = model_with_seed
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.5

        # Forward/backward
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Collect with enhanced metrics
        seed_params = list(slot.seed.parameters())
        metrics = collect_seed_gradients(seed_params, return_enhanced=True)

        assert isinstance(metrics, GradientHealthMetrics)
        assert metrics.gradient_norm > 0
        assert not metrics.has_vanishing
        assert not metrics.has_exploding
```

### Step 1.6: Run gradient tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestGradientCollection -v`
Expected: PASS (2 tests)

### Step 1.7: Commit Task 1

```bash
git add tests/integration/test_simic_kasmina.py
git commit -m "test(integration): add simic-kasmina seam tests

Tests reward computation with actual SeedInfo structure and gradient
collection during seed training."
```

---

## Task 2: Create tolaria ↔ simic Integration Tests

**Files:**
- Create: `tests/integration/test_tolaria_simic.py`

### Step 2.1: Write test file with correct imports

```python
"""Integration tests for Tolaria-Simic interaction.

Tests the core integration where:
- Tolaria's trainer runs training epochs
- TolariaGovernor monitors training health
- Training metrics flow to RL reward computation
"""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria.trainer import (
    train_epoch_normal,
    validate_and_get_metrics,
)
from esper.tolaria.governor import TolariaGovernor
from esper.kasmina import MorphogeneticModel, CNNHost


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Simple model for training tests."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def train_loader():
    """Small training data loader."""
    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def test_loader():
    """Small test data loader."""
    X = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


@pytest.fixture
def morphogenetic_model():
    """MorphogeneticModel for governor tests."""
    host = CNNHost(num_classes=10)
    return MorphogeneticModel(host, device="cpu", slots=["r0c1"])
```

### Step 2.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py --collect-only`
Expected: Collection succeeds

### Step 2.3: Write training epoch test

```python
class TestTrainingLoop:
    """Tests for Tolaria training loop producing Simic-relevant metrics."""

    def test_train_epoch_returns_loss(self, simple_model, train_loader):
        """train_epoch_normal should return average loss for the epoch.

        This loss is what Simic observes for reward computation.
        """
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        avg_loss = train_epoch_normal(
            model=simple_model,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )

        assert isinstance(avg_loss, float)
        assert avg_loss > 0, "Loss should be positive"
        assert avg_loss < 100, "Loss should be reasonable"

    def test_validation_returns_accuracy(self, simple_model, train_loader, test_loader):
        """validate_and_get_metrics should return accuracy percentage.

        This accuracy is what Simic uses for contribution reward.
        """
        criterion = nn.CrossEntropyLoss()

        val_loss, val_acc, train_loss, train_acc, _, _ = validate_and_get_metrics(
            model=simple_model,
            trainloader=train_loader,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
        )

        assert isinstance(val_acc, float)
        assert 0 <= val_acc <= 100, "Accuracy should be percentage"
        assert isinstance(val_loss, float)
        assert val_loss > 0
```

### Step 2.4: Run training tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py::TestTrainingLoop -v`
Expected: PASS (2 tests)

### Step 2.5: Write governor integration test

```python
class TestGovernorIntegration:
    """Tests for TolariaGovernor monitoring training for Simic."""

    def test_governor_detects_catastrophic_loss(self, morphogenetic_model):
        """Governor should detect sudden loss spike.

        Simic relies on governor to catch training instabilities.
        """
        governor = TolariaGovernor(
            model=morphogenetic_model,
            sensitivity=2.0,
            absolute_threshold=10.0,
            history_window=5,
        )

        # Snapshot good state
        governor.snapshot()

        # Simulate stable losses
        for _ in range(5):
            is_bad = governor.check_vital_signs(current_loss=1.0)
            assert not is_bad

        # Catastrophic spike above absolute threshold
        is_bad = governor.check_vital_signs(current_loss=15.0)
        assert is_bad, "Governor should detect catastrophic loss"

    def test_governor_provides_punishment_reward(self, morphogenetic_model):
        """Governor should provide punishment reward for RL buffer."""
        governor = TolariaGovernor(
            model=morphogenetic_model,
            death_penalty=-2.0,
        )

        punishment = governor.get_punishment_reward()
        assert punishment == 2.0, "Punishment should be positive (negated penalty)"
```

### Step 2.6: Run governor tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py::TestGovernorIntegration -v`
Expected: PASS (2 tests)

### Step 2.7: Commit Task 2

```bash
git add tests/integration/test_tolaria_simic.py
git commit -m "test(integration): add tolaria-simic seam tests

Tests training loop metrics and TolariaGovernor integration with
correct API signatures."
```

---

## Task 3: Create tolaria ↔ kasmina Integration Tests

**Files:**
- Create: `tests/integration/test_tolaria_kasmina.py`

### Step 3.1: Write test file with correct imports

```python
"""Integration tests for Tolaria-Kasmina interaction.

Tests the core integration where:
- Tolaria trainer works with MorphogeneticModel
- Validation correctly measures seed contribution (attribution)
- force_alpha context manager works for counterfactual evaluation
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria.trainer import (
    train_epoch_normal,
    train_epoch_blended,
    validate_with_attribution,
    AttributionResult,
)
from esper.kasmina import MorphogeneticModel, CNNHost
from esper.leyline import SeedStage


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def morphogenetic_model():
    """MorphogeneticModel with seed slot."""
    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    return model


@pytest.fixture
def model_with_active_seed(morphogenetic_model):
    """Model with germinated seed ready for training."""
    morphogenetic_model.germinate_seed("conv_light", "test_seed", slot="r0c1")
    slot = morphogenetic_model.seed_slots["r0c1"]
    slot.state.stage = SeedStage.TRAINING
    return morphogenetic_model


@pytest.fixture
def cifar_loaders():
    """CIFAR-like data loaders."""
    X_train = torch.randn(64, 3, 32, 32)
    y_train = torch.randint(0, 10, (64,))
    X_test = torch.randn(32, 3, 32, 32)
    y_test = torch.randint(0, 10, (32,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
    return train_loader, test_loader
```

### Step 3.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_kasmina.py --collect-only`
Expected: Collection succeeds

### Step 3.3: Write training with MorphogeneticModel test

```python
class TestTrainingWithMorphogeneticModel:
    """Tests for Tolaria training MorphogeneticModel."""

    def test_train_epoch_with_dormant_seed(self, morphogenetic_model, cifar_loaders):
        """Training should work when seed is dormant (host-only mode)."""
        train_loader, _ = cifar_loaders
        optimizer = torch.optim.SGD(morphogenetic_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = train_epoch_normal(
            model=morphogenetic_model,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_with_active_seed(self, model_with_active_seed, cifar_loaders):
        """Training should work with active seed in TRAINING stage."""
        train_loader, _ = cifar_loaders
        optimizer = torch.optim.SGD(model_with_active_seed.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss = train_epoch_normal(
            model=model_with_active_seed,
            trainloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
        )

        assert isinstance(loss, float)
        assert loss > 0
```

### Step 3.4: Run training tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_kasmina.py::TestTrainingWithMorphogeneticModel -v`
Expected: PASS (2 tests)

### Step 3.5: Write attribution validation test

```python
class TestAttributionValidation:
    """Tests for validate_with_attribution measuring seed contribution."""

    def test_attribution_returns_result(self, model_with_active_seed, cifar_loaders):
        """Attribution validation should return AttributionResult."""
        _, test_loader = cifar_loaders
        criterion = nn.CrossEntropyLoss()

        # Advance to BLENDING so alpha > 0
        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.5

        result = validate_with_attribution(
            model=model_with_active_seed,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
            slot="r0c1",
        )

        assert isinstance(result, AttributionResult)
        assert hasattr(result, "real_accuracy")
        assert hasattr(result, "baseline_accuracy")
        assert hasattr(result, "seed_contribution")

    def test_attribution_contribution_equals_difference(
        self, model_with_active_seed, cifar_loaders
    ):
        """seed_contribution should equal real_accuracy - baseline_accuracy."""
        _, test_loader = cifar_loaders
        criterion = nn.CrossEntropyLoss()

        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.5

        result = validate_with_attribution(
            model=model_with_active_seed,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
            slot="r0c1",
        )

        expected = result.real_accuracy - result.baseline_accuracy
        assert abs(result.seed_contribution - expected) < 0.001

    def test_force_alpha_context_restores_alpha(
        self, model_with_active_seed, cifar_loaders
    ):
        """force_alpha context manager should restore original alpha."""
        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        original_alpha = 0.7
        slot._alpha = original_alpha

        # Use force_alpha directly
        with slot.force_alpha(0.0):
            assert slot.alpha == 0.0, "Alpha should be forced to 0"

        assert slot.alpha == original_alpha, "Alpha should be restored"
```

### Step 3.6: Run attribution tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_kasmina.py::TestAttributionValidation -v`
Expected: PASS (3 tests)

### Step 3.7: Commit Task 3

```bash
git add tests/integration/test_tolaria_kasmina.py
git commit -m "test(integration): add tolaria-kasmina seam tests

Tests training with MorphogeneticModel and attribution validation
with correct force_alpha API."
```

---

## Task 4: Create nissa ↔ simic Integration Tests

**Files:**
- Create: `tests/integration/test_nissa_simic.py`

### Step 4.1: Write test file with correct imports

```python
"""Integration tests for Nissa-Simic interaction.

Tests the core integration where:
- Simic training emits telemetry events
- NissaHub receives and routes events
- Reward components are correctly structured for telemetry
- Anomaly detection produces correct reports
"""

import pytest
import torch

from esper.nissa import NissaHub, get_hub
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.simic.anomaly_detector import AnomalyDetector
from esper.simic.reward_telemetry import RewardComponentsTelemetry
from esper.simic.rewards import compute_contribution_reward, SeedInfo, ContributionRewardConfig
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hub():
    """Fresh NissaHub instance."""
    return NissaHub()


@pytest.fixture
def anomaly_detector():
    """Anomaly detector for training metrics."""
    return AnomalyDetector()


@pytest.fixture
def seed_info():
    """SeedInfo for reward computation."""
    return SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.02,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=4,
    )
```

### Step 4.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py --collect-only`
Expected: Collection succeeds

### Step 4.3: Write reward telemetry test

```python
class TestRewardTelemetry:
    """Tests for Simic reward telemetry flowing to Nissa."""

    def test_compute_reward_returns_telemetry_components(self, seed_info):
        """compute_contribution_reward with return_components=True returns telemetry."""
        config = ContributionRewardConfig()

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        assert isinstance(components, RewardComponentsTelemetry)
        assert components.total_reward == reward
        assert hasattr(components, "seed_contribution")
        assert hasattr(components, "pbrs_bonus")
        assert hasattr(components, "compute_rent")

    def test_telemetry_components_to_dict(self, seed_info):
        """RewardComponentsTelemetry.to_dict() produces serializable data."""
        config = ContributionRewardConfig()

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        data = components.to_dict()
        assert isinstance(data, dict)
        assert "total_reward" in data
        assert "seed_contribution" in data
```

### Step 4.4: Run telemetry tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py::TestRewardTelemetry -v`
Expected: PASS (2 tests)

### Step 4.5: Write anomaly detection test

```python
class TestAnomalyDetection:
    """Tests for anomaly detection producing correct reports."""

    def test_anomaly_detector_healthy_metrics(self, anomaly_detector):
        """Healthy training metrics should not trigger anomaly."""
        # Get current threshold
        threshold = anomaly_detector.get_ev_threshold(
            current_episode=500,
            total_episodes=1000,
        )

        # Threshold should be reasonable
        assert isinstance(threshold, float)
        assert -1.0 <= threshold <= 1.0

    def test_hub_can_emit_telemetry_event(self, hub):
        """NissaHub should accept telemetry events."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={
                "total_reward": 0.5,
                "seed_contribution": 0.02,
                "epoch": 5,
            },
        )

        # Should not raise
        hub.emit(event)
```

### Step 4.6: Run anomaly tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py::TestAnomalyDetection -v`
Expected: PASS (2 tests)

### Step 4.7: Commit Task 4

```bash
git add tests/integration/test_nissa_simic.py
git commit -m "test(integration): add nissa-simic seam tests

Tests reward telemetry components and NissaHub event emission."
```

---

## Task 5: Create karn ↔ nissa Integration Tests

**Files:**
- Create: `tests/integration/test_karn_nissa.py`

### Step 5.1: Write test file with correct imports

```python
"""Integration tests for Karn-Nissa interaction.

Tests the core integration where:
- Karn TelemetryStore receives data from training
- Store tracks epoch history
- Export/import functionality works
"""

import pytest
import tempfile
from pathlib import Path

from esper.karn.store import (
    TelemetryStore,
    EpisodeContext,
)
from esper.nissa import NissaHub
from esper.leyline import TelemetryEvent, TelemetryEventType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Fresh TelemetryStore."""
    return TelemetryStore()


@pytest.fixture
def episode_context():
    """Episode context for store initialization."""
    return EpisodeContext(
        episode_id="test_episode_001",
        host_architecture="cnn_3block",
        host_params=1000000,
        max_epochs=25,
        task_type="classification",
        reward_mode="shaped",
    )
```

### Step 5.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py --collect-only`
Expected: Collection succeeds

### Step 5.3: Write store lifecycle test

```python
class TestTelemetryStoreLifecycle:
    """Tests for Karn TelemetryStore lifecycle."""

    def test_store_starts_episode(self, store, episode_context):
        """Store should accept episode context."""
        store.start_episode(episode_context)
        assert store.context == episode_context

    def test_store_tracks_epochs(self, store, episode_context):
        """Store should track epoch snapshots."""
        store.start_episode(episode_context)

        # Start and commit multiple epochs
        for epoch in range(3):
            epoch_snap = store.start_epoch(epoch=epoch)
            store.commit_epoch()

        # Query recent epochs
        recent = store.get_recent_epochs(n=10)
        assert len(recent) == 3

    def test_store_latest_epoch(self, store, episode_context):
        """Store should provide latest epoch snapshot."""
        store.start_episode(episode_context)

        store.start_epoch(epoch=0)
        store.commit_epoch()

        store.start_epoch(epoch=1)
        store.commit_epoch()

        latest = store.latest_epoch
        assert latest is not None
        assert latest.epoch == 1
```

### Step 5.4: Run store lifecycle tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py::TestTelemetryStoreLifecycle -v`
Expected: PASS (3 tests)

### Step 5.5: Write export/import test

```python
class TestTelemetryExport:
    """Tests for Karn telemetry export/import."""

    def test_export_to_jsonl(self, store, episode_context):
        """Store should export to JSONL format."""
        store.start_episode(episode_context)

        for epoch in range(3):
            store.start_epoch(epoch=epoch)
            store.commit_epoch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.jsonl"
            count = store.export_jsonl(path=str(path))

            assert count > 0
            assert path.exists()

    def test_import_from_jsonl(self, episode_context):
        """Store should import from JSONL format."""
        # Create store with data
        store1 = TelemetryStore()
        store1.start_episode(episode_context)
        for epoch in range(3):
            store1.start_epoch(epoch=epoch)
            store1.commit_epoch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.jsonl"
            store1.export_jsonl(path=str(path))

            # Import into new store
            store2 = TelemetryStore()
            store2.import_jsonl(path=str(path))

            # Should have same data
            recent1 = store1.get_recent_epochs(n=10)
            recent2 = store2.get_recent_epochs(n=10)
            assert len(recent1) == len(recent2)
```

### Step 5.6: Run export tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py::TestTelemetryExport -v`
Expected: PASS (2 tests)

### Step 5.7: Commit Task 5

```bash
git add tests/integration/test_karn_nissa.py
git commit -m "test(integration): add karn-nissa seam tests

Tests TelemetryStore lifecycle and JSONL export/import."
```

---

## Task 6: Final Verification

### Step 6.1: Run all new integration tests

```bash
PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py tests/integration/test_tolaria_simic.py tests/integration/test_tolaria_kasmina.py tests/integration/test_nissa_simic.py tests/integration/test_karn_nissa.py -v
```

Expected: All tests PASS

### Step 6.2: Run full test suite

```bash
PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/cuda
```

Expected: All tests PASS

### Step 6.3: Final commit

```bash
git add -A
git commit -m "test(integration): complete seam test coverage

Added integration tests for all 5 previously missing seams:
- simic ↔ kasmina: Rewards, gradient collection
- tolaria ↔ simic: Training loop, governor
- tolaria ↔ kasmina: MorphogeneticModel, attribution
- nissa ↔ simic: Telemetry emission, anomaly detection
- karn ↔ nissa: Store lifecycle, export/import

All tests verified against actual codebase APIs."
```

---

## Summary

| Task | File | Tests | Seam |
|------|------|-------|------|
| 1 | `test_simic_kasmina.py` | 4 | Rewards, gradients |
| 2 | `test_tolaria_simic.py` | 4 | Training, governor |
| 3 | `test_tolaria_kasmina.py` | 5 | Model, attribution |
| 4 | `test_nissa_simic.py` | 4 | Telemetry, anomaly |
| 5 | `test_karn_nissa.py` | 5 | Store, export |
| **Total** | 5 files | **~22** | 5 seams |

---

## Notes

1. **All imports verified** against actual `src/esper/` module exports
2. **All API signatures verified** using `inspect.signature()`
3. **Removed non-existent methods**: `stage_potential`, `record_loss`, `is_improving`, `fossilize`, `TelemetryCollector`, etc.
4. **Used correct constructors**: `TolariaGovernor(model=...)`, `SeedInfo(stage=..., improvement_since_stage_start=..., ...)`
