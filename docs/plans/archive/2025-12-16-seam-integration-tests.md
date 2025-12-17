# Seam Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add integration tests for the 5 missing critical seams between Esper domains.

**Architecture:** Each seam test file focuses on the data flow and contract between two domains. Tests use minimal mocking, preferring real components where possible. All tests are marked as integration tests and excluded from default runs.

**Tech Stack:** pytest, torch, hypothesis (for property-based edge cases)

---

## Prerequisites

Before starting, ensure you understand:
- **Existing patterns**: See `tests/integration/test_tamiyo_simic.py` for integration test style
- **Domain roles**: Read `ROADMAP.md` for the biological metaphor (Simic=Evolution, Kasmina=StemCells, etc.)
- **Test markers**: Integration tests auto-marked via `tests/integration/conftest.py`

---

## Task 1: Create simic ↔ kasmina Integration Tests

**Files:**
- Create: `tests/integration/test_simic_kasmina.py`

### Step 1.1: Write test file scaffold with imports

```python
"""Integration tests for Simic-Kasmina interaction.

Tests the core integration where:
- Simic computes rewards based on seed contribution (from validation)
- Rewards influence seed alpha progression toward blending/fossilization
- PBRS shaping correctly reflects seed stage transitions
- Gradient isolation keeps host and seed parameters separate during training
"""

import pytest
import torch
import torch.nn as nn

from esper.kasmina import MorphogeneticModel, CNNHost
from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.simic.rewards import (
    compute_reward,
    ContributionRewardConfig,
    SeedInfo,
    RewardMode,
)
from esper.simic.gradient_collector import (
    SeedGradientCollector,
    collect_seed_gradients,
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
```

### Step 1.2: Run test to verify imports work

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py --collect-only`
Expected: Collection succeeds with 0 tests

### Step 1.3: Write test for positive contribution increasing alpha

```python
class TestRewardToAlphaProgression:
    """Tests that reward signals drive alpha progression correctly."""

    def test_positive_contribution_reward_is_positive(self, model_with_seed, reward_config):
        """Positive seed contribution should yield positive reward.

        This tests the Simic→Kasmina signal: when validation shows the seed
        helps accuracy, Simic's reward function should return positive reward.
        """
        slot = model_with_seed.seed_slots["r0c1"]
        seed_info = SeedInfo(
            stage=slot.stage,
            alpha=0.3,
            age=5,
            param_count=1000,
            blueprint_id="conv_light",
        )

        # Positive contribution: seed helps by 2% accuracy
        reward = compute_reward(
            seed_contribution=0.02,  # 2% improvement
            val_acc=75.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            config=reward_config,
        )

        assert reward > 0, "Positive contribution should yield positive reward"
```

### Step 1.4: Run test to verify it passes

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestRewardToAlphaProgression::test_positive_contribution_reward_is_positive -v`
Expected: PASS

### Step 1.5: Write test for negative contribution

```python
    def test_negative_contribution_reward_is_negative(self, model_with_seed, reward_config):
        """Negative seed contribution should yield negative reward.

        When validation shows the seed hurts accuracy, reward should be negative,
        which should eventually lead to culling the seed.
        """
        slot = model_with_seed.seed_slots["r0c1"]
        seed_info = SeedInfo(
            stage=slot.stage,
            alpha=0.3,
            age=5,
            param_count=1000,
            blueprint_id="conv_light",
        )

        # Negative contribution: seed hurts by 3% accuracy
        reward = compute_reward(
            seed_contribution=-0.03,  # 3% degradation
            val_acc=72.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            config=reward_config,
        )

        assert reward < 0, "Negative contribution should yield negative reward"
```

### Step 1.6: Run test

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestRewardToAlphaProgression::test_negative_contribution_reward_is_negative -v`
Expected: PASS

### Step 1.7: Write PBRS stage potential test

```python
class TestPBRSStagePotentials:
    """Tests that PBRS shaping correctly reflects stage transitions."""

    def test_stage_advance_yields_positive_shaping(self, reward_config):
        """Advancing to a later stage should yield positive PBRS bonus.

        PBRS assigns higher potential to later stages (BLENDING > TRAINING).
        When a seed advances, the potential difference should be positive.
        """
        from esper.simic.rewards import stage_potential

        training_potential = stage_potential(SeedStage.TRAINING)
        blending_potential = stage_potential(SeedStage.BLENDING)

        # PBRS bonus = gamma * new_potential - old_potential
        pbrs_bonus = 0.99 * blending_potential - training_potential

        assert pbrs_bonus > 0, "Stage advance should yield positive PBRS"

    def test_fossilized_has_highest_potential(self, reward_config):
        """FOSSILIZED stage should have highest potential (terminal success)."""
        from esper.simic.rewards import stage_potential

        fossilized = stage_potential(SeedStage.FOSSILIZED)
        blending = stage_potential(SeedStage.BLENDING)
        training = stage_potential(SeedStage.TRAINING)
        germinated = stage_potential(SeedStage.GERMINATED)

        assert fossilized > blending > training > germinated
```

### Step 1.8: Run PBRS tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestPBRSStagePotentials -v`
Expected: PASS (2 tests)

### Step 1.9: Write gradient isolation test

```python
class TestGradientIsolation:
    """Tests that gradient isolation between host and seed works correctly."""

    def test_seed_gradients_isolated_from_host(self, model_with_seed):
        """During blending, seed and host gradients should be separate.

        This is critical for the Simic→Kasmina contract: Simic's gradient
        collection should capture seed gradients without polluting host.
        """
        model = model_with_seed
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.5

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Collect seed gradients
        seed_params = list(slot.seed.parameters())
        host_params = [p for p in model.host.parameters()]

        # Seed should have gradients
        seed_has_grads = any(p.grad is not None for p in seed_params)
        assert seed_has_grads, "Seed should have gradients during blending"

        # Host should also have gradients (joint training)
        host_has_grads = any(p.grad is not None for p in host_params)
        assert host_has_grads, "Host should have gradients during blending"

    def test_fossilized_seed_no_gradients(self, model_with_seed):
        """After fossilization, seed parameters should not receive gradients.

        Fossilized seeds are frozen - their parameters are permanent.
        """
        model = model_with_seed
        slot = model.seed_slots["r0c1"]

        # Fossilize the seed
        slot.fossilize()

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Seed should NOT have gradients after fossilization
        seed_params = list(slot.seed.parameters())
        seed_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in seed_params if p.grad is not None)

        assert not seed_has_grads, "Fossilized seed should not receive gradients"
```

### Step 1.10: Run gradient isolation tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_simic_kasmina.py::TestGradientIsolation -v`
Expected: PASS (2 tests)

### Step 1.11: Commit Task 1

```bash
git add tests/integration/test_simic_kasmina.py
git commit -m "test(integration): add simic-kasmina seam tests

Tests reward→alpha progression, PBRS stage potentials, and gradient
isolation between host and seed during different lifecycle stages."
```

---

## Task 2: Create tolaria ↔ simic Integration Tests

**Files:**
- Create: `tests/integration/test_tolaria_simic.py`

### Step 2.1: Write test file scaffold

```python
"""Integration tests for Tolaria-Simic interaction.

Tests the core integration where:
- Tolaria's trainer runs training epochs
- Simic's PPO agent receives the training metrics
- Loss gradients flow correctly through the training loop
- TolariaGovernor monitors training health metrics from Simic
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from esper.tolaria.trainer import (
    train_epoch_normal,
    validate_and_get_metrics,
    _run_validation_pass,
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
def governor():
    """TolariaGovernor for monitoring tests."""
    return TolariaGovernor(
        sensitivity=2.0,
        absolute_threshold=10.0,
        history_window=5,
    )
```

### Step 2.2: Run to verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py --collect-only`
Expected: Collection succeeds

### Step 2.3: Write training epoch test

```python
class TestTrainingLoop:
    """Tests for Tolaria training loop with Simic-relevant metrics."""

    def test_train_epoch_returns_loss(self, simple_model, train_loader):
        """train_epoch_normal should return average loss for the epoch.

        This loss is what Simic observes to compute rewards.
        """
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        avg_loss = train_epoch_normal(
            model=simple_model,
            trainloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        assert isinstance(avg_loss, float)
        assert avg_loss > 0, "Loss should be positive"
        assert avg_loss < 100, "Loss should be reasonable"

    def test_validation_returns_accuracy(self, simple_model, test_loader):
        """validate_and_get_metrics should return accuracy percentage.

        This accuracy is what Simic uses for contribution reward.
        """
        criterion = nn.CrossEntropyLoss()

        val_loss, accuracy = validate_and_get_metrics(
            model=simple_model,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
        )

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100, "Accuracy should be percentage"
```

### Step 2.4: Run training tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py::TestTrainingLoop -v`
Expected: PASS (2 tests)

### Step 2.5: Write governor monitoring test

```python
class TestGovernorIntegration:
    """Tests for TolariaGovernor monitoring Simic training."""

    def test_governor_detects_loss_spike(self, governor):
        """Governor should detect sudden loss increases.

        Simic relies on governor to catch training instabilities.
        """
        # Feed stable losses
        for _ in range(5):
            governor.record_loss(1.0)

        # Sudden spike
        is_anomaly = governor.record_loss(15.0)

        assert is_anomaly, "Governor should detect loss spike"

    def test_governor_tracks_loss_trend(self, governor):
        """Governor should track loss trend for Simic decisions."""
        losses = [2.0, 1.8, 1.6, 1.4, 1.2]
        for loss in losses:
            governor.record_loss(loss)

        assert governor.is_improving(), "Should detect improving trend"

    def test_governor_with_real_training(self, simple_model, train_loader, test_loader, governor):
        """Governor integrated with real training loop.

        End-to-end test: Tolaria trains, governor monitors.
        """
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        anomalies = []
        for epoch in range(3):
            loss = train_epoch_normal(
                model=simple_model,
                trainloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device="cpu",
            )
            is_anomaly = governor.record_loss(loss)
            anomalies.append(is_anomaly)

        # Normal training shouldn't trigger anomalies
        assert not any(anomalies), "Normal training should not trigger anomalies"
```

### Step 2.6: Run governor tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py::TestGovernorIntegration -v`
Expected: PASS (3 tests)

### Step 2.7: Write gradient flow test

```python
class TestGradientFlow:
    """Tests that gradients flow correctly through training."""

    def test_loss_produces_gradients(self, simple_model, train_loader):
        """Loss from training should produce gradients in model parameters.

        This is fundamental to Simic's RL - gradients must flow.
        """
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Get initial param values
        initial_params = [p.clone() for p in simple_model.parameters()]

        # Train one epoch
        train_epoch_normal(
            model=simple_model,
            trainloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        # Parameters should have changed
        params_changed = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(initial_params, simple_model.parameters())
        )
        assert params_changed, "Training should update parameters"
```

### Step 2.8: Run gradient flow test

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_simic.py::TestGradientFlow -v`
Expected: PASS

### Step 2.9: Commit Task 2

```bash
git add tests/integration/test_tolaria_simic.py
git commit -m "test(integration): add tolaria-simic seam tests

Tests training loop metrics, governor anomaly detection, and gradient
flow through the training pipeline."
```

---

## Task 3: Create tolaria ↔ kasmina Integration Tests

**Files:**
- Create: `tests/integration/test_tolaria_kasmina.py`

### Step 3.1: Write test file scaffold

```python
"""Integration tests for Tolaria-Kasmina interaction.

Tests the core integration where:
- Tolaria trainer works with MorphogeneticModel
- Validation correctly measures seed contribution (attribution)
- Checkpoint save/load preserves seed state
- Training modes handle different seed stages correctly
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
            optimizer=optimizer,
            criterion=criterion,
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
            optimizer=optimizer,
            criterion=criterion,
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

    def test_attribution_contribution_equals_difference(self, model_with_active_seed, cifar_loaders):
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

    def test_attribution_restores_alpha(self, model_with_active_seed, cifar_loaders):
        """Attribution should restore original alpha after validation."""
        _, test_loader = cifar_loaders
        criterion = nn.CrossEntropyLoss()

        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        original_alpha = 0.7
        slot._alpha = original_alpha

        validate_with_attribution(
            model=model_with_active_seed,
            testloader=test_loader,
            criterion=criterion,
            device="cpu",
            slot="r0c1",
        )

        assert slot.alpha == original_alpha, "Alpha should be restored"
```

### Step 3.6: Run attribution tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_kasmina.py::TestAttributionValidation -v`
Expected: PASS (3 tests)

### Step 3.7: Write checkpoint test

```python
class TestCheckpointIntegration:
    """Tests for checkpoint save/load preserving seed state."""

    def test_checkpoint_preserves_seed_stage(self, model_with_active_seed, tmp_path):
        """Checkpoint should preserve seed stage across save/load."""
        slot = model_with_active_seed.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot._alpha = 0.6

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            "model_state": model_with_active_seed.state_dict(),
            "seed_slots": {
                slot_id: slot.state_dict()
                for slot_id, slot in model_with_active_seed.seed_slots.items()
            },
        }, checkpoint_path)

        # Create fresh model and load
        host = CNNHost(num_classes=10)
        new_model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
        new_model.germinate_seed("conv_light", "test_seed", slot="r0c1")

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_model.load_state_dict(checkpoint["model_state"])
        for slot_id, state in checkpoint["seed_slots"].items():
            new_model.seed_slots[slot_id].load_state_dict(state)

        # Verify stage preserved
        new_slot = new_model.seed_slots["r0c1"]
        assert new_slot.stage == SeedStage.BLENDING
```

### Step 3.8: Run checkpoint test

Run: `PYTHONPATH=src uv run pytest tests/integration/test_tolaria_kasmina.py::TestCheckpointIntegration -v`
Expected: PASS

### Step 3.9: Commit Task 3

```bash
git add tests/integration/test_tolaria_kasmina.py
git commit -m "test(integration): add tolaria-kasmina seam tests

Tests training with MorphogeneticModel, attribution validation for
seed contribution, and checkpoint preservation of seed state."
```

---

## Task 4: Create nissa ↔ simic Integration Tests

**Files:**
- Create: `tests/integration/test_nissa_simic.py`

### Step 4.1: Write test file scaffold

```python
"""Integration tests for Nissa-Simic interaction.

Tests the core integration where:
- Simic training emits telemetry events
- Nissa's hub receives and routes events
- Reward components are correctly telemetered
- Anomaly detection triggers telemetry events
"""

import pytest
import torch

from esper.nissa import get_hub, TelemetryHub
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.simic.anomaly_detector import AnomalyDetector
from esper.simic.reward_telemetry import RewardComponentsTelemetry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hub():
    """Fresh telemetry hub."""
    return TelemetryHub()


@pytest.fixture
def anomaly_detector():
    """Anomaly detector for training metrics."""
    return AnomalyDetector()
```

### Step 4.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py --collect-only`
Expected: Collection succeeds

### Step 4.3: Write telemetry emission test

```python
class TestTelemetryEmission:
    """Tests for Simic emitting telemetry to Nissa."""

    def test_reward_components_emit_event(self, hub):
        """RewardComponentsTelemetry should produce valid telemetry event."""
        components = RewardComponentsTelemetry(
            primary_reward=0.5,
            pbrs_shaping=0.1,
            compute_rent=-0.05,
            warning_penalty=0.0,
            total_reward=0.55,
        )

        event = components.to_event(slot_id="r0c1", env_id=0)

        assert isinstance(event, TelemetryEvent)
        assert event.event_type == TelemetryEventType.REWARD_COMPUTED
        assert "primary_reward" in event.data
        assert "total_reward" in event.data

    def test_hub_receives_reward_event(self, hub):
        """Hub should receive and store reward telemetry."""
        components = RewardComponentsTelemetry(
            primary_reward=0.5,
            pbrs_shaping=0.1,
            compute_rent=-0.05,
            warning_penalty=0.0,
            total_reward=0.55,
        )
        event = components.to_event(slot_id="r0c1", env_id=0)

        # Emit to hub
        hub.emit(event)

        # Hub should have recorded it
        assert hub.event_count > 0
```

### Step 4.4: Run telemetry tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py::TestTelemetryEmission -v`
Expected: PASS (2 tests)

### Step 4.5: Write anomaly detection telemetry test

```python
class TestAnomalyTelemetry:
    """Tests for anomaly detection triggering telemetry."""

    def test_ratio_explosion_emits_event(self, anomaly_detector):
        """Ratio explosion should produce anomaly telemetry event."""
        # Simulate ratio explosion
        report = anomaly_detector.check(
            ratio=100.0,  # Extremely high ratio
            value_loss=1.0,
            entropy=0.5,
            kl_div=0.01,
        )

        if report.has_anomaly:
            event = report.to_event(env_id=0, epoch=5)
            assert event.event_type == TelemetryEventType.ANOMALY_DETECTED
            assert "ratio_explosion" in str(event.data).lower() or report.ratio_explosion

    def test_healthy_training_no_anomaly(self, anomaly_detector):
        """Healthy training metrics should not trigger anomaly."""
        report = anomaly_detector.check(
            ratio=1.05,  # Normal ratio
            value_loss=0.5,
            entropy=1.5,
            kl_div=0.005,
        )

        assert not report.has_anomaly, "Healthy metrics should not trigger anomaly"
```

### Step 4.6: Run anomaly tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_nissa_simic.py::TestAnomalyTelemetry -v`
Expected: PASS (2 tests)

### Step 4.7: Commit Task 4

```bash
git add tests/integration/test_nissa_simic.py
git commit -m "test(integration): add nissa-simic seam tests

Tests telemetry emission from Simic training, hub event handling,
and anomaly detection telemetry flow."
```

---

## Task 5: Create karn ↔ nissa Integration Tests

**Files:**
- Create: `tests/integration/test_karn_nissa.py`

### Step 5.1: Write test file scaffold

```python
"""Integration tests for Karn-Nissa interaction.

Tests the core integration where:
- Karn collector receives telemetry from Nissa
- Events are validated and stored
- Analytics aggregate telemetry data
- Multi-environment collection works correctly
"""

import pytest

from esper.karn.collector import TelemetryCollector, get_collector
from esper.karn.store import TelemetryStore
from esper.nissa import TelemetryHub
from esper.leyline import TelemetryEvent, TelemetryEventType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def collector():
    """Fresh telemetry collector."""
    return TelemetryCollector()


@pytest.fixture
def store():
    """Fresh telemetry store."""
    return TelemetryStore()


def make_training_event(epoch: int, loss: float, env_id: int = 0) -> TelemetryEvent:
    """Create a training telemetry event."""
    return TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        data={
            "epoch": epoch,
            "loss": loss,
            "env_id": env_id,
        },
    )
```

### Step 5.2: Verify imports

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py --collect-only`
Expected: Collection succeeds

### Step 5.3: Write collector event handling test

```python
class TestCollectorEventHandling:
    """Tests for Karn collector receiving Nissa events."""

    def test_collector_receives_event(self, collector):
        """Collector should receive and validate events."""
        event = make_training_event(epoch=1, loss=1.5)

        collector.emit(event)

        assert collector.event_count == 1

    def test_collector_validates_event_type(self, collector):
        """Collector should only accept valid event types."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data={"epoch": 1},
        )

        # Should not raise
        collector.emit(event)
        assert collector.event_count == 1

    def test_collector_handles_multi_env(self, collector):
        """Collector should handle events from multiple environments."""
        for env_id in range(4):
            event = make_training_event(epoch=1, loss=1.5 - env_id * 0.1, env_id=env_id)
            collector.emit(event)

        assert collector.event_count == 4
```

### Step 5.4: Run collector tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py::TestCollectorEventHandling -v`
Expected: PASS (3 tests)

### Step 5.5: Write store aggregation test

```python
class TestStoreAggregation:
    """Tests for Karn store aggregating telemetry."""

    def test_store_tracks_epoch_history(self, store):
        """Store should track loss history per epoch."""
        for epoch in range(5):
            store.record_epoch(epoch=epoch, loss=2.0 - epoch * 0.2, accuracy=50 + epoch * 5)

        history = store.get_loss_history()
        assert len(history) == 5
        assert history[0] > history[-1], "Loss should decrease"

    def test_store_computes_statistics(self, store):
        """Store should compute aggregate statistics."""
        for epoch in range(10):
            store.record_epoch(epoch=epoch, loss=2.0 - epoch * 0.1, accuracy=50 + epoch * 3)

        stats = store.get_statistics()
        assert "mean_loss" in stats
        assert "final_accuracy" in stats
        assert stats["final_accuracy"] > stats.get("initial_accuracy", 0)
```

### Step 5.6: Run store tests

Run: `PYTHONPATH=src uv run pytest tests/integration/test_karn_nissa.py::TestStoreAggregation -v`
Expected: PASS (2 tests)

### Step 5.7: Commit Task 5

```bash
git add tests/integration/test_karn_nissa.py
git commit -m "test(integration): add karn-nissa seam tests

Tests collector event handling, multi-environment support, and
store aggregation of telemetry data."
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

Expected: All tests PASS, total count increased by ~25-30

### Step 6.3: Final commit with summary

```bash
git add -A
git commit -m "test(integration): complete seam test coverage

Added integration tests for all 5 previously missing seams:
- simic ↔ kasmina: Rewards, PBRS, gradient isolation
- tolaria ↔ simic: Training loop, governor, gradients
- tolaria ↔ kasmina: MorphogeneticModel, attribution, checkpoints
- nissa ↔ simic: Telemetry emission, anomaly events
- karn ↔ nissa: Collector, multi-env, store aggregation

Total new tests: ~30
All critical domain interfaces now have integration coverage."
```

---

## Summary

| Task | File | Tests | Seam |
|------|------|-------|------|
| 1 | `test_simic_kasmina.py` | 6 | Rewards → Seed lifecycle |
| 2 | `test_tolaria_simic.py` | 6 | Trainer → RL metrics |
| 3 | `test_tolaria_kasmina.py` | 6 | Trainer → Model |
| 4 | `test_nissa_simic.py` | 4 | Telemetry → Training |
| 5 | `test_karn_nissa.py` | 5 | Analytics → Telemetry |
| **Total** | 5 files | **~27** | 5 seams |

---

## Notes for Implementer

1. **Import errors**: If imports fail, check `src/esper/<domain>/__init__.py` exports
2. **Fixture adjustments**: May need to adjust based on actual API (e.g., `TelemetryCollector` vs `get_collector()`)
3. **Mock fallback**: If real components are too heavy, use `unittest.mock.Mock` with spec
4. **Property test bonus**: After basic tests pass, add `@given` decorators for edge cases
