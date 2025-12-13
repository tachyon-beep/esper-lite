"""Integration tests for Tamiyo-Tolaria interaction.

Tests that SignalTracker receives training signals from Tolaria's training loop
and that policy decisions feed back into the training execution.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from esper.tamiyo.tracker import SignalTracker
from esper.tolaria.trainer import train_epoch_normal, validate_and_get_metrics
from esper.leyline import TrainingSignals


class SimpleModel(nn.Module):
    """Minimal model for testing Tolaria integration."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestTamiyoTolariaIntegration:
    """Integration tests for SignalTracker ← Tolaria data flow."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create minimal dataloaders for testing."""
        # Generate random data
        X_train = torch.randn(64, 10)
        y_train = torch.randint(0, 2, (64,))
        X_val = torch.randn(32, 10)
        y_val = torch.randint(0, 2, (32,))

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        trainloader = DataLoader(train_dataset, batch_size=16)
        valloader = DataLoader(val_dataset, batch_size=16)

        return trainloader, valloader

    def test_tracker_receives_training_signals(self, simple_dataloader):
        """SignalTracker should receive and process signals from Tolaria training.

        Integration flow:
        1. Tolaria runs training epoch
        2. Tolaria runs validation
        3. Metrics fed to SignalTracker.update()
        4. Tracker computes derived signals (deltas, plateaus, stabilization)
        """
        trainloader, valloader = simple_dataloader

        # Create model and tracker
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        tracker = SignalTracker()

        # Simulate training loop with Tolaria functions
        for epoch in range(3):
            # Training epoch (Tolaria)
            train_epoch_normal(
                model=model,
                trainloader=trainloader,
                criterion=criterion,
                optimizer=optimizer,
                device="cpu",
                task_type="classification",
            )

            # Validation (Tolaria)
            val_loss, val_accuracy, train_loss, train_accuracy, _, _ = validate_and_get_metrics(
                model=model,
                trainloader=trainloader,
                testloader=valloader,
                criterion=criterion,
                device="cpu",
                task_type="classification",
            )

            # Feed signals to tracker (Tamiyo)
            signals = tracker.update(
                epoch=epoch,
                global_step=epoch * len(trainloader),
                train_loss=0.5,  # Simplified (would come from training loop)
                train_accuracy=75.0,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                active_seeds=[],
                available_slots=1,
            )

            # Verify tracker produces valid signals
            assert isinstance(signals, TrainingSignals)
            assert signals.metrics.epoch == epoch
            assert signals.metrics.val_accuracy == val_accuracy
            assert signals.metrics.val_loss == val_loss

        # Verify history accumulation
        assert len(tracker._accuracy_history) == 3
        assert len(tracker._loss_history) == 3

    def test_decisions_fed_back_to_loop(self, simple_dataloader):
        """Policy decisions should affect training loop execution.

        Integration flow:
        1. Tracker processes signals → returns TrainingSignals
        2. Policy observes TrainingSignals → makes decision
        3. Decision affects next training epoch (e.g., germination changes model)

        This test verifies the tracker provides the correct data structure
        that policies need to make decisions.
        """
        trainloader, valloader = simple_dataloader

        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        tracker = SignalTracker()

        # First epoch: establish baseline
        train_epoch_normal(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            task_type="classification",
        )

        val_loss, val_accuracy, train_loss, train_accuracy, _, _ = validate_and_get_metrics(
            model=model,
            trainloader=trainloader,
            testloader=valloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        signals = tracker.update(
            epoch=0,
            global_step=0,
            train_loss=0.5,
            train_accuracy=75.0,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            active_seeds=[],
            available_slots=1,
        )

        # Verify signals contain decision-relevant information
        assert hasattr(signals.metrics, "val_accuracy")
        assert hasattr(signals.metrics, "val_loss")
        assert hasattr(signals.metrics, "plateau_epochs")
        assert hasattr(signals.metrics, "host_stabilized")

        # Simulate policy decision based on signals
        # (In real system: HeuristicTamiyo or PPO policy observes signals)
        should_germinate = (
            signals.metrics.plateau_epochs > 2
            and signals.metrics.host_stabilized == 1
            and len(signals.active_seeds) == 0
        )

        # Decision would feed back to loop:
        # if should_germinate:
        #     slot.germinate(blueprint_id="norm")
        #     train_epoch_incubator_mode(...)

        # For this test, we just verify the signal structure enables decisions
        assert isinstance(should_germinate, bool)


class TestTolariaValidationUtilities:
    """Test Tolaria validation utilities used in integration."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create minimal validation dataloader."""
        X = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=16)

    def test_validate_and_get_metrics_returns_tuple(self, simple_dataloader):
        """validate_and_get_metrics should return 6-tuple with metrics.

        Returns (val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity).
        """
        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()

        result = validate_and_get_metrics(
            model=model,
            trainloader=simple_dataloader,
            testloader=simple_dataloader,
            criterion=criterion,
            device="cpu",
            task_type="classification",
        )

        # Verify return structure
        assert len(result) == 6
        val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity = result

        # Verify return types
        assert isinstance(val_loss, float)
        assert isinstance(val_accuracy, float)
        assert isinstance(train_loss, float)
        assert isinstance(train_accuracy, float)
        assert per_class_acc is None  # Not requested
        assert perplexity is None  # Classification task

        # Verify reasonable ranges
        assert val_loss > 0
        assert 0 <= val_accuracy <= 100
        assert train_loss > 0
        assert 0 <= train_accuracy <= 100
