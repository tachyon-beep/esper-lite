"""Tests for WandbBackend step accounting and data quality.

Verifies that:
- Batch-level metrics use event.epoch for correct x-axis alignment
- Skipped updates don't collapse multiple batches onto the same step
- None values are filtered before logging (no data quality issues)
- API contract is correct (no dead parameters)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    BatchEpochCompletedPayload,
    EpochCompletedPayload,
    PPOUpdatePayload,
    SeedGerminatedPayload,
    AnomalyDetectedPayload,
)


@pytest.fixture
def mock_wandb():
    """Mock wandb module for testing without actual wandb installation."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}):
        import wandb
        wandb.init = MagicMock(return_value=MagicMock(url="http://test"))
        wandb.log = MagicMock()
        wandb.config = MagicMock()
        wandb.alert = MagicMock()
        wandb.AlertLevel = MagicMock()
        yield wandb


@pytest.fixture
def backend(mock_wandb):
    """Create a WandbBackend with mocked wandb."""
    # Patch WANDB_AVAILABLE before importing
    with patch("esper.nissa.wandb_backend.WANDB_AVAILABLE", True):
        with patch("esper.nissa.wandb_backend.wandb", mock_wandb):
            from esper.nissa.wandb_backend import WandbBackend
            backend = WandbBackend(project="test", mode="disabled")
            backend._run = MagicMock(url="http://test")
            return backend


class TestBatchStepFromEventEpoch:
    """Batch-level events use event.epoch for step, not internal counter."""

    def test_batch_epoch_completed_uses_event_epoch(self, backend, mock_wandb):
        """BATCH_EPOCH_COMPLETED logs at step=event.epoch."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=42,
            data=BatchEpochCompletedPayload(
                batch_idx=5,
                episodes_completed=42,
                avg_accuracy=0.75,
                rolling_accuracy=0.73,
                avg_reward=1.5,
                n_envs=4,
                total_episodes=100,
            ),
        )

        backend.emit(event)

        mock_wandb.log.assert_called_once()
        call_kwargs = mock_wandb.log.call_args[1]
        assert call_kwargs["step"] == 42
        # commit=False was removed - all events now commit independently
        # wandb handles step alignment when same step value is used
        assert "commit" not in call_kwargs or call_kwargs.get("commit") is True

    def test_ppo_update_uses_event_epoch(self, backend, mock_wandb):
        """PPO_UPDATE_COMPLETED logs at step=event.epoch."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=100,
            data=PPOUpdatePayload(
                policy_loss=0.5,
                value_loss=0.3,
                entropy=0.1,
                grad_norm=1.0,
                kl_divergence=0.01,
                clip_fraction=0.1,
                nan_grad_count=0,
            ),
        )

        backend.emit(event)

        mock_wandb.log.assert_called_once()
        call_kwargs = mock_wandb.log.call_args[1]
        assert call_kwargs["step"] == 100
        # PPO is the commit point, so commit should be True (default)
        assert "commit" not in call_kwargs or call_kwargs.get("commit") is True

    def test_seed_germinated_uses_event_epoch(self, backend, mock_wandb):
        """SEED_GERMINATED logs at step=event.epoch."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            epoch=25,
            slot_id="r0c0",
            data=SeedGerminatedPayload(
                slot_id="r0c0",
                env_id=0,
                blueprint_id="conv_heavy",
                params=1000,
                alpha_curve="linear",
            ),
        )

        backend.emit(event)

        mock_wandb.log.assert_called_once()
        call_kwargs = mock_wandb.log.call_args[1]
        assert call_kwargs["step"] == 25

    def test_anomaly_uses_event_epoch(self, backend, mock_wandb):
        """Anomaly events log at step=event.epoch."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            epoch=50,
            data=AnomalyDetectedPayload(
                anomaly_type="ratio_explosion",
                episode=50,
                batch=1,
                inner_epoch=10,
                total_episodes=100,
                detail="Ratio explosion at 15.0",
            ),
        )

        backend.emit(event)

        # log is called once for metrics
        assert mock_wandb.log.call_count == 1
        call_kwargs = mock_wandb.log.call_args[1]
        assert call_kwargs["step"] == 50


class TestMissingEpochHandling:
    """Events without epoch are logged with warning and skipped."""

    def test_batch_epoch_completed_missing_epoch_skipped(self, backend, mock_wandb, caplog):
        """BATCH_EPOCH_COMPLETED without epoch logs warning and skips."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=None,  # Missing epoch!
            data=BatchEpochCompletedPayload(
                batch_idx=5,
                episodes_completed=42,
                avg_accuracy=0.75,
                rolling_accuracy=0.73,
                avg_reward=1.5,
                n_envs=4,
                total_episodes=100,
            ),
        )

        import logging
        with caplog.at_level(logging.WARNING):
            backend.emit(event)

        # Should NOT log to wandb
        mock_wandb.log.assert_not_called()
        # Should log warning
        assert "missing epoch" in caplog.text.lower()


class TestStepMonotonicity:
    """Step values should be monotonically increasing across batches."""

    def test_realistic_event_sequence_monotonic_steps(self, backend, mock_wandb):
        """Realistic batch sequence produces monotonic step values."""
        logged_steps: list[tuple[str, int]] = []

        def capture_log(metrics: dict[str, Any], step: int, commit: bool = True):
            # Identify event type from metrics keys
            if any(k.startswith("ppo/") for k in metrics):
                event_type = "ppo"
            elif any(k.startswith("batch/") for k in metrics):
                event_type = "batch"
            else:
                event_type = "other"
            logged_steps.append((event_type, step))

        mock_wandb.log.side_effect = capture_log

        # Simulate 3 batches: PPO then BATCH for each
        for batch_epoch_id in [10, 11, 12]:
            # PPO update
            ppo_event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                epoch=batch_epoch_id,
                data=PPOUpdatePayload(
                    policy_loss=0.5,
                    value_loss=0.3,
                    entropy=0.1,
                    grad_norm=1.0,
                    kl_divergence=0.01,
                    clip_fraction=0.1,
                    nan_grad_count=0,
                ),
            )
            backend.emit(ppo_event)

            # Batch completed
            batch_event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                epoch=batch_epoch_id,
                data=BatchEpochCompletedPayload(
                    batch_idx=batch_epoch_id,
                    episodes_completed=batch_epoch_id,
                    avg_accuracy=0.75,
                    rolling_accuracy=0.73,
                    avg_reward=1.5,
                    n_envs=4,
                    total_episodes=100,
                ),
            )
            backend.emit(batch_event)

        # Verify monotonicity
        steps_only = [s for _, s in logged_steps]
        assert steps_only == sorted(steps_only), f"Steps not monotonic: {steps_only}"

        # Verify PPO and BATCH at same batch share same step
        for i in range(0, len(logged_steps), 2):
            ppo_type, ppo_step = logged_steps[i]
            batch_type, batch_step = logged_steps[i + 1]
            assert ppo_type == "ppo"
            assert batch_type == "batch"
            assert ppo_step == batch_step, (
                f"PPO and BATCH should share step, got {ppo_step} vs {batch_step}"
            )

    def test_skipped_update_batch_still_has_correct_step(self, backend, mock_wandb):
        """Skipped PPO update still logs batch at correct step (no step collapse)."""
        logged_steps: list[tuple[str, int]] = []

        def capture_log(metrics: dict[str, Any], step: int, commit: bool = True):
            if any(k.startswith("ppo/") for k in metrics):
                event_type = "ppo"
            elif any(k.startswith("batch/") for k in metrics):
                event_type = "batch"
            else:
                event_type = "other"
            logged_steps.append((event_type, step))

        mock_wandb.log.side_effect = capture_log

        # Batch 10: Normal PPO update
        ppo1 = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=10,
            data=PPOUpdatePayload(
                policy_loss=0.5,
                value_loss=0.3,
                entropy=0.1,
                grad_norm=1.0,
                kl_divergence=0.01,
                clip_fraction=0.1,
                nan_grad_count=0,
                update_skipped=False,
            ),
        )
        backend.emit(ppo1)

        batch1 = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=10,
            data=BatchEpochCompletedPayload(
                batch_idx=10,
                episodes_completed=10,
                avg_accuracy=0.75,
                rolling_accuracy=0.73,
                avg_reward=1.5,
                n_envs=4,
                total_episodes=100,
            ),
        )
        backend.emit(batch1)

        # Batch 11: Skipped PPO update (KL too high)
        ppo2 = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=11,
            data=PPOUpdatePayload(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                grad_norm=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                nan_grad_count=0,
                update_skipped=True,
            ),
        )
        backend.emit(ppo2)

        batch2 = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=11,
            data=BatchEpochCompletedPayload(
                batch_idx=11,
                episodes_completed=11,
                avg_accuracy=0.76,
                rolling_accuracy=0.74,
                avg_reward=1.6,
                n_envs=4,
                total_episodes=100,
            ),
        )
        backend.emit(batch2)

        # Batch 12: Normal again
        ppo3 = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            epoch=12,
            data=PPOUpdatePayload(
                policy_loss=0.4,
                value_loss=0.2,
                entropy=0.15,
                grad_norm=0.8,
                kl_divergence=0.005,
                clip_fraction=0.08,
                nan_grad_count=0,
                update_skipped=False,
            ),
        )
        backend.emit(ppo3)

        batch3 = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=12,
            data=BatchEpochCompletedPayload(
                batch_idx=12,
                episodes_completed=12,
                avg_accuracy=0.77,
                rolling_accuracy=0.75,
                avg_reward=1.7,
                n_envs=4,
                total_episodes=100,
            ),
        )
        backend.emit(batch3)

        # Extract step values for each batch
        ppo_steps = [s for t, s in logged_steps if t == "ppo"]
        batch_steps = [s for t, s in logged_steps if t == "batch"]

        # All 3 batches should have distinct, monotonic steps
        assert ppo_steps == [10, 11, 12], f"PPO steps not monotonic: {ppo_steps}"
        assert batch_steps == [10, 11, 12], f"Batch steps not monotonic: {batch_steps}"

        # Skipped update batch (11) should NOT collapse onto previous step (10)
        assert 11 in batch_steps, "Skipped update batch should have its own step"


class TestNoneValueFiltering:
    """Verify that None values are not logged to wandb (data quality fix)."""

    def test_epoch_completed_filters_none_train_loss(self, backend, mock_wandb):
        """EPOCH_COMPLETED should NOT log train_loss when it's None."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=5,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=0.85,
                val_loss=0.15,
                inner_epoch=5,
                train_loss=None,  # Not computed
                train_accuracy=None,  # Not computed
            ),
        )

        backend.emit(event)

        mock_wandb.log.assert_called_once()
        logged_metrics = mock_wandb.log.call_args[0][0]

        # Required metrics should be present
        assert "env_0/val_loss" in logged_metrics
        assert "env_0/val_accuracy" in logged_metrics
        assert "env_0/epoch" in logged_metrics

        # Optional None metrics should NOT be present
        assert "env_0/train_loss" not in logged_metrics
        assert "env_0/train_accuracy" not in logged_metrics

    def test_epoch_completed_includes_train_loss_when_present(self, backend, mock_wandb):
        """EPOCH_COMPLETED should include train_loss when it's not None."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=5,
            data=EpochCompletedPayload(
                env_id=1,
                val_accuracy=0.85,
                val_loss=0.15,
                inner_epoch=5,
                train_loss=0.25,  # Present
                train_accuracy=0.80,  # Present
            ),
        )

        backend.emit(event)

        mock_wandb.log.assert_called_once()
        logged_metrics = mock_wandb.log.call_args[0][0]

        # All metrics should be present
        assert logged_metrics["env_1/train_loss"] == 0.25
        assert logged_metrics["env_1/train_accuracy"] == 0.80
