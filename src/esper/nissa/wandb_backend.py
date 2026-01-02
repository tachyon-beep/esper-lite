"""Weights & Biases integration for Esper telemetry.

This backend logs training metrics, seed lifecycle events, and PPO training
progress to wandb for experiment tracking and visualization.

Usage:
    from esper.nissa.wandb_backend import WandbBackend
    from esper.karn import get_collector

    collector = get_collector()

    wandb_backend = WandbBackend(
        project="esper-morphogenesis",
        config={"lr": 3e-4, "gamma": 0.99},
        tags=["ppo", "cifar_baseline"],
    )

    collector.add_backend(wandb_backend)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from esper.nissa.output import OutputBackend
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    TrainingStartedPayload,
    AnomalyDetectedPayload,
)

_logger = logging.getLogger(__name__)


class WandbBackend(OutputBackend):
    """Weights & Biases output backend for Esper telemetry.

    Logs training metrics, seed lifecycle events, and PPO training progress
    to wandb for experiment tracking and visualization.

    Step Model:
        Two independent step axes are used for different metric granularities:

        1. Batch-level step (from event.epoch): The monotonic batch_epoch_id
           carried on all batch-level events (PPO_UPDATE_COMPLETED,
           BATCH_EPOCH_COMPLETED, seed lifecycle, anomalies).
           All batch-level metrics share this x-axis for correct alignment.
           Events without epoch are logged with a warning.

        2. _env_epoch_step: Incremented once per EPOCH_COMPLETED event.
           Used by: env_<id>/ metrics (per-environment training curves).
           These are high-frequency events (n_envs × epochs_per_episode) and
           need their own x-axis to avoid crowding the batch-level charts.

    Thread Safety:
        This backend is designed to be called from BackendWorker threads in
        NissaHub/KarnCollector. All wandb.log() calls are thread-safe per
        the wandb SDK documentation.

    Args:
        project: wandb project name (e.g., "esper-morphogenesis")
        entity: wandb team/user name (optional)
        config: Training configuration dict to log
        tags: List of tags for this run
        group: Group name for related runs (e.g., "ablation-study-1")
        name: Custom run name (auto-generated if None)
        mode: "online", "offline", or "disabled"
        log_code: Whether to log git commit and code diff
        log_system: Whether to log system metrics (GPU, CPU, memory)

    Raises:
        ImportError: If wandb is not installed
    """

    def __init__(
        self,
        project: str = "esper",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        name: str | None = None,
        mode: str = "online",
        log_code: bool = True,
        log_system: bool = True,
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.name = name
        self.mode = mode
        self.log_code = log_code
        self.log_system = log_system

        self._run: Any | None = None  # wandb.sdk.wandb_run.Run
        # Step counter for per-env metrics - see class docstring "Step Model"
        self._env_epoch_step = 0  # Incremented per env epoch (per-env x-axis)

    def start(self) -> None:
        """Initialize wandb run.

        Called by KarnCollector.add_backend() before the backend is used.
        Idempotent - if run already exists, logs warning and returns.
        """
        if self._run is not None:
            _logger.warning("WandbBackend.start() called but run already exists")
            return

        # wandb.init is type-hinted to return Run | RunDisabled | None
        # We check for None and log appropriately
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            group=self.group,
            name=self.name,
            mode=self.mode,
            save_code=self.log_code,
        )

        if self._run is None:
            _logger.error("wandb.init() returned None - run not initialized")
            return

        # Log system metrics if requested
        if self.log_system and self.mode != "disabled":
            # wandb automatically tracks system metrics when init() is called
            # with default settings
            pass

        # hasattr AUTHORIZED by Code Review 2026-01-01
        # Justification: Defensive check - wandb.Run may not have .url in all modes
        run_url = self._run.url if hasattr(self._run, "url") else "N/A"
        _logger.info(f"Wandb run initialized: {run_url}")

    def emit(self, event: TelemetryEvent) -> None:
        """Emit telemetry event to wandb.

        Routes events to type-specific handlers based on event_type.
        Safe to call from background worker threads.

        Args:
            event: The telemetry event to log
        """
        if self._run is None:
            # This can happen if start() failed or wasn't called
            # Log at debug level to avoid spam - collector handles this
            _logger.debug("WandbBackend.emit() called before successful start()")
            return

        # Route event to appropriate handler based on type
        event_type = event.event_type

        if event_type == TelemetryEventType.TRAINING_STARTED:
            self._handle_training_started(event)
        elif event_type == TelemetryEventType.EPOCH_COMPLETED:
            self._handle_epoch_completed(event)
        elif event_type == TelemetryEventType.BATCH_EPOCH_COMPLETED:
            self._handle_batch_epoch_completed(event)
        elif event_type == TelemetryEventType.PPO_UPDATE_COMPLETED:
            self._handle_ppo_update(event)
        elif event_type == TelemetryEventType.SEED_GERMINATED:
            self._handle_seed_germinated(event)
        elif event_type == TelemetryEventType.SEED_STAGE_CHANGED:
            self._handle_seed_stage_changed(event)
        elif event_type == TelemetryEventType.SEED_FOSSILIZED:
            self._handle_seed_fossilized(event)
        elif event_type == TelemetryEventType.SEED_PRUNED:
            self._handle_seed_pruned(event)
        elif event_type in {
            TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            TelemetryEventType.RATIO_COLLAPSE_DETECTED,
            TelemetryEventType.VALUE_COLLAPSE_DETECTED,
            TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
            TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED,
        }:
            self._handle_anomaly(event)

    def _get_batch_step(self, event: TelemetryEvent, event_name: str) -> int | None:
        """Extract batch step from event.epoch, logging warning if missing.

        Returns None if epoch is not set, which signals the caller to skip
        logging (emitter bug - batch-level events must carry epoch).
        """
        if event.epoch is None:
            _logger.warning(
                "%s missing epoch (required for wandb step alignment) - skipping",
                event_name,
            )
            return None
        return event.epoch

    def _handle_training_started(self, event: TelemetryEvent) -> None:
        """Log training configuration to wandb config.

        This updates the wandb config with runtime training parameters
        that may not have been available at init time.
        """
        if not isinstance(event.data, TrainingStartedPayload):
            _logger.warning("TRAINING_STARTED event has non-TrainingStartedPayload data")
            return

        # Update wandb config with training parameters
        # wandb.config.update is safe to call multiple times
        wandb.config.update({
            "n_envs": event.data.n_envs,
            "max_epochs": event.data.max_epochs,
            "task": event.data.task,
        })

    def _handle_epoch_completed(self, event: TelemetryEvent) -> None:
        """Log per-environment epoch metrics.

        These are logged with per-env grouping to allow comparing
        individual environment performance over time.
        """
        if not isinstance(event.data, EpochCompletedPayload):
            _logger.debug("EPOCH_COMPLETED event has non-EpochCompletedPayload data")
            return

        env_id = event.data.env_id
        epoch = event.epoch or 0

        # Log per-env metrics (grouped by env_id for easy filtering in UI)
        metrics = {
            f"env_{env_id}/val_loss": event.data.val_loss,
            f"env_{env_id}/val_accuracy": event.data.val_accuracy,
            f"env_{env_id}/train_loss": event.data.train_loss,
            f"env_{env_id}/epoch": epoch,
        }

        wandb.log(metrics, step=self._env_epoch_step)
        self._env_epoch_step += 1

    def _handle_batch_epoch_completed(self, event: TelemetryEvent) -> None:
        """Log batch-level aggregated metrics.

        These are the main training curves - mean and std across all envs
        for each inner epoch within an episode.

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, BatchEpochCompletedPayload):
            _logger.debug("BATCH_EPOCH_COMPLETED event has non-BatchEpochCompletedPayload data")
            return

        step = self._get_batch_step(event, "BATCH_EPOCH_COMPLETED")
        if step is None:
            return

        metrics = {
            "batch/avg_accuracy": event.data.avg_accuracy,
            "batch/avg_reward": event.data.avg_reward,
            "batch/episodes_completed": event.data.episodes_completed,
            "batch/batch_idx": event.data.batch_idx,
            "batch/rolling_accuracy": event.data.rolling_accuracy,
        }

        # commit=False: PPO update is the commit point for batch-level metrics
        wandb.log(metrics, step=step, commit=False)

    def _handle_ppo_update(self, event: TelemetryEvent) -> None:
        """Log PPO training metrics.

        This is the COMMIT POINT for batch-level metrics. All batch/, ppo/,
        seeds/, and anomalies/ metrics share event.epoch as their x-axis.
        This handler commits (default) to finalize all pending batch metrics.

        These metrics track the RL agent's learning progress:
        - Policy/value losses
        - KL divergence (for early stopping)
        - Clip fraction (how often policy ratio was clipped)
        - Explained variance (how well value function predicts returns)
        """
        if not isinstance(event.data, PPOUpdatePayload):
            _logger.debug("PPO_UPDATE_COMPLETED event has non-PPOUpdatePayload data")
            return

        step = self._get_batch_step(event, "PPO_UPDATE_COMPLETED")
        if step is None:
            return

        if event.data.update_skipped:
            # Log skipped update marker at the correct step
            wandb.log({"ppo/update_skipped": 1}, step=step)
            return

        metrics = {
            "ppo/policy_loss": event.data.policy_loss,
            "ppo/value_loss": event.data.value_loss,
            "ppo/entropy": event.data.entropy,
            "ppo/kl_divergence": event.data.kl_divergence,
            "ppo/clip_fraction": event.data.clip_fraction,
            "ppo/grad_norm": event.data.grad_norm,
        }

        # Optional metrics (may be None)
        if event.data.explained_variance is not None:
            metrics["ppo/explained_variance"] = event.data.explained_variance
        if event.data.entropy_coef is not None:
            metrics["ppo/entropy_coef"] = event.data.entropy_coef
        if event.data.lr is not None:
            metrics["ppo/lr"] = event.data.lr

        # Commit point: this log() commits all pending batch-level metrics
        wandb.log(metrics, step=step)

    def _handle_seed_germinated(self, event: TelemetryEvent) -> None:
        """Log seed germination events.

        Tracks when new neural modules are created and what blueprint
        (architecture) was chosen.

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, SeedGerminatedPayload):
            _logger.debug("SEED_GERMINATED event has non-SeedGerminatedPayload data")
            return

        step = self._get_batch_step(event, "SEED_GERMINATED")
        if step is None:
            return

        slot_id = event.data.slot_id or event.slot_id or "unknown"

        metrics = {
            "seeds/germinated_count": 1,
            f"seeds/{slot_id}/blueprint": event.data.blueprint_id,
            f"seeds/{slot_id}/params": event.data.params,
        }

        wandb.log(metrics, step=step, commit=False)

    def _handle_seed_stage_changed(self, event: TelemetryEvent) -> None:
        """Log seed stage transitions.

        Tracks the seed lifecycle: DORMANT → GERMINATED → TRAINING →
        BLENDING → HOLDING → FOSSILIZED (or PRUNED at any stage).

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, SeedStageChangedPayload):
            _logger.debug("SEED_STAGE_CHANGED event has non-SeedStageChangedPayload data")
            return

        step = self._get_batch_step(event, "SEED_STAGE_CHANGED")
        if step is None:
            return

        slot_id = event.slot_id or "unknown"

        # Log stage name
        metrics = {
            f"seeds/{slot_id}/stage": event.data.to_stage,
        }

        wandb.log(metrics, step=step, commit=False)

    def _handle_seed_fossilized(self, event: TelemetryEvent) -> None:
        """Log seed fossilization (successful integration).

        This is a successful outcome - the seed improved the host model
        and was permanently integrated.

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, SeedFossilizedPayload):
            _logger.debug("SEED_FOSSILIZED event has non-SeedFossilizedPayload data")
            return

        step = self._get_batch_step(event, "SEED_FOSSILIZED")
        if step is None:
            return

        slot_id = event.slot_id or "unknown"

        metrics = {
            "seeds/fossilized_count": 1,
            f"seeds/{slot_id}/improvement": event.data.improvement,
            f"seeds/{slot_id}/params_added": event.data.params_added,
        }
        if event.data.blending_delta is not None:
            metrics[f"seeds/{slot_id}/blending_delta"] = event.data.blending_delta

        wandb.log(metrics, step=step, commit=False)

        # Also log to run summary for easy comparison across runs
        # hasattr AUTHORIZED by Code Review 2026-01-01
        # Justification: Defensive check - wandb.Run may not have .summary in disabled mode
        if self._run is not None and hasattr(self._run, "summary"):
            summary_key = f"best_improvement_{slot_id}"
            # Only update if this is better than previous best
            current_best = self._run.summary.get(summary_key, float("-inf"))
            if event.data.improvement > current_best:
                self._run.summary[summary_key] = event.data.improvement

    def _handle_seed_pruned(self, event: TelemetryEvent) -> None:
        """Log seed pruning (failed integration).

        This is a failure outcome - the seed didn't improve the host
        or caused regression, so it was removed.

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, SeedPrunedPayload):
            _logger.debug("SEED_PRUNED event has non-SeedPrunedPayload data")
            return

        step = self._get_batch_step(event, "SEED_PRUNED")
        if step is None:
            return

        slot_id = event.slot_id or "unknown"

        metrics = {
            "seeds/pruned_count": 1,
            f"seeds/{slot_id}/prune_reason": event.data.reason or "unknown",
        }

        wandb.log(metrics, step=step, commit=False)

    def _handle_anomaly(self, event: TelemetryEvent) -> None:
        """Log training anomalies and send alerts.

        These are critical issues that may indicate training is going off
        the rails: ratio explosion, value collapse, gradient pathology, etc.

        Alerts are sent to wandb for real-time notification.

        Uses commit=False because PPO_UPDATE_COMPLETED is the commit point.
        """
        if not isinstance(event.data, AnomalyDetectedPayload):
            _logger.debug("Anomaly event has non-AnomalyDetectedPayload data")
            return

        step = self._get_batch_step(event, event.event_type.name)
        if step is None:
            return

        anomaly_type = event.event_type.name

        # Log anomaly counts
        metrics = {
            f"anomalies/{anomaly_type}": 1,
            "anomalies/total_count": 1,
        }

        wandb.log(metrics, step=step, commit=False)

        # Send alert for real-time notification
        # wandb>=0.16.0 is required (pyproject.toml), so alert() always exists
        try:
            wandb.alert(
                title=f"Training Anomaly: {anomaly_type}",
                text=f"Episode {event.data.episode}: {event.data.detail}",
                level=wandb.AlertLevel.WARN,
            )
        except Exception as e:
            # Network errors or offline mode - don't crash training
            _logger.warning(f"Failed to send wandb alert: {e}")

    def close(self) -> None:
        """Finish wandb run and release resources.

        Called by KarnCollector.close() during shutdown.
        Idempotent - safe to call multiple times.
        """
        if self._run is not None:
            try:
                self._run.finish()
            except Exception as e:
                _logger.error(f"Error finishing wandb run: {e}")
            finally:
                self._run = None
