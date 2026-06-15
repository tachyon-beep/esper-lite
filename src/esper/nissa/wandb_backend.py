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
        #
        # Settings for system metrics: _disable_stats controls GPU/CPU/memory logging
        # See: https://docs.wandb.ai/support/how_can_i_disable_logging_of_system_metrics_to_wb/
        settings = wandb.Settings(_disable_stats=not self.log_system)
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            group=self.group,
            name=self.name,
            mode=self.mode,
            save_code=self.log_code,
            settings=settings,
        )

        if self._run is None:
            _logger.error("wandb.init() returned None - run not initialized")
            return

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
        d = event.data
        config_update: dict[str, Any] = {
            "n_envs": d.n_envs,
            "max_epochs": d.max_epochs,
            "max_batches": d.max_batches,
            "task": d.task,
            # host_params: model size baseline; param efficiency is measured
            # against this, so it must land in the run config.
            "host_params": d.host_params,
            "param_budget": d.param_budget,
            "n_episodes": d.n_episodes,
            "seed": d.seed,
            # Key PPO hyperparameters for cross-run comparison.
            "lr": d.lr,
            "clip_ratio": d.clip_ratio,
            "entropy_coef": d.entropy_coef,
            "reward_mode": d.reward_mode,
            "policy_device": d.policy_device,
            # Distributed / precision posture.
            "world_size": d.world_size,
            "amp_enabled": d.amp_enabled,
            "compile_enabled": d.compile_enabled,
        }
        # Optional diagnostics (omit when absent - never fabricate).
        if d.amp_dtype is not None:
            config_update["amp_dtype"] = d.amp_dtype
        if d.compile_backend is not None:
            config_update["compile_backend"] = d.compile_backend
        if d.compile_mode is not None:
            config_update["compile_mode"] = d.compile_mode
        if d.proof_baseline_mode is not None:
            config_update["proof_baseline_mode"] = d.proof_baseline_mode
        if d.proof_baseline_pair_id is not None:
            config_update["proof_baseline_pair_id"] = d.proof_baseline_pair_id
        if d.proof_baseline_lifecycle_policy is not None:
            config_update["proof_baseline_lifecycle_policy"] = (
                d.proof_baseline_lifecycle_policy
            )
        if d.proof_baseline_schedule_id is not None:
            config_update["proof_baseline_schedule_id"] = d.proof_baseline_schedule_id
        if d.proof_baseline_schedule_hash is not None:
            config_update["proof_baseline_schedule_hash"] = d.proof_baseline_schedule_hash
        if d.proof_baseline_schedule_version is not None:
            config_update["proof_baseline_schedule_version"] = (
                d.proof_baseline_schedule_version
            )
        if d.proof_baseline_schedule_action_count is not None:
            config_update["proof_baseline_schedule_action_count"] = (
                d.proof_baseline_schedule_action_count
            )

        wandb.config.update(config_update)

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
        # Only include non-None values to avoid data quality issues in wandb
        metrics: dict[str, Any] = {
            f"env_{env_id}/val_loss": event.data.val_loss,
            f"env_{env_id}/val_accuracy": event.data.val_accuracy,
            f"env_{env_id}/epoch": epoch,
        }
        # Optional training metrics - only log if present
        if event.data.train_loss is not None:
            metrics[f"env_{env_id}/train_loss"] = event.data.train_loss
        if event.data.train_accuracy is not None:
            metrics[f"env_{env_id}/train_accuracy"] = event.data.train_accuracy

        wandb.log(metrics, step=self._env_epoch_step)
        self._env_epoch_step += 1

    def _handle_batch_epoch_completed(self, event: TelemetryEvent) -> None:
        """Log batch-level aggregated metrics.

        These are the main training curves - mean and std across all envs
        for each inner epoch within an episode.
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
        }
        # Only log rolling accuracy when it was actually measured; None = not
        # measured and must not be logged as a fabricated point (LN-004).
        if event.data.rolling_accuracy is not None:
            metrics["batch/rolling_accuracy"] = event.data.rolling_accuracy

        wandb.log(metrics, step=step)

    def _handle_ppo_update(self, event: TelemetryEvent) -> None:
        """Log PPO training metrics.

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

        p = event.data
        metrics = {
            "ppo/policy_loss": p.policy_loss,
            "ppo/value_loss": p.value_loss,
            "ppo/entropy": p.entropy,
            "ppo/kl_divergence": p.kl_divergence,
            "ppo/clip_fraction": p.clip_fraction,
            "ppo/grad_norm": p.grad_norm,
            # Gradient health: pre-clip norm reveals true gradient magnitude
            # (post-clip ~1.0 hides explosion); NaN/Inf grad counts are the
            # fail-fast signal; dead/exploding layer counts localize pathology.
            "ppo/pre_clip_grad_norm": p.pre_clip_grad_norm,
            "ppo/nan_grad_count": p.nan_grad_count,
            "ppo/inf_grad_count": p.inf_grad_count,
            "ppo/dead_layers": p.dead_layers,
            "ppo/exploding_layers": p.exploding_layers,
            # Ratio extremes: explosion/collapse precede training divergence.
            "ppo/ratio_max": p.ratio_max,
            "ppo/ratio_min": p.ratio_min,
            "ppo/joint_ratio_max": p.joint_ratio_max,
            # Directional clip fractions: WHERE clipping bites (up vs down).
            "ppo/clip_fraction_positive": p.clip_fraction_positive,
            "ppo/clip_fraction_negative": p.clip_fraction_negative,
            # Value-function quality: scale, calibration, Bellman residual.
            "ppo/value_target_scale": p.value_target_scale,
            "ppo/v_return_correlation": p.v_return_correlation,
            "ppo/td_error_mean": p.td_error_mean,
            "ppo/bellman_error": p.bellman_error,
        }

        # Optional metrics (may be None - omit rather than fabricate)
        if p.explained_variance is not None:
            metrics["ppo/explained_variance"] = p.explained_variance
        if p.entropy_coef is not None:
            metrics["ppo/entropy_coef"] = p.entropy_coef
        if p.lr is not None:
            metrics["ppo/lr"] = p.lr

        # LSTM hidden-state health (B7-DRL-04 / SIMIC-PROD-003).
        # Update-time vs rollout-time are distinct signals; log both when present.
        # RMS metrics are scale-free; nan/inf flags catch irrecoverable corruption.
        if p.lstm_h_rms is not None:
            metrics["ppo/lstm_h_rms"] = p.lstm_h_rms
        if p.lstm_c_rms is not None:
            metrics["ppo/lstm_c_rms"] = p.lstm_c_rms
        if p.lstm_h_env_rms_max is not None:
            metrics["ppo/lstm_h_env_rms_max"] = p.lstm_h_env_rms_max
        if p.lstm_c_env_rms_max is not None:
            metrics["ppo/lstm_c_env_rms_max"] = p.lstm_c_env_rms_max
        metrics["ppo/lstm_has_nan"] = int(p.lstm_has_nan)
        metrics["ppo/lstm_has_inf"] = int(p.lstm_has_inf)
        if p.rollout_lstm_h_rms is not None:
            metrics["ppo/rollout_lstm_h_rms"] = p.rollout_lstm_h_rms
        if p.rollout_lstm_c_rms is not None:
            metrics["ppo/rollout_lstm_c_rms"] = p.rollout_lstm_c_rms
        if p.rollout_lstm_h_env_rms_max is not None:
            metrics["ppo/rollout_lstm_h_env_rms_max"] = p.rollout_lstm_h_env_rms_max
        if p.rollout_lstm_c_env_rms_max is not None:
            metrics["ppo/rollout_lstm_c_env_rms_max"] = p.rollout_lstm_c_env_rms_max
        metrics["ppo/rollout_lstm_has_nan"] = int(p.rollout_lstm_has_nan)
        metrics["ppo/rollout_lstm_has_inf"] = int(p.rollout_lstm_has_inf)

        # AMP finiteness: overflow detection drives loss-scale backoff / skips.
        metrics["ppo/amp_overflow_detected"] = int(p.amp_overflow_detected)
        if p.loss_scale is not None:
            metrics["ppo/loss_scale"] = p.loss_scale

        wandb.log(metrics, step=step)

    def _handle_seed_germinated(self, event: TelemetryEvent) -> None:
        """Log seed germination events.

        Tracks when new neural modules are created and what blueprint
        (architecture) was chosen.
        """
        if not isinstance(event.data, SeedGerminatedPayload):
            _logger.debug("SEED_GERMINATED event has non-SeedGerminatedPayload data")
            return

        step = self._get_batch_step(event, "SEED_GERMINATED")
        if step is None:
            return

        slot_id = event.slot_id or event.data.slot_id
        if slot_id is None:
            raise ValueError("slot_id required for SEED_GERMINATED event")

        metrics = {
            "seeds/germinated_count": 1,
            f"seeds/{slot_id}/blueprint": event.data.blueprint_id,
            f"seeds/{slot_id}/params": event.data.params,
        }

        wandb.log(metrics, step=step)

    def _handle_seed_stage_changed(self, event: TelemetryEvent) -> None:
        """Log seed stage transitions.

        Tracks the seed lifecycle: DORMANT → GERMINATED → TRAINING →
        BLENDING → HOLDING → FOSSILIZED (or PRUNED at any stage).
        """
        if not isinstance(event.data, SeedStageChangedPayload):
            _logger.debug("SEED_STAGE_CHANGED event has non-SeedStageChangedPayload data")
            return

        step = self._get_batch_step(event, "SEED_STAGE_CHANGED")
        if step is None:
            return

        slot_id = event.slot_id or event.data.slot_id
        if slot_id is None:
            raise ValueError("slot_id required for SEED_STAGE_CHANGED event")

        # Log stage name
        metrics = {
            f"seeds/{slot_id}/stage": event.data.to_stage,
        }

        wandb.log(metrics, step=step)

    def _handle_seed_fossilized(self, event: TelemetryEvent) -> None:
        """Log seed fossilization (successful integration).

        This is a successful outcome - the seed improved the host model
        and was permanently integrated.
        """
        if not isinstance(event.data, SeedFossilizedPayload):
            _logger.debug("SEED_FOSSILIZED event has non-SeedFossilizedPayload data")
            return

        step = self._get_batch_step(event, "SEED_FOSSILIZED")
        if step is None:
            return

        slot_id = event.slot_id or event.data.slot_id
        if slot_id is None:
            raise ValueError("slot_id required for SEED_FOSSILIZED event")

        metrics = {
            "seeds/fossilized_count": 1,
            f"seeds/{slot_id}/improvement": event.data.improvement,
            f"seeds/{slot_id}/params_added": event.data.params_added,
        }
        if event.data.blending_delta is not None:
            metrics[f"seeds/{slot_id}/blending_delta"] = event.data.blending_delta

        wandb.log(metrics, step=step)

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
        """
        if not isinstance(event.data, SeedPrunedPayload):
            _logger.debug("SEED_PRUNED event has non-SeedPrunedPayload data")
            return

        step = self._get_batch_step(event, "SEED_PRUNED")
        if step is None:
            return

        slot_id = event.slot_id or event.data.slot_id
        if slot_id is None:
            raise ValueError("slot_id required for SEED_PRUNED event")

        metrics = {
            "seeds/pruned_count": 1,
            f"seeds/{slot_id}/prune_reason": event.data.reason or "unknown",
        }

        wandb.log(metrics, step=step)

    def _handle_anomaly(self, event: TelemetryEvent) -> None:
        """Log training anomalies and send alerts.

        These are critical issues that may indicate training is going off
        the rails: ratio explosion, value collapse, gradient pathology, etc.

        Alerts are sent to wandb for real-time notification.
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

        wandb.log(metrics, step=step)

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
