"""PPO update coordination extracted from VectorizedPPOTrainer.

This module handles the PPO update phase after rollout collection, including:
- Rollback handling and penalty injection
- PPO updates execution with target KL early stopping
- Anomaly detection and telemetry escalation
- Gradient drift and LSTM health monitoring
- Per-head entropy collapse detection
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from esper.simic.agent import PPOAgent
    from esper.simic.control import RewardNormalizer
    from esper.simic.telemetry import AnomalyDetector, GradientEMATracker
    from esper.simic.training.parallel_env_state import ParallelEnvState
    from esper.simic.vectorized_types import EpisodeRecord
    from esper.leyline import EpisodeOutcome


_logger = logging.getLogger(__name__)


@dataclass
class PPOUpdateResult:
    """Result from PPO update phase."""

    metrics: dict[str, Any]
    update_skipped: bool
    ppo_grad_norm: float | None
    ppo_update_time_ms: float | None
    consecutive_finiteness_failures: int
    should_continue: bool  # False if training should abort or skip anomaly detection


@dataclass(frozen=True)
class PPOCoordinatorConfig:
    """Configuration for PPO update coordination."""

    ppo_updates_per_batch: int
    max_epochs: int
    total_env_episodes: int
    amp_enabled: bool
    resolved_amp_dtype: torch.dtype | None


class PPOCoordinator:
    """Coordinates PPO updates after rollout collection.

    Extracted from VectorizedPPOTrainer.run() to reduce method size
    and improve testability.
    """

    def __init__(
        self,
        agent: "PPOAgent",
        config: PPOCoordinatorConfig,
        reward_normalizer: "RewardNormalizer",
        anomaly_detector: "AnomalyDetector",
        env_reward_configs: list[Any],
        reward_family_enum: Any,
        hub: Any | None,
        telemetry_config: Any | None,
        group_id: str,
        run_ppo_updates_fn: Callable[..., dict[str, Any]],
        handle_telemetry_escalation_fn: Callable[..., None],
        emit_anomaly_diagnostics_fn: Callable[..., None],
        logger: logging.Logger,
    ):
        self.agent = agent
        self.config = config
        self.reward_normalizer = reward_normalizer
        self.anomaly_detector = anomaly_detector
        self.env_reward_configs = env_reward_configs
        self.reward_family_enum = reward_family_enum
        self.hub = hub
        self.telemetry_config = telemetry_config
        self.group_id = group_id
        self.run_ppo_updates_fn = run_ppo_updates_fn
        self.handle_telemetry_escalation_fn = handle_telemetry_escalation_fn
        self.emit_anomaly_diagnostics_fn = emit_anomaly_diagnostics_fn
        self.logger = logger

    def handle_rollbacks(
        self,
        env_states: list["ParallelEnvState"],
        env_rollback_occurred: list[bool],
        env_total_rewards: list[float],
        episode_history: list["EpisodeRecord"],
        episode_outcomes: list["EpisodeOutcome"],
    ) -> list[int]:
        """Handle rollback penalty injection and recompute metrics.

        B1-DRL-01 fix: Inject death penalty so PPO learns to avoid catastrophic actions.
        B11-CR-02 fix: Recompute metrics after penalty injection.
        B11-CR-03 fix: Overwrite last reward with raw penalty for telemetry interpretability.

        Returns list of environment indices that experienced rollback.
        """
        rollback_env_indices = [
            i for i, occurred in enumerate(env_rollback_occurred) if occurred
        ]

        if not rollback_env_indices:
            return rollback_env_indices

        for env_idx in rollback_env_indices:
            # B1-DRL-01 fix: Inject death penalty so PPO learns to avoid
            # catastrophic actions. Previously get_punishment_reward() was dead code.
            # P1-NORM fix: Normalize penalty to match other rewards' scale.
            # Use normalize_only to avoid polluting running stats with rare outliers.
            penalty = env_states[env_idx].governor.get_punishment_reward()
            normalized_penalty = self.reward_normalizer.normalize_only(penalty)
            self.agent.buffer.mark_terminal_with_penalty(env_idx, normalized_penalty)
            # B11-CR-03 fix: OVERWRITE last reward with RAW penalty (for telemetry interpretability).
            # Buffer gets normalized_penalty (for PPO training stability).
            # Telemetry gets raw penalty (for cross-run comparability).
            if env_states[env_idx].episode_rewards:
                env_states[env_idx].episode_rewards[-1] = penalty

        # B11-CR-02 fix: Recompute metrics after penalty injection
        # Metrics were computed in the epoch loop BEFORE penalty was applied.
        # This caused EpisodeOutcome, episode_history, and stability to reflect PRE-PENALTY
        # rewards, making rollback episodes appear ~2x more rewarding and ~1.6x more stable.
        for env_idx in rollback_env_indices:
            env_state = env_states[env_idx]

            # 1. Recompute total reward from post-penalty episode_rewards
            env_total_rewards[env_idx] = sum(env_state.episode_rewards)

            # 2. Update episode_history entry for this env
            for entry in reversed(episode_history):
                if entry.env_id == env_idx:
                    entry.episode_reward = env_total_rewards[env_idx]
                    break

            # 3. Recompute stability from post-penalty variance
            recent_ep_rewards = (
                env_state.episode_rewards[-20:]
                if len(env_state.episode_rewards) >= 20
                else env_state.episode_rewards
            )
            if len(recent_ep_rewards) > 1:
                reward_var = float(np.var(recent_ep_rewards))
                stability = 1.0 / (1.0 + reward_var)
            else:
                stability = 1.0

            # 4. Find and replace EpisodeOutcome for this env
            for idx, outcome in enumerate(episode_outcomes):
                if outcome.env_id == env_idx:
                    corrected_outcome = dataclasses.replace(
                        outcome,
                        episode_reward=env_total_rewards[env_idx],
                        stability_score=stability,
                    )
                    episode_outcomes[idx] = corrected_outcome
                    break

        return rollback_env_indices

    def run_update(
        self,
        raw_states_for_normalizer_update: list[torch.Tensor],
        obs_normalizer: Any,
        envs_this_batch: int,
        throughput_step_time_ms_sum: float,
        throughput_dataloader_wait_ms_sum: float,
    ) -> tuple[dict[str, Any], bool, float | None]:
        """Execute PPO updates on collected rollout data.

        Returns:
            Tuple of (metrics dict, update_skipped flag, ppo_update_time_ms)
        """
        if len(self.agent.buffer) == 0:
            return {}, True, None

        update_start = time.perf_counter()
        metrics = self.run_ppo_updates_fn(
            agent=self.agent,
            ppo_updates_per_batch=self.config.ppo_updates_per_batch,
            raw_states_for_normalizer_update=raw_states_for_normalizer_update,
            obs_normalizer=obs_normalizer,
            use_amp=self.config.amp_enabled,
            amp_dtype=self.config.resolved_amp_dtype,
        )
        ppo_update_time_ms = (time.perf_counter() - update_start) * 1000.0

        if metrics:
            metrics["ppo_update_time_ms"] = ppo_update_time_ms
            metrics["ppo_grad_norm"] = metrics["pre_clip_grad_norm"]
            metrics["rollout_length"] = self.config.max_epochs
            metrics["rollout_episodes"] = envs_this_batch
            metrics["rollout_total_steps"] = len(self.agent.buffer)
            metrics["reward_mode"] = self.env_reward_configs[0].reward_mode.value
            metrics["reward_family"] = self.reward_family_enum.value
            metrics["entropy_coef"] = self.agent.entropy_coef

            metrics["throughput_step_time_ms_sum"] = throughput_step_time_ms_sum
            metrics["throughput_dataloader_wait_ms_sum"] = throughput_dataloader_wait_ms_sum

        return metrics, False, ppo_update_time_ms

    def check_finiteness_gate(
        self,
        metrics: dict[str, Any],
        consecutive_finiteness_failures: int,
    ) -> tuple[int, bool]:
        """Check finiteness gate and escalate on repeated failures.

        Returns:
            Tuple of (updated consecutive_failures count, should_continue flag)
        """
        if not metrics.get("ppo_update_performed", True):
            # All epochs skipped due to non-finite values
            skip_count = metrics.get("finiteness_gate_skip_count", 0)
            consecutive_finiteness_failures += 1
            self.logger.warning(
                f"PPO update skipped (all {skip_count} epochs hit finiteness gate). "
                f"Consecutive failures: {consecutive_finiteness_failures}/3"
            )

            # Escalate after 3 consecutive failures (DRL best practice)
            if consecutive_finiteness_failures >= 3:
                raise RuntimeError(
                    f"PPO training failed: {consecutive_finiteness_failures} consecutive updates "
                    "skipped due to non-finite values. Check policy/value network outputs for NaN. "
                    f"Last failure: {metrics.get('finiteness_gate_failures')}"
                )
            return consecutive_finiteness_failures, False

        # Reset counter on successful update
        return 0, True

    def check_gradient_drift(
        self,
        grad_ema_tracker: "GradientEMATracker | None",
        ppo_grad_norm: float | None,
    ) -> dict[str, Any] | None:
        """Compute gradient drift metrics if tracker is available.

        Returns drift_metrics dict or None if not tracking.
        """
        if grad_ema_tracker is None or ppo_grad_norm is None:
            return None

        # Compute gradient health (0-1)
        if ppo_grad_norm < 1e-7:
            grad_health = 0.3  # Vanishing gradients
        elif ppo_grad_norm > 100.0:
            grad_health = 0.3  # Exploding gradients
        else:
            grad_health = 1.0  # Healthy range

        return grad_ema_tracker.update(ppo_grad_norm, grad_health)

    def run_anomaly_detection(
        self,
        metrics: dict[str, Any],
        drift_metrics: dict[str, Any] | None,
        batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None,
        batch_epoch_id: int,
        batch_idx: int,
    ) -> None:
        """Run anomaly detection on PPO update metrics.

        Includes:
        - Basic anomaly checks (ratio explosion, NaN, inf, etc.)
        - Gradient drift detection
        - LSTM health monitoring
        - Per-head entropy collapse detection
        """
        from esper.simic.telemetry import compute_lstm_health

        # Get numeric metric values for NaN/inf checks
        metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]

        anomaly_report = self.anomaly_detector.check_all(
            # MANDATORY metrics after PPO update - fail loudly if missing
            ratio_max=metrics["ratio_max"],
            ratio_min=metrics["ratio_min"],
            explained_variance=metrics.get("explained_variance", 0.0),
            has_nan=any(math.isnan(v) for v in metric_values),
            has_inf=any(math.isinf(v) for v in metric_values),
            current_episode=batch_epoch_id,
            total_episodes=self.config.total_env_episodes,
        )

        # B7-DRL-01: Check gradient drift and merge into anomaly report
        if drift_metrics is not None:
            drift_report = self.anomaly_detector.check_gradient_drift(
                norm_drift=drift_metrics["norm_drift"],
                health_drift=drift_metrics["health_drift"],
            )
            if drift_report.has_anomaly:
                anomaly_report.has_anomaly = True
                anomaly_report.anomaly_types.extend(drift_report.anomaly_types)
                anomaly_report.details.update(drift_report.details)

        # B7-DRL-04: Check LSTM hidden state health after PPO update
        # LSTM hidden states can become corrupted during BPTT - monitor for
        # explosion/saturation (RMS > threshold), vanishing (RMS < 1e-6), or NaN/Inf.
        lstm_health = compute_lstm_health(batched_lstm_hidden)
        if lstm_health is not None:
            lstm_report = self.anomaly_detector.check_lstm_health(
                h_rms=lstm_health.h_rms,
                c_rms=lstm_health.c_rms,
                h_env_rms_max=lstm_health.h_env_rms_max,
                c_env_rms_max=lstm_health.c_env_rms_max,
                has_nan=lstm_health.has_nan,
                has_inf=lstm_health.has_inf,
            )
            if lstm_report.has_anomaly:
                anomaly_report.has_anomaly = True
                anomaly_report.anomaly_types.extend(lstm_report.anomaly_types)
                anomaly_report.details.update(lstm_report.details)
            # Add LSTM health to metrics for telemetry display in Sanctum
            metrics.update(lstm_health.to_dict())

        # Per-head entropy collapse detection (Task 6)
        # Check individual action heads for collapse even when total entropy appears healthy
        head_entropies_raw = metrics.get("head_entropies")
        if head_entropies_raw:
            # Convert per-epoch lists to mean per head
            mean_head_entropies = {
                head: sum(values) / len(values) if values else 0.0
                for head, values in head_entropies_raw.items()
            }
            per_head_report = self.anomaly_detector.check_per_head_entropy_collapse(
                mean_head_entropies
            )
            if per_head_report.has_anomaly:
                # Log warnings but don't halt - this is early warning
                for anomaly_type in per_head_report.anomaly_types:
                    detail = per_head_report.details.get(anomaly_type, "")
                    self.logger.warning(
                        f"Per-head entropy anomaly: {anomaly_type} - {detail}"
                    )
                # Merge into main anomaly report for telemetry escalation
                anomaly_report.has_anomaly = True
                anomaly_report.anomaly_types.extend(per_head_report.anomaly_types)
                anomaly_report.details.update(per_head_report.details)

        self.handle_telemetry_escalation_fn(anomaly_report, self.telemetry_config)
        self.emit_anomaly_diagnostics_fn(
            self.hub,
            anomaly_report,
            self.agent,
            batch_epoch_id,
            batch_idx,
            self.config.max_epochs,
            self.config.total_env_episodes,
            False,
            group_id=self.group_id,
        )


__all__ = ["PPOCoordinator", "PPOCoordinatorConfig", "PPOUpdateResult"]
