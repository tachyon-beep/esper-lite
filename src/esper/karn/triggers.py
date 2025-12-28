"""Karn Triggers - Anomaly detection and DenseTrace triggering.

Monitors telemetry events for anomalies and triggers dense trace capture
when thresholds are exceeded. Uses exponential moving averages for
baseline comparison.

Usage:
    from esper.karn.triggers import AnomalyDetector

    detector = AnomalyDetector()
    detector.check_epoch(epoch_snapshot)  # Returns trigger reason or None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from esper.karn.constants import AnomalyThresholds, PolicyThresholds
from esper.karn.store import (
    DenseTrace,
    DenseTraceTrigger,
    BatchMetrics,
    GateEvaluationTrace,
)

if TYPE_CHECKING:
    from esper.karn.store import EpochSnapshot

_logger = logging.getLogger(__name__)


@dataclass
class RollingStats:
    """Exponential moving average statistics for anomaly detection."""

    # EMA parameters
    alpha: float = 0.1  # Smoothing factor (higher = more weight to recent)

    # Loss statistics
    loss_ema: float = 0.0
    loss_initialized: bool = False

    # Accuracy statistics
    prev_accuracy: float = 0.0
    accuracy_initialized: bool = False

    # Gradient statistics
    grad_norm_ema: float = 1.0  # Avoid division by zero
    grad_initialized: bool = False

    def update_loss(self, loss: float) -> float:
        """Update loss EMA and return ratio to baseline."""
        if not self.loss_initialized:
            self.loss_ema = loss
            self.loss_initialized = True
            return 1.0

        ratio = loss / self.loss_ema if self.loss_ema > 0 else 1.0
        self.loss_ema = self.alpha * loss + (1 - self.alpha) * self.loss_ema
        return ratio

    def update_accuracy(self, accuracy: float) -> float:
        """Update accuracy tracking and return drop from previous."""
        if not self.accuracy_initialized:
            self.prev_accuracy = accuracy
            self.accuracy_initialized = True
            return 0.0

        drop = (self.prev_accuracy - accuracy) * 100  # Percentage points
        self.prev_accuracy = accuracy
        return drop

    def update_grad_norm(self, grad_norm: float) -> float:
        """Update gradient norm EMA and return ratio to baseline."""
        if not self.grad_initialized:
            self.grad_norm_ema = max(grad_norm, 0.01)  # Avoid zero
            self.grad_initialized = True
            return 1.0

        ratio = grad_norm / self.grad_norm_ema if self.grad_norm_ema > 0 else 1.0
        self.grad_norm_ema = self.alpha * grad_norm + (1 - self.alpha) * self.grad_norm_ema
        return ratio


@dataclass
class AnomalyDetector:
    """Detects training anomalies and triggers dense trace capture.

    Monitors epoch snapshots for:
    - Loss spikes (> threshold × rolling average)
    - Accuracy drops (> threshold percentage points)
    - Gradient explosions (> threshold × typical)
    - Gate failures (when enabled)
    - Stage transitions (when enabled)
    """

    config: DenseTraceTrigger = field(default_factory=DenseTraceTrigger)
    stats: RollingStats = field(default_factory=RollingStats)

    # Active trace being captured
    _active_trace: DenseTrace | None = field(default=None, repr=False)
    _trace_window_epochs: int = AnomalyThresholds.TRACE_WINDOW_EPOCHS

    def check_epoch(self, snapshot: "EpochSnapshot") -> str | None:
        """Check epoch for anomalies and return trigger reason if detected.

        Args:
            snapshot: The epoch snapshot to check

        Returns:
            Trigger reason string if anomaly detected, None otherwise
        """
        reasons = []

        # Check loss spike
        if snapshot.host.val_loss > 0:
            loss_ratio = self.stats.update_loss(snapshot.host.val_loss)
            if loss_ratio > self.config.loss_spike_threshold:
                reasons.append(f"loss_spike:{loss_ratio:.1f}x")
                _logger.warning(
                    f"Loss spike detected: {loss_ratio:.1f}x baseline "
                    f"(loss={snapshot.host.val_loss:.4f})"
                )

        # Check accuracy drop
        if snapshot.host.val_accuracy > 0 or self.stats.accuracy_initialized:
            acc_drop = self.stats.update_accuracy(snapshot.host.val_accuracy)
            if acc_drop > self.config.accuracy_drop_threshold:
                reasons.append(f"accuracy_drop:{acc_drop:.1f}pp")
                _logger.warning(
                    f"Accuracy drop detected: {acc_drop:.1f}pp "
                    f"(acc={snapshot.host.val_accuracy:.1f}%)"
                )

        # Check gradient explosion
        if snapshot.host.host_grad_norm > 0:
            grad_ratio = self.stats.update_grad_norm(snapshot.host.host_grad_norm)
            if grad_ratio > self.config.gradient_explosion:
                reasons.append(f"gradient_explosion:{grad_ratio:.0f}x")
                _logger.warning(
                    f"Gradient explosion detected: {grad_ratio:.0f}x baseline "
                    f"(norm={snapshot.host.host_grad_norm:.2f})"
                )

        # Check stage transitions
        if self.config.stage_transition:
            for slot_id, slot in snapshot.slots.items():
                if slot.epochs_in_stage == 0:  # Just transitioned
                    reasons.append(f"stage_transition:{slot_id}:{slot.stage.name}")

        # Check gate failures
        if self.config.gate_failure:
            for slot_id, slot in snapshot.slots.items():
                if slot.last_gate_passed is False:
                    reasons.append(f"gate_failure:{slot_id}:{slot.last_gate_attempted}")

        # Force dense mode
        if self.config.force_dense:
            reasons.append("force_dense")

        return ",".join(reasons) if reasons else None

    def start_trace(self, epoch: int, reason: str) -> DenseTrace:
        """Start a new dense trace capture.

        Args:
            epoch: The epoch that triggered the trace
            reason: The trigger reason string

        Returns:
            The new DenseTrace being captured
        """
        self._active_trace = DenseTrace(
            trigger_reason=reason,
            window_start_epoch=epoch,
            window_end_epoch=epoch + self._trace_window_epochs,
        )
        _logger.info(f"Started dense trace: {reason} (epochs {epoch}-{epoch + self._trace_window_epochs})")
        return self._active_trace

    def add_batch_metrics(self, metrics: BatchMetrics) -> None:
        """Add batch metrics to active trace if one exists."""
        if self._active_trace:
            self._active_trace.batch_metrics.append(metrics)

    def add_gate_evaluation(self, gate_trace: GateEvaluationTrace) -> None:
        """Add gate evaluation details to active trace."""
        if self._active_trace:
            self._active_trace.gate_evaluation_details = gate_trace

    def finalize_trace(self, epoch: int) -> DenseTrace | None:
        """Check if trace window has ended and return completed trace.

        Args:
            epoch: Current epoch

        Returns:
            Completed DenseTrace if window ended, None otherwise
        """
        if self._active_trace and epoch >= self._active_trace.window_end_epoch:
            completed = self._active_trace
            self._active_trace = None
            _logger.info(
                f"Completed dense trace: {completed.trigger_reason} "
                f"({len(completed.batch_metrics)} batch samples)"
            )
            return completed
        return None

    @property
    def is_capturing(self) -> bool:
        """True if currently capturing a dense trace."""
        return self._active_trace is not None

    def reset(self) -> None:
        """Reset detector state for new episode."""
        self.stats = RollingStats()
        self._active_trace = None


@dataclass
class PolicyAnomalyDetector:
    """Detects policy-specific anomalies (PPO diagnostics).

    Monitors for:
    - Value collapse (critic outputs same value)
    - Entropy collapse (policy becomes deterministic)
    - KL divergence spikes
    """

    # Thresholds
    value_std_threshold: float = PolicyThresholds.VALUE_STD_COLLAPSE
    entropy_threshold: float = PolicyThresholds.ENTROPY_COLLAPSE
    kl_threshold: float = PolicyThresholds.KL_SPIKE

    # Rolling stats
    value_stds: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    window_size: int = PolicyThresholds.WINDOW_SIZE

    def check_value_collapse(self, value_std: float) -> bool:
        """Check if critic is collapsing to constant output."""
        self.value_stds.append(value_std)
        if len(self.value_stds) > self.window_size:
            self.value_stds.pop(0)

        # Collapse if recent values all have low variance
        if len(self.value_stds) >= 3:
            recent_avg = sum(self.value_stds[-3:]) / 3
            if recent_avg < self.value_std_threshold:
                _logger.warning(f"Value collapse detected: std={recent_avg:.4f}")
                return True
        return False

    def check_entropy_collapse(self, entropy: float) -> bool:
        """Check if policy is becoming deterministic."""
        self.entropies.append(entropy)
        if len(self.entropies) > self.window_size:
            self.entropies.pop(0)

        if entropy < self.entropy_threshold:
            _logger.warning(f"Entropy collapse detected: H={entropy:.4f}")
            return True
        return False

    def check_kl_spike(self, kl: float) -> bool:
        """Check for large policy changes."""
        if kl > self.kl_threshold:
            _logger.warning(f"KL spike detected: KL={kl:.4f}")
            return True
        return False

    def reset(self) -> None:
        """Reset for new episode."""
        self.value_stds.clear()
        self.entropies.clear()
