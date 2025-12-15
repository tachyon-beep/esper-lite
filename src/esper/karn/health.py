"""Karn Health - System health monitoring and diagnostics.

Monitors system health during training:
- Memory usage (GPU and CPU)
- Training throughput and timing
- Gradient health (norms, NaN/Inf detection)
- Resource utilization

Usage:
    from esper.karn.health import HealthMonitor, SystemHealth

    monitor = HealthMonitor()
    health = monitor.check_health()

    if not health.is_healthy:
        print(f"Warning: {health.warnings}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.store import TelemetryStore

_logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    # GPU memory (bytes)
    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_total: int = 0

    # CPU memory (bytes)
    cpu_used: int = 0
    cpu_available: int = 0

    @property
    def gpu_utilization(self) -> float:
        """GPU memory utilization (0-1)."""
        return self.gpu_allocated / self.gpu_total if self.gpu_total > 0 else 0.0

    @property
    def gpu_allocated_gb(self) -> float:
        """GPU allocated memory in GB."""
        return self.gpu_allocated / (1024**3)

    @property
    def gpu_total_gb(self) -> float:
        """GPU total memory in GB."""
        return self.gpu_total / (1024**3)

    def __str__(self) -> str:
        return f"GPU: {self.gpu_allocated_gb:.2f}/{self.gpu_total_gb:.2f}GB ({self.gpu_utilization:.1%})"


@dataclass
class ThroughputStats:
    """Training throughput statistics."""

    epochs_per_minute: float = 0.0
    batches_per_second: float = 0.0
    samples_per_second: float = 0.0

    # Timing breakdown
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    data_loading_ms: float = 0.0


@dataclass
class GradientHealth:
    """Gradient health indicators."""

    mean_norm: float = 0.0
    max_norm: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    clip_ratio: float = 0.0  # Fraction of gradients clipped

    @property
    def is_healthy(self) -> bool:
        """True if gradients are healthy."""
        return not self.has_nan and not self.has_inf and self.mean_norm < 100.0


@dataclass
class SystemHealth:
    """Overall system health status."""

    is_healthy: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Component health
    memory: MemoryStats = field(default_factory=MemoryStats)
    throughput: ThroughputStats = field(default_factory=ThroughputStats)
    gradients: GradientHealth = field(default_factory=GradientHealth)

    # Timing
    check_time_ms: float = 0.0
    last_epoch: int = 0

    def add_warning(self, msg: str) -> None:
        """Add a warning and mark as potentially unhealthy."""
        self.warnings.append(msg)
        _logger.warning(f"Health warning: {msg}")

    def add_error(self, msg: str) -> None:
        """Add an error and mark as unhealthy."""
        self.errors.append(msg)
        self.is_healthy = False
        _logger.error(f"Health error: {msg}")


class HealthMonitor:
    """Monitors system health during training.

    Integrates with TelemetryStore to provide health metrics
    and detect potential issues before they become critical.
    """

    def __init__(
        self,
        store: "TelemetryStore | None" = None,
        gpu_warning_threshold: float = 0.9,
        grad_norm_warning: float = 50.0,
        grad_norm_error: float = 100.0,
    ):
        self.store = store
        self.gpu_warning_threshold = gpu_warning_threshold
        self.grad_norm_warning = grad_norm_warning
        self.grad_norm_error = grad_norm_error

        # Timing tracking
        self._last_check_time = time.monotonic()
        self._epoch_times: list[float] = []

    def check_health(self) -> SystemHealth:
        """Perform comprehensive health check.

        Returns:
            SystemHealth with current status and any warnings/errors
        """
        start = time.monotonic()
        health = SystemHealth()

        # Check memory
        health.memory = self._check_memory(health)

        # Check gradients from store
        if self.store:
            health.gradients = self._check_gradients(health)
            health.last_epoch = (
                self.store.latest_epoch.epoch if self.store.latest_epoch else 0
            )

        # Calculate throughput
        health.throughput = self._calculate_throughput()

        health.check_time_ms = (time.monotonic() - start) * 1000
        return health

    def _check_memory(self, health: SystemHealth) -> MemoryStats:
        """Check GPU and CPU memory usage."""
        stats = MemoryStats()

        # Try to get GPU memory (requires torch)
        try:
            import torch

            if torch.cuda.is_available():
                stats.gpu_allocated = torch.cuda.memory_allocated()
                stats.gpu_reserved = torch.cuda.memory_reserved()
                stats.gpu_total = torch.cuda.get_device_properties(0).total_memory

                if stats.gpu_utilization > self.gpu_warning_threshold:
                    health.add_warning(
                        f"GPU memory high: {stats.gpu_utilization:.1%} "
                        f"({stats.gpu_allocated_gb:.2f}GB)"
                    )
        except ImportError:
            pass  # torch not available
        except Exception as e:
            _logger.debug(f"Could not check GPU memory: {e}")

        # CPU memory (requires psutil)
        try:
            import psutil

            mem = psutil.virtual_memory()
            stats.cpu_used = mem.used
            stats.cpu_available = mem.available
        except ImportError:
            pass  # psutil not available

        return stats

    def _check_gradients(self, health: SystemHealth) -> GradientHealth:
        """Check gradient health from recent epochs."""
        grad_health = GradientHealth()

        if not self.store or not self.store.epoch_snapshots:
            return grad_health

        # Get recent gradient norms
        recent = list(self.store.epoch_snapshots)[-10:]
        norms = [s.host.host_grad_norm for s in recent if s.host.host_grad_norm > 0]

        if not norms:
            return grad_health

        grad_health.mean_norm = sum(norms) / len(norms)
        grad_health.max_norm = max(norms)

        # Check for NaN/Inf (indicated by very large norms)
        if grad_health.max_norm > 1e10:
            grad_health.has_inf = True
            health.add_error("Gradient explosion detected (possible Inf)")

        if grad_health.mean_norm != grad_health.mean_norm:  # NaN check
            grad_health.has_nan = True
            health.add_error("NaN gradients detected")

        # Warnings for high gradients
        if grad_health.max_norm > self.grad_norm_error:
            health.add_warning(f"Very high gradient norm: {grad_health.max_norm:.2f}")
        elif grad_health.max_norm > self.grad_norm_warning:
            health.add_warning(f"High gradient norm: {grad_health.max_norm:.2f}")

        return grad_health

    def _calculate_throughput(self) -> ThroughputStats:
        """Calculate training throughput from epoch timing."""
        stats = ThroughputStats()

        if not self.store or len(self.store.epoch_snapshots) < 2:
            return stats

        # Calculate epochs per minute from timestamps
        recent = list(self.store.epoch_snapshots)[-10:]
        if len(recent) >= 2:
            first_ts = recent[0].timestamp
            last_ts = recent[-1].timestamp
            duration_seconds = (last_ts - first_ts).total_seconds()

            if duration_seconds > 0:
                epochs = len(recent) - 1
                stats.epochs_per_minute = (epochs / duration_seconds) * 60

        return stats

    def record_epoch_time(self, duration_seconds: float) -> None:
        """Record epoch duration for throughput tracking."""
        self._epoch_times.append(duration_seconds)
        # Keep last 100 epochs
        if len(self._epoch_times) > 100:
            self._epoch_times.pop(0)

    def get_epoch_stats(self) -> dict[str, float]:
        """Get epoch timing statistics."""
        if not self._epoch_times:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        import statistics

        times = self._epoch_times
        return {
            "mean": statistics.mean(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        }


@dataclass
class VitalSigns:
    """Critical vital signs for training health.

    Based on Tolaria Governor's vital signs pattern.
    """

    # Loss stability
    loss_stable: bool = True
    loss_trend: str = "stable"  # "improving", "stable", "degrading"
    loss_spike_count: int = 0

    # Accuracy health
    accuracy_improving: bool = True
    epochs_without_improvement: int = 0

    # Seed health
    seed_failure_rate: float = 0.0  # Fraction of seeds culled
    active_seeds: int = 0

    # Overall
    critical: bool = False
    reason: str = ""


class VitalSignsMonitor:
    """Monitors critical training vital signs.

    Provides early warning for training failures based on
    patterns that typically precede collapse.
    """

    def __init__(
        self,
        store: "TelemetryStore | None" = None,
        loss_spike_threshold: float = 2.0,
        stagnation_epochs: int = 20,
    ):
        self.store = store
        self.loss_spike_threshold = loss_spike_threshold
        self.stagnation_epochs = stagnation_epochs

        self._best_accuracy = 0.0
        self._epochs_since_improvement = 0
        self._loss_history: list[float] = []

    def check_vitals(self) -> VitalSigns:
        """Check all vital signs.

        Returns:
            VitalSigns with current status
        """
        vitals = VitalSigns()

        if not self.store or not self.store.epoch_snapshots:
            return vitals

        latest = self.store.latest_epoch
        if not latest:
            return vitals

        # Check accuracy improvement
        current_acc = latest.host.val_accuracy
        if current_acc > self._best_accuracy:
            self._best_accuracy = current_acc
            self._epochs_since_improvement = 0
            vitals.accuracy_improving = True
        else:
            self._epochs_since_improvement += 1
            vitals.accuracy_improving = False

        vitals.epochs_without_improvement = self._epochs_since_improvement

        # Check loss stability
        self._loss_history.append(latest.host.val_loss)
        if len(self._loss_history) > 50:
            self._loss_history.pop(0)

        vitals.loss_stable, vitals.loss_trend = self._analyze_loss_trend()

        # Count loss spikes
        if len(self._loss_history) >= 2:
            recent_losses = self._loss_history[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            spikes = sum(1 for loss_value in recent_losses if loss_value > avg_loss * self.loss_spike_threshold)
            vitals.loss_spike_count = spikes

        # Check seed health
        total_seeds = 0
        culled_seeds = 0
        active = 0
        for slot in latest.slots.values():
            if slot.stage.value >= 2:  # GERMINATED or beyond
                total_seeds += 1
            if slot.stage.value == 7:  # CULLED
                culled_seeds += 1
            if slot.stage.value in (2, 3, 4, 5):  # Active states
                active += 1

        vitals.active_seeds = active
        vitals.seed_failure_rate = culled_seeds / total_seeds if total_seeds > 0 else 0.0

        # Determine if critical
        if self._epochs_since_improvement > self.stagnation_epochs:
            vitals.critical = True
            vitals.reason = f"No improvement for {self._epochs_since_improvement} epochs"
        elif vitals.loss_spike_count > 3:
            vitals.critical = True
            vitals.reason = f"Multiple loss spikes ({vitals.loss_spike_count})"
        elif vitals.seed_failure_rate > 0.8:
            vitals.critical = True
            vitals.reason = f"High seed failure rate ({vitals.seed_failure_rate:.0%})"

        return vitals

    def _analyze_loss_trend(self) -> tuple[bool, str]:
        """Analyze loss trend from history."""
        if len(self._loss_history) < 5:
            return True, "stable"

        recent = self._loss_history[-5:]
        earlier = self._loss_history[-10:-5] if len(self._loss_history) >= 10 else self._loss_history[:5]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        if recent_avg < earlier_avg * 0.95:
            return True, "improving"
        elif recent_avg > earlier_avg * 1.1:
            return False, "degrading"
        else:
            return True, "stable"

    def reset(self) -> None:
        """Reset vital signs tracking."""
        self._best_accuracy = 0.0
        self._epochs_since_improvement = 0
        self._loss_history.clear()
