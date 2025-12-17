"""Gradient EMA Tracking for Drift Detection.

Uses exponential moving average to track gradient statistics over time.
Detects gradual drift that single-step anomaly checks miss.

Why EMA for gradients:
- Single-step checks catch sudden spikes but miss slow degradation
- Training can slowly drift into instability over hundreds of steps
- EMA smooths noise while tracking the underlying trend
- Drift indicator = |current - ema| / (ema + epsilon) catches divergence
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GradientEMATracker:
    """Tracks gradient statistics with EMA for drift detection.

    Drift indicator = |current - ema| / (ema + epsilon)
    High drift suggests training instability.

    Args:
        momentum: EMA momentum (0.99 = slow adaptation, 0.9 = faster)
        epsilon: Small constant for numerical stability in division
    """

    momentum: float = 0.99
    epsilon: float = 1e-8

    # EMA state (initialized on first update)
    ema_norm: float = field(default=0.0, init=False)
    ema_health: float = field(default=1.0, init=False)
    _initialized: bool = field(default=False, init=False)
    _update_count: int = field(default=0, init=False)

    def update(self, grad_norm: float, grad_health: float) -> dict[str, float]:
        """Update EMA and return drift indicators.

        Args:
            grad_norm: Current gradient norm
            grad_health: Current gradient health (0-1)

        Returns:
            Dict with ema values and drift indicators
        """
        self._update_count += 1

        if not self._initialized:
            # First update: initialize to current values
            self.ema_norm = grad_norm
            self.ema_health = grad_health
            self._initialized = True
            return {
                "ema_grad_norm": self.ema_norm,
                "ema_grad_health": self.ema_health,
                "norm_drift": 0.0,
                "health_drift": 0.0,
            }

        # Compute drift before updating EMA
        norm_drift = abs(grad_norm - self.ema_norm) / (self.ema_norm + self.epsilon)
        health_drift = abs(grad_health - self.ema_health) / (self.ema_health + self.epsilon)

        # Update EMA
        self.ema_norm = self.momentum * self.ema_norm + (1 - self.momentum) * grad_norm
        self.ema_health = self.momentum * self.ema_health + (1 - self.momentum) * grad_health

        return {
            "ema_grad_norm": self.ema_norm,
            "ema_grad_health": self.ema_health,
            "norm_drift": norm_drift,
            "health_drift": health_drift,
        }

    def check_drift(
        self,
        grad_norm: float,
        grad_health: float,
        drift_threshold: float = 0.5,
    ) -> tuple[bool, dict[str, float]]:
        """Update and check if drift exceeds threshold.

        Args:
            grad_norm: Current gradient norm
            grad_health: Current gradient health
            drift_threshold: Threshold for drift warning (0.5 = 50% deviation)

        Returns:
            Tuple of (has_drift, metrics_dict)
        """
        metrics = self.update(grad_norm, grad_health)
        has_drift = (
            metrics["norm_drift"] > drift_threshold
            or metrics["health_drift"] > drift_threshold
        )
        return has_drift, metrics

    def state_dict(self) -> dict[str, float | bool | int]:
        """Return state for checkpointing."""
        return {
            "ema_norm": self.ema_norm,
            "ema_health": self.ema_health,
            "initialized": self._initialized,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, float | bool | int]) -> None:
        """Load state from checkpoint."""
        self.ema_norm = float(state["ema_norm"])
        self.ema_health = float(state["ema_health"])
        self._initialized = bool(state["initialized"])
        self._update_count = int(state["update_count"])


__all__ = ["GradientEMATracker"]
