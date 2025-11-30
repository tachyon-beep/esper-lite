"""Tolaria Governor - The fail-safe watchdog mechanism.

Monitors model training for catastrophic failures (NaN, loss explosions)
and can rollback to Last Known Good state while punishing the RL agent.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import torch.nn as nn


@dataclass
class GovernorReport:
    """Report from a rollback event."""
    reason: str
    loss_at_panic: float
    loss_threshold: float
    consecutive_panics: int
    rollback_occurred: bool


class TolariaGovernor:
    """The Super-Ego of the training system.

    Monitors model training for catastrophic failures and can rollback
    to Last Known Good state while signaling punishment to the RL agent.

    Capabilities:
    1. Anomaly Detection - NaN/Inf and statistical outliers
    2. State Reversion - RAM checkpoint for instant rollback
    3. RL Punishment - Returns negative reward for PPO buffer injection
    """

    def __init__(
        self,
        model: nn.Module,
        sensitivity: float = 6.0,  # 6 sigma = very rare
        multiplier: float = 3.0,   # Loss must be 3x average
        absolute_threshold: float = 10.0,  # Loss must exceed this absolute value
        death_penalty: float = 10.0,
        history_window: int = 20,  # Longer window for stable estimates
        min_panics_before_rollback: int = 2,  # Require consecutive panics
        num_classes: int = 10,  # For random-guess detection (CIFAR-10 default)
    ):
        self.model = model
        self.sensitivity = sensitivity
        self.multiplier = multiplier
        self.absolute_threshold = absolute_threshold
        self.death_penalty = death_penalty
        self.loss_history: deque[float] = deque(maxlen=history_window)
        self.last_good_state: dict | None = None
        self.consecutive_panics: int = 0
        self.min_panics_before_rollback = min_panics_before_rollback
        self._pending_panic: bool = False
        # Random guessing loss = ln(num_classes), the "lobotomy signature"
        self.random_guess_loss = math.log(num_classes)
        # Capture an initial snapshot so rollback is always possible, even on first panic
        self.snapshot()

    def snapshot(self) -> None:
        """Save Last Known Good state (lightweight RAM copy)."""
        self.last_good_state = copy.deepcopy(self.model.state_dict())

    def check_vital_signs(self, current_loss: float) -> bool:
        """Check if the system is irreparably damaged.

        Returns True only for truly catastrophic failures:
        - NaN or Inf in loss (immediate)
        - Loss exceeds absolute threshold AND statistical threshold AND multiplier
        - Only after consecutive panics (to avoid false positives)

        This is a NUCLEAR OPTION - should almost never trigger during normal training.
        """
        # Immediate panic on NaN/Inf - these are always catastrophic
        if math.isnan(current_loss) or math.isinf(current_loss):
            self._pending_panic = False
            self.consecutive_panics = self.min_panics_before_rollback  # Skip to rollback
            return True

        # Lobotomy detection: loss jumped to exactly random guessing
        # This catches "silent failures" where model outputs uniform probabilities
        if len(self.loss_history) >= 10:
            avg = sum(self.loss_history) / len(self.loss_history)
            # If we were doing well (loss < 60% of random guess) and suddenly
            # hit exactly the random guess loss (±0.15), that's a lobotomy
            if (avg < self.random_guess_loss * 0.6 and
                abs(current_loss - self.random_guess_loss) < 0.15):
                self._pending_panic = False
                self.consecutive_panics = self.min_panics_before_rollback
                return True

        # Need sufficient history for stable estimates
        if len(self.loss_history) < 10:
            self.loss_history.append(current_loss)
            return False

        # Statistical anomaly detection
        history = list(self.loss_history)
        avg = sum(history) / len(history)
        variance = sum((x - avg) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        statistical_threshold = avg + (self.sensitivity * std)
        multiplier_threshold = avg * self.multiplier

        # ALL conditions must be met for panic:
        # 1. Loss exceeds absolute threshold (e.g., > 10.0)
        # 2. Loss exceeds statistical threshold (6 sigma)
        # 3. Loss exceeds multiplier threshold (3x average)
        is_anomaly = (
            current_loss > self.absolute_threshold and
            current_loss > statistical_threshold and
            current_loss > multiplier_threshold
        )

        if is_anomaly:
            self.consecutive_panics += 1
            self._pending_panic = True
            # Only actually panic after consecutive anomalies
            if self.consecutive_panics >= self.min_panics_before_rollback:
                return True
            return False
        else:
            self.loss_history.append(current_loss)
            self.consecutive_panics = 0
            self._pending_panic = False
            return False

    def execute_rollback(self) -> GovernorReport:
        """Emergency stop: restore LKG state and return punishment info."""
        print(f"[GOVERNOR] CRITICAL INSTABILITY DETECTED. INITIATING ROLLBACK.")

        if self.last_good_state is None:
            raise RuntimeError("Governor panic before first snapshot!")

        # Restore model weights
        self.model.load_state_dict(self.last_good_state)

        self.consecutive_panics += 1

        # Calculate what the threshold was
        history = list(self.loss_history)
        avg = sum(history) / len(history) if history else 0.0
        variance = sum((x - avg) ** 2 for x in history) / len(history) if history else 0.0
        std = math.sqrt(variance) if variance > 0 else 0.0
        threshold = avg + (self.sensitivity * std)

        return GovernorReport(
            reason="Structural Collapse",
            loss_at_panic=float('nan'),  # We don't store the panic loss
            loss_threshold=threshold,
            consecutive_panics=self.consecutive_panics,
            rollback_occurred=True,
        )

    def get_punishment_reward(self) -> float:
        """Return the negative reward for RL agent punishment."""
        return -self.death_penalty

    def reset(self) -> None:
        """Reset governor state (for new episode)."""
        self.loss_history.clear()
        self.consecutive_panics = 0
        self._pending_panic = False
        self.snapshot()


__all__ = ["TolariaGovernor", "GovernorReport"]
