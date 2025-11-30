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
        sensitivity: float = 3.0,
        death_penalty: float = 10.0,
        history_window: int = 10,
    ):
        self.model = model
        self.sensitivity = sensitivity
        self.death_penalty = death_penalty
        self.loss_history: deque[float] = deque(maxlen=history_window)
        self.last_good_state: dict | None = None
        self.consecutive_panics: int = 0

    def snapshot(self) -> None:
        """Save Last Known Good state (lightweight RAM copy)."""
        self.last_good_state = copy.deepcopy(self.model.state_dict())

    def check_vital_signs(self, current_loss: float) -> bool:
        """Check if the system is dying.

        Returns True if panic condition detected:
        - NaN or Inf in loss
        - Loss exceeds (mean + k*std) AND (mean * 1.2)
        """
        # Immediate panic on NaN/Inf
        if math.isnan(current_loss) or math.isinf(current_loss):
            return True

        # Need history for statistical detection
        if len(self.loss_history) < 5:
            self.loss_history.append(current_loss)
            return False

        # Statistical anomaly detection
        history = list(self.loss_history)
        avg = sum(history) / len(history)
        variance = sum((x - avg) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        threshold = avg + (self.sensitivity * std)

        # Panic if loss exceeds threshold AND is 20% above average
        is_panic = current_loss > threshold and current_loss > avg * 1.2

        if not is_panic:
            self.loss_history.append(current_loss)
            self.consecutive_panics = 0

        return is_panic

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
        self.last_good_state = None
        self.consecutive_panics = 0


__all__ = ["TolariaGovernor", "GovernorReport"]
