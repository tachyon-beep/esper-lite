"""Tolaria Governor - The fail-safe watchdog mechanism.

Monitors model training for catastrophic failures (NaN, loss explosions)
and can rollback to Last Known Good state while punishing the RL agent.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

from esper.leyline import (
    TelemetryEvent,
    TelemetryEventType,
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_GOVERNOR_HISTORY_WINDOW,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    DEFAULT_GOVERNOR_LOSS_MULTIPLIER,
    SeedStage,
)
from esper.nissa import get_hub


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
        sensitivity: float = DEFAULT_GOVERNOR_SENSITIVITY,  # 6 sigma = very rare
        multiplier: float = DEFAULT_GOVERNOR_LOSS_MULTIPLIER,   # Loss must be Nx average
        absolute_threshold: float = DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,  # From leyline
        death_penalty: float = DEFAULT_GOVERNOR_DEATH_PENALTY,
        history_window: int = DEFAULT_GOVERNOR_HISTORY_WINDOW,  # From leyline
        min_panics_before_rollback: int = DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,  # From leyline
        random_guess_loss: float | None = None,  # Task-specific baseline (default: ln(10) for CIFAR-10)
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
        self._panic_loss: float | None = None  # Track loss that triggered panic
        # Random guessing loss = "lobotomy signature"
        # - CIFAR-10: ln(10) ≈ 2.3 (default)
        # - TinyStories: ln(50257) ≈ 10.8
        # - ImageNet: ln(1000) ≈ 6.9
        self.random_guess_loss = random_guess_loss if random_guess_loss is not None else math.log(10)
        # Capture an initial snapshot so rollback is always possible, even on first panic
        self.snapshot()

    def snapshot(self) -> None:
        """Save Last Known Good state to CPU memory to reduce GPU memory pressure.

        Only snapshots host parameters and fossilized seeds. Experimental
        (non-fossilized) seeds are excluded because they may be culled
        before rollback, causing state_dict key mismatches.

        Tensors are moved to CPU; non-tensor values are deep copied.
        This trades slightly slower rollback for significant GPU memory savings,
        especially for large models where snapshots could double GPU memory usage.
        """
        # Explicitly free old snapshot to prevent memory fragmentation
        if self.last_good_state is not None:
            del self.last_good_state
            self.last_good_state = None

        # Get model state, filtering out experimental seed keys
        full_state = self.model.state_dict()

        # If model has seed_slots, filter out non-fossilized seed parameters
        # hasattr AUTHORIZED by John on 2025-12-17 00:00:00 UTC
        # Justification: Feature detection - MorphogeneticModel has seed_slots, base models don't
        if hasattr(self.model, 'seed_slots'):
            experimental_prefixes = []
            for slot_id, slot in self.model.seed_slots.items():
                if slot.state is not None and slot.state.stage != SeedStage.FOSSILIZED:
                    # This seed is experimental - exclude its keys from snapshot
                    experimental_prefixes.append(f"seed_slots.{slot_id}.seed.")
                    experimental_prefixes.append(f"seed_slots.{slot_id}.alpha_schedule.")

            # Filter state dict
            filtered_state = {
                k: v for k, v in full_state.items()
                if not any(k.startswith(prefix) for prefix in experimental_prefixes)
            }
        else:
            filtered_state = full_state

        # Store on CPU to save GPU memory (rollback is rare, memory savings are constant)
        # Use no_grad() to prevent any autograd overhead during state extraction
        with torch.no_grad():
            self.last_good_state = {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in filtered_state.items()
            }

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
            self._panic_loss = current_loss
            self.consecutive_panics = self.min_panics_before_rollback  # Skip to rollback
            return True

        # Lobotomy detection: loss jumped to exactly random guessing
        # This catches "silent failures" where model outputs uniform probabilities
        if len(self.loss_history) >= 10:
            avg = sum(self.loss_history) / len(self.loss_history)
            # Relative tolerance: ~6.5% of random guess loss
            # - CIFAR-10 (ln(10)=2.3): tolerance = 0.15
            # - TinyStories (ln(50257)=10.8): tolerance = 0.70
            lobotomy_tolerance = 0.065 * self.random_guess_loss
            # If we were doing well (loss < 60% of random guess) and suddenly
            # hit exactly the random guess loss (±tolerance), that's a lobotomy
            if (avg < self.random_guess_loss * 0.6 and
                abs(current_loss - self.random_guess_loss) < lobotomy_tolerance):
                self._pending_panic = False
                self._panic_loss = current_loss
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
            self._panic_loss = current_loss
            # Only actually panic after consecutive anomalies
            if self.consecutive_panics >= self.min_panics_before_rollback:
                return True
            return False
        else:
            self.loss_history.append(current_loss)
            self.consecutive_panics = 0
            self._pending_panic = False
            return False

    def execute_rollback(
        self,
        *,
        env_id: int = 0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GovernorReport:
        """Emergency stop: restore LKG state and return punishment info.

        Rollback semantics (Option B):
        - Restore host + fossilized seeds from snapshot
        - Discard any live/experimental seeds (not fossilized)
        - Reset seed slots to empty/DORMANT state
        - (Optional) Reset optimizer momentum to prevent immediate re-crash

        Philosophy: Fossils are committed stable memory. Live seeds are
        experimental hypotheses - a catastrophic event means they failed
        the safety test and should be discarded.
        """
        # Get device from parameters, falling back to CPU if no parameters
        try:
            raw_device = next(self.model.parameters()).device
            device = raw_device if isinstance(raw_device, torch.device) else torch.device(raw_device)
        except StopIteration:
            device = torch.device("cpu")

        history = list(self.loss_history)
        avg = sum(history) / len(history) if history else 0.0
        variance = sum((x - avg) ** 2 for x in history) / len(history) if history else 0.0
        std = math.sqrt(variance) if variance > 0 else 0.0
        threshold = avg + (self.sensitivity * std)

        # Emit telemetry event (replaces print)
        hub = get_hub()
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            severity="critical",
            message="Critical instability detected - initiating rollback",
            data={
                "env_id": env_id,
                "device": str(device),
                "reason": "Structural Collapse",
                "loss_at_panic": self._panic_loss,
                "loss_threshold": threshold,
                "consecutive_panics": self.consecutive_panics,
            },
        ))

        if self.last_good_state is None:
            raise RuntimeError("Governor panic before first snapshot!")

        # Clear any live (non-fossilized) seeds FIRST
        # This removes seed parameters so state_dict keys match snapshot
        # Implements "revert to stable organism, dump all temporary grafts"
        # hasattr AUTHORIZED by John on 2025-12-01 16:30:00 UTC
        # Justification: Feature detection - MorphogeneticModel has seed_slots, base models may not
        if hasattr(self.model, 'seed_slots'):
            for slot in self.model.seed_slots.values():
                slot.cull("governor_rollback")

        # Restore host + fossilized seeds from snapshot
        # Use strict=False because:
        # 1. Snapshot excludes experimental seeds (by design)
        # 2. Current model may have different seed state than snapshot
        # 3. execute_rollback culls all seeds before restore, so missing keys are expected
        # Move all tensors to model device in one batch before loading, avoiding
        # individual CPU->GPU transfers for each parameter.
        # Use non_blocking=True for async CPU->GPU transfer
        state_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in self.last_good_state.items()
        }

        # CRITICAL: Synchronize CUDA stream before load_state_dict
        # load_state_dict() does NOT synchronize - without this, we load garbage
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        missing_keys, unexpected_keys = self.model.load_state_dict(state_on_device, strict=False)

        # Log if there are key mismatches (diagnostic for snapshot filtering issues)
        if missing_keys or unexpected_keys:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
                severity="warning",
                message="Rollback load_state_dict had key mismatches",
                data={
                    "missing_keys": list(missing_keys),
                    "unexpected_keys": list(unexpected_keys),
                    "env_id": env_id,
                    "device": str(device),
                    "reason": "State Dict Key Mismatch",
                },
            ))

        # Reset optimizer momentum (BUG-015 fix)
        # If we don't clear momentum, the optimizer will push the restored weights
        # in the same direction that caused the crash.
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state:
                        for value in state.values():
                            if isinstance(value, torch.Tensor) and value.is_floating_point():
                                value.zero_()

        # Reset panic counter after successful rollback to allow fresh recovery
        self.consecutive_panics = 0

        return GovernorReport(
            reason="Structural Collapse",
            loss_at_panic=self._panic_loss if self._panic_loss is not None else float('nan'),
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
        self._panic_loss = None
        self.snapshot()


__all__ = ["TolariaGovernor", "GovernorReport"]
