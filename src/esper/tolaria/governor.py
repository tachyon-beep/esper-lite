"""Tolaria Governor - The fail-safe watchdog mechanism.

Monitors model training for catastrophic failures (NaN, loss explosions)
and can rollback to Last Known Good state while punishing the RL agent.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any, cast

import torch
import torch.nn as nn

from esper.leyline import (
    TelemetryEvent,
    TelemetryEventType,
    GovernorRollbackPayload,
    GovernorReport,  # Protocol dataclass from leyline
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_GOVERNOR_HISTORY_WINDOW,
    MIN_GOVERNOR_HISTORY_SAMPLES,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    DEFAULT_GOVERNOR_LOSS_MULTIPLIER,
    SeedStage,
)
from esper.leyline.telemetry import GovernorPanicReason

# NOTE: get_hub imported at function scope to defer telemetry hub initialization


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
        # Fail fast on invalid config that would silently disable detection
        if history_window < MIN_GOVERNOR_HISTORY_SAMPLES:
            raise ValueError(
                f"history_window ({history_window}) must be >= MIN_GOVERNOR_HISTORY_SAMPLES "
                f"({MIN_GOVERNOR_HISTORY_SAMPLES}). Smaller windows disable anomaly detection."
            )

        self.model = model
        self.sensitivity = sensitivity
        self.multiplier = multiplier
        self.absolute_threshold = absolute_threshold
        self.death_penalty = death_penalty
        self.loss_history: deque[float] = deque(maxlen=history_window)
        self.last_good_state: dict[str, Any] | None = None
        self.consecutive_panics: int = 0
        self.min_panics_before_rollback = min_panics_before_rollback
        self._pending_panic: bool = False
        self._panic_loss: float | None = None  # Track loss that triggered panic
        self._panic_reason: str | None = None  # Seed prune reason for catastrophic events
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
        (non-fossilized) seeds are excluded because they may be pruned
        before rollback, causing state_dict key mismatches.

        Tensors are moved to CPU; non-tensor values are deep copied.
        This trades slightly slower rollback for significant GPU memory savings,
        especially for large models where snapshots could double GPU memory usage.

        STREAM CONTRACT: This method must be called OUTSIDE any non-default CUDA
        stream context, after all async training operations have been synchronized.
        The model.state_dict() operation runs on the default stream, and tensor.cpu()
        synchronizes with the source device. Caller is responsible for ensuring no
        concurrent writes to model parameters during snapshot.

        Current call site (vectorized.py) satisfies this by calling snapshot() after
        stream.synchronize() and outside the per-env stream context.
        """
        # C7 FIX: Explicitly free old snapshot to allow garbage collection
        # NOTE: We intentionally do NOT call torch.cuda.empty_cache() here.
        # The CUDA caching allocator is designed to hold freed memory for fast
        # reallocation. Calling empty_cache() forces:
        #   1. Full GPU synchronization (blocking)
        #   2. Memory returned to OS, then re-allocated on next use
        #   3. Potential memory fragmentation
        # The caching allocator handles memory efficiently on its own.
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
            for slot_id, slot in self.model.seed_slots.items():  # type: ignore[union-attr, operator]
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
        # PERF: For CUDA tensors, .cpu() already allocates a fresh CPU tensor, so
        # .clone() is redundant. Only clone() for CPU tensors to ensure independence.
        with torch.no_grad():
            self.last_good_state = {
                k: (v.detach().clone() if v.device.type == "cpu" else v.detach().cpu())
                   if isinstance(v, torch.Tensor)
                   else copy.deepcopy(v)
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
        # NOTE: We do NOT mutate consecutive_panics here. The counter should reflect
        # actual consecutive anomalies, not be used as a control knob. The panic_reason
        # ("governor_nan") already distinguishes this path for telemetry analysis.
        if math.isnan(current_loss) or math.isinf(current_loss):
            self._pending_panic = False
            self._panic_loss = current_loss
            self._panic_reason = "governor_nan"
            return True

        # Lobotomy detection: loss jumped to exactly random guessing
        # This catches "silent failures" where model outputs uniform probabilities
        if len(self.loss_history) >= MIN_GOVERNOR_HISTORY_SAMPLES:
            avg = sum(self.loss_history) / len(self.loss_history)
            # Relative tolerance: ~6.5% of random guess loss
            # - CIFAR-10 (ln(10)=2.3): tolerance = 0.15
            # - TinyStories (ln(50257)=10.8): tolerance = 0.70
            lobotomy_tolerance = 0.065 * self.random_guess_loss
            # If we were doing well (loss < 60% of random guess) and suddenly
            # hit exactly the random guess loss (±tolerance), that's a lobotomy
            # NOTE: We do NOT mutate consecutive_panics here (see NaN path comment).
            if (avg < self.random_guess_loss * 0.6 and
                abs(current_loss - self.random_guess_loss) < lobotomy_tolerance):
                self._pending_panic = False
                self._panic_loss = current_loss
                self._panic_reason = "governor_lobotomy"
                return True

        # Need sufficient history for stable estimates
        if len(self.loss_history) < MIN_GOVERNOR_HISTORY_SAMPLES:
            self.loss_history.append(current_loss)
            self._panic_reason = None
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
            self._panic_reason = "governor_divergence"
            # Intentionally NOT adding anomaly loss to history. This keeps
            # statistical thresholds (mean + k*std) based on "healthy" samples,
            # ensuring robust anomaly detection. Contaminated statistics would
            # inflate variance, triggering fewer alarms.
            if self.consecutive_panics >= self.min_panics_before_rollback:
                return True
            return False
        else:
            self.loss_history.append(current_loss)
            self.consecutive_panics = 0
            self._pending_panic = False
            self._panic_loss = None  # Clear stale panic loss on recovery
            self._panic_reason = None
            return False

    def execute_rollback(
        self,
        *,
        env_id: int = 0,
    ) -> GovernorReport:
        """Emergency stop: restore LKG state and return punishment info.

        Rollback semantics:
        - Restore host + fossilized seeds from snapshot
        - Prune any live/experimental seeds (sets them to PRUNED with embargo)
        - After embargo period expires, pruned slots become dormant again

        IMPORTANT: Caller must clear optimizer state after rollback.
        PyTorch's load_state_dict() copies weights IN-PLACE (same Parameter
        objects), so optimizer momentum/variance buffers SURVIVE rollback.
        Call optimizer.state.clear() to prevent re-divergence.

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

        # Prepare telemetry data (emitted after rollback completes)
        # Import hub at function scope to defer initialization
        from esper.nissa import get_hub
        hub = get_hub()
        # Cast is safe: _panic_reason is only set to valid GovernorPanicReason values
        panic_reason = cast(GovernorPanicReason, self._panic_reason or "governor_rollback")

        if self.last_good_state is None:
            raise RuntimeError("Governor panic before first snapshot!")

        # Clear any live (non-fossilized) seeds FIRST
        # This removes seed parameters so state_dict keys match snapshot
        # Implements "revert to stable organism, dump all temporary grafts"
        # hasattr AUTHORIZED by John on 2025-12-01 16:30:00 UTC
        # Justification: Feature detection - MorphogeneticModel has seed_slots, base models may not
        if hasattr(self.model, 'seed_slots'):
            for slot in self.model.seed_slots.values():  # type: ignore[operator]
                slot.prune(panic_reason, initiator="governor")

        # Restore host + fossilized seeds from snapshot
        # Use strict=False because:
        # 1. Snapshot excludes experimental seeds (by design)
        # 2. Current model may have different seed state than snapshot
        # 3. execute_rollback prunes all seeds before restore, so missing keys are expected
        #
        # MEMORY OPTIMIZATION: Load CPU snapshot directly instead of pre-copying to GPU.
        # PyTorch's load_state_dict handles CPU->GPU transfer per-parameter, avoiding
        # a full GPU duplicate that would double memory usage during rollback.
        # This is critical because rollback often occurs when memory is already tight.
        #
        # CRITICAL: Synchronize ALL CUDA streams before load_state_dict
        # Using device-level sync (not just current_stream) for safety - ensures
        # no other operations are modifying model parameters during rollback.
        # Cost is acceptable since rollback is rare. See B1-PT-02 for analysis.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            self.last_good_state, strict=False
        )

        # Emit single rollback event with all context (B1-CR-02: no duplicate emissions)
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            severity="critical",
            message="Critical instability detected - rollback complete",
            data=GovernorRollbackPayload(
                env_id=env_id,
                device=str(device),
                reason="Structural Collapse",
                loss_at_panic=self._panic_loss,
                loss_threshold=threshold,
                consecutive_panics=self.consecutive_panics,
                panic_reason=panic_reason,
                # Include key mismatch info if present (diagnostic context)
                missing_keys=list(missing_keys) if missing_keys else None,
                unexpected_keys=list(unexpected_keys) if unexpected_keys else None,
            ),
        ))

        # IMPORTANT: Optimizer state must be cleared by the CALLER after rollback.
        #
        # PyTorch load_state_dict() behavior (verified PyTorch 2.9):
        # - Default (assign=False): Copies weights IN-PLACE via param.copy_()
        # - Parameter objects RETAIN their identity (same id())
        # - Optimizer state is keyed by Parameter objects
        # - Therefore: momentum/variance buffers SURVIVE rollback unchanged
        #
        # If optimizer.state is not cleared, SGD/Adam momentum continues pushing
        # toward the diverged state that caused the panic, risking re-divergence.
        #
        # The caller (vectorized.py) must call optimizer.state.clear() after
        # execute_rollback() returns. We don't do it here because:
        # 1. Governor doesn't own the optimizer (separation of concerns)
        # 2. Multiple optimizers may exist (host + per-seed optimizers)
        #
        # TRAP: Using load_state_dict(assign=True) would replace Parameter objects,
        # but then optimizer.param_groups references orphaned tensors and
        # optimizer.step() updates garbage. Would require optimizer recreation.
        #
        # B1-PT-01 CORRECTION: The original B1-PT-01 ticket incorrectly claimed
        # load_state_dict() creates new Parameter tensors. This was wrong - it
        # copies in-place. The "fix" that removed momentum zeroing actually
        # introduced the bug. Verified via id() checks on Parameters before/after.

        # Reset ALL panic state after successful rollback to prevent stale context
        # from leaking into future reports. Store loss locally first for the report.
        loss_at_panic = self._panic_loss
        self.consecutive_panics = 0
        self._pending_panic = False
        self._panic_loss = None
        self._panic_reason = None

        return GovernorReport(
            reason="Structural Collapse",
            loss_at_panic=loss_at_panic if loss_at_panic is not None else float('nan'),
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
        self._panic_reason = None
        self.snapshot()


__all__ = ["TolariaGovernor", "GovernorReport"]
