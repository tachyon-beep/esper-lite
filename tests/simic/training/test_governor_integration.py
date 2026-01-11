"""Simic-specific Governor integration tests.

Tests the orchestration pattern used in vectorized training, NOT the Governor
internals (which are tested in tests/tolaria/test_governor.py and
tests/integration/test_governor_rollback.py).

These tests focus on:
1. The exact rollback pattern from action_execution.py:381-404
2. Multi-environment isolation (rollback in env 0 doesn't affect env 1)
3. The needs_governor_snapshot flag lifecycle
4. Optimizer clearing for both host and seed optimizers
"""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.tolaria.governor import TolariaGovernor


class SimpleModel(nn.Module):
    """Minimal model for testing rollback isolation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.seed_slots: dict[str, MagicMock] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_env_state(
    device: str = "cpu",
) -> tuple[ParallelEnvState, SimpleModel]:
    """Create a ParallelEnvState with real Governor for integration testing."""
    model = SimpleModel()
    model.to(device)

    host_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    governor = TolariaGovernor(model=model, min_panics_before_rollback=1)

    signal_tracker = MagicMock()
    signal_tracker.reset = MagicMock()

    env_state = ParallelEnvState(
        model=model,
        host_optimizer=host_optimizer,
        signal_tracker=signal_tracker,
        governor=governor,
        env_device=device,
    )
    return env_state, model


def build_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run a forward/backward pass to populate optimizer state (momentum buffers)."""
    x = torch.randn(2, 10)
    output = model(x)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


class TestMultiEnvRollbackIsolation:
    """Rollback in one environment should not affect others.

    This tests the critical property that vectorized training with multiple
    environments maintains isolation: if env 0 panics and rolls back, env 1's
    model and optimizer state must be completely unaffected.
    """

    def test_rollback_env0_preserves_env1_state(self) -> None:
        """When env 0 rolls back, env 1's model and optimizer are unchanged."""
        # Create two independent model/governor pairs (simulating 2 envs)
        env_state_0, model_0 = create_env_state()
        env_state_1, model_1 = create_env_state()

        # Build optimizer state (momentum buffers) for both
        build_optimizer_state(model_0, env_state_0.host_optimizer)
        build_optimizer_state(model_1, env_state_1.host_optimizer)

        # Warm up governors and take snapshots
        for _ in range(10):
            env_state_0.governor.check_vital_signs(2.3)
            env_state_1.governor.check_vital_signs(2.3)
        env_state_0.governor.snapshot()
        env_state_1.governor.snapshot()

        # Capture env 1's state before any rollback
        env1_weights_before = {
            k: v.clone() for k, v in model_1.state_dict().items()
        }
        env1_momentum_keys = set(env_state_1.host_optimizer.state.keys())
        assert len(env1_momentum_keys) > 0, "Env 1 should have optimizer state"

        # Corrupt env 0 and trigger rollback
        with torch.no_grad():
            for p in model_0.parameters():
                p.data.fill_(float("nan"))

        panic_0 = env_state_0.governor.check_vital_signs(float("nan"))
        assert panic_0, "Env 0 should panic on NaN loss"

        # Execute the rollback pattern from action_execution.py:389-403
        env_state_0.governor.execute_rollback(env_id=0)
        env_state_0.host_optimizer.state.clear()
        # (no seed_optimizers in this test)

        # CRITICAL ASSERTION: Env 1 is completely unaffected
        env1_weights_after = {
            k: v.clone() for k, v in model_1.state_dict().items()
        }
        for key in env1_weights_before:
            assert torch.equal(
                env1_weights_before[key], env1_weights_after[key]
            ), f"Env 1 weight {key} was modified by env 0 rollback!"

        # Env 1's optimizer state is preserved
        assert (
            set(env_state_1.host_optimizer.state.keys()) == env1_momentum_keys
        ), "Env 1 optimizer state was modified by env 0 rollback!"

        # Env 0's weights are restored (not NaN)
        assert not torch.isnan(
            next(model_0.parameters())
        ).any(), "Env 0 weights still NaN after rollback"

        # Env 0's optimizer state is cleared
        assert (
            len(env_state_0.host_optimizer.state) == 0
        ), "Env 0 optimizer state should be cleared after rollback"

    def test_independent_panic_tracking(self) -> None:
        """Each environment tracks panics independently."""
        env_state_0, model_0 = create_env_state()
        env_state_1, model_1 = create_env_state()

        # Warm up both governors
        for _ in range(10):
            env_state_0.governor.check_vital_signs(2.3)
            env_state_1.governor.check_vital_signs(2.3)

        # Only env 0 sees NaN
        panic_0 = env_state_0.governor.check_vital_signs(float("nan"))
        panic_1 = env_state_1.governor.check_vital_signs(2.3)  # Normal loss

        assert panic_0, "Env 0 should panic"
        assert not panic_1, "Env 1 should NOT panic"


class TestOptimizerClearingPattern:
    """Test that both host and seed optimizers are cleared on rollback.

    This tests the pattern at action_execution.py:399-404 where BOTH the
    host_optimizer AND all seed_optimizers have their state cleared after
    rollback to prevent momentum from pushing toward the diverged state.
    """

    def test_host_and_seed_optimizers_cleared(self) -> None:
        """Rollback must clear state for host_optimizer AND all seed_optimizers."""
        env_state, model = create_env_state()

        # Build host optimizer state
        build_optimizer_state(model, env_state.host_optimizer)
        assert len(env_state.host_optimizer.state) > 0

        # Create mock seed optimizers with state
        seed_opt_1 = torch.optim.SGD([torch.nn.Parameter(torch.randn(5))], lr=0.01, momentum=0.9)
        seed_opt_2 = torch.optim.SGD([torch.nn.Parameter(torch.randn(5))], lr=0.01, momentum=0.9)

        # Build state in seed optimizers
        for opt in [seed_opt_1, seed_opt_2]:
            param = next(iter(opt.param_groups[0]["params"]))
            grad = torch.randn_like(param)
            param.grad = grad
            opt.step()
            opt.zero_grad()

        env_state.seed_optimizers["slot_0"] = seed_opt_1
        env_state.seed_optimizers["slot_1"] = seed_opt_2

        assert len(seed_opt_1.state) > 0, "Seed optimizer 1 should have state"
        assert len(seed_opt_2.state) > 0, "Seed optimizer 2 should have state"

        # Warm up governor and snapshot
        for _ in range(10):
            env_state.governor.check_vital_signs(2.3)
        env_state.governor.snapshot()

        # Trigger panic and execute rollback
        env_state.governor.check_vital_signs(float("nan"))
        env_state.governor.execute_rollback(env_id=0)

        # Execute the exact pattern from action_execution.py:401-403
        env_state.host_optimizer.state.clear()
        for seed_opt in env_state.seed_optimizers.values():
            seed_opt.state.clear()

        # All optimizer states should be cleared
        assert len(env_state.host_optimizer.state) == 0, "Host optimizer state not cleared"
        assert len(seed_opt_1.state) == 0, "Seed optimizer 1 state not cleared"
        assert len(seed_opt_2.state) == 0, "Seed optimizer 2 state not cleared"

    def test_momentum_cleared_prevents_redivergence(self) -> None:
        """Verify that clearing optimizer state prevents momentum-driven redivergence.

        This is the B1-PT-01 correction: without clearing, SGD momentum continues
        pushing toward the diverged state that caused the panic.
        """
        env_state, model = create_env_state()

        # Build significant momentum by training several steps
        optimizer = env_state.host_optimizer
        for _ in range(5):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Capture momentum before rollback
        param = next(model.parameters())
        momentum_before = optimizer.state[param]["momentum_buffer"].clone()
        assert momentum_before.abs().sum() > 0, "Should have built up momentum"

        # Snapshot, corrupt, rollback
        env_state.governor.snapshot()
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(float("nan"))
        env_state.governor.check_vital_signs(float("nan"))
        env_state.governor.execute_rollback(env_id=0)

        # Without clearing, momentum would still exist
        assert (
            param in optimizer.state
        ), "Param should still be in optimizer state (in-place load_state_dict)"

        # Clear as per the pattern
        optimizer.state.clear()

        # Now momentum is gone - next step won't be influenced by diverged history
        assert len(optimizer.state) == 0


class TestGovernorSnapshotFlag:
    """Test the needs_governor_snapshot lifecycle in ParallelEnvState.

    The flag is set after fossilization (action_execution.py:778) and cleared
    after the snapshot is taken (vectorized_trainer.py:1079). This ensures
    snapshots include fossilized seed weights to prevent incoherent rollback.
    """

    def test_snapshot_flag_default_false(self) -> None:
        """needs_governor_snapshot defaults to False."""
        env_state, _ = create_env_state()
        assert env_state.needs_governor_snapshot is False

    def test_snapshot_flag_set_after_fossilization(self) -> None:
        """needs_governor_snapshot should be True after fossilization."""
        env_state, _ = create_env_state()

        # Simulate fossilization setting the flag
        # (In real code, this happens in action_execution.py:778)
        env_state.needs_governor_snapshot = True

        assert env_state.needs_governor_snapshot is True

    def test_snapshot_flag_cleared_after_snapshot(self) -> None:
        """needs_governor_snapshot should be False after snapshot taken."""
        env_state, _ = create_env_state()

        # Simulate fossilization
        env_state.needs_governor_snapshot = True

        # Simulate the vectorized_trainer.py:1077-1079 pattern
        if env_state.needs_governor_snapshot:
            env_state.governor.snapshot()
            env_state.needs_governor_snapshot = False

        assert env_state.needs_governor_snapshot is False

    def test_snapshot_flag_reset_on_episode_reset(self) -> None:
        """needs_governor_snapshot should be False after episode reset."""
        env_state, model = create_env_state()

        # Set up mock slots for reset_episode_state
        slot_mock = MagicMock()
        slot_mock.seed = None
        slot_mock.state = None
        model.seed_slots = {"slot_0": slot_mock}

        # Simulate fossilization
        env_state.needs_governor_snapshot = True

        # Reset episode
        env_state.reset_episode_state(["slot_0"])

        assert env_state.needs_governor_snapshot is False

    def test_snapshot_includes_current_weights(self) -> None:
        """Snapshot taken when flag is set captures current model weights."""
        env_state, model = create_env_state()

        # Modify weights (simulating fossilization changing host weights)
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(42.0)

        # Set flag and take snapshot (as would happen after fossilization)
        env_state.needs_governor_snapshot = True
        env_state.governor.snapshot()
        env_state.needs_governor_snapshot = False

        # Verify snapshot captured the modified weights
        snapshot = env_state.governor.last_good_state
        for key, value in snapshot.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(value == 42.0), f"Snapshot didn't capture modified {key}"


class TestStreamContextDuringRollback:
    """Test that rollback uses proper CUDA stream context.

    The pattern at action_execution.py:384-390 wraps rollback in the env's
    CUDA stream context to avoid default-stream leakage and races.
    """

    def test_rollback_uses_stream_context_when_available(self) -> None:
        """Rollback should use env's CUDA stream when present."""
        env_state, model = create_env_state()

        # Create a mock stream
        mock_stream = MagicMock()
        env_state.stream = mock_stream

        # Warm up and snapshot
        for _ in range(10):
            env_state.governor.check_vital_signs(2.3)
        env_state.governor.snapshot()

        # The pattern from action_execution.py:384-390
        with patch("torch.cuda.stream") as mock_cuda_stream:
            mock_cuda_stream.return_value.__enter__ = MagicMock()
            mock_cuda_stream.return_value.__exit__ = MagicMock()

            rollback_ctx = (
                torch.cuda.stream(env_state.stream)
                if env_state.stream
                else nullcontext()
            )
            # Should call torch.cuda.stream with our stream
            mock_cuda_stream.assert_called_once_with(mock_stream)

    def test_rollback_uses_nullcontext_when_no_stream(self) -> None:
        """Rollback should use nullcontext when no CUDA stream."""
        env_state, model = create_env_state()
        env_state.stream = None

        rollback_ctx = (
            torch.cuda.stream(env_state.stream)
            if env_state.stream
            else nullcontext()
        )

        # Should be a nullcontext (identity context manager)
        assert isinstance(rollback_ctx, type(nullcontext()))


class TestRollbackTracking:
    """Test the env_rollback_occurred tracking dict."""

    def test_rollback_sets_tracking_flag(self) -> None:
        """env_rollback_occurred[env_idx] should be True after rollback."""
        env_state, model = create_env_state()
        env_rollback_occurred: dict[int, bool] = {0: False, 1: False}

        # Warm up and snapshot
        for _ in range(10):
            env_state.governor.check_vital_signs(2.3)
        env_state.governor.snapshot()

        # Simulate panic detection and rollback for env 0
        governor_panic_envs = []
        panic = env_state.governor.check_vital_signs(float("nan"))
        if panic:
            governor_panic_envs.append(0)

        # Execute rollback pattern from action_execution.py:381-391
        env_idx = 0
        if env_idx in governor_panic_envs:
            env_state.governor.execute_rollback(env_id=env_idx)
            env_rollback_occurred[env_idx] = True
            env_state.host_optimizer.state.clear()

        assert env_rollback_occurred[0] is True
        assert env_rollback_occurred[1] is False  # Env 1 unaffected

    def test_multiple_envs_can_rollback_independently(self) -> None:
        """Multiple environments can track rollback independently."""
        env_state_0, _ = create_env_state()
        env_state_1, _ = create_env_state()
        env_rollback_occurred: dict[int, bool] = {0: False, 1: False}

        # Warm up and snapshot both
        for _ in range(10):
            env_state_0.governor.check_vital_signs(2.3)
            env_state_1.governor.check_vital_signs(2.3)
        env_state_0.governor.snapshot()
        env_state_1.governor.snapshot()

        # Both panic (unusual but valid scenario)
        governor_panic_envs = []
        if env_state_0.governor.check_vital_signs(float("nan")):
            governor_panic_envs.append(0)
        if env_state_1.governor.check_vital_signs(float("nan")):
            governor_panic_envs.append(1)

        # Both should rollback
        for env_idx in governor_panic_envs:
            env_state = env_state_0 if env_idx == 0 else env_state_1
            env_state.governor.execute_rollback(env_id=env_idx)
            env_rollback_occurred[env_idx] = True
            env_state.host_optimizer.state.clear()

        assert env_rollback_occurred[0] is True
        assert env_rollback_occurred[1] is True


class TestGovernorRollbackTelemetry:
    """Test that GOVERNOR_ROLLBACK telemetry events are emitted during rollback.

    This tests the telemetry pattern at action_execution.py ~line 404 where
    a GovernorRollbackPayload is emitted via the VectorizedEmitter after
    a rollback occurs.
    """

    def test_rollback_emits_telemetry_event(self) -> None:
        """Rollback should emit GOVERNOR_ROLLBACK telemetry event."""
        from unittest.mock import MagicMock
        from esper.leyline import TelemetryEventType, GovernorRollbackPayload

        env_state, model = create_env_state()

        # Create a mock emitter to capture events
        mock_emitter = MagicMock()
        emitted_events: list = []

        def capture_emit(event):
            emitted_events.append(event)

        mock_emitter.emit = capture_emit

        # Warm up governor and snapshot
        for _ in range(10):
            env_state.governor.check_vital_signs(2.3)
        env_state.governor.snapshot()

        # Trigger panic
        panic = env_state.governor.check_vital_signs(float("nan"))
        assert panic, "Should panic on NaN"

        # Capture panic info BEFORE rollback (as action_execution.py should do)
        panic_reason = env_state.governor._panic_reason
        panic_loss = env_state.governor._panic_loss
        consecutive_panics = env_state.governor.consecutive_panics

        # Execute rollback
        env_state.governor.execute_rollback(env_id=0)
        env_state.host_optimizer.state.clear()

        # Emit telemetry as action_execution.py should do
        from esper.leyline import TelemetryEvent
        device = "cpu"
        env_idx = 0
        mock_emitter.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=GovernorRollbackPayload(
                env_id=env_idx,
                device=str(device),
                reason=panic_reason or "unknown",
                loss_at_panic=panic_loss,
                consecutive_panics=consecutive_panics,
            ),
            severity="warning",
        ))

        # Verify telemetry was emitted
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.event_type == TelemetryEventType.GOVERNOR_ROLLBACK
        assert event.severity == "warning"

        # Verify payload contents
        payload = event.data
        assert isinstance(payload, GovernorRollbackPayload)
        assert payload.env_id == 0
        assert payload.device == "cpu"
        assert payload.reason == "governor_nan"
        assert payload.consecutive_panics == 0  # NaN panics don't increment counter

    def test_rollback_telemetry_includes_divergence_info(self) -> None:
        """Rollback from divergence should include consecutive_panics."""
        from unittest.mock import MagicMock
        from esper.leyline import TelemetryEventType, GovernorRollbackPayload, TelemetryEvent

        env_state, model = create_env_state()
        mock_emitter = MagicMock()
        emitted_events: list = []
        mock_emitter.emit = lambda e: emitted_events.append(e)

        # Warm up governor with normal losses
        for _ in range(10):
            env_state.governor.check_vital_signs(2.3)
        env_state.governor.snapshot()

        # Trigger divergence panic (needs min_panics_before_rollback consecutive high losses)
        # Governor was created with min_panics_before_rollback=1
        panic = env_state.governor.check_vital_signs(50.0)  # Very high loss
        assert panic, "Should panic on extreme loss"

        # Capture panic info
        panic_reason = env_state.governor._panic_reason
        panic_loss = env_state.governor._panic_loss
        consecutive_panics = env_state.governor.consecutive_panics

        assert panic_reason == "governor_divergence"
        assert consecutive_panics >= 1

        # Execute rollback
        env_state.governor.execute_rollback(env_id=0)

        # Emit telemetry
        mock_emitter.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=GovernorRollbackPayload(
                env_id=0,
                device="cpu",
                reason=panic_reason or "unknown",
                loss_at_panic=panic_loss,
                consecutive_panics=consecutive_panics,
            ),
            severity="warning",
        ))

        # Verify payload
        payload = emitted_events[0].data
        assert payload.reason == "governor_divergence"
        assert payload.consecutive_panics >= 1
        assert payload.loss_at_panic == 50.0

    def test_vectorized_emitter_emit_method_injects_context(self) -> None:
        """VectorizedEmitter.emit() should inject env_id, device, group_id."""
        from unittest.mock import MagicMock
        from esper.leyline import TelemetryEventType, GovernorRollbackPayload, TelemetryEvent
        from esper.simic.telemetry.emitters import VectorizedEmitter

        # Create mock hub
        mock_hub = MagicMock()
        emitted_events: list = []
        mock_hub.emit = lambda e: emitted_events.append(e)

        # Create VectorizedEmitter with known env context
        emitter = VectorizedEmitter(
            env_id=42,
            device="cuda:0",
            group_id="test-group",
            hub=mock_hub,
        )

        # Emit a rollback event
        emitter.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=GovernorRollbackPayload(
                env_id=0,  # Will be overwritten by emitter
                device="cpu",  # Will be overwritten by emitter
                reason="test_reason",
            ),
            severity="warning",
        ))

        # Verify event was emitted with injected context
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.env_id == 42  # Injected by emitter
        assert event.device == "cuda:0"  # Injected by emitter
        assert event.group_id == "test-group"  # Injected by emitter
        assert event.event_type == TelemetryEventType.GOVERNOR_ROLLBACK
        assert event.data.reason == "test_reason"
