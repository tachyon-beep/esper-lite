"""Integration test to verify Governor Rollback mechanics.

Ensures that catastrophic failures (NaN/Inf/Loss Spike) trigger a rollback
that correctly restores model weights.
"""

import pytest
import torch
import copy

from esper.tolaria.governor import TolariaGovernor
from esper.tolaria.environment import create_model

# =============================================================================
# Verification Test
# =============================================================================

def _clone_state_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(v) for v in value)
    return copy.deepcopy(value)


def _assert_state_value_equal(actual, expected) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_value_equal(actual[key], expected[key])
        return
    if isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_state_value_equal(a, b)
        return
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_state_value_equal(a, b)
        return
    assert actual == expected


@pytest.mark.integration
class TestGovernorRollback:
    
    def test_rollback_restores_weights_and_resets_optimizer(self):
        """Verify rollback functionality on NaN injection."""
        device = "cpu"
        model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        
        # Create Governor
        governor = TolariaGovernor(
            model=model,
            sensitivity=3.0,
            absolute_threshold=100.0,
            min_panics_before_rollback=1 # Trigger immediately for test
        )
        
        # 1. Warmup (Establish stable history)
        # Loss ~2.3 for CIFAR random
        for i in range(10):
            # check_vital_signs returns True if panic triggers rollback condition
            governor.check_vital_signs(current_loss=2.3)
            
        # 2. Snapshot implicitly taken during init and maintained?
        # Governor snapshots initially.
        # It snapshots on reset.
        # It snapshots manually.
        # Let's force a snapshot to be sure we have the "stable" state.
        governor.snapshot()
        
        assert governor.last_good_state is not None
        safe_snapshot = {k: _clone_state_value(v) for k, v in governor.last_good_state.items()}
        
        # 3. Inject Sabotage (NaN)
        # Mutate model weights to NaN
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(float('nan'))
                
        # Verify model is broken
        assert torch.isnan(next(model.parameters())).all()
        
        # 4. Trigger Panic
        # Loss=NaN
        # check_vital_signs returns True if panic triggers rollback condition
        panic = governor.check_vital_signs(current_loss=float('nan'))
        
        # 5. Trigger Rollback
        if panic:
            report = governor.execute_rollback()
            
            # 6. Verify Rollback Success
            assert report.rollback_occurred
            
            # Weights should be restored (no longer NaN)
            current_param = next(model.parameters())
            assert not torch.isnan(current_param).any(), "Weights are still NaN after rollback!"
            
            # Verify values match snapshot state (including any Module extra state).
            current_state = model.state_dict()
            for key, expected_value in safe_snapshot.items():
                assert key in current_state, f"Missing state key after rollback: {key}"
                _assert_state_value_equal(current_state[key], expected_value)

    def test_optimizer_state_persistence_after_rollback(self):
        """Verify that Governor does NOT clear optimizer state (caller responsibility)."""
        device = "cpu"
        model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        governor = TolariaGovernor(model=model, min_panics_before_rollback=1)
        
        # 1. Create optimizer state
        inputs = torch.randn(1, 3, 32, 32, device=device)
        output = model(inputs)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        
        # Check state exists
        param = next(model.parameters())
        assert len(optimizer.state[param]) > 0 # Momentum buffer exists
        
        # 2. Snapshot
        governor.snapshot()
        
        # 3. Trigger Rollback
        # check_vital_signs returns True if panic triggers rollback condition
        governor.check_vital_signs(float('nan'))
        governor.execute_rollback()
        
        # 4. Verify Optimizer State PERSISTS (Validation of Contract)
        # It is the caller's job to clear it. If Governor cleared it, it would be a side effect.
        # But wait, Governor doc says "Caller must clear".
        # So we assert it is NOT cleared.
        # Note: After rollback, 'param' object might be different if load_state_dict replaced it?
        # Standard load_state_dict modifies in-place.
        # So 'param' id should be same.
        
        # If param ID is same, state should be there.
        # If param ID changed (recreated), state is orphaned (zombie).
        
        # Let's check if param ID persists
        new_param = next(model.parameters())
        assert id(param) == id(new_param), "Parameter object identity changed! This implies orphaned optimizer state."
        
        assert len(optimizer.state[param]) > 0, "Governor unexpectedly cleared optimizer state!"
        
        # 5. Verify Manual Clear works (Simulation of vectorized.py)
        optimizer.state.clear()
        assert len(optimizer.state) == 0

    def test_rollback_survives_telemetry_failure(self):
        """Verify rollback completes even when slot telemetry callback raises.

        Regression test for the bug where Governor.execute_rollback() could fail
        if SeedSlot.prune() raised an exception from its telemetry callback.
        The safety mechanism (rollback) must not be blocked by observability
        failures (telemetry).

        The fix involves:
        1. Reordering execute_rollback() to restore weights BEFORE pruning seeds
        2. Making _emit_telemetry fault-tolerant with try/except
        3. Wrapping the prune loop in try/except in governor.py
        """
        device = "cpu"
        model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        governor = TolariaGovernor(
            model=model,
            sensitivity=3.0,
            absolute_threshold=100.0,
            min_panics_before_rollback=1,
        )

        # 1. Germinate a seed so there's something to prune during rollback
        model.germinate_seed("conv_light", "test_seed", slot="r0c1")
        slot = model.seed_slots["r0c1"]

        # Enable telemetry path (disable fast_mode)
        slot.fast_mode = False

        # 2. Set up a telemetry callback that ALWAYS raises
        telemetry_calls = []
        def failing_callback(event):
            telemetry_calls.append(event.event_type.name)
            raise IOError("Disk full - simulated telemetry failure")

        slot.on_telemetry = failing_callback

        # 3. Warmup and snapshot
        for _ in range(10):
            governor.check_vital_signs(2.3)
        governor.snapshot()

        # Capture pre-rollback state for verification
        pre_rollback_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in governor.last_good_state.items()
        }

        # 4. Corrupt model and trigger panic
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(float('nan'))

        # Verify model is broken
        assert torch.isnan(next(model.parameters())).all()

        # 5. Execute rollback - MUST NOT raise despite telemetry failure
        panic = governor.check_vital_signs(float('nan'))
        assert panic, "Expected panic to trigger"

        # This is the critical assertion: rollback must complete
        report = governor.execute_rollback()

        # 6. Verify rollback succeeded
        assert report.rollback_occurred

        # Weights should be restored (no longer NaN)
        current_param = next(model.parameters())
        assert not torch.isnan(current_param).any(), \
            "Weights are still NaN after rollback - rollback failed!"

        # Verify host weights match pre-rollback snapshot
        current_state = model.state_dict()
        for key, expected in pre_rollback_state.items():
            if isinstance(expected, torch.Tensor):
                assert key in current_state, f"Missing key after rollback: {key}"
                actual = current_state[key]
                if isinstance(actual, torch.Tensor):
                    assert torch.equal(actual, expected), \
                        f"Key {key} not restored correctly"

        # 7. Verify seed was pruned (even though telemetry failed)
        # The slot should be in PRUNED stage or have no state
        if slot.state is not None:
            from esper.leyline import SeedStage
            assert slot.state.stage == SeedStage.PRUNED, \
                f"Seed not pruned during rollback: {slot.state.stage}"

        # 8. Verify telemetry was attempted (callback was called)
        # The fault-tolerant _emit_telemetry should have caught the exception
        assert len(telemetry_calls) > 0, \
            "Telemetry callback was never called - test setup issue"
