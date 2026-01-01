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
