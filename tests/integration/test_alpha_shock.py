"""Integration test to verify Alpha Shock Transient.

Quantifies the destabilizing jolt when a seed transitions from TRAINING (Alpha=0)
to BLENDING (Alpha>0). Ensuring this shock is minimal is critical for the RL agent
to accept new seeds.
"""

import pytest
import torch
import torch.nn as nn
from typing import cast

from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.tolaria.environment import create_model

# =============================================================================
# Helper Functions
# =============================================================================

def train_seed_one_step(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, slot_id: str):
    """Train the seed for one step in ISOLATION mode."""
    slot = cast(SeedSlot, model.seed_slots[slot_id])
    
    # CRITICAL FIX: Optimize SEED only. 
    # Host must remain chemically invariant during incubation.
    optimizer = torch.optim.SGD(slot.seed.parameters(), lr=0.1)
    optimizer.zero_grad()
    
    # Force Host to be immutable for this operation
    # (Just in case gradients leak via the connection point)
    # Note: We need gradients to flow *through* the host to the seed input,
    # but we don't want to update host weights.
    # We can't set requires_grad=False on host, because that might stop gradient flow if host is used.
    # But ste_forward uses host_features (detached or not).
    # If we use isolate_gradients=True, seed input is detached.
    # So gradient flow stops at seed input. Host weights don't get grad from seed path.
    # But host weights might get grad from host path?
    # ste_forward: host + (seed - detached).
    # Backward: dL/dHost = 1. dL/dSeed = 1.
    # If we run backward(), host weights WILL get gradients from the host path!
    # We must ensuring optimizer only updates seed.
    # And ideally we don't even compute host grads to save time, but standard backward() computes all.
    # Since we construct optimizer with ONLY seed params, host weights won't update.
    
    # Forward pass
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    
    # DEBUG: Check gradients
    total_grad = 0.0
    for p in slot.seed.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    print(f"Step grad norm: {total_grad}")
    
    optimizer.step()
    # Zero all grads to be clean for next step
    model.zero_grad()
        
    return loss.item()

def measure_shock(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, slot_id: str, alpha_step: float = 0.01) -> float:
    """Measure the loss delta when alpha moves from 0 to alpha_step."""
    slot = cast(SeedSlot, model.seed_slots[slot_id])
    
    # Baseline (Alpha=0)
    slot.state.alpha = 0.0
    slot.state.alpha_controller.alpha = 0.0
    slot._cached_alpha_tensor = None  # <--- CRITICAL FIX: Invalidate cache
    
    with torch.no_grad():
        loss_0 = criterion(model(inputs), targets).item()
        
    # Shock (Alpha=step)
    slot.state.alpha = alpha_step
    slot.state.alpha_controller.alpha = alpha_step
    slot._cached_alpha_tensor = None  # <--- CRITICAL FIX: Invalidate cache
    
    # IMPORTANT: When alpha > 0, we are in BLENDING.
    # We must ensure the forward pass logic respects this.
    # SeedSlot.forward uses alpha from state/controller.
    
    with torch.no_grad():
        loss_step = criterion(model(inputs), targets).item()
        
    return loss_step - loss_0

def perturb_seed_weights(seed: nn.Module):
    """Perturb seed weights to ensure non-identity behavior."""
    with torch.no_grad():
        for param in seed.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                param.add_(0.01)

# =============================================================================
# Verification Test
# =============================================================================

@pytest.mark.integration
class TestAlphaShock:
    
    def test_random_seed_shock_is_bounded(self):
        """Verify that even a random seed doesn't cause catastrophic loss explosion."""
        device = "cpu"
        torch.manual_seed(42)
        
        inputs = torch.randn(16, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (16,), device=device)
        criterion = nn.CrossEntropyLoss()
        
        model = create_model(task="cifar10", device=device, slots=["r0c1"])
        slot = cast(SeedSlot, model.seed_slots["r0c1"])
        slot.germinate("conv_small")
        perturb_seed_weights(slot.seed) # Make it random/noisy
        slot.state.stage = SeedStage.BLENDING # Simulate transition
        
        shock = measure_shock(model, inputs, targets, criterion, "r0c1", alpha_step=0.05)
        
        print(f"\nRandom Seed Shock (Alpha=0.05): {shock:.4f}")
        
        # Criterion: Loss shouldn't increase by more than 2.0 (arbitrary safety bound for stability)
        # If it increases by 10.0, the gradients will explode and Governor will panic.
        assert shock < 2.0, f"Catastrophic alpha shock detected: {shock}"

    def test_trained_seed_reduces_shock(self):
        """Verify that training the seed in the incubator REDUCES the alpha shock."""
        device = "cpu"
        torch.manual_seed(42)
        
        inputs = torch.randn(16, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (16,), device=device)
        criterion = nn.CrossEntropyLoss()
        
        # --- Control: Random Seed ---
        model_control = create_model(task="cifar10", device=device, slots=["r0c1"])
        model_control.load_state_dict(model_control.state_dict()) # Clone start state not really needed for fresh
        slot_c = cast(SeedSlot, model_control.seed_slots["r0c1"])
        slot_c.germinate("conv_small")
        perturb_seed_weights(slot_c.seed)
        slot_c.state.stage = SeedStage.BLENDING
        
        shock_random = measure_shock(model_control, inputs, targets, criterion, "r0c1")
        
        # --- Experiment: Incubated Seed ---
        model_exp = create_model(task="cifar10", device=device, slots=["r0c1"])
        slot_e = cast(SeedSlot, model_exp.seed_slots["r0c1"])
        slot_e.germinate("conv_small")
        
        # Important: Start from same host state (now that topology matches)
        model_exp.load_state_dict(model_control.state_dict())
        # Start with same random weights as control
        slot_e.seed.load_state_dict(slot_c.seed.state_dict())
        
        # Incubate! (Train with alpha=0)
        slot_e.state.stage = SeedStage.TRAINING
        slot_e.isolate_gradients = True
        
        # Train for a few steps to align seed with residual
        print("\nIncubating seed...")
        initial_loss = 0
        for i in range(50):
            loss = train_seed_one_step(model_exp, inputs, targets, criterion, "r0c1")
            if i == 0: initial_loss = loss
            
        # Switch to blending
        slot_e.state.stage = SeedStage.BLENDING
        shock_trained = measure_shock(model_exp, inputs, targets, criterion, "r0c1")
        
        print(f"Random Shock: {shock_random:.6f}")
        print(f"Trained Shock: {shock_trained:.6f}")
        
        # Verify Benefit
        # Ideally shock_trained should be negative (immediate help) or at least smaller than random
        assert shock_trained < shock_random, \
            "Incubation failed! Trained seed caused WORSE shock than random seed."
            
        # Optional: Stronger assertion - trained seed should be beneficial or neutral
        # assert shock_trained <= 0.1, "Trained seed still causes significant shock"
