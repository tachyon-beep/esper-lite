"""Integration test to verify Gradient Isolation Fidelity.

This test ensures that the "Incubator" mechanism works as intended:
1. Seeds in TRAINING stage do not affect Host gradients (Isolation).
2. Seeds in TRAINING stage DO receive gradients (Learning).
3. Seeds in BLENDING stage DO affect Host gradients (Integration).
"""

import pytest
import torch
import torch.nn as nn
from typing import cast

from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.tolaria.environment import create_model


# =============================================================================
# Helper: Gradient Capture
# =============================================================================

def get_param_grad(module: nn.Module, param_name: str) -> torch.Tensor | None:
    """Helper to fetch gradient clone for a specific named parameter."""
    for name, param in module.named_parameters():
        if name == param_name:
            if param.grad is None:
                return None
            return param.grad.clone()
    return None

def get_all_host_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture dict of all host parameter gradients."""
    grads = {}
    # Iterate over host only (excluding slots)
    for name, param in model.host.named_parameters():
        if "slots" in name:
            continue
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads

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
class TestGradientIsolationFidelity:
    
    def test_isolation_in_training_stage(self):
        """Verify strict isolation for seeds in TRAINING stage."""
        device = "cpu"
        torch.manual_seed(42)
        
        # 1. Setup Data
        inputs = torch.randn(4, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (4,), device=device)
        criterion = nn.CrossEntropyLoss()
        
        # 2. Baseline Run (Clean Host)
        # We create a fresh model to ensure no state contamination
        baseline_model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        
        # Run forward/backward
        baseline_out = baseline_model(inputs)
        baseline_loss = criterion(baseline_out, targets)
        baseline_loss.backward()
        
        baseline_grads = get_all_host_grads(baseline_model)
        
        # 3. Experiment Run (Host + Incubating Seed)
        exp_model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        # Copy weights to ensure identical start point
        exp_model.load_state_dict(baseline_model.state_dict())
        
        # Germinate seed in TRAINING stage
        slot = cast(SeedSlot, exp_model.seed_slots["r0c1"])
        slot.germinate("conv_small")
        perturb_seed_weights(slot.seed)  # Ensure non-zero gradients
        slot.state.stage = SeedStage.TRAINING
        slot.state.alpha = 0.0
        # Enforce isolation (this is the flag we are testing)
        slot.isolate_gradients = True
        
        # Clear grads before run
        exp_model.zero_grad()
        
        # Run forward/backward
        exp_out = exp_model(inputs)
        exp_loss = criterion(exp_out, targets)
        exp_loss.backward()
        
        exp_grads = get_all_host_grads(exp_model)
        
        # 4. Verify Forward Identity
        # Output should be bitwise identical because alpha=0 and STE masks output
        assert torch.equal(baseline_out, exp_out), \
            "Forward pass output changed despite TRAINING stage (STE failure)"
            
        # 5. Verify Host Gradient Identity (Metric 1)
        for name, base_grad in baseline_grads.items():
            if name not in exp_grads:
                pytest.fail(f"Gradient missing in experiment for {name}")
            
            exp_grad = exp_grads[name]
            
            # Strict equality expected
            if not torch.equal(base_grad, exp_grad):
                diff = (base_grad - exp_grad).abs().max().item()
                pytest.fail(
                    f"Host gradient leakage detected in {name}. "
                    f"Max diff: {diff:.2e}. "
                    "Input isolation (detach) failed."
                )
                
        # 6. Verify Seed Learning (Metric 2)
        # Seed should have received gradients despite isolation
        seed_has_grad = False
        for param in slot.seed.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                seed_has_grad = True
                break
        
        assert seed_has_grad, \
            "Seed received no gradients! It is detached from the loss graph completely."

    def test_integration_in_blending_stage(self):
        """Verify co-adaptation (NO isolation) for seeds in BLENDING stage."""
        device = "cpu"
        torch.manual_seed(42)
        
        inputs = torch.randn(4, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (4,), device=device)
        criterion = nn.CrossEntropyLoss()
        
        # Baseline
        baseline_model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        baseline_loss = criterion(baseline_model(inputs), targets)
        baseline_loss.backward()
        baseline_grads = get_all_host_grads(baseline_model)
        
        # Experiment (BLENDING)
        exp_model = create_model(task="cifar_baseline", device=device, slots=["r0c1"])
        exp_model.load_state_dict(baseline_model.state_dict())
        
        slot = cast(SeedSlot, exp_model.seed_slots["r0c1"])
        slot.germinate("conv_small")
        perturb_seed_weights(slot.seed)  # Ensure divergence
        slot.state.stage = SeedStage.BLENDING
        slot.state.alpha = 0.5 # Partial blend
        slot.state.alpha_controller.alpha = 0.5
        # Disable isolation (host should see seed gradients)
        slot.isolate_gradients = False
        
        exp_model.zero_grad()
        exp_loss = criterion(exp_model(inputs), targets)
        exp_loss.backward()
        
        exp_grads = get_all_host_grads(exp_model)
        
        # 4. Verify Divergence
        # Gradients SHOULD differ because host is now "hearing" the seed
        divergence_detected = False
        for name, base_grad in baseline_grads.items():
            exp_grad = exp_grads.get(name)
            if exp_grad is not None and not torch.allclose(base_grad, exp_grad):
                divergence_detected = True
                break
                
        assert divergence_detected, \
            "Host gradients identical to baseline despite BLENDING stage! Seed is ignored."
