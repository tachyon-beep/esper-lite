"""Integration test to verify Fused Validation fidelity.

This test ensures that the "Fused Validation" mechanism (used for high-performance
counterfactual attribution) yields results identical to standard Sequential Validation.
It catches regressions in batch normalization handling, state leakage, and alpha
override logic.
"""

import pytest
import torch
import torch.nn as nn
from typing import cast

from esper.kasmina.slot import SeedSlot
from esper.leyline import AlphaAlgorithm, SeedStage
from esper.tolaria.environment import create_model


# =============================================================================
# Helper: Minimal Host for Testing
# =============================================================================

class SimpleHost(nn.Module):
    """Minimal host with BatchNorm to test stats freezing."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, 2)
        
        # Injection point manually exposed
        self.injection_point = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.injection_point(x)  # Seed slots attach here
        return self.classifier(x)


# =============================================================================
# Helper: Sequential Validation (Ground Truth)
# =============================================================================

def run_sequential_validation(
    model: nn.Module,
    inputs: torch.Tensor,
    configs: list[dict[str, float]],
    slots: list[str]
) -> list[torch.Tensor]:
    """Run validation sequentially for each config.
    
    Args:
        model: SlottedHost model
        inputs: Input batch
        configs: List of alpha configs (e.g. [{"r0c0": 0.0}, {"r0c0": 1.0}])
        slots: List of active slot IDs
        
    Returns:
        List of output tensors, one per config.
    """
    results = []
    
    # Ensure model is in eval mode (crucial for BN stats freezing)
    model.eval()
    
    for config in configs:
        # Apply alpha configuration manually
        for slot_id in slots:
            slot = cast(SeedSlot, model.seed_slots[slot_id])
            target_alpha = config.get(slot_id, slot.alpha)
            
            # Use the slot's state controller to set alpha
            # This mimics what happens during training steps
            if slot.state:
                slot.state.alpha = target_alpha
                slot.state.alpha_controller.alpha = target_alpha
                # Clear alpha schedule if forcing to 0/1 (except for GATED)
                if slot.state.alpha_algorithm != AlphaAlgorithm.GATE:
                    slot.alpha_schedule = None
                # Invalidate cache
                slot._cached_alpha_tensor = None

        with torch.no_grad():
            output = model(inputs)
            results.append(output)
            
    return results


# =============================================================================
# Helper: Fused Validation (Target)
# =============================================================================

def run_fused_validation(
    model: nn.Module,
    inputs: torch.Tensor,
    configs: list[dict[str, float]],
    slots: list[str]
) -> list[torch.Tensor]:
    """Run validation using the Fused approach (batch tiling).
    
    Args:
        model: SlottedHost model
        inputs: Input batch
        configs: List of alpha configs
        slots: List of active slot IDs
        
    Returns:
        List of output tensors (split from the fused batch).
    """
    model.eval()
    batch_size = inputs.size(0)
    num_configs = len(configs)
    
    # 1. Tile Inputs
    # Shape: [K * B, ...]
    inputs_fused = inputs.repeat(num_configs, *([1] * (inputs.dim() - 1)))
    
    # 2. Build Alpha Overrides
    # This logic replicates `vectorized.py` lines ~2050-2100
    alpha_overrides: dict[str, torch.Tensor] = {}
    
    for slot_id in slots:
        slot = cast(SeedSlot, model.seed_slots[slot_id])
        current_alpha = slot.alpha
        
        # Alpha overrides must match MorphogeneticModel.fused_forward() expectations:
        # - CNN: (K*B, 1, 1, 1)
        # - Transformer: (K*B, 1, 1)
        alpha_shape = (num_configs * batch_size, *([1] * (inputs.dim() - 1)))
        
        override_vec = torch.full(
            alpha_shape,
            current_alpha,
            device=inputs.device,
            dtype=inputs.dtype
        )
        
        for cfg_idx, cfg in enumerate(configs):
            if slot_id in cfg:
                start = cfg_idx * batch_size
                end = (cfg_idx + 1) * batch_size
                alpha_value = cfg[slot_id]
                override_vec[start:end].fill_(alpha_value)
                
        alpha_overrides[slot_id] = override_vec
        
    # 3. Run Fused Forward Pass
    with torch.no_grad():
        # MorphogeneticModel exposes fused_forward() for alpha override batches.
        output_fused = model.fused_forward(inputs_fused, alpha_overrides=alpha_overrides)  # type: ignore[attr-defined]
        
    # 4. Split Results
    results = list(torch.split(output_fused, batch_size, dim=0))
    return results


# =============================================================================
# Verification Test
# =============================================================================

class TestFusedValidationFidelity:
    
    def test_fused_vs_sequential_accuracy(self):
        """Verify bitwise fidelity between fused and sequential validation."""
        device = "cpu"
        
        # 1. Setup Model (Standard CIFAR-10 like setup but smaller)
        # using create_model to get the full SlottedHost wrapper
        model = create_model(
            task="cifar10",  # Triggers ResNet host usually, but we'll mock internals if needed
            device=device,
            slots=["r0c1", "r0c2"]
        )
        
        # Germinate seeds to make them active
        # r0c1: BLENDING (alpha 0.5)
        # r0c2: FOSSILIZED (alpha 1.0)
        slot1 = cast(SeedSlot, model.seed_slots["r0c1"])
        slot1.germinate("conv_small", alpha_target=1.0)
        slot1.state.stage = SeedStage.BLENDING
        slot1.state.alpha = 0.5
        slot1.state.alpha_controller.alpha = 0.5
        
        slot2 = cast(SeedSlot, model.seed_slots["r0c2"])
        slot2.germinate("conv_small", alpha_target=1.0)
        slot2.state.stage = SeedStage.FOSSILIZED
        slot2.state.alpha = 1.0
        slot2.state.alpha_controller.alpha = 1.0

        # 2. Setup Data
        batch_size = 4
        # ResNet18 input is [B, 3, 32, 32]
        inputs = torch.randn(batch_size, 3, 32, 32, device=device)
        
        # 3. Define Configs
        configs = [
            {"_kind": "main"},                                   # Baseline (0.5, 1.0)
            {"_kind": "solo_1", "r0c1": 0.0},                    # r0c1 off
            {"_kind": "solo_2", "r0c2": 0.0},                    # r0c2 off
            {"_kind": "all_off", "r0c1": 0.0, "r0c2": 0.0},      # Both off
        ]
        
        # 4. Run Sequential
        # Save original states to restore after
        state_backup = {
            "r0c1": slot1.state.alpha,
            "r0c2": slot2.state.alpha
        }
        
        seq_outputs = run_sequential_validation(
            model, 
            inputs, 
            configs, 
            slots=["r0c1", "r0c2"]
        )
        
        # Restore state
        slot1.state.alpha = state_backup["r0c1"]
        slot1.state.alpha_controller.alpha = state_backup["r0c1"]
        slot2.state.alpha = state_backup["r0c2"]
        slot2.state.alpha_controller.alpha = state_backup["r0c2"]
        
        # 5. Run Fused
        fused_outputs = run_fused_validation(
            model,
            inputs,
            configs,
            slots=["r0c1", "r0c2"]
        )
        
        # 6. Verify
        assert len(seq_outputs) == len(fused_outputs)
        
        for i, (seq, fused) in enumerate(zip(seq_outputs, fused_outputs)):
            # Check shape
            assert seq.shape == fused.shape
            
            # Check values (Strict tolerance)
            # BN stats mismatch often causes ~1e-3 error
            # Correct implementation should be ~1e-6 or better (float32 precision)
            if not torch.allclose(seq, fused, atol=1e-5, rtol=1e-5):
                max_diff = (seq - fused).abs().max().item()
                config_kind = configs[i]["_kind"]
                pytest.fail(
                    f"Divergence in config {i} ({config_kind}): "
                    f"Max diff {max_diff:.2e}. "
                    "Likely caused by BatchNorm stats leakage or alpha override failure."
                )
                
    def test_batch_norm_stats_frozen(self):
        """Verify that Fused Validation does not update BN running stats."""
        device = "cpu"
        model = create_model(task="cifar10", device=device, slots=["r0c1"])
        
        # Access a BN layer
        bn_layer = model.host.blocks[0].bn
        initial_mean = bn_layer.running_mean.clone()
        
        # Run fused pass
        inputs = torch.randn(4, 3, 32, 32, device=device)
        configs = [{"r0c1": 0.0}, {"r0c1": 1.0}]
        
        run_fused_validation(model, inputs, configs, ["r0c1"])
        
        # Verify stats didn't change
        current_mean = bn_layer.running_mean
        assert torch.equal(initial_mean, current_mean), "BN running stats changed during fused validation!"
