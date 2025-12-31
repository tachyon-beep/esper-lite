"""Integration test to verify Dynamic Topology Persistence.

Ensures that models with grown seed architectures can be saved and loaded,
restoring the exact topology and state (Stage, Alpha, Weights).
"""

import pytest
import torch
import os
from typing import cast

from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.tolaria.environment import create_model

# =============================================================================
# Helper Functions
# =============================================================================

def germinate_and_grow(model, slot_id, stage, alpha):
    """Germinate a seed and set its state."""
    slot = cast(SeedSlot, model.seed_slots[slot_id])
    slot.germinate("conv_small")
    slot.state.stage = stage
    slot.state.alpha = alpha
    slot.state.alpha_controller.alpha = alpha
    # Perturb weights to make them unique
    with torch.no_grad():
        for p in slot.seed.parameters():
            p.add_(0.1)

# =============================================================================
# Verification Test
# =============================================================================

@pytest.mark.integration
class TestTopologyPersistence:
    
    def test_save_load_grown_topology(self, tmp_path):
        """Verify saving a grown model and loading it into a fresh instance."""
        device = "cpu"
        checkpoint_path = tmp_path / "grown_model.pt"
        
        # --- Phase 1: Grow Model ---
        original_model = create_model(task="cifar10", device=device, slots=["r0c1", "r0c2"])
        
        # Grow seeds in different states
        germinate_and_grow(original_model, "r0c1", SeedStage.BLENDING, 0.5)
        germinate_and_grow(original_model, "r0c2", SeedStage.TRAINING, 0.0)
        
        # Verify state before save
        slot1 = cast(SeedSlot, original_model.seed_slots["r0c1"])
        assert slot1.is_active
        assert slot1.state.stage == SeedStage.BLENDING
        
        # Save
        # We save the state_dict AND a topology manifest (simulated)
        # In a real system, the state_dict might be enough IF we have a way to infer topology,
        # but standard PyTorch requires manual reconstruction.
        
        # Let's try to save JUST the state_dict first, as naive users would.
        torch.save(original_model.state_dict(), checkpoint_path)
        
        # Capture outputs for verification
        inputs = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            original_output = original_model(inputs)
            
        # --- Phase 2: Resume (Fresh Instance) ---
        # Create fresh model (starts DORMANT)
        restored_model = create_model(task="cifar10", device=device, slots=["r0c1", "r0c2"])
        
        # Verify it is empty
        assert not restored_model.has_active_seed
        
        # Attempt Load
        # Standard load fails because topology is missing.
        # We must inspect the checkpoint first.
        state_dict = torch.load(checkpoint_path)
        
        # --- Topology Reconstruction ---
        # 1. Detect active slots from keys
        detected_slots = set()
        for key in state_dict.keys():
            if key.startswith("seed_slots.") and ".seed." in key:
                parts = key.split(".")
                # seed_slots.{slot_id}.seed...
                slot_id = parts[1]
                detected_slots.add(slot_id)
        
        print(f"Detected slots in checkpoint: {detected_slots}")
        
        # 2. Pre-Germinate seeds to match topology
        # Note: We assume "conv_small" blueprint for this test.
        # In a real system, we'd need to save blueprint_id in extra_state or metadata.
        for slot_id in detected_slots:
            slot = cast(SeedSlot, restored_model.seed_slots[slot_id])
            # Germinate initializes seed module and default state
            slot.germinate("conv_small")
            
        # 3. Load State (Weights + SeedState)
        # Now that topology matches, this should succeed without error
        restored_model.load_state_dict(state_dict)
        
        # --- Verify State Fidelity ---
        slot1 = cast(SeedSlot, restored_model.seed_slots["r0c1"])
        print(f"Restored r0c1 stage: {slot1.state.stage}")
        assert slot1.state.stage == SeedStage.BLENDING
        assert slot1.state.alpha == 0.5
        
        slot2 = cast(SeedSlot, restored_model.seed_slots["r0c2"])
        print(f"Restored r0c2 stage: {slot2.state.stage}")
        assert slot2.state.stage == SeedStage.TRAINING
        assert slot2.state.alpha == 0.0
        
        # Verify outputs match
        with torch.no_grad():
            restored_output = restored_model(inputs)
            
        assert torch.allclose(original_output, restored_output), "Output mismatch after restore"
