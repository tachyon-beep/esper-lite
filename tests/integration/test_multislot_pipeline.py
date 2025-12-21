"""Integration Tests for Multi-Slot Pipeline.

Tests that verify multi-slot model components work together end-to-end:
- MorphogeneticModel with multiple slots
- Seed lifecycle (germinate, forward, cull)

Focus: Integration - verifying components work together, not re-testing individual components.
"""

import torch


def test_multislot_model_creation_and_forward():
    """Multi-slot model creation with all 3 slots and forward pass."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    # Create host and multi-slot model
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Verify slots are created
    assert len(model.seed_slots) == 3
    assert "r0c0" in model.seed_slots
    assert "r0c1" in model.seed_slots
    assert "r0c2" in model.seed_slots

    # Verify correct channel dimensions from host
    assert model.seed_slots["r0c0"].channels == 32
    assert model.seed_slots["r0c1"].channels == 64
    assert model.seed_slots["r0c2"].channels == 128

    # Forward pass through model (no seeds yet)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), "Model should output class logits"

    # Verify no active seeds initially
    assert not model.has_active_seed


def test_end_to_end_multislot_lifecycle():
    """Full lifecycle: Model creation → Germinate → Forward → Cull."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Initially no active seeds
    assert not model.has_active_seed

    # Germinate in different slots (use actual available blueprints)
    model.germinate_seed("conv_light", "seed_early", slot="r0c0")
    assert model.has_active_seed_in_slot("r0c0")
    assert not model.has_active_seed_in_slot("r0c1")
    assert not model.has_active_seed_in_slot("r0c2")

    model.germinate_seed("attention", "seed_late", slot="r0c2")
    assert model.has_active_seed_in_slot("r0c0")
    assert not model.has_active_seed_in_slot("r0c1")
    assert model.has_active_seed_in_slot("r0c2")

    # Forward pass with active seeds
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

    # Verify parameter counts increased
    assert model.active_seed_params > 0

    # Cull a seed
    model.prune_seed(slot="r0c0")
    assert not model.has_active_seed_in_slot("r0c0")
    assert model.has_active_seed_in_slot("r0c2")

    # Model still works after culling
    out = model(x)
    assert out.shape == (2, 10)

    # Cull remaining seed
    model.prune_seed(slot="r0c2")
    assert not model.has_active_seed
    assert model.active_seed_params == 0
