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
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Verify slots are created
    assert len(model.seed_slots) == 3
    assert "early" in model.seed_slots
    assert "mid" in model.seed_slots
    assert "late" in model.seed_slots

    # Verify correct channel dimensions from host
    assert model.seed_slots["early"].channels == 32
    assert model.seed_slots["mid"].channels == 64
    assert model.seed_slots["late"].channels == 128

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
    model = MorphogeneticModel(host, device="cpu", slots=["early", "mid", "late"])

    # Initially no active seeds
    assert not model.has_active_seed

    # Germinate in different slots (use actual available blueprints)
    model.germinate_seed("conv_light", "seed_early", slot="early")
    assert model.has_active_seed_in_slot("early")
    assert not model.has_active_seed_in_slot("mid")
    assert not model.has_active_seed_in_slot("late")

    model.germinate_seed("attention", "seed_late", slot="late")
    assert model.has_active_seed_in_slot("early")
    assert not model.has_active_seed_in_slot("mid")
    assert model.has_active_seed_in_slot("late")

    # Forward pass with active seeds
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

    # Verify parameter counts increased
    assert model.active_seed_params > 0

    # Cull a seed
    model.cull_seed(slot="early")
    assert not model.has_active_seed_in_slot("early")
    assert model.has_active_seed_in_slot("late")

    # Model still works after culling
    out = model(x)
    assert out.shape == (2, 10)

    # Cull remaining seed
    model.cull_seed(slot="late")
    assert not model.has_active_seed
    assert model.active_seed_params == 0
