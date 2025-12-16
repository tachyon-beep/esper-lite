"""Tests for CNNHost segment boundaries for multi-slot support."""

import torch


def test_host_segment_channels():
    """CNNHost should expose channel counts at each segment boundary."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()

    # Should expose channels at each injection point (access directly - AttributeError if missing)
    assert host.segment_channels == {
        "r0c0": 32,   # After block1
        "r0c1": 64,     # After block2
        "r0c2": 128,   # After block3
    }


def test_host_forward_segments():
    """CNNHost should support segmented forward pass."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to each segment
    x_early = host.forward_to_segment("r0c0", x)
    assert x_early.shape == (2, 32, 16, 16)  # After block1 + pool

    x_mid = host.forward_to_segment("r0c1", x)
    assert x_mid.shape == (2, 64, 8, 8)  # After block2 + pool

    x_late = host.forward_to_segment("r0c2", x)
    assert x_late.shape == (2, 128, 4, 4)  # After block3 + pool

    # Forward from segment to output
    out = host.forward_from_segment("r0c2", x_late)
    assert out.shape == (2, 10)


def test_forward_from_early_segment():
    """Should be able to forward from early segment through rest of network."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to early, then from early to output
    x_early = host.forward_to_segment("r0c0", x)
    out = host.forward_from_segment("r0c0", x_early)
    assert out.shape == (2, 10)


def test_forward_from_mid_segment():
    """Should be able to forward from mid segment through rest of network."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to mid, then from mid to output
    x_mid = host.forward_to_segment("r0c1", x)
    out = host.forward_from_segment("r0c1", x_mid)
    assert out.shape == (2, 10)


def test_segment_channels_match_injection_points():
    """segment_channels should match the injection_points property."""
    from esper.kasmina.host import CNNHost

    host = CNNHost()

    # Both should expose the same channel information
    assert "block2_post" in host.injection_points
    assert host.injection_points["block2_post"] == host.segment_channels["r0c1"]


def test_multislot_model_creation():
    """MorphogeneticModel should support multiple slots."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    assert len(model.seed_slots) == 3
    assert "r0c0" in model.seed_slots
    assert "r0c1" in model.seed_slots
    assert "r0c2" in model.seed_slots

    # Each slot should have correct channels
    assert model.seed_slots["r0c0"].channels == 32
    assert model.seed_slots["r0c1"].channels == 64
    assert model.seed_slots["r0c2"].channels == 128


def test_multislot_forward_pass():
    """Multi-slot model forward should pass through all slots."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel
    import torch

    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_multislot_germinate_specific_slot():
    """Should germinate seed in specific slot."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Germinate in mid slot (use actual blueprint name)
    model.germinate_seed("conv_light", "test_seed", slot="r0c1")

    assert model.seed_slots["r0c1"].is_active
    assert not model.seed_slots["r0c0"].is_active
    assert not model.seed_slots["r0c2"].is_active


def test_transformer_forward_matches_host():
    """MorphogeneticModel with TransformerHost should match host output when no seeds active."""
    from esper.kasmina.host import TransformerHost, MorphogeneticModel
    import torch

    # Create host and model with multiple slots
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly when no seeds are active
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"


def test_transformer_single_slot_forward():
    """MorphogeneticModel with single slot should process all layers."""
    from esper.kasmina.host import TransformerHost, MorphogeneticModel
    import torch

    # Create host and model with only mid slot
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly - all layers must be processed
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"
