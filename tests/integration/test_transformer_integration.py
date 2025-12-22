"""Integration tests for TransformerHost with seed lifecycle via MorphogeneticModel."""

import torch


def test_transformer_with_seed_lifecycle():
    """Full seed lifecycle on TransformerHost through MorphogeneticModel."""
    from esper.kasmina.host import TransformerHost, MorphogeneticModel
    from esper.leyline import SeedStage
    from esper.tamiyo.policy.features import TaskConfig

    host = TransformerHost(
        vocab_size=1000, n_embd=64, n_head=2, n_layer=6,
        block_size=32, dropout=0.0, num_segments=3
    )
    config = TaskConfig.for_tinystories()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"], task_config=config)
    model.eval()

    x = torch.randint(0, 1000, (2, 16))
    out_before = model(x)

    # Germinate seed at first segment
    model.germinate_seed("lora", "test-lora", slot="r0c0")

    # Initialize weights for deterministic test
    slot = model.seed_slots["r0c0"]
    torch.nn.init.ones_(slot.seed.up.weight)
    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    out_with_seed = model(x)

    assert out_before.shape == out_with_seed.shape
    assert not torch.allclose(out_before, out_with_seed)

    # Cull seed
    model.prune_seed(slot="r0c0")

    out_after = model(x)
    assert torch.allclose(out_before, out_after)


def test_transformer_gradient_flow():
    """Gradients flow through TransformerHost with seed via MorphogeneticModel."""
    from esper.kasmina.host import TransformerHost, MorphogeneticModel
    from esper.leyline import SeedStage
    from esper.tamiyo.policy.features import TaskConfig

    host = TransformerHost(
        vocab_size=1000, n_embd=64, n_head=2, n_layer=6,
        block_size=32, dropout=0.0, num_segments=3
    )
    config = TaskConfig.for_tinystories()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0"], task_config=config)

    # Germinate seed and set to training mode
    model.germinate_seed("lora", "test-lora", slot="r0c0")
    slot = model.seed_slots["r0c0"]
    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    x = torch.randint(0, 1000, (2, 16))
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradients flow to seed parameters
    for param in model.get_seed_parameters("r0c0"):
        assert param.grad is not None, "No gradient for seed parameter"


def test_cnn_with_seed_lifecycle():
    """CNNHost seed lifecycle through MorphogeneticModel."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel
    from esper.leyline import SeedStage
    from esper.tamiyo.policy.features import TaskConfig

    host = CNNHost()
    config = TaskConfig.for_cifar10()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"], task_config=config)

    # Germinate attention seed at second block
    model.germinate_seed("attention", "test-attention", slot="r0c1")

    # Set to training mode for gradients to flow
    slot = model.seed_slots["r0c1"]
    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    x = torch.randn(2, 3, 32, 32)
    out = model(x)

    assert out.shape == (2, 10)

    loss = out.sum()
    loss.backward()

    # Check gradients flow to seed parameters
    for param in model.get_seed_parameters("r0c1"):
        assert param.grad is not None, "No gradient for seed parameter"
