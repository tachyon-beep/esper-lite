"""Integration tests for TransformerHost with seed lifecycle."""

import torch


def test_transformer_with_seed_lifecycle():
    """Full seed lifecycle on TransformerHost."""
    from esper.kasmina.host import TransformerHost
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import SeedStage
    from esper.simic.features import TaskConfig

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32, dropout=0.0)
    host.eval()

    slot_id = list(host.injection_points.keys())[0]
    dim = host.injection_points[slot_id]
    config = TaskConfig.for_tinystories()
    slot = SeedSlot(slot_id=slot_id, channels=dim, fast_mode=True, task_config=config)

    x = torch.randint(0, 1000, (2, 16))
    out_before = host(x)

    slot.germinate("lora", "test-lora")
    torch.nn.init.ones_(slot.seed.up.weight)
    host.register_slot(slot_id, slot.seed)

    slot.state.transition(SeedStage.TRAINING)
    slot.set_alpha(1.0)

    out_with_seed = host(x)

    assert out_before.shape == out_with_seed.shape
    assert not torch.allclose(out_before, out_with_seed)

    host.unregister_slot(slot_id)

    out_after = host(x)

    assert torch.allclose(out_before, out_after)


def test_transformer_gradient_flow():
    """Gradients flow through TransformerHost with seed."""
    from esper.kasmina.host import TransformerHost
    from esper.kasmina.blueprints import BlueprintRegistry

    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=2, block_size=32, dropout=0.0)

    slot_id = "layer_0_post_block"
    lora = BlueprintRegistry.create("transformer", "lora", dim=64)
    host.register_slot(slot_id, lora)

    x = torch.randint(0, 1000, (2, 16))
    out = host(x)
    loss = out.sum()
    loss.backward()

    for name, param in lora.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_host_cnn_still_works():
    """HostCNN still works with new protocol."""
    from esper.kasmina.host import HostCNN
    from esper.kasmina.blueprints import BlueprintRegistry

    host = HostCNN()

    attention = BlueprintRegistry.create("cnn", "attention", dim=64)
    host.register_slot("block2_post", attention)

    x = torch.randn(2, 3, 32, 32)
    out = host(x)

    assert out.shape == (2, 10)

    loss = out.sum()
    loss.backward()

    for param in attention.parameters():
        assert param.grad is not None
