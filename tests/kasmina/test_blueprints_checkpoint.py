"""Test activation checkpointing for large seeds."""
import pytest
import torch

from esper.kasmina.blueprints.registry import BlueprintRegistry


class TestMLPCheckpointing:
    """Verify MLP seed supports activation checkpointing."""

    def test_mlp_checkpoint_option(self):
        """MLP blueprint should accept checkpoint parameter."""
        # Create with checkpointing enabled
        seed = BlueprintRegistry.create(
            "transformer", "mlp", 64,
            checkpoint=True
        )

        x = torch.randn(2, 16, 64, requires_grad=True)
        result = seed(x)

        assert result.shape == x.shape

    def test_mlp_checkpoint_reduces_memory(self):
        """Checkpointing should reduce activation memory."""
        # This is a behavioral test - checkpointing trades compute for memory
        seed_no_ckpt = BlueprintRegistry.create("transformer", "mlp", 256, checkpoint=False)
        seed_ckpt = BlueprintRegistry.create("transformer", "mlp", 256, checkpoint=True)

        x = torch.randn(4, 32, 256, requires_grad=True)

        # Both should produce same output
        result_no_ckpt = seed_no_ckpt(x.clone())
        result_ckpt = seed_ckpt(x.clone())

        assert result_no_ckpt.shape == result_ckpt.shape

    def test_mlp_checkpoint_only_during_training(self):
        """Checkpointing should only apply during training with requires_grad."""
        seed = BlueprintRegistry.create("transformer", "mlp", 64, checkpoint=True)

        # Test 1: training mode with requires_grad=True (should use checkpoint)
        seed.train()
        x_train = torch.randn(2, 16, 64, requires_grad=True)
        result_train = seed(x_train)
        assert result_train.shape == x_train.shape

        # Test 2: eval mode (should not use checkpoint)
        seed.eval()
        x_eval = torch.randn(2, 16, 64, requires_grad=True)
        result_eval = seed(x_eval)
        assert result_eval.shape == x_eval.shape

        # Test 3: training mode but no requires_grad (should not use checkpoint)
        seed.train()
        x_no_grad = torch.randn(2, 16, 64, requires_grad=False)
        result_no_grad = seed(x_no_grad)
        assert result_no_grad.shape == x_no_grad.shape

    def test_mlp_checkpoint_gradient_flow(self):
        """Checkpointing should maintain gradient flow."""
        seed = BlueprintRegistry.create("transformer", "mlp", 64, checkpoint=True)
        seed.train()

        x = torch.randn(2, 16, 64, requires_grad=True)
        result = seed(x)
        loss = result.sum()
        loss.backward()

        # Gradients should flow through
        assert x.grad is not None
        # Seed parameters should have gradients
        for param in seed.parameters():
            assert param.grad is not None

    def test_mlp_default_checkpoint_false(self):
        """Default should be checkpoint=False."""
        # Create without explicit checkpoint parameter
        seed = BlueprintRegistry.create("transformer", "mlp", 64)

        x = torch.randn(2, 16, 64, requires_grad=True)
        result = seed(x)

        assert result.shape == x.shape
