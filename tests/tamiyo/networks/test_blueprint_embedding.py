"""Tests for BlueprintEmbedding component of Tamiyo's neural network.

Critical validation tests identified by PyTorch specialist review:
- Bounds checking for blueprint indices
- Device migration of _null_idx buffer
- Embedding correctness for edge cases
"""

import pytest
import torch

from esper.leyline import NUM_BLUEPRINTS, OBS_V3_NON_BLUEPRINT_DIM
from esper.tamiyo.networks.factored_lstm import BlueprintEmbedding


class TestBlueprintEmbedding:
    """Tests for BlueprintEmbedding module."""

    def test_basic_embedding(self):
        """Basic embedding lookup should work for valid indices."""
        embedding = BlueprintEmbedding()

        # Valid indices: 0 to NUM_BLUEPRINTS-1
        valid_idx = torch.tensor([[0, 5, NUM_BLUEPRINTS - 1]])
        result = embedding(valid_idx)

        # Output shape should be [batch, num_slots, embed_dim]
        assert result.shape == (1, 3, 4)  # embed_dim=4 by default
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_inactive_slot_handling(self):
        """Inactive slots (index -1) should map to null embedding."""
        embedding = BlueprintEmbedding()

        # Mix of active and inactive slots
        idx = torch.tensor([[0, -1, NUM_BLUEPRINTS - 1]])
        result = embedding(idx)

        assert result.shape == (1, 3, 4)

        # The null embedding should be a valid vector (not zeros)
        # It's index 13 in the embedding table (NUM_BLUEPRINTS)
        null_result = result[0, 1, :]  # The -1 slot
        assert not torch.allclose(null_result, torch.zeros(4)), (
            "Null embedding should be a learned vector, not zeros"
        )

    def test_null_idx_buffer_device_migration(self):
        """_null_idx buffer should migrate with model.to(device)."""
        embedding = BlueprintEmbedding()

        # Initially on CPU
        assert embedding._null_idx.device.type == "cpu"

        # Move to target device
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding = embedding.to(target_device)

        # Buffer should follow - compare device types to handle cuda vs cuda:0
        actual_device = embedding._null_idx.device
        assert actual_device.type == target_device.type, (
            f"_null_idx on {actual_device.type}, expected {target_device.type}"
        )

        # Embedding table should also follow
        assert embedding.embedding.weight.device.type == target_device.type

    def test_out_of_bounds_blueprint_index_fails(self):
        """Verify out-of-bounds blueprint indices are caught.

        Critical test from PyTorch specialist review: nn.Embedding does not
        bounds-check by default, which can cause silent memory corruption or
        crashes. This test ensures invalid indices are caught.

        When validation is enabled: ValueError with clear message
        When validation is disabled: IndexError from embedding layer
        """
        from esper.tamiyo.policy.action_masks import MaskedCategorical

        embedding = BlueprintEmbedding()

        # Valid range: -1 (inactive) or 0 to NUM_BLUEPRINTS-1
        valid_idx = torch.tensor([[0, NUM_BLUEPRINTS - 1, -1]])
        result = embedding(valid_idx)  # Should succeed
        assert result.shape == (1, 3, 4)

        # Out of bounds (positive) - index >= NUM_BLUEPRINTS
        invalid_idx_high = torch.tensor([[NUM_BLUEPRINTS + 1]])  # Index 14

        # With validation enabled: ValueError with helpful message
        original = MaskedCategorical.validate
        MaskedCategorical.validate = True
        try:
            with pytest.raises(ValueError, match="invalid indices"):
                embedding(invalid_idx_high)
        finally:
            MaskedCategorical.validate = original

        # With validation disabled: IndexError from embedding layer
        MaskedCategorical.validate = False
        try:
            with pytest.raises(IndexError):
                embedding(invalid_idx_high)
        finally:
            MaskedCategorical.validate = original

    def test_out_of_bounds_negative_below_minus_one_when_validation_enabled(self):
        """Blueprint index -2 or lower should fail when validation is enabled.

        Only -1 is valid for inactive slots. Other negative values are bugs.
        FAIL-FAST: Validation catches upstream bugs that emit invalid sentinels.
        """
        from esper.tamiyo.policy.action_masks import MaskedCategorical

        # Enable validation mode
        original = MaskedCategorical.validate
        MaskedCategorical.validate = True
        try:
            embedding = BlueprintEmbedding()

            # -2 is not a valid inactive marker - should raise
            invalid_negative = torch.tensor([[-2]])
            with pytest.raises(ValueError, match="invalid indices.*-2"):
                embedding(invalid_negative)

            # -999 should also fail
            invalid_extreme = torch.tensor([[-999]])
            with pytest.raises(ValueError, match="invalid indices.*-999"):
                embedding(invalid_extreme)
        finally:
            MaskedCategorical.validate = original

    def test_out_of_bounds_negative_below_minus_one_fails_with_index_error_without_validation(self):
        """Without validation, -2 causes IndexError (fail-fast behavior).

        The sentinel is exactly -1 for inactive slots. Other negative values
        are passed directly to the embedding, causing IndexError. This is the
        desired fail-fast behavior even when explicit validation is disabled.
        """
        from esper.tamiyo.policy.action_masks import MaskedCategorical

        # Disable validation mode
        original = MaskedCategorical.validate
        MaskedCategorical.validate = False
        try:
            embedding = BlueprintEmbedding()

            # Without validation, -2 passes directly to embedding → IndexError
            invalid_negative = torch.tensor([[-2]])
            with pytest.raises(IndexError):
                embedding(invalid_negative)
        finally:
            MaskedCategorical.validate = original

    def test_batch_and_sequence_dims(self):
        """Embedding should handle various batch and sequence dimensions."""
        embedding = BlueprintEmbedding()

        # [batch, seq, num_slots] format
        idx = torch.randint(0, NUM_BLUEPRINTS, (4, 10, 3))
        result = embedding(idx)

        assert result.shape == (4, 10, 3, 4)  # [batch, seq, num_slots, embed_dim]

    def test_embedding_gradients_flow(self):
        """Gradients should flow through the embedding layer."""
        embedding = BlueprintEmbedding()

        idx = torch.tensor([[0, 5, -1]])
        result = embedding(idx)

        # Sum and backprop
        loss = result.sum()
        loss.backward()

        # Embedding weights should have gradients
        assert embedding.embedding.weight.grad is not None
        assert not torch.allclose(
            embedding.embedding.weight.grad,
            torch.zeros_like(embedding.embedding.weight.grad)
        )


def test_blueprint_embedding_in_model_context():
    """Verify BlueprintEmbedding works correctly within full model."""
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic

    model = FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)

    # Test with valid indices including inactive slots
    state = torch.randn(2, 1, OBS_V3_NON_BLUEPRINT_DIM)
    bp_idx = torch.tensor([[[0, 5, -1]], [[NUM_BLUEPRINTS - 1, -1, -1]]])

    output = model(state, bp_idx)

    # Model should complete forward pass without error
    assert output["value"].shape == (2, 1)
    assert not torch.isnan(output["value"]).any()
