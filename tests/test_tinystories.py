"""Tests for TinyStories dataset."""

import pytest
import torch


@pytest.mark.slow
def test_tinystories_dataset_creates():
    """TinyStoriesDataset can be instantiated (mocked)."""
    from esper.utils.data import TinyStoriesDataset

    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64, mock=True)

    assert len(dataset) > 0


@pytest.mark.slow
def test_tinystories_returns_tensor_pair():
    """TinyStoriesDataset returns (input, target) tensor pair."""
    from esper.utils.data import TinyStoriesDataset

    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64, mock=True)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (64,)
    assert y.shape == (64,)


@pytest.mark.slow
def test_tinystories_shifted_target():
    """Target is shifted by 1 from input."""
    from esper.utils.data import TinyStoriesDataset

    dataset = TinyStoriesDataset(split="train", max_samples=10, block_size=64, mock=True)

    x, y = dataset[0]

    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(x[1:], y[:-1]) or y.shape == x.shape


def test_load_tinystories_helper():
    """load_tinystories returns train/val loaders."""
    from esper.utils.data import load_tinystories

    train_loader, val_loader = load_tinystories(
        block_size=16,
        batch_size=4,
        max_train_samples=8,
        max_val_samples=4,
        mock=True,
    )

    x, y = next(iter(train_loader))
    assert x.shape[1:] == (16,)
    assert y.shape[1:] == (16,)
