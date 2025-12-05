"""Data loading utilities.

Dataset loaders for training Model Alpha. Currently supports CIFAR-10,
with room to grow for ImageNet, synthetic datasets, etc.
"""

from __future__ import annotations

import warnings

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms


def load_cifar10(
    batch_size: int = 128,
    generator: torch.Generator | None = None,
    data_root: str = "./data",
    num_workers: int = 4,
    mock: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size for DataLoaders.
        generator: Optional torch.Generator for reproducible shuffling.
            Use different generators per environment to avoid GIL contention
            when multiple CUDA streams iterate shared DataLoaders.
        data_root: Root directory for dataset storage.
        num_workers: DataLoader workers (set to 0 for sandboxed/CI).
        mock: If True, return synthetic data instead of downloading CIFAR-10.

    Returns:
        Tuple of (trainloader, testloader).
    """
    if mock:
        train_x = torch.randn(batch_size * 2, 3, 32, 32)
        train_y = torch.randint(0, 10, (batch_size * 2,))
        test_x = torch.randn(batch_size * 2, 3, 32, 32)
        test_y = torch.randint(0, 10, (batch_size * 2,))
        trainset = TensorDataset(train_x, train_y)
        testset = TensorDataset(test_x, test_y)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        return trainloader, testloader

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
    except Exception as exc:
        warnings.warn(f"Falling back to synthetic CIFAR-10 data: {exc}")
        return load_cifar10(
            batch_size=batch_size,
            generator=generator,
            data_root=data_root,
            num_workers=0,
            mock=True,
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    trainloader = DataLoader(
        trainset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    testloader = DataLoader(
        testset,
        shuffle=False,
        **loader_kwargs,
    )

    return trainloader, testloader


# =============================================================================
# TinyStories Language Modeling Dataset
# =============================================================================


class TinyStoriesDataset(Dataset):
    """TinyStories for causal language modeling."""

    def __init__(
        self,
        split: str = "train",
        block_size: int = 256,
        max_samples: int | None = None,
        mock: bool = False,
        vocab_size: int = 50257,
    ):
        self.block_size = block_size

        if mock:
            # Synthetic token sequences for offline/testing use
            rng = torch.Generator().manual_seed(0)
            num_samples = max_samples or 16
            self.examples = [
                torch.randint(0, vocab_size, (block_size + 1,), generator=rng).tolist()
                for _ in range(num_samples)
            ]
            return

        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
        except ImportError as exc:
            raise ImportError(
                "TinyStories requires 'datasets' and 'transformers' packages. "
                "Install with: pip install datasets transformers"
            ) from exc

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.examples: list[list[int]] = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            for i in range(0, len(tokens) - self.block_size, self.block_size):
                self.examples.append(tokens[i : i + self.block_size + 1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def load_tinystories(
    block_size: int = 256,
    batch_size: int = 32,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
    mock: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load TinyStories train and validation dataloaders."""
    train_dataset = TinyStoriesDataset(
        split="train",
        block_size=block_size,
        max_samples=max_train_samples,
        mock=mock,
    )
    val_dataset = TinyStoriesDataset(
        split="validation",
        block_size=block_size,
        max_samples=max_val_samples,
        mock=mock,
    )

    pin = False if mock else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader
