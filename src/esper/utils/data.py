"""Data loading utilities.

Dataset loaders for training Model Alpha. Currently supports CIFAR-10,
with room to grow for ImageNet, synthetic datasets, etc.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# Shared Batch Iterator (single DataLoader serving multiple environments)
# =============================================================================

class SharedBatchIterator:
    """Single DataLoader serving multiple parallel environments.

    Instead of N independent DataLoaders (N × workers = massive IPC overhead),
    this uses ONE DataLoader with a combined batch size, then splits batches
    across environments.

    Performance comparison (4 envs, 4 workers each):
    - Independent DataLoaders: 16 worker processes, 16× IPC overhead
    - SharedBatchIterator: 4 worker processes, 1× IPC overhead

    Args:
        dataset: PyTorch Dataset to iterate
        batch_size_per_env: Batch size each environment receives
        n_envs: Number of parallel environments
        env_devices: List of device strings for each env (e.g., ["cuda:0", "cuda:1"])
        num_workers: DataLoader workers (shared across all envs)
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Drop incomplete batches (ensures even splits)
        generator: Optional torch.Generator for reproducible shuffling
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size_per_env: int,
        n_envs: int,
        env_devices: list[str],
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.n_envs = n_envs
        self.env_devices = env_devices
        self.batch_size_per_env = batch_size_per_env

        # Single DataLoader with combined batch size
        total_batch = batch_size_per_env * n_envs
        loader_kwargs = {
            "batch_size": total_batch,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        if generator is not None:
            loader_kwargs["generator"] = generator

        self.loader = DataLoader(dataset, **loader_kwargs)
        self._iter = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "SharedBatchIterator":
        self._iter = iter(self.loader)
        return self

    def __next__(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns list of (inputs, targets) tuples, one per environment.

        Each tuple's tensors are already moved to the corresponding env's device
        with non_blocking=True for async transfer.
        """
        inputs, targets = next(self._iter)

        # Split into per-env chunks (tensor_split preserves env count on partial batches)
        input_chunks = torch.tensor_split(inputs, self.n_envs)
        target_chunks = torch.tensor_split(targets, self.n_envs)

        # Move each chunk to its env's device (async)
        batches = []
        for i, (inp, tgt) in enumerate(zip(input_chunks, target_chunks)):
            if inp.numel() == 0:
                break
            device = self.env_devices[i % len(self.env_devices)]
            batches.append((
                inp.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            ))

        return batches


# =============================================================================
# GPU-Resident Data Loading (8x faster for small datasets)
# =============================================================================

# Global cache for GPU-resident datasets (avoid re-uploading).
#
# Keying includes data_root so callers can safely vary dataset locations within
# a long-lived interpreter (e.g., notebooks/tests) without stale tensor reuse.
_GPU_DATASET_CACHE: dict[
    tuple[str, str, str, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

_CIFAR10_GPU_CACHE_VERSION = "v1"


def _normalize_data_root(data_root: str) -> str:
    return str(Path(data_root).expanduser().resolve())


def _cifar10_cache_key(device: str, data_root: str) -> tuple[str, str, str, str]:
    return ("cifar10", device, _normalize_data_root(data_root), _CIFAR10_GPU_CACHE_VERSION)


def clear_gpu_dataset_cache(
    *,
    dataset: str | None = None,
    device: str | None = None,
    data_root: str | None = None,
) -> None:
    """Clear cached GPU-resident datasets.

    Intended for long-lived processes and tests where callers may want to vary
    data roots or free GPU memory.
    """
    global _GPU_DATASET_CACHE
    normalized_root = _normalize_data_root(data_root) if data_root is not None else None

    keys_to_delete: list[tuple[str, str, str, str]] = []
    for cache_key in _GPU_DATASET_CACHE:
        key_dataset, key_device, key_root, _key_version = cache_key
        if dataset is not None and key_dataset != dataset:
            continue
        if device is not None and key_device != device:
            continue
        if normalized_root is not None and key_root != normalized_root:
            continue
        keys_to_delete.append(cache_key)

    for cache_key in keys_to_delete:
        del _GPU_DATASET_CACHE[cache_key]


def _ensure_cifar10_cached(device: str, data_root: str = "./data", *, refresh: bool = False) -> None:
    """Ensure CIFAR-10 is cached on the specified device."""
    global _GPU_DATASET_CACHE
    cache_key = _cifar10_cache_key(device, data_root)

    if refresh or cache_key not in _GPU_DATASET_CACHE:
        # Load raw data (no augmentation for GPU-resident - applied at batch time)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )

        # Preload to GPU (one-time cost ~12s, amortized over training)
        train_x = torch.stack([trainset[i][0] for i in range(len(trainset))])
        train_y = torch.tensor([trainset[i][1] for i in range(len(trainset))])
        test_x = torch.stack([testset[i][0] for i in range(len(testset))])
        test_y = torch.tensor([testset[i][1] for i in range(len(testset))])

        _GPU_DATASET_CACHE[cache_key] = (
            train_x.to(device),
            train_y.to(device),
            test_x.to(device),
            test_y.to(device),
        )


class SharedGPUBatchIterator:
    """Single GPU-resident DataLoader serving multiple parallel environments per device.

    This is the GPU-preload equivalent of SharedBatchIterator. Instead of having N
    independent DataLoaders sharing cached GPU tensors (which causes race conditions),
    this uses ONE DataLoader per device and splits batches across environments.

    CRITICAL: Multiple DataLoaders iterating shared GPU tensors concurrently causes
    data corruption. This class avoids that by using a single DataLoader per device.

    Architecture:
    - Groups environments by device
    - Creates one DataLoader per device with combined batch size for that device's envs
    - Iteration returns per-environment tensors already on the correct device

    Args:
        batch_size_per_env: Batch size each environment receives
        n_envs: Total number of parallel environments
        env_devices: List of device strings for each env (e.g., ["cuda:0", "cuda:0", "cuda:1", "cuda:1"])
        shuffle: Whether to shuffle data
        generator: Optional torch.Generator for reproducible shuffling
        data_root: Root directory for dataset storage
        is_train: If True, use training set; if False, use test set
    """

    def __init__(
        self,
        batch_size_per_env: int,
        n_envs: int,
        env_devices: list[str],
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        data_root: str = "./data",
        is_train: bool = True,
    ):
        self.n_envs = n_envs
        self.env_devices = env_devices
        self.batch_size_per_env = batch_size_per_env
        self.is_train = is_train

        # Group environments by device
        # e.g., {cuda:0: [0, 1, 2, 3], cuda:1: [4, 5, 6, 7]}
        self._device_to_env_indices: dict[str, list[int]] = {}
        for env_idx, device in enumerate(env_devices):
            if device not in self._device_to_env_indices:
                self._device_to_env_indices[device] = []
            self._device_to_env_indices[device].append(env_idx)

        # Ensure data is cached on all devices
        for device in self._device_to_env_indices.keys():
            _ensure_cifar10_cached(device, data_root)

        # CRITICAL: Synchronize all devices after cache initialization.
        # Although .to(device) is synchronous, ensuring all GPU memory transfers
        # are complete before creating DataLoaders prevents race conditions when
        # multiple devices access cached tensors concurrently.
        if torch.cuda.is_available():
            for device in self._device_to_env_indices.keys():
                if device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))

        # Create ONE DataLoader per device with combined batch size
        self._device_loaders: dict[str, DataLoader] = {}
        self._device_iters: dict[str, object] = {}

        for device, env_indices in self._device_to_env_indices.items():
            n_envs_on_device = len(env_indices)
            total_batch = batch_size_per_env * n_envs_on_device

            cache_key = _cifar10_cache_key(device, data_root)
            train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

            if is_train:
                dataset = TensorDataset(train_x, train_y)
            else:
                dataset = TensorDataset(test_x, test_y)

            loader_kwargs = {
                "batch_size": total_batch,
                "shuffle": shuffle,
                "num_workers": 0,  # Data already on GPU, no workers needed
                "drop_last": is_train,  # Drop last for training, keep for validation
            }
            if generator is not None:
                # Each device gets a deterministically offset generator
                device_gen = torch.Generator(device='cpu')
                device_gen.manual_seed(generator.initial_seed() + hash(device) % 2**31)
                loader_kwargs["generator"] = device_gen

            self._device_loaders[device] = DataLoader(dataset, **loader_kwargs)

        # Compute length as minimum across devices (ensures all devices provide data)
        self._len = min(len(loader) for loader in self._device_loaders.values())

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> "SharedGPUBatchIterator":
        # Reset iterators for all device loaders
        self._device_iters = {
            device: iter(loader)
            for device, loader in self._device_loaders.items()
        }
        return self

    def __next__(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Returns list of (inputs, targets) tuples, one per environment.

        Tensors are already on the correct device for each environment.
        """
        # Collect batches from each device and split across its environments
        result: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.n_envs

        for device, env_indices in self._device_to_env_indices.items():
            # Get combined batch from this device's DataLoader
            try:
                inputs, targets = next(self._device_iters[device])
            except StopIteration:
                raise StopIteration

            # Split into per-env chunks for environments on this device
            n_envs_on_device = len(env_indices)
            input_chunks = torch.tensor_split(inputs, n_envs_on_device)
            target_chunks = torch.tensor_split(targets, n_envs_on_device)

            # Assign to correct environment indices
            for local_idx, (inp, tgt) in enumerate(zip(input_chunks, target_chunks)):
                global_env_idx = env_indices[local_idx]
                result[global_env_idx] = (inp, tgt)

        # Type narrowing - all slots should be filled
        return [(inp, tgt) for inp, tgt in result if inp is not None and inp.numel() > 0]


def load_cifar10_gpu(
    batch_size: int = 512,
    generator: torch.Generator | None = None,
    data_root: str = "./data",
    device: str = "cuda:0",
    *,
    refresh: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with data pre-loaded to GPU for maximum throughput.

    This eliminates DataLoader worker overhead by keeping the entire dataset
    GPU-resident. For CIFAR-10 (~0.75 GB), this is 8x faster than CPU loading.

    Use this for small datasets (< 2GB) where GPU memory isn't a constraint.
    For larger datasets, use load_cifar10() with num_workers.

    Args:
        batch_size: Batch size for DataLoaders (default 512 for GPU throughput).
        generator: Optional torch.Generator for reproducible shuffling.
        data_root: Root directory for dataset storage.
        device: GPU device to preload data to.
        refresh: If True, reload tensors even if cached.

    Returns:
        Tuple of (trainloader, testloader) with GPU-resident data.
    """
    _ensure_cifar10_cached(device, data_root, refresh=refresh)
    cache_key = _cifar10_cache_key(device, data_root)
    train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

    # Create GPU-resident DataLoaders (no workers needed - data already on GPU)
    gpu_trainset = TensorDataset(train_x, train_y)
    gpu_testset = TensorDataset(test_x, test_y)

    trainloader = DataLoader(
        gpu_trainset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,  # Data already on GPU, no workers needed
    )
    testloader = DataLoader(
        gpu_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, testloader


def get_cifar10_datasets(
    data_root: str = "./data",
    mock: bool = False,
) -> tuple[Dataset, Dataset]:
    """Get raw CIFAR-10 datasets (for SharedBatchIterator).

    Args:
        data_root: Root directory for dataset storage.
        mock: If True, return synthetic data instead of downloading CIFAR-10.

    Returns:
        Tuple of (trainset, testset) Dataset objects.
    """
    if mock:
        train_x = torch.randn(1000, 3, 32, 32)
        train_y = torch.randint(0, 10, (1000,))
        test_x = torch.randn(200, 3, 32, 32)
        test_y = torch.randint(0, 10, (200,))
        return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)

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
        return trainset, testset
    except Exception as exc:
        warnings.warn(f"Falling back to synthetic CIFAR-10 data: {exc}")
        return get_cifar10_datasets(data_root=data_root, mock=True)


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
        eos_id = self.tokenizer.eos_token_id

        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.examples: list[list[int]] = []
        target_len = self.block_size + 1
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            # Stride by block_size, capturing all tokens including trailing chunks
            for i in range(0, len(tokens), self.block_size):
                chunk = tokens[i : i + target_len]
                if len(chunk) < 2:
                    # Skip 1-token chunks (already in previous chunk's y, no new signal)
                    continue
                if len(chunk) < target_len:
                    # Pad short chunks (short stories or trailing tokens) with EOS
                    chunk = chunk + [eos_id] * (target_len - len(chunk))
                self.examples.append(chunk)

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
