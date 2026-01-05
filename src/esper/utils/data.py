"""Data loading utilities.

Dataset loaders for training Model Alpha. Currently supports CIFAR-10,
with room to grow for ImageNet, synthetic datasets, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 normalization constants (match CPU pipeline).
_CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32)
_CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32)
_CIFAR10_PAD_VALUE = -_CIFAR10_MEAN / _CIFAR10_STD


def _cifar10_pad_value(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Return per-channel pad value for normalized CIFAR-10 tensors."""
    return _CIFAR10_PAD_VALUE.to(device=device, dtype=dtype)


def augment_cifar10_batch(
    inputs: torch.Tensor,
    *,
    generator: torch.Generator,
    padding: int = 4,
    flip_prob: float = 0.5,
) -> torch.Tensor:
    """Apply CIFAR-10 RandomCrop + RandomHorizontalFlip on normalized tensors."""
    if inputs.ndim != 4:
        raise ValueError(f"Expected inputs with shape (B,C,H,W), got {inputs.shape}")
    batch_size, channels, height, width = inputs.shape

    if padding > 0:
        pad_value = _cifar10_pad_value(inputs.device, inputs.dtype)
        padded = torch.empty(
            (batch_size, channels, height + 2 * padding, width + 2 * padding),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        padded[:] = pad_value.view(1, channels, 1, 1)
        padded[:, :, padding:padding + height, padding:padding + width] = inputs
    else:
        padded = inputs

    if padding > 0:
        max_offset = padding * 2
        top = torch.randint(
            0, max_offset + 1, (batch_size,), generator=generator, device=inputs.device
        )
        left = torch.randint(
            0, max_offset + 1, (batch_size,), generator=generator, device=inputs.device
        )
        rows = top[:, None] + torch.arange(height, device=inputs.device)[None, :]
        cols = left[:, None] + torch.arange(width, device=inputs.device)[None, :]
        gathered = padded.gather(
            2, rows[:, None, :, None].expand(-1, channels, -1, padded.size(3))
        )
        cropped = gathered.gather(
            3, cols[:, None, None, :].expand(-1, channels, height, -1)
        )
    else:
        cropped = padded

    if flip_prob > 0.0:
        flip_mask = torch.rand(
            (batch_size,), generator=generator, device=inputs.device
        ) < flip_prob
        if flip_mask.any():
            cropped[flip_mask] = torch.flip(cropped[flip_mask], dims=[3])

    if inputs.is_contiguous(memory_format=torch.channels_last):
        return cropped.contiguous(memory_format=torch.channels_last)

    return cropped


def precompute_cifar10_augment(
    inputs: torch.Tensor,
    *,
    seed: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Apply deterministic CIFAR-10 augmentations in-place over a cached dataset."""
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed)

    total = inputs.size(0)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch = inputs[start:end]
        inputs[start:end] = augment_cifar10_batch(batch, generator=generator)

    return inputs


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
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
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
        loader_kwargs: dict[str, Any] = {
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

        self.loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset, **loader_kwargs)
        self._iter: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None

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
        assert self._iter is not None, "Iterator not initialized - call __iter__ first"
        inputs, targets = next(self._iter)

        # Split into per-env chunks (tensor_split preserves env count on partial batches)
        input_chunks = torch.tensor_split(inputs, self.n_envs)
        target_chunks = torch.tensor_split(targets, self.n_envs)

        # Move each chunk to its env's device (async)
        # CRITICAL: Clone after tensor_split to prevent CUDA stream race conditions.
        # tensor_split returns views, and multiple CUDA streams accessing the same
        # underlying memory causes data corruption. This matches SharedGPUBatchIterator.
        batches = []
        for i, (inp, tgt) in enumerate(zip(input_chunks, target_chunks)):
            if inp.numel() == 0:
                break
            device = self.env_devices[i % len(self.env_devices)]
            # Clone AFTER moving to device for efficiency (clone on GPU is fast)
            batches.append((
                inp.to(device, non_blocking=True).clone(),
                tgt.to(device, non_blocking=True).clone(),
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


def _cifar10_cache_key(
    device: str,
    data_root: str,
    augment_mode: str = "base",
    seed: int | None = None,
) -> tuple[str, str, str, str]:
    version = f"{_CIFAR10_GPU_CACHE_VERSION}-{augment_mode}"
    if augment_mode == "precompute":
        if seed is None:
            raise ValueError("seed is required for CIFAR-10 precompute augmentation")
        version = f"{version}-seed{seed}"
    return ("cifar10", device, _normalize_data_root(data_root), version)


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


def _ensure_cifar10_cached(
    device: str,
    data_root: str = "./data",
    *,
    refresh: bool = False,
    augment_mode: str = "base",
    seed: int | None = None,
) -> None:
    """Ensure CIFAR-10 is cached on the specified device."""
    global _GPU_DATASET_CACHE
    if augment_mode not in ("base", "precompute"):
        raise ValueError(f"Unsupported CIFAR-10 augment_mode: {augment_mode}")
    if augment_mode == "precompute" and seed is None:
        raise ValueError("seed is required for CIFAR-10 precompute augmentation")

    cache_key = _cifar10_cache_key(
        device, data_root, augment_mode=augment_mode, seed=seed
    )

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

        # Move to device with channels-last format in a single operation to avoid
        # 2x VRAM spike from separate .to(device) then .to(memory_format=...).
        # Channels-last enables Tensor Core acceleration on modern GPUs (Volta+).
        train_x = train_x.to(device, memory_format=torch.channels_last)
        train_y = train_y.to(device)
        test_x = test_x.to(device, memory_format=torch.channels_last)
        test_y = test_y.to(device)

        if augment_mode == "precompute":
            if seed is None:
                raise ValueError("seed is required for CIFAR-10 precompute augmentation")
            precompute_cifar10_augment(train_x, seed=seed)

        _GPU_DATASET_CACHE[cache_key] = (train_x, train_y, test_x, test_y)


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
        cifar_precompute_aug: bool = False,
        seed: int | None = None,
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

        if cifar_precompute_aug and seed is None:
            raise ValueError("seed is required for CIFAR-10 precompute augmentation")
        augment_mode = "precompute" if cifar_precompute_aug else "base"

        # Ensure data is cached on all devices
        for device in self._device_to_env_indices.keys():
            _ensure_cifar10_cached(
                device, data_root, augment_mode=augment_mode, seed=seed
            )

        # CRITICAL: Synchronize all devices after cache initialization.
        # Although .to(device) is synchronous, ensuring all GPU memory transfers
        # are complete before creating DataLoaders prevents race conditions when
        # multiple devices access cached tensors concurrently.
        if torch.cuda.is_available():
            for device in self._device_to_env_indices.keys():
                if device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))

        # Create ONE DataLoader per device with combined batch size
        self._device_loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._device_iters: dict[str, Iterator[tuple[torch.Tensor, torch.Tensor]]] = {}

        for device, env_indices in self._device_to_env_indices.items():
            n_envs_on_device = len(env_indices)
            total_batch = batch_size_per_env * n_envs_on_device

            cache_key = _cifar10_cache_key(
                device, data_root, augment_mode=augment_mode, seed=seed
            )
            train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

            dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]
            if is_train:
                dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(train_x, train_y))
            else:
                dataset = cast(Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(test_x, test_y))

            loader_kwargs: dict[str, Any] = {
                "batch_size": total_batch,
                "shuffle": shuffle,
                "num_workers": 0,  # Data already on GPU, no workers needed
                "drop_last": is_train,  # Drop last for training, keep for validation
            }
            if generator is not None:
                # Each device gets a deterministically offset generator.
                # Use _stable_device_index() instead of hash() for reproducibility
                # across Python processes (hash() is randomized unless PYTHONHASHSEED is set).
                device_gen = torch.Generator(device='cpu')
                device_gen.manual_seed(generator.initial_seed() + _stable_device_index(device))
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
            device_iter = self._device_iters[device]
            try:
                inputs, targets = next(device_iter)
            except StopIteration:
                raise StopIteration

            # Split into per-env chunks for environments on this device
            n_envs_on_device = len(env_indices)
            input_chunks = torch.tensor_split(inputs, n_envs_on_device)
            target_chunks = torch.tensor_split(targets, n_envs_on_device)

            # Assign to correct environment indices
            # CRITICAL: Clone tensors to break view aliasing from tensor_split().
            # tensor_split() returns views that share storage with the batch tensor.
            # Without cloning, concurrent CUDA stream operations across environments
            # or across multiple training runs cause data corruption (nll_loss assertion
            # failures with targets outside valid class range). This is the source fix
            # that eliminates the need for consumer-side cloning.
            for local_idx, (inp, tgt) in enumerate(zip(input_chunks, target_chunks, strict=True)):
                global_env_idx = env_indices[local_idx]
                result[global_env_idx] = (inp.clone(), tgt.clone())

        # Type narrowing - all slots should be filled
        return [item for item in result if item is not None and item[0].numel() > 0]


def _stable_device_index(device: str) -> int:
    """Return a stable numeric index for a device string.

    Used for deterministic per-device shuffling without relying on Python's hash()
    (which is randomized across processes unless PYTHONHASHSEED is pinned).
    """
    if device == "cpu":
        return 0
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        idx_str = device.split(":", 1)[1]
        try:
            return int(idx_str)
        except ValueError as e:
            raise ValueError(f"Invalid CUDA device string: {device}") from e
    raise ValueError(f"Unsupported device string: {device}")


@dataclass
class _GatherDeviceState:
    device: str
    env_indices: list[int]
    dataset_x: torch.Tensor
    dataset_y: torch.Tensor
    total_batch: int
    drop_last: bool
    shuffle: bool
    cpu_gen: torch.Generator
    perm: torch.Tensor | None = None
    cursor: int = 0


class SharedGPUGatherBatchIterator:
    """GPU-preload iterator that avoids DataLoader and uses direct gathers.

    This is an experimental alternative to SharedGPUBatchIterator. It iterates the
    GPU-cached CIFAR tensors directly, generating a per-device permutation and
    slicing index windows to build per-env batches via index_select.
    """

    def __init__(
        self,
        *,
        batch_size_per_env: int,
        n_envs: int,
        env_devices: list[str],
        shuffle: bool,
        data_root: str = "./data",
        is_train: bool = True,
        seed: int,
        cifar_precompute_aug: bool = False,
    ):
        if batch_size_per_env < 1:
            raise ValueError(
                f"batch_size_per_env must be >= 1 (got {batch_size_per_env})"
            )
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1 (got {n_envs})")
        if len(env_devices) != n_envs:
            raise ValueError(
                f"env_devices length ({len(env_devices)}) must match n_envs ({n_envs})"
            )

        self.n_envs = n_envs
        self.env_devices = env_devices
        self.batch_size_per_env = batch_size_per_env
        self.shuffle = shuffle
        self.is_train = is_train
        self.drop_last = is_train

        self._device_to_env_indices: dict[str, list[int]] = {}
        for env_idx, device in enumerate(env_devices):
            if device not in self._device_to_env_indices:
                self._device_to_env_indices[device] = []
            self._device_to_env_indices[device].append(env_idx)

        augment_mode = "precompute" if cifar_precompute_aug else "base"
        for device in self._device_to_env_indices.keys():
            _ensure_cifar10_cached(
                device, data_root, augment_mode=augment_mode, seed=seed
            )

        if torch.cuda.is_available():
            for device in self._device_to_env_indices.keys():
                if device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))

        self._device_states: dict[str, _GatherDeviceState] = {}
        per_device_lens: list[int] = []
        for device, env_indices in self._device_to_env_indices.items():
            cache_key = _cifar10_cache_key(
                device, data_root, augment_mode=augment_mode, seed=seed
            )
            train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]
            if is_train:
                dataset_x, dataset_y = train_x, train_y
            else:
                dataset_x, dataset_y = test_x, test_y

            n_envs_on_device = len(env_indices)
            total_batch = batch_size_per_env * n_envs_on_device
            dataset_len = dataset_x.size(0)

            if self.drop_last:
                device_len = dataset_len // total_batch
            else:
                device_len = int(math.ceil(dataset_len / total_batch))
            per_device_lens.append(device_len)

            device_gen = torch.Generator(device="cpu")
            device_seed = seed + 1009 * _stable_device_index(device)
            device_gen.manual_seed(device_seed)

            self._device_states[device] = _GatherDeviceState(
                device=device,
                env_indices=env_indices,
                dataset_x=dataset_x,
                dataset_y=dataset_y,
                total_batch=total_batch,
                drop_last=self.drop_last,
                shuffle=shuffle,
                cpu_gen=device_gen,
            )

        self._len = min(per_device_lens)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> "SharedGPUGatherBatchIterator":
        for state in self._device_states.values():
            dataset_len = state.dataset_x.size(0)
            if state.shuffle:
                perm_cpu = torch.randperm(dataset_len, generator=state.cpu_gen)
                state.perm = perm_cpu.to(state.device)
            else:
                state.perm = torch.arange(dataset_len, device=state.device)
            state.cursor = 0
        return self

    def __next__(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        result: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.n_envs

        for state in self._device_states.values():
            if state.perm is None:
                raise RuntimeError("Iterator not initialized - call __iter__ first")

            dataset_len = state.dataset_x.size(0)
            start = state.cursor
            if start >= dataset_len:
                raise StopIteration

            end = start + state.total_batch
            if end > dataset_len:
                if state.drop_last:
                    raise StopIteration
                end = dataset_len

            batch_indices = state.perm[start:end]
            state.cursor = end

            n_envs_on_device = len(state.env_indices)
            index_chunks = torch.tensor_split(batch_indices, n_envs_on_device)
            for local_idx, idx_chunk in enumerate(index_chunks):
                if idx_chunk.numel() == 0:
                    continue
                env_idx = state.env_indices[local_idx]
                # Use advanced indexing to preserve channels-last memory format.
                # torch.index_select returns NCHW-contiguous by default, but
                # dataset_x[idx_chunk] preserves the source tensor's memory format.
                inputs = state.dataset_x[idx_chunk]
                targets = state.dataset_y[idx_chunk]
                result[env_idx] = (inputs, targets)

        first_missing: int | None = None
        for i, item in enumerate(result):
            if item is None or item[0].numel() == 0:
                first_missing = i
                break
        if first_missing is None:
            return cast(list[tuple[torch.Tensor, torch.Tensor]], result)

        for j in range(first_missing + 1, self.n_envs):
            item = result[j]
            if item is not None and item[0].numel() > 0:
                raise RuntimeError(
                    "Non-contiguous per-env batches detected; this would misalign env indexing."
                )

        prefix = result[:first_missing]
        return cast(list[tuple[torch.Tensor, torch.Tensor]], [item for item in prefix if item is not None])


def load_cifar10_gpu(
    batch_size: int = 512,
    generator: torch.Generator | None = None,
    data_root: str = "./data",
    device: str = "cuda:0",
    *,
    refresh: bool = False,
    cifar_precompute_aug: bool = False,
    seed: int | None = None,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
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
    if cifar_precompute_aug and seed is None:
        raise ValueError("seed is required for CIFAR-10 precompute augmentation")
    augment_mode = "precompute" if cifar_precompute_aug else "base"
    _ensure_cifar10_cached(
        device, data_root, refresh=refresh, augment_mode=augment_mode, seed=seed
    )
    cache_key = _cifar10_cache_key(
        device, data_root, augment_mode=augment_mode, seed=seed
    )
    train_x, train_y, test_x, test_y = _GPU_DATASET_CACHE[cache_key]

    # Create GPU-resident DataLoaders (no workers needed - data already on GPU)
    gpu_trainset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
        Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(train_x, train_y)
    )
    gpu_testset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
        Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(test_x, test_y)
    )

    trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        gpu_trainset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,  # Data already on GPU, no workers needed
    )
    testloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        gpu_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, testloader


def get_cifar10_datasets(
    data_root: str = "./data",
    mock: bool = False,
) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], Dataset[tuple[torch.Tensor, torch.Tensor]]]:
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
        trainset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
            Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(train_x, train_y)
        )
        testset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
            Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(test_x, test_y)
        )
        return trainset, testset

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
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
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
        mock_trainset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
            Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(train_x, train_y)
        )
        mock_testset: Dataset[tuple[torch.Tensor, torch.Tensor]] = cast(
            Dataset[tuple[torch.Tensor, torch.Tensor]], TensorDataset(test_x, test_y)
        )
        mock_trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            mock_trainset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        mock_testloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            mock_testset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return mock_trainloader, mock_testloader

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

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        trainset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    testloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        testset,
        shuffle=False,
        **loader_kwargs,
    )

    return trainloader, testloader


# =============================================================================
# TinyStories Language Modeling Dataset
# =============================================================================


class TinyStoriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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
        self.examples: list[list[int]]

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

        self.examples = []
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
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
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
