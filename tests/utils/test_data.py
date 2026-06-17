"""Tests for esper.utils package."""

import pytest
import torch
from torch.utils.data import TensorDataset

import esper.utils.data as data
from esper.utils.data import (
    AugmentationBuffers,
    SharedGPUBatchIterator,
    SharedGPUGatherBatchIterator,
    SharedBatchIterator,
    _cifar10_cache_key,
    _cifar10_pad_value,
    _ensure_cifar10_cached,
    _stable_device_index,
    _split_batch_span,
    augment_cifar10_batch,
    clear_gpu_dataset_cache,
    get_cifar10_datasets,
    load_cifar10,
    load_cifar10_gpu,
    precompute_cifar10_augment,
)


class TestData:
    """Tests for utils.data module."""

    def test_load_cifar10_returns_loaders(self):
        """Test CIFAR-10 loading returns DataLoaders."""
        trainloader, testloader = load_cifar10(batch_size=32, mock=True)
        assert len(trainloader) > 0
        assert len(testloader) > 0

    def test_load_cifar10_batch_size(self):
        """Test CIFAR-10 respects batch size."""
        trainloader, _ = load_cifar10(batch_size=64, mock=True)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape[0] == 64
        assert labels.shape[0] == 64

    def test_load_cifar10_data_shape(self):
        """Test CIFAR-10 data has correct shape."""
        trainloader, _ = load_cifar10(batch_size=32, mock=True)
        inputs, labels = next(iter(trainloader))
        assert inputs.shape == (32, 3, 32, 32)  # CIFAR-10 is 32x32 RGB

def test_shared_batch_iterator_retains_env_splits_on_partial_batch() -> None:
    """SharedBatchIterator should keep one batch per env on partial batches."""
    dataset = TensorDataset(
        torch.arange(10).float().unsqueeze(1),
        torch.arange(10),
    )
    iterator = SharedBatchIterator(
        dataset=dataset,
        batch_size_per_env=2,
        n_envs=3,
        env_devices=["cpu"] * 3,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    batches = list(iter(iterator))
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert [batch[0].shape[0] for batch in batches[0]] == [2, 2, 2]
    assert [batch[0].shape[0] for batch in batches[1]] == [2, 1, 1]


def test_split_batch_span_preserves_global_env_order_for_partial_tail() -> None:
    assert _split_batch_span(start=0, end=6, n_chunks=3) == [
        (0, 2),
        (2, 4),
        (4, 6),
    ]
    assert _split_batch_span(start=0, end=5, n_chunks=3) == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]

def test_load_cifar10_gpu_cache_key_includes_data_root(monkeypatch) -> None:
    """GPU dataset cache should not serve tensors across different data_root values."""
    import torchvision

    class FakeCIFAR10(torch.utils.data.Dataset):
        def __init__(self, root, train, download, transform):
            self.root = str(root)
            self.train = bool(train)
            self.transform = transform

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx):
            value = 1.0 if "root_a" in self.root else 2.0
            x = torch.full((3, 32, 32), value, dtype=torch.float32)
            y = 0
            return x, y

    clear_gpu_dataset_cache()
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", FakeCIFAR10)

    gen = torch.Generator().manual_seed(0)
    trainloader_a, _ = load_cifar10_gpu(
        batch_size=2, generator=gen, data_root="root_a", device="cpu"
    )
    x_a, _ = next(iter(trainloader_a))

    trainloader_b, _ = load_cifar10_gpu(
        batch_size=2, generator=gen, data_root="root_b", device="cpu"
    )
    x_b, _ = next(iter(trainloader_b))

    assert x_a.mean().item() == 1.0
    assert x_b.mean().item() == 2.0


def test_cifar10_pad_value_normalizes_per_channel() -> None:
    pad_value = _cifar10_pad_value("cpu", torch.float64)

    assert pad_value.dtype == torch.float64
    assert pad_value.shape == (3,)
    assert torch.all(pad_value < 0)


def test_augmentation_buffers_reallocate_on_shape_or_dtype_change() -> None:
    buffers = AugmentationBuffers.create(batch_size=2, device="cpu")
    first_buffer = buffers.padded

    buffers.ensure_capacity(2, 3, 32, 32, torch.float32)
    assert buffers.padded is first_buffer

    buffers.ensure_capacity(1, 3, 32, 32, torch.float64)

    assert buffers.padded is not first_buffer
    assert buffers.padded is not None
    assert buffers.padded.shape == (1, 3, 40, 40)
    assert buffers.padded.dtype == torch.float64
    assert buffers.batch_size == 1


def test_augment_cifar10_batch_rejects_non_image_batch() -> None:
    generator = torch.Generator().manual_seed(0)

    try:
        augment_cifar10_batch(torch.zeros(3, 32, 32), generator=generator)
    except ValueError as exc:
        assert "Expected inputs" in str(exc)
    else:
        raise AssertionError("Expected shape validation failure")


def test_augment_cifar10_batch_preserves_channels_last_and_can_skip_padding() -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)

    output = augment_cifar10_batch(
        inputs,
        generator=generator,
        padding=0,
        flip_prob=0.0,
    )

    assert output.shape == inputs.shape
    assert output.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(output, inputs)


def test_augment_cifar10_batch_reuses_buffers_and_flips_when_requested() -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    buffers = AugmentationBuffers.create(
        batch_size=2,
        device="cpu",
        padding=1,
        height=4,
        width=4,
    )

    output = augment_cifar10_batch(
        inputs,
        generator=generator,
        padding=1,
        flip_prob=1.0,
        buffers=buffers,
    )

    assert output.shape == inputs.shape
    assert buffers.padded is not None
    assert buffers.padded.shape == (2, 3, 6, 6)


def test_precompute_cifar10_augment_updates_chunks_in_place() -> None:
    inputs = torch.zeros(5, 3, 4, 4)

    output = precompute_cifar10_augment(inputs, seed=123, chunk_size=2)

    assert output is inputs
    assert inputs.shape == (5, 3, 4, 4)


def test_clear_gpu_dataset_cache_filters_entries() -> None:
    clear_gpu_dataset_cache()
    key_a = _cifar10_cache_key("cpu", "root_a")
    key_b = _cifar10_cache_key("cpu", "root_b")
    tensors = (
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 3, 32, 32),
        torch.zeros(1, dtype=torch.long),
    )
    data._GPU_DATASET_CACHE[key_a] = tensors
    data._GPU_DATASET_CACHE[key_b] = tensors

    cleared = clear_gpu_dataset_cache(data_root="root_a")

    assert cleared == 1
    assert key_a not in data._GPU_DATASET_CACHE
    assert key_b in data._GPU_DATASET_CACHE
    assert clear_gpu_dataset_cache() == 1


def test_cifar10_cache_key_requires_seed_for_precompute() -> None:
    try:
        _cifar10_cache_key("cpu", "./data", augment_mode="precompute")
    except ValueError as exc:
        assert "seed is required" in str(exc)
    else:
        raise AssertionError("Expected seed validation failure")


def test_ensure_cifar10_cached_rejects_invalid_augmentation_options() -> None:
    try:
        _ensure_cifar10_cached("cpu", augment_mode="unknown")
    except ValueError as exc:
        assert "Unsupported CIFAR-10 augment_mode" in str(exc)
    else:
        raise AssertionError("Expected augment mode validation failure")

    try:
        _ensure_cifar10_cached("cpu", augment_mode="precompute")
    except ValueError as exc:
        assert "seed is required" in str(exc)
    else:
        raise AssertionError("Expected seed validation failure")


def test_stable_device_index_parses_known_devices() -> None:
    assert _stable_device_index("cpu") == 0
    assert _stable_device_index("cuda") == 0
    assert _stable_device_index("cuda:2") == 2

    for device in ("cuda:abc", "mps"):
        try:
            _stable_device_index(device)
        except ValueError as exc:
            assert device in str(exc)
        else:
            raise AssertionError(f"Expected invalid device failure for {device}")


def test_load_cifar10_raises_on_load_failure(monkeypatch) -> None:
    import torchvision

    class BrokenCIFAR10:
        def __init__(self, root, train, download, transform) -> None:
            raise RuntimeError("offline")

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", BrokenCIFAR10)

    with pytest.raises(RuntimeError, match="offline"):
        load_cifar10(batch_size=4, num_workers=0)


def test_get_cifar10_datasets_raises_on_load_failure(monkeypatch) -> None:
    import torchvision

    class BrokenCIFAR10:
        def __init__(self, root, train, download, transform) -> None:
            raise RuntimeError("offline")

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", BrokenCIFAR10)

    with pytest.raises(RuntimeError, match="offline"):
        get_cifar10_datasets()


def _cached_cpu_cifar(data_root: str = "./data", *, seed: int | None = None) -> None:
    train_x = torch.arange(12 * 3 * 4 * 4, dtype=torch.float32).reshape(12, 3, 4, 4)
    train_y = torch.arange(12)
    test_x = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    test_y = torch.arange(5)
    key = _cifar10_cache_key(
        "cpu",
        data_root,
        augment_mode="precompute" if seed is not None else "base",
        seed=seed,
    )
    data._GPU_DATASET_CACHE[key] = (train_x, train_y, test_x, test_y)


def test_load_cifar10_gpu_uses_existing_cpu_cache(monkeypatch) -> None:
    clear_gpu_dataset_cache()
    _cached_cpu_cifar()
    monkeypatch.setattr(data, "_ensure_cifar10_cached", lambda *args, **kwargs: None)

    trainloader, testloader = load_cifar10_gpu(batch_size=3, device="cpu")
    train_x, train_y = next(iter(trainloader))
    test_x, test_y = next(iter(testloader))

    assert train_x.shape == (3, 3, 4, 4)
    assert train_y.shape == (3,)
    assert test_x.shape == (3, 3, 4, 4)
    assert test_y.shape == (3,)
    assert clear_gpu_dataset_cache() == 1


def test_load_cifar10_gpu_precompute_requires_seed() -> None:
    try:
        load_cifar10_gpu(device="cpu", cifar_precompute_aug=True)
    except ValueError as exc:
        assert "seed is required" in str(exc)
    else:
        raise AssertionError("Expected seed validation failure")


def test_shared_gpu_batch_iterator_splits_cached_batches(monkeypatch) -> None:
    clear_gpu_dataset_cache()
    _cached_cpu_cifar()
    monkeypatch.setattr(data, "_ensure_cifar10_cached", lambda *args, **kwargs: None)
    generator = torch.Generator().manual_seed(5)

    iterator = SharedGPUBatchIterator(
        batch_size_per_env=2,
        n_envs=2,
        env_devices=["cpu", "cpu"],
        shuffle=False,
        generator=generator,
        data_root="./data",
        is_train=True,
    )

    batches = next(iter(iterator))

    assert len(iterator) == 3
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 4, 4)
    assert batches[1][0].shape == (2, 3, 4, 4)
    assert clear_gpu_dataset_cache() == 1


def test_shared_gpu_batch_iterator_precompute_requires_seed() -> None:
    try:
        SharedGPUBatchIterator(
            batch_size_per_env=2,
            n_envs=1,
            env_devices=["cpu"],
            cifar_precompute_aug=True,
        )
    except ValueError as exc:
        assert "seed is required" in str(exc)
    else:
        raise AssertionError("Expected seed validation failure")


def test_shared_gpu_gather_iterator_validation() -> None:
    invalid_args = [
        {"batch_size_per_env": 0, "n_envs": 1, "env_devices": ["cpu"]},
        {"batch_size_per_env": 1, "n_envs": 0, "env_devices": []},
        {"batch_size_per_env": 1, "n_envs": 2, "env_devices": ["cpu"]},
    ]
    for kwargs in invalid_args:
        try:
            SharedGPUGatherBatchIterator(shuffle=False, seed=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {kwargs}")


def test_shared_gpu_gather_iterator_splits_cached_batches(monkeypatch) -> None:
    clear_gpu_dataset_cache()
    _cached_cpu_cifar(seed=11)
    monkeypatch.setattr(data, "_ensure_cifar10_cached", lambda *args, **kwargs: None)

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=2,
        n_envs=2,
        env_devices=["cpu", "cpu"],
        shuffle=False,
        seed=11,
        cifar_precompute_aug=True,
    )
    batches = next(iter(iterator))

    assert len(iterator) == 3
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 4, 4)
    assert batches[1][1].tolist() == [2, 3]
    assert clear_gpu_dataset_cache() == 1


def test_shared_gpu_gather_iterator_rejects_mismatched_cached_tensor_lengths(
    monkeypatch,
) -> None:
    clear_gpu_dataset_cache()
    key = _cifar10_cache_key("cpu", "./data")
    data._GPU_DATASET_CACHE[key] = (
        torch.zeros(4, 3, 4, 4),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(2, 3, 4, 4),
        torch.zeros(2, dtype=torch.long),
    )
    monkeypatch.setattr(data, "_ensure_cifar10_cached", lambda *args, **kwargs: None)

    try:
        SharedGPUGatherBatchIterator(
            batch_size_per_env=2,
            n_envs=1,
            env_devices=["cpu"],
            shuffle=False,
            seed=11,
        )
    except ValueError as exc:
        assert "cached CIFAR train tensors length mismatch" in str(exc)
    else:
        raise AssertionError("Expected cached tensor length validation failure")
    finally:
        clear_gpu_dataset_cache()


def test_shared_gpu_gather_iterator_non_shuffle_avoids_epoch_permutation(
    monkeypatch,
) -> None:
    clear_gpu_dataset_cache()
    _cached_cpu_cifar()
    monkeypatch.setattr(data, "_ensure_cifar10_cached", lambda *args, **kwargs: None)

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=2,
        n_envs=3,
        env_devices=["cpu", "cpu", "cpu"],
        shuffle=False,
        data_root="./data",
        is_train=False,
        seed=11,
    )

    batches = list(iterator)

    assert iterator._global_perm is None
    assert [[targets.tolist() for _inputs, targets in batch] for batch in batches] == [
        [[0, 1], [2, 3], [4]]
    ]
    input_storage_ptrs = [
        inputs.untyped_storage().data_ptr()
        for inputs, _targets in batches[0]
    ]
    assert len(set(input_storage_ptrs)) == 3
    assert clear_gpu_dataset_cache() == 1
