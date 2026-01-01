"""Tests for B8-PT-01 record_stream fix assumptions.

The B8-PT-01 fix adds record_stream() calls in process_fused_val_batch() after
.to(device, non_blocking=True). The fix comment notes that when using
SharedGPUBatchIterator, tensors are already on the target device, making .to()
a no-op.

This test verifies that assumption: SharedGPUBatchIterator must return tensors
already on the environment's device, ensuring .to() returns the same tensor.
"""

import pytest
import torch

from esper.utils.data import SharedGPUBatchIterator, SharedGPUGatherBatchIterator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_gpu_iterator_returns_device_resident_tensors():
    """SharedGPUBatchIterator returns tensors already on the target device.

    This is a critical assumption for B8-PT-01: when tensors are already on
    env_dev, .to(env_dev, non_blocking=True) is a no-op returning the same
    tensor object. The caller's record_stream() already protects them.

    If this test fails, process_fused_val_batch() may have unprotected tensors
    in the GPU-resident path.
    """
    device = "cuda:0"
    batch_size = 4
    n_envs = 2

    iterator = SharedGPUBatchIterator(
        batch_size_per_env=batch_size,
        n_envs=n_envs,
        env_devices=[device] * n_envs,
        shuffle=False,
        is_train=False,  # Use smaller test set
    )

    # Get first batch
    env_batches = next(iter(iterator))

    for env_idx, (inputs, targets) in enumerate(env_batches):
        # Verify tensors are on expected device
        assert inputs.is_cuda, f"Env {env_idx}: inputs expected CUDA tensor"
        assert targets.is_cuda, f"Env {env_idx}: targets expected CUDA tensor"
        assert inputs.device == torch.device(device), (
            f"Env {env_idx}: inputs on {inputs.device}, expected {device}"
        )
        assert targets.device == torch.device(device), (
            f"Env {env_idx}: targets on {targets.device}, expected {device}"
        )

        # Verify .to() is a no-op (returns same tensor object)
        inputs_after_to = inputs.to(device, non_blocking=True)
        targets_after_to = targets.to(device, non_blocking=True)

        assert inputs_after_to is inputs, (
            f"Env {env_idx}: .to() created new tensor for inputs "
            "(expected no-op for GPU-resident data)"
        )
        assert targets_after_to is targets, (
            f"Env {env_idx}: .to() created new tensor for targets "
            "(expected no-op for GPU-resident data)"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_gpu_gather_iterator_returns_device_resident_tensors():
    """SharedGPUGatherBatchIterator returns tensors already on the target device."""
    device = "cuda:0"
    batch_size = 4
    n_envs = 2

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=batch_size,
        n_envs=n_envs,
        env_devices=[device] * n_envs,
        shuffle=False,
        is_train=False,  # Use smaller test set
        seed=123,
    )

    env_batches = next(iter(iterator))

    for env_idx, (inputs, targets) in enumerate(env_batches):
        assert inputs.is_cuda, f"Env {env_idx}: inputs expected CUDA tensor"
        assert targets.is_cuda, f"Env {env_idx}: targets expected CUDA tensor"
        assert inputs.device == torch.device(device), (
            f"Env {env_idx}: inputs on {inputs.device}, expected {device}"
        )
        assert targets.device == torch.device(device), (
            f"Env {env_idx}: targets on {targets.device}, expected {device}"
        )

        inputs_after_to = inputs.to(device, non_blocking=True)
        targets_after_to = targets.to(device, non_blocking=True)
        assert inputs_after_to is inputs
        assert targets_after_to is targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_gpu_gather_iterator_no_storage_aliasing_across_envs():
    """SharedGPUGatherBatchIterator must not return views sharing one allocation."""
    device = "cuda:0"
    batch_size = 4
    n_envs = 2

    iterator = SharedGPUGatherBatchIterator(
        batch_size_per_env=batch_size,
        n_envs=n_envs,
        env_devices=[device] * n_envs,
        shuffle=False,
        is_train=False,
        seed=123,
    )

    env_batches = next(iter(iterator))
    assert len(env_batches) == n_envs

    input_storage_ptrs = [
        inputs.untyped_storage().data_ptr() for inputs, _targets in env_batches
    ]
    target_storage_ptrs = [
        targets.untyped_storage().data_ptr() for _inputs, targets in env_batches
    ]

    assert len(set(input_storage_ptrs)) == n_envs
    assert len(set(target_storage_ptrs)) == n_envs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_to_noop_identity_on_same_device():
    """Verify .to(device, non_blocking=True) returns same tensor when already on device.

    This is a PyTorch contract test: .to() with matching device should return
    the same tensor object, not a copy. This behavior is what makes the B8-PT-01
    fix harmless in the GPU-resident path.
    """
    device = torch.device("cuda:0")
    tensor = torch.randn(4, 3, 32, 32, device=device)

    # Same device .to() should return identity
    tensor_after = tensor.to(device, non_blocking=True)

    assert tensor_after is tensor, (
        ".to() on same device should return same tensor object"
    )
    assert tensor_after.data_ptr() == tensor.data_ptr(), (
        ".to() on same device should share storage"
    )


def test_to_creates_new_tensor_on_device_change():
    """Verify .to() creates new tensor when moving between devices.

    This is the scenario B8-PT-01 protects: when data comes from CPU,
    .to(cuda, non_blocking=True) creates a NEW tensor that needs
    record_stream() protection.
    """
    cpu_tensor = torch.randn(4, 3, 32, 32, device="cpu")

    if torch.cuda.is_available():
        cuda_tensor = cpu_tensor.to("cuda:0", non_blocking=True)

        # Different devices must create new tensor
        assert cuda_tensor is not cpu_tensor, (
            ".to() across devices should create new tensor"
        )
        assert cuda_tensor.device.type == "cuda", (
            "New tensor should be on CUDA"
        )
