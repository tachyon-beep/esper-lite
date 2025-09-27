from __future__ import annotations

import math
from typing import Sequence

import pytest
import torch

from esper.tolaria.aggregation import (
    aggregate_weighted,
    combine_flat_grads,
    flat_to_grads,
    grads_to_flat,
)


def _cpu_device() -> torch.device:
    return torch.device("cpu")


def _build_vector(values: Sequence[float], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def test_grads_to_flat_preserves_device_and_dtype():
    device = _cpu_device()
    grads = [_build_vector([1.0, 2.0], device=device), _build_vector([3.0], device=device)]

    flat, shapes = grads_to_flat(grads)

    assert flat.device == grads[0].device
    assert flat.dtype == grads[0].dtype
    assert shapes == [torch.Size([2]), torch.Size([1])]
    assert torch.allclose(flat, _build_vector([1.0, 2.0, 3.0], device=device))


def test_grads_to_flat_raises_on_mixed_devices():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to create a mixed device gradient")

    cpu_grad = torch.ones(2, device=torch.device("cpu"))
    try:
        gpu_grad = torch.ones(2, device=torch.device("cuda"))
    except RuntimeError:
        pytest.skip("CUDA device unavailable or cannot allocate test tensor")

    with pytest.raises(RuntimeError):
        grads_to_flat([cpu_grad, gpu_grad])


def test_flat_to_grads_validates_element_counts():
    flat = torch.ones(5)
    shapes = [torch.Size([2]), torch.Size([2])]

    with pytest.raises(RuntimeError):
        flat_to_grads(flat, shapes)


def test_pcgrad_projects_conflicting_vectors():
    device = _cpu_device()
    g1 = _build_vector([1.0, 0.0], device=device)
    g2 = _build_vector([-1.0, 0.0], device=device)

    combined, conflicts = combine_flat_grads([g1, g2], use_pcgrad=True)

    assert conflicts == 1
    assert torch.allclose(combined, g1) or torch.allclose(combined, g2)


def test_pcgrad_no_projection_when_non_conflicting():
    device = _cpu_device()
    g1 = _build_vector([1.0, 0.0], device=device)
    g2 = _build_vector([0.0, 1.0], device=device)

    combined, conflicts = combine_flat_grads([g1, g2], use_pcgrad=True)

    assert conflicts == 0
    assert torch.allclose(combined, g1 + g2)


def test_weighted_aggregation_broadcasts_over_matrix():
    device = _cpu_device()
    g1 = torch.ones(2, 3, device=device)
    g2 = 2 * torch.ones(2, 3, device=device)

    result = aggregate_weighted([g1, g2], [1.0, 3.0])

    expected = (1.0 * g1 + 3.0 * g2) / 4.0
    assert torch.allclose(result, expected)


def test_weighted_aggregation_rejects_invalid_weights():
    device = _cpu_device()
    g1 = torch.ones(1, device=device)
    g2 = torch.ones(1, device=device)

    with pytest.raises(RuntimeError):
        aggregate_weighted([g1, g2], [float("nan"), 1.0])


def test_weighted_aggregation_requires_matching_shape():
    device = _cpu_device()
    g1 = torch.ones(1, device=device)
    g2 = torch.ones(2, device=device)

    with pytest.raises(RuntimeError):
        aggregate_weighted([g1, g2], [1.0, 1.0])


@pytest.mark.parametrize("use_pcgrad", [True, False])
def test_combine_flat_grads_raises_on_empty_input(use_pcgrad: bool):
    with pytest.raises(RuntimeError):
        combine_flat_grads([], use_pcgrad=use_pcgrad)
