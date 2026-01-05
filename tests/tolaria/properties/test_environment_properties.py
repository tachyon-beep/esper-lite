"""Property-based tests for Tolaria environment device parsing/validation."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from unittest.mock import patch

from esper.tolaria.environment import parse_device, validate_device

pytestmark = pytest.mark.property


@given(index=st.integers(min_value=0, max_value=127))
@settings(max_examples=50)
def test_parse_device_cuda_index_roundtrips(index: int) -> None:
    dev = parse_device(f"cuda:{index}")
    assert dev.type == "cuda"
    assert dev.index == index


@given(index=st.integers(min_value=0, max_value=4096))
@settings(max_examples=50)
def test_parse_device_rejects_cuda_missing_colon(index: int) -> None:
    with pytest.raises(ValueError, match="Invalid device"):
        parse_device(f"cuda{index}")


@given(device_type=st.sampled_from(["meta", "xla", "xpu", "hpu", "privateuseone"]))
@settings(max_examples=20)
def test_validate_device_rejects_unsupported_types(device_type: str) -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        validate_device(device_type)


def test_validate_device_cuda_requires_explicit_index_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="must include an explicit index"):
        validate_device("cuda", require_explicit_index=True)


@given(index=st.integers(min_value=0, max_value=5))
@settings(max_examples=30)
def test_validate_device_cuda_index_bounds(index: int) -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=2),
    ):
        if index < 2:
            dev = validate_device(f"cuda:{index}", require_explicit_index=True)
            assert dev.type == "cuda"
            assert dev.index == index
        else:
            with pytest.raises(RuntimeError, match="only .* device"):
                validate_device(f"cuda:{index}", require_explicit_index=True)


@given(index=st.integers(min_value=1, max_value=16))
@settings(max_examples=30)
def test_validate_device_mps_rejects_nonzero_index_when_available(
    index: int
) -> None:
    with patch.object(torch.backends.mps, "is_available", return_value=True):
        with pytest.raises(ValueError, match="Invalid MPS device index"):
            validate_device(f"mps:{index}")
