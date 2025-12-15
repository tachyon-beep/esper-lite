"""Tests enforcing removal of legacy (V3) signals API.

Esper has a strict no-legacy-code policy (see CLAUDE.md). These tests ensure we
do not reintroduce backwards-compatibility shims for the older V3 observation
schema (TensorSchema/FastTrainingSignals).
"""

import pytest


def test_v3_tensor_schema_removed() -> None:
    with pytest.raises(ImportError):
        from esper.leyline.signals import TensorSchema  # noqa: F401


def test_v3_tensor_schema_size_removed() -> None:
    with pytest.raises(ImportError):
        from esper.leyline.signals import TENSOR_SCHEMA_SIZE  # noqa: F401


def test_v3_fast_training_signals_removed() -> None:
    with pytest.raises(ImportError):
        from esper.leyline.signals import FastTrainingSignals  # noqa: F401


def test_leyline_no_longer_exports_v3_fast_signals() -> None:
    with pytest.raises(ImportError):
        from esper.leyline import FastTrainingSignals  # noqa: F401

