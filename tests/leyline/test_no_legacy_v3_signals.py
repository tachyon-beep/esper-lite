"""Enforce removal of legacy (V3) signals exports.

Esper has a strict no-legacy-code policy (see CLAUDE.md). These tests ensure we
do not reintroduce V3 vector primitives (TensorSchema/FastTrainingSignals) after
the V4 multislot observation contract became the only supported path.
"""

from __future__ import annotations

import importlib


_FORBIDDEN_V3_EXPORTS = (
    "FastTrainingSignals",
    "TensorSchema",
    "TENSOR_SCHEMA_SIZE",
)


def test_leyline_does_not_export_v3_vector_primitives() -> None:
    leyline = importlib.import_module("esper.leyline")
    for name in _FORBIDDEN_V3_EXPORTS:
        assert name not in leyline.__dict__, f"Unexpected V3 export in esper.leyline: {name}"


def test_signals_module_has_no_v3_vector_primitives() -> None:
    signals = importlib.import_module("esper.leyline.signals")
    for name in _FORBIDDEN_V3_EXPORTS:
        assert name not in signals.__dict__, f"Unexpected V3 export in esper.leyline.signals: {name}"
