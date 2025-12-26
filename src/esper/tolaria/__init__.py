"""Tolaria - Model Training Infrastructure

This package provides:
- environment: Model factory (create_model)
- governor: Fail-safe watchdog for catastrophic failure detection

Training loops are implemented inline in simic/training/vectorized.py
for performance (CUDA streams, AMP, multi-env parallelism).
"""

from esper.tolaria.environment import create_model
from esper.tolaria.governor import GovernorReport, TolariaGovernor

__all__ = [
    "create_model",
    "TolariaGovernor",
    "GovernorReport",
]
