"""Tolaria training orchestrator package.

Implements the epoch-driven control loop described in
`docs/design/detailed_design/01-tolaria.md` and legacy spec `old/01-tolaria.md`.
"""

from .trainer import KasminaClient, TamiyoClient, TolariaTrainer, TrainingLoopConfig

__all__ = ["TolariaTrainer", "TrainingLoopConfig", "TamiyoClient", "KasminaClient"]
