"""Esper - Adaptive Neural Architecture System.

Esper implements dynamic model adaptation through the seed lifecycle:
germination -> training -> blending -> fossilization.

Subpackages:
- leyline: Data contracts and schemas
- kasmina: Seed mechanics and host models
- tamiyo: Strategic decision-making
- simic: RL training infrastructure
- nissa: System telemetry
"""

__version__ = "1.0.0"

# Re-export key types for convenience
from esper.leyline import SimicAction, SeedStage, TrainingSignals
from esper.kasmina import MorphogeneticModel, SeedSlot

__all__ = [
    "SimicAction",
    "SeedStage",
    "TrainingSignals",
    "MorphogeneticModel",
    "SeedSlot",
]
