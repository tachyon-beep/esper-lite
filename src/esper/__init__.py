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

# Public convenience exports (keep this module lightweight; no torch imports)
from esper.leyline import SeedStage, TrainingSignals

__all__ = [
    "SeedStage",
    "TrainingSignals",
]
