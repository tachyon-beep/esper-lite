"""Kasmina lifecycle using Leyline enums as the single source of truth.

All lifecycle stages are represented by `leyline_pb2.SeedLifecycleStage` values.
No parallel internal enums are used.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from esper.leyline import leyline_pb2

SeedStage = int


@dataclass(slots=True)
class LifecycleTransition:
    """Represents a transition attempt between stages."""

    current: SeedStage
    next_stage: SeedStage


class KasminaLifecycle:
    """State machine enforcing Leyline seed lifecycle stages."""

    def __init__(self) -> None:
        self._stage: SeedStage = leyline_pb2.SEED_STAGE_DORMANT

    @property
    def state(self) -> SeedStage:
        return self._stage

    def transition(self, next_stage: SeedStage) -> LifecycleTransition:
        """Transition to a Leyline stage if allowed."""

        current = self._stage
        if next_stage not in self.allowed_next(current):
            raise ValueError(
                f"Transition {leyline_pb2.SeedLifecycleStage.Name(current)} -> "
                f"{leyline_pb2.SeedLifecycleStage.Name(next_stage)} not allowed"
            )
        self._stage = next_stage
        return LifecycleTransition(current, next_stage)

    def allowed_next(self, stage: SeedStage | None = None) -> Iterable[SeedStage]:
        """Return allowed next stages given current stage using Leyline enums."""

        s = stage if stage is not None else self._stage
        m: dict[SeedStage, tuple[SeedStage, ...]] = {
            leyline_pb2.SEED_STAGE_UNKNOWN: (leyline_pb2.SEED_STAGE_DORMANT,),
            leyline_pb2.SEED_STAGE_DORMANT: (
                leyline_pb2.SEED_STAGE_GERMINATED,
                leyline_pb2.SEED_STAGE_TERMINATED,
            ),
            leyline_pb2.SEED_STAGE_GERMINATED: (
                leyline_pb2.SEED_STAGE_TRAINING,
                leyline_pb2.SEED_STAGE_CULLED,
            ),
            leyline_pb2.SEED_STAGE_TRAINING: (
                leyline_pb2.SEED_STAGE_BLENDING,
                leyline_pb2.SEED_STAGE_CULLED,
            ),
            leyline_pb2.SEED_STAGE_BLENDING: (
                leyline_pb2.SEED_STAGE_SHADOWING,
                leyline_pb2.SEED_STAGE_CULLED,
            ),
            leyline_pb2.SEED_STAGE_SHADOWING: (
                leyline_pb2.SEED_STAGE_PROBATIONARY,
                leyline_pb2.SEED_STAGE_CULLED,
            ),
            leyline_pb2.SEED_STAGE_PROBATIONARY: (
                leyline_pb2.SEED_STAGE_FOSSILIZED,
                leyline_pb2.SEED_STAGE_CULLED,
            ),
            leyline_pb2.SEED_STAGE_FOSSILIZED: (leyline_pb2.SEED_STAGE_TERMINATED,),
            leyline_pb2.SEED_STAGE_CULLED: (leyline_pb2.SEED_STAGE_EMBARGOED,),
            leyline_pb2.SEED_STAGE_EMBARGOED: (leyline_pb2.SEED_STAGE_RESETTING,),
            leyline_pb2.SEED_STAGE_RESETTING: (leyline_pb2.SEED_STAGE_DORMANT,),
            leyline_pb2.SEED_STAGE_TERMINATED: (),
        }
        return m.get(s, ())

    def stage_name(self) -> str:
        return leyline_pb2.SeedLifecycleStage.Name(self._stage)


__all__ = ["KasminaLifecycle", "LifecycleTransition", "SeedStage"]
