"""Kasmina lifecycle state machine scaffolding.

The detailed behaviour must follow the eleven-state lifecycle defined in
`docs/design/detailed_design/old/02-kasmina.md`. This placeholder provides the
state transitions and guard hooks that future work will implement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class LifecycleEvent(str, Enum):
    """Lifecycle events that drive state transitions."""

    REGISTER = "register"
    GERMINATE = "germinate"
    ACTIVATE = "activate"
    OBSERVE = "observe"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    ROLLBACK = "rollback"
    RETIRE = "retire"
    TERMINATE = "terminate"


class LifecycleState(str, Enum):
    """Kasmina lifecycle states (subset placeholder)."""

    DORMANT = "dormant"
    REGISTERED = "registered"
    GERMINATING = "germinating"
    ACTIVE = "active"
    OBSERVING = "observing"
    DEGRADED = "degraded"
    ISOLATED = "isolated"
    ROLLING_BACK = "rolling_back"
    RETIRED = "retired"
    TERMINATED = "terminated"


@dataclass(slots=True)
class LifecycleTransition:
    """Represents a transition attempt between states."""

    current: LifecycleState
    event: LifecycleEvent
    next_state: LifecycleState


class KasminaLifecycle:
    """State machine shell for seed lifecycle enforcement."""

    def __init__(self) -> None:
        self._state = LifecycleState.DORMANT

    @property
    def state(self) -> LifecycleState:
        return self._state

    def apply(self, event: LifecycleEvent) -> LifecycleTransition:
        """Apply an event and transition to the resulting state."""

        current = self._state
        next_state = self._resolve_next_state(current, event)
        self._state = next_state
        return LifecycleTransition(current, event, next_state)

    def allowed_events(self, state: LifecycleState | None = None) -> Iterable[LifecycleEvent]:
        """Return the allowed events for the current or provided state."""

        mapping = {
            LifecycleState.DORMANT: (LifecycleEvent.REGISTER,),
            LifecycleState.REGISTERED: (LifecycleEvent.GERMINATE, LifecycleEvent.TERMINATE),
            LifecycleState.GERMINATING: (LifecycleEvent.ACTIVATE, LifecycleEvent.DEGRADE),
            LifecycleState.ACTIVE: (
                LifecycleEvent.OBSERVE,
                LifecycleEvent.DEGRADE,
                LifecycleEvent.RETIRE,
            ),
            LifecycleState.OBSERVING: (
                LifecycleEvent.DEGRADE,
                LifecycleEvent.RETIRE,
            ),
            LifecycleState.DEGRADED: (
                LifecycleEvent.ISOLATE,
                LifecycleEvent.ROLLBACK,
            ),
            LifecycleState.ISOLATED: (LifecycleEvent.ROLLBACK, LifecycleEvent.RETIRE),
            LifecycleState.ROLLING_BACK: (LifecycleEvent.OBSERVE,),
            LifecycleState.RETIRED: (LifecycleEvent.TERMINATE,),
            LifecycleState.TERMINATED: (),
        }

        return mapping[state or self._state]

    def _resolve_next_state(
        self, state: LifecycleState, event: LifecycleEvent
    ) -> LifecycleState:
        """Resolve the next state; raises ValueError when disallowed."""

        if event not in self.allowed_events(state):
            msg = f"Lifecycle event {event} not allowed from {state}"
            raise ValueError(msg)

        transitions = {
            (LifecycleState.DORMANT, LifecycleEvent.REGISTER): LifecycleState.REGISTERED,
            (LifecycleState.REGISTERED, LifecycleEvent.GERMINATE): LifecycleState.GERMINATING,
            (LifecycleState.REGISTERED, LifecycleEvent.TERMINATE): LifecycleState.TERMINATED,
            (LifecycleState.GERMINATING, LifecycleEvent.ACTIVATE): LifecycleState.ACTIVE,
            (LifecycleState.GERMINATING, LifecycleEvent.DEGRADE): LifecycleState.DEGRADED,
            (LifecycleState.ACTIVE, LifecycleEvent.OBSERVE): LifecycleState.OBSERVING,
            (LifecycleState.ACTIVE, LifecycleEvent.DEGRADE): LifecycleState.DEGRADED,
            (LifecycleState.ACTIVE, LifecycleEvent.RETIRE): LifecycleState.RETIRED,
            (LifecycleState.OBSERVING, LifecycleEvent.DEGRADE): LifecycleState.DEGRADED,
            (LifecycleState.OBSERVING, LifecycleEvent.RETIRE): LifecycleState.RETIRED,
            (LifecycleState.DEGRADED, LifecycleEvent.ISOLATE): LifecycleState.ISOLATED,
            (LifecycleState.DEGRADED, LifecycleEvent.ROLLBACK): LifecycleState.ROLLING_BACK,
            (LifecycleState.ISOLATED, LifecycleEvent.ROLLBACK): LifecycleState.ROLLING_BACK,
            (LifecycleState.ISOLATED, LifecycleEvent.RETIRE): LifecycleState.RETIRED,
            (LifecycleState.ROLLING_BACK, LifecycleEvent.OBSERVE): LifecycleState.OBSERVING,
            (LifecycleState.RETIRED, LifecycleEvent.TERMINATE): LifecycleState.TERMINATED,
        }

        return transitions[(state, event)]


__all__ = ["KasminaLifecycle", "LifecycleEvent", "LifecycleState", "LifecycleTransition"]
