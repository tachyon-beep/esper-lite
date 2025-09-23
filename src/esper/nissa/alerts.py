"""Alert evaluation helpers for Nissa (TKT-302)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable


Comparator = Callable[[float, float], bool]


def _greater_than(value: float, threshold: float) -> bool:
    return value > threshold


COMPARATORS: dict[str, Comparator] = {
    ">": _greater_than,
    "<": lambda v, t: v < t,
}


@dataclass(slots=True)
class AlertRule:
    name: str
    metric: str
    threshold: float
    comparator: str = ">"
    for_count: int = 1
    routes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class AlertEvent:
    name: str
    metric: str
    value: float
    source: str
    routes: tuple[str, ...]


class AlertRouter:
    """Collects alert events and mirrors them to routing stubs."""

    def __init__(self, handlers: dict[str, Callable[[AlertEvent], None]] | None = None) -> None:
        self._events: deque[AlertEvent] = deque()
        self._handlers = handlers or {}
        self._logs: dict[str, list[AlertEvent]] = {
            "slack": [],
            "pagerduty": [],
            "email": [],
        }

    def dispatch(self, event: AlertEvent) -> None:
        self._events.append(event)
        for route in event.routes:
            handler = self._handlers.get(route)
            if handler:
                handler(event)
            elif route in self._logs:
                self._logs[route].append(event)

    def events(self) -> list[AlertEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
        for log in self._logs.values():
            log.clear()

    def slack_notifications(self) -> list[AlertEvent]:
        return list(self._logs["slack"])

    def pagerduty_notifications(self) -> list[AlertEvent]:
        return list(self._logs["pagerduty"])

    def email_notifications(self) -> list[AlertEvent]:
        return list(self._logs["email"])


@dataclass(slots=True)
class _AlertState:
    consecutive: int = 0
    active: bool = False


class AlertEngine:
    """Evaluate telemetry metrics against alert rules."""

    def __init__(self, rules: Iterable[AlertRule], router: AlertRouter | None = None) -> None:
        self._rules = list(rules)
        self._router = router or AlertRouter()
        self._states: dict[str, _AlertState] = {rule.name: _AlertState() for rule in self._rules}
        self._active: dict[str, AlertEvent] = {}

    @property
    def active_alerts(self) -> dict[str, AlertEvent]:
        return dict(self._active)

    @property
    def router(self) -> AlertRouter:
        return self._router

    def evaluate(self, metrics: dict[str, float], source: str) -> None:
        for rule in self._rules:
            value = metrics.get(rule.metric)
            state = self._states[rule.name]
            comparator = COMPARATORS.get(rule.comparator, _greater_than)
            if value is not None and comparator(value, rule.threshold):
                state.consecutive += 1
            else:
                if state.active and rule.name in self._active:
                    self._active.pop(rule.name, None)
                state.consecutive = 0
                state.active = False
                continue

            if state.consecutive >= rule.for_count and not state.active:
                state.active = True
                event = AlertEvent(rule.name, rule.metric, value, source, rule.routes)
                self._active[rule.name] = event
                self._router.dispatch(event)


DEFAULT_ALERT_RULES: tuple[AlertRule, ...] = (
    # Coverage low for 3 consecutive packets → Slack
    AlertRule(
        name="tamiyo_coverage_low",
        metric="tamiyo.gnn.feature_coverage",
        threshold=0.7,
        comparator="<",
        for_count=3,
        routes=("slack",),
    ),
    AlertRule(
        name="training_latency_high",
        metric="tolaria.training.latency_ms",
        threshold=18.0,
        for_count=3,
        routes=("pagerduty", "slack"),
    ),
    AlertRule(
        name="kasmina_isolation_violation",
        metric="kasmina.isolation.violations",
        threshold=3.0,
        for_count=1,
        routes=("pagerduty",),
    ),
    AlertRule(
        name="oona_queue_depth",
        metric="oona.queue.depth",
        threshold=4000.0,
        for_count=2,
        routes=("slack",),
    ),
    AlertRule(
        name="tezzeret_compile_retry_high",
        metric="tezzeret.compile.retry_count",
        threshold=1.0,
        for_count=1,
        routes=("email",),
    ),
    # BSDS (Tamiyo) — prototype signals
    AlertRule(
        name="tamiyo_bsds_critical",
        metric="tamiyo.bsds.hazard_critical_signal",
        threshold=0.5,
        for_count=1,
        routes=("pagerduty",),
    ),
    AlertRule(
        name="tamiyo_bsds_high",
        metric="tamiyo.bsds.hazard_high_signal",
        threshold=0.5,
        for_count=3,
        routes=("slack",),
    ),
    # Elevated blueprint risk flag (ingested as a boolean gauge) → PagerDuty
    AlertRule(
        name="tamiyo_bsds_elevated_risk",
        metric="tamiyo.bsds.elevated_risk_flag",
        threshold=0.5,
        for_count=1,
        routes=("pagerduty",),
    ),
)


__all__ = ["AlertRule", "AlertEvent", "AlertRouter", "AlertEngine", "DEFAULT_ALERT_RULES"]
