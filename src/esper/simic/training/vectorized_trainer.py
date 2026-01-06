from __future__ import annotations

from typing import Any, Callable

from esper.simic.agent import PPOAgent


class VectorizedPPOTrainer:
    def __init__(
        self,
        train_fn: Callable[..., tuple[PPOAgent, list[dict[str, Any]]]],
    ) -> None:
        self._train_fn = train_fn

    def run(self, *args: Any, **kwargs: Any) -> tuple[PPOAgent, list[dict[str, Any]]]:
        return self._train_fn(*args, **kwargs)
