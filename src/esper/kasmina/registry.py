"""Parameter registration and validation for Kasmina."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from torch import nn


@dataclass(slots=True)
class RegistrationRecord:
    seed_id: str
    parameter_ids: set[int] = field(default_factory=set)


class SeedParameterRegistry:
    """Tracks parameter ownership across seeds and teacher models."""

    def __init__(self) -> None:
        self._seed_records: dict[str, RegistrationRecord] = {}
        self._parameter_index: dict[int, str] = {}
        self._teacher_parameter_ids: set[int] = set()

    def register_seed(self, seed_id: str, module: nn.Module) -> None:
        record = self._seed_records.setdefault(seed_id, RegistrationRecord(seed_id))
        for param in _iter_parameters(module):
            pid = id(param)
            owner = self._parameter_index.get(pid)
            if owner is not None and owner != seed_id:
                raise ValueError(f"parameter {pid} already registered to {owner}")
            record.parameter_ids.add(pid)
            self._parameter_index[pid] = seed_id

    def deregister_seed(self, seed_id: str) -> None:
        record = self._seed_records.pop(seed_id, None)
        if not record:
            return
        for pid in record.parameter_ids:
            self._parameter_index.pop(pid, None)

    def register_teacher(self, module: nn.Module) -> None:
        for param in _iter_parameters(module):
            self._teacher_parameter_ids.add(id(param))

    def validate_update(self, seed_id: str, parameters: Iterable[nn.Parameter]) -> bool:
        for param in parameters:
            pid = id(param)
            if pid in self._teacher_parameter_ids:
                return False
            owner = self._parameter_index.get(pid)
            if owner is None or owner != seed_id:
                return False
        return True

    def owner_of(self, parameter: nn.Parameter) -> str | None:
        return self._parameter_index.get(id(parameter))


def _iter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for param in module.parameters(recurse=True):
        if param.requires_grad:
            yield param


__all__ = ["SeedParameterRegistry", "RegistrationRecord"]
