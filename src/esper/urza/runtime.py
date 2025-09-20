"""Runtime helpers for loading compiled blueprints from Urza."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Tuple

import torch
from torch import nn

from torch.serialization import add_safe_globals

from esper.tezzeret.compiler import CompiledBlueprint
from esper.urza.library import UrzaLibrary


class UrzaRuntime:
    """Implements the BlueprintRuntime protocol expected by Kasmina."""

    def __init__(self, library: UrzaLibrary) -> None:
        self._library = library

    def fetch_kernel(self, blueprint_id: str) -> Tuple[nn.Module, float]:
        start = perf_counter()
        record = self._library.get(blueprint_id)
        if record is None:
            raise KeyError(f"Blueprint '{blueprint_id}' not found in Urza")
        artifact_path = Path(record.artifact_path)
        add_safe_globals([CompiledBlueprint])
        module = torch.load(artifact_path, weights_only=False)
        latency_ms = (perf_counter() - start) * 1000.0
        return module, latency_ms

    def load_kernel(self, blueprint_id: str) -> nn.Module:
        module, _ = self.fetch_kernel(blueprint_id)
        return module


__all__ = ["UrzaRuntime"]
