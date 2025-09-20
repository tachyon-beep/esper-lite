"""Runtime helpers for loading compiled blueprints from Urza."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from torch.serialization import add_safe_globals

from esper.tezzeret.compiler import CompiledBlueprint
from esper.urza.library import UrzaLibrary


class UrzaRuntime:
    """Implements the BlueprintRuntime protocol expected by Kasmina."""

    def __init__(self, library: UrzaLibrary) -> None:
        self._library = library

    def load_kernel(self, blueprint_id: str) -> nn.Module:
        record = self._library.get(blueprint_id)
        if record is None:
            raise KeyError(f"Blueprint '{blueprint_id}' not found in Urza")
        artifact_path = Path(record.artifact_path)
        add_safe_globals([CompiledBlueprint])
        return torch.load(artifact_path, weights_only=False)


__all__ = ["UrzaRuntime"]
