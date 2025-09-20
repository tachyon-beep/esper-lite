"""Tezzeret compiler scaffolding.

Future work will execute PyTorch 2.8 `torch.compile` pipelines and persist
artifacts to Urza (see `docs/design/detailed_design/06-tezzeret.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from torch.serialization import add_safe_globals

from esper.karn import BlueprintMetadata


@dataclass(slots=True)
class CompileJobConfig:
    """Configuration for a Tezzeret compile job."""

    artifact_dir: Path
    use_cuda: bool = torch.cuda.is_available()
    max_retries: int = 1


class CompiledBlueprint(nn.Module):
    """Simple placeholder module representing a compiled blueprint."""

    def __init__(self, blueprint_id: str, parameters: dict[str, float]) -> None:
        super().__init__()
        self.blueprint_id = blueprint_id
        self.blueprint_params = parameters

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - stub
        return inputs


# Allow CompiledBlueprint deserialisation with torch.load(weights_only=True).
add_safe_globals([CompiledBlueprint])


class TezzeretCompiler:
    """Stub compiler that fakes torch.compile execution."""

    def __init__(self, config: CompileJobConfig) -> None:
        self._config = config

    def compile(
        self,
        metadata: BlueprintMetadata,
        parameters: dict[str, float] | None = None,
    ) -> Path:
        """Compile the blueprint and persist the artifact."""

        artifact_path = self._config.artifact_dir / f"{metadata.blueprint_id}.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        module = CompiledBlueprint(metadata.blueprint_id, parameters or {})
        torch.save(module, artifact_path)
        return artifact_path


__all__ = ["TezzeretCompiler", "CompileJobConfig", "CompiledBlueprint"]
