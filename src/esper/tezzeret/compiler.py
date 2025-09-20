"""Tezzeret compiler scaffolding.

Future work will execute PyTorch 2.8 `torch.compile` pipelines and persist
artifacts to Urza (see `docs/design/detailed_design/06-tezzeret.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from esper.karn import BlueprintMetadata


@dataclass(slots=True)
class CompileJobConfig:
    """Configuration for a Tezzeret compile job."""

    artifact_dir: Path
    use_cuda: bool = torch.cuda.is_available()
    max_retries: int = 1


class UrzaArtifactStore(Protocol):
    """Protocol for Urza artifact persistence."""

    def save_kernel(self, blueprint_id: str, artifact_path: Path) -> None:
        """Persist the compiled kernel for later retrieval."""


class TezzeretCompiler:
    """Stub compiler that fakes torch.compile execution."""

    def __init__(self, artifact_store: UrzaArtifactStore, config: CompileJobConfig) -> None:
        self._store = artifact_store
        self._config = config

    def compile(self, metadata: BlueprintMetadata) -> Path:
        """Compile the blueprint and persist the artifact."""

        artifact_path = self._config.artifact_dir / f"{metadata.blueprint_id}.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"blueprint": metadata.blueprint_id}, artifact_path)
        self._store.save_kernel(metadata.blueprint_id, artifact_path)
        return artifact_path


__all__ = ["TezzeretCompiler", "CompileJobConfig", "UrzaArtifactStore"]
