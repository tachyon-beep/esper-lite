"""Tezzeret compiler scaffolding.

Future work will execute PyTorch 2.8 `torch.compile` pipelines and persist
artifacts to Urza (see `docs/design/detailed_design/06-tezzeret.md`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from torch.serialization import add_safe_globals

from esper.karn import BlueprintDescriptor


@dataclass(slots=True)
class CompileJobConfig:
    """Configuration for a Tezzeret compile job."""

    artifact_dir: Path
    use_cuda: bool = torch.cuda.is_available()
    max_retries: int = 1
    wal_path: Path | None = None


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

    def __init__(
        self,
        config: CompileJobConfig,
        *,
        error_sampler: Callable[[BlueprintDescriptor], bool] | None = None,
    ) -> None:
        self._config = config
        self._error_sampler = error_sampler or (lambda _: False)
        self._wal_path = config.wal_path or (config.artifact_dir / "tezzeret_wal.json")
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)

    def compile(
        self,
        metadata: BlueprintDescriptor,
        parameters: dict[str, float] | None = None,
    ) -> Path:
        """Compile the blueprint and persist the artifact."""

        artifact_path = self._config.artifact_dir / f"{metadata.blueprint_id}.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        params = parameters or {}
        self._persist_wal(metadata, params)

        attempts = 0
        last_error: Exception | None = None
        while attempts <= self._config.max_retries:
            attempts += 1
            try:
                if self._error_sampler(metadata):
                    raise RuntimeError("Simulated compile failure")
                module = CompiledBlueprint(metadata.blueprint_id, params)
                torch.save(module, artifact_path)
                self._clear_wal()
                return artifact_path
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                if attempts > self._config.max_retries:
                    break
        raise RuntimeError(f"Failed to compile blueprint {metadata.blueprint_id}: {last_error}")

    def _persist_wal(self, metadata: BlueprintDescriptor, parameters: dict[str, float]) -> None:
        record = {
            "blueprint_id": metadata.blueprint_id,
            "parameters": parameters,
        }
        self._wal_path.write_text(json.dumps(record), encoding="utf-8")

    def _clear_wal(self) -> None:
        if self._wal_path.exists():
            self._wal_path.unlink()


__all__ = ["TezzeretCompiler", "CompileJobConfig", "CompiledBlueprint"]
