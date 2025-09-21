"""Tezzeret compiler scaffolding.

Future work will execute PyTorch 2.8 `torch.compile` pipelines and persist
artifacts to Urza (see `docs/design/detailed_design/06-tezzeret.md`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from torch.serialization import add_safe_globals

from esper.karn import BlueprintDescriptor
from esper.leyline import leyline_pb2


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
        self._latest_catalog_update: leyline_pb2.KernelCatalogUpdate | None = None

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
        compile_start = time.perf_counter()
        while attempts <= self._config.max_retries:
            attempts += 1
            try:
                if self._error_sampler(metadata):
                    raise RuntimeError("Simulated compile failure")
                module = CompiledBlueprint(metadata.blueprint_id, params)
                torch.save(module, artifact_path)
                compile_ms = (time.perf_counter() - compile_start) * 1000.0
                prewarm_start = time.perf_counter()
                with torch.inference_mode():
                    sample = torch.randn(1)
                    module(sample)
                prewarm_ms = (time.perf_counter() - prewarm_start) * 1000.0
                self._latest_catalog_update = self._build_catalog_update(
                    metadata,
                    artifact_path,
                    compile_ms,
                    prewarm_ms,
                )
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

    def latest_catalog_update(self) -> leyline_pb2.KernelCatalogUpdate | None:
        return self._latest_catalog_update

    def _build_catalog_update(
        self,
        metadata: BlueprintDescriptor,
        artifact_path: Path,
        compile_ms: float,
        prewarm_ms: float,
    ) -> leyline_pb2.KernelCatalogUpdate:
        checksum = self._compute_checksum(artifact_path)
        guard_digest = hashlib.sha256(
            f"{metadata.blueprint_id}:{metadata.tier}".encode("utf-8")
        ).hexdigest()
        update = leyline_pb2.KernelCatalogUpdate(
            blueprint_id=metadata.blueprint_id,
            artifact_ref=str(artifact_path),
            checksum=checksum,
            guard_digest=guard_digest,
            compile_ms=compile_ms,
            prewarm_ms=prewarm_ms,
        )
        return update

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["TezzeretCompiler", "CompileJobConfig", "CompiledBlueprint"]
