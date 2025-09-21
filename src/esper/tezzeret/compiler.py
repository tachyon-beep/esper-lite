"""Tezzeret torch.compile pipeline."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable

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
    inductor_cache_dir: Path | None = None


class CompiledBlueprint(nn.Module):
    """Serialized blueprint artifact with metadata."""

    def __init__(
        self,
        module: nn.Module,
        *,
        blueprint_id: str,
        parameters: dict[str, float],
        guard_spec: Iterable[dict[str, Any]] | None = None,
        guard_digest: str | None = None,
        compile_strategy: str | None = None,
        eager_fallback: bool = False,
    ) -> None:
        super().__init__()
        self._module = module
        self.blueprint_id = blueprint_id
        self.blueprint_params = parameters
        self.guard_spec = list(guard_spec or ())
        self.guard_digest = guard_digest or ""
        self.compile_strategy = compile_strategy or "standard"
        self.eager_fallback = eager_fallback

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised downstream
        return self._module(*inputs)


# Allow CompiledBlueprint deserialisation with torch.load(weights_only=True).
add_safe_globals(
    [
        CompiledBlueprint,
        nn.Sequential,
        nn.Linear,
        nn.SiLU,
        nn.GELU,
        nn.LayerNorm,
        nn.Conv2d,
        nn.ReLU,
        nn.MultiheadAttention,
    ]
)


@dataclass(slots=True)
class CompilationResult:
    """Details captured for each compilation run."""

    artifact_path: Path
    guard_spec: list[dict[str, Any]]
    guard_digest: str
    compile_ms: float
    prewarm_ms: float
    eager_fallback: bool
    compile_strategy: str
    inductor_cache_dir: str | None


class TezzeretCompiler:
    """Torch 2.8 compilation pipeline for Tezzeret."""

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
        self._latest_result: CompilationResult | None = None

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
        self._latest_result = None
        while attempts <= self._config.max_retries:
            attempts += 1
            try:
                if self._error_sampler(metadata):
                    raise RuntimeError("Simulated compile failure")
                result = self._compile_blueprint(metadata, params, artifact_path)
                self._latest_result = result
                self._latest_catalog_update = self._build_catalog_update(
                    metadata,
                    artifact_path,
                    result.compile_ms,
                    result.prewarm_ms,
                    result.guard_digest,
                )
                self._clear_wal()
                return artifact_path
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                if attempts > self._config.max_retries:
                    break
        raise RuntimeError(f"Failed to compile blueprint {metadata.blueprint_id}: {last_error}")

    def latest_result(self) -> CompilationResult | None:
        return self._latest_result

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
        guard_digest: str,
    ) -> leyline_pb2.KernelCatalogUpdate:
        checksum = self._compute_checksum(artifact_path)
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

    def _compile_blueprint(
        self,
        metadata: BlueprintDescriptor,
        params: dict[str, float],
        artifact_path: Path,
    ) -> CompilationResult:
        device = torch.device("cuda" if self._config.use_cuda and torch.cuda.is_available() else "cpu")
        module, example_inputs = _build_blueprint_module(metadata, params, device)
        module.eval()
        guard_spec = _build_guard_spec(example_inputs)
        guard_digest = _guard_digest(guard_spec)

        compile_ms: float = 0.0
        prewarm_ms: float = 0.0
        eager_fallback = False
        strategy = "standard"

        with _inductor_cache(self._config.inductor_cache_dir):
            try:
                compile_start = time.perf_counter()
                compiled = torch.compile(module, dynamic=True)  # type: ignore[attr-defined]
                with torch.inference_mode():
                    compiled(*example_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                compile_ms = (time.perf_counter() - compile_start) * 1000.0
            except Exception:
                eager_fallback = True
                strategy = "eager"
                compiled = module

            prewarm_start = time.perf_counter()
            with torch.inference_mode():
                compiled(*example_inputs)
            if device.type == "cuda":  # pragma: no cover - GPU specific
                torch.cuda.synchronize()
            prewarm_ms = (time.perf_counter() - prewarm_start) * 1000.0

        module_to_save = module.to("cpu")
        module_to_save.eval()
        artifact = CompiledBlueprint(
            module_to_save,
            blueprint_id=metadata.blueprint_id,
            parameters=params,
            guard_spec=guard_spec,
            guard_digest=guard_digest,
            compile_strategy=strategy,
            eager_fallback=eager_fallback,
        )
        torch.save(artifact, artifact_path)

        return CompilationResult(
            artifact_path=artifact_path,
            guard_spec=guard_spec,
            guard_digest=guard_digest,
            compile_ms=compile_ms,
            prewarm_ms=prewarm_ms,
            eager_fallback=eager_fallback,
            compile_strategy=strategy,
            inductor_cache_dir=str(self._config.inductor_cache_dir) if self._config.inductor_cache_dir else None,
        )


def _build_guard_spec(inputs: Iterable[torch.Tensor]) -> list[dict[str, Any]]:
    spec: list[dict[str, Any]] = []
    for index, tensor in enumerate(inputs):
        tensor_cpu = tensor.detach().cpu()
        spec.append(
            {
                "index": index,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
                "stride": list(tensor_cpu.stride()),
                "requires_grad": bool(tensor.requires_grad),
            }
        )
    return spec


def _guard_digest(guard_spec: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(guard_spec), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_blueprint_module(
    metadata: BlueprintDescriptor,
    params: dict[str, float],
    device: torch.device,
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    feature_dim = 32
    batch = 8
    blueprint_id = metadata.blueprint_id.lower()

    if "linear" in blueprint_id or metadata.name.lower().startswith("linear"):
        module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.SiLU(),
        )
        scale = float(params.get("scale", 1.0))
        with torch.no_grad():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(scale)
        inputs = (torch.randn(batch, feature_dim, device=device),)
    elif "layer_norm" in blueprint_id:
        module = nn.LayerNorm(feature_dim)
        inputs = (torch.randn(batch, feature_dim, device=device),)
    elif "conv" in blueprint_id or "spatial" in blueprint_id or "channel" in blueprint_id:
        channels = 8
        module = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        inputs = (torch.randn(batch, channels, 16, 16, device=device),)
    elif "attention" in blueprint_id:
        heads = int(params.get("heads", 2))
        embed = heads * 16
        module = nn.MultiheadAttention(embed_dim=embed, num_heads=heads, batch_first=True)
        seq = 12
        inputs = (
            torch.randn(batch, seq, embed, device=device),
            torch.randn(batch, seq, embed, device=device),
            torch.randn(batch, seq, embed, device=device),
        )
    else:
        module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        inputs = (torch.randn(batch, feature_dim, device=device),)

    module = module.to(device)
    return module, inputs


@contextlib.contextmanager
def _inductor_cache(cache_dir: Path | None):
    if cache_dir is None:
        yield
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    original = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    try:
        yield
    finally:  # pragma: no cover - env restore
        if original is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original


__all__ = [
    "TezzeretCompiler",
    "CompileJobConfig",
    "CompiledBlueprint",
    "CompilationResult",
]
