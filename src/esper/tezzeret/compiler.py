"""Tezzeret torch.compile pipeline."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

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

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        if self.wal_path is not None:
            self.wal_path = Path(self.wal_path)
        if self.inductor_cache_dir is None:
            env_value = (
                os.environ.get("TEZZERET_INDUCTOR_CACHE_DIR")
                or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            )
            if env_value:
                self.inductor_cache_dir = Path(env_value)
        elif not isinstance(self.inductor_cache_dir, Path):
            self.inductor_cache_dir = Path(self.inductor_cache_dir)


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
    guard_summary: tuple[str, ...]
    compile_ms: float
    prewarm_ms: float
    eager_fallback: bool
    compile_strategy: str
    inductor_cache_dir: str | None


@dataclass(slots=True)
class CompilerMetrics:
    total_jobs: int = 0
    failed_jobs: int = 0
    failed_attempts: int = 0
    retried_jobs: int = 0
    eager_fallbacks: int = 0
    last_compile_ms: float = 0.0
    last_prewarm_ms: float = 0.0
    last_strategy: str = ""
    duration_by_strategy: dict[str, float] = field(default_factory=dict)
    prewarm_by_strategy: dict[str, float] = field(default_factory=dict)


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
        self._metrics = CompilerMetrics()

    def compile(
        self,
        metadata: BlueprintDescriptor,
        parameters: dict[str, float] | None = None,
        *,
        strategy: Literal["standard", "conservative"] = "standard",
    ) -> Path:
        """Compile the blueprint and persist the artifact."""

        if strategy not in {"standard", "conservative"}:
            raise ValueError(f"Unsupported compile strategy: {strategy}")

        artifact_path = self._config.artifact_dir / f"{metadata.blueprint_id}.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        params = parameters or {}
        self._persist_wal(metadata, params)

        last_error: Exception | None = None
        self._latest_result = None
        max_attempts = max(1, self._config.max_retries + 1)
        had_failures = False
        for attempt in range(1, max_attempts + 1):
            try:
                if self._error_sampler(metadata):
                    raise RuntimeError("Simulated compile failure")
                result = self._compile_blueprint(
                    metadata,
                    params,
                    artifact_path,
                    strategy=strategy,
                )
                self._latest_result = result
                self._latest_catalog_update = self._build_catalog_update(
                    metadata,
                    artifact_path,
                    result.compile_ms,
                    result.prewarm_ms,
                    result.guard_digest,
                )
                self._record_success(result, had_failures=had_failures)
                self._clear_wal()
                return artifact_path
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                self._record_attempt_failure()
                had_failures = True
                if attempt >= max_attempts:
                    break
        self._record_job_failure()
        raise RuntimeError(f"Failed to compile blueprint {metadata.blueprint_id}: {last_error}")

    def latest_result(self) -> CompilationResult | None:
        return self._latest_result

    def metrics_snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {
            "tezzeret.compilation.total": float(self._metrics.total_jobs),
            "tezzeret.compilation.failed": float(self._metrics.failed_jobs),
            "tezzeret.compilation.failed_attempts": float(self._metrics.failed_attempts),
            "tezzeret.compilation.eager_fallback": float(self._metrics.eager_fallbacks),
            "tezzeret.jobs.total": float(self._metrics.total_jobs),
            "tezzeret.jobs.failed": float(self._metrics.failed_jobs),
            "tezzeret.jobs.retried": float(self._metrics.retried_jobs),
            "tezzeret.compilation.last_compile_ms": self._metrics.last_compile_ms,
            "tezzeret.compilation.last_prewarm_ms": self._metrics.last_prewarm_ms,
        }
        for strategy, value in self._metrics.duration_by_strategy.items():
            snapshot[f"tezzeret.compilation.duration_ms.{strategy}"] = value
        for strategy, value in self._metrics.prewarm_by_strategy.items():
            snapshot[f"tezzeret.prewarm.ms.{strategy}"] = value
        return snapshot

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
        *,
        strategy: Literal["standard", "conservative"],
    ) -> CompilationResult:
        device = torch.device("cuda" if self._config.use_cuda and torch.cuda.is_available() else "cpu")
        try:
            module, example_inputs = _build_blueprint_module(metadata, params, device)
            module.eval()
        except Exception:
            # Fallback to CPU module build on CUDA initialisation issues
            device = torch.device("cpu")
            module, example_inputs = _build_blueprint_module(metadata, params, device)
            module.eval()
        guard_spec = _build_guard_spec(example_inputs)
        guard_digest = _guard_digest(guard_spec)
        guard_summary = _guard_summary(guard_spec)

        compile_ms: float = 0.0
        prewarm_ms: float = 0.0
        eager_fallback = False
        selected_strategy = strategy
        cache_dir = self._resolve_inductor_cache_dir()

        with _inductor_cache(cache_dir):
            disable_env = str(os.environ.get("TEZZERET_ENABLE_COMPILE", "false")).lower()
            compile_disabled = disable_env in ("0", "false", "no", "off")
            if strategy == "conservative":
                eager_fallback = True
                selected_strategy = "conservative"
                compiled = module
            elif compile_disabled:
                eager_fallback = True
                selected_strategy = "eager"
                # Force CPU to avoid GPU init/graphs in test environments
                if device.type == "cuda":
                    try:
                        module = module.to("cpu")
                        example_inputs = tuple(t.detach().cpu() for t in example_inputs)
                        device = torch.device("cpu")
                    except Exception:
                        pass
                compiled = module
            else:
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
                    selected_strategy = "eager"
                    # If CUDA path failed, fall back to CPU to avoid backend init issues
                    if device.type == "cuda":
                        try:
                            module = module.to("cpu")
                            example_inputs = tuple(t.detach().cpu() for t in example_inputs)
                            device = torch.device("cpu")
                        except Exception:
                            pass
                    compiled = module

            prewarm_start = time.perf_counter()
            with torch.inference_mode():
                compiled(*example_inputs)
            if device.type == "cuda":  # pragma: no cover - GPU specific
                with contextlib.suppress(Exception):
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
            compile_strategy=selected_strategy,
            eager_fallback=eager_fallback,
        )
        torch.save(artifact, artifact_path)

        return CompilationResult(
            artifact_path=artifact_path,
            guard_spec=guard_spec,
            guard_digest=guard_digest,
            guard_summary=guard_summary,
            compile_ms=compile_ms,
            prewarm_ms=prewarm_ms,
            eager_fallback=eager_fallback,
            compile_strategy=selected_strategy,
            inductor_cache_dir=str(cache_dir) if cache_dir else None,
        )

    def _record_success(self, result: CompilationResult, *, had_failures: bool) -> None:
        self._metrics.total_jobs += 1
        if had_failures:
            self._metrics.retried_jobs += 1
        self._metrics.last_compile_ms = result.compile_ms
        self._metrics.last_prewarm_ms = result.prewarm_ms
        self._metrics.last_strategy = result.compile_strategy
        self._metrics.duration_by_strategy[result.compile_strategy] = result.compile_ms
        self._metrics.prewarm_by_strategy[result.compile_strategy] = result.prewarm_ms
        if result.eager_fallback:
            self._metrics.eager_fallbacks += 1

    def _record_attempt_failure(self) -> None:
        self._metrics.failed_attempts += 1

    def _record_job_failure(self) -> None:
        self._metrics.total_jobs += 1
        self._metrics.failed_jobs += 1

    def _resolve_inductor_cache_dir(self) -> Path | None:
        if self._config.inductor_cache_dir is not None:
            return self._config.inductor_cache_dir
        env_value = (
            os.environ.get("TEZZERET_INDUCTOR_CACHE_DIR")
            or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        )
        return Path(env_value) if env_value else None


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


def _guard_summary(guard_spec: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    summary: list[str] = []
    for entry in guard_spec:
        shape = entry.get("shape", [])
        shape_descriptor = "x".join(str(dim) for dim in shape) or "scalar"
        dtype = entry.get("dtype", "unknown")
        summary.append(f"{dtype}[{shape_descriptor}]")
    return tuple(summary)


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
