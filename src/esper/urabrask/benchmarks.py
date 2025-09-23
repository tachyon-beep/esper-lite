"""Benchmark runner for Urabrask (prototype).

Strict plan conformance (URA7):
- API exposes BenchmarkProfile and BenchmarkConfig with in_shape, warmup/measure
  on config and device preference.
- run_benchmarks returns (BlueprintBenchmark, list[dict]) where the list is a
  JSON mirror per profile with the planned keys.
- Attempts runtime.fetch_kernel(...). If unavailable or any error occurs, falls
  back to a deterministic synthetic path and marks provenance as "fallback".
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Tuple

try:  # optional torch dependency at runtime path; fallback path works without
    import torch  # type: ignore
    from torch import nn  # type: ignore
except Exception:  # pragma: no cover - fallback path in environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore

from esper.leyline import leyline_pb2


@dataclass(slots=True)
class BenchmarkProfile:
    """Per-profile inputs and dtype.

    - name: short identifier (e.g. "batch16_f32")
    - in_shape: input tensor shape excluding batch dimension
    - dtype: one of {"float32", "bfloat16", "float16"}
    """

    name: str
    batch_size: int
    in_shape: Tuple[int, ...]
    dtype: str = "float32"


@dataclass(slots=True)
class BenchmarkConfig:
    profiles: Tuple[BenchmarkProfile, ...] = (
        BenchmarkProfile(name="batch16_f32", batch_size=16, in_shape=(128,), dtype="float32"),
        BenchmarkProfile(name="batch32_f32", batch_size=32, in_shape=(128,), dtype="float32"),
    )
    warmup_iters: int = 5
    measure_iters: int = 20
    device_preference: str = "cpu"  # default CPU-only; {"auto", "cpu", "cuda"}
    allow_cuda_profiles: bool = False


def _select_device(preference: str, *, allow_cuda: bool) -> str:
    if torch is None:
        return "cpu"
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        try:
            if allow_cuda and torch.cuda.is_available():  # type: ignore[attr-defined]
                return "cuda:0"
        except Exception:
            return "cpu"
        return "cpu"
    # auto
    try:
        if allow_cuda and torch.cuda.is_available():  # type: ignore[attr-defined]
            return "cuda:0"
    except Exception:
        return "cpu"
    return "cpu"


def _torch_version() -> str:
    try:
        # mypy: ignore if torch is None
        return getattr(torch, "__version__", "n/a") if torch is not None else "n/a"
    except Exception:
        return "n/a"


def _to_dtype(name: str):  # type: ignore[no-untyped-def]
    if torch is None:
        return None
    key = (name or "float32").lower()
    if key in {"fp32", "float32", "f32"}:
        return torch.float32  # type: ignore[attr-defined]
    if key in {"fp16", "float16", "f16"}:
        return torch.float16  # type: ignore[attr-defined]
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16  # type: ignore[attr-defined]
    return torch.float32  # type: ignore[attr-defined]


def _identity_module(in_shape: Tuple[int, ...]):  # type: ignore[no-untyped-def]
    # Provide a callable that touches data proportionally to in_shape to create
    # a small, deterministic amount of Python work per call.
    class Identity:
        def __init__(self) -> None:
            self._ops = max(1, int(sum(in_shape)))

        def __call__(self, x):  # type: ignore[no-untyped-def]
            s = 0
            for i in range(self._ops):
                s += i
            return x

    return Identity()


def run_benchmarks(
    blueprint_id: str,
    *,
    runtime: object | None = None,
    config: BenchmarkConfig | None = None,
) -> Tuple[leyline_pb2.BlueprintBenchmark, list[dict]]:
    """Run benchmarks for a blueprint.

    - Prefers runtime.fetch_kernel to obtain a callable module when possible.
    - Falls back to a deterministic synthetic identity model on error/unavailable.
    - Uses inference_mode for safety; conditionally autocasts on CUDA for bf16.
    """

    cfg = config or BenchmarkConfig()
    device = _select_device(cfg.device_preference, allow_cuda=cfg.allow_cuda_profiles)
    torch_ver = _torch_version()

    proto = leyline_pb2.BlueprintBenchmark(
        version=1,
        blueprint_id=blueprint_id,
        device=device,
        torch_version=torch_ver,
    )

    # Attempt runtime path
    provenance = "runtime" if (runtime is not None and torch is not None) else "fallback"
    module = None
    if provenance == "runtime":
        try:
            fetched = runtime.fetch_kernel(blueprint_id)  # type: ignore[attr-defined]
            module = fetched[0] if isinstance(fetched, (tuple, list)) else fetched
        except Exception:
            module = None
            provenance = "fallback"

    mirror: list[dict] = []

    # Optionally add bf16 profile on CUDA if not already present
    profiles = list(cfg.profiles)
    try:
        if torch is not None and device.startswith("cuda") and cfg.allow_cuda_profiles:
            if all(p.dtype != "bfloat16" for p in profiles):
                profiles.append(BenchmarkProfile(name="batch32_bf16", batch_size=32, in_shape=(128,), dtype="bfloat16"))
    except Exception:
        pass

    for profile in profiles:
        p50_ms = 0.0
        p95_ms = 0.0
        thr = 0.0
        used_prov = provenance
        if used_prov == "runtime" and torch is not None and module is not None:
            try:
                # Build input tensor
                dtype = _to_dtype(profile.dtype)
                dev = device if device.startswith("cuda") else "cpu"
                try:
                    x = torch.randn((profile.batch_size, *profile.in_shape), device=dev, dtype=dtype)  # type: ignore[attr-defined]
                except Exception:
                    x = torch.randn((profile.batch_size, *profile.in_shape), device=dev, dtype=torch.float32)  # type: ignore[attr-defined]

                # Prepare forward callable
                if nn is not None and isinstance(module, nn.Module):  # type: ignore[arg-type]
                    forward = module.forward  # type: ignore[attr-defined]
                else:
                    forward = module  # callable

                # Context managers
                ctx = torch.inference_mode if torch is not None else None  # type: ignore[attr-defined]
                use_autocast = device.startswith("cuda") and (dtype is not None and str(dtype).endswith("bfloat16"))

                # Warmup
                warm = max(0, cfg.warmup_iters)
                meas = max(1, cfg.measure_iters)
                if ctx is not None:
                    with torch.inference_mode():  # type: ignore[attr-defined]
                        if use_autocast:
                            try:
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                                    for _ in range(warm):
                                        _ = forward(x)
                            except Exception:
                                for _ in range(warm):
                                    _ = forward(x)
                        else:
                            for _ in range(warm):
                                _ = forward(x)
                        samples: list[float] = []
                        for _ in range(meas):
                            t0 = perf_counter()
                            _ = forward(x)
                            dt_ms = (perf_counter() - t0) * 1000.0
                            samples.append(dt_ms)
                else:
                    samples = []
                    for _ in range(max(1, cfg.measure_iters)):
                        t0 = perf_counter()
                        _ = forward(x)
                        samples.append((perf_counter() - t0) * 1000.0)

                samples.sort()
                p50_ms = samples[len(samples) // 2]
                p95_ms = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
                thr = float(profile.batch_size) / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
            except Exception:
                used_prov = "fallback"
                module = None

        if used_prov == "fallback":
            # Deterministic synthetic timings scaled by problem size
            base = 0.05  # ms baseline
            scale = max(1, profile.batch_size) * max(1, int(sum(profile.in_shape)))
            p50_ms = base * (1.0 + 0.0005 * (scale ** 0.5))
            p95_ms = p50_ms * 1.50
            thr = float(profile.batch_size) / max(p50_ms / 1000.0, 1e-6)

        proto.profiles.add(
            name=profile.name,
            p50_latency_ms=float(p50_ms),
            p95_latency_ms=float(p95_ms),
            throughput_samples_per_s=float(thr),
        )
        mirror.append(
            {
                "name": profile.name,
                "batch_size": int(profile.batch_size),
                "in_shape": list(profile.in_shape),
                "dtype": profile.dtype,
                "p50_latency_ms": float(p50_ms),
                "p95_latency_ms": float(p95_ms),
                "throughput_samples_per_s": float(thr),
                "provenance": used_prov,
            }
        )

    return proto, mirror


__all__ = [
    "BenchmarkProfile",
    "BenchmarkConfig",
    "run_benchmarks",
]
