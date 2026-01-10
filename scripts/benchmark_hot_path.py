#!/usr/bin/env python3
"""Benchmark script for hot-path optimizations.

Compares performance across:
- AMP modes: off, FP16, BF16 (auto-detected)
- Compile modes: off, default, max-autotune

Outputs events/second (eps) for each configuration.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

# Check CUDA availability first
if not torch.cuda.is_available():
    print("CUDA not available - benchmark requires GPU")
    sys.exit(1)

from typing import Any

from esper.nissa import reset_hub
from esper.simic.training.vectorized import train_ppo_vectorized


def run_benchmark(
    amp: bool,
    amp_dtype: str,
    compile_mode: str,
    n_episodes: int = 10,
    n_envs: int = 4,
    max_epochs: int = 16,
) -> dict[str, Any]:
    """Run a single benchmark configuration."""
    # Reset global hub to prevent backend accumulation across runs
    # (each train_ppo_vectorized adds BlueprintAnalytics backend)
    reset_hub()

    config_name = f"amp={amp}, dtype={amp_dtype}, compile={compile_mode}"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")

    # Warmup: torch.compile needs first run to compile
    if compile_mode != "off":
        print("Warmup run (compile)...")
        try:
            train_ppo_vectorized(
                n_episodes=2,
                n_envs=2,
                max_epochs=8,
                amp=amp,
                amp_dtype=amp_dtype,
                compile_mode=compile_mode,
                use_telemetry=False,
                device="cuda:0",
                slots=["r0c1"],  # Use default mid slot
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
            return {"config": config_name, "eps": 0.0, "error": str(e)}

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Benchmark run
    print(f"Benchmark run: {n_episodes} episodes, {n_envs} envs, {max_epochs} epochs...")
    start_time = time.perf_counter()

    try:
        agent, history = train_ppo_vectorized(
            n_episodes=n_episodes,
            n_envs=n_envs,
            max_epochs=max_epochs,
            amp=amp,
            amp_dtype=amp_dtype,
            compile_mode=compile_mode,
            use_telemetry=False,  # Disable telemetry for pure training perf
            device="cuda:0",
            slots=["r0c1"],  # Use default mid slot
        )
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return {"config": config_name, "eps": 0.0, "error": str(e)}

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Calculate events per second
    # Events = episodes * envs (each episode processes all envs in parallel)
    # But more accurately: events = total_steps = episodes * max_epochs * n_envs
    total_steps = n_episodes * max_epochs * n_envs
    eps = total_steps / elapsed

    result = {
        "config": config_name,
        "eps": eps,
        "elapsed": elapsed,
        "total_steps": total_steps,
    }

    print(f"Completed in {elapsed:.2f}s")
    print(f"Events per second: {eps:.1f}")

    return result


def main() -> None:
    print("Hot-Path Benchmark")
    print("=" * 60)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    bf16_supported = torch.cuda.is_bf16_supported()

    print(f"GPU: {gpu_name}")
    print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"BF16 supported: {bf16_supported}")

    # Benchmark parameters - smaller for faster iteration
    n_episodes = 5
    n_envs = 4
    max_epochs = 16

    print(f"\nBenchmark config: {n_episodes} episodes, {n_envs} envs, {max_epochs} epochs")

    results = []

    # 1. Baseline: No AMP, no compile (fastest to run)
    results.append(run_benchmark(
        amp=False, amp_dtype="off", compile_mode="off",
        n_episodes=n_episodes, n_envs=n_envs, max_epochs=max_epochs
    ))

    # 2. AMP FP16 (no compile) - compare AMP overhead
    results.append(run_benchmark(
        amp=True, amp_dtype="float16", compile_mode="off",
        n_episodes=n_episodes, n_envs=n_envs, max_epochs=max_epochs
    ))

    # 3. AMP auto/BF16 (no compile) - compare BF16 vs FP16
    results.append(run_benchmark(
        amp=True, amp_dtype="auto", compile_mode="off",
        n_episodes=n_episodes, n_envs=n_envs, max_epochs=max_epochs
    ))

    # 4. AMP auto + compile=default (the recommended production config)
    results.append(run_benchmark(
        amp=True, amp_dtype="auto", compile_mode="default",
        n_episodes=n_episodes, n_envs=n_envs, max_epochs=max_epochs
    ))

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<50} {'EPS':>10}")
    print("-" * 60)

    baseline_eps = results[0]["eps"] if results[0]["eps"] > 0 else 1.0
    for r in results:
        speedup = r["eps"] / baseline_eps if r["eps"] > 0 else 0
        speedup_str = f"({speedup:.2f}x)" if speedup > 0 else "(failed)"
        print(f"{r['config']:<50} {r['eps']:>8.1f} {speedup_str}")

    print("-" * 60)
    print(f"Baseline: {baseline_eps:.1f} eps")

    # Find best configuration
    best = max(results, key=lambda x: x["eps"])
    if best["eps"] > 0:
        print(f"\nBest: {best['config']} @ {best['eps']:.1f} eps ({best['eps']/baseline_eps:.2f}x speedup)")


if __name__ == "__main__":
    main()
