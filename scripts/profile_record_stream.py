#!/usr/bin/env python3
"""Profile record_stream() overhead in the hot path.

This script measures the overhead of record_stream() calls added in B8-PT-01.
It isolates the record_stream() cost from actual training to determine if this
is the source of the performance regression.

Run with: PYTHONPATH=src uv run python scripts/profile_record_stream.py
"""

import time
import torch
import statistics


def benchmark_record_stream(
    num_tensors: int = 4,  # Number of tensor pairs (inputs, targets)
    batch_size: int = 64,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    iterations: int = 1000,
    num_streams: int = 4,  # Typical multi-env setup
) -> dict:
    """Benchmark record_stream() overhead.

    Simulates the hot path pattern:
    1. Create tensors on GPU
    2. Call record_stream() on multiple streams
    3. Measure per-call overhead
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}

    device = torch.device("cuda:0")

    # Create streams (simulating multi-env parallel execution)
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    # Pre-allocate tensors (inputs and targets for each env)
    # Shape: [batch_size, channels, height, width] for inputs
    # Shape: [batch_size] for targets (classification labels)
    inputs = [
        torch.randn(batch_size, channels, height, width, device=device)
        for _ in range(num_tensors)
    ]
    targets = [
        torch.randint(0, 10, (batch_size,), device=device)
        for _ in range(num_tensors)
    ]

    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        for i, stream in enumerate(streams):
            if i < len(inputs):
                inputs[i].record_stream(stream)
                targets[i].record_stream(stream)
    torch.cuda.synchronize()

    # Benchmark: With record_stream()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        for i, stream in enumerate(streams):
            if i < len(inputs):
                inputs[i].record_stream(stream)
                targets[i].record_stream(stream)
    torch.cuda.synchronize()
    with_record_stream_time = time.perf_counter() - start

    # Benchmark: Without record_stream() (baseline)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        for i, stream in enumerate(streams):
            if i < len(inputs):
                # Just access tensors without record_stream
                _ = inputs[i].data_ptr()
                _ = targets[i].data_ptr()
    torch.cuda.synchronize()
    without_record_stream_time = time.perf_counter() - start

    # Calculate overhead
    overhead_per_iteration = (with_record_stream_time - without_record_stream_time) / iterations
    overhead_per_call = overhead_per_iteration / (num_tensors * 2)  # 2 tensors per pair

    return {
        "with_record_stream_ms": with_record_stream_time * 1000,
        "without_record_stream_ms": without_record_stream_time * 1000,
        "overhead_total_ms": (with_record_stream_time - without_record_stream_time) * 1000,
        "overhead_per_iteration_us": overhead_per_iteration * 1e6,
        "overhead_per_call_us": overhead_per_call * 1e6,
        "iterations": iterations,
        "num_tensors": num_tensors,
        "num_streams": num_streams,
    }


def benchmark_process_fused_val_pattern(
    batch_size: int = 64,
    num_configs: int = 4,  # K configurations for fused pass
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    iterations: int = 500,
) -> dict:
    """Benchmark the exact pattern from process_fused_val_batch.

    This is the pattern from lines 1619-1634 in vectorized.py:
    1. inputs/targets arrive (already on GPU or via .to())
    2. record_stream(inputs), record_stream(targets)
    3. fused_inputs = inputs.repeat(num_configs, ...)
    4. fused_targets = targets.repeat(num_configs, ...)
    5. record_stream(fused_inputs), record_stream(fused_targets)
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}

    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)

    # Pre-allocate
    inputs = torch.randn(batch_size, channels, height, width, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)

    torch.cuda.synchronize()

    # Pattern A: With record_stream() (current code after B8-PT-01)
    times_with_rs = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.cuda.stream(stream):
            # First record_stream pair (lines 1620-1621)
            inputs.record_stream(stream)
            targets.record_stream(stream)

            # Repeat (lines 1625-1626)
            fused_inputs = inputs.repeat(num_configs, *([1] * (inputs.dim() - 1)))
            fused_targets = targets.repeat(num_configs, *([1] * (targets.dim() - 1)))

            # Second record_stream pair (lines 1633-1634)
            fused_inputs.record_stream(stream)
            fused_targets.record_stream(stream)

        torch.cuda.synchronize()
        times_with_rs.append(time.perf_counter() - start)

    # Pattern B: Without record_stream() (what code was before B8-PT-01)
    times_without_rs = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.cuda.stream(stream):
            # No record_stream calls
            fused_inputs = inputs.repeat(num_configs, *([1] * (inputs.dim() - 1)))
            fused_targets = targets.repeat(num_configs, *([1] * (targets.dim() - 1)))

        torch.cuda.synchronize()
        times_without_rs.append(time.perf_counter() - start)

    avg_with = statistics.mean(times_with_rs) * 1000
    avg_without = statistics.mean(times_without_rs) * 1000
    std_with = statistics.stdev(times_with_rs) * 1000
    std_without = statistics.stdev(times_without_rs) * 1000

    return {
        "with_record_stream_avg_ms": avg_with,
        "with_record_stream_std_ms": std_with,
        "without_record_stream_avg_ms": avg_without,
        "without_record_stream_std_ms": std_without,
        "overhead_ms": avg_with - avg_without,
        "overhead_percent": ((avg_with - avg_without) / avg_without) * 100,
        "batch_size": batch_size,
        "num_configs": num_configs,
        "iterations": iterations,
    }


def benchmark_accumulator_pattern(
    iterations: int = 1000,
) -> dict:
    """Benchmark record_stream on accumulators (lines 1800-1801).

    This is called once per epoch, not per batch.
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}

    device = torch.device("cuda:0")
    num_envs = 4  # Typical multi-env setup

    streams = [torch.cuda.Stream(device=device) for _ in range(num_envs)]
    loss_accums = [torch.zeros(1, device=device) for _ in range(num_envs)]
    correct_accums = [torch.zeros(1, device=device) for _ in range(num_envs)]

    torch.cuda.synchronize()

    # With record_stream
    start = time.perf_counter()
    for _ in range(iterations):
        for i, stream in enumerate(streams):
            loss_accums[i].record_stream(stream)
            correct_accums[i].record_stream(stream)
    torch.cuda.synchronize()
    time_with = time.perf_counter() - start

    # Without record_stream
    start = time.perf_counter()
    for _ in range(iterations):
        for i, stream in enumerate(streams):
            _ = loss_accums[i].data_ptr()
            _ = correct_accums[i].data_ptr()
    torch.cuda.synchronize()
    time_without = time.perf_counter() - start

    return {
        "with_record_stream_ms": time_with * 1000,
        "without_record_stream_ms": time_without * 1000,
        "overhead_per_epoch_us": (time_with - time_without) / iterations * 1e6,
        "num_envs": num_envs,
        "iterations": iterations,
    }


def main():
    print("=" * 70)
    print("record_stream() Overhead Profiling")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available - cannot profile GPU operations")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Test 1: Raw record_stream overhead
    print("\n" + "-" * 70)
    print("Test 1: Raw record_stream() overhead")
    print("-" * 70)

    result = benchmark_record_stream()
    print(f"  Iterations: {result['iterations']}")
    print(f"  Tensors per iteration: {result['num_tensors'] * 2}")
    print(f"  Streams: {result['num_streams']}")
    print(f"  With record_stream: {result['with_record_stream_ms']:.3f} ms")
    print(f"  Without record_stream: {result['without_record_stream_ms']:.3f} ms")
    print(f"  Overhead per call: {result['overhead_per_call_us']:.3f} us")

    # Test 2: process_fused_val_batch pattern
    print("\n" + "-" * 70)
    print("Test 2: process_fused_val_batch pattern (B8-PT-01 hot path)")
    print("-" * 70)

    result = benchmark_process_fused_val_pattern()
    print(f"  Batch size: {result['batch_size']}")
    print(f"  Num configs (K): {result['num_configs']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  With record_stream: {result['with_record_stream_avg_ms']:.3f} +/- {result['with_record_stream_std_ms']:.3f} ms")
    print(f"  Without record_stream: {result['without_record_stream_avg_ms']:.3f} +/- {result['without_record_stream_std_ms']:.3f} ms")
    print(f"  Overhead: {result['overhead_ms']:.3f} ms ({result['overhead_percent']:.1f}%)")

    # Test 3: Accumulator pattern (per-epoch overhead)
    print("\n" + "-" * 70)
    print("Test 3: Accumulator record_stream (per-epoch, lines 1800-1801)")
    print("-" * 70)

    result = benchmark_accumulator_pattern()
    print(f"  Environments: {result['num_envs']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  With record_stream: {result['with_record_stream_ms']:.3f} ms")
    print(f"  Without record_stream: {result['without_record_stream_ms']:.3f} ms")
    print(f"  Overhead per epoch: {result['overhead_per_epoch_us']:.3f} us")

    # Impact analysis
    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)

    # Typical training parameters
    batches_per_epoch = 50  # Approximate for CIFAR-10 with batch_size=64
    epochs_per_episode = 16
    episodes = 100

    # Per-batch overhead from process_fused_val_batch
    fused_result = benchmark_process_fused_val_pattern(iterations=100)
    per_batch_overhead_ms = fused_result['overhead_ms']

    # Per-epoch overhead from accumulator recording
    accum_result = benchmark_accumulator_pattern(iterations=100)
    per_epoch_overhead_us = accum_result['overhead_per_epoch_us']

    total_batches = batches_per_epoch * epochs_per_episode * episodes
    total_epochs = epochs_per_episode * episodes

    total_overhead_ms = (
        (per_batch_overhead_ms * total_batches) +
        (per_epoch_overhead_us * total_epochs / 1000)
    )

    print(f"\nAssuming {batches_per_epoch} batches/epoch, {epochs_per_episode} epochs/episode, {episodes} episodes:")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Total epochs: {total_epochs:,}")
    print(f"  Per-batch overhead: {per_batch_overhead_ms:.4f} ms")
    print(f"  Per-epoch overhead: {per_epoch_overhead_us:.2f} us")
    print(f"  Total record_stream overhead: {total_overhead_ms:.1f} ms ({total_overhead_ms/1000:.1f} seconds)")

    if per_batch_overhead_ms < 0.1:
        print("\n  CONCLUSION: record_stream() overhead is NEGLIGIBLE")
        print("  The performance regression is likely elsewhere.")
    else:
        print("\n  CONCLUSION: record_stream() may contribute to slowdown")
        print("  Consider optimizing the pattern or reducing call frequency.")


if __name__ == "__main__":
    main()
