#!/usr/bin/env python3
"""Profile the float64 conversion overhead in gradient collection.

This script measures the overhead of the B7-PT-01 fix that converts
gradients to float64 before computing norms to prevent overflow.

Run with: PYTHONPATH=src uv run python scripts/profile_float64_conversion.py
"""

import time
import statistics
import torch
import torch.nn as nn


class MockResNet(nn.Module):
    """A simplified ResNet-like model for profiling."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


def benchmark_gradient_collection(
    model: nn.Module,
    iterations: int = 100,
    with_float64: bool = True,
) -> dict:
    """Benchmark gradient collection with/without float64 conversion."""
    device = next(model.parameters()).device

    # Create fake gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Get gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]

        if with_float64:
            # B7-PT-01 pattern: convert to float64 first
            grads_for_norm = [g.double() for g in grads]
        else:
            # Original pattern: use float32 directly
            grads_for_norm = grads

        # Compute norms
        per_param_norms = torch._foreach_norm(grads_for_norm, ord=2)
        all_norms = torch.stack(per_param_norms)
        total_squared = (all_norms ** 2).sum()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "avg_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000,
        "with_float64": with_float64,
    }


def benchmark_double_conversion_only(
    model: nn.Module,
    iterations: int = 100,
) -> dict:
    """Benchmark just the .double() conversion."""
    device = next(model.parameters()).device

    # Create fake gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    grads = [p.grad for p in model.parameters() if p.grad is not None]

    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Just the conversion
        grads_double = [g.double() for g in grads]

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "avg_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000,
    }


def benchmark_foreach_norm_overhead(
    model: nn.Module,
    iterations: int = 100,
) -> dict:
    """Benchmark _foreach_norm with different dtypes."""
    device = next(model.parameters()).device

    # Create fake gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grads_double = [g.double() for g in grads]

    torch.cuda.synchronize()

    # Float32 norms
    times_fp32 = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch._foreach_norm(grads, ord=2)
        torch.cuda.synchronize()
        times_fp32.append(time.perf_counter() - start)

    # Float64 norms
    times_fp64 = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch._foreach_norm(grads_double, ord=2)
        torch.cuda.synchronize()
        times_fp64.append(time.perf_counter() - start)

    return {
        "foreach_norm_fp32_avg_ms": statistics.mean(times_fp32) * 1000,
        "foreach_norm_fp64_avg_ms": statistics.mean(times_fp64) * 1000,
        "overhead_ms": (statistics.mean(times_fp64) - statistics.mean(times_fp32)) * 1000,
    }


def main():
    print("=" * 70)
    print("Float64 Conversion Overhead Profiling (B7-PT-01)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available - cannot profile GPU operations")
        return

    device = torch.device("cuda:0")
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Create model
    model = MockResNet().to(device)
    model.to(memory_format=torch.channels_last)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Warmup
    print("\nWarming up...")
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    _ = [g.double() for g in [p.grad for p in model.parameters()]]
    torch.cuda.synchronize()

    # Test 1: Just the .double() conversion
    print("\n" + "-" * 70)
    print("Test 1: .double() conversion overhead only")
    print("-" * 70)

    result = benchmark_double_conversion_only(model, iterations=100)
    print(f"  Conversion time: {result['avg_ms']:.4f} +/- {result['std_ms']:.4f} ms")

    # Test 2: _foreach_norm with different dtypes
    print("\n" + "-" * 70)
    print("Test 2: _foreach_norm() dtype comparison")
    print("-" * 70)

    result = benchmark_foreach_norm_overhead(model, iterations=100)
    print(f"  FP32 norms: {result['foreach_norm_fp32_avg_ms']:.4f} ms")
    print(f"  FP64 norms: {result['foreach_norm_fp64_avg_ms']:.4f} ms")
    print(f"  Overhead: {result['overhead_ms']:.4f} ms")

    # Test 3: Full gradient collection pattern
    print("\n" + "-" * 70)
    print("Test 3: Full gradient collection (with vs without float64)")
    print("-" * 70)

    result_with = benchmark_gradient_collection(model, with_float64=True)
    result_without = benchmark_gradient_collection(model, with_float64=False)

    overhead = result_with['avg_ms'] - result_without['avg_ms']
    overhead_pct = (overhead / result_without['avg_ms']) * 100

    print(f"  WITH float64: {result_with['avg_ms']:.4f} +/- {result_with['std_ms']:.4f} ms")
    print(f"  WITHOUT float64: {result_without['avg_ms']:.4f} +/- {result_without['std_ms']:.4f} ms")
    print(f"  Overhead: {overhead:.4f} ms ({overhead_pct:.1f}%)")

    # Impact analysis
    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)

    # Gradient collection is called per-batch when use_telemetry=True
    # But with gradient_telemetry_stride, it's sampled
    gradient_telemetry_stride = 10  # Typical value
    batches_per_epoch = 50
    epochs_per_episode = 16
    episodes = 100

    # Actual calls = total_batches / stride (when telemetry enabled)
    total_batches = batches_per_epoch * epochs_per_episode * episodes
    telemetry_calls = total_batches // gradient_telemetry_stride

    total_overhead_ms = overhead * telemetry_calls
    total_overhead_sec = total_overhead_ms / 1000

    print(f"\nWith gradient_telemetry_stride={gradient_telemetry_stride}:")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Telemetry collection calls: {telemetry_calls:,}")
    print(f"  Per-call overhead: {overhead:.4f} ms")
    print(f"  Total float64 conversion overhead: {total_overhead_sec:.1f} seconds")

    # Without stride (worst case)
    worst_case_overhead = overhead * total_batches / 1000
    print("\nWithout stride (worst case):")
    print(f"  Total overhead: {worst_case_overhead:.1f} seconds")

    if total_overhead_sec < 5:
        print("\n  CONCLUSION: Float64 conversion overhead is NEGLIGIBLE with stride")
    else:
        print("\n  CONCLUSION: Float64 conversion adds NOTABLE overhead")
        print("  The stride helps, but this is still significant")


if __name__ == "__main__":
    main()
