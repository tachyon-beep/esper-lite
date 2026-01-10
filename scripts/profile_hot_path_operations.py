#!/usr/bin/env python3
"""Profile hot path operations in vectorized training.

This script profiles individual operations in the training hot path to identify
which components contribute most to training time:
1. record_stream() calls (B8-PT-01)
2. gradient clipping (d22c2f9d)
3. GradScaler operations
4. Parameter iteration for grad checks

Run with: PYTHONPATH=src uv run python scripts/profile_hot_path_operations.py
"""

import time
import statistics
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]


class MockResNet(nn.Module):
    """A simplified ResNet-like model for profiling."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Mimic a small ResNet structure
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).flatten(1)
        result: torch.Tensor = self.fc(x)
        return result


def benchmark_grad_clipping(
    model: nn.Module,
    max_grad_norm: float = 1.0,
    iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark gradient clipping overhead."""
    _device = next(model.parameters()).device  # verify model is on expected device
    params = list(model.parameters())

    # Create fake gradients
    for p in params:
        p.grad = torch.randn_like(p)

    torch.cuda.synchronize()

    # Benchmark clip_grad_norm_
    times = []
    for _ in range(iterations):
        # Restore grads (clipping modifies them)
        for p in params:
            p.grad = torch.randn_like(p)
        torch.cuda.synchronize()

        start = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "avg_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000,
        "num_params": len(params),
        "total_param_count": sum(p.numel() for p in params),
    }


def benchmark_scaler_operations(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark GradScaler operations."""
    device = next(model.parameters()).device
    scaler = GradScaler()

    # Create a simple loss for backward
    inputs = torch.randn(32, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (32,), device=device)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()

    scale_times = []
    unscale_times = []
    step_times = []
    update_times = []

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)

        # Forward + backward with scaling
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        torch.cuda.synchronize()
        start = time.perf_counter()
        scaled_loss = scaler.scale(loss)
        torch.cuda.synchronize()
        scale_times.append(time.perf_counter() - start)

        scaled_loss.backward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        scaler.unscale_(optimizer)
        torch.cuda.synchronize()
        unscale_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        scaler.step(optimizer)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        scaler.update()
        torch.cuda.synchronize()
        update_times.append(time.perf_counter() - start)

    return {
        "scale_avg_ms": statistics.mean(scale_times) * 1000,
        "unscale_avg_ms": statistics.mean(unscale_times) * 1000,
        "step_avg_ms": statistics.mean(step_times) * 1000,
        "update_avg_ms": statistics.mean(update_times) * 1000,
        "total_scaler_overhead_ms": (
            statistics.mean(scale_times) +
            statistics.mean(unscale_times) +
            statistics.mean(update_times)
        ) * 1000,
    }


def benchmark_grad_presence_check(
    optimizer: torch.optim.Optimizer,
    iterations: int = 1000,
) -> dict[str, Any]:
    """Benchmark the has_grads check pattern from process_train_batch."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        has_grads = any(
            p.grad is not None for group in optimizer.param_groups for p in group["params"]
        )
        times.append(time.perf_counter() - start)

    return {
        "avg_us": statistics.mean(times) * 1e6,
        "std_us": statistics.stdev(times) * 1e6,
        "has_grads": has_grads,
    }


def benchmark_param_iteration(
    model: nn.Module,
    num_slots: int = 4,
    iterations: int = 1000,
) -> dict[str, Any]:
    """Benchmark parameter iteration patterns used in gradient clipping."""
    params = list(model.parameters())

    times_list = []
    times_extend = []

    for _ in range(iterations):
        # Pattern 1: list(model.parameters()) - what the code does
        start = time.perf_counter()
        _ = list(model.parameters())
        times_list.append(time.perf_counter() - start)

        # Pattern 2: extend pattern (like in the old code)
        start = time.perf_counter()
        all_params: list[nn.Parameter] = []
        all_params.extend(model.parameters())
        times_extend.append(time.perf_counter() - start)

    return {
        "list_avg_us": statistics.mean(times_list) * 1e6,
        "extend_avg_us": statistics.mean(times_extend) * 1e6,
        "num_params": len(params),
    }


def benchmark_full_train_batch_pattern(
    model: nn.Module,
    iterations: int = 50,
    with_grad_clipping: bool = True,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Benchmark the full process_train_batch pattern."""
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(64, 3, 32, 32, device=device)
    targets = torch.randint(0, 10, (64,), device=device)

    torch.cuda.synchronize()
    times = []

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Forward pass with AMP
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward with scaling
        scaler.scale(loss).backward()

        # Gradient clipping (if enabled)
        if with_grad_clipping and max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "avg_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000,
        "with_grad_clipping": with_grad_clipping,
        "iterations": iterations,
    }


def main() -> None:
    print("=" * 70)
    print("Hot Path Operations Profiling")
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
    model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Warmup
    print("\nWarming up...")
    inputs = torch.randn(32, 3, 32, 32, device=device)
    for _ in range(5):
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Test 1: Gradient clipping overhead
    print("\n" + "-" * 70)
    print("Test 1: Gradient Clipping (torch.nn.utils.clip_grad_norm_)")
    print("-" * 70)

    result = benchmark_grad_clipping(model, max_grad_norm=1.0, iterations=100)
    print(f"  Parameters: {result['num_params']}")
    print(f"  Total param count: {result['total_param_count']:,}")
    print(f"  Time per clip: {result['avg_ms']:.4f} +/- {result['std_ms']:.4f} ms")

    # Test 2: GradScaler operations
    print("\n" + "-" * 70)
    print("Test 2: GradScaler Operations")
    print("-" * 70)

    result = benchmark_scaler_operations(model, optimizer, iterations=50)
    print(f"  scale(): {result['scale_avg_ms']:.4f} ms")
    print(f"  unscale_(): {result['unscale_avg_ms']:.4f} ms")
    print(f"  step(): {result['step_avg_ms']:.4f} ms")
    print(f"  update(): {result['update_avg_ms']:.4f} ms")
    print(f"  Total scaler overhead: {result['total_scaler_overhead_ms']:.4f} ms")

    # Test 3: Grad presence check
    print("\n" + "-" * 70)
    print("Test 3: Gradient Presence Check (has_grads pattern)")
    print("-" * 70)

    result = benchmark_grad_presence_check(optimizer, iterations=1000)
    print(f"  Time per check: {result['avg_us']:.3f} +/- {result['std_us']:.3f} us")

    # Test 4: Parameter iteration
    print("\n" + "-" * 70)
    print("Test 4: Parameter Iteration Patterns")
    print("-" * 70)

    result = benchmark_param_iteration(model, iterations=1000)
    print(f"  list(model.parameters()): {result['list_avg_us']:.3f} us")
    print(f"  extend pattern: {result['extend_avg_us']:.3f} us")

    # Test 5: Full train batch comparison
    print("\n" + "-" * 70)
    print("Test 5: Full Train Batch Pattern (with vs without grad clipping)")
    print("-" * 70)

    result_with = benchmark_full_train_batch_pattern(model, with_grad_clipping=True)
    result_without = benchmark_full_train_batch_pattern(model, with_grad_clipping=False)

    overhead = result_with['avg_ms'] - result_without['avg_ms']
    overhead_pct = (overhead / result_without['avg_ms']) * 100

    print(f"  WITH grad clipping: {result_with['avg_ms']:.3f} +/- {result_with['std_ms']:.3f} ms")
    print(f"  WITHOUT grad clipping: {result_without['avg_ms']:.3f} +/- {result_without['std_ms']:.3f} ms")
    print(f"  Overhead: {overhead:.3f} ms ({overhead_pct:.1f}%)")

    # Impact analysis
    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS")
    print("=" * 70)

    # Typical training scenario
    batches_per_epoch = 50
    epochs_per_episode = 16
    episodes = 100
    total_batches = batches_per_epoch * epochs_per_episode * episodes

    grad_clip_total = result_with['avg_ms'] - result_without['avg_ms']
    total_overhead_seconds = (grad_clip_total * total_batches) / 1000

    print(f"\nFor {total_batches:,} batches ({episodes} episodes x {epochs_per_episode} epochs x {batches_per_epoch} batches):")
    print(f"  Gradient clipping overhead per batch: {grad_clip_total:.3f} ms")
    print(f"  Total gradient clipping overhead: {total_overhead_seconds:.1f} seconds")

    # Check if this is significant
    if total_overhead_seconds < 10:
        print("\n  CONCLUSION: Gradient clipping overhead is MINOR")
    elif total_overhead_seconds < 60:
        print("\n  CONCLUSION: Gradient clipping adds MODERATE overhead")
    else:
        print("\n  CONCLUSION: Gradient clipping adds SIGNIFICANT overhead")
        print("  Consider reducing clipping frequency or optimizing the implementation")


if __name__ == "__main__":
    main()
