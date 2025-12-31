"""Profile V3 feature extraction performance.

Follows 4-phase profiling framework:
1. Establish baseline timing
2. Identify bottleneck type
3. Narrow to component
4. Recommend optimizations
"""

import torch
import time
from torch.profiler import profile, ProfilerActivity

from esper.tamiyo.policy.features import batch_obs_to_features
from esper.leyline import SlotConfig
from esper.leyline.signals import TrainingSignals, TrainingMetrics
from esper.simic.training.parallel_env_state import ParallelEnvState

MAX_EPOCHS = 100


def _make_mock_training_signals():
    """Create mock TrainingSignals for profiling."""
    metrics = TrainingMetrics(
        epoch=10,
        global_step=1000,
        train_loss=1.5,
        val_loss=1.7,
        loss_delta=-0.02,
        train_accuracy=67.0,
        val_accuracy=65.0,
        accuracy_delta=0.5,
        plateau_epochs=2,
        best_val_accuracy=65.0,
        best_val_loss=1.6,
    )

    return TrainingSignals(
        metrics=metrics,
        loss_history=[0.6, 0.55, 0.5, 0.52, 0.5],
        accuracy_history=[65.0, 66.0, 67.0, 68.0, 70.0],
    )


def _make_mock_parallel_env_state():
    """Create minimal mock ParallelEnvState."""
    class MockModel:
        pass

    class MockOptimizer:
        pass

    class MockSignalTracker:
        def reset(self):
            pass

    class MockGovernor:
        def reset(self):
            pass

    return ParallelEnvState(
        model=MockModel(),
        host_optimizer=MockOptimizer(),
        signal_tracker=MockSignalTracker(),
        governor=MockGovernor(),
        last_action_success=True,
        last_action_op=0,
    )


def phase1_baseline_timing(num_warmup=10, num_iterations=100):
    """Phase 1: Establish baseline timing for feature extraction."""
    print("=" * 60)
    print("PHASE 1: Baseline Timing")
    print("=" * 60)

    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Create mock inputs (single batch)
    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [{}]
    batch_env_states = [_make_mock_parallel_env_state()]

    # Warmup
    for _ in range(num_warmup):
        _ = batch_obs_to_features(
            batch_signals,
            batch_slot_reports,
            batch_env_states,
            slot_config,
            device,
            max_epochs=MAX_EPOCHS,
        )

    # Timed runs
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        obs, bp_indices = batch_obs_to_features(
            batch_signals,
            batch_slot_reports,
            batch_env_states,
            slot_config,
            device,
            max_epochs=MAX_EPOCHS,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5

    print(f"Batch size: 1 environment")
    print(f"Feature dim: {obs.shape[-1]}")
    print(f"Blueprint indices shape: {bp_indices.shape}")
    print(f"\nTiming (ms):")
    print(f"  Mean:  {mean_time:.4f}ms")
    print(f"  Min:   {min_time:.4f}ms")
    print(f"  Max:   {max_time:.4f}ms")
    print(f"  Std:   {std_time:.4f}ms")

    # Validate against target
    target_ms = 1.0
    if mean_time < target_ms:
        print(f"\n✓ PASS: Mean time {mean_time:.4f}ms < {target_ms}ms target")
    else:
        print(f"\n✗ FAIL: Mean time {mean_time:.4f}ms >= {target_ms}ms target")

    return mean_time, obs, bp_indices


def phase2_bottleneck_type(batch_signals, batch_slot_reports, batch_env_states, slot_config, device):
    """Phase 2: Identify bottleneck type (CPU vs memory)."""
    print("\n" + "=" * 60)
    print("PHASE 2: Bottleneck Type Identification")
    print("=" * 60)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            _ = batch_obs_to_features(
                batch_signals,
                batch_slot_reports,
                batch_env_states,
                slot_config,
                device,
                max_epochs=MAX_EPOCHS,
            )

    print("\n=== CPU Time (Top 10) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\n=== Memory Usage (Top 10) ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    return prof


def phase3_batch_scaling(slot_config, device):
    """Phase 3: Test how performance scales with batch size."""
    print("\n" + "=" * 60)
    print("PHASE 3: Batch Size Scaling")
    print("=" * 60)

    batch_sizes = [1, 2, 4, 8, 16]
    results = []

    for batch_size in batch_sizes:
        # Create batched inputs
        batch_signals = [_make_mock_training_signals() for _ in range(batch_size)]
        batch_slot_reports = [{} for _ in range(batch_size)]
        batch_env_states = [_make_mock_parallel_env_state() for _ in range(batch_size)]

        # Warmup
        for _ in range(5):
            _ = batch_obs_to_features(
                batch_signals,
                batch_slot_reports,
                batch_env_states,
                slot_config,
                device,
                max_epochs=MAX_EPOCHS,
            )

        # Timed runs
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = batch_obs_to_features(
                batch_signals,
                batch_slot_reports,
                batch_env_states,
                slot_config,
                device,
                max_epochs=MAX_EPOCHS,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        mean_time = sum(times) / len(times)
        per_sample = mean_time / batch_size

        results.append((batch_size, mean_time, per_sample))
        print(f"Batch size {batch_size:2d}: {mean_time:6.3f}ms total, {per_sample:6.3f}ms per sample")

    # Calculate efficiency
    baseline_per_sample = results[0][2]  # batch_size=1
    print(f"\nScaling efficiency (vs batch_size=1):")
    for batch_size, mean_time, per_sample in results:
        efficiency = (baseline_per_sample / per_sample) * 100
        print(f"  Batch {batch_size:2d}: {efficiency:5.1f}% efficient")

    return results


def main():
    """Run all profiling phases."""
    slot_config = SlotConfig.default()
    device = torch.device("cpu")

    # Phase 1: Baseline
    mean_time, obs, bp_indices = phase1_baseline_timing()

    # Phase 2: Bottleneck analysis
    batch_signals = [_make_mock_training_signals()]
    batch_slot_reports = [{}]
    batch_env_states = [_make_mock_parallel_env_state()]
    prof = phase2_bottleneck_type(batch_signals, batch_slot_reports, batch_env_states, slot_config, device)

    # Phase 3: Batch scaling
    results = phase3_batch_scaling(slot_config, device)

    # Summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    print(f"Single batch mean time: {mean_time:.4f}ms")
    print(f"Target: <1.0ms")
    print(f"Status: {'✓ PASS' if mean_time < 1.0 else '✗ FAIL'}")

    if mean_time >= 1.0:
        print("\nRecommended optimizations:")
        print("1. Vectorize slot report processing")
        print("2. Pre-allocate tensors instead of concatenating")
        print("3. Use torch.jit.script for hot path functions")
        print("4. Consider caching static features")


if __name__ == "__main__":
    main()
