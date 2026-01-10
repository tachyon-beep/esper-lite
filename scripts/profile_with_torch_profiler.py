#!/usr/bin/env python3
"""Profile vectorized training with torch.profiler.

This script runs a short training session with the built-in torch.profiler
to generate detailed traces that can be viewed in TensorBoard or Chrome.

Run with: PYTHONPATH=src uv run python scripts/profile_with_torch_profiler.py
View with: tensorboard --logdir=./profiler_traces
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

if not torch.cuda.is_available():
    print("CUDA not available - profiling requires GPU")
    sys.exit(1)

# Check memory before starting
total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total GPU memory: {total_mem:.1f} GB")

# Clear any existing memory
torch.cuda.empty_cache()

from esper.nissa import reset_hub  # noqa: E402 - import after CUDA check
from esper.simic.training.vectorized import train_ppo_vectorized  # noqa: E402


def profile_training() -> torch.profiler.profile | None:
    """Run profiled training session."""
    print("\n" + "=" * 70)
    print("Starting Profiled Training")
    print("=" * 70)

    # Reset global hub
    reset_hub()

    output_dir = "./profiler_traces"
    os.makedirs(output_dir, exist_ok=True)

    # Profile schedule: wait 1 batch, warmup 1 batch, profile 3 batches
    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    # Training parameters - small for profiling
    n_episodes = 3
    n_envs = 2
    max_epochs = 8

    print(f"\nConfig: {n_episodes} episodes, {n_envs} envs, {max_epochs} epochs")
    print(f"Output: {output_dir}")

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        try:
            # Run training with profiler
            agent, history = train_ppo_vectorized(
                n_episodes=n_episodes,
                n_envs=n_envs,
                max_epochs=max_epochs,
                amp=True,
                amp_dtype="auto",
                compile_mode="off",  # Disable compile for cleaner profiling
                use_telemetry=False,
                device="cuda:0",
                slots=["r0c1"],
            )

            # Step the profiler for each epoch
            # Note: The profiler needs to be stepped manually if we want per-batch traces
            # For now we just get an overall trace

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"\nProfiling complete! Traces saved to: {output_dir}")
    print("View with: tensorboard --logdir=./profiler_traces")

    # Print key averages
    print("\n" + "=" * 70)
    print("Top 10 CUDA Operations by Self CUDA Time")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    print("\n" + "=" * 70)
    print("Top 10 CPU Operations by Self CPU Time")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    result: torch.profiler.profile = prof
    return result


def analyze_profile(prof: torch.profiler.profile | None) -> None:
    """Analyze profiling results."""
    if prof is None:
        return

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Get key averages
    averages = prof.key_averages()

    # Find record_stream calls
    record_stream_events = [
        e for e in averages
        if "record_stream" in str(e.key).lower()
    ]

    if record_stream_events:
        print("\nrecord_stream() Operations:")
        for e in record_stream_events:
            print(f"  {e.key}: count={e.count}, self_cuda={e.self_cuda_time_total/1000:.2f}ms")
    else:
        print("\nNo record_stream() operations found in top operations")

    # Find clip_grad_norm calls
    clip_events = [
        e for e in averages
        if "clip" in str(e.key).lower() or "norm" in str(e.key).lower()
    ]

    if clip_events:
        print("\nGradient Clipping Operations:")
        for e in clip_events[:5]:
            print(f"  {e.key}: count={e.count}, self_cuda={e.self_cuda_time_total/1000:.2f}ms")

    # Find memory operations
    memory_events = [
        e for e in averages
        if any(op in str(e.key).lower() for op in ["cudamemcpy", "memcpy", "alloc"])
    ]

    if memory_events:
        print("\nMemory Operations:")
        for e in memory_events[:5]:
            print(f"  {e.key}: count={e.count}, self_cuda={e.self_cuda_time_total/1000:.2f}ms")


if __name__ == "__main__":
    prof = profile_training()
    analyze_profile(prof)
