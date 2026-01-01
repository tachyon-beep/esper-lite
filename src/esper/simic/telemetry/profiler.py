"""torch.profiler Integration for Simic Training.

Provides context manager for on-demand profiling of training loops.
Outputs TensorBoard-compatible traces for GPU bottleneck analysis.

Usage:
    with training_profiler(enabled=args.profile) as prof:
        for step in training_loop:
            train_step()
            if prof:
                prof.step()

The profiler schedule:
- wait: Steps to skip before starting (let training stabilize)
- warmup: Steps to warmup the profiler (discard initial overhead)
- active: Steps to actually profile
- repeat: Number of profiling cycles
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def training_profiler(
    output_dir: str = "./profiler_traces",
    enabled: bool = True,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
) -> Iterator[torch.profiler.profile | None]:
    """Context manager for profiling training steps.

    Args:
        output_dir: Directory for trace output
        enabled: Whether profiling is enabled
        wait: Steps to wait before warmup (let training stabilize)
        warmup: Warmup steps before active profiling (discard JIT overhead)
        active: Steps to actively profile (collect traces)
        repeat: Number of profiling cycles
        record_shapes: Record tensor shapes (larger traces)
        profile_memory: Track memory allocations (larger traces)
        with_stack: Record Python stacks (very large traces)

    Yields:
        Profiler instance or None if disabled

    Example:
        with training_profiler(enabled=args.profile) as prof:
            for batch in dataloader:
                train_step(batch)
                if prof:
                    prof.step()

    Note:
        Profiling adds overhead. Enable only for debugging/optimization.
        Output can be viewed with TensorBoard:
            tensorboard --logdir=./profiler_traces
    """
    if not enabled:
        yield None
        return

    os.makedirs(output_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof


__all__ = ["training_profiler"]
